# AI Engine backend

Compiles a trained BDT to an AMD AI Engine project, using a vectorized QuickScorer
kernel mapped across one or more AIE tiles.

Target: VEK280 (`xcve2802`), AIE-ML.

Platforms are located from the Vitis environment, so sourcing `settings64.sh` is
enough - `PLATFORM_REPO_PATHS` does not have to be set. Both the `vek280_base` and
`xilinx_vek280_base_<version>` layouts are recognised. Set `Platform` to an `.xpfm`
path to override.

## Quick start

```python
import conifer

cfg = conifer.backends.aie.auto_config()
cfg['OutputDir'] = 'prj_aie'
cfg['Priority'] = 'latency'          # or 'throughput'

model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()                        # no toolchain needed
model.compile()                      # aiecompiler
y = model.decision_function(X)       # x86simulator
model.build()                        # aiecompiler + aiesimulator
print(model.read_report())
```

## The four stages

| call | needs | gives |
|---|---|---|
| `write()` | nothing | the project, the resolved mapping, and a forward cost estimate |
| `compile()` | `aiecompiler` | that the project builds, for the x86 functional path |
| `decision_function(X)` | `x86simulator` | scores, bit-accurate to the hardware arithmetic |
| `build()` | `aiesimulator` | cycles: `cyc_per_sample`, `latency_ss_ns`, occupancy |

`read_report()` returns whatever stage is on disk, with a `stage` key naming it and a
`next_step` hint for the rest. It never fails for a stage that has not run.

Tile placement, per-tile memory and program size come from `aiecompiler -target=hw`, which
`build()` runs; the x86 compile produces no map report.

## Configuration

Any handle may be `'auto'`, in which case the backend chooses it and reports what it
chose. `model.resolved_config()` returns the configuration with every `'auto'` filled
in; passing that back reproduces the same project.

| field | default | meaning |
|---|---|---|
| `Priority` | `latency` | `latency` splits trees across tiles; `throughput` splits samples. Also chooses the tile count and vector width |
| `NTiles` | `auto` | 1 to the outgoing PLIO channels the platform routes (112 on `vek280_base`) |
| `SplitAxis` | `auto` | `tree` or `sample` |
| `VectorWidth` | `auto` | samples per invocation; auto chooses from 8, 16, 32 |
| `PlioRate` | `auto` | offered input rate in MHz; at most half the array clock |
| `Tau` | `auto` | trees per tile under tree-split |
| `NSamples` | `auto` | rows the graph is compiled to score in one run |
| `Shard` | `auto` | `auto` searches the layout, `fast` skips the search, `False` disables sharding |
| `Feed` | `auto` | `memtile` shares one input across the array; `plio` gives each tile a port |
| `XilinxPart` | `xcve2802-vsvh1760-2MP-e-S` | selects the device record |
| `ElfgenJobs` | unset | caps `aiecompiler` ELF generation fan-out |

Precisions follow conifer's usual fields. The compare path must be 16 bits and the
accumulator 32, both with `AP_RND_CONV,AP_SAT`:

```
Precision / InputPrecision / ThresholdPrecision   ap_fixed<16,I,AP_RND_CONV,AP_SAT>
ScorePrecision                                    ap_fixed<32,I,AP_RND_CONV,AP_SAT>
WeightPrecision                                   ap_fixed<16,I,AP_RND_CONV,AP_SAT>   (oblique only)
```

`AP_RND_CONV,AP_SAT` is required because the kernels are bit-exact against that grid;
the `ap_fixed` default `AP_TRN,AP_WRAP` would score on a different one.

## Sharding and the memtile feed

Under tree-split with more than one tile the backend permutes trees and feature rows so
each tile reads a contiguous window of rows rather than all of them, and feeds the array
from a memtile: one input port writes each group into a shared buffer that every tile
reads its own window from. Measured on VEK280 this is better on latency, period, port
count and balance at every tile count.

Sample-split is never sharded - a sample-split tile holds the whole ensemble, so it
reads every row. Set `Shard=False` or `Feed='plio'` to decline either.

The tree assignment and the feature order are searched together, seeded so a build is
reproducible; `Shard='fast'` takes a deterministic heuristic instead, at a few percent
more rows.

Every sharded model is checked before it is emitted: the per-tile partial scores are
replayed and required to sum to exactly what the unsharded tables score.

## Kernels

| model | kernel | tiles |
|---|---|---|
| axis-aligned | QuickScorer, per-tree layout, multi-tile | 1–112 |
| oblique, weights in {0, ±1} | QuickScorer with a partial-projection basis | 1 |

An oblique split tests `w · x <= threshold`. Only binary ±1 projection weights are
supported, which is what ydf's default `sparse_oblique_weights="BINARY"` emits.

## Limits

Raised at `write()`:

- `max_depth > 6` — the result bitvector holds one bit per leaf and reaches two words
- oblique projection weights outside {0, ±1}
- more than two classes — the kernels score one value per sample
- an oblique model with `n_tiles > 1` — no multi-tile oblique kernel exists
- `n_tiles` above the device's tile count, or above the outgoing PLIO channels the
  platform routes: every tile emits its own partial score on its own channel, on both
  split axes, and the outgoing side binds first
- unsupported precision width, rounding or overflow mode

Warned, not raised:

- estimated tile memory over budget — the AIE mapper has the final say
- a score precision too narrow for `n_trees × max|leaf|`, which may saturate

Padded, with an INFO message: oblique `n_features` up to a power of two, and the sample
count up to a multiple of the vector width.

## `n_samples` is a batch size

The graph is compiled for a fixed number of rows, so `n_samples` is how many samples one
run scores - a property of the project, not of the model.

**`decision_function(X)` accepts any number of rows.** A shorter `X` is padded up to the
batch; a longer one is split across as many runs as it needs. Either way you get exactly
`len(X)` scores back, and an INFO line says what it cost - padding rows that are computed
and discarded, or the number of runs - so `NSamples` can be set to match a known workload.

`NSamples` itself also accepts any value: a run is a whole number of `W`-sample groups
(and, under sample-split, of tiles), so a value that does not divide is rounded up rather
than refused, with an INFO saying so.

Auto holds the batch at one target across mappings so two configurations of the same
model stay comparable.

## How auto chooses

`Priority` fixes the split axis; the tile count and the vector width are then chosen
together, on the metric the priority names, using the cost model below. They interact -
a narrower vector is a smaller invocation, which helps latency only while the
per-invocation setup is comparable to the per-tree work, so neither can be picked alone.

Auto only chooses vector widths the study measured. Wider ones build and the cost model
prices them, but set `VectorWidth` explicitly to use one.

## What `latency_ss` means

The residence of one group: from the array accepting its first input word to its last
score existing. Under tree-split every tile holds the same group, so the array first
held it when the earliest tile accepted it and the answer exists when the last partial
does. Acceptance is reconstructed from each tile's own output, since an invocation
begins by reading: `accepted = emitted - invocation`.

It is reported as the **intercept of a fit** over group index, not a mean, with the
slope beside it as `latency_ss_drift_ns_per_group`. A pipelined mapping can hold a
perfectly steady period while residence climbs, in which case the mean is a function of
how long the run was and the intercept is not.

This is the steady-state residence, not the cold latency of a drained graph.

## The estimate

`write()` reports `est_cyc_per_sample` and `est_latency_ss_ns` from a cost model fitted
on VEK280 measurements. It reproduces every measured point of the
study's width sweep - depths 4 to 6 at widths 8 to 32 - within 9%, and both tile-count
anchors exactly. It is still an estimate, not a measurement, and carries a `validity`
list naming every extrapolation in play — a depth or vector width outside the
fitted range, or an oblique model, for which no cost law was fitted. Use `build()` for
real numbers.
