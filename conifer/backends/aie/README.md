# AI Engine backend

Compiles a trained BDT to an AMD AI Engine project, using a vectorized QuickScorer
kernel mapped across one or more AIE tiles.

Target: VEK280 (`xcve2802`), AIE-ML.

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
| `compile()` | `aiecompiler` | the real mapping: tile placement, tile memory, program size |
| `decision_function(X)` | `x86simulator` | scores, bit-accurate to the hardware arithmetic |
| `build()` | `aiesimulator` | cycles: `cyc_per_sample`, `latency_ss_ns`, occupancy |

`read_report()` returns whatever stage is on disk, with a `stage` key naming it and a
`next_step` hint for the rest. It never fails for a stage that has not run.

## Configuration

Any handle may be `'auto'`, in which case the backend chooses it and reports what it
chose. `model.resolved_config()` returns the configuration with every `'auto'` filled
in; passing that back reproduces the same project.

| field | default | meaning |
|---|---|---|
| `Priority` | `latency` | `latency` splits trees across tiles; `throughput` splits samples |
| `NTiles` | `auto` | 1–64 |
| `SplitAxis` | `auto` | `tree` or `sample` |
| `VectorWidth` | `auto` | samples per invocation |
| `Tau` | `auto` | trees per tile under tree-split |
| `NSamples` | `auto` | rows the graph is compiled for |
| `Shard` | `auto` | cut each tile's feature rows to a contiguous window (tree-split only) |
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

Every sharded model is checked before it is emitted: the per-tile partial scores are
replayed and required to sum to exactly what the unsharded tables score.

## Kernels

| model | kernel | tiles |
|---|---|---|
| axis-aligned | QuickScorer, per-tree layout, multi-tile | 1–64 |
| oblique, weights in {0, ±1} | QuickScorer with a partial-projection basis | 1 |

An oblique split tests `w · x <= threshold`. Only binary ±1 projection weights are
supported, which is what ydf's default `sparse_oblique_weights="BINARY"` emits.

## Limits

Raised at `write()`:

- `max_depth > 6` — the result bitvector holds one bit per leaf and reaches two words
- oblique projection weights outside {0, ±1}
- more than two classes — the kernels score one value per sample
- an oblique model with `n_tiles > 1` — no multi-tile oblique kernel exists
- `n_tiles` above the device's tile count, or above the 64 the templates expand to
- unsupported precision width, rounding or overflow mode
- `n_features > 64` for axis-aligned

Warned, not raised:

- estimated tile memory over budget — the AIE mapper has the final say
- a score precision too narrow for `n_trees × max|leaf|`, which may saturate

Padded, with an INFO message: oblique `n_features` up to a power of two, and the sample
count up to a multiple of the vector width.

## The estimate

`write()` reports `est_cyc_per_sample` and `est_latency_ss_ns` from a cost model fitted
on VEK280 measurements. It is an estimate, not a measurement, and it carries a
`validity` list naming every extrapolation in play — a depth or vector width outside the
fitted range, or an oblique model, for which no cost law was fitted. Use `build()` for
real numbers.
