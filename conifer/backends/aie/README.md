# AI Engine backend

Compiles a trained BDT to an AMD AI Engine project, using vectorized branch-free kernels
mapped across one or more AIE tiles.

Target: VEK280 (`xcve2802`), AIE-ML.

Platforms are located from the Vitis environment, so sourcing `settings64.sh` is
enough - `PLATFORM_REPO_PATHS` does not have to be set. Both the `vek280_base` and
`xilinx_vek280_base_<version>` layouts are recognised. Set `Platform` to an `.xpfm`
path to override.

## Where things are

This backend is larger than the other hardware ones because it has no HLS step to hand
the hard problem to: `aiecompiler` places kernels but does not choose the parallelism,
the data layout or the numeric grid, so the mapping is decided here.

    writer.py     AIEConfig, AIEModel, and the four stages. Bulk-writes parameters.h
    mapper.py     the cost model, and the policy that picks tiles, width and axis
    tables.py     the node and leaf tables, derived from the conifer ensemble
    shard.py      tree and feature assignment, so each tile reads a row window
    precision.py  ap_fixed -> integer width and binary point
    checks.py     the guards, each raising with what to change
    report.py     the staged report reader
    roles.py      the per-tile kernel symbol ladder, generated per project
    devices.py    device records (devices/*.json), read without any toolchain
    platforms.py  locating a platform from the Vitis environment
    tools.py      toolchain discovery, and running one Makefile target
    firmware/     the vendored kernels: common/, axis/, oblique/
    template/     the project Makefile

Start at `writer.py`: it owns the model class and calls everything else.

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

`examples/sklearn_to_aie.py` is the same thing end to end and runnable: it writes a
project with no toolchain, and `--build` runs the three stages that need one and prints
what each returns.

## The four stages

| call | needs | gives |
|---|---|---|
| `write()` | nothing | the project, the resolved mapping, and a forward cost estimate |
| `compile()` | `aiecompiler` | that the project builds, for the x86 functional path |
| `decision_function(X)` | `x86simulator` | scores, bit-accurate to the hardware arithmetic |
| `build(simulate=False)` | `aiecompiler` | the placement, tile memory and program memory of the mapped design |
| `build(X)` | `aiesimulator` | that, plus cycles: `cyc_per_sample`, `latency_ss_ns` (steady-state latency, below), `slowest_tile_ratio`, and `X` scored |

`read_report()` returns whatever stage is on disk, with a `stage` key naming it and a
`next_step` hint for the rest. It never fails for a stage that has not run.

`build()` runs two toolchain stages, and its arguments say which.
`build(simulate=False)` stops after the hardware compile: the placement and the memory a
tile needs, with no stimulus to supply and no cycle-accurate run to wait for. Expect a
modest saving rather than a large one - on the sklearn example the compile is 430 s of the
483 s a full `build()` costs, and the simulator about 50 s. The simulator is
cycle-accurate, so its share grows with `NSamples` while the compile's does not.
`build(X)` names the rows the cycle-accurate run scores; read them back with
`read_scores(simulator='aie')`. With no `X` it simulates whatever `data/x.dat` holds, or
zeros, and says which: the kernels are branch-free, so the timing does not depend on the
data.

Each stage captures its tool output to `<target>.log` in the project directory -
`x86sim_build.log`, `x86sim.log`, `aiesim.log` - and logs where it is. A stage that
fails says so, names its log, and quotes the first error the tools reported.

Tile placement, per-tile memory and program size come from `aiecompiler -target=hw`, which
`build()` runs; the x86 compile produces no map report.

## The vocabulary

Five words the rest of this page assumes, none of them conifer's:

- **tile** - one AI Engine core, with its own program and 64 kB of data memory. A VEK280
  has 304 of them.
- **PLIO** - the stream ports between the array and the rest of the chip. They are a
  scarcer resource than the cores: this platform routes 112 outgoing ones.
- **memtile** - a shared on-chip buffer that several tiles can read from, each at its own
  offset. It is how different tiles get *different* data from one input port.
- **tree-split** - use more tiles by giving each a subset of the trees. Every tile sees
  every sample and emits a partial score, and the partials are summed. Shortens latency.
- **sample-split** - use more tiles by giving each a subset of the *samples*, with the
  whole ensemble on each. Nothing to sum. Raises throughput.

One **invocation** is one call of a tile's kernel, scoring `VectorWidth` samples against
its share of the trees. It is the unit every cycle count on this page is built from.

## Configuration

The **mapping** handles - everything above `XilinxPart` - may be `'auto'`, in which case
the backend chooses one and reports what it chose. `model.resolved_config()` returns the
configuration with each filled in; passing that back reproduces the same project.
`Priority` is the exception among them: it always holds a value, because it is what the
others are chosen against.

`XilinxPart`, `Platform` and `ElfgenJobs` name the target and the toolchain rather than
the mapping. Nothing chooses them and `resolved_config()` returns them unchanged.

| field | default | meaning |
|---|---|---|
| `Priority` | `latency` | `latency` splits trees across tiles; `throughput` splits samples. Also chooses the tile count and vector width |
| `NTiles` | `auto` | 1 to the outgoing PLIO channels the platform routes (112 on `vek280_base`) |
| `SplitAxis` | `auto` | `tree` or `sample` |
| `VectorWidth` | `auto` | samples per invocation; auto fills a vector register, 32 for latency and 64 for throughput |
| `PlioRate` | `auto` | offered input rate in MHz; at most half the array clock |
| `TreesPerTile` | `auto` | how many trees each tile takes under tree-split |
| `NSamples` | `auto` | rows the graph is compiled to score in one run |
| `Shard` | `auto` | narrow each tile's feature rows to a window: `auto` searches the layout, `fast` uses a heuristic, `False` reads every row |
| `Feed` | `auto` | `memtile` shares one input across the array; `plio` gives each tile a port |
| `XilinxPart` | `xcve2802-vsvh1760-2MP-e-S` | selects the device record: core count, tile memory, PLIO channels, clock. Read at `write()`, with or without a toolchain |
| `Platform` | unset | overrides the `.xpfm` the toolchain builds against. The device record names one, so this is only for a custom or renamed platform, or an absolute path |
| `ElfgenJobs` | unset | caps `aiecompiler` ELF generation fan-out |

Precisions follow conifer's usual fields. The compare path must be 16 bits and the
accumulator 32, both with `AP_RND_CONV,AP_SAT`:

```
Precision / InputPrecision / ThresholdPrecision   ap_fixed<16,I,AP_RND_CONV,AP_SAT>
ScorePrecision                                    ap_fixed<32,I,AP_RND_CONV,AP_SAT>
WeightPrecision                                   ap_fixed<16,I,AP_RND_CONV,AP_SAT>   (oblique only)
```

`AP_RND_CONV,AP_SAT` is required because the tables are quantized round-half-to-even
and saturating, and that mode is what names it;
the `ap_fixed` default `AP_TRN,AP_WRAP` would score on a different one.

## Giving each tile less to read

Splitting trees across tiles divides the *work*, but not the *input*: by default every
tile still reads every feature of every sample, and then waits on rows most of its trees
never test.

It does not have to. A tile's trees are fixed when the project is written, so the features
those trees test are known too. The backend uses that twice - it chooses **which trees go
on which tile**, and it **reorders the feature rows** - so that the features one tile
needs end up next to each other. Each tile then reads a single contiguous window of rows
instead of the whole sample. The code calls this *sharding*.

Delivering different rows to different tiles needs the **memtile feed**: one input port
writes each sample group into a shared buffer, and each tile reads only its own window out
of it. A multicast PLIO cannot do it - every tile gets the same stream - so `Feed='plio'`
declines the windowing along with the memtile.

`write()` says what it achieved, and is honest when that is nothing:

    sharded: each tile reads 13 of 16 feature rows at worst (46 across the array)

Ten of ten would mean the trees on every tile touch every feature, so there was nothing to
cut. Measured on VEK280 the memtile feed is better on latency, period, port count and
balance at every tile count, whether or not the windowing saves rows.

Sample-split never does this: such a tile holds the whole ensemble, so it reads every row
regardless. `Shard=False` declines the windowing on its own.

The tree assignment and the feature order are searched together, seeded so a build is
reproducible; `Shard='fast'` takes a deterministic heuristic instead, at a few percent
more rows.

Every windowed model is checked before it is emitted: the per-tile partial scores are
replayed and required to sum to exactly what the unwindowed tables score.

## Kernels

| model | kernel | tiles |
|---|---|---|
| axis-aligned | per-tree table layout, multi-tile | 1–112 |
| oblique, weights in {0, ±1} | partial-projection basis, multi-tile | 1–112 |

Both are written for this backend. Neither walks a tree: every node in a tile's trees is
evaluated, each failing node clears the leaves it rules out from a per-tree bitmask, and
the surviving lowest bit is the exit leaf. There are no data-dependent branches and no
pointer chasing, which is what makes the work vectorize across samples.

That exit-leaf scheme is the one **QuickScorer** introduced (Lucchese et al., *QuickScorer:
A Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees*, SIGIR
2015), and the credit for it belongs there. Everything around it is this work: the
vectorization across a sample group, the table layout, the multi-word bitmask that carries
depth, the leaf select chain, the split across tiles, and the whole oblique path - the
published algorithm is axis-aligned only and has no notion of a projection.

An oblique split tests `w · x <= threshold`. Only binary ±1 projection weights are
supported, which is what ydf's default `sparse_oblique_weights="BINARY"` emits.

Tree-split divides an oblique ensemble as it divides an axis-aligned one, with one
exception that governs how well it scales. Before testing any node, a tile builds the
**basis**: the signed feature pairs every oblique split is expressed in. That is built
once per sample group over the whole feature set, and it belongs to the ensemble rather
than to any tile's share of it - so adding tiles divides the tree work and leaves the
basis exactly where it was. It is what an oblique mapping saturates against, and the
estimate prices it as its own term for that reason. Four tiles measured 2.58x, not 4x.

An oblique model also reads every feature row on every tile, and takes the plain PLIO
feed. Narrowing rows needs to know which features a tile's trees test; an oblique node
tests a dense weight row over all of them, so there is nothing to narrow.

## Limits

Raised at `write()`:

- `max_depth > 6` — the result bitvector holds one bit per leaf and reaches two words
- oblique projection weights outside {0, ±1}
- more than two classes — the kernels score one value per sample
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

`Priority` fixes two things directly and one through the cost model.

**The split axis** follows it: `latency` splits trees, `throughput` splits samples.

**The vector width** follows it too, and is not searched. A group of `W` samples is one
vector per feature, so `W` is just the register the priority wants divided by the compare
width: **512 bits for latency, 1024 for throughput**, which at the mandatory 16-bit
compare is `W=32` and `W=64`. A partly filled register wastes the lanes it does not use,
and measurement says so on both arms. What varies is what a lane costs: the result
bitvector is one bit per leaf, so a deeper ensemble wants a narrower group to fill the
same register, and the measured winner flips exactly where the bitvector doubles.

| depth | bits per lane | `W` for 512 bits | measured `latency_ss` |
|---|---|---|---|
| 4 | 16 | **32** | W=16 4316 ns, **W=32 4292 ns** |
| 5 | 32 | **16** | **W=16 7979 ns**, W=32 10816 ns |

Set `VectorWidth` to override.

**The tile count** is the one the cost model chooses, by minimising whichever metric the
priority names over the powers of two up to the auto ceiling.

## The three per-sample numbers, and why they differ

`build()` reports all three, and they are not the same quantity:

| | what it counts | independent of `NSamples`? |
|---|---|---|
| `cyc_per_sample` | the kernel's own cycles, per score | yes, in steady state |
| `throughput_ns_per_sample` | the steady-state invocation period, per score | yes, in steady state |
| `run_ns_per_sample` | **every** cycle of the run, graph startup and teardown included | **no** |

`throughput_ns_per_sample` is the period the array holds - one group's last output to the
next group's - over the samples an invocation retires: `W` on a tree-split, `W x n_tiles`
on a sample-split, which is what the estimate divides by too. Its excess over
`cyc_per_sample` is reported as `io_ns_per_sample`, the time the array spends not
computing. On the shipped examples that is a fraction of a nanosecond: these designs are
compute-bound. It is a steady-state median against a whole-run mean, so on a short run it
can come out slightly negative - that is the first invocation's transient sitting in
`cyc_per_sample`, and it means no I/O cost is resolvable.

**`run_ns_per_sample` is not a throughput.** Its excess over the kernel's own time is a
fixed graph startup and teardown - 7300 to 9600 cycles whatever the model - so it falls as
`NSamples` rises: on the sklearn example that constant adds 18.7 cyc/sample over 512
samples and 62.7 over 128. **Two of them compare only at equal `NSamples`**, and neither
compares against the estimate.

## Reading the array's balance

`slowest_tile_ratio` is the busiest tile's cycles over the average tile's. **1.0 is a
perfectly balanced array**; 1.05 means the busiest tile does 5% more work than the
average one, and the whole array waits for it. Narrowing each tile's rows, above,
is what keeps it near 1.

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
anchors exactly. It is still an estimate, not a measurement: use `build()` for real
numbers.

Two arms are weaker than the rest, both at narrow vectors. At `W=8` the fixed term is
under-predicted, so an axis-aligned estimate runs **optimistic** - 16% on cyc/sample and
14% on latency for the example above - and a latency mapping is where auto picks a narrow
vector. The oblique basis term is priced flat per entry from a `W=32` measurement and is
cheaper at narrower vectors, so an oblique estimate runs **high** at small `W`. The error
grows with the tile count, because the mispriced term is fixed while the tree work beside
it is divided: +8% on one tile, +19% on sixteen.
Both are sizing errors, not correctness ones: `build()` is where reported numbers come
from.
