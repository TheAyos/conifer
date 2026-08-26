import math
import logging
logger = logging.getLogger(__name__)

AIE_CLOCK_GHZ = 1.25

# cyc/invocation = base + n_features * row + per_tree * tau, measured on the QuickScorer
# per-tree kernel at n_features=16, W=32, int16. Piecewise in the bitvector width rather
# than smooth in depth: the cost steps where bv_t widens.
_DEPTH_COST = {2: (107, 50.5),
               3: (104, 91.0),
               4: (106, 156.0),
               5: (125, 410.4),
               6: (125, 735.0),
               }

# Depths whose per_tree is extrapolated from the bitvector step rather than fitted.
_EXTRAPOLATED_DEPTHS = (6,)

# Bits of result bitvector per lane, which sets the vector width the priority knob picks.
_PRIORITY_BITS = {'latency': 512, 'throughput': 1024}

# A latency mapping grows while a doubling of the tile count still buys this much.
_MARGINAL_GAIN = 0.10

# Sample-split throughput is linear in the tile count, and tree-split latency keeps paying
# until tau reaches 1, so auto stops here to keep compile times reasonable and says so.
_THROUGHPUT_DEFAULT_TILES = 8
_AUTO_TILE_CEILING = 16

# The kernel templates expand a per-tile ladder that stops here.
MAX_TEMPLATE_TILES = 64

MIN_VECTOR_WIDTH = 8


def bitvector_bits(max_depth):
    '''Bits of result bitvector per lane: one bit per leaf, rounded to the word type'''
    leaves = 1 << max_depth
    if leaves <= 16:
        return 16
    if leaves <= 32:
        return 32
    return 64


def vector_width(priority, max_depth):
    '''The vector width the priority knob asks for, as a bitvector bit-target'''
    w = _PRIORITY_BITS[priority] // bitvector_bits(max_depth)
    return max(MIN_VECTOR_WIDTH, w)


# A memtile row is two 256-bit loads against sixteen 32-bit stream reads (FINDINGS 18:
# the fill stops being a heterogeneity in time).
_MEMTILE_ROW_FRACTION = 0.125


def _row_cycles(W, feat_bytes):
    # A feature row is W * sizeof(feat_t) bytes taken four at a time, one 32-bit stream
    # read an instruction.
    return W * feat_bytes / 4


def _register_scale(W, feat_bytes):
    # per_tree was fitted at one 512-bit register of lanes. Below that the vector ops do
    # not get cheaper; above it they multiply.
    return max(1.0, W * feat_bytes / 64.0)


def invocation_cycles(n_features, max_depth, tau, W, feat_bytes, feed='plio'):
    '''Cycles for one invocation, which scores W samples against tau trees

    n_features is the straggler's row count, which sharding reduces. A memtile row is
    two wide loads rather than sixteen stream reads, so the fill term nearly vanishes.
    '''
    base, per_tree = _DEPTH_COST[max_depth]
    row = _row_cycles(W, feat_bytes) * (_MEMTILE_ROW_FRACTION if feed == 'memtile' else 1.0)
    return base + n_features * row + per_tree * _register_scale(W, feat_bytes) * tau


def _roundup(x, to):
    return ((x + to - 1) // to) * to


def table_bytes(n_trees, max_depth, max_leaves, n_features, W, feat_bytes, leaf_bytes,
                oblique, basis_n=0, max_terms=1):
    '''Tile data memory, mirroring the heap and stack the graph declares

    Sharding is not implemented, so every tile carries the whole ensemble's tables and
    this does not shrink with the tile count.
    '''
    slots = n_trees * ((1 << max_depth) - 1)
    bv_bytes = bitvector_bits(max_depth) // 8
    tables = {'qt_thr': slots * feat_bytes,
              'qt_bv': slots * bv_bytes,
              'qt_feat': slots * 2,
              'leaves': n_trees * max_leaves * leaf_bytes,
              'init_v': n_trees * bv_bytes,
              }
    x_lanes = n_features
    if oblique:
        tables['qt_bterm'] = (slots * max_terms + 64) * 2
        tables['qt_bsign'] = (slots * max_terms + 64) * feat_bytes
        tables['basis_ij'] = basis_n * 2 * 2
        tables['basis_w'] = basis_n * 2 * feat_bytes
        x_lanes = n_features + basis_n

    b = dict(tables)
    b['heap'] = _roundup(sum(tables.values()) + 2 * 1024, 1024)
    b['stack'] = _roundup(x_lanes * W * feat_bytes + 4 * 1024, 1024)
    b['total'] = b['heap'] + b['stack']
    return b


def _split_axis(priority):
    return 'tree' if priority == 'latency' else 'sample'


def _tau_for(n_trees, n_tiles, split_axis):
    # Sample-split gives every tile the whole ensemble; tree-split divides it, with the
    # straggler taking the ceiling.
    if split_axis == 'sample':
        return n_trees
    return int(math.ceil(n_trees / n_tiles))


def estimate(n_trees, max_depth, n_features, n_tiles, W, feat_bytes, leaf_bytes,
             priority, oblique=False, feed='plio'):
    '''Forward estimate for one mapping. An estimate, not a measurement: see validity'''
    split_axis = _split_axis(priority)
    tau = _tau_for(n_trees, n_tiles, split_axis)
    inv = invocation_cycles(n_features, max_depth, tau, W, feat_bytes, feed)
    samples_per_inv = W * (n_tiles if split_axis == 'sample' else 1)

    validity = []
    if max_depth in _EXTRAPOLATED_DEPTHS:
        validity.append(f'per_tree at depth {max_depth} is extrapolated from the bitvector '
                        'step, not fitted')
    if W != 32:
        validity.append(f'the cost law was fitted at W=32; W={W} scales it by register count')
    if feat_bytes != 2:
        validity.append('the cost law was fitted at int16')
    if oblique:
        validity.append('the oblique kernel has no fitted cost law; this is the '
                        'axis-aligned law and understates it')

    return {'est_cyc_per_invocation': inv,
            'est_cyc_per_sample': inv / samples_per_inv,
            'est_latency_ss_ns': 1.1 * inv / AIE_CLOCK_GHZ,
            'est_throughput_ns_per_sample': inv / samples_per_inv / AIE_CLOCK_GHZ,
            'tau': tau,
            'split_axis': split_axis,
            'validity': validity,
            }


def choose_n_tiles(n_trees, max_depth, n_features, W, feat_bytes, leaf_bytes, priority,
                   max_tiles, tile_memory_bytes, oblique=False):
    '''Pick a tile count, and explain the choice. Returns (n_tiles, [notes])'''
    notes = []
    ceiling = min(max_tiles, MAX_TEMPLATE_TILES)
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    if oblique:
        return 1, ['the oblique kernel is single-tile']

    candidates = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= ceiling]

    if priority == 'throughput':
        n = min(_THROUGHPUT_DEFAULT_TILES, ceiling)
        if ceiling > n:
            notes.append(f'sample-split throughput scales linearly with tiles; stopping at '
                         f'{n} to keep compile time down, raise n_tiles for more')
        return n, notes

    chosen = 1
    for n in [c for c in candidates if c > 1]:
        if n > auto_ceiling:
            notes.append(f'stopping at {chosen} tiles: auto does not go past {auto_ceiling} '
                         'to keep compile time down, raise n_tiles for lower latency')
            break
        here = estimate(n_trees, max_depth, n_features, chosen, W, feat_bytes, leaf_bytes,
                        priority)['est_latency_ss_ns']
        there = estimate(n_trees, max_depth, n_features, n, W, feat_bytes, leaf_bytes,
                         priority)['est_latency_ss_ns']
        if here - there < _MARGINAL_GAIN * here:
            notes.append(f'stopping at {chosen} tiles: doubling to {n} buys '
                         f'{100 * (here - there) / here:.1f}%, under the '
                         f'{100 * _MARGINAL_GAIN:.0f}% threshold. Set n_tiles manually for '
                         'lower latency')
            break
        chosen = n
    else:
        if chosen == ceiling:
            notes.append(f'stopping at {chosen} tiles, the maximum available')

    return chosen, notes
