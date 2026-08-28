import math
import logging
logger = logging.getLogger(__name__)

AIE_CLOCK_GHZ = 1.25

# cyc/invocation = base + n_features * row + per_tree * tau, measured at f16/W=32/int16.
# Piecewise rather than smooth in depth: the cost steps where bv_t widens.
_DEPTH_COST = {2: (107, 50.5),
               3: (104, 91.0),
               4: (106, 156.0),
               5: (125, 410.4),
               6: (125, 1150.5),
               }

# Measured: W=32 wins throughput at every depth, W=16 wins latency at every depth. The
# invocation is what latency pays and W is what throughput divides by, so they pull apart.
_PRIORITY_W = {'latency': 16, 'throughput': 32}

# A latency mapping grows while a doubling of the tile count still buys this much.
_MARGINAL_GAIN = 0.10

# Both priorities keep paying past this, so auto stops here for compile time and says so.
# One ceiling for both: per-priority ceilings let one priority win the other's metric.
_AUTO_TILE_CEILING = 16


def tile_candidates(ceiling):
    """Powers of two up to a ceiling. The ladder is generated, so this follows the device."""
    out, n = [], 1
    while n <= ceiling:
        out.append(n)
        n *= 2
    return out


def bitvector_bits(max_depth):
    '''Bits of result bitvector per lane: one bit per leaf, rounded to the word type'''
    leaves = 1 << max_depth
    if leaves <= 16:
        return 16
    if leaves <= 32:
        return 32
    return 64


def vector_width(priority, max_depth):
    '''Samples per invocation, from the priority knob

    The measured optimum at the tau a latency mapping lands on. choose_mapping picks it
    from the cost model instead when the tile count is free.
    '''
    return _PRIORITY_W[priority]


VECTOR_WIDTHS = (8, 16, 32, 64)

# The widths the study swept. Wider builds and is priced, but is unmeasured on this
# device, so a user has to ask for it by name.
AUTO_VECTOR_WIDTHS = (8, 16, 32)


def _metric(priority):
    return ('est_latency_ss_ns' if priority == 'latency'
            else 'est_throughput_ns_per_sample')


def choose_mapping(n_trees, max_depth, n_features, feat_bytes, leaf_bytes, priority,
                   max_tiles, tile_memory_bytes, oblique=False, feed='plio',
                   n_tiles=None, W=None, basis_n=0):
    '''Pick the tile count and vector width together, on the metric the priority names

    They interact: a narrower vector is a smaller invocation, which helps latency only
    while the per-invocation setup is comparable to the per-tree work.
    '''
    notes = []
    ceiling = max_tiles
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    tiles = [n_tiles] if n_tiles else tile_candidates(auto_ceiling)
    widths = [W] if W else list(AUTO_VECTOR_WIDTHS)

    key = _metric(priority)
    best, best_score = None, None
    for n in tiles:
        for w in widths:
            if w * feat_bytes % 4:
                continue
            e = estimate(n_trees, max_depth, n_features, n, w, feat_bytes, leaf_bytes,
                         priority, oblique, feed, basis_n=basis_n)
            if best_score is None or e[key] < best_score:
                best, best_score = (n, w), e[key]

    n, w = best
    if n_tiles is None and n == auto_ceiling < ceiling:
        notes.append(f'stopping at {n} tiles: auto does not go past {auto_ceiling} to keep '
                     f'compile time down, raise n_tiles for more')
    if oblique:
        notes.append('oblique: tree-split does not divide the basis, so the speedup '
                     'saturates against it however many tiles are added')
    return n, w, notes


# A memtile row is two 256-bit loads against sixteen 32-bit stream reads (FINDINGS 18:
# the fill stops being a heterogeneity in time).
_MEMTILE_ROW_FRACTION = 0.125

# The width the cost table was fitted at, and one vector register.
_FIT_W = 32
_FIT_BYTES = 64
_REGISTER_BITS = 512

# The share of per-tree work that does not scale with the bitvector's register count.
_PER_TREE_FLOOR = 0.5

# Measured: fixed + 11.9 * basis_n + 380 * tau. The basis belongs to the ensemble, not
# the shard, so tree-split does not divide it -- and one multiplier would.
_OBLIQUE_PER_TREE_TAX = 380.0 / 156.0
_BASIS_CYC_PER_ENTRY = 11.9


def _row_cycles(W, feat_bytes):
    # A feature row is W * sizeof(feat_t) bytes taken four at a time, one 32-bit stream
    # read an instruction.
    return W * feat_bytes / 4


def _bitvector_registers(W, max_depth):
    return max(1, math.ceil(bitvector_bits(max_depth) * W / _REGISTER_BITS))


def _register_scale(W, max_depth):
    '''How per_tree moves with W: the bitvector's register count is what drives it

    Only part of the per-tree work is bitvector-wide, so the scaling has a floor -
    fitted across the measured depth and width sweep.
    '''
    ratio = _bitvector_registers(W, max_depth) / _bitvector_registers(_FIT_W, max_depth)
    return _PER_TREE_FLOOR + (1.0 - _PER_TREE_FLOOR) * ratio


def _invocation_parts(n_features, max_depth, tau, W, feat_bytes, feed='plio'):
    '''(what one invocation pays whatever tau is, what it pays per tree)

    n_features is the busiest tile's row count, which sharding reduces. A memtile row is
    two wide loads rather than sixteen stream reads, so the fill term nearly vanishes.
    '''
    base, per_tree = _DEPTH_COST[max_depth]
    row = _row_cycles(W, feat_bytes) * (_MEMTILE_ROW_FRACTION if feed == 'memtile' else 1.0)
    # The setup term scales with the group, like the rows do: measured fixed roughly
    # halves from W=32 to W=16.
    base *= W * feat_bytes / _FIT_BYTES
    return base + n_features * row, per_tree * _register_scale(W, max_depth) * tau


def invocation_cycles(n_features, max_depth, tau, W, feat_bytes, feed='plio'):
    '''Cycles for one invocation, which scores W samples against tau trees'''
    fixed, trees = _invocation_parts(n_features, max_depth, tau, W, feat_bytes, feed)
    return fixed + trees


def _roundup(x, to):
    return ((x + to - 1) // to) * to


def table_bytes(n_trees, max_depth, max_leaves, n_features, W, feat_bytes, leaf_bytes,
                oblique, basis_n=0, max_terms=1):
    '''Tile data memory, mirroring the heap and stack the graph declares

    Every tile is handed the same generated tables and indexes its own tree range into
    them, so this does not shrink with the tile count. Sharding cuts the feature rows a
    tile reads, not the tables it holds.
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
    # busiest tile taking the ceiling.
    if split_axis == 'sample':
        return n_trees
    return int(math.ceil(n_trees / n_tiles))


def estimate(n_trees, max_depth, n_features, n_tiles, W, feat_bytes, leaf_bytes,
             priority, oblique=False, feed='plio', split_axis=None, basis_n=0):
    '''Forward estimate for one mapping

    split_axis is normally the priority's, because a priority IS a choice of axis. It is
    overridable so a REQUIREMENT can search both: given a rate to hold and a latency
    budget to stay inside, which axis serves them is an answer rather than an input.
    '''
    split_axis = split_axis or _split_axis(priority)
    tau = _tau_for(n_trees, n_tiles, split_axis)
    fixed, trees = _invocation_parts(n_features, max_depth, tau, W, feat_bytes, feed)
    if oblique:
        inv = fixed + _OBLIQUE_PER_TREE_TAX * trees + _BASIS_CYC_PER_ENTRY * basis_n
    else:
        inv = fixed + trees
    samples_per_inv = W * (n_tiles if split_axis == 'sample' else 1)

    return {'est_cyc_per_invocation': inv,
            'est_cyc_per_sample': inv / samples_per_inv,
            'est_latency_ss_ns': 1.1 * inv / AIE_CLOCK_GHZ,
            'est_throughput_ns_per_sample': inv / samples_per_inv / AIE_CLOCK_GHZ,
            'tau': tau,
            'split_axis': split_axis,
            }


def choose_n_tiles(n_trees, max_depth, n_features, W, feat_bytes, leaf_bytes, priority,
                   max_tiles, tile_memory_bytes, oblique=False, basis_n=0):
    '''Pick a tile count, and explain the choice. Returns (n_tiles, [notes])'''
    notes = []
    ceiling = max_tiles
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    candidates = tile_candidates(ceiling)

    if priority == 'throughput':
        n = min(auto_ceiling, ceiling)
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
                        priority, oblique, basis_n=basis_n)['est_latency_ss_ns']
        there = estimate(n_trees, max_depth, n_features, n, W, feat_bytes, leaf_bytes,
                         priority, oblique, basis_n=basis_n)['est_latency_ss_ns']
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
