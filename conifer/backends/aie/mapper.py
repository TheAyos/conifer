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
               6: (125, 1150.5),
               }

# Measured: W=32 wins throughput at every depth, W=16 wins latency at every depth. The
# invocation is what latency pays and W is what throughput divides by, so they pull apart.
_PRIORITY_W = {'latency': 16, 'throughput': 32}

# A latency mapping grows while a doubling of the tile count still buys this much.
_MARGINAL_GAIN = 0.10

# Sample-split throughput is linear in the tile count, and tree-split latency keeps paying
# until tau reaches 1, so auto stops here to keep compile times reasonable and says so.
# One ceiling for both priorities: a different one per priority would let the latency
# mapping out-run the throughput mapping on throughput.
_AUTO_TILE_CEILING = 16


MIN_VECTOR_WIDTH = 8


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

# What auto will choose from: the widths the study actually swept. Wider still builds and
# the model will price it, but nothing on this device has measured it, so a user has to
# ask for it by name.
AUTO_VECTOR_WIDTHS = (8, 16, 32)


def _metric(priority):
    return ('est_latency_ss_ns' if priority == 'latency'
            else 'est_throughput_ns_per_sample')


def choose_mapping(n_trees, max_depth, n_features, feat_bytes, leaf_bytes, priority,
                   max_tiles, tile_memory_bytes, oblique=False, feed='plio',
                   n_tiles=None, W=None):
    '''Pick the tile count and vector width together, on the metric the priority names

    They interact: a narrower vector is a smaller invocation, which helps latency only
    while the per-invocation setup is comparable to the per-tree work.
    '''
    notes = []
    ceiling = max_tiles
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    tiles = [n_tiles] if n_tiles else (
        [1] if oblique else tile_candidates(auto_ceiling))
    widths = [W] if W else list(AUTO_VECTOR_WIDTHS)

    key = _metric(priority)
    best, best_score = None, None
    for n in tiles:
        for w in widths:
            if w * feat_bytes % 4:
                continue
            e = estimate(n_trees, max_depth, n_features, n, w, feat_bytes, leaf_bytes,
                         priority, oblique, feed)
            if best_score is None or e[key] < best_score:
                best, best_score = (n, w), e[key]

    n, w = best
    if n_tiles is None and n == auto_ceiling < ceiling:
        notes.append(f'stopping at {n} tiles: auto does not go past {auto_ceiling} to keep '
                     f'compile time down, raise n_tiles for more')
    if oblique:
        notes.append('the oblique kernel is single-tile')
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

# The oblique basis kernel's cost against an equally-shaped axis-aligned one, measured.
_OBLIQUE_TAX = 2.73


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


def invocation_cycles(n_features, max_depth, tau, W, feat_bytes, feed='plio'):
    '''Cycles for one invocation, which scores W samples against tau trees

    n_features is the busiest tile's row count, which sharding reduces. A memtile row is
    two wide loads rather than sixteen stream reads, so the fill term nearly vanishes.
    '''
    base, per_tree = _DEPTH_COST[max_depth]
    row = _row_cycles(W, feat_bytes) * (_MEMTILE_ROW_FRACTION if feed == 'memtile' else 1.0)
    # The setup term scales with the group, like the rows do: measured fixed roughly
    # halves from W=32 to W=16.
    base *= W * feat_bytes / _FIT_BYTES
    return base + n_features * row + per_tree * _register_scale(W, max_depth) * tau


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
    # busiest tile taking the ceiling.
    if split_axis == 'sample':
        return n_trees
    return int(math.ceil(n_trees / n_tiles))


def estimate(n_trees, max_depth, n_features, n_tiles, W, feat_bytes, leaf_bytes,
             priority, oblique=False, feed='plio', split_axis=None):
    '''Forward estimate for one mapping

    split_axis is normally the priority's, because a priority IS a choice of axis. It is
    overridable so a REQUIREMENT can search both: given a rate to hold and a latency
    budget to stay inside, which axis serves them is an answer rather than an input.
    '''
    split_axis = split_axis or _split_axis(priority)
    tau = _tau_for(n_trees, n_tiles, split_axis)
    inv = invocation_cycles(n_features, max_depth, tau, W, feat_bytes, feed)
    if oblique:
        inv *= _OBLIQUE_TAX
    samples_per_inv = W * (n_tiles if split_axis == 'sample' else 1)

    return {'est_cyc_per_invocation': inv,
            'est_cyc_per_sample': inv / samples_per_inv,
            'est_latency_ss_ns': 1.1 * inv / AIE_CLOCK_GHZ,
            'est_throughput_ns_per_sample': inv / samples_per_inv / AIE_CLOCK_GHZ,
            'tau': tau,
            'split_axis': split_axis,
            }


# --------------------------------------------------------------------------- #
# The long form: a required rate and a latency budget, instead of a priority
# --------------------------------------------------------------------------- #
#
# `priority` asks the mapper to be good at one of two things. A trigger does not want
# that: it wants a rate it MUST hold, because the arrival rate is a physical constant,
# and a latency it must stay inside. Those are requirements, and a mapping either meets
# them or it does not.
#
# It is the same law, used backwards, and the reason it was not shipped in the first pass
# is worth keeping in view: inverting a fitted law PROMISES something. The forward form
# reports an estimate a user reads and judges; the inverse form says "this mapping meets
# your requirement", and if the point it picked rests on an extrapolation then the promise
# rests on one too. So the search reports which, prefers a point that does not, and never
# silently returns one that does.


def _requirement_candidates(n_trees, max_depth, n_features, feat_bytes, leaf_bytes,
                            max_tiles, oblique, feed, n_tiles, W):
    """Every mapping the search may return, with its forward estimate.

    BOTH AXES, and that is the point of the long form. Tree-split divides the invocation
    and is what a latency budget buys; sample-split multiplies the samples per invocation
    and is what a rate buys. Which of them serves a given (rate, budget) pair is exactly
    the question being asked, so neither is assumed.
    """
    ceiling = max_tiles
    tiles = [n_tiles] if n_tiles else (
        [1] if oblique else tile_candidates(ceiling))
    widths = [W] if W else list(AUTO_VECTOR_WIDTHS)
    out = []
    for axis in (('tree',) if oblique else ('tree', 'sample')):
        for n in tiles:
            for w in widths:
                if w * feat_bytes % 4:
                    continue
                e = estimate(n_trees, max_depth, n_features, n, w, feat_bytes,
                             leaf_bytes, 'latency', oblique, feed, split_axis=axis)
                out.append((n, w, axis, e))
    return out


def meet_requirement(n_trees, max_depth, n_features, feat_bytes, leaf_bytes,
                     max_tiles, tile_memory_bytes, max_ns_per_sample=None,
                     max_latency_ns=None, oblique=False, feed='plio',
                     n_tiles=None, W=None):
    """The smallest mapping that holds a rate and stays inside a latency budget.

    Returns (n_tiles, W, split_axis, notes). Either requirement may be None, in which
    case it is not a constraint -- one of the two alone is a perfectly good question.

    SMALLEST MEANS FEWEST TILES, then narrowest vector. A requirement is a floor, not an
    objective: once two mappings both meet it, the one that spends less silicon is the
    answer, and spending more to beat a requirement nobody stated is how a mapper ends up
    recommending sixty-four tiles for a model that fits on four.

    """
    if max_ns_per_sample is None and max_latency_ns is None:
        raise ValueError('meet_requirement needs a rate, a latency budget, or both')

    cands = _requirement_candidates(n_trees, max_depth, n_features, feat_bytes,
                                    leaf_bytes, max_tiles, oblique, feed, n_tiles, W)

    def meets(e):
        if max_ns_per_sample is not None \
                and e['est_throughput_ns_per_sample'] > max_ns_per_sample:
            return False
        if max_latency_ns is not None and e['est_latency_ss_ns'] > max_latency_ns:
            return False
        return True

    def fits(w):
        # A mapping that does not fit tile memory is not a mapping. Sharding is not
        # implemented in this backend, so every tile carries the whole ensemble's tables
        # and the footprint does not shrink with the tile count -- which is why this is
        # checked against the WIDTH and not against the tiles.
        b = table_bytes(n_trees, max_depth, 1 << max_depth, n_features, w, feat_bytes,
                        leaf_bytes, oblique)
        return b['total'] <= tile_memory_bytes

    pool = [c for c in cands if meets(c[3]) and fits(c[1])]
    notes = []

    if not pool:
        # NOTHING MEETS IT, and saying WHICH HALF failed is most of the value of the
        # answer: a rate that cannot be held and a latency that cannot be met call for
        # different changes to the model.
        best_rate = min(cands, key=lambda c: c[3]['est_throughput_ns_per_sample'])
        best_lat = min(cands, key=lambda c: c[3]['est_latency_ss_ns'])
        # WHICH FAILURE IT IS MATTERS, and there are two quite different ones. Either a
        # requirement is out of reach on its own -- no mapping is fast enough, or none is
        # prompt enough -- or each is reachable and NO SINGLE MAPPING does both. The
        # second is the interesting one: it means the rate wants sample-split and the
        # latency wants tree-split, and the model has to change rather than the mapping.
        rate_ok = (max_ns_per_sample is None
                   or best_rate[3]['est_throughput_ns_per_sample'] <= max_ns_per_sample)
        lat_ok = (max_latency_ns is None
                  or best_lat[3]['est_latency_ss_ns'] <= max_latency_ns)
        if rate_ok and lat_ok:
            msg = ['each requirement is reachable but no single mapping meets BOTH: the '
                   'rate wants sample-split (many tiles, whole ensemble each) and the '
                   'latency wants tree-split (many tiles, few trees each), and one graph '
                   'is one axis. Shrink the ensemble, lower a requirement, or split the '
                   'model across two graphs']
        else:
            msg = ['no mapping in range meets the requirement']
        if max_ns_per_sample is not None:
            msg.append('best rate reachable is '
                       '{:.2f} ns/sample at {} tiles W={} ({}-split), against the '
                       'required {:.2f}'.format(
                           best_rate[3]['est_throughput_ns_per_sample'], best_rate[0],
                           best_rate[1], best_rate[2], max_ns_per_sample))
        if max_latency_ns is not None:
            msg.append('best latency reachable is '
                       '{:.1f} ns at {} tiles W={} ({}-split), against the budget '
                       '{:.1f}'.format(
                           best_lat[3]['est_latency_ss_ns'], best_lat[0], best_lat[1],
                           best_lat[2], max_latency_ns))
        raise ValueError('; '.join(msg))

    # Fewest tiles, then narrowest vector, then the better of the two on whichever
    # requirement was given. A deterministic order, so the same request maps the same way
    # twice.
    def rank(c):
        n, w, ax, e = c
        tie = (e['est_latency_ss_ns'] if max_latency_ns is not None
               else e['est_throughput_ns_per_sample'])
        return (n, w, tie)

    n, w, axis, e = min(pool, key=rank)
    margin = []
    if max_ns_per_sample is not None:
        margin.append('{:.0f}% rate headroom'.format(
            100 * (1 - e['est_throughput_ns_per_sample'] / max_ns_per_sample)))
    if max_latency_ns is not None:
        margin.append('{:.0f}% latency headroom'.format(
            100 * (1 - e['est_latency_ss_ns'] / max_latency_ns)))
    notes.append('{} tile(s), W={}, {}-split meets the requirement with {}'.format(
        n, w, axis, ' and '.join(margin)))
    return n, w, axis, notes


def choose_n_tiles(n_trees, max_depth, n_features, W, feat_bytes, leaf_bytes, priority,
                   max_tiles, tile_memory_bytes, oblique=False):
    '''Pick a tile count, and explain the choice. Returns (n_tiles, [notes])'''
    notes = []
    ceiling = max_tiles
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    if oblique:
        return 1, ['the oblique kernel is single-tile']

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
