#pragma once

#ifndef BDT_SHARDED
#define BDT_SHARDED 0
#endif

#include <cstdint>
#include "parameters.h"

#ifndef BDT_W
#define BDT_W 16
#endif

#ifndef BDT_N_TILES
#define BDT_N_TILES 1
#endif

// 1 = tree-split   : disjoint tree subsets, same samples, partial scores merged
// 0 = sample-split : full ensemble per tile, distinct samples, no merge
#ifndef BDT_SPLIT_TREE
#define BDT_SPLIT_TREE 0
#endif

// Tree-split merge: 0 = cascade chain, 1 = per-tile PLIO (the control).
#ifndef BDT_MERGE_PLIO
#define BDT_MERGE_PLIO 0
#endif

#ifndef BDT_MERGE_REDUCE
#define BDT_MERGE_REDUCE 0
#endif

#ifndef BDT_REDUCE_K
#define BDT_REDUCE_K 0
#endif

#ifndef BDT_REDUCE_ROOT_K
#define BDT_REDUCE_ROOT_K 0
#endif

#ifndef BDT_REDUCE_NODES
#define BDT_REDUCE_NODES 0
#endif

#ifndef BDT_FEED_PLIO
#define BDT_FEED_PLIO 0
#endif

#ifndef BDT_FEED_MEMTILE
#define BDT_FEED_MEMTILE 0
#endif

#ifndef BDT_MT_BUFFERS
#define BDT_MT_BUFFERS 2
#endif

#ifndef BDT_MT_FANOUT
#define BDT_MT_FANOUT 8
#endif

#ifndef BDT_DELTA
#define BDT_DELTA BDT_W
#endif

#ifndef BDT_TAP
#define BDT_TAP 0
#endif

#ifndef BDT_PLIO_RATE
#define BDT_PLIO_RATE 625
#endif

#ifndef BDT_PLACE
#define BDT_PLACE 0
#endif

#define BDT_PLIO_COL_FIRST 5
#define BDT_PLIO_COL_COUNT 28

#ifndef BDT_FEAT_MASK
#define BDT_FEAT_MASK 0xFFFF
#endif

#ifndef BDT_T_BEGIN
#define BDT_T_BEGIN 0
#endif

#ifndef BDT_TAU
#define BDT_TAU 0        /* 0 = derive from N_TILES */
#endif

namespace bdtmt {

constexpr unsigned N_TILES    = BDT_N_TILES;
constexpr bool     SPLIT_TREE = (BDT_SPLIT_TREE != 0);
constexpr bool     MERGE_PLIO = (BDT_MERGE_PLIO != 0);
constexpr bool     MERGE_REDUCE = (BDT_MERGE_REDUCE != 0);
static_assert(!(MERGE_REDUCE && MERGE_PLIO),
              "cascade, plio and reduce are three merges, not two flags -- pick one");

// Sample-split gives every tile the whole ensemble; only tree-split shards it.
constexpr unsigned N_SHARDS = SPLIT_TREE ? N_TILES : 1u;
constexpr unsigned DELTA    = BDT_DELTA;
constexpr bool     FEED_PLIO = (BDT_FEED_PLIO != 0);
constexpr bool     FEED_MEMTILE = (BDT_FEED_MEMTILE != 0);
constexpr unsigned MT_BUFFERS = BDT_MT_BUFFERS;
static_assert(MT_BUFFERS >= 2, "a ping-pong needs at least two buffers");
constexpr unsigned MT_FANOUT = BDT_MT_FANOUT;
static_assert(MT_FANOUT >= 1, "a memtile feeds at least one tile");
// How many memtiles the array needs, which tile each one feeds, and how many it feeds.
constexpr unsigned N_MEMTILE =
    FEED_MEMTILE ? (N_TILES + MT_FANOUT - 1) / MT_FANOUT : 0u;
constexpr unsigned mt_of(unsigned t) { return t / MT_FANOUT; }
constexpr unsigned mt_first(unsigned m) { return m * MT_FANOUT; }
constexpr unsigned mt_count(unsigned m) {
    return (m + 1) * MT_FANOUT <= N_TILES ? MT_FANOUT : N_TILES - m * MT_FANOUT;
}
static_assert(!(FEED_MEMTILE && FEED_PLIO),
              "broadcast, per-tile PLIO and memtile are three feeds, not two flags");

#if BDT_FEED_MEMTILE
static_assert(SPLIT_TREE && N_TILES > 1,
              "the memtile feed writes one group for the whole array to share; a "
              "sample-split tile holds different samples and shares nothing");
static_assert(BDT_SHARDED,
              "FEED=memtile hands a tile a RANGE of rows, so the model must have been "
              "sharded to say which range -- re-run gen/shard_model.py --assign span");
static_assert(bdtsh::WINDOWED,
              "this sharding's rows are a SET, not a contiguous window, and a memtile "
              "tiling cannot ask for a set. Use --assign span (ADR-0021 §5)");
static_assert(!MERGE_REDUCE,
              "goal 2 changes the FEED and holds the merge at the control's, so that "
              "the two axes stay separable; `both` is ADR-0021 §10's 4.3 arm");
#endif

constexpr bool     TAP = (BDT_TAP != 0);

constexpr unsigned REDUCE_K = (BDT_REDUCE_K != 0) ? (unsigned)BDT_REDUCE_K : N_TILES;

constexpr unsigned reduce_levels() {
    unsigned l = 0, span = 1;
    while (span < N_TILES) { span *= REDUCE_K; l++; }
    return l;
}
// Nodes on level `j`, counting the payload tiles as level 0.
constexpr unsigned reduce_count(unsigned j) {
    unsigned c = N_TILES;
    for (unsigned i = 0; i < j; i++) c = (c + REDUCE_K - 1) / REDUCE_K;
    return c;
}
constexpr unsigned REDUCE_LEVELS = MERGE_REDUCE ? reduce_levels() : 0u;

constexpr unsigned reduce_nodes() {
    unsigned n = 0;
    for (unsigned j = 1; j <= REDUCE_LEVELS; j++) n += reduce_count(j);
    return n;
}
constexpr unsigned N_REDUCE = MERGE_REDUCE ? reduce_nodes() : 0u;

constexpr unsigned reduce_offset(unsigned j) {
    unsigned o = 0;
    for (unsigned i = 1; i < j; i++) o += reduce_count(i);
    return o;
}
constexpr unsigned ROOT_ARITY =
    MERGE_REDUCE ? (REDUCE_LEVELS > 1 ? reduce_count(REDUCE_LEVELS - 1) : N_TILES) : 0u;

constexpr bool reduce_regular() {
    unsigned c = N_TILES;
    for (unsigned j = 1; j < REDUCE_LEVELS; j++) {
        if (c % REDUCE_K) return false;
        c /= REDUCE_K;
    }
    return true;
}

#if BDT_MERGE_REDUCE
static_assert(SPLIT_TREE && N_TILES > 1,
              "a reducer sums PARTIAL scores; a sample-split tile emits a whole one and "
              "has nothing to reduce");
static_assert(REDUCE_K >= 2, "a reducer takes at least two partials");
static_assert(REDUCE_K <= 4,
              "pktorderedmerge does not route above arity 4 on this device -- measured, "
              "probe/pktmerge; at N > 4 the reduction is a TREE and k is 2, 3 or 4");
static_assert(reduce_regular(),
              "this (N_TILES, k) needs a ragged interior level -- every level but the "
              "root must divide exactly by k");
static_assert(ROOT_ARITY >= 2 && ROOT_ARITY <= 4, "the root's arity is 2, 3 or 4");
static_assert(BDT_REDUCE_ROOT_K == (int)ROOT_ARITY,
              "BDT_REDUCE_ROOT_K disagrees with the tree this file derives -- the "
              "Makefile and params.h have come apart");
static_assert(BDT_REDUCE_NODES == (int)N_REDUCE,
              "BDT_REDUCE_NODES disagrees with the tree this file derives -- the "
              "Makefile and params.h have come apart");
// BDT_PLACE, the macro, and not bdtmt::PLACE -- the constexpr is declared further down
// this file than this block.
static_assert(BDT_PLACE == 0,
              "the placement arms constrain payload tiles and their ports only; a "
              "reducer tree would be left to the mapper and the arm would measure half "
              "a placement (ADR-0021 §4)");
#endif
constexpr double   PLIO_RATE = BDT_PLIO_RATE;  // MHz, on every input port
constexpr unsigned PLACE = BDT_PLACE;
static_assert(PLACE <= 2, "BDT_PLACE is 0 (mapper), 1 (column) or 2 (pessimum)");

constexpr unsigned place_col(unsigned t) {
    return PLACE == 2 ? 3u + (t % 16u)
                      : BDT_PLIO_COL_FIRST + (t % BDT_PLIO_COL_COUNT);
}
constexpr unsigned place_row(unsigned t) {
    return PLACE == 2 ? 6u + (t / 16u) : (t / BDT_PLIO_COL_COUNT);
}
static_assert(PLACE == 0 || (N_TILES - 1) / (PLACE == 2 ? 16u : BDT_PLIO_COL_COUNT)
                            + (PLACE == 2 ? 6u : 0u) < 8u,
              "this placement arm would need more than eight AIE rows at this tile "
              "count -- widen the column span or lower N_TILES");

constexpr unsigned port_col(unsigned t) {
    return PLACE == 2
        ? BDT_PLIO_COL_FIRST + (BDT_PLIO_COL_COUNT - 1u - (t % BDT_PLIO_COL_COUNT))
        : BDT_PLIO_COL_FIRST + (t % BDT_PLIO_COL_COUNT);
}
static_assert(!TAP || (SPLIT_TREE && !MERGE_PLIO && N_TILES > 1),
              "TAP=1 is only defined for a tree-split cascade chain or reducer tree of "
              "more than one tile; every other topology emits from every tile already "
              "(ADR-0020 §3, ADR-0021 §4)");

static_assert(BDT_SHARDED || BDT_TAU != 0 || bdtm::N_TREES % N_SHARDS == 0,
              "n_trees must divide evenly across the tiles; round 1 does not map "
              "a ragged ensemble (ADR-0019). A SHARDED model is exempt: its tree counts "
              "come from a generated table and are allowed to be ragged");
constexpr unsigned TAU = (BDT_TAU != 0) ? (unsigned)BDT_TAU
                                        : bdtm::N_TREES / N_SHARDS;
static_assert(TAU <= bdtm::N_TREES, "tau exceeds the ensemble");

static_assert(!(SPLIT_TREE && !MERGE_PLIO) || N_TILES <= 32,
              "the cascade chain is declared to 32 tiles; beyond that use MERGE=plio, "
              "which is the merge this study settled on (ADR-0020 §12)");
static_assert(N_TILES <= 64, "no per-shard symbol exists past 64 tiles");

constexpr bool SHARDED = (BDT_SHARDED != 0);

#if BDT_SHARDED
static_assert(bdtsh::N_SHARDS == N_SHARDS,
              "this model was sharded for a different tile count -- re-run "
              "gen/shard_model.py with --tiles matching BDT_N_TILES");
static_assert(SPLIT_TREE || N_TILES == 1,
              "sharding is a property of the TREE axis; a sample-split tile holds the "
              "whole ensemble and therefore reads every feature");
static_assert(BDT_FEAT_MASK == 0xFFFF,
              "a sharded model has already dropped its rows -- BDT_FEAT_MASK would drop "
              "them again, in a frame where a bit no longer names the same feature");
static_assert(BDT_T_BEGIN == 0,
              "a sharded model starts each tile where its own table says, not at an "
              "offset -- BDT_T_BEGIN is for the unsharded single-tile gate");
#endif

constexpr unsigned t_begin(unsigned shard) {
#if BDT_SHARDED
    return bdtsh::T_BEGIN[shard];
#else
    return BDT_T_BEGIN + shard * TAU;
#endif
}
constexpr unsigned t_count(unsigned shard) {
#if BDT_SHARDED
    return bdtsh::T_COUNT[shard];
#else
    (void)shard;
    return TAU;
#endif
}

using featmask_t = std::uint64_t;
constexpr featmask_t ALL_FEATURES =
    bdtm::N_FEATURES >= 64 ? ~featmask_t(0)
                           : ((featmask_t(1) << bdtm::N_FEATURES) - featmask_t(1));
constexpr featmask_t FEAT_MASK =
    (BDT_FEAT_MASK == 0xFFFFu && bdtm::N_FEATURES > 16) ? ALL_FEATURES
                                                        : (featmask_t)BDT_FEAT_MASK;
constexpr bool reads_feature(unsigned f) { return (FEAT_MASK >> f) & featmask_t(1); }
constexpr unsigned n_feat_read() {
    unsigned n = 0;
    for (unsigned f = 0; f < bdtm::N_FEATURES; f++) n += reads_feature(f);
    return n;
}
static_assert(n_feat_read() > 0, "a mask that reads no feature reads no samples");

constexpr unsigned n_feat(unsigned shard) {
#if BDT_SHARDED
    return bdtsh::N_FEAT[shard];
#else
    (void)shard;
    return bdtm::N_FEATURES;
#endif
}

constexpr unsigned feat_offset(unsigned shard) {
#if BDT_SHARDED
    return bdtsh::OFFSET[shard];
#else
    (void)shard;
    return 0u;
#endif
}

// The rows this tile is fed, as a bit per GLOBAL feature -- what names its input file.
constexpr unsigned feat_mask(unsigned shard) {
#if BDT_SHARDED
    return bdtsh::MASK[shard];
#else
    (void)shard;
    return FEAT_MASK;
#endif
}

constexpr bool adds_init(unsigned shard) { return !MERGE_REDUCE && shard == 0; }

}  // namespace bdtmt

static_assert(bdtm::N_SAMPLES % BDT_W == 0,
              "N_SAMPLES must be a whole number of W-sample groups; "
              "regenerate with --n-samples a multiple of W");
static_assert(bdtmt::SPLIT_TREE || (bdtm::N_SAMPLES / BDT_W) % bdtmt::N_TILES == 0,
              "sample-split needs the group count to divide across the tiles");
constexpr unsigned iter_count =
    bdtmt::SPLIT_TREE ? (bdtm::N_SAMPLES / BDT_W)
                      : (bdtm::N_SAMPLES / BDT_W / bdtmt::N_TILES);
