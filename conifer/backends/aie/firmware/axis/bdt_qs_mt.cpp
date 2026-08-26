#include "bdt_qs_mt.hpp"
#include "bdt_qs_pertree_v2.hpp"

using namespace adf;
using namespace bdtm;
using namespace bdtv;

template <unsigned SHARD>
__attribute__((always_inline))
static inline vscore score_shard(input_stream<feat_t>* xin) {
    return qs_score_group<bdtmt::t_begin(SHARD),
                          bdtmt::t_count(SHARD),
                          bdtmt::adds_init(SHARD),
                          bdtmt::n_feat(SHARD)>(
#if BDT_SHARDED
        [&](unsigned) __attribute__((always_inline)) {
            return readincr_v<W>(xin);
        });
#else
        [&](unsigned f) __attribute__((always_inline)) {
            return bdtmt::reads_feature(f) ? readincr_v<W>(xin)
                                           : aie::zeros<feat_t, W>();
        });
#endif
}

template <unsigned SHARD>
__attribute__((always_inline))
static inline vscore score_shard_buf(const feat_t* __restrict p) {
    return qs_score_group<bdtmt::t_begin(SHARD),
                          bdtmt::t_count(SHARD),
                          bdtmt::adds_init(SHARD),
                          bdtmt::n_feat(SHARD)>(
        [&](unsigned f) __attribute__((always_inline)) {
            return aie::load_v<W>(p + f * W);
        });
}

#define BDT_DEF_BUF(S)                                                                 \
    void bdt_qs_tile_##S(adf::input_buffer<feat_t>& xin,                                    \
                         output_stream<score_t>* __restrict sout) {                    \
        writeincr(sout, score_shard_buf<S>(xin.data()));                               \
    }

void bdt_qs_mt(input_stream<feat_t>* __restrict xin,
               output_stream<score_t>* __restrict sout) {
    writeincr(sout, score_shard<0>(xin));
}

#define BDT_DEF_PLIO(S)                                                                \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         output_stream<score_t>* __restrict sout) {                    \
        writeincr(sout, score_shard<S>(xin));                                          \
    }

namespace {

// The packet TYPE field, which selects nothing here: routing is by packet ID, and every
// packet in this graph is the same kind of thing.
constexpr unsigned BDT_PKT_TYPE = 0;

__attribute__((always_inline))
static inline void write_partial(output_pktstream* pout, unsigned idx, const vscore& v) {
    alignas(32) score_t buf[W];
    aie::store_v(buf, v);
    writeHeader(pout, BDT_PKT_TYPE, getPacketid(pout, idx));
    for (unsigned j = 0; j < W; j++) writeincr(pout, buf[j], j + 1 == W);
}

template <unsigned A>
__attribute__((always_inline))
static inline vscore reduce_partials(input_pktstream* pin) {
    alignas(32) score_t buf[W];
    vscore acc = aie::zeros<score_t, W>();
#pragma unroll
    for (unsigned p = 0; p < A; p++) {
        readincr(pin);  // header, discarded
        for (unsigned j = 0; j < W; j++) buf[j] = readincr(pin);
        acc = aie::add(acc, aie::load_v<W>(buf));
    }
    return acc;
}

}  // namespace

// A payload tile: its own partial, as a packet. Destination 0 of its output port,
// because an untapped tile has only one.
#define BDT_DEF_PKT(S)                                                                 \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         output_pktstream* __restrict pout) {                          \
        write_partial(pout, 0, score_shard<S>(xin));                                   \
    }

#define BDT_DEF_PKT_TAP(S)                                                             \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         output_pktstream* __restrict pout) {                          \
        writeHeader(pout, BDT_PKT_TYPE, getPacketid(pout, 0));                         \
        writeincr(pout, (score_t)S, true);                                             \
        write_partial(pout, 1, score_shard<S>(xin));                                   \
    }

// An interior reducer: A partials in, their sum out as one packet.
#define BDT_DEF_REDUCE(A)                                                              \
    void bdt_qs_reduce_##A(input_pktstream* __restrict pin,                            \
                           output_pktstream* __restrict pout) {                        \
        write_partial(pout, 0, reduce_partials<A>(pin));                               \
    }

#define BDT_DEF_REDUCE_ROOT(A)                                                         \
    void bdt_qs_reduce_root_##A(input_pktstream* __restrict pin,                       \
                                output_stream<score_t>* __restrict sout) {             \
        writeincr(sout, aie::add(reduce_partials<A>(pin),                              \
                                 aie::broadcast<score_t, W>(bdtm::INIT_PREDICT)));     \
    }

#if BDT_MERGE_REDUCE
BDT_DEF_REDUCE(2)      BDT_DEF_REDUCE(3)      BDT_DEF_REDUCE(4)
BDT_DEF_REDUCE_ROOT(2) BDT_DEF_REDUCE_ROOT(3) BDT_DEF_REDUCE_ROOT(4)
#endif

// The payload role this build maps, and tile 0's, which may carry the tap. Mirrors the
// header's BDT_DECL_ROLE exactly.
#if BDT_FEED_MEMTILE
#  define BDT_DEF_ROLE(S)  BDT_DEF_BUF(S)
#elif BDT_MERGE_PLIO
#  define BDT_DEF_ROLE(S)  BDT_DEF_PLIO(S)
#elif BDT_MERGE_REDUCE
#  define BDT_DEF_ROLE(S)  BDT_DEF_PKT(S)
#endif
#if BDT_MERGE_REDUCE && BDT_TAP
#  define BDT_DEF_ROLE0(S) BDT_DEF_PKT_TAP(S)
#else
#  define BDT_DEF_ROLE0(S) BDT_DEF_ROLE(S)
#endif

#define BDT_DEF_HEAD(S)                                                                \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         output_cascade<score_t>* __restrict cout) {                   \
        writeincr(cout, score_shard<S>(xin));                                          \
    }

#define BDT_DEF_HEAD_TAP(S)                                                            \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         output_cascade<score_t>* __restrict cout,                     \
                         output_stream<score_t>* __restrict tap) {                     \
        writeincr(tap, (score_t)S);                                                    \
        writeincr(cout, score_shard<S>(xin));                                               \
    }

#define BDT_DEF_LINK(S)                                                                \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         input_cascade<score_t>* __restrict cin,                       \
                         output_cascade<score_t>* __restrict cout) {                   \
        const vscore mine = score_shard<S>(xin);                                       \
        writeincr(cout, aie::add(readincr_v<W>(cin), mine));                           \
    }

#define BDT_DEF_TAIL(S)                                                                \
    void bdt_qs_tile_##S(input_stream<feat_t>* __restrict xin,                         \
                         input_cascade<score_t>* __restrict cin,                       \
                         output_stream<score_t>* __restrict sout) {                    \
        const vscore mine = score_shard<S>(xin);                                       \
        writeincr(sout, aie::add(readincr_v<W>(cin), mine));                           \
    }

// The definitions this build maps, and only those -- same guards as the header, so a
// declaration and its definition cannot disagree about which role a tile plays.
#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#if BDT_MERGE_PLIO || BDT_MERGE_REDUCE
#if BDT_N_TILES > 0
BDT_DEF_ROLE0(0)
#endif
#if BDT_N_TILES > 1
BDT_DEF_ROLE(1)
#endif
#if BDT_N_TILES > 2
BDT_DEF_ROLE(2)
#endif
#if BDT_N_TILES > 3
BDT_DEF_ROLE(3)
#endif
#if BDT_N_TILES > 4
BDT_DEF_ROLE(4)
#endif
#if BDT_N_TILES > 5
BDT_DEF_ROLE(5)
#endif
#if BDT_N_TILES > 6
BDT_DEF_ROLE(6)
#endif
#if BDT_N_TILES > 7
BDT_DEF_ROLE(7)
#endif
#if BDT_N_TILES > 8
BDT_DEF_ROLE(8)
#endif
#if BDT_N_TILES > 9
BDT_DEF_ROLE(9)
#endif
#if BDT_N_TILES > 10
BDT_DEF_ROLE(10)
#endif
#if BDT_N_TILES > 11
BDT_DEF_ROLE(11)
#endif
#if BDT_N_TILES > 12
BDT_DEF_ROLE(12)
#endif
#if BDT_N_TILES > 13
BDT_DEF_ROLE(13)
#endif
#if BDT_N_TILES > 14
BDT_DEF_ROLE(14)
#endif
#if BDT_N_TILES > 15
BDT_DEF_ROLE(15)
#endif
#if BDT_N_TILES > 16
BDT_DEF_ROLE(16)
#endif
#if BDT_N_TILES > 17
BDT_DEF_ROLE(17)
#endif
#if BDT_N_TILES > 18
BDT_DEF_ROLE(18)
#endif
#if BDT_N_TILES > 19
BDT_DEF_ROLE(19)
#endif
#if BDT_N_TILES > 20
BDT_DEF_ROLE(20)
#endif
#if BDT_N_TILES > 21
BDT_DEF_ROLE(21)
#endif
#if BDT_N_TILES > 22
BDT_DEF_ROLE(22)
#endif
#if BDT_N_TILES > 23
BDT_DEF_ROLE(23)
#endif
#if BDT_N_TILES > 24
BDT_DEF_ROLE(24)
#endif
#if BDT_N_TILES > 25
BDT_DEF_ROLE(25)
#endif
#if BDT_N_TILES > 26
BDT_DEF_ROLE(26)
#endif
#if BDT_N_TILES > 27
BDT_DEF_ROLE(27)
#endif
#if BDT_N_TILES > 28
BDT_DEF_ROLE(28)
#endif
#if BDT_N_TILES > 29
BDT_DEF_ROLE(29)
#endif
#if BDT_N_TILES > 30
BDT_DEF_ROLE(30)
#endif
#if BDT_N_TILES > 31
BDT_DEF_ROLE(31)
#endif
#if BDT_N_TILES > 32
BDT_DEF_ROLE(32)
#endif
#if BDT_N_TILES > 33
BDT_DEF_ROLE(33)
#endif
#if BDT_N_TILES > 34
BDT_DEF_ROLE(34)
#endif
#if BDT_N_TILES > 35
BDT_DEF_ROLE(35)
#endif
#if BDT_N_TILES > 36
BDT_DEF_ROLE(36)
#endif
#if BDT_N_TILES > 37
BDT_DEF_ROLE(37)
#endif
#if BDT_N_TILES > 38
BDT_DEF_ROLE(38)
#endif
#if BDT_N_TILES > 39
BDT_DEF_ROLE(39)
#endif
#if BDT_N_TILES > 40
BDT_DEF_ROLE(40)
#endif
#if BDT_N_TILES > 41
BDT_DEF_ROLE(41)
#endif
#if BDT_N_TILES > 42
BDT_DEF_ROLE(42)
#endif
#if BDT_N_TILES > 43
BDT_DEF_ROLE(43)
#endif
#if BDT_N_TILES > 44
BDT_DEF_ROLE(44)
#endif
#if BDT_N_TILES > 45
BDT_DEF_ROLE(45)
#endif
#if BDT_N_TILES > 46
BDT_DEF_ROLE(46)
#endif
#if BDT_N_TILES > 47
BDT_DEF_ROLE(47)
#endif
#if BDT_N_TILES > 48
BDT_DEF_ROLE(48)
#endif
#if BDT_N_TILES > 49
BDT_DEF_ROLE(49)
#endif
#if BDT_N_TILES > 50
BDT_DEF_ROLE(50)
#endif
#if BDT_N_TILES > 51
BDT_DEF_ROLE(51)
#endif
#if BDT_N_TILES > 52
BDT_DEF_ROLE(52)
#endif
#if BDT_N_TILES > 53
BDT_DEF_ROLE(53)
#endif
#if BDT_N_TILES > 54
BDT_DEF_ROLE(54)
#endif
#if BDT_N_TILES > 55
BDT_DEF_ROLE(55)
#endif
#if BDT_N_TILES > 56
BDT_DEF_ROLE(56)
#endif
#if BDT_N_TILES > 57
BDT_DEF_ROLE(57)
#endif
#if BDT_N_TILES > 58
BDT_DEF_ROLE(58)
#endif
#if BDT_N_TILES > 59
BDT_DEF_ROLE(59)
#endif
#if BDT_N_TILES > 60
BDT_DEF_ROLE(60)
#endif
#if BDT_N_TILES > 61
BDT_DEF_ROLE(61)
#endif
#if BDT_N_TILES > 62
BDT_DEF_ROLE(62)
#endif
#if BDT_N_TILES > 63
BDT_DEF_ROLE(63)
#endif
#else
#if BDT_TAP
BDT_DEF_HEAD_TAP(0)
#else
BDT_DEF_HEAD(0)
#endif
#if BDT_N_TILES == 2
BDT_DEF_TAIL(1)
#elif BDT_N_TILES > 2
BDT_DEF_LINK(1)
#endif
#if BDT_N_TILES == 3
BDT_DEF_TAIL(2)
#elif BDT_N_TILES > 3
BDT_DEF_LINK(2)
#endif
#if BDT_N_TILES == 4
BDT_DEF_TAIL(3)
#elif BDT_N_TILES > 4
BDT_DEF_LINK(3)
#endif
#if BDT_N_TILES == 5
BDT_DEF_TAIL(4)
#elif BDT_N_TILES > 5
BDT_DEF_LINK(4)
#endif
#if BDT_N_TILES == 6
BDT_DEF_TAIL(5)
#elif BDT_N_TILES > 6
BDT_DEF_LINK(5)
#endif
#if BDT_N_TILES == 7
BDT_DEF_TAIL(6)
#elif BDT_N_TILES > 7
BDT_DEF_LINK(6)
#endif
#if BDT_N_TILES == 8
BDT_DEF_TAIL(7)
#elif BDT_N_TILES > 8
BDT_DEF_LINK(7)
#endif
#if BDT_N_TILES == 9
BDT_DEF_TAIL(8)
#elif BDT_N_TILES > 9
BDT_DEF_LINK(8)
#endif
#if BDT_N_TILES == 10
BDT_DEF_TAIL(9)
#elif BDT_N_TILES > 10
BDT_DEF_LINK(9)
#endif
#if BDT_N_TILES == 11
BDT_DEF_TAIL(10)
#elif BDT_N_TILES > 11
BDT_DEF_LINK(10)
#endif
#if BDT_N_TILES == 12
BDT_DEF_TAIL(11)
#elif BDT_N_TILES > 12
BDT_DEF_LINK(11)
#endif
#if BDT_N_TILES == 13
BDT_DEF_TAIL(12)
#elif BDT_N_TILES > 13
BDT_DEF_LINK(12)
#endif
#if BDT_N_TILES == 14
BDT_DEF_TAIL(13)
#elif BDT_N_TILES > 14
BDT_DEF_LINK(13)
#endif
#if BDT_N_TILES == 15
BDT_DEF_TAIL(14)
#elif BDT_N_TILES > 15
BDT_DEF_LINK(14)
#endif
#if BDT_N_TILES == 16
BDT_DEF_TAIL(15)
#elif BDT_N_TILES > 16
BDT_DEF_LINK(15)
#endif
#if BDT_N_TILES == 17
BDT_DEF_TAIL(16)
#elif BDT_N_TILES > 17
BDT_DEF_LINK(16)
#endif
#if BDT_N_TILES == 18
BDT_DEF_TAIL(17)
#elif BDT_N_TILES > 18
BDT_DEF_LINK(17)
#endif
#if BDT_N_TILES == 19
BDT_DEF_TAIL(18)
#elif BDT_N_TILES > 19
BDT_DEF_LINK(18)
#endif
#if BDT_N_TILES == 20
BDT_DEF_TAIL(19)
#elif BDT_N_TILES > 20
BDT_DEF_LINK(19)
#endif
#if BDT_N_TILES == 21
BDT_DEF_TAIL(20)
#elif BDT_N_TILES > 21
BDT_DEF_LINK(20)
#endif
#if BDT_N_TILES == 22
BDT_DEF_TAIL(21)
#elif BDT_N_TILES > 22
BDT_DEF_LINK(21)
#endif
#if BDT_N_TILES == 23
BDT_DEF_TAIL(22)
#elif BDT_N_TILES > 23
BDT_DEF_LINK(22)
#endif
#if BDT_N_TILES == 24
BDT_DEF_TAIL(23)
#elif BDT_N_TILES > 24
BDT_DEF_LINK(23)
#endif
#if BDT_N_TILES == 25
BDT_DEF_TAIL(24)
#elif BDT_N_TILES > 25
BDT_DEF_LINK(24)
#endif
#if BDT_N_TILES == 26
BDT_DEF_TAIL(25)
#elif BDT_N_TILES > 26
BDT_DEF_LINK(25)
#endif
#if BDT_N_TILES == 27
BDT_DEF_TAIL(26)
#elif BDT_N_TILES > 27
BDT_DEF_LINK(26)
#endif
#if BDT_N_TILES == 28
BDT_DEF_TAIL(27)
#elif BDT_N_TILES > 28
BDT_DEF_LINK(27)
#endif
#if BDT_N_TILES == 29
BDT_DEF_TAIL(28)
#elif BDT_N_TILES > 29
BDT_DEF_LINK(28)
#endif
#if BDT_N_TILES == 30
BDT_DEF_TAIL(29)
#elif BDT_N_TILES > 30
BDT_DEF_LINK(29)
#endif
#if BDT_N_TILES == 31
BDT_DEF_TAIL(30)
#elif BDT_N_TILES > 31
BDT_DEF_LINK(30)
#endif
#if BDT_N_TILES == 32
BDT_DEF_TAIL(31)
#elif BDT_N_TILES > 32
BDT_DEF_LINK(31)
#endif
#endif
#endif
