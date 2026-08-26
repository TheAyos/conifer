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
        [&](unsigned) __attribute__((always_inline)) {
            return readincr_v<W>(xin);
        });
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

// The payload role this build maps, and tile 0's, which may carry the tap. Mirrors the
// header's BDT_DECL_ROLE exactly.
#if BDT_FEED_MEMTILE
#  define BDT_DEF_ROLE(S)  BDT_DEF_BUF(S)
#else
#  define BDT_DEF_ROLE(S)  BDT_DEF_PLIO(S)
#endif
#define BDT_DEF_ROLE0(S) BDT_DEF_ROLE(S)

// The definitions this build maps, and only those -- same guards as the header, so a
// declaration and its definition cannot disagree about which role a tile plays.
#if BDT_SPLIT_TREE && BDT_N_TILES > 1
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
#if BDT_N_TILES == 2
#elif BDT_N_TILES > 2
#endif
#if BDT_N_TILES == 3
#elif BDT_N_TILES > 3
#endif
#if BDT_N_TILES == 4
#elif BDT_N_TILES > 4
#endif
#if BDT_N_TILES == 5
#elif BDT_N_TILES > 5
#endif
#if BDT_N_TILES == 6
#elif BDT_N_TILES > 6
#endif
#if BDT_N_TILES == 7
#elif BDT_N_TILES > 7
#endif
#if BDT_N_TILES == 8
#elif BDT_N_TILES > 8
#endif
#if BDT_N_TILES == 9
#elif BDT_N_TILES > 9
#endif
#if BDT_N_TILES == 10
#elif BDT_N_TILES > 10
#endif
#if BDT_N_TILES == 11
#elif BDT_N_TILES > 11
#endif
#if BDT_N_TILES == 12
#elif BDT_N_TILES > 12
#endif
#if BDT_N_TILES == 13
#elif BDT_N_TILES > 13
#endif
#if BDT_N_TILES == 14
#elif BDT_N_TILES > 14
#endif
#if BDT_N_TILES == 15
#elif BDT_N_TILES > 15
#endif
#if BDT_N_TILES == 16
#elif BDT_N_TILES > 16
#endif
#if BDT_N_TILES == 17
#elif BDT_N_TILES > 17
#endif
#if BDT_N_TILES == 18
#elif BDT_N_TILES > 18
#endif
#if BDT_N_TILES == 19
#elif BDT_N_TILES > 19
#endif
#if BDT_N_TILES == 20
#elif BDT_N_TILES > 20
#endif
#if BDT_N_TILES == 21
#elif BDT_N_TILES > 21
#endif
#if BDT_N_TILES == 22
#elif BDT_N_TILES > 22
#endif
#if BDT_N_TILES == 23
#elif BDT_N_TILES > 23
#endif
#if BDT_N_TILES == 24
#elif BDT_N_TILES > 24
#endif
#if BDT_N_TILES == 25
#elif BDT_N_TILES > 25
#endif
#if BDT_N_TILES == 26
#elif BDT_N_TILES > 26
#endif
#if BDT_N_TILES == 27
#elif BDT_N_TILES > 27
#endif
#if BDT_N_TILES == 28
#elif BDT_N_TILES > 28
#endif
#if BDT_N_TILES == 29
#elif BDT_N_TILES > 29
#endif
#if BDT_N_TILES == 30
#elif BDT_N_TILES > 30
#endif
#if BDT_N_TILES == 31
#elif BDT_N_TILES > 31
#endif
#if BDT_N_TILES == 32
#elif BDT_N_TILES > 32
#endif
#endif
