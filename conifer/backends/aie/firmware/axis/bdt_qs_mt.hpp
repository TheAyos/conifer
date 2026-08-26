#pragma once
#include <adf.h>
#include "params.h"

#define BDT_DECL_PLIO(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, output_stream<bdtm::score_t>*);
#define BDT_DECL_HEAD(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, output_cascade<bdtm::score_t>*);
#define BDT_DECL_HEAD_TAP(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, output_cascade<bdtm::score_t>*, \
                        output_stream<bdtm::score_t>*);
#define BDT_DECL_LINK(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, input_cascade<bdtm::score_t>*, \
                         output_cascade<bdtm::score_t>*);
// TREE-SPLIT, merge_mode=reduce (ADR-0021 §4). The partial leaves as a PACKET on the
// tile's one stream output, to be ordered-merged with its k-1 siblings onto a reducer.
#define BDT_DECL_PKT(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, output_pktstream*);
#define BDT_DECL_BUF(S) \
    void bdt_qs_tile_##S(adf::input_buffer<bdtm::feat_t>&, output_stream<bdtm::score_t>*);
#define BDT_DECL_TAIL(S) \
    void bdt_qs_tile_##S(input_stream<bdtm::feat_t>*, input_cascade<bdtm::score_t>*, \
                         output_stream<bdtm::score_t>*);

#define BDT_DECL_REDUCE(A) \
    void bdt_qs_reduce_##A(input_pktstream*, output_pktstream*);
#define BDT_DECL_REDUCE_ROOT(A) \
    void bdt_qs_reduce_root_##A(input_pktstream*, output_stream<bdtm::score_t>*);

#if BDT_FEED_MEMTILE
#  define BDT_DECL_ROLE(S)  BDT_DECL_BUF(S)
#elif BDT_MERGE_PLIO
#  define BDT_DECL_ROLE(S)  BDT_DECL_PLIO(S)
#elif BDT_MERGE_REDUCE
#  define BDT_DECL_ROLE(S)  BDT_DECL_PKT(S)
#endif

// Whole ensemble: N=1, and every tile of a sample-split (they differ only in which
// samples arrive, so they share one symbol).
void bdt_qs_mt(input_stream<bdtm::feat_t>* xin,
               output_stream<bdtm::score_t>* sout);

#if BDT_MERGE_REDUCE
BDT_DECL_REDUCE(2)      BDT_DECL_REDUCE(3)      BDT_DECL_REDUCE(4)
BDT_DECL_REDUCE_ROOT(2) BDT_DECL_REDUCE_ROOT(3) BDT_DECL_REDUCE_ROOT(4)
#endif

#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#if BDT_MERGE_PLIO || BDT_MERGE_REDUCE
#if BDT_N_TILES > 0
BDT_DECL_ROLE(0)
#endif
#if BDT_N_TILES > 1
BDT_DECL_ROLE(1)
#endif
#if BDT_N_TILES > 2
BDT_DECL_ROLE(2)
#endif
#if BDT_N_TILES > 3
BDT_DECL_ROLE(3)
#endif
#if BDT_N_TILES > 4
BDT_DECL_ROLE(4)
#endif
#if BDT_N_TILES > 5
BDT_DECL_ROLE(5)
#endif
#if BDT_N_TILES > 6
BDT_DECL_ROLE(6)
#endif
#if BDT_N_TILES > 7
BDT_DECL_ROLE(7)
#endif
#if BDT_N_TILES > 8
BDT_DECL_ROLE(8)
#endif
#if BDT_N_TILES > 9
BDT_DECL_ROLE(9)
#endif
#if BDT_N_TILES > 10
BDT_DECL_ROLE(10)
#endif
#if BDT_N_TILES > 11
BDT_DECL_ROLE(11)
#endif
#if BDT_N_TILES > 12
BDT_DECL_ROLE(12)
#endif
#if BDT_N_TILES > 13
BDT_DECL_ROLE(13)
#endif
#if BDT_N_TILES > 14
BDT_DECL_ROLE(14)
#endif
#if BDT_N_TILES > 15
BDT_DECL_ROLE(15)
#endif
#if BDT_N_TILES > 16
BDT_DECL_ROLE(16)
#endif
#if BDT_N_TILES > 17
BDT_DECL_ROLE(17)
#endif
#if BDT_N_TILES > 18
BDT_DECL_ROLE(18)
#endif
#if BDT_N_TILES > 19
BDT_DECL_ROLE(19)
#endif
#if BDT_N_TILES > 20
BDT_DECL_ROLE(20)
#endif
#if BDT_N_TILES > 21
BDT_DECL_ROLE(21)
#endif
#if BDT_N_TILES > 22
BDT_DECL_ROLE(22)
#endif
#if BDT_N_TILES > 23
BDT_DECL_ROLE(23)
#endif
#if BDT_N_TILES > 24
BDT_DECL_ROLE(24)
#endif
#if BDT_N_TILES > 25
BDT_DECL_ROLE(25)
#endif
#if BDT_N_TILES > 26
BDT_DECL_ROLE(26)
#endif
#if BDT_N_TILES > 27
BDT_DECL_ROLE(27)
#endif
#if BDT_N_TILES > 28
BDT_DECL_ROLE(28)
#endif
#if BDT_N_TILES > 29
BDT_DECL_ROLE(29)
#endif
#if BDT_N_TILES > 30
BDT_DECL_ROLE(30)
#endif
#if BDT_N_TILES > 31
BDT_DECL_ROLE(31)
#endif
#if BDT_N_TILES > 32
BDT_DECL_ROLE(32)
#endif
#if BDT_N_TILES > 33
BDT_DECL_ROLE(33)
#endif
#if BDT_N_TILES > 34
BDT_DECL_ROLE(34)
#endif
#if BDT_N_TILES > 35
BDT_DECL_ROLE(35)
#endif
#if BDT_N_TILES > 36
BDT_DECL_ROLE(36)
#endif
#if BDT_N_TILES > 37
BDT_DECL_ROLE(37)
#endif
#if BDT_N_TILES > 38
BDT_DECL_ROLE(38)
#endif
#if BDT_N_TILES > 39
BDT_DECL_ROLE(39)
#endif
#if BDT_N_TILES > 40
BDT_DECL_ROLE(40)
#endif
#if BDT_N_TILES > 41
BDT_DECL_ROLE(41)
#endif
#if BDT_N_TILES > 42
BDT_DECL_ROLE(42)
#endif
#if BDT_N_TILES > 43
BDT_DECL_ROLE(43)
#endif
#if BDT_N_TILES > 44
BDT_DECL_ROLE(44)
#endif
#if BDT_N_TILES > 45
BDT_DECL_ROLE(45)
#endif
#if BDT_N_TILES > 46
BDT_DECL_ROLE(46)
#endif
#if BDT_N_TILES > 47
BDT_DECL_ROLE(47)
#endif
#if BDT_N_TILES > 48
BDT_DECL_ROLE(48)
#endif
#if BDT_N_TILES > 49
BDT_DECL_ROLE(49)
#endif
#if BDT_N_TILES > 50
BDT_DECL_ROLE(50)
#endif
#if BDT_N_TILES > 51
BDT_DECL_ROLE(51)
#endif
#if BDT_N_TILES > 52
BDT_DECL_ROLE(52)
#endif
#if BDT_N_TILES > 53
BDT_DECL_ROLE(53)
#endif
#if BDT_N_TILES > 54
BDT_DECL_ROLE(54)
#endif
#if BDT_N_TILES > 55
BDT_DECL_ROLE(55)
#endif
#if BDT_N_TILES > 56
BDT_DECL_ROLE(56)
#endif
#if BDT_N_TILES > 57
BDT_DECL_ROLE(57)
#endif
#if BDT_N_TILES > 58
BDT_DECL_ROLE(58)
#endif
#if BDT_N_TILES > 59
BDT_DECL_ROLE(59)
#endif
#if BDT_N_TILES > 60
BDT_DECL_ROLE(60)
#endif
#if BDT_N_TILES > 61
BDT_DECL_ROLE(61)
#endif
#if BDT_N_TILES > 62
BDT_DECL_ROLE(62)
#endif
#if BDT_N_TILES > 63
BDT_DECL_ROLE(63)
#endif
#else
#if BDT_TAP
BDT_DECL_HEAD_TAP(0)
#else
BDT_DECL_HEAD(0)
#endif
#if BDT_N_TILES == 2
BDT_DECL_TAIL(1)
#elif BDT_N_TILES > 2
BDT_DECL_LINK(1)
#endif
#if BDT_N_TILES == 3
BDT_DECL_TAIL(2)
#elif BDT_N_TILES > 3
BDT_DECL_LINK(2)
#endif
#if BDT_N_TILES == 4
BDT_DECL_TAIL(3)
#elif BDT_N_TILES > 4
BDT_DECL_LINK(3)
#endif
#if BDT_N_TILES == 5
BDT_DECL_TAIL(4)
#elif BDT_N_TILES > 5
BDT_DECL_LINK(4)
#endif
#if BDT_N_TILES == 6
BDT_DECL_TAIL(5)
#elif BDT_N_TILES > 6
BDT_DECL_LINK(5)
#endif
#if BDT_N_TILES == 7
BDT_DECL_TAIL(6)
#elif BDT_N_TILES > 7
BDT_DECL_LINK(6)
#endif
#if BDT_N_TILES == 8
BDT_DECL_TAIL(7)
#elif BDT_N_TILES > 8
BDT_DECL_LINK(7)
#endif
#if BDT_N_TILES == 9
BDT_DECL_TAIL(8)
#elif BDT_N_TILES > 9
BDT_DECL_LINK(8)
#endif
#if BDT_N_TILES == 10
BDT_DECL_TAIL(9)
#elif BDT_N_TILES > 10
BDT_DECL_LINK(9)
#endif
#if BDT_N_TILES == 11
BDT_DECL_TAIL(10)
#elif BDT_N_TILES > 11
BDT_DECL_LINK(10)
#endif
#if BDT_N_TILES == 12
BDT_DECL_TAIL(11)
#elif BDT_N_TILES > 12
BDT_DECL_LINK(11)
#endif
#if BDT_N_TILES == 13
BDT_DECL_TAIL(12)
#elif BDT_N_TILES > 13
BDT_DECL_LINK(12)
#endif
#if BDT_N_TILES == 14
BDT_DECL_TAIL(13)
#elif BDT_N_TILES > 14
BDT_DECL_LINK(13)
#endif
#if BDT_N_TILES == 15
BDT_DECL_TAIL(14)
#elif BDT_N_TILES > 15
BDT_DECL_LINK(14)
#endif
#if BDT_N_TILES == 16
BDT_DECL_TAIL(15)
#elif BDT_N_TILES > 16
BDT_DECL_LINK(15)
#endif
#if BDT_N_TILES == 17
BDT_DECL_TAIL(16)
#elif BDT_N_TILES > 17
BDT_DECL_LINK(16)
#endif
#if BDT_N_TILES == 18
BDT_DECL_TAIL(17)
#elif BDT_N_TILES > 18
BDT_DECL_LINK(17)
#endif
#if BDT_N_TILES == 19
BDT_DECL_TAIL(18)
#elif BDT_N_TILES > 19
BDT_DECL_LINK(18)
#endif
#if BDT_N_TILES == 20
BDT_DECL_TAIL(19)
#elif BDT_N_TILES > 20
BDT_DECL_LINK(19)
#endif
#if BDT_N_TILES == 21
BDT_DECL_TAIL(20)
#elif BDT_N_TILES > 21
BDT_DECL_LINK(20)
#endif
#if BDT_N_TILES == 22
BDT_DECL_TAIL(21)
#elif BDT_N_TILES > 22
BDT_DECL_LINK(21)
#endif
#if BDT_N_TILES == 23
BDT_DECL_TAIL(22)
#elif BDT_N_TILES > 23
BDT_DECL_LINK(22)
#endif
#if BDT_N_TILES == 24
BDT_DECL_TAIL(23)
#elif BDT_N_TILES > 24
BDT_DECL_LINK(23)
#endif
#if BDT_N_TILES == 25
BDT_DECL_TAIL(24)
#elif BDT_N_TILES > 25
BDT_DECL_LINK(24)
#endif
#if BDT_N_TILES == 26
BDT_DECL_TAIL(25)
#elif BDT_N_TILES > 26
BDT_DECL_LINK(25)
#endif
#if BDT_N_TILES == 27
BDT_DECL_TAIL(26)
#elif BDT_N_TILES > 27
BDT_DECL_LINK(26)
#endif
#if BDT_N_TILES == 28
BDT_DECL_TAIL(27)
#elif BDT_N_TILES > 28
BDT_DECL_LINK(27)
#endif
#if BDT_N_TILES == 29
BDT_DECL_TAIL(28)
#elif BDT_N_TILES > 29
BDT_DECL_LINK(28)
#endif
#if BDT_N_TILES == 30
BDT_DECL_TAIL(29)
#elif BDT_N_TILES > 30
BDT_DECL_LINK(29)
#endif
#if BDT_N_TILES == 31
BDT_DECL_TAIL(30)
#elif BDT_N_TILES > 31
BDT_DECL_LINK(30)
#endif
#if BDT_N_TILES == 32
BDT_DECL_TAIL(31)
#elif BDT_N_TILES > 32
BDT_DECL_LINK(31)
#endif
#endif
#endif
