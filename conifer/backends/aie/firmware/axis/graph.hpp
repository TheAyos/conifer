#pragma once
#include <cstdio>
#include <string>

#include "params.h"
#include "bdt_qs_mt.hpp"

using namespace adf;

#if defined(__has_include)
#  if __has_include("xin_file.h")
#    include "xin_file.h"
#  endif
#endif
#ifndef XIN_FILE
#define XIN_FILE "../../gen/out/t32-d4-f16/int16/data/X_fm_w32_plio64.dat"
#endif
namespace bdtmt {

constexpr unsigned N_IN  = FEED_MEMTILE ? N_MEMTILE
                         : (SPLIT_TREE && !FEED_PLIO) ? 1u : N_TILES;
constexpr unsigned N_OUT = N_TILES;  // every tile emits its own partial

inline std::string tile_in_file(const std::string& base, unsigned t) {
    const std::string key = ".n" + std::to_string(N_TILES) + "d" + std::to_string(DELTA);
    const auto dot = base.rfind('.');
    const std::string tag = key + ".t" + std::to_string(t);
    return dot == std::string::npos ? base + tag
                                    : base.substr(0, dot) + tag + base.substr(dot);
}

}  // namespace bdtmt

class theGraph : public graph {
private:
    kernel k[bdtmt::N_TILES];
#if BDT_FEED_MEMTILE
    shared_buffer<bdtm::feat_t> mtx[bdtmt::N_MEMTILE];
#endif

    // Every tile gets the same sizing and source; only its symbol differs.
    void configure(kernel& kk) {
        source(kk) = "src/bdt_qs_mt.cpp";
        runtime<ratio>(kk) = 1.0;

        constexpr unsigned KIB = 1024;
        constexpr unsigned TABLES =
              sizeof(bdtm::QT_THR)  + sizeof(bdtm::QT_BV)
            + sizeof(bdtm::QT_FEAT) + sizeof(bdtm::LEAVES)
            + sizeof(bdtm::INIT_V);
        constexpr unsigned XBYTES = bdtm::N_FEATURES * BDT_W * sizeof(bdtm::feat_t);
        heap_size(kk)  = ((TABLES + 2 * KIB - 1) / KIB) * KIB;
        stack_size(kk) = ((XBYTES + 4 * KIB - 1) / KIB) * KIB;
    }

public:
    input_plio  xin[bdtmt::N_IN];
    output_plio sout[bdtmt::N_OUT];

    theGraph() {
#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#if BDT_N_TILES > 0
        k[0] = kernel::create(bdt_qs_tile_0);
#endif
#if BDT_N_TILES > 1
        k[1] = kernel::create(bdt_qs_tile_1);
#endif
#if BDT_N_TILES > 2
        k[2] = kernel::create(bdt_qs_tile_2);
#endif
#if BDT_N_TILES > 3
        k[3] = kernel::create(bdt_qs_tile_3);
#endif
#if BDT_N_TILES > 4
        k[4] = kernel::create(bdt_qs_tile_4);
#endif
#if BDT_N_TILES > 5
        k[5] = kernel::create(bdt_qs_tile_5);
#endif
#if BDT_N_TILES > 6
        k[6] = kernel::create(bdt_qs_tile_6);
#endif
#if BDT_N_TILES > 7
        k[7] = kernel::create(bdt_qs_tile_7);
#endif
#if BDT_N_TILES > 8
        k[8] = kernel::create(bdt_qs_tile_8);
#endif
#if BDT_N_TILES > 9
        k[9] = kernel::create(bdt_qs_tile_9);
#endif
#if BDT_N_TILES > 10
        k[10] = kernel::create(bdt_qs_tile_10);
#endif
#if BDT_N_TILES > 11
        k[11] = kernel::create(bdt_qs_tile_11);
#endif
#if BDT_N_TILES > 12
        k[12] = kernel::create(bdt_qs_tile_12);
#endif
#if BDT_N_TILES > 13
        k[13] = kernel::create(bdt_qs_tile_13);
#endif
#if BDT_N_TILES > 14
        k[14] = kernel::create(bdt_qs_tile_14);
#endif
#if BDT_N_TILES > 15
        k[15] = kernel::create(bdt_qs_tile_15);
#endif
#if BDT_N_TILES > 16
        k[16] = kernel::create(bdt_qs_tile_16);
#endif
#if BDT_N_TILES > 17
        k[17] = kernel::create(bdt_qs_tile_17);
#endif
#if BDT_N_TILES > 18
        k[18] = kernel::create(bdt_qs_tile_18);
#endif
#if BDT_N_TILES > 19
        k[19] = kernel::create(bdt_qs_tile_19);
#endif
#if BDT_N_TILES > 20
        k[20] = kernel::create(bdt_qs_tile_20);
#endif
#if BDT_N_TILES > 21
        k[21] = kernel::create(bdt_qs_tile_21);
#endif
#if BDT_N_TILES > 22
        k[22] = kernel::create(bdt_qs_tile_22);
#endif
#if BDT_N_TILES > 23
        k[23] = kernel::create(bdt_qs_tile_23);
#endif
#if BDT_N_TILES > 24
        k[24] = kernel::create(bdt_qs_tile_24);
#endif
#if BDT_N_TILES > 25
        k[25] = kernel::create(bdt_qs_tile_25);
#endif
#if BDT_N_TILES > 26
        k[26] = kernel::create(bdt_qs_tile_26);
#endif
#if BDT_N_TILES > 27
        k[27] = kernel::create(bdt_qs_tile_27);
#endif
#if BDT_N_TILES > 28
        k[28] = kernel::create(bdt_qs_tile_28);
#endif
#if BDT_N_TILES > 29
        k[29] = kernel::create(bdt_qs_tile_29);
#endif
#if BDT_N_TILES > 30
        k[30] = kernel::create(bdt_qs_tile_30);
#endif
#if BDT_N_TILES > 31
        k[31] = kernel::create(bdt_qs_tile_31);
#endif
#if BDT_N_TILES > 32
        k[32] = kernel::create(bdt_qs_tile_32);
#endif
#if BDT_N_TILES > 33
        k[33] = kernel::create(bdt_qs_tile_33);
#endif
#if BDT_N_TILES > 34
        k[34] = kernel::create(bdt_qs_tile_34);
#endif
#if BDT_N_TILES > 35
        k[35] = kernel::create(bdt_qs_tile_35);
#endif
#if BDT_N_TILES > 36
        k[36] = kernel::create(bdt_qs_tile_36);
#endif
#if BDT_N_TILES > 37
        k[37] = kernel::create(bdt_qs_tile_37);
#endif
#if BDT_N_TILES > 38
        k[38] = kernel::create(bdt_qs_tile_38);
#endif
#if BDT_N_TILES > 39
        k[39] = kernel::create(bdt_qs_tile_39);
#endif
#if BDT_N_TILES > 40
        k[40] = kernel::create(bdt_qs_tile_40);
#endif
#if BDT_N_TILES > 41
        k[41] = kernel::create(bdt_qs_tile_41);
#endif
#if BDT_N_TILES > 42
        k[42] = kernel::create(bdt_qs_tile_42);
#endif
#if BDT_N_TILES > 43
        k[43] = kernel::create(bdt_qs_tile_43);
#endif
#if BDT_N_TILES > 44
        k[44] = kernel::create(bdt_qs_tile_44);
#endif
#if BDT_N_TILES > 45
        k[45] = kernel::create(bdt_qs_tile_45);
#endif
#if BDT_N_TILES > 46
        k[46] = kernel::create(bdt_qs_tile_46);
#endif
#if BDT_N_TILES > 47
        k[47] = kernel::create(bdt_qs_tile_47);
#endif
#if BDT_N_TILES > 48
        k[48] = kernel::create(bdt_qs_tile_48);
#endif
#if BDT_N_TILES > 49
        k[49] = kernel::create(bdt_qs_tile_49);
#endif
#if BDT_N_TILES > 50
        k[50] = kernel::create(bdt_qs_tile_50);
#endif
#if BDT_N_TILES > 51
        k[51] = kernel::create(bdt_qs_tile_51);
#endif
#if BDT_N_TILES > 52
        k[52] = kernel::create(bdt_qs_tile_52);
#endif
#if BDT_N_TILES > 53
        k[53] = kernel::create(bdt_qs_tile_53);
#endif
#if BDT_N_TILES > 54
        k[54] = kernel::create(bdt_qs_tile_54);
#endif
#if BDT_N_TILES > 55
        k[55] = kernel::create(bdt_qs_tile_55);
#endif
#if BDT_N_TILES > 56
        k[56] = kernel::create(bdt_qs_tile_56);
#endif
#if BDT_N_TILES > 57
        k[57] = kernel::create(bdt_qs_tile_57);
#endif
#if BDT_N_TILES > 58
        k[58] = kernel::create(bdt_qs_tile_58);
#endif
#if BDT_N_TILES > 59
        k[59] = kernel::create(bdt_qs_tile_59);
#endif
#if BDT_N_TILES > 60
        k[60] = kernel::create(bdt_qs_tile_60);
#endif
#if BDT_N_TILES > 61
        k[61] = kernel::create(bdt_qs_tile_61);
#endif
#if BDT_N_TILES > 62
        k[62] = kernel::create(bdt_qs_tile_62);
#endif
#if BDT_N_TILES > 63
        k[63] = kernel::create(bdt_qs_tile_63);
#endif
#else
        for (unsigned i = 0; i < bdtmt::N_TILES; i++)
            k[i] = kernel::create(bdt_qs_mt);
#endif
        for (unsigned i = 0; i < bdtmt::N_TILES; i++) configure(k[i]);

        for (unsigned i = 0; i < bdtmt::N_IN; i++) {
            const std::string name = "xin" + std::to_string(i);
            const std::string file =
                  (bdtmt::FEED_MEMTILE || bdtmt::SPLIT_TREE || bdtmt::N_TILES == 1)
                      ? std::string(XIN_FILE)
                      : bdtmt::tile_in_file(XIN_FILE, i);
            // The offered rate, matched to the mapping under a latency run and left at
            // the transport default otherwise -- see params.h.
            xin[i] = input_plio::create(name.c_str(), plio_64_bits, file.c_str(),
                                        bdtmt::PLIO_RATE);
        }
        for (unsigned i = 0; i < bdtmt::N_OUT; i++) {
            const std::string name = i == 0 ? "scores" : "scores" + std::to_string(i);
            const std::string file = i == 0 ? "scores.dat"
                                            : "scores.t" + std::to_string(i) + ".dat";
            sout[i] = output_plio::create(name.c_str(), plio_32_bits, file.c_str(),
                                          bdtmt::PLIO_RATE);
        }

#if BDT_FEED_MEMTILE
        for (unsigned m = 0; m < bdtmt::N_MEMTILE; m++) {
            mtx[m] = shared_buffer<bdtm::feat_t>::create(
                {BDT_W, bdtm::N_FEATURES}, 1, bdtmt::mt_count(m));
            num_buffers(mtx[m]) = bdtmt::MT_BUFFERS;

            connect<>(xin[m].out[0], mtx[m].in[0]);
            write_access(mtx[m].in[0]) = tiling({
                .buffer_dimension = {BDT_W, bdtm::N_FEATURES},
                .tiling_dimension = {BDT_W, bdtm::N_FEATURES},
                .offset = {0, 0}});

            for (unsigned c = 0; c < bdtmt::mt_count(m); c++) {
                const unsigned i = bdtmt::mt_first(m) + c;
                connect<>(mtx[m].out[c], k[i].in[0]);
                read_access(mtx[m].out[c]) = tiling({
                    .buffer_dimension = {BDT_W, bdtm::N_FEATURES},
                    .tiling_dimension = {BDT_W, bdtmt::n_feat(i)},
                    .offset = {0, (int)bdtmt::feat_offset(i)}});
                dimensions(k[i].in[0]) = {BDT_W * bdtmt::n_feat(i)};
            }
        }
#else
        for (unsigned i = 0; i < bdtmt::N_TILES; i++) {
            const unsigned src = (bdtmt::SPLIT_TREE && !bdtmt::FEED_PLIO) ? 0 : i;
            connect<stream>(xin[src].out[0], k[i].in[0]);
        }
#endif

        for (unsigned i = 0; i < bdtmt::N_TILES; i++)
            connect<stream>(k[i].out[0], sout[i].in[0]);

    }
};
