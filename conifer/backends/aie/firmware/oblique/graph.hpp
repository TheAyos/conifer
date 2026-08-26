#pragma once
#include "params.h"
#include "bdt_qs_obl.hpp"

using namespace adf;

#if defined(__has_include)
#  if __has_include("xin_file.h")
#    include "xin_file.h"
#  endif
#endif
#ifndef XIN_FILE
#define XIN_FILE "../../gen/out/fp32/data/X_fm_w16.dat"
#endif

class theGraph : public graph {
private:
    kernel k1;

public:
    input_plio  xin;  // one W-sample group, feature-major, per iteration
    output_plio sout;  // W scores per iteration

    theGraph() {
        k1 = kernel::create(bdt_qs_obl);
        source(k1) = "src/bdt_qs_obl.cpp";
        runtime<ratio>(k1) = 1.0;

        constexpr unsigned KIB = 1024;
        constexpr unsigned TABLES =
              sizeof(bdtm::QT_THR)  + sizeof(bdtm::QT_BV)
            + sizeof(bdtm::QT_FEAT) + sizeof(bdtm::LEAVES)
            + sizeof(bdtm::INIT_V)
            + sizeof(bdtm::QT_BTERM) + sizeof(bdtm::QT_BSIGN)
            + sizeof(bdtm::BASIS_I) + sizeof(bdtm::BASIS_J)
            + sizeof(bdtm::BASIS_WI) + sizeof(bdtm::BASIS_WJ);
        constexpr unsigned XBYTES =
              (bdtm::N_FEATURES + bdtm::BASIS_N) * BDT_W * sizeof(bdtm::feat_t);
        constexpr unsigned HEAP  = ((TABLES + 2 * KIB - 1) / KIB) * KIB;
        constexpr unsigned STACK = ((XBYTES + 4 * KIB - 1) / KIB) * KIB;
        heap_size(k1)  = HEAP;
        stack_size(k1) = STACK;

        xin  = input_plio ::create("xin",    plio_64_bits, XIN_FILE,     BDT_PLIO_RATE);
        sout = output_plio::create("scores", plio_32_bits, "scores.dat", BDT_PLIO_RATE);

        connect<stream> net_in (xin.out[0], k1.in[0]);
        connect<stream> net_out(k1.out[0],  sout.in[0]);

    }
};
