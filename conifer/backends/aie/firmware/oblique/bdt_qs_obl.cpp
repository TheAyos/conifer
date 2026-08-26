#include "bdt_qs_obl.hpp"
#include "bdt_qs_oblique.hpp"

using namespace adf;
using namespace bdtm;
using namespace bdtv;

void bdt_qs_obl(input_stream<feat_t>* __restrict xin,
               output_stream<score_t>* __restrict sout) {
    const vscore acc =
        qs_score_group([&](unsigned) { return readincr_v<W>(xin); });
    writeincr(sout, acc);
}
