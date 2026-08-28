#include "bdt_qs_obl.hpp"
#include "bdt_qs_oblique.hpp"

using namespace adf;
using namespace bdtm;
using namespace bdtv;

// The templates live here, not in the header: aiecompiler's frontend
// introspects a kernel's argument types from its declaration, and a template
// instantiation defeats it.
//
// always_inline on the closure, not only on the body it is passed to: left
// standing as a real function on one tile of a multi-tile build, the row reader
// costs that tile call overhead its neighbours do not pay, and the slowest tile
// sets the reported number.
template <unsigned SHARD>
__attribute__((always_inline)) static inline vscore
score_shard(input_stream<feat_t> *xin) {
  return qs_score_group<bdtmt::t_begin(SHARD), bdtmt::t_count(SHARD),
                        bdtmt::adds_init(SHARD)>(
      [&](unsigned)
          __attribute__((always_inline)) { return readincr_v<W>(xin); });
}

#define BDT_DEF_PLIO(S)                                                        \
  void bdt_qs_tile_##S(input_stream<feat_t> *__restrict xin,                   \
                       output_stream<score_t> *__restrict sout) {              \
    writeincr(sout, score_shard<S>(xin));                                      \
  }

#define BDT_DEF_ROLE(S) BDT_DEF_PLIO(S)
#define BDT_DEF_ROLE0(S) BDT_DEF_PLIO(S)

// At N_TILES == 1 this is the single-tile kernel's arithmetic exactly; under
// sample-split every tile runs it, on its own samples.
void bdt_qs_obl(input_stream<feat_t> *__restrict xin,
                output_stream<score_t> *__restrict sout) {
  writeincr(sout, score_shard<0>(xin));
}

#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#define BDT_LADDER_DEF
#include "tile_roles.h"
#undef BDT_LADDER_DEF
#endif
