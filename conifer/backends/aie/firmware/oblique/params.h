#pragma once
#include "parameters.h"

#ifndef BDT_W
#define BDT_W 16
#endif

// One group of W samples per kernel invocation, so the graph runs
// N_SAMPLES / W times and the profile's cycles/call covers W samples.
static_assert(bdtm::N_SAMPLES % BDT_W == 0,
              "N_SAMPLES must be a whole number of W-sample groups; "
              "regenerate with --n-samples a multiple of W");
constexpr unsigned iter_count = bdtm::N_SAMPLES / BDT_W;
