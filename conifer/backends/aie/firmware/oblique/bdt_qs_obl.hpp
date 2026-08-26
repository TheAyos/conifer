#pragma once
#include <adf.h>
#include "params.h"

void bdt_qs_obl(input_stream<bdtm::feat_t>* xin,
               output_stream<bdtm::score_t>* sout);
