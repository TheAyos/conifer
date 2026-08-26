#pragma once
#include <type_traits>
#include <aie_api/aie.hpp>
#include "parameters.h"

#ifndef BDT_W
#define BDT_W 16
#endif

namespace bdtv {

constexpr unsigned W = BDT_W;

using feat_t = bdtm::feat_t;
using score_t = bdtm::score_t;
using bv_t = bdtm::bv_t;

using vfeat = aie::vector<feat_t, W>;
using vscore = aie::vector<score_t, W>;
using vidx = aie::vector<int16_t, W>;
using vmask = aie::mask<W>;

template <typename A>
using elem_of = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<A &>()[0])>>;

using acc_tag = std::conditional_t<std::is_floating_point_v<score_t>,
                                   accfloat, acc32>;
using vacc = aie::accum<acc_tag, W>;

__attribute__((always_inline))
inline vmask is_false_node(feat_t thr, const vfeat &x) {
    const vfeat t = aie::broadcast<feat_t, W>(thr);
    return bdtm::SPLIT_LE ? aie::gt(x, t) : aie::ge(x, t);
}

// Which way a naive traversal branches: left iff SPLIT_LE ? x <= thr : x < thr.
__attribute__((always_inline))
inline vmask goes_left(const vfeat &x, const vfeat &thr) {
    return bdtm::SPLIT_LE ? aie::le(x, thr) : aie::lt(x, thr);
}

}  // namespace bdtv
