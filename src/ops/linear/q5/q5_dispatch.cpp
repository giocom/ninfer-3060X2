#include "ops/linear/q5/q5_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// --- TP2 shard geometries ---------------------------------------------------------------------
//
// A tp2 shard is the same row-split tensor with one axis halved: a column-parallel shard halves N
// (attention/query_key, attention/gate_value, mlp/gate_up, gdn/*), a row-parallel shard halves K
// (attention/output, gdn/output, mlp/down). No new kernel exists -- every launcher named below
// reads its extents from the tensors, so a shard entry only records which launcher serves the
// halved extent and the halved grid follows automatically.
//
// Two deliberate rules, stated once here and followed by every groupwise family:
//
//   1. Shard entries are ADDITIVE and looked up AFTER the tp1 table (tp1 wins on any collision),
//      but no shard extent coincides with a registered tp1 extent, so the tp1 path is bit-for-bit
//      the one it was.
//   2. A shard uses the family's GENERIC runtime-dimension launchers. It deliberately does not
//      inherit the parent's compile-time EXACT instantiations (q5's gemv_r16_s2_x is N-exact and
//      its split2 SIMT is K-exact, so neither covers a halved extent), and it deliberately does
//      not get its own exact instantiation: that would be a second tuned kernel set to qualify.
//      This is a performance choice only -- the generic launchers are the same qualified compute
//      body -- and re-measuring the halved crossovers is a separate, benchmarked change.
//
// Returns nullptr when (n, k) is not a registered shard extent, so the caller falls through to the
// tp1 table and its error message.
Q5Launch select_q5_tp2_shard_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool column_shard = k == 5120 && (n == 512 ||   // 1024  / 2
                                            n == 3072 ||  // 6144  / 2
                                            n == 3584);   // 7168  / 2
    const bool row_shard    = n == 5120 && (k == 3072 ||  // 6144  / 2 (attention/gdn output)
                                            k == 8704);   // 17408 / 2 (mlp/down)
    if (!column_shard && !row_shard) { return nullptr; }
    if (t <= 4) { return launch_q5_simt_r8_c4; }
    if (t <= 24) { return launch_q5_simt_r8_c8; }
    return launch_q5_mma_r64_c128;
}

// The tp1 table, exactly as it was: returns nullptr rather than throwing so the caller
// can fall back to the tp2 shard table. It is consulted FIRST, so a geometry that is
// both registered here and listed as a shard extent keeps its tuned tp1 launcher.
Q5Launch select_q5_a16_registered(std::int32_t n, std::int32_t k, std::int32_t t) {
    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_q5_simt_r8_c4; }
            if (t <= 16) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            if (t <= 64) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 16) { return launch_q5_simt_r8_c4; }
            return launch_q5_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 1152:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 76) { return launch_q5_simt_r8_c4; }
            if (t <= 636) { return launch_q5_mma_r64_c64; }
            if (t <= 700) { return launch_q5_mma_r64_c128; }
            if (t == 704) { return launch_q5_mma_r64_c64; }
            if (t <= 828) { return launch_q5_mma_r64_c128; }
            if (t == 832) { return launch_q5_mma_r64_c64; }
            if (t <= 896) { return launch_q5_mma_r64_c128; }
            if (t <= 960) { return launch_q5_mma_r64_c64; }
            if (t <= 1024) { return launch_q5_mma_r64_c128; }
            if (t <= 1088) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 4304:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 120) { return launch_q5_simt_r8_c4; }
            if (t <= 1148) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    default:
        break;
    }

    return nullptr;
}

} // namespace

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q5 linear: unsupported shape or T"); }
    if (const Q5Launch tp1 = select_q5_a16_registered(n, k, t); tp1 != nullptr) {
        return tp1;
    }
    if (const Q5Launch shard = select_q5_tp2_shard_launch(n, k, t); shard != nullptr) {
        return shard;
    }
    throw std::invalid_argument("q5 linear: unsupported shape or T");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_q5_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q5 linear: unsupported policy");
}

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q5Launch launch = select_q5_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
