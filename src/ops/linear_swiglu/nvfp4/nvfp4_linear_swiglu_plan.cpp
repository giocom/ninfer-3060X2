#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
};

constexpr std::int32_t kPrimaryT = 1024;

// A pure function of (policy, token count), not of N/K, so it is inherited unchanged by the tp2
// shard -- the same "route selection depends only on the family, not the halved extent" rule
// every split family here follows: tuning is inherited from the tp1 parent, never re-measured.
Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_swiglu admits only A16 or A4");
    }
    if (policy == LinearPolicy::A16Only) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        throw std::invalid_argument("nvfp4 linear_swiglu A16 is registered only through T=16");
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    if (tokens <= 48) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Geometry, class Allocator>
Nvfp4LinearSwiGluWorkspace allocate_baseline_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens}, 256);
    const std::size_t linear_bytes =
        linear_workspace_capacity_bytes(QType::NVFP4, Geometry::kOutputRows, Geometry::kInputRows,
                                        LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Geometry, class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, Geometry::kInputRows);
}

template <class Geometry>
std::size_t baseline_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

template <class Geometry>
std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

template <class Geometry>
std::size_t capacity_bytes_impl(LinearPolicy policy, std::int32_t min_tokens,
                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (policy == LinearPolicy::A16Only || max_tokens <= 4) { return 0; }

    std::size_t maximum = 0;
    if (min_tokens <= 48 && max_tokens >= 5) {
        maximum = fused_workspace_bytes<Geometry>(std::min(max_tokens, 48));
    }
    if (min_tokens <= kPrimaryT && max_tokens >= kPrimaryT) {
        maximum = fused_workspace_bytes<Geometry>(kPrimaryT);
    }

    std::int32_t last_baseline = max_tokens;
    if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
        --last_baseline;
    }
    if (last_baseline >= std::max(min_tokens, 49)) {
        maximum = std::max(maximum, baseline_workspace_bytes<Geometry>(last_baseline));
    }
    return maximum;
}

// Bridges the Geometry-templated dispatch below to the two concretely-named launch symbols each
// .cu file exports (tp1 unchanged name, tp2 shard suffixed `_shard`). Selected at compile time, so
// dispatch_impl<Geometry> never pays a runtime branch for something the type already determines.
template <class Geometry>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
    } else {
        nvfp4_linear_swiglu_decode_launch_shard(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
    } else {
        nvfp4_linear_swiglu_small_t_launch_shard(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_w4a4(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
    } else {
        nvfp4_linear_swiglu_w4a4_launch_shard(x, weight, out, workspace, stream);
    }
}

template <class Geometry>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        launch_nvfp4_linear_swiglu_w4a4_tma(activation_codes, activation_scales, weight_codes,
                                            weight_scales, output, tokens, alpha, stream);
    } else {
        launch_nvfp4_linear_swiglu_w4a4_tma_shard(activation_codes, activation_scales, weight_codes,
                                                  weight_scales, output, tokens, alpha, stream);
    }
}

// The one route body, parameterized on Geometry. `workspace` is nullable (see
// nvfp4_linear_swiglu_plan.h): the A16 routes never touch it; a W4A4-family route with a null
// workspace throws, matching ops::linear's own nvfp4_dispatch guard.
template <class Geometry>
void dispatch_impl(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        launch_decode<Geometry>(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        launch_small_t<Geometry>(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu W4A4 requires caller workspace");
        }
        launch_w4a4<Geometry>(x, weight, out, *workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu TMA W4A4 requires caller workspace");
        }
        auto scope                       = workspace->scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace<Geometry>(*workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_tma<Geometry>(scratch.codes, scratch.scales,
                             static_cast<const std::uint8_t*>(weight.qdata),
                             static_cast<const std::uint8_t*>(weight.scales),
                             static_cast<__nv_bfloat16*>(out.data), x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    if (workspace == nullptr) {
        throw std::invalid_argument(
            "nvfp4 linear_swiglu materialized W4A4 requires caller workspace");
    }
    auto scope                         = workspace->scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace<Geometry>(*workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    silu_mul(scratch.projected.slice(0, 0, kIntermediate),
             scratch.projected.slice(0, kIntermediate, kIntermediate), out, stream);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    return capacity_bytes_impl<Nvfp4MlpGateUpGeometry>(policy, min_tokens, max_tokens);
}

std::size_t nvfp4_linear_swiglu_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens) {
    return capacity_bytes_impl<Nvfp4MlpGateUpTp2ColumnGeometry>(policy, min_tokens, max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    dispatch_impl<Nvfp4MlpGateUpGeometry>(x, weight, out, policy, &workspace, stream);
}

void nvfp4_linear_swiglu_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                        LinearPolicy policy, WorkspaceArena* workspace,
                                        cudaStream_t stream) {
    dispatch_impl<Nvfp4MlpGateUpTp2ColumnGeometry>(x, weight, out, policy, workspace, stream);
}

} // namespace ninfer::ops::detail
