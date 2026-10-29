#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
namespace {
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}
struct alignas(16) PackedBf16x8 {
  __nv_bfloat162 elts[4];
};
template <bool Contract>
__device__ __forceinline__ float gelu_tanh_f(float f) {
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  if constexpr (Contract) {
    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube);
    return 0.5f * f * (1.0f + ::tanhf(inner));
  } else {
    float x_cube = __fmul_rn(__fmul_rn(f, f), f);
    float inner = __fmul_rn(BETA, __fadd_rn(f, __fmul_rn(KAPPA, x_cube)));
    return __fmul_rn(__fmul_rn(0.5f, f), __fadd_rn(1.0f, ::tanhf(inner)));
  }
}
template <bool Contract>
__device__ __forceinline__ __nv_bfloat162 packed_gelu_tanh(const __nv_bfloat162& val) {
  float2 fval = __bfloat1622float2(val);
  fval.x = gelu_tanh_f<Contract>(fval.x);
  fval.y = gelu_tanh_f<Contract>(fval.y);
  return __float22bfloat162_rn(fval);
}
template <bool Contract>
__device__ __forceinline__ __nv_bfloat162 gelu_and_mul(const __nv_bfloat162& gate, const __nv_bfloat162& up,
                                                       const float beta) {
  float2 u = __bfloat1622float2(up);
  float2 activated = __bfloat1622float2(packed_gelu_tanh<Contract>(gate));
  activated.x *= u.x + beta;
  activated.y *= u.y + beta;
  return __float22bfloat162_rn(activated);
}
template <bool Contract>
__global__ void __launch_bounds__(512) reax_gact_kernel(
    int32_t numRows, int32_t numCols, __nv_bfloat16 const* in, float const* SFScale, uint32_t* out, uint32_t* SFout,
    uint32_t const* input_offset_by_experts, uint32_t const* output_scale_offset_by_experts, int n_experts,
    float beta) {
  int32_t const numKTiles = (numCols + 63) / 64;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / 8;
  int32_t const numValidRows = min(numRows, static_cast<int32_t>(__ldca(&input_offset_by_experts[n_experts])));
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;
    int rowIdx_in_expert = 0;
    int expert_idx = 0;
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) >> 1;
      uint32_t mid_offset = __ldca(&input_offset_by_experts[mid]);
      uint32_t next_offset = __ldca(&input_offset_by_experts[mid + 1]);
      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    PackedBf16x8 const* row = reinterpret_cast<PackedBf16x8 const*>(in) + static_cast<int64_t>(rowIdx) * 2 * colsPerRow;
    PackedBf16x8 g8 = row[colIdx];
    PackedBf16x8 u8 = row[colsPerRow + colIdx];
    PackedBf16x8 vec;
#pragma unroll
    for (int i = 0; i < 4; i++) vec.elts[i] = gelu_and_mul<Contract>(g8.elts[i], u8.elts[i], beta);
    int64_t outOffset = static_cast<int64_t>(rowIdx) * colsPerRow + colIdx;
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;
    uint8_t* sf_out = nullptr;
    if (threadIdx.x % 2 == 0) {
      int32_t kIdx = colIdx / 2;
      int32_t mIdx = rowIdx_in_expert;
      int32_t mTileIdx = mIdx >> 7;
      int32_t outerMIdx = mIdx & 31;
      int32_t innerMIdx = (mIdx >> 5) & 3;
      int32_t kTileIdx = kIdx >> 2;
      int32_t innerKIdx = kIdx & 3;
      int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9 | (outerMIdx << 4) |
                         (innerMIdx << 2) | innerKIdx;
      sf_out = reinterpret_cast<uint8_t*>(SFout_in_expert) + SFOffset;
    }
    auto localMax = __habs2(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < 4; i++) {
      localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
    float vecMax = float(__hmax(localMax.x, localMax.y));
    float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    uint8_t fp8SFVal;
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    SFValue = float(tmp);
    if (sf_out) *sf_out = fp8SFVal;
    float outputScale =
        SFValue != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
    float2 fp2Vals[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
      fp2Vals[i].x *= outputScale;
      fp2Vals[i].y *= outputScale;
    }
    out[outOffset] = fp32_vec8_to_e2m1(fp2Vals);
  }
}
struct Cfg {
  int block, cap;
};
constexpr Cfg kCfgs[] = {{256, 8}, {128, 8}, {128, 16}, {256, 4}, {256, 16}, {256, 0}, {512, 2}, {512, 4}, {512, 0}};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(kCfgs[0]);
}
extern "C" int reax_gact_configs() { return kNumCfgs; }
extern "C" int reax_gact_launch(int variant, int m_topk, int k, void const* in, void const* gscale, void* out, void* sf,
                                void const* in_off, void const* sf_off, int n_experts, int num_sms, float beta,
                                void* stream) {
  int const total = m_topk * (k / 8);
  if (total == 0) return 0;
  if (variant < 0 || (variant >> 1) >= kNumCfgs) return 1;
  Cfg const c = kCfgs[variant >> 1];
  int grid = (total + c.block - 1) / c.block;
  if (c.cap && grid > num_sms * c.cap) grid = num_sms * c.cap;
  auto kern = (variant & 1) == 0 ? reax_gact_kernel<true> : reax_gact_kernel<false>;
  kern<<<grid, c.block, 0, static_cast<cudaStream_t>(stream)>>>(
      m_topk, k, static_cast<__nv_bfloat16 const*>(in), static_cast<float const*>(gscale), static_cast<uint32_t*>(out),
      static_cast<uint32_t*>(sf), static_cast<uint32_t const*>(in_off), static_cast<uint32_t const*>(sf_off), n_experts,
      beta);
  return static_cast<int>(cudaGetLastError());
}
