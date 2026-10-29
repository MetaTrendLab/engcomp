#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
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
__global__ void __launch_bounds__(512) reax_gquant_kernel(
    int32_t numRows, int32_t numCols, __nv_bfloat16 const* in, int32_t const* rowmap, float const* SFScale,
    uint32_t* out, uint32_t* SFout, uint32_t const* input_offset_by_experts,
    uint32_t const* output_scale_offset_by_experts, int n_experts) {
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
    int64_t inOffset = static_cast<int64_t>(__ldca(&rowmap[rowIdx])) * colsPerRow + colIdx;
    PackedBf16x8 vec = reinterpret_cast<PackedBf16x8 const*>(in)[inOffset];
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
__global__ void __launch_bounds__(512) reax_gquant1_kernel(
    int32_t numTokens, int32_t topk, int32_t numCols, __nv_bfloat16 const* in, int32_t const* cmap,
    int32_t const* topk_ids, float const* SFScale, uint32_t* out, uint32_t* SFout,
    uint32_t const* input_offset_by_experts, uint32_t const* output_scale_offset_by_experts, int n_experts) {
  int32_t const numKTiles = (numCols + 63) / 64;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / 8;
  uint32_t const numValidRows =
      min(static_cast<uint32_t>(numTokens * topk), __ldca(&input_offset_by_experts[n_experts]));
  for (int globalIdx = tid; globalIdx < numTokens * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    int tok = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;
    PackedBf16x8 vec = reinterpret_cast<PackedBf16x8 const*>(in)[static_cast<int64_t>(tok) * colsPerRow + colIdx];
    auto localMax = __habs2(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < 4; i++) {
      localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
    float vecMax = float(__hmax(localMax.x, localMax.y));
    bool have = false;
    float lastScale = 0.0f;
    uint8_t fp8SFVal = 0;
    uint32_t packed = 0;
    for (int j = 0; j < topk; j++) {
      int32_t const pair = tok * topk + j;
      int32_t const expert_idx = __ldca(&topk_ids[pair]);
      if (expert_idx < 0 || expert_idx >= n_experts) continue;
      uint32_t const rowIdx = static_cast<uint32_t>(__ldca(&cmap[pair]));
      if (rowIdx >= numValidRows) continue;
      float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];
      if (!have || SFScaleVal != lastScale) {
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
        SFValue = float(tmp);
        float outputScale =
            SFValue != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
        float2 fp2Vals[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
          fp2Vals[i].x *= outputScale;
          fp2Vals[i].y *= outputScale;
        }
        packed = fp32_vec8_to_e2m1(fp2Vals);
        lastScale = SFScaleVal;
        have = true;
      }
      out[static_cast<int64_t>(rowIdx) * colsPerRow + colIdx] = packed;
      if (threadIdx.x % 2 == 0) {
        int32_t mIdx = static_cast<int32_t>(rowIdx - __ldca(&input_offset_by_experts[expert_idx]));
        int32_t kIdx = colIdx / 2;
        int32_t mTileIdx = mIdx >> 7;
        int32_t outerMIdx = mIdx & 31;
        int32_t innerMIdx = (mIdx >> 5) & 3;
        int32_t kTileIdx = kIdx >> 2;
        int32_t innerKIdx = kIdx & 3;
        int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9 | (outerMIdx << 4) |
                           (innerMIdx << 2) | innerKIdx;
        uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;
        reinterpret_cast<uint8_t*>(SFout_in_expert)[SFOffset] = fp8SFVal;
      }
    }
  }
}
__global__ void __launch_bounds__(512) reax_tquant_kernel(int32_t numTokens, int32_t numCols,
                                                          __nv_bfloat16 const* in, float const* SFScale, uint32_t* out,
                                                          uint8_t* SFout) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / 8;
  float const SFScaleVal = SFScale[0];
  for (int globalIdx = tid; globalIdx < numTokens * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    int tok = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;
    PackedBf16x8 vec = reinterpret_cast<PackedBf16x8 const*>(in)[static_cast<int64_t>(tok) * colsPerRow + colIdx];
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
    float outputScale =
        SFValue != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
    float2 fp2Vals[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
      fp2Vals[i].x *= outputScale;
      fp2Vals[i].y *= outputScale;
    }
    out[static_cast<int64_t>(tok) * colsPerRow + colIdx] = fp32_vec8_to_e2m1(fp2Vals);
    if (threadIdx.x % 2 == 0) SFout[static_cast<int64_t>(tok) * (numCols / 16) + colIdx / 2] = fp8SFVal;
  }
}
}
extern "C" int reax_gquant_launch(int m_topk, int k, void const* in, void const* rowmap, void const* gscale, void* out,
                                  void* sf, void const* in_off, void const* sf_off, int n_experts, int num_sms,
                                  void* stream) {
  int const total = m_topk * (k / 8);
  if (total == 0) return 0;
  int const block = 256;
  int grid = (total + block - 1) / block;
  if (grid > num_sms * 8) grid = num_sms * 8;
  reax_gquant_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      m_topk, k, static_cast<__nv_bfloat16 const*>(in), static_cast<int32_t const*>(rowmap),
      static_cast<float const*>(gscale), static_cast<uint32_t*>(out), static_cast<uint32_t*>(sf),
      static_cast<uint32_t const*>(in_off), static_cast<uint32_t const*>(sf_off), n_experts);
  return static_cast<int>(cudaGetLastError());
}
extern "C" int reax_gquant1_launch(int m, int topk, int k, void const* in, void const* cmap, void const* topk_ids,
                                   void const* gscale, void* out, void* sf, void const* in_off, void const* sf_off,
                                   int n_experts, int num_sms, void* stream) {
  int const total = m * (k / 8);
  if (total == 0) return 0;
  int const block = 256;
  int grid = (total + block - 1) / block;
  if (grid > num_sms * 8) grid = num_sms * 8;
  reax_gquant1_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      m, topk, k, static_cast<__nv_bfloat16 const*>(in), static_cast<int32_t const*>(cmap),
      static_cast<int32_t const*>(topk_ids), static_cast<float const*>(gscale), static_cast<uint32_t*>(out),
      static_cast<uint32_t*>(sf), static_cast<uint32_t const*>(in_off), static_cast<uint32_t const*>(sf_off),
      n_experts);
  return static_cast<int>(cudaGetLastError());
}
extern "C" int reax_tquant_launch(int m, int k, void const* in, void const* gscale, void* out, void* sf, int num_sms,
                                  void* stream) {
  int const total = m * (k / 8);
  if (total == 0) return 0;
  int const block = 256;
  int grid = (total + block - 1) / block;
  if (grid > num_sms * 8) grid = num_sms * 8;
  reax_tquant_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      m, k, static_cast<__nv_bfloat16 const*>(in), static_cast<float const*>(gscale), static_cast<uint32_t*>(out),
      static_cast<uint8_t*>(sf));
  return static_cast<int>(cudaGetLastError());
}
