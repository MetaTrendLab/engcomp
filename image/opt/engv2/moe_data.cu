#include <cuda_runtime.h>
#include <cstdint>
namespace {
constexpr int kThreads = 1024;
template <int E>
__global__ void __launch_bounds__(kThreads)
    reax_moe_data_kernel(int32_t const* __restrict__ topk_ids, int const topk_length, int const topk,
                         int32_t* __restrict__ expert_offsets, int32_t* __restrict__ problem_sizes1,
                         int32_t* __restrict__ problem_sizes2, int32_t* __restrict__ a_map, int32_t* __restrict__ c_map,
                         int32_t* __restrict__ blockscale_offsets, int const n, int const k, int const n1) {
  static_assert(E % 32 == 0, "experts in whole warps");
  constexpr int PER = E / 32;
  __shared__ int cnt[E];
  __shared__ int cur[E];
  for (int i = threadIdx.x; i < E; i += kThreads) cnt[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < topk_length; i += kThreads) {
    int const e = topk_ids[i];
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int const lane = threadIdx.x;
    int c[PER], r[PER];
    int s = 0, sr = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      c[j] = cnt[lane * PER + j];
      r[j] = (c[j] + 127) / 128 * 128;
      s += c[j];
      sr += r[j];
    }
    int is = s, isr = sr;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int const t = __shfl_up_sync(0xffffffffu, is, o);
      int const tr = __shfl_up_sync(0xffffffffu, isr, o);
      if (lane >= o) {
        is += t;
        isr += tr;
      }
    }
    int base = is - s, baser = isr - sr;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      int const e = lane * PER + j;
      expert_offsets[e] = base;
      blockscale_offsets[e] = baser;
      cur[e] = base;
      problem_sizes1[e * 3 + 0] = c[j];
      problem_sizes1[e * 3 + 1] = n1;
      problem_sizes1[e * 3 + 2] = k;
      problem_sizes2[e * 3 + 0] = c[j];
      problem_sizes2[e * 3 + 1] = k;
      problem_sizes2[e * 3 + 2] = n;
      base += c[j];
      baser += r[j];
    }
    if (lane == 31) {
      expert_offsets[E] = base;
      blockscale_offsets[E] = baser;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < topk_length; i += kThreads) {
    int const e = topk_ids[i];
    if (e >= 0 && e < E) {
      int const p = atomicAdd(&cur[e], 1);
      a_map[p] = i / topk;
      c_map[i] = p;
    }
  }
}
}
extern "C" int reax_moe_data_launch(int topk_length, int topk, int n_experts, void const* topk_ids, void* expert_offsets,
                                    void* problem_sizes1, void* problem_sizes2, void* a_map, void* c_map,
                                    void* blockscale_offsets, int n, int k, int gated, void* stream) {
  if (n_experts != 128) return -1;
  reax_moe_data_kernel<128><<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<int32_t const*>(topk_ids), topk_length, topk, static_cast<int32_t*>(expert_offsets),
      static_cast<int32_t*>(problem_sizes1), static_cast<int32_t*>(problem_sizes2), static_cast<int32_t*>(a_map),
      static_cast<int32_t*>(c_map), static_cast<int32_t*>(blockscale_offsets), n, k, gated ? 2 * n : n);
  return static_cast<int>(cudaGetLastError());
}
