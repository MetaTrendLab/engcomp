#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace {
__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int t) {
  const bool hi = t & 2, lo = t & 1;
  uint32_t s0 = hi ? v[0] : v[2], s1 = hi ? v[1] : v[3];
  uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
  if (hi) { v[0] = r0; v[1] = r1; } else { v[2] = r0; v[3] = r1; }
  s0 = lo ? v[0] : v[1];
  s1 = lo ? v[2] : v[3];
  r0 = __shfl_xor_sync(0xffffffffu, s0, 1);
  r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
  if (lo) { v[0] = r0; v[2] = r1; } else { v[1] = r0; v[3] = r1; }
}
__device__ __forceinline__ uint4 ld_stream(const uint4* p, uint64_t pol) {
  uint4 x;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(p), "l"(pol));
  return x;
}
__device__ __forceinline__ int sf_off(int r, int kk, int KS) {
  return ((r >> 7) * KS + kk) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4;
}
template <int K, int W, int R, int J, int MINB, int SP, int CPS>
__global__ void __launch_bounds__(W * 32, MINB) reax_moe_dn(__nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ a,
                                                           const uint8_t* __restrict__ b,
                                                           const uint8_t* __restrict__ a_sf,
                                                           const uint8_t* __restrict__ b_sf,
                                                           const float* __restrict__ alphas,
                                                           const int32_t* __restrict__ problem_sizes,
                                                           const int32_t* __restrict__ expert_offsets,
                                                           const int32_t* __restrict__ sf_offsets, int N) {
  constexpr int KB = K / 2;
  constexpr int KS = K / 64;
  constexpr int LD = KB + 16;
  constexpr int kCols = W * 8 * J;
  __shared__ __align__(16) uint8_t sa[R * LD];
  const int e = blockIdx.z;
  const int m_e = problem_sizes[e * 3];
  const int y = blockIdx.y, m_lo = (CPS ? y * CPS : y) * R;
  const int m_hi = CPS && y + 1 < SP ? min(m_e, (y + 1) * CPS * R) : m_e, STEP = CPS ? R : SP * R;
  if (m_e <= m_lo) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * kCols + warp * 8 * J;
  const uint8_t* bw = b + (size_t)e * N * KB;
  const uint8_t* bs = b_sf + (size_t)e * N * (K / 16);
  constexpr int NQ = 2 * KS, NC = (NQ + 3) / 4;
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  uint32_t wb[J][KS][2], ws[J][KS];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int n = n0 + j * 8 + g;
    const uint4* row = reinterpret_cast<const uint4*>(bw + (size_t)n * KB);
    uint32_t v[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int q = 4 * c + t;
      const uint4 x = q < NQ ? ld_stream(row + q, pol) : make_uint4(0u, 0u, 0u, 0u);
      v[c][0] = x.x, v[c][1] = x.y, v[c][2] = x.z, v[c][3] = x.w;
    }
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) ws[j][kk] = __ldg(reinterpret_cast<const uint32_t*>(bs + sf_off(n, kk, KS)));
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      quad_transpose(v[c], t);
#pragma unroll
      for (int s = 0; s < 4; ++s)
        if (4 * c + s < NQ) wb[j][(4 * c + s) >> 1][s & 1] = v[c][s];
    }
  }
  const int eo = expert_offsets[e];
  const uint8_t* ae = a + (size_t)eo * KB;
  const uint8_t* aes = a_sf + (size_t)sf_offsets[e] * (K / 16);
  const float alpha = alphas[e];
  for (int mc = m_lo; mc < m_hi; mc += STEP) {
    const int rows = min(R, m_e - mc);
    if (mc != m_lo) __syncthreads();
    for (int c = threadIdx.x; c < rows * (KB / 16); c += W * 32) {
      const int r = c / (KB / 16), q = c % (KB / 16);
      const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(sa + r * LD + q * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(ae + (size_t)(mc + r) * KB + q * 16));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int tiles = (rows + 15) >> 4;
    for (int i = 0; i < tiles; ++i) {
      float acc[J][4];
#pragma unroll
      for (int j = 0; j < J; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      const uint8_t* r0 = sa + (i * 16 + g) * LD + 4 * t;
      const uint8_t* r1 = r0 + 8 * LD;
      const int sr = mc + i * 16 + g + 8 * (lane & 1);
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        uint32_t af[4];
        af[0] = *reinterpret_cast<const uint32_t*>(r0 + kk * 32);
        af[1] = *reinterpret_cast<const uint32_t*>(r1 + kk * 32);
        af[2] = *reinterpret_cast<const uint32_t*>(r0 + kk * 32 + 16);
        af[3] = *reinterpret_cast<const uint32_t*>(r1 + kk * 32 + 16);
        const uint32_t sfa = __ldg(reinterpret_cast<const uint32_t*>(aes + sf_off(sr, kk, KS)));
#pragma unroll
        for (int j = 0; j < J; ++j) mma_nvf4(acc[j], af, wb[j][kk][0], wb[j][kk][1], sfa, ws[j][kk]);
      }
      const int ra = mc + i * 16 + g, rb = ra + 8;
#pragma unroll
      for (int j = 0; j < J; ++j) {
        const int col = n0 + j * 8 + 2 * t;
        if (ra < m_e)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(eo + ra) * N + col) =
              __floats2bfloat162_rn(alpha * acc[j][0], alpha * acc[j][1]);
        if (rb < m_e)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(eo + rb) * N + col) =
              __floats2bfloat162_rn(alpha * acc[j][2], alpha * acc[j][3]);
      }
    }
  }
}
template <int K, int W, int R, int J, int SP>
__global__ void __launch_bounds__(W * 32) reax_moe_dn_b(__nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ a,
                                                       const uint8_t* __restrict__ b, const uint8_t* __restrict__ a_sf,
                                                       const uint8_t* __restrict__ b_sf,
                                                       const float* __restrict__ alphas,
                                                       const int32_t* __restrict__ problem_sizes,
                                                       const int32_t* __restrict__ expert_offsets,
                                                       const int32_t* __restrict__ sf_offsets, int N) {
  constexpr int KB = K / 2, KS = K / 64, LD = KB + 16, kCols = W * 8 * J;
  constexpr int BB = kCols * KB, SM = BB > R * LD ? BB : R * LD;
  __shared__ __align__(128) uint8_t sm[SM];
  __shared__ __align__(8) uint64_t bar;
  const int e = blockIdx.z;
  const int m_e = problem_sizes[e * 3];
  if (m_e <= (int)blockIdx.y * R) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int c0 = blockIdx.x * kCols, n0 = c0 + warp * 8 * J;
  const uint32_t s_b = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
  const uint32_t s_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\nfence.mbarrier_init.release.cluster;\n" ::"r"(s_bar));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(s_bar), "r"(BB) : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
            s_b),
        "l"(b + ((size_t)e * N + c0) * KB), "r"(BB), "r"(s_bar), "l"(pol)
        : "memory");
  }
  const uint8_t* bs = b_sf + (size_t)e * N * (K / 16);
  uint32_t wb[J][KS][2], ws[J][KS];
#pragma unroll
  for (int j = 0; j < J; ++j)
#pragma unroll
    for (int kk = 0; kk < KS; ++kk)
      ws[j][kk] = __ldg(reinterpret_cast<const uint32_t*>(bs + sf_off(n0 + j * 8 + g, kk, KS)));
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra WAIT_%=;\n}\n" ::"r"(s_bar)
      : "memory");
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const uint8_t* row = sm + (warp * 8 * J + j * 8 + g) * KB + 4 * t;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      wb[j][kk][0] = *reinterpret_cast<const uint32_t*>(row + kk * 32);
      wb[j][kk][1] = *reinterpret_cast<const uint32_t*>(row + kk * 32 + 16);
    }
  }
  const int eo = expert_offsets[e];
  const uint8_t* ae = a + (size_t)eo * KB;
  const uint8_t* aes = a_sf + (size_t)sf_offsets[e] * (K / 16);
  const float alpha = alphas[e];
  for (int mc = blockIdx.y * R; mc < m_e; mc += SP * R) {
    const int rows = min(R, m_e - mc);
    __syncthreads();
    for (int c = threadIdx.x; c < rows * (KB / 16); c += W * 32) {
      const int r = c / (KB / 16), q = c % (KB / 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s_b + r * LD + q * 16),
                   "l"(ae + (size_t)(mc + r) * KB + q * 16));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int tiles = (rows + 15) >> 4;
    for (int i = 0; i < tiles; ++i) {
      float acc[J][4];
#pragma unroll
      for (int j = 0; j < J; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      const uint8_t* r0 = sm + (i * 16 + g) * LD + 4 * t;
      const uint8_t* r1 = r0 + 8 * LD;
      const int sr = mc + i * 16 + g + 8 * (lane & 1);
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        uint32_t af[4];
        af[0] = *reinterpret_cast<const uint32_t*>(r0 + kk * 32);
        af[1] = *reinterpret_cast<const uint32_t*>(r1 + kk * 32);
        af[2] = *reinterpret_cast<const uint32_t*>(r0 + kk * 32 + 16);
        af[3] = *reinterpret_cast<const uint32_t*>(r1 + kk * 32 + 16);
        const uint32_t sfa = __ldg(reinterpret_cast<const uint32_t*>(aes + sf_off(sr, kk, KS)));
#pragma unroll
        for (int j = 0; j < J; ++j) mma_nvf4(acc[j], af, wb[j][kk][0], wb[j][kk][1], sfa, ws[j][kk]);
      }
      const int ra = mc + i * 16 + g, rb = ra + 8;
#pragma unroll
      for (int j = 0; j < J; ++j) {
        const int col = n0 + j * 8 + 2 * t;
        if (ra < m_e)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(eo + ra) * N + col) =
              __floats2bfloat162_rn(alpha * acc[j][0], alpha * acc[j][1]);
        if (rb < m_e)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(eo + rb) * N + col) =
              __floats2bfloat162_rn(alpha * acc[j][2], alpha * acc[j][3]);
      }
    }
  }
}
using Launch = int (*)(void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*,
                       const void*, int, int, cudaStream_t);
template <int W, int R, int J, int MINB, int SP = 1, int CPS = 0>
int launch(void* out, const void* a, const void* b, const void* a_sf, const void* b_sf, const void* alphas,
           const void* problem_sizes, const void* expert_offsets, const void* sf_offsets, int num_experts, int n,
           cudaStream_t stream) {
  constexpr int kCols = W * 8 * J;
  if (n % kCols) return 1;
  dim3 grid(n / kCols, SP, num_experts);
  reax_moe_dn<704, W, R, J, MINB, SP, CPS><<<grid, W * 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b),
      static_cast<const uint8_t*>(a_sf), static_cast<const uint8_t*>(b_sf), static_cast<const float*>(alphas),
      static_cast<const int32_t*>(problem_sizes), static_cast<const int32_t*>(expert_offsets),
      static_cast<const int32_t*>(sf_offsets), n);
  return static_cast<int>(cudaGetLastError());
}
template <int W, int R, int J, int SP = 1>
int launch_b(void* out, const void* a, const void* b, const void* a_sf, const void* b_sf, const void* alphas,
             const void* problem_sizes, const void* expert_offsets, const void* sf_offsets, int num_experts, int n,
             cudaStream_t stream) {
  constexpr int kCols = W * 8 * J;
  if (n % kCols) return 1;
  dim3 grid(n / kCols, SP, num_experts);
  reax_moe_dn_b<704, W, R, J, SP><<<grid, W * 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b),
      static_cast<const uint8_t*>(a_sf), static_cast<const uint8_t*>(b_sf), static_cast<const float*>(alphas),
      static_cast<const int32_t*>(problem_sizes), static_cast<const int32_t*>(expert_offsets),
      static_cast<const int32_t*>(sf_offsets), n);
  return static_cast<int>(cudaGetLastError());
}
struct Variant {
  const char* name;
  Launch fn;
};
#define REAX_V(W, R, J, MB) {"w" #W "_r" #R "_j" #J "_b" #MB, launch<W, R, J, MB>}
#define REAX_B(W, R, J) {"t_w" #W "_r" #R "_j" #J, launch_b<W, R, J>}
#define REAX_C(W, R, J, SP, CPS) {"w" #W "_r" #R "_j" #J "_s" #SP "c" #CPS, launch<W, R, J, 1, SP, CPS>}
constexpr Variant kVariants[] = {
    REAX_V(4, 64, 2, 1),
    REAX_V(2, 32, 2, 1), REAX_V(2, 64, 2, 1), REAX_V(2, 128, 2, 1), REAX_V(2, 32, 4, 1), REAX_V(2, 64, 4, 1),
    REAX_V(2, 128, 4, 1), REAX_V(4, 32, 2, 1), REAX_V(4, 128, 2, 1), REAX_V(4, 32, 4, 1), REAX_V(4, 64, 4, 1),
    REAX_V(4, 128, 4, 1), REAX_V(8, 32, 2, 1), REAX_V(8, 64, 2, 1), REAX_V(8, 128, 2, 1), REAX_V(8, 32, 4, 1),
    REAX_V(8, 64, 4, 1), REAX_V(8, 128, 4, 1),
    REAX_V(4, 64, 4, 4), REAX_V(8, 64, 4, 2),
    REAX_B(4, 64, 2), REAX_B(8, 64, 2), REAX_B(4, 64, 4),
    REAX_C(8, 64, 4, 4, 2), REAX_C(8, 64, 4, 4, 3), REAX_C(8, 64, 4, 3, 4), REAX_C(8, 64, 4, 8, 2), REAX_C(4, 32, 4, 8, 4),
    REAX_C(4, 32, 4, 6, 6), REAX_C(8, 32, 2, 8, 4), REAX_C(8, 128, 4, 4, 2),
};
#undef REAX_V
#undef REAX_B
#undef REAX_C
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);
}
extern "C" int reax_moedn_variants() { return kNumVariants; }
extern "C" const char* reax_moedn_variant_name(int v) { return v >= 0 && v < kNumVariants ? kVariants[v].name : ""; }
extern "C" int reax_moedn_mm_v(int variant, void* out, const void* a, const void* b, const void* a_sf, const void* b_sf,
                               const void* alphas, const void* problem_sizes, const void* expert_offsets,
                               const void* sf_offsets, int num_experts, int n, int k, void* stream) {
  if (k != 704 || variant < 0 || variant >= kNumVariants) return 1;
  return kVariants[variant].fn(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, num_experts, n,
                               static_cast<cudaStream_t>(stream));
}
extern "C" int reax_moedn_mm(void* out, const void* a, const void* b, const void* a_sf, const void* b_sf,
                             const void* alphas, const void* problem_sizes, const void* expert_offsets,
                             const void* sf_offsets, int num_experts, int n, int k, void* stream) {
  return reax_moedn_mm_v(0, out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, num_experts, n, k,
                         stream);
}
