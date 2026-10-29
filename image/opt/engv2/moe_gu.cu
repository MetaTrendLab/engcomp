#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
namespace {
constexpr int K = 2816, KB = K / 2, KS = K / 64;
constexpr int NG = 704;
constexpr int SFK = (NG / 16 + 3) / 4;
constexpr int kMaxExperts = 256;
__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
}
__device__ __forceinline__ void cp4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ int sf_off(int r, int kk, int ks) {
  return ((r >> 7) * ks + kk) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4;
}
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
__device__ __forceinline__ uint32_t e2m1_pair(float x, float y) {
  float2 v[4] = {make_float2(x, y), make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
  return fp32_vec8_to_e2m1(v) & 0xffu;
}
struct Seg {
  int it, e, nt, mc, m;
};
template <int R, int WM, int ST, int BKB, bool Contract, bool Gather>
__global__ void __launch_bounds__(128 * WM, 1) reax_moe_gu(
    uint32_t* __restrict__ out, uint8_t* __restrict__ sf_out, const uint8_t* __restrict__ a, const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ a_sf, const uint8_t* __restrict__ b_sf, const float* __restrict__ alphas,
    const float* __restrict__ gscale, const int32_t* __restrict__ problem_sizes,
    const int32_t* __restrict__ expert_offsets, const int32_t* __restrict__ sf_offsets, const int32_t* __restrict__ amap,
    int num_experts, float beta) {
  constexpr int THREADS = 128 * WM, MTW = R / 16 / WM, NT = NG / 64;
  constexpr int KC = KB / BKB, KKC = BKB / 32, LD = BKB + 16;
  static_assert(KB % BKB == 0, "K chunks");
  __shared__ int pre[kMaxExperts + 1];
  constexpr int SA = R * LD, SB = 128 * LD;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t sA = smem_u32(smem), sB = sA + ST * SA, sFA = sB + ST * SB, sFB = sFA + ST * KKC * R * 4;
  const uint32_t* fA = reinterpret_cast<const uint32_t*>(smem + ST * (SA + SB));
  const uint32_t* fB = fA + ST * KKC * R;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp & 3, wm = warp >> 2;
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  for (int x = tid; x < num_experts; x += THREADS) pre[x + 1] = (problem_sizes[x * 3] + R - 1) / R;
  __syncthreads();
  if (tid == 0) {
    pre[0] = 0;
    for (int x = 1; x <= num_experts; ++x) pre[x] += pre[x - 1];
  }
  __syncthreads();
  const int items = NT * pre[num_experts];
  auto next_seg = [&](Seg& s) -> bool {
    const int it = s.it < 0 ? (int)blockIdx.x : s.it + (int)gridDim.x;
    if (it >= items) return false;
    const int ci = it / NT;
    int lo = 0, hi = num_experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (pre[mid] <= ci) lo = mid;
      else hi = mid;
    }
    s = Seg{it, lo, it % NT, (ci - pre[lo]) * R, problem_sizes[lo * 3]};
    return true;
  };
  auto load = [&](int st, int kt, const Seg& s) {
    const int n0 = s.nt * 64;
    const int eo = expert_offsets[s.e];
    const uint8_t* ae = a + (size_t)eo * KB;
    const uint8_t* aes = a_sf + (size_t)sf_offsets[s.e] * (K / 16);
    const uint8_t* bw = b + (size_t)s.e * (2 * NG) * KB;
    const uint8_t* bs = b_sf + (size_t)s.e * (2 * NG) * (K / 16);
    for (int c = tid; c < R * (BKB / 16); c += THREADS) {
      const int r = c / (BKB / 16), q = c % (BKB / 16);
      const bool ok = s.mc + r < s.m;
      if constexpr (Gather) {
        const uint8_t* row = a + (size_t)__ldg(&amap[eo + (ok ? s.mc + r : 0)]) * KB;
        cp16(sA + st * SA + r * LD + q * 16, row + kt * BKB + q * 16, ok ? 16 : 0);
      } else {
        cp16(sA + st * SA + r * LD + q * 16, ae + (size_t)(ok ? s.mc + r : 0) * KB + kt * BKB + q * 16, ok ? 16 : 0);
      }
    }
    for (int c = tid; c < 128 * (BKB / 16); c += THREADS) {
      const int r = c / (BKB / 16), q = c % (BKB / 16);
      const int n = r < 64 ? n0 + r : NG + n0 + r - 64;
      cp16_ef(sB + st * SB + r * LD + q * 16, bw + (size_t)n * KB + kt * BKB + q * 16, pol);
    }
    for (int c = tid; c < KKC * R; c += THREADS) {
      const int kk = c / R, r = c % R;
      if constexpr (Gather) {
        const int tok = __ldg(&amap[eo + (s.mc + r < s.m ? s.mc + r : 0)]);
        cp4(sFA + (st * KKC * R + c) * 4, a_sf + (size_t)tok * (K / 16) + (kt * KKC + kk) * 4);
      } else {
        cp4(sFA + (st * KKC * R + c) * 4, aes + sf_off(s.mc + r, kt * KKC + kk, KS));
      }
    }
    for (int c = tid; c < KKC * 128; c += THREADS) {
      const int kk = c / 128, r = c % 128;
      const int n = r < 64 ? n0 + r : NG + n0 + r - 64;
      cp4(sFB + (st * KKC * 128 + c) * 4, bs + sf_off(n, kt * KKC + kk, KS));
    }
  };
  Seg cur{-1, 0, 0, 0, 0};
  if (!next_seg(cur)) return;
  Seg ld = cur;
  int ldk = 0, ql = 0;
  bool ld_ok = true;
  auto issue = [&]() {
    if (ld_ok) {
      load(ql % ST, ldk, ld);
      if (++ldk == KC) {
        ldk = 0;
        ld_ok = next_seg(ld);
      }
    }
    cp_commit();
    ++ql;
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) issue();
  int q = 0;
  while (true) {
    const int ntile = (min(R, cur.m - cur.mc) + 15) >> 4;
    float acc[MTW][4][4];
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    for (int kt = 0; kt < KC; ++kt, ++q) {
      cp_wait<ST - 2>();
      __syncthreads();
      issue();
      const int s = q % ST;
      const uint32_t pa = sA + s * SA, pb = sB + s * SB;
      const int jm = lane >> 3, rr = lane & 7;
      const uint32_t aoff = (rr + (jm & 1) * 8) * LD + (jm >> 1) * 16;
      const uint32_t boff = ((jm >> 1) * 8 + rr) * LD + (jm & 1) * 16;
#pragma unroll
      for (int kk = 0; kk < KKC; ++kk) {
        uint32_t bf[4][2], sfb[4];
        ldsm4(pb + (wn * 16) * LD + boff + kk * 32, bf[0][0], bf[0][1], bf[1][0], bf[1][1]);
        ldsm4(pb + (64 + wn * 16) * LD + boff + kk * 32, bf[2][0], bf[2][1], bf[3][0], bf[3][1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) sfb[j] = fB[(s * KKC + kk) * 128 + (j < 2 ? 0 : 64) + wn * 16 + (j & 1) * 8 + g];
#pragma unroll
        for (int ii = 0; ii < MTW; ++ii) {
          const int i = wm + ii * WM;
          if (i < ntile) {
            uint32_t af[4];
            ldsm4(pa + (i * 16) * LD + aoff + kk * 32, af[0], af[1], af[2], af[3]);
            const uint32_t sfa = fA[(s * KKC + kk) * R + i * 16 + g + 8 * (lane & 1)];
#pragma unroll
            for (int j = 0; j < 4; ++j) mma_nvf4(acc[ii][j], af, bf[j][0], bf[j][1], sfa, sfb[j]);
          }
        }
      }
    }
    {
      const float alpha = alphas[cur.e], sfscale = gscale[cur.e];
      const int eo = expert_offsets[cur.e], n0 = cur.nt * 64, kblk = (n0 + wn * 16) >> 4;
      uint8_t* sfe = sf_out + (size_t)sf_offsets[cur.e] * SFK * 4;
#pragma unroll
      for (int ii = 0; ii < MTW; ++ii) {
        const int i = wm + ii * WM;
        if (i >= ntile) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = cur.mc + i * 16 + g + 8 * h;
          const __nv_bfloat162 g0 = __floats2bfloat162_rn(alpha * acc[ii][0][2 * h], alpha * acc[ii][0][2 * h + 1]);
          const __nv_bfloat162 g1 = __floats2bfloat162_rn(alpha * acc[ii][1][2 * h], alpha * acc[ii][1][2 * h + 1]);
          const __nv_bfloat162 u0 = __floats2bfloat162_rn(alpha * acc[ii][2][2 * h], alpha * acc[ii][2][2 * h + 1]);
          const __nv_bfloat162 u1 = __floats2bfloat162_rn(alpha * acc[ii][3][2 * h], alpha * acc[ii][3][2 * h + 1]);
          const __nv_bfloat162 c0 = gelu_and_mul<Contract>(g0, u0, beta);
          const __nv_bfloat162 c1 = gelu_and_mul<Contract>(g1, u1, beta);
          auto lm = __hmax2(__habs2(c0), __habs2(c1));
          lm = __hmax2(__shfl_xor_sync(0xffffffffu, lm, 1), lm);
          lm = __hmax2(__shfl_xor_sync(0xffffffffu, lm, 2), lm);
          const float vecMax = float(__hmax(lm.x, lm.y));
          float SFValue = sfscale * (vecMax * reciprocal_approximate_ftz(6.0f));
          const __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
          uint8_t fp8SFVal;
          reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
          SFValue = float(tmp);
          const float outputScale =
              SFValue != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(sfscale)) : 0.0f;
          float2 f0 = __bfloat1622float2(c0), f1 = __bfloat1622float2(c1);
          f0.x *= outputScale;
          f0.y *= outputScale;
          f1.x *= outputScale;
          f1.y *= outputScale;
          uint32_t w0 = e2m1_pair(f0.x, f0.y) << (8 * t), w1 = e2m1_pair(f1.x, f1.y) << (8 * t);
          w0 |= __shfl_xor_sync(0xffffffffu, w0, 1);
          w1 |= __shfl_xor_sync(0xffffffffu, w1, 1);
          w0 |= __shfl_xor_sync(0xffffffffu, w0, 2);
          w1 |= __shfl_xor_sync(0xffffffffu, w1, 2);
          if (r < cur.m) {
            uint32_t* orow = out + (size_t)(eo + r) * (NG / 8) + (n0 + wn * 16) / 8;
            if (t == 0) orow[0] = w0;
            if (t == 1) orow[1] = w1;
            if (t == 2) {
              const int off =
                  (((r >> 7) * SFK + (kblk >> 2)) << 9) | ((r & 31) << 4) | (((r >> 5) & 3) << 2) | (kblk & 3);
              sfe[off] = fp8SFVal;
            }
          }
        }
      }
    }
    if (!next_seg(cur)) break;
  }
  cp_wait<0>();
}
using Launch = int (*)(int, void*, void*, const void*, const void*, const void*, const void*, const void*, const void*,
                       const void*, const void*, const void*, const void*, int, int, float, cudaStream_t);
template <int R, int WM, int ST, int P, int BKB = 64>
int launch(int numerics, void* out, void* sf_out, const void* a, const void* b, const void* a_sf, const void* b_sf,
           const void* alphas, const void* gscale, const void* problem_sizes, const void* expert_offsets,
           const void* sf_offsets, const void* amap, int num_experts, int rows, float beta, cudaStream_t stream) {
  constexpr int LD = BKB + 16, KKC = BKB / 32;
  constexpr int SMEM = ST * (R * LD + 128 * LD) + ST * KKC * (R + 128) * 4;
  const int ki = (numerics != 0) + 2 * (amap != nullptr);
  auto kern = ki == 0 ? reax_moe_gu<R, WM, ST, BKB, true, false>
              : ki == 1 ? reax_moe_gu<R, WM, ST, BKB, false, false>
              : ki == 2 ? reax_moe_gu<R, WM, ST, BKB, true, true>
                        : reax_moe_gu<R, WM, ST, BKB, false, true>;
  static bool attr[4] = {false, false, false, false};
  static int sms = 0;
  if (!attr[ki]) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) return 2;
    attr[ki] = true;
  }
  if (P && !sms) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
      return 3;
  }
  const int items = (NG / 64) * (num_experts + (rows + R - 1) / R);
  const int grid = P ? (P * sms < items ? P * sms : items) : items;
  kern<<<grid, 128 * WM, SMEM, stream>>>(
      static_cast<uint32_t*>(out), static_cast<uint8_t*>(sf_out), static_cast<const uint8_t*>(a),
      static_cast<const uint8_t*>(b), static_cast<const uint8_t*>(a_sf), static_cast<const uint8_t*>(b_sf),
      static_cast<const float*>(alphas), static_cast<const float*>(gscale), static_cast<const int32_t*>(problem_sizes),
      static_cast<const int32_t*>(expert_offsets), static_cast<const int32_t*>(sf_offsets),
      static_cast<const int32_t*>(amap), num_experts, beta);
  return static_cast<int>(cudaGetLastError());
}
struct Variant {
  const char* name;
  Launch fn;
};
#define REAX_V(R, WM, ST) {"r" #R "_m" #WM "_s" #ST, launch<R, WM, ST, 0>}
#define REAX_K(R, WM, ST) {"r" #R "_m" #WM "_s" #ST "_k128", launch<R, WM, ST, 0, 128>}
#define REAX_P(P, R, WM, ST) {"p" #P "_r" #R "_m" #WM "_s" #ST, launch<R, WM, ST, P>}
constexpr Variant kVariants[] = {
    REAX_V(64, 1, 4), REAX_V(64, 2, 4), REAX_V(64, 1, 3), REAX_V(64, 2, 3), REAX_V(128, 2, 3),
    REAX_V(128, 2, 4), REAX_V(128, 1, 3), REAX_V(32, 1, 3), REAX_V(32, 2, 3), REAX_V(32, 1, 4),
    REAX_P(1, 128, 2, 4), REAX_P(1, 128, 2, 3),
    REAX_K(64, 2, 3), REAX_K(64, 1, 3), REAX_K(128, 2, 2), REAX_K(32, 1, 3),
};
#undef REAX_V
#undef REAX_K
#undef REAX_P
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);
}
extern "C" int reax_moegu_variants() { return kNumVariants; }
extern "C" const char* reax_moegu_variant_name(int v) { return v >= 0 && v < kNumVariants ? kVariants[v].name : ""; }
extern "C" int reax_moegu_launch(int variant, int numerics, void* out, void* sf_out, const void* a, const void* b,
                                 const void* a_sf, const void* b_sf, const void* alphas, const void* gscale,
                                 const void* problem_sizes, const void* expert_offsets, const void* sf_offsets,
                                 int num_experts, int rows, float beta, void* stream) {
  if (variant < 0 || variant >= kNumVariants || numerics < 0 || numerics > 1 || num_experts > kMaxExperts) return 1;
  return kVariants[variant].fn(numerics, out, sf_out, a, b, a_sf, b_sf, alphas, gscale, problem_sizes, expert_offsets,
                               sf_offsets, nullptr, num_experts, rows, beta, static_cast<cudaStream_t>(stream));
}
extern "C" int reax_moegu_launch_g(int variant, int numerics, void* out, void* sf_out, const void* a, const void* b,
                                   const void* a_sf, const void* b_sf, const void* alphas, const void* gscale,
                                   const void* problem_sizes, const void* expert_offsets, const void* sf_offsets,
                                   const void* amap, int num_experts, int rows, float beta, void* stream) {
  if (variant < 0 || variant >= kNumVariants || numerics < 0 || numerics > 1 || num_experts > kMaxExperts || !amap)
    return 1;
  return kVariants[variant].fn(numerics, out, sf_out, a, b, a_sf, b_sf, alphas, gscale, problem_sizes, expert_offsets,
                               sf_offsets, amap, num_experts, rows, beta, static_cast<cudaStream_t>(stream));
}
