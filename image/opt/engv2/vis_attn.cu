#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
constexpr int kD = 72;
constexpr int kKC = 5;
constexpr int kNT = kD / 8;
constexpr int kBK = 128;
constexpr int kLD = 88;
constexpr float kLog2e = 1.4426950408889634074f;
__device__ __forceinline__ void mma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm2t(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
template <int NT>
__device__ __forceinline__ void load_tile(uint32_t s_base, const __nv_bfloat16* g, int64_t ts, int kv) {
  for (int c = threadIdx.x; c < kBK * (kD / 8); c += NT) {
    const int r = c / (kD / 8), q = c % (kD / 8);
    const bool ok = r < kv;
    cp16(s_base + (r * kLD + q * 8) * 2, ok ? (const void*)(g + r * ts + q * 8) : (const void*)g, ok);
  }
}
template <int W, int MINB>
__global__ void __launch_bounds__(W * 32, MINB) reax_vattn(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ q,
                                                          const __nv_bfloat16* __restrict__ k,
                                                          const __nv_bfloat16* __restrict__ v,
                                                          const int32_t* __restrict__ cu, int64_t ts, float scale) {
  __shared__ __align__(16) __nv_bfloat16 sk[kBK * kLD];
  __shared__ __align__(16) __nv_bfloat16 sv[kBK * kLD];
  const int seq = blockIdx.z, head = blockIdx.y;
  const int t0 = cu[seq], len = cu[seq + 1] - t0;
  const int q0 = blockIdx.x * (16 * W);
  if (q0 >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const __nv_bfloat16* kb = k + (int64_t)t0 * ts + head * kD;
  const __nv_bfloat16* vb = v + (int64_t)t0 * ts + head * kD;
  const uint32_t s_k = static_cast<uint32_t>(__cvta_generic_to_shared(sk));
  const uint32_t s_v = static_cast<uint32_t>(__cvta_generic_to_shared(sv));
  const int nblk = (len + kBK - 1) / kBK;
  load_tile<W * 32>(s_k, kb, ts, min(kBK, len));
  cp_commit();
  load_tile<W * 32>(s_v, vb, ts, min(kBK, len));
  cp_commit();
  const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  uint32_t qa[kKC][4];
#pragma unroll
  for (int kc = 0; kc < kKC; ++kc) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int colq = kc * 16 + h * 8 + 2 * t;
      const bool cv = colq < kD;
      qa[kc][2 * h] = (cv && r0 < len) ? *reinterpret_cast<const uint32_t*>(q + (int64_t)(t0 + r0) * ts + head * kD + colq) : 0u;
      qa[kc][2 * h + 1] = (cv && r1 < len) ? *reinterpret_cast<const uint32_t*>(q + (int64_t)(t0 + r1) * ts + head * kD + colq) : 0u;
    }
  }
  float o[kNT][4];
#pragma unroll
  for (int n = 0; n < kNT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mi[2] = {-INFINITY, -INFINITY}, sp[2] = {0.f, 0.f};
  for (int blk = 0; blk < nblk; ++blk) {
    const int kbase = blk * kBK, kv = min(kBK, len - kbase);
    const bool more = blk + 1 < nblk;
    cp_wait<1>();
    __syncthreads();
    float s[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t b[4];
        const int row = 16 * np + (lane & 7) + ((lane >> 4) << 3);
        const int col = 16 * kc + (((lane >> 3) & 1) << 3);
        ldsm4(b, s_k + (row * kLD + col) * 2);
        mma_k8(s[2 * np], qa[kc][0], qa[kc][1], b[0]);
        mma_k8(s[2 * np + 1], qa[kc][0], qa[kc][1], b[2]);
        if (kc < kKC - 1) {
          mma_k8(s[2 * np], qa[kc][2], qa[kc][3], b[1]);
          mma_k8(s[2 * np + 1], qa[kc][2], qa[kc][3], b[3]);
        }
      }
    }
    __syncthreads();
    if (more) {
      load_tile<W * 32>(s_k, kb + (int64_t)(kbase + kBK) * ts, ts, min(kBK, len - kbase - kBK));
      cp_commit();
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          s[n][2 * h + c] = __fmul_rn(kLog2e, __fmul_rn(scale, s[n][2 * h + c]));
          if (8 * n + 2 * t + c < kv) mx = fmaxf(mx, s[n][2 * h + c]);
        }
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mp = mi[h];
      const float mnew = fmaxf(mp, mx);
      float r = 1.f;
      if (mp < mnew) {
        r = exp2f(mp - mnew);
        sp[h] = __fmul_rn(sp[h], r);
      }
      mi[h] = mnew;
      if (blk) {
#pragma unroll
        for (int n = 0; n < kNT; ++n) {
          o[n][2 * h] = o[n][2 * h] * r;
          o[n][2 * h + 1] = o[n][2 * h + 1] * r;
        }
      }
      float tot = sp[h];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        float part = 0.f;
#pragma unroll
        for (int n = 4 * w; n < 4 * w + 4; ++n) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const float p = (8 * n + 2 * t + c < kv) ? exp2f(__fsub_rn(s[n][2 * h + c], mnew)) : 0.f;
            s[n][2 * h + c] = p;
            part += p;
          }
        }
        part = part + __shfl_xor_sync(0xffffffffu, part, 1);
        part = part + __shfl_xor_sync(0xffffffffu, part, 2);
        tot += part;
      }
      sp[h] = tot;
    }
    if (more) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      uint32_t pa[4];
      pa[0] = pack_bf16(s[2 * kc][0], s[2 * kc][1]);
      pa[1] = pack_bf16(s[2 * kc][2], s[2 * kc][3]);
      pa[2] = pack_bf16(s[2 * kc + 1][0], s[2 * kc + 1][1]);
      pa[3] = pack_bf16(s[2 * kc + 1][2], s[2 * kc + 1][3]);
      const int row = 16 * kc + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        ldsm4t(b, s_v + (row * kLD + 8 * (2 * dp + (lane >> 4))) * 2);
        mma_k8(o[2 * dp], pa[0], pa[1], b[0]);
        mma_k8(o[2 * dp], pa[2], pa[3], b[1]);
        mma_k8(o[2 * dp + 1], pa[0], pa[1], b[2]);
        mma_k8(o[2 * dp + 1], pa[2], pa[3], b[3]);
      }
      uint32_t b2[2];
      ldsm2t(b2, s_v + (row * kLD + 64) * 2);
      mma_k8(o[8], pa[0], pa[1], b2[0]);
      mma_k8(o[8], pa[2], pa[3], b2[1]);
    }
    __syncthreads();
    if (more) {
      load_tile<W * 32>(s_v, vb + (int64_t)(kbase + kBK) * ts, ts, min(kBK, len - kbase - kBK));
      cp_commit();
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = h ? r1 : r0;
    if (row >= len) continue;
    const float den = sp[h] == 0.f ? 1.f : sp[h];
    const float alpha = 1.f / den;
    __nv_bfloat16* dst = out + (int64_t)(t0 + row) * ts + head * kD + 2 * t;
#pragma unroll
    for (int n = 0; n < kNT; ++n)
      *reinterpret_cast<__nv_bfloat162*>(dst + 8 * n) = __floats2bfloat162_rn(alpha * o[n][2 * h], alpha * o[n][2 * h + 1]);
  }
}
template <int W, int MINB>
int launch(void* out, const void* q, const void* k, const void* v, const void* cu, int seqs, int max_len, int heads,
           long long ts, float scale, cudaStream_t stream) {
  dim3 grid((max_len + 16 * W - 1) / (16 * W), heads, seqs);
  reax_vattn<W, MINB><<<grid, W * 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const int32_t*>(cu), ts, scale);
  return static_cast<int>(cudaGetLastError());
}
using Launch = int (*)(void*, const void*, const void*, const void*, const void*, int, int, int, long long, float,
                       cudaStream_t);
struct Variant {
  const char* name;
  Launch fn;
};
#define REAX_V(W, MB) {"w" #W "_b" #MB, launch<W, MB>}
constexpr Variant kVariants[] = {REAX_V(4, 1), REAX_V(2, 1), REAX_V(8, 1)};
#undef REAX_V
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);
}
extern "C" int reax_vattn_variants() { return kNumVariants; }
extern "C" const char* reax_vattn_variant_name(int v) { return v >= 0 && v < kNumVariants ? kVariants[v].name : ""; }
extern "C" int reax_vattn_fwd_v(int variant, void* out, const void* q, const void* k, const void* v, const void* cu, int seqs,
                                int max_len, int heads, int head_dim, long long ts, float scale, void* stream) {
  if (head_dim != kD || ts % 8 || variant < 0 || variant >= kNumVariants) return 1;
  return kVariants[variant].fn(out, q, k, v, cu, seqs, max_len, heads, ts, scale, static_cast<cudaStream_t>(stream));
}
extern "C" int reax_vattn_fwd(void* out, const void* q, const void* k, const void* v, const void* cu, int seqs, int max_len,
                              int heads, int head_dim, long long ts, float scale, void* stream) {
  return reax_vattn_fwd_v(0, out, q, k, v, cu, seqs, max_len, heads, head_dim, ts, scale, stream);
}
