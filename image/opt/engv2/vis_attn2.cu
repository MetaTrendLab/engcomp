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
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
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
template <bool FULL>
__device__ __forceinline__ void softmax_blk(float (&s)[16][4], float (&o)[kNT][4], float (&mi)[2], float (&sp)[2], int t, int kv, int blk, float scale) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float mx = -INFINITY;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        s[n][2 * h + c] = __fmul_rn(kLog2e, __fmul_rn(scale, s[n][2 * h + c]));
        if (FULL || 8 * n + 2 * t + c < kv) mx = fmaxf(mx, s[n][2 * h + c]);
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
          const float p = (FULL || 8 * n + 2 * t + c < kv) ? exp2f(__fsub_rn(s[n][2 * h + c], mnew)) : 0.f;
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
}
template <bool SKIP, bool X2>
__device__ __forceinline__ void s_phase(float (&s)[16][4], const uint32_t (&qa)[kKC][4], uint32_t s_k, int lane, int kv) {
#pragma unroll
  for (int kc = 0; kc < kKC; ++kc) {
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      if (SKIP && 16 * np >= kv) continue;
      if (X2 && kc == kKC - 1) {
        uint32_t b2[2];
        ldsm2(b2, s_k + ((16 * np + (lane & 7) + (((lane >> 3) & 1) << 3)) * kLD + 16 * kc) * 2);
        mma_k8(s[2 * np], qa[kc][0], qa[kc][1], b2[0]);
        mma_k8(s[2 * np + 1], qa[kc][0], qa[kc][1], b2[1]);
        continue;
      }
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
}
template <bool SKIP>
__device__ __forceinline__ void pv_phase(float (&o)[kNT][4], const float (&s)[16][4], uint32_t s_v, int lane, int kv) {
#pragma unroll
  for (int kc = 0; kc < 8; ++kc) {
    if (SKIP && 16 * kc >= kv) continue;
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
}
template <int W, int MINB, int OPT>
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
  const bool live = !(OPT & 8) || q0 + warp * 16 < len;
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
    if (live) {
      if ((OPT & 1) && kv < kBK)
        s_phase<true, (OPT & 4) != 0>(s, qa, s_k, lane, kv);
      else
        s_phase<false, (OPT & 4) != 0>(s, qa, s_k, lane, kv);
    }
    __syncthreads();
    if (more) {
      load_tile<W * 32>(s_k, kb + (int64_t)(kbase + kBK) * ts, ts, min(kBK, len - kbase - kBK));
      cp_commit();
    }
    if (live) {
      if ((OPT & 2) && kv == kBK)
        softmax_blk<true>(s, o, mi, sp, t, kv, blk, scale);
      else
        softmax_blk<false>(s, o, mi, sp, t, kv, blk, scale);
    }
    if (more) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    if (live) {
      if ((OPT & 1) && kv < kBK)
        pv_phase<true>(o, s, s_v, lane, kv);
      else
        pv_phase<false>(o, s, s_v, lane, kv);
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
template <int W, int MINB, int OPT>
int launch(void* out, const void* q, const void* k, const void* v, const void* cu, int seqs, int max_len, int heads,
           long long ts, float scale, cudaStream_t stream) {
  dim3 grid((max_len + 16 * W - 1) / (16 * W), heads, seqs);
  reax_vattn<W, MINB, OPT><<<grid, W * 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const int32_t*>(cu), ts, scale);
  return static_cast<int>(cudaGetLastError());
}
template <bool QREG>
__device__ __forceinline__ void q_frag2(uint32_t (&qa)[2][4], const uint32_t (&qr)[2][kKC][4], const __nv_bfloat16* qb,
                                        int64_t ts, const int (&rr)[2][2], int len, int t, int kc) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    if (QREG) {
#pragma unroll
      for (int i = 0; i < 4; ++i) qa[m][i] = qr[m][kc][i];
    } else {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int colq = kc * 16 + h * 8 + 2 * t;
        const bool cv = colq < kD;
        qa[m][2 * h] = (cv && rr[m][0] < len) ? __ldg(reinterpret_cast<const unsigned int*>(qb + (int64_t)rr[m][0] * ts + colq)) : 0u;
        qa[m][2 * h + 1] = (cv && rr[m][1] < len) ? __ldg(reinterpret_cast<const unsigned int*>(qb + (int64_t)rr[m][1] * ts + colq)) : 0u;
      }
    }
  }
}
template <bool SKIP, bool QREG>
__device__ __forceinline__ void s_phase2(float (&s)[2][16][4], const uint32_t (&qr)[2][kKC][4], const __nv_bfloat16* qb,
                                         int64_t ts, const int (&rr)[2][2], int len, int t, uint32_t s_k, int lane, int kv) {
#pragma unroll 1
  for (int kc = 0; kc < kKC; ++kc) {
    uint32_t qa[2][4];
    q_frag2<QREG>(qa, qr, qb, ts, rr, len, t, kc);
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      if (SKIP && 16 * np >= kv) continue;
      uint32_t b[4];
      const int row = 16 * np + (lane & 7) + ((lane >> 4) << 3);
      const int col = 16 * kc + (((lane >> 3) & 1) << 3);
      ldsm4(b, s_k + (row * kLD + col) * 2);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        mma_k8(s[m][2 * np], qa[m][0], qa[m][1], b[0]);
        mma_k8(s[m][2 * np + 1], qa[m][0], qa[m][1], b[2]);
        if (kc < kKC - 1) {
          mma_k8(s[m][2 * np], qa[m][2], qa[m][3], b[1]);
          mma_k8(s[m][2 * np + 1], qa[m][2], qa[m][3], b[3]);
        }
      }
    }
  }
}
template <bool SKIP>
__device__ __forceinline__ void pv_phase2(float (&o)[2][kNT][4], const float (&s)[2][16][4], uint32_t s_v, int lane, int kv) {
#pragma unroll
  for (int kc = 0; kc < 8; ++kc) {
    if (SKIP && 16 * kc >= kv) continue;
    uint32_t pa[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      pa[m][0] = pack_bf16(s[m][2 * kc][0], s[m][2 * kc][1]);
      pa[m][1] = pack_bf16(s[m][2 * kc][2], s[m][2 * kc][3]);
      pa[m][2] = pack_bf16(s[m][2 * kc + 1][0], s[m][2 * kc + 1][1]);
      pa[m][3] = pack_bf16(s[m][2 * kc + 1][2], s[m][2 * kc + 1][3]);
    }
    const int row = 16 * kc + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
    for (int dp = 0; dp < 4; ++dp) {
      uint32_t b[4];
      ldsm4t(b, s_v + (row * kLD + 8 * (2 * dp + (lane >> 4))) * 2);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        mma_k8(o[m][2 * dp], pa[m][0], pa[m][1], b[0]);
        mma_k8(o[m][2 * dp], pa[m][2], pa[m][3], b[1]);
        mma_k8(o[m][2 * dp + 1], pa[m][0], pa[m][1], b[2]);
        mma_k8(o[m][2 * dp + 1], pa[m][2], pa[m][3], b[3]);
      }
    }
    uint32_t b2[2];
    ldsm2t(b2, s_v + (row * kLD + 64) * 2);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      mma_k8(o[m][8], pa[m][0], pa[m][1], b2[0]);
      mma_k8(o[m][8], pa[m][2], pa[m][3], b2[1]);
    }
  }
}
template <int W, int MINB, int OPT, bool QREG>
__global__ void __launch_bounds__(W * 32, MINB) reax_vattn_m2(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ q,
                                                             const __nv_bfloat16* __restrict__ k,
                                                             const __nv_bfloat16* __restrict__ v,
                                                             const int32_t* __restrict__ cu, int64_t ts, float scale) {
  __shared__ __align__(16) __nv_bfloat16 sk[kBK * kLD];
  __shared__ __align__(16) __nv_bfloat16 sv[kBK * kLD];
  const int seq = blockIdx.z, head = blockIdx.y;
  const int t0 = cu[seq], len = cu[seq + 1] - t0;
  const int q0 = blockIdx.x * (32 * W);
  if (q0 >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const __nv_bfloat16* kb = k + (int64_t)t0 * ts + head * kD;
  const __nv_bfloat16* vb = v + (int64_t)t0 * ts + head * kD;
  const __nv_bfloat16* qb = q + (int64_t)t0 * ts + head * kD;
  const uint32_t s_k = static_cast<uint32_t>(__cvta_generic_to_shared(sk));
  const uint32_t s_v = static_cast<uint32_t>(__cvta_generic_to_shared(sv));
  const int nblk = (len + kBK - 1) / kBK;
  load_tile<W * 32>(s_k, kb, ts, min(kBK, len));
  cp_commit();
  load_tile<W * 32>(s_v, vb, ts, min(kBK, len));
  cp_commit();
  int rr[2][2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    rr[m][0] = q0 + warp * 32 + m * 16 + g;
    rr[m][1] = rr[m][0] + 8;
  }
  uint32_t qr[2][kKC][4];
  if (QREG) {
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
      uint32_t qa[2][4];
      q_frag2<false>(qa, qr, qb, ts, rr, len, t, kc);
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) qr[m][kc][i] = qa[m][i];
    }
  }
  float o[2][kNT][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < kNT; ++n) o[m][n][0] = o[m][n][1] = o[m][n][2] = o[m][n][3] = 0.f;
  float mi[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, sp[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int blk = 0; blk < nblk; ++blk) {
    const int kbase = blk * kBK, kv = min(kBK, len - kbase);
    const bool more = blk + 1 < nblk;
    cp_wait<1>();
    __syncthreads();
    float s[2][16][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 16; ++n) s[m][n][0] = s[m][n][1] = s[m][n][2] = s[m][n][3] = 0.f;
    if ((OPT & 1) && kv < kBK)
      s_phase2<true, QREG>(s, qr, qb, ts, rr, len, t, s_k, lane, kv);
    else
      s_phase2<false, QREG>(s, qr, qb, ts, rr, len, t, s_k, lane, kv);
    __syncthreads();
    if (more) {
      load_tile<W * 32>(s_k, kb + (int64_t)(kbase + kBK) * ts, ts, min(kBK, len - kbase - kBK));
      cp_commit();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if ((OPT & 2) && kv == kBK)
        softmax_blk<true>(s[m], o[m], mi[m], sp[m], t, kv, blk, scale);
      else
        softmax_blk<false>(s[m], o[m], mi[m], sp[m], t, kv, blk, scale);
    }
    if (more) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    if ((OPT & 1) && kv < kBK)
      pv_phase2<true>(o, s, s_v, lane, kv);
    else
      pv_phase2<false>(o, s, s_v, lane, kv);
    __syncthreads();
    if (more) {
      load_tile<W * 32>(s_v, vb + (int64_t)(kbase + kBK) * ts, ts, min(kBK, len - kbase - kBK));
      cp_commit();
    }
  }
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = rr[m][h];
      if (row >= len) continue;
      const float den = sp[m][h] == 0.f ? 1.f : sp[m][h];
      const float alpha = 1.f / den;
      __nv_bfloat16* dst = out + (int64_t)(t0 + row) * ts + head * kD + 2 * t;
#pragma unroll
      for (int n = 0; n < kNT; ++n)
        *reinterpret_cast<__nv_bfloat162*>(dst + 8 * n) =
            __floats2bfloat162_rn(alpha * o[m][n][2 * h], alpha * o[m][n][2 * h + 1]);
    }
  }
}
template <int W, int MINB, int OPT, bool QREG>
int launch_m2(void* out, const void* q, const void* k, const void* v, const void* cu, int seqs, int max_len, int heads,
              long long ts, float scale, cudaStream_t stream) {
  dim3 grid((max_len + 32 * W - 1) / (32 * W), heads, seqs);
  reax_vattn_m2<W, MINB, OPT, QREG><<<grid, W * 32, 0, stream>>>(
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
#define REAX_V(W, MB) {"w" #W "_b" #MB, launch<W, MB, 0>}
#define REAX_VO(W, MB, O) {"w" #W "_b" #MB "_o" #O, launch<W, MB, O>}
#define REAX_VM(W, MB, O, QR) {"m2w" #W "_b" #MB "_o" #O "_q" #QR, launch_m2<W, MB, O, QR != 0>}
constexpr Variant kVariants[] = {REAX_V(4, 1),      REAX_V(2, 1),      REAX_V(8, 1),      REAX_VO(4, 1, 1),
                                 REAX_VO(4, 1, 2),  REAX_VO(4, 1, 3),  REAX_VO(4, 2, 3),  REAX_VO(4, 1, 7),
                                 REAX_VO(2, 1, 3),  REAX_VO(3, 2, 3),  REAX_VO(5, 1, 3),  REAX_VO(5, 2, 3),
                                 REAX_VO(6, 1, 3),  REAX_VO(7, 1, 3),  REAX_VO(8, 1, 3),
                                 REAX_VM(4, 1, 2, 0), REAX_VM(4, 2, 2, 0), REAX_VM(4, 1, 2, 1), REAX_VM(4, 1, 3, 0),
                                 REAX_VM(2, 1, 2, 0), REAX_VM(2, 2, 2, 0), REAX_VM(3, 2, 2, 0), REAX_VM(8, 1, 2, 0),
                                 REAX_VM(2, 2, 2, 1)};
#undef REAX_V
#undef REAX_VO
#undef REAX_VM
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
