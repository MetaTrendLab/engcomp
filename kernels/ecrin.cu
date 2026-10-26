// engcomp ROTIN: the "in" families' int8 inputs rotated per 128-group of the lo order (randomised Hadamard, the same
// R = diag(d) H_128 / sqrt(128) as q8.py's rot_matrix() and ecrot.cu: d_i = -1 where bit i of EC_NEG is set), so the
// per-row scale sees outliers spread over their group; the family's weight rows carry the same rotation (q8.py Block),
// so the q8g / q8h kernels and their epilogues are unchanged. Standalone cubin: no kernel of q8 / ecq8 / ecrot changes.
//   ec_addnormr: ec_addnorm (q8n_addnorm with x optional) whose prepare rotates before the amax: each thread's 8
//                consecutive lo columns are one sixteenth of a group (16 threads = half a warp): signs, butterflies
//                over index bits 0-2 in registers and 3-6 over lanes (xor 1, 2, 4, 8); H unnormalised, 1 / sqrt(128)
//                folded into the stored scale: int8 = rn(w' 127 / amax(w')), sa = amax(w') / 127 / sqrt(128).
//   ec_prepr:    the same prepare from a bf16 row x (q8x_prep's arguments, one CTA of 320 threads per row).
//   ec_addnormrf / ec_preprf (engcomp ROTF, Kl = 2560): the whole row rotated, R = diag(d20) (H_20 / sqrt(20) x R_128)
//                (Kronecker; d20 and the Paley I H_20 in rotf_consts.inc = q8.py rot_full()): signs d20, the 128-group
//                rotation above, then each position j of a group mixed over the 20 groups through shared memory,
//                y[h][j] = sum_g H20[g][h] x[g][j]; 1 / sqrt(20) also folded into the stored scale.
// Build: nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o ecrin.cubin ecrin.cu (rotf_consts.inc next to it)
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

__device__ __constant__ uint32_t EC_NEG[4] = {0x1208d64eu, 0x42bddcf8u, 0x6324f493u, 0x8e8d7a2bu};
constexpr float EC_RS = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr float EC_RS20 = 0.22360679774997896f;  // 1 / sqrt(20)
#include "rotf_consts.inc"
constexpr int D = 2560;

// rotate the 8 values of thread t (group positions 8 (t % 16) .. + 7); every lane of the warp must call it
DEVI void ec_rot8(float (&v)[8], int t) {
    const int l = t & 15;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = 8 * l + e;
        if ((EC_NEG[i >> 5] >> (i & 31)) & 1u) v[e] = -v[e];
    }
#pragma unroll
    for (int m = 1; m < 8; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (!(e & m)) {
                const float a = v[e], b = v[e | m];
                v[e] = a + b;
                v[e | m] = a - b;
            }
#pragma unroll
    for (int m = 1; m <= 8; m <<= 1) {
        const bool hi = l & m;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? o - v[e] : v[e] + o;
        }
    }
}

// the rotated prepare of the row staged in xs (320 threads, Kl % 128 == 0): xq [r, Kl], sa[r], xh [r, KH]
// FULL (ROTF, Kl == 2560): d20 signs first, the 20-group mix after the 128-group rotation (fs: 2560 floats of smem)
template <bool FULL>
DEVI void prep_rot(const __nv_bfloat16* xs, int r, int t, float* red, int8_t* __restrict__ xq, float* __restrict__ sa,
                   __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo, const float* __restrict__ mu,
                   const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH, float* fs = nullptr) {
    const int c = t * 8;
    float q[8];
    const bool act = c < Kl;
    if (act) {
        const int4 i0 = *reinterpret_cast<const int4*>(lo + c), i1 = *reinterpret_cast<const int4*>(lo + c + 4);
        const int ii[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
        const float4 m0 = *reinterpret_cast<const float4*>(mu + c), m1 = *reinterpret_cast<const float4*>(mu + c + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(is + c), s1 = *reinterpret_cast<const float4*>(is + c + 4);
        const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = (__bfloat162float(xs[ii[e]]) - mm[e]) * sv[e];
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = 0.f;
    }
    if constexpr (FULL) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if ((EC_NEG20[(c + e) >> 5] >> ((c + e) & 31)) & 1u) q[e] = -q[e];
    }
    ec_rot8(q, t);
    if constexpr (FULL) {   // thread t: group h = t / 16, positions j = 8 (t % 16) + e; fs[g * 128 + j]
        *reinterpret_cast<float4*>(fs + c) = make_float4(q[0], q[1], q[2], q[3]);
        *reinterpret_cast<float4*>(fs + c + 4) = make_float4(q[4], q[5], q[6], q[7]);
        __syncthreads();
        const int h = t >> 4, j0 = 8 * (t & 15);
        const uint32_t col = EC_P20[h];
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = 0.f;
#pragma unroll 4
        for (int g = 0; g < 20; ++g) {
            const float4 a = *reinterpret_cast<const float4*>(fs + g * 128 + j0);
            const float4 b = *reinterpret_cast<const float4*>(fs + g * 128 + j0 + 4);
            const float sg = ((col >> g) & 1u) ? -1.f : 1.f;
            q[0] = fmaf(sg, a.x, q[0]); q[1] = fmaf(sg, a.y, q[1]); q[2] = fmaf(sg, a.z, q[2]); q[3] = fmaf(sg, a.w, q[3]);
            q[4] = fmaf(sg, b.x, q[4]); q[5] = fmaf(sg, b.y, q[5]); q[6] = fmaf(sg, b.z, q[6]); q[7] = fmaf(sg, b.w, q[7]);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(q[e]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __syncthreads();
    if ((t & 31) == 0) red[t >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 10; ++w) amax = fmaxf(amax, red[w]);
    amax = fmaxf(amax, 1e-12f);
    const float inv = 127.f / amax;
    if (t == 0) sa[r] = FULL ? amax / 127.f * EC_RS * EC_RS20 : amax / 127.f * EC_RS;
    if (act) {
        uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(q[e] * inv) & 0xff) << (8 * e);
#pragma unroll
        for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(q[4 + e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint2*>(xq + (size_t)r * Kl + c) = make_uint2(lo4, hi4);
    }
    for (int j = t; j < KH; j += 320) xh[(size_t)r * KH + j] = xs[hi[j]];
}

// ec_addnorm (q8n_addnorm's add + RMSNorm text, x optional) + the rotated prepare
template <bool FULL>
DEVI void addnormr(
    const float* __restrict__ part, int P, const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ w1p, float eps, __nv_bfloat16* __restrict__ hn, __nv_bfloat16* __restrict__ x, int M,
    int8_t* __restrict__ xq, float* __restrict__ sa, __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo,
    const float* __restrict__ mu, const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH, float* fs) {
    const int r = blockIdx.x, t = threadIdx.x, c = t * 8;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[D];
    float v[8];
    {
        const uint4 hr = *reinterpret_cast<const uint4*>(h + (size_t)r * D + c);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hr);
        float yy[8];
        if (P > 0) {
            const float* pr = part + (size_t)r * D + c;
            float4 a0 = *reinterpret_cast<const float4*>(pr), a1 = *reinterpret_cast<const float4*>(pr + 4);
            for (int p = 1; p < P; ++p) {
                const float* q = pr + (size_t)p * M * D;
                const float4 b0 = *reinterpret_cast<const float4*>(q), b1 = *reinterpret_cast<const float4*>(q + 4);
                a0.x += b0.x; a0.y += b0.y; a0.z += b0.z; a0.w += b0.w;
                a1.x += b1.x; a1.y += b1.y; a1.z += b1.z; a1.w += b1.w;
            }
            const float s[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) yy[e] = __bfloat162float(__float2bfloat16_rn(s[e]));
        } else {
            const uint4 yr = *reinterpret_cast<const uint4*>(y + (size_t)r * D + c);
            const __nv_bfloat16* yb = reinterpret_cast<const __nv_bfloat16*>(&yr);
#pragma unroll
            for (int e = 0; e < 8; ++e) yy[e] = __bfloat162float(yb[e]);
        }
        uint4 ho;
        __nv_bfloat16* hob = reinterpret_cast<__nv_bfloat16*>(&ho);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            hob[e] = __float2bfloat16_rn(__bfloat162float(hb[e]) + yy[e]);
            v[e] = __bfloat162float(hob[e]);
        }
        *reinterpret_cast<uint4*>(hn + (size_t)r * D + c) = ho;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss = fmaf(v[e], v[e], ss);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 10; ++w) tot += red[w];
    const float rr = rsqrtf(tot * (1.0f / D) + eps);
    {
        const float4 w0 = *reinterpret_cast<const float4*>(w1p + c), w1 = *reinterpret_cast<const float4*>(w1p + c + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        uint4 xo;
        __nv_bfloat16* xob = reinterpret_cast<__nv_bfloat16*>(&xo);
#pragma unroll
        for (int e = 0; e < 8; ++e) xob[e] = __float2bfloat16_rn((v[e] * rr) * ww[e]);
        if (x != nullptr) *reinterpret_cast<uint4*>(x + (size_t)r * D + c) = xo;
        *reinterpret_cast<uint4*>(xs + c) = xo;
    }
    __syncthreads();
    prep_rot<FULL>(xs, r, t, red, xq, sa, xh, lo, mu, is, Kl, hi, KH, fs);
}

// the rotated prepare of x [M, ldx] bf16 rows (2560 columns), one CTA of 320 threads per row
template <bool FULL>
DEVI void prepr(const __nv_bfloat16* __restrict__ x, int ldx, int8_t* __restrict__ xq, float* __restrict__ sa,
                __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo, const float* __restrict__ mu,
                const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH, float* fs) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[D];
    const __nv_bfloat16* xr = x + (size_t)r * ldx;
    for (int j = t; j < D / 8; j += 320) *reinterpret_cast<uint4*>(xs + 8 * j) = *reinterpret_cast<const uint4*>(xr + 8 * j);
    __syncthreads();
    prep_rot<FULL>(xs, r, t, red, xq, sa, xh, lo, mu, is, Kl, hi, KH, fs);
}

#define EC_ADDNORM_ARGS                                                                                            \
    const float *__restrict__ part, int P, const __nv_bfloat16 *__restrict__ y, const __nv_bfloat16 *__restrict__ h, \
        const float *__restrict__ w1p, float eps, __nv_bfloat16 *__restrict__ hn, __nv_bfloat16 *__restrict__ x,   \
        int M, int8_t *__restrict__ xq, float *__restrict__ sa, __nv_bfloat16 *__restrict__ xh,                     \
        const int *__restrict__ lo, const float *__restrict__ mu, const float *__restrict__ is, int Kl,           \
        const int *__restrict__ hi, int KH
#define EC_PREP_ARGS                                                                                               \
    const __nv_bfloat16 *__restrict__ x, int ldx, int8_t *__restrict__ xq, float *__restrict__ sa,                \
        __nv_bfloat16 *__restrict__ xh, const int *__restrict__ lo, const float *__restrict__ mu,                   \
        const float *__restrict__ is, int Kl, const int *__restrict__ hi, int KH
extern "C" __global__ void __launch_bounds__(320) ec_addnormr(EC_ADDNORM_ARGS) {
    addnormr<false>(part, P, y, h, w1p, eps, hn, x, M, xq, sa, xh, lo, mu, is, Kl, hi, KH, nullptr);
}
extern "C" __global__ void __launch_bounds__(320) ec_prepr(EC_PREP_ARGS) {
    prepr<false>(x, ldx, xq, sa, xh, lo, mu, is, Kl, hi, KH, nullptr);
}
extern "C" __global__ void __launch_bounds__(320) ec_addnormrf(EC_ADDNORM_ARGS) {
    __shared__ __align__(16) float fs[D];
    addnormr<true>(part, P, y, h, w1p, eps, hn, x, M, xq, sa, xh, lo, mu, is, Kl, hi, KH, fs);
}
extern "C" __global__ void __launch_bounds__(320) ec_preprf(EC_PREP_ARGS) {
    __shared__ __align__(16) float fs[D];
    prepr<true>(x, ldx, xq, sa, xh, lo, mu, is, Kl, hi, KH, fs);
}
