// engcomp EP (ours, on G9XP): the q8gr twins (gate|up GEMM + SwiGLU + ROT 128-point rotation -> the down family's int8
// input) with our own epilogue. G9XP's epi_g_rotK stages the 128 SwiGLU outputs of a row in smem and rotates them with
// one warp per row (4 values per lane: 2 butterfly stages in registers, 5 by shuffles, then a 5-shuffle amax), reading
// mu / is from global memory for every element. Here:
//   * each thread loads the mu / is of its 8 columns once (registers);
//   * staging rows of 4 quarters x (32 + 4) floats (row stride 144 = 16 mod 32): the row pass reads conflict-free;
//   * the row pass: 4 lanes per row, 32 consecutive elements per lane (8 rows per warp per step): index bits 0..4 by
//     butterflies in registers, bits 5 / 6 by lane xor 1 / 2, amax over the 32 values then 2 shuffles; int8 stored as
//     two 16-byte words per lane.
// Every element sees G9XP's operations in G9XP's order (the same SwiGLU / LUT, centring, sign flips, the butterfly
// stages over index bits 0, 1, ..., 6 with the same lower x + y / upper x - y ops, fmaxf amax, 127 / amax, rn), so hq /
// hs are byte-identical to q8gr's (tools/engcomp/ecep_test.py). Mainloop, dequant and loaders are G9XP's (q8k).
// Build (G9XP's q8.cu next to it): nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o ecep.cubin ecep.cu
#define Q8K 1
#include "q8.cu"

namespace ecep {
using namespace q8;

constexpr int QS = 36, LDS = 4 * QS;   // quarter stride / row stride in floats

template <class C, int LV>
DEVI void epi(const ArgsG& a, const float (&fa)[C::MT][C::NT][4], int m0, int o0, int H, U* smem) {
    static_assert(32 % (C::WTN / 2) == 0, "a warp column's outputs inside one 32-column quarter");
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* S = reinterpret_cast<float*>(smem);   // [BM][LDS] (smem free after the loops)
    float mu[NH][2], is[NH][2];
    const bool cen = a.mu != nullptr;
#pragma unroll
    for (int j = 0; j < NH; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
            mu[j][e] = cen ? a.mu[c] : 0.f;
            is[j][e] = cen ? a.is[c] : 1.f;
        }
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int cb = wn * (C::WTN / 2);   // the warp column's first output (one quarter)
            float* row = S + rl * LDS + (cb / 32) * QS + cb % 32 + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                float x2[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x;
                    if constexpr ((LV & 1) != 0) {
                        const float s = __bfloat162float(__ushort_as_bfloat16(__ldg(&q8k_silu[__bfloat16_as_ushort(g)])));
                        x = __bfloat162float(__float2bfloat16_rn(s * __bfloat162float(u)));
                    } else {
                        x = __bfloat162float(swiglu1(g, u));
                    }
                    if (cen) x = (x - mu[j][e]) * is[j][e];
                    x2[e] = x;
                }
                *reinterpret_cast<float2*>(row + j * 8) = make_float2(x2[0], x2[1]);
            }
        }
    __syncthreads();
    const int g = o0 / 128;
    const int q = lane & 3;   // quarter: index bits 5 (q & 1) and 6 (q & 2)
    for (int rg = warp * 8; rg < C::BM; rg += C::CW * 8) {
        const int rl = rg + lane / 4;
        const float* src = S + rl * LDS + q * QS;
        float y[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 v = *reinterpret_cast<const float4*>(src + 4 * k);
            y[4 * k] = v.x, y[4 * k + 1] = v.y, y[4 * k + 2] = v.z, y[4 * k + 3] = v.w;
        }
#pragma unroll
        for (int e = 0; e < 32; ++e) y[e] = rot_sign(q * 32 + e, y[e]);
        fwht_reg<32>(y);   // index bits 0 .. 4
#pragma unroll
        for (int m = 1; m <= 2; m <<= 1)   // bits 5, 6: G9XP's fwht_lanes ops
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const float o = __shfl_xor_sync(0xffffffffu, y[e], m);
                y[e] = (lane & m) ? o - y[e] : y[e] + o;
            }
        float mx = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) mx = fmaxf(mx, fabsf(y[e]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, 1e-12f);
        const int r = m0 + rl;
        if (r >= a.M) continue;
        const float inv = 127.f / mx;
        if (q == 0) a.hs[(size_t)g * a.M + r] = mx / 127.f * ROT_RS;
        uint32_t w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t q4 = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) q4 |= (uint32_t)(__float2int_rn(y[4 * k + e] * inv) & 0xff) << (8 * e);
            w[k] = q4;
        }
        uint4* dst = reinterpret_cast<uint4*>(a.hq + (size_t)r * H + o0 + q * 32);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

template <class C, int LV>
DEVI void tile(const ArgsG& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    float fa[C::MT][C::NT][4];
    q8k::in_passes<C, LV>(a.A, a.B, a.AH, a.BH, a.sa, a.sw, a.bias, a.M, a.K, a.KH, a.lda, a.ldb, m0, o0, H + o0, H,
                          smem, fa);
    epi<C, LV>(a, fa, m0, o0, H, smem);
}

template <class CB, class CS, int LV>
DEVI void hetero(const ArgsG& a, U* smem) {
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile<CB, LV>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile<CS, LV>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}
}  // namespace ecep

#define ECEP(NAME, CB, CS, LV)                                                                                    \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsG a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        ecep::hetero<CB, CS, LV>(a, reinterpret_cast<q8::U*>(smem_raw));                                          \
    }
#define ECEP8(S, CS)                                                                                              \
    ECEP(epgr_b128x256_##S##_k1, W_B128x256, CS, 1)                                                                \
    ECEP(epgr_b128x256_##S##_k2, W_B128x256, CS, 2)                                                                \
    ECEP(epgr_b128x256_##S##_k3, W_B128x256, CS, 3)                                                                \
    ECEP(epgr_b128x256_##S##_k4, W_B128x256, CS, 4)                                                                \
    ECEP(epgr_b128x256_##S##_k5, W_B128x256, CS, 5)                                                                \
    ECEP(epgr_b128x256_##S##_k6, W_B128x256, CS, 6)                                                                \
    ECEP(epgr_b128x256_##S##_k7, W_B128x256, CS, 7)
ECEP8(s64x256, W_S64x256)
ECEP8(s32x256, W_S32x256)
ECEP8(s96x256, W_S96x256)
ECEP(epgr_b128x256_s96x256_k0, W_B128x256, W_S96x256, 0)
