#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
#ifdef F8_L2HINT
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
#endif
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int BK>
DEVI uint32_t swz(int r, int c) {
    if constexpr (BK >= 128) return r * BK + ((c ^ (r & 7)) << 4);
    else return r * BK + ((c ^ ((r >> 1) & 3)) << 4);
}
template <bool V>
struct BoolC { static constexpr bool value = V; };
DEVI float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__device__ __forceinline__ void gemm_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                          __nv_bfloat16* __restrict__ C, int M, int N, int K, int m_base, int tm, int tn,
                                          uint8_t* smem) {
    constexpr int THREADS = WM * WN * 32;
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int MT = WTM / 16, NT = WTN / 8;
    constexpr int CH = BK / 16;
    static_assert(CH == 4 || CH >= 8, "swizzle needs BK 64 or >= 128");
    static_assert(NT % 2 == 0, "B fragments are loaded two n8 tiles at a time");
    uint8_t* sA = smem;
    uint8_t* sB = smem + STAGES * BM * BK;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = m_base + tm * BM, n0 = tn * BN;
    const int KT = K / BK;
    constexpr int LA = BM * CH / THREADS, LB = BN * CH / THREADS;
    static_assert(BM * CH % THREADS == 0 && BN * CH % THREADS == 0, "copy slots");
    uint32_t soA[LA], soB[LB];
    const uint8_t* gA[LA];
    const uint8_t* gB[LB];
    bool pA[LA];
#pragma unroll
    for (int t = 0; t < LA; t++) {
        int i = tid + t * THREADS, r = i / CH, c = i % CH, gm = m0 + r;
        soA[t] = swz<BK>(r, c);
        pA[t] = gm < M;
        gA[t] = A + (size_t)(pA[t] ? gm : 0) * K + c * 16;
    }
#pragma unroll
    for (int t = 0; t < LB; t++) {
        int i = tid + t * THREADS, r = i / CH, c = i % CH;
        soB[t] = swz<BK>(r, c);
        gB[t] = B + (size_t)(n0 + r) * K + c * 16;
    }
    const uint32_t sA0 = smem_u32(sA), sB0 = smem_u32(sB);
    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        const uint32_t a_base = sA0 + stage * BM * BK, b_base = sB0 + stage * BN * BK;
#pragma unroll
        for (int t = 0; t < LA; t++) cp_async16(a_base + soA[t], gA[t] + k0, pA[t]);
#pragma unroll
        for (int t = 0; t < LB; t++) cp_async16(b_base + soB[t], gB[t] + k0, true);
    };
    constexpr int KK = BK / 32;
    static_assert(KK >= 2, "BK >= 64");
    uint32_t oA[MT][KK], oB[NT / 2][KK];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int kk = 0; kk < KK; kk++) oA[i][kk] = swz<BK>(wm * WTM + i * 16 + (lane % 16), kk * 2 + (lane / 16));
#pragma unroll
    for (int j = 0; j < NT / 2; j++)
#pragma unroll
        for (int kk = 0; kk < KK; kk++)
            oB[j][kk] = swz<BK>(wn * WTN + j * 16 + (lane % 8) + ((lane / 16) * 8), kk * 2 + ((lane / 8) % 2));
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load(s, s);
        cp_commit();
    }
    uint32_t af[2][MT][4], bf[2][NT][2];
    auto frags = [&](uint32_t a_base, uint32_t b_base, int kk, int buf) {
#pragma unroll
        for (int i = 0; i < MT; i++) ldsm_x4(a_base + oA[i][kk], af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < NT / 2; j++)
            ldsm_x4(b_base + oB[j][kk], bf[buf][2 * j][0], bf[buf][2 * j][1], bf[buf][2 * j + 1][0], bf[buf][2 * j + 1][1]);
    };
    cp_wait<STAGES - 2>();
    __syncthreads();
    frags(sA0, sB0, 0, 0);
    auto tile = [&](int kt, auto last_tag) {
        constexpr bool LAST = decltype(last_tag)::value;
        const int st = kt % STAGES;
        const uint32_t a_base = sA0 + st * BM * BK, b_base = sB0 + st * BN * BK;
        if constexpr (!LAST) {
            int nk = kt + STAGES - 1;
            if (nk < KT) load(nk % STAGES, nk);
            cp_commit();
        }
#pragma unroll
        for (int kk = 0; kk < KK; kk++) {
            const int cur = kk & 1, nxt = cur ^ 1;
            if (kk + 1 < KK) {
                frags(a_base, b_base, kk + 1, nxt);
            } else if constexpr (!LAST) {
                const int st2 = (kt + 1) % STAGES;
                frags(sA0 + st2 * BM * BK, sB0 + st2 * BN * BK, 0, nxt);
            }
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < NT; j++) mma_bf16(acc[i][j], af[cur][i], bf[cur][j]);
            if constexpr (!LAST) {
                if (kk + 2 == KK) {
                    cp_wait<STAGES - 2>();
                    __syncthreads();
                }
            }
        }
    };
    for (int kt = 0; kt < KT - 1; kt++) tile(kt, BoolC<false>{});
    tile(KT - 1, BoolC<true>{});
    cp_wait<0>();
    const int g = lane / 4, t4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int m = m0 + wm * WTM + i * 16 + g + h * 8;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                int n = n0 + wn * WTN + j * 8 + t4 * 2;
                float v0 = acc[i][j][h * 2 + 0], v1 = acc[i][j][h * 2 + 1];
                __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m * N + n) = o;
            }
        }
    }
}
DEVI void raster(int pid, int npm, int npn, int gm, int& tm, int& tn) {
    int grp = pid / (gm * npn), fm = grp * gm, gs = min(npm - fm, gm);
    tm = fm + (pid % (gm * npn)) % gs;
    tn = (pid % (gm * npn)) / gs;
}
#define DEF(NAME, BM, BN, BK, WM, WN, ST, MINB)                                                                         \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB) NAME(const uint8_t* A, const uint8_t* B,          \
                                                                        __nv_bfloat16* C, int M, int N, int K, int gm) { \
        extern __shared__ __align__(128) uint8_t smem[];                                                              \
        int tm, tn;                                                                                                   \
        raster(blockIdx.x, (M + BM - 1) / BM, N / BN, gm, tm, tn);                                                    \
        gemm_body<BM, BN, BK, WM, WN, ST>(A, B, C, M, N, K, 0, tm, tn, smem);                                         \
    }
#define DEFHY(NAME, BM, BMS, BN, BK, WM, WN, WMS, ST, MINB)                                                             \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB) NAME(const uint8_t* A, const uint8_t* B,          \
                                                                        __nv_bfloat16* C, int M, int N, int K, int gm, \
                                                                        int MB) {                                      \
        extern __shared__ __align__(128) uint8_t smem[];                                                              \
        const int npn = N / BN, nbig = (MB / BM) * npn;                                                               \
        int tm, tn;                                                                                                   \
        if ((int)blockIdx.x < nbig) {                                                                                 \
            raster(blockIdx.x, MB / BM, npn, gm, tm, tn);                                                             \
            gemm_body<BM, BN, BK, WM, WN, ST>(A, B, C, MB, N, K, 0, tm, tn, smem);                                    \
        } else {                                                                                                      \
            static_assert(WMS * WN == WM * WN, "same thread count");                                                  \
            raster(blockIdx.x - nbig, (M - MB + BMS - 1) / BMS, npn, gm, tm, tn);                                     \
            gemm_body<BMS, BN, BK, WMS, WN, ST>(A, B, C, M, N, K, MB, tm, tn, smem);                                  \
        }                                                                                                             \
    }
DEF(bg_128x128x128_w2x2_s3, 128, 128, 128, 2, 2, 3, 1)
DEF(bg_128x128x128_w2x4_s3, 128, 128, 128, 2, 4, 3, 1)
DEF(bg_128x128x128_w4x2_s3, 128, 128, 128, 4, 2, 3, 1)
DEF(bg_128x256x128_w2x4_s2, 128, 256, 128, 2, 4, 2, 1)
DEF(bg_256x128x128_w4x2_s2, 256, 128, 128, 4, 2, 2, 1)
DEF(bg_64x128x128_w1x4_s4, 64, 128, 128, 1, 4, 4, 1)
DEF(bg_64x256x128_w1x4_s2, 64, 256, 128, 1, 4, 2, 1)
DEF(bg_64x128x128_w2x2_s4, 64, 128, 128, 2, 2, 4, 2)
DEF(bg_128x64x128_w2x2_s4, 128, 64, 128, 2, 2, 4, 2)
DEF(bg_128x128x64_w2x4_s5, 128, 128, 64, 2, 4, 5, 1)
DEF(bg_128x128x128_w2x2_s3_b2, 128, 128, 128, 2, 2, 3, 2)
DEFHY(hy_128_32x128x128_w2x2, 128, 32, 128, 128, 2, 2, 2, 3, 1)
DEFHY(hy_128_64x128x128_w2x2, 128, 64, 128, 128, 2, 2, 2, 3, 1)
DEFHY(hy_128_32x256x128_w2x4, 128, 32, 256, 128, 2, 4, 2, 2, 1)
DEFHY(hy_256_64x128x128_w4x2, 256, 64, 128, 128, 4, 2, 4, 2, 1)
DEFHY(hy_128_32x128x128_w2x4, 128, 32, 128, 128, 2, 4, 2, 3, 1)
DEF(bg_128x128x64_w2x2_s3_b2, 128, 128, 64, 2, 2, 3, 2)
DEF(bg_64x128x128_w2x2_s2_b2, 64, 128, 128, 2, 2, 2, 2)
DEF(bg_64x128x64_w2x2_s4_b2, 64, 128, 64, 2, 2, 4, 2)
DEF(bg_128x64x128_w2x2_s2_b2, 128, 64, 128, 2, 2, 2, 2)
DEF(bg_128x64x64_w2x2_s4_b2, 128, 64, 64, 2, 2, 4, 2)
DEFHY(hy_128_32x128x64_w2x2_b2, 128, 32, 128, 64, 2, 2, 2, 3, 2)
DEFHY(hy_128_64x128x64_w2x2_b2, 128, 64, 128, 64, 2, 2, 2, 3, 2)
