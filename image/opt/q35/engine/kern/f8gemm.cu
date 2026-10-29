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
DEVI void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int BK>
DEVI uint32_t swz(int r, int c) {
    if constexpr (BK >= 128) return r * BK + ((c ^ (r & 7)) << 4);
    else return r * BK + ((c ^ ((r >> 1) & 3)) << 4);
}
DEVI float silu_tri(float g) {
    float t, e, d, r;
    asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(g), "f"(d));
    return r;
}
template <bool V>
struct BoolC { static constexpr bool value = V; };
DEVI float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool GU = false>
__device__ void gemm_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, __nv_bfloat16* __restrict__ C,
                          const float* __restrict__ sa, const float* __restrict__ sb, int M, int N, int K, int tm, int tn,
                          float* __restrict__ amax = nullptr) {
    constexpr int THREADS = WM * WN * 32;
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int MT = WTM / 16, NT = WTN / 8;
    constexpr int CH = BK / 16;
    static_assert(CH == 4 || CH >= 8, "swizzle needs BK 64 or >= 128");
    static_assert(NT % 2 == 0, "B fragments are loaded two n8 tiles at a time");
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* sA = smem;
    uint8_t* sB = smem + STAGES * BM * BK;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = tm * BM, n0 = tn * BN;
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
                for (int j = 0; j < NT; j++) mma_e4m3(acc[i][j], af[cur][i], bf[cur][j]);
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
    if constexpr (GU) {
        const int I = N / 2;
#pragma unroll
        for (int i = 0; i < MT; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * WTM + i * 16 + g + h * 8;
                bool ok = m < M;
                float xs = ok ? sa[m] : 0.f;
                float am = 0.f;
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    int n = n0 + wn * WTN + j * 8 + t4 * 2;
                    float gv = bfr(acc[i][j][h * 2 + 0]), uv = bfr(acc[i][j][h * 2 + 1]);
                    gv = __fmul_rn(__fmul_rn(gv, xs), sb[n]);
                    uv = __fmul_rn(__fmul_rn(uv, xs), sb[n + 1]);
                    float a = bfr(silu_tri(gv));
                    __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(a, bfr(uv)));
                    if (ok) C[(size_t)m * I + n / 2] = y;
                    am = fmaxf(am, fabsf(__bfloat162float(y)));
                }
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
                if (ok && t4 == 0) atomicMax(reinterpret_cast<int*>(amax) + m, __float_as_int(am));
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int m = m0 + wm * WTM + i * 16 + g + h * 8;
            if (m >= M) continue;
            float s_a = sa ? sa[m] : 1.f;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                int n = n0 + wn * WTN + j * 8 + t4 * 2;
                float v0 = acc[i][j][h * 2 + 0], v1 = acc[i][j][h * 2 + 1];
                if (sa) {
                    v0 = (v0 * s_a) * sb[n];
                    v1 = (v1 * s_a) * sb[n + 1];
                }
                __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m * N + n) = o;
            }
        }
    }
}
#define DEF(NAME, BM, BN, BK, WM, WN, ST) DEF2(NAME, BM, BN, BK, WM, WN, ST, 1)
#define DEF2(NAME, BM, BN, BK, WM, WN, ST, MINB)                                                                       \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB) NAME(const uint8_t* A, const uint8_t* B,                \
                                                                    __nv_bfloat16* C, const float* sa, const float* sb, \
                                                                    int M, int N, int K, int gm) {                      \
                                                  \
        int npm = (M + BM - 1) / BM, npn = N / BN;                                                                    \
        int pid = blockIdx.x, grp = pid / (gm * npn), fm = grp * gm;                                                   \
        int gs = min(npm - fm, gm);                                                                                     \
        int tm = fm + (pid % (gm * npn)) % gs, tn = (pid % (gm * npn)) / gs;                                           \
        gemm_body<BM, BN, BK, WM, WN, ST>(A, B, C, sa, sb, M, N, K, tm, tn);                                           \
    }
DEF(f8_128x128x128_w2x2_s3, 128, 128, 128, 2, 2, 3)
DEF(f8_128x128x128_w2x4_s3, 128, 128, 128, 2, 4, 3)
DEF(f8_128x256x128_w2x4_s2, 128, 256, 128, 2, 4, 2)
DEF(f8_256x128x128_w4x2_s2, 256, 128, 128, 4, 2, 2)
DEF(f8_64x256x128_w1x4_s2, 64, 256, 128, 1, 4, 2)
DEF(f8_128x128x128_w4x2_s3, 128, 128, 128, 4, 2, 3)
DEF(f8_128x256x64_w2x4_s4, 128, 256, 64, 2, 4, 4)
DEF(f8_256x128x64_w4x2_s4, 256, 128, 64, 4, 2, 4)
DEF(f8_128x256x64_w2x4_s3, 128, 256, 64, 2, 4, 3)
DEF(f8_128x128x64_w2x2_s4, 128, 128, 64, 2, 2, 4)
DEF(f8_128x128x64_w2x4_s5, 128, 128, 64, 2, 4, 5)
DEF2(f8_128x128x64_w2x2_s3_b2, 128, 128, 64, 2, 2, 3, 2)
DEF2(f8_128x128x64_w2x4_s3_b2, 128, 128, 64, 2, 4, 3, 2)
DEF2(f8_128x128x128_w2x4_s2_b2, 128, 128, 128, 2, 4, 2, 2)
#define DEFGU(NAME, BM, BN, BK, WM, WN, ST)                                                                            \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(const uint8_t* A, const uint8_t* B,                \
                                                                    __nv_bfloat16* Y, const float* xs, const float* ws, \
                                                                    int M, int N, int K, int gm, float* amax) {         \
        int npm = (M + BM - 1) / BM, npn = N / BN;                                                                    \
        int pid = blockIdx.x, grp = pid / (gm * npn), fm = grp * gm;                                                   \
        int gs = min(npm - fm, gm);                                                                                     \
        int tm = fm + (pid % (gm * npn)) % gs, tn = (pid % (gm * npn)) / gs;                                           \
        gemm_body<BM, BN, BK, WM, WN, ST, true>(A, B, Y, xs, ws, M, N, K, tm, tn, amax);                               \
    }
DEFGU(gu_256x128x64_w4x2_s4, 256, 128, 64, 4, 2, 4)
DEFGU(gu_128x256x128_w2x4_s2, 128, 256, 128, 2, 4, 2)
DEFGU(gu_256x128x128_w4x2_s2, 256, 128, 128, 4, 2, 2)
