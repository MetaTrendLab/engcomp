#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
template <bool L2PF = false>
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    if constexpr (L2PF)
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
    else
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
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
template <bool V>
struct BoolC { static constexpr bool value = V; };
DEVI float silu_tri(float g) {
    float t, e, d, r;
    asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(g), "f"(d));
    return r;
}
DEVI float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool L2PF = false>
struct Gemm {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int WTM = BM / WM, WTN = BN / WN;
    static constexpr int MT = WTM / 16, NT = WTN / 8;
    static constexpr int CH = BK / 16;
    static constexpr int KK = BK / 32;
    static constexpr int RSTEP = THREADS / CH;
    static constexpr int LA = (BM * CH + THREADS - 1) / THREADS, LB = (BN * CH + THREADS - 1) / THREADS;
    static constexpr bool A_REM = BM * CH % THREADS != 0, B_REM = BN * CH % THREADS != 0;
    static_assert(WTM % 16 == 0 && WTN % 16 == 0, "warp tile");
    static_assert(CH == 4 || CH == 8, "BK 64 or 128");
    static_assert(RSTEP % 8 == 0, "copy slots keep the swizzle phase");
    static_assert(KK >= 2, "BK >= 64");
    static constexpr int SMEM = STAGES * (BM + BN) * BK;
    const uint8_t* A;
    const uint8_t* B;
    int M, N, K, KT;
    uint32_t sA0, sB0;
    int tid, lane, wm, wn;
    int cr, cc;
    uint32_t so0;
    uint32_t oA[KK], oB[KK];
    DEVI Gemm(const uint8_t* A_, const uint8_t* B_, int M_, int N_, int K_, uint8_t* smem)
        : A(A_), B(B_), M(M_), N(N_), K(K_) {
        KT = K / BK;
        sA0 = smem_u32(smem);
        sB0 = sA0 + STAGES * BM * BK;
        tid = threadIdx.x;
        lane = tid % 32;
        int warp = tid / 32;
        wm = warp / WN;
        wn = warp % WN;
        cr = tid / CH;
        cc = tid % CH;
        so0 = swz<BK>(cr, cc);
#pragma unroll
        for (int kk = 0; kk < KK; kk++) {
            oA[kk] = swz<BK>(wm * WTM + (lane % 16), kk * 2 + (lane / 16));
            oB[kk] = swz<BK>(wn * WTN + (lane % 8) + ((lane / 16) * 8), kk * 2 + ((lane / 8) % 2));
        }
    }
    DEVI void load(int st, int kt, int m0, int n0) {
        const int k0 = kt * BK + cc * 16;
        const uint32_t a_base = sA0 + st * BM * BK + so0, b_base = sB0 + st * BN * BK + so0;
#pragma unroll
        for (int t = 0; t < LA; t++) {
            const int r = cr + t * RSTEP;
            if (!A_REM || r < BM) {
                const int gm = m0 + r;
                const bool ok = gm < M;
                const uint8_t* src = A + (uint32_t)((ok ? gm : 0) * K + k0);
                cp_async16<L2PF>(a_base + t * RSTEP * BK, src, ok);
            }
        }
#pragma unroll
        for (int t = 0; t < LB; t++) {
            const int r = cr + t * RSTEP;
            if (!B_REM || r < BN) cp_async16<L2PF>(b_base + t * RSTEP * BK, B + (uint32_t)((n0 + r) * K + k0), true);
        }
    }
    DEVI void prologue(int m0, int n0) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < KT) load(s, s, m0, n0);
            cp_commit();
        }
    }
    DEVI void mainloop(float (&acc)[MT][NT][4], int m0, int n0, bool has_next, int nm0, int nn0) {
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
        uint32_t af[2][MT][4], bf[2][NT][2];
        auto frags = [&](uint32_t a_st, uint32_t b_st, int kk, int buf) {
#pragma unroll
            for (int i = 0; i < MT; i++)
                ldsm_x4(a_st + oA[kk] + i * 16 * BK, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
            for (int j = 0; j < NT / 2; j++)
                ldsm_x4(b_st + oB[kk] + j * 16 * BK, bf[buf][2 * j][0], bf[buf][2 * j][1], bf[buf][2 * j + 1][0],
                        bf[buf][2 * j + 1][1]);
        };
        cp_wait<STAGES - 2>();
        __syncthreads();
        frags(sA0, sB0, 0, 0);
        auto tile = [&](int kt, auto last_tag) {
            constexpr bool LAST = decltype(last_tag)::value;
            const int st = kt % STAGES;
            const uint32_t a_st = sA0 + st * BM * BK, b_st = sB0 + st * BN * BK;
            if constexpr (!LAST) {
                const int nk = kt + STAGES - 1;
                if (nk < KT) load(nk % STAGES, nk, m0, n0);
                cp_commit();
            }
#pragma unroll
            for (int kk = 0; kk < KK; kk++) {
                const int cur = kk & 1, nxt = cur ^ 1;
                if (kk + 1 < KK) {
                    frags(a_st, b_st, kk + 1, nxt);
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
        if (has_next) {
            __syncthreads();
            prologue(nm0, nn0);
        }
    }
    DEVI void epilogue(const float (&acc)[MT][NT][4], __nv_bfloat16* __restrict__ C, const float* __restrict__ sa,
                       const float* __restrict__ sb, int m0, int n0) {
        const int g = lane / 4, t4 = lane % 4;
#pragma unroll
        for (int i = 0; i < MT; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * WTM + i * 16 + g + h * 8;
                if (m >= M) continue;
                const float s_a = sa ? sa[m] : 1.f;
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const int n = n0 + wn * WTN + j * 8 + t4 * 2;
                    float v0 = acc[i][j][h * 2 + 0], v1 = acc[i][j][h * 2 + 1];
                    if (sa) {
                        v0 = (v0 * s_a) * sb[n];
                        v1 = (v1 * s_a) * sb[n + 1];
                    }
                    *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m * N + n) = __floats2bfloat162_rn(v0, v1);
                }
            }
        }
    }
    template <bool BFIRST = false>
    DEVI void epilogue16(const float (&acc)[MT][NT][4], __nv_bfloat16* __restrict__ C, const float* __restrict__ sa,
                         const float* __restrict__ sb, int m0, int n0) {
        static_assert(NT % 4 == 0, "quad transpose needs 4 n8 tiles");
        const int g = lane / 4, t4 = lane % 4;
        const int nw = n0 + wn * WTN;
        float sbv[NT][2], sav[MT][2];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n = nw + j * 8 + t4 * 2;
            sbv[j][0] = sa ? sb[n] : 1.f;
            sbv[j][1] = sa ? sb[n + 1] : 1.f;
        }
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * WTM + i * 16 + g + h * 8;
                sav[i][h] = (sa && m < M) ? sa[m] : 1.f;
            }
        const bool lo2 = t4 < 2, even = (t4 & 1) == 0;
#pragma unroll
        for (int i = 0; i < MT; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * WTM + i * 16 + g + h * 8;
#pragma unroll
                for (int jb = 0; jb < NT; jb += 4) {
                    uint32_t w[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        float v0 = acc[i][jb + q][h * 2 + 0], v1 = acc[i][jb + q][h * 2 + 1];
                        if (sa) {
                            if constexpr (BFIRST) {
                                v0 = (v0 * sbv[jb + q][0]) * sav[i][h];
                                v1 = (v1 * sbv[jb + q][1]) * sav[i][h];
                            } else {
                                v0 = (v0 * sav[i][h]) * sbv[jb + q][0];
                                v1 = (v1 * sav[i][h]) * sbv[jb + q][1];
                            }
                        }
                        __nv_bfloat162 p = __floats2bfloat162_rn(v0, v1);
                        w[q] = *reinterpret_cast<uint32_t*>(&p);
                    }
                    uint32_t s0 = lo2 ? w[2] : w[0], s1 = lo2 ? w[3] : w[1];
                    uint32_t r0 = __shfl_xor_sync(0xffffffff, s0, 2), r1 = __shfl_xor_sync(0xffffffff, s1, 2);
                    uint32_t n0_ = lo2 ? w[0] : r0, n1 = lo2 ? w[1] : r1, n2 = lo2 ? r0 : w[2], n3 = lo2 ? r1 : w[3];
                    s0 = even ? n1 : n0_;
                    s1 = even ? n3 : n2;
                    r0 = __shfl_xor_sync(0xffffffff, s0, 1);
                    r1 = __shfl_xor_sync(0xffffffff, s1, 1);
                    uint4 o;
                    o.x = even ? n0_ : r0;
                    o.y = even ? r0 : n1;
                    o.z = even ? n2 : r1;
                    o.w = even ? r1 : n3;
                    if (m < M) *reinterpret_cast<uint4*>(C + (size_t)m * N + nw + (jb + t4) * 8) = o;
                }
            }
        }
    }
    DEVI void epilogue_gu(const float (&acc)[MT][NT][4], __nv_bfloat16* __restrict__ Y, const float* __restrict__ xs,
                          const float* __restrict__ ws, float* __restrict__ amax, int m0, int n0) {
        static_assert(NT % 4 == 0, "quad transpose needs 4 n8 tiles");
        const int g = lane / 4, t4 = lane % 4;
        const int I = N / 2, nw = n0 + wn * WTN;
        float wv[NT][2];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n = nw + j * 8 + t4 * 2;
            wv[j][0] = ws[n];
            wv[j][1] = ws[n + 1];
        }
        const bool lo2 = t4 < 2, even = (t4 & 1) == 0;
#pragma unroll
        for (int i = 0; i < MT; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * WTM + i * 16 + g + h * 8;
                const bool ok = m < M;
                const float x = ok ? xs[m] : 0.f;
                float am = 0.f;
#pragma unroll
                for (int jb = 0; jb < NT; jb += 4) {
                    uint32_t e[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        float gv = bfr(acc[i][jb + q][h * 2 + 0]), uv = bfr(acc[i][jb + q][h * 2 + 1]);
                        gv = __fmul_rn(__fmul_rn(gv, x), wv[jb + q][0]);
                        uv = __fmul_rn(__fmul_rn(uv, x), wv[jb + q][1]);
                        const float a = bfr(silu_tri(gv));
                        const __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(a, bfr(uv)));
                        am = fmaxf(am, fabsf(__bfloat162float(y)));
                        e[q] = (uint32_t)(*reinterpret_cast<const uint16_t*>(&y));
                    }
                    const uint32_t w0 = e[0] | (e[1] << 16), w1 = e[2] | (e[3] << 16);
                    const uint32_t r = __shfl_xor_sync(0xffffffff, lo2 ? w1 : w0, 2);
                    const uint32_t p0 = lo2 ? w0 : r, p1 = lo2 ? r : w1;
                    const uint32_t snd = even ? __byte_perm(p0, p1, 0x7632) : __byte_perm(p0, p1, 0x5410);
                    const uint32_t got = __shfl_xor_sync(0xffffffff, snd, 1);
                    const uint32_t own = even ? __byte_perm(p0, p1, 0x5410) : __byte_perm(p0, p1, 0x7632);
                    const uint32_t lo = even ? __byte_perm(own, got, 0x5410) : __byte_perm(got, own, 0x5410);
                    const uint32_t hi = even ? __byte_perm(own, got, 0x7632) : __byte_perm(got, own, 0x7632);
                    if (ok) *reinterpret_cast<uint2*>(Y + (size_t)m * I + nw / 2 + 4 * (jb + t4)) = make_uint2(lo, hi);
                }
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
                if (ok && t4 == 0) atomicMax(reinterpret_cast<int*>(amax) + m, __float_as_int(am));
            }
        }
    }
};
DEVI void tile_of(int pid, int npm, int npn, int gm, int& tm, int& tn) {
    const int grp = pid / (gm * npn), fm = grp * gm, gs = min(npm - fm, gm), r = pid % (gm * npn);
    tm = fm + r % gs;
    tn = r / gs;
}
template <int BM, int BN, int BK, int WM, int WN, int ST, bool PERSIST, int EPI = 0>
__device__ void run(const uint8_t* A, const uint8_t* B, __nv_bfloat16* C, const float* sa, const float* sb, int M, int N,
                    int K, int gm) {
    using G = Gemm<BM, BN, BK, WM, WN, ST, EPI == 2>;
    extern __shared__ __align__(128) uint8_t smem[];
    G g(A, B, M, N, K, smem);
    const int npm = (M + BM - 1) / BM, npn = N / BN, T = npm * npn;
    float acc[G::MT][G::NT][4];
    int pid = blockIdx.x;
    if (pid >= T) return;
    int tm, tn;
    tile_of(pid, npm, npn, gm, tm, tn);
    g.prologue(tm * BM, tn * BN);
    while (true) {
        const int nxt = PERSIST ? pid + (int)gridDim.x : T;
        const bool has_next = nxt < T;
        int ntm = 0, ntn = 0;
        if (has_next) tile_of(nxt, npm, npn, gm, ntm, ntn);
        g.mainloop(acc, tm * BM, tn * BN, has_next, ntm * BM, ntn * BN);
        if constexpr (EPI == 3) g.template epilogue16<true>(acc, C, sa, sb, tm * BM, tn * BN);
        else if constexpr (EPI >= 1) g.epilogue16(acc, C, sa, sb, tm * BM, tn * BN);
        else g.epilogue(acc, C, sa, sb, tm * BM, tn * BN);
        if (!has_next) break;
        pid = nxt;
        tm = ntm;
        tn = ntn;
    }
}
#define DEF6(NAME, BM, BN, BK, WM, WN, ST, MINB, PERSIST) DEFE(NAME, BM, BN, BK, WM, WN, ST, MINB, PERSIST, 0)
#define DEFE(NAME, BM, BN, BK, WM, WN, ST, MINB, PERSIST, EPI)                                                         \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB)                                                   \
        NAME(const uint8_t* A, const uint8_t* B, __nv_bfloat16* C, const float* sa, const float* sb, int M, int N,    \
             int K, int gm) {                                                                                          \
        run<BM, BN, BK, WM, WN, ST, PERSIST, EPI>(A, B, C, sa, sb, M, N, K, gm);                                       \
    }
#define DEF6P(BM, BN, BK, WM, WN, ST, MINB)                                                                            \
    DEF6(g6_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, false)                  \
    DEF6(g6p_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, true)
DEF6P(128, 256, 128, 2, 4, 2, 1)
DEF6P(256, 128, 128, 4, 2, 2, 1)
DEF6P(256, 128, 64, 4, 2, 4, 1)
DEF6P(128, 256, 64, 2, 4, 4, 1)
DEF6P(128, 128, 64, 2, 2, 3, 2)
DEF6P(224, 128, 64, 2, 4, 4, 1)
DEF6P(224, 128, 128, 2, 4, 2, 1)
DEF6P(192, 128, 64, 4, 2, 4, 1)
DEF6P(192, 128, 128, 4, 2, 2, 1)
DEF6P(160, 128, 64, 2, 4, 4, 1)
DEF6P(160, 128, 128, 2, 4, 2, 1)
DEF6P(112, 128, 64, 1, 4, 3, 2)
DEF6P(96, 128, 64, 2, 2, 3, 2)
DEF6P(96, 256, 64, 2, 4, 4, 1)
DEF6P(112, 256, 64, 1, 8, 4, 1)
DEF6P(112, 256, 128, 1, 8, 2, 1)
#define DEF7P(BM, BN, BK, WM, WN, ST, MINB)                                                                            \
    DEFE(g7_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, false, 1)               \
    DEFE(g7p_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, true, 1)
DEF7P(128, 256, 128, 2, 4, 2, 1)
DEF7P(256, 128, 128, 4, 2, 2, 1)
DEF7P(256, 128, 64, 4, 2, 4, 1)
DEF7P(128, 256, 64, 2, 4, 4, 1)
DEF7P(128, 128, 64, 2, 2, 3, 2)
DEF7P(224, 128, 64, 2, 4, 4, 1)
DEF7P(224, 128, 128, 2, 4, 2, 1)
DEF7P(192, 128, 64, 4, 2, 4, 1)
DEF7P(192, 128, 128, 4, 2, 2, 1)
DEF7P(160, 128, 128, 2, 4, 2, 1)
DEF7P(112, 128, 64, 1, 4, 3, 2)
DEF7P(96, 128, 64, 2, 2, 3, 2)
DEF7P(96, 256, 64, 2, 4, 4, 1)
DEF7P(112, 256, 128, 1, 8, 2, 1)
#define DEF8P(BM, BN, BK, WM, WN, ST, MINB)                                                                            \
    DEFE(g8_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, false, 2)               \
    DEFE(g8p_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, true, 2)
DEF8P(128, 256, 128, 2, 4, 2, 1)
DEF8P(256, 128, 64, 4, 2, 4, 1)
DEF8P(128, 128, 64, 2, 2, 3, 2)
DEF8P(224, 128, 128, 2, 4, 2, 1)
DEF8P(192, 128, 128, 4, 2, 2, 1)
DEF8P(112, 128, 64, 1, 4, 3, 2)
template <int BM, int BN, int BK, int WM, int WN, int ST>
__device__ void run_gu(const uint8_t* A, const uint8_t* B, __nv_bfloat16* Y, const float* xs, const float* ws, int M,
                       int N, int K, int gm, float* amax) {
    using G = Gemm<BM, BN, BK, WM, WN, ST>;
    extern __shared__ __align__(128) uint8_t smem[];
    G g(A, B, M, N, K, smem);
    const int npm = (M + BM - 1) / BM, npn = N / BN, T = npm * npn;
    float acc[G::MT][G::NT][4];
    const int pid = blockIdx.x;
    if (pid >= T) return;
    int tm, tn;
    tile_of(pid, npm, npn, gm, tm, tn);
    g.prologue(tm * BM, tn * BN);
    g.mainloop(acc, tm * BM, tn * BN, false, 0, 0);
    g.epilogue_gu(acc, Y, xs, ws, amax, tm * BM, tn * BN);
}
#define DEFGU7(BM, BN, BK, WM, WN, ST, MINB)                                                                           \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB)                                                   \
        g7gu_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB(const uint8_t* A, const uint8_t* B, __nv_bfloat16* Y, \
                                                                 const float* xs, const float* ws, int M, int N, int K, \
                                                                 int gm, float* amax) {                                \
        run_gu<BM, BN, BK, WM, WN, ST>(A, B, Y, xs, ws, M, N, K, gm, amax);                                            \
    }
DEFGU7(256, 128, 64, 4, 2, 4, 1)
DEFGU7(256, 128, 128, 4, 2, 2, 1)
DEFGU7(128, 256, 128, 2, 4, 2, 1)
DEFGU7(224, 128, 128, 2, 4, 2, 1)
DEFGU7(192, 128, 128, 4, 2, 2, 1)
DEFGU7(160, 128, 128, 2, 4, 2, 1)
DEFGU7(128, 128, 64, 2, 2, 3, 2)
DEFGU7(112, 128, 64, 1, 4, 3, 2)
#define DEF9P(BM, BN, BK, WM, WN, ST, MINB)                                                                            \
    DEFE(g9_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, false, 3)               \
    DEFE(g9p_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, true, 3)
DEF9P(128, 128, 128, 4, 2, 2, 1)
DEF9P(128, 128, 64, 2, 2, 3, 2)
DEF9P(256, 128, 64, 4, 2, 4, 1)
DEF9P(256, 128, 128, 4, 2, 2, 1)
DEF9P(224, 128, 128, 2, 4, 2, 1)
DEF9P(192, 128, 128, 4, 2, 2, 1)
DEF9P(160, 128, 128, 2, 4, 2, 1)
DEF9P(112, 128, 64, 1, 4, 3, 2)
DEF9P(96, 128, 64, 2, 2, 3, 2)
DEF9P(64, 128, 128, 2, 2, 2, 2)
