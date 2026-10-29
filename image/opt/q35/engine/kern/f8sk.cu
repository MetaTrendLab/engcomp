#include "f8g6.cu"
DEVI int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DEVI void st_release(int* p, int v) { asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
template <class G>
DEVI void seg_prologue(G& g, int ST, int m0, int n0, int kb, int ke) {
    for (int s = 0; s < ST - 1; s++) {
        if (kb + s < ke) g.load(s, kb + s, m0, n0);
        cp_commit();
    }
}
template <int BM, int BN, int BK, int WM, int WN, int ST, bool L2PF>
DEVI void seg_main(Gemm<BM, BN, BK, WM, WN, ST, L2PF>& g, float (&acc)[Gemm<BM, BN, BK, WM, WN, ST, L2PF>::MT]
                   [Gemm<BM, BN, BK, WM, WN, ST, L2PF>::NT][4], int m0, int n0, int kb, int ke) {
    using G = Gemm<BM, BN, BK, WM, WN, ST, L2PF>;
    constexpr int MT = G::MT, NT = G::NT, KK = G::KK;
    uint32_t af[2][MT][4], bf[2][NT][2];
    auto frags = [&](uint32_t a_st, uint32_t b_st, int kk, int buf) {
#pragma unroll
        for (int i = 0; i < MT; i++)
            ldsm_x4(a_st + g.oA[kk] + i * 16 * BK, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < NT / 2; j++)
            ldsm_x4(b_st + g.oB[kk] + j * 16 * BK, bf[buf][2 * j][0], bf[buf][2 * j][1], bf[buf][2 * j + 1][0],
                    bf[buf][2 * j + 1][1]);
    };
    cp_wait<ST - 2>();
    __syncthreads();
    frags(g.sA0, g.sB0, 0, 0);
    auto tile = [&](int kt, auto last_tag) {
        constexpr bool LAST = decltype(last_tag)::value;
        const int st = (kt - kb) % ST;
        const uint32_t a_st = g.sA0 + st * BM * BK, b_st = g.sB0 + st * BN * BK;
        if constexpr (!LAST) {
            const int nk = kt + ST - 1;
            if (nk < ke) g.load((nk - kb) % ST, nk, m0, n0);
            cp_commit();
        }
#pragma unroll
        for (int kk = 0; kk < KK; kk++) {
            const int cur = kk & 1, nxt = cur ^ 1;
            if (kk + 1 < KK) {
                frags(a_st, b_st, kk + 1, nxt);
            } else if constexpr (!LAST) {
                const int st2 = (kt + 1 - kb) % ST;
                frags(g.sA0 + st2 * BM * BK, g.sB0 + st2 * BN * BK, 0, nxt);
            }
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < NT; j++) mma_e4m3(acc[i][j], af[cur][i], bf[cur][j]);
            if constexpr (!LAST) {
                if (kk + 2 == KK) {
                    cp_wait<ST - 2>();
                    __syncthreads();
                }
            }
        }
    };
    for (int kt = kb; kt < ke - 1; kt++) tile(kt, BoolC<false>{});
    tile(ke - 1, BoolC<true>{});
    cp_wait<0>();
}
template <int BM, int BN, int BK, int WM, int WN, int ST, int EPI>
__device__ void run_sk(const uint8_t* A, const uint8_t* B, __nv_bfloat16* C, const float* sa, const float* sb, int M,
                       int N, int K, int gm, float* amax, float* ws, int* flags) {
    using G = Gemm<BM, BN, BK, WM, WN, ST, EPI == 2>;
    constexpr int MT = G::MT, NT = G::NT, NACC = MT * NT * 4;
    extern __shared__ __align__(128) uint8_t smem[];
    G g(A, B, M, N, K, smem);
    const int npm = (M + BM - 1) / BM, npn = N / BN, T = npm * npn, KT = g.KT;
    const int P = gridDim.x, c = blockIdx.x;
    const int fw = T / P, rem = T % P;
    const int R = rem == 0 ? 0 : (fw == 0 ? T : P + rem), base = T - R, nd = (base - c + P - 1) / P;
    const long long W = (long long)R * KT;
    const long long r0 = W * c / P, r1 = W * (c + 1) / P;
    const bool sk = r1 > r0;
    const int t0 = sk ? (int)(r0 / KT) : 0, k0 = sk ? (int)(r0 % KT) : 0;
    const int t1 = sk ? (int)((r1 - 1) / KT) : 0, k1 = sk ? (int)((r1 - 1) % KT) + 1 : 0;
    float acc[MT][NT][4];
    const bool one = sk && t0 == t1;
    const bool pre = sk && (one ? k0 == 0 : k1 < KT), post = sk && (one ? k0 > 0 : k0 > 0);
    const int fa = k0 == 0 ? t0 : t0 + 1, fb = k1 == KT ? t1 : t1 - 1, nfull = (!sk || one) ? 0 : fb - fa + 1;
    const int nseg = pre + nd + nfull + post;
    if (nseg == 0) return;
    auto seg_of = [&](int i, int& t, int& kb, int& ke) {
        if (pre && i == 0) {
            t = base + t1;
            kb = one ? k0 : 0;
            ke = k1;
        } else if (post && i == nseg - 1) {
            t = base + t0;
            kb = k0;
            ke = one ? k1 : KT;
        } else if (i - pre < nd) {
            t = c + (i - pre) * P;
            kb = 0;
            ke = KT;
        } else {
            t = base + fa + (i - pre - nd);
            kb = 0;
            ke = KT;
        }
    };
    auto origin = [&](int t, int& m0, int& n0) {
        int tm, tn;
        tile_of(t, npm, npn, gm, tm, tn);
        m0 = tm * BM;
        n0 = tn * BN;
    };
    {
        int t, kb, ke, m0, n0;
        seg_of(0, t, kb, ke);
        origin(t, m0, n0);
        seg_prologue(g, ST, m0, n0, kb, ke);
    }
    for (int i = 0; i < nseg; i++) {
        int t, kb, ke, m0, n0;
        seg_of(i, t, kb, ke);
        origin(t, m0, n0);
        const bool first = kb == 0, last = ke == KT;
        const int c0 = first ? c : (int)((((long long)(t - base) * KT + 1) * P - 1) / W);
        float* slot = ws + (size_t)c0 * (BM * BN) + g.tid;
        float* a = &acc[0][0][0];
        if (first) {
#pragma unroll
            for (int q = 0; q < NACC; q++) a[q] = 0.f;
        } else {
            if (g.tid == 0) {
                while (ld_acquire(flags + c0) != c - c0) __nanosleep(32);
                if (last) flags[c0] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int q = 0; q < NACC; q++) a[q] = __ldcg(slot + q * G::THREADS);
        }
        seg_main(g, acc, m0, n0, kb, ke);
        if (i + 1 < nseg) {
            int t2, kb2, ke2, m2, n2;
            seg_of(i + 1, t2, kb2, ke2);
            origin(t2, m2, n2);
            __syncthreads();
            seg_prologue(g, ST, m2, n2, kb2, ke2);
        }
        if (last) {
            if constexpr (EPI == 4) g.epilogue_gu(acc, C, sa, sb, amax, m0, n0);
            else if constexpr (EPI == 3) g.template epilogue16<true>(acc, C, sa, sb, m0, n0);
            else g.epilogue16(acc, C, sa, sb, m0, n0);
        } else {
#pragma unroll
            for (int q = 0; q < NACC; q++) __stcg(slot + q * G::THREADS, a[q]);
            __threadfence();
            __syncthreads();
            if (g.tid == 0) st_release(flags + c0, c - c0 + 1);
        }
    }
}
#define DEFSK(NAME, BM, BN, BK, WM, WN, ST, MINB, EPI)                                                                 \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB)                                                   \
        NAME(const uint8_t* A, const uint8_t* B, __nv_bfloat16* C, const float* sa, const float* sb, int M, int N,    \
             int K, int gm, float* amax, float* ws, int* flags) {                                                      \
        run_sk<BM, BN, BK, WM, WN, ST, EPI>(A, B, C, sa, sb, M, N, K, gm, amax, ws, flags);                            \
    }
#define DEFS(BM, BN, BK, WM, WN, ST, MINB)                                                                             \
    DEFSK(s7_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, 1)                     \
    DEFSK(s8_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, 2)                     \
    DEFSK(s9_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, 3)                     \
    DEFSK(s7gu_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB, BM, BN, BK, WM, WN, ST, MINB, 4)
DEFS(128, 256, 128, 2, 4, 2, 1)
DEFS(256, 128, 128, 4, 2, 2, 1)
DEFS(256, 128, 64, 4, 2, 4, 1)
DEFS(128, 256, 64, 2, 4, 4, 1)
DEFS(128, 128, 64, 2, 2, 3, 2)
DEFS(128, 128, 128, 4, 2, 2, 1)
DEFS(192, 128, 128, 4, 2, 2, 1)
DEFS(224, 128, 128, 2, 4, 2, 1)
DEFS(160, 128, 128, 2, 4, 2, 1)
DEFS(112, 128, 64, 1, 4, 3, 2)
