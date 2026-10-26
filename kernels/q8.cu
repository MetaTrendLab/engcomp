// Q8 (rgh, native rewrite 10-24): W8A8 int8 projections for sm_120 (RTX PRO 6000).
//   out[r, n] = sa[r] * sw[n] * (Aq[r, :] . Bq[n, :]) + bias[n] + AH[r, :] . BH[n, :]
// Aq [M, K] / Bq [N, K] int8 (per-row activation scales sa, per-output-column weight scales sw, an fp32 bias from the
// calibrated channel centres), AH [M, KH] / BH [N, KH] bf16 = the protected (outlier) channels, exact bf16 products.
// Mainloop = NG9B's (ngemm9b.cu: cp.async ring, XOR-swizzled smem + ldmatrix, double-buffered fragments, one barrier
// per k-tile, serpentine mma order) with the int8 operands addressed as 2-byte units: a 64-unit k-tile = 128 int8
// columns, each 16-unit k-step one mma.sync m16n8k32 s8 (A / B / C fragments have the m16n8k16 bf16 register layout
// byte for byte, so the ldmatrix addressing and the accumulator map are unchanged). The int32 sums are exact; after
// the int8 loop they are dequantised in registers and the protected channels run the same mainloop as bf16 m16n8k16
// into those fp32 accumulators.
// Kernels (all: M rows, rows >= M never loaded / stored; N % BN == 0; K % (2 BK) == 0; KH % BK == 0):
//   q8h_<big>_<small>: bf16 out [M, N] (ldo), rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), the rest small
//   q8w_<big>_<small>: N = 2H stacked [Wg; Wu], out [M, H] = SwiGLU(g, u) (gate / up rows interleaved per warp column)
//   q8p_<big>_<small>: grouped split-K, blockIdx.y = group p over int8 columns [p G, (p + 1) G) of Aq / Bq (K = G):
//                      fp32 part[p][M][N] = sa[p M + r] * sw[p N + n] * acc + (p == 0 ? bias[n] : 0) (+ AH . BH, p 0)
//   q8r_sum: out[M, N] bf16 = sum_p part[p] (p ascending, fp32), N % 4 == 0
//   q8x_prep: activation prepare, grid (M, P): x[r, lo[c]] -> v = (x - mu[c]) * is[c]; group p of G columns: amax,
//             sa[p M + r] = amax / 127, xq[r, c] = rn(v * 127 / amax) (int8); hi: xh[r, j] = x[r, hi[j]] (group 0 CTAs)
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#define DEVI __device__ __forceinline__

namespace q8 {

using U = uint16_t;   // the 2-byte addressing unit of both operand kinds

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma_s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void mma_s8z(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {   // d = a . b (C = 0)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
DEVI void mma_bf(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <class T>
DEVI void mma(T* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, int>::value) mma_s8(d, a, b0, b1);
    else mma_bf(d, a, b0, b1);
}
DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
DEVI float silu_f32(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
DEVI __nv_bfloat16 swiglu1(__nv_bfloat16 g, __nv_bfloat16 u) {
    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));
    return __float2bfloat16_rn(s * __bfloat162float(u));
}

// ROT (rgh 10-24): randomised 128-point Hadamard rotation of a fine group, v -> H_128 (d o v) / sqrt(128) with d_i = -1
// where bit i of ROT_NEG is set: the same R = diag(d) H / sqrt(128) as the rotated weights of q8_gptq_rot.pt (the
// 1 / sqrt(128) is folded into the stored activation scale)
__device__ __constant__ uint32_t ROT_NEG[4] = {0x09bab9abu, 0xfc955699u, 0xfc9121c0u, 0xafc1071cu};
constexpr float ROT_RS = 0.08838834764831845f;
DEVI float rot_sign(int i, float v) { return ((ROT_NEG[i >> 5] >> (i & 31)) & 1u) ? -v : v; }
// butterflies over the index bits held in one thread (N consecutive elements) ...
template <int N>
DEVI void fwht_reg(float (&v)[N]) {
#pragma unroll
    for (int st = 1; st < N; st <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e)
            if ((e & st) == 0) {
                const float x = v[e], y = v[e + st];
                v[e] = x + y;
                v[e + st] = x - y;
            }
}
// ... then over the higher index bits held by lanes (lane masks 1 .. MMAX)
template <int N, int MMAX>
DEVI void fwht_lanes(float (&v)[N], int lane) {
#pragma unroll
    for (int m = 1; m <= MMAX; m <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? o - v[e] : v[e] + o;
        }
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

// EPI: 2 = plain (bf16 stmatrix epilogue or fp32 partials), 1 = SwiGLU (interleaved B rows); BK in 2-byte units
template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_, int EPI_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N, EPI = EPI_;
    static constexpr int CW = WARPS_M * WARPS_N;
    static constexpr int THREADS = CW * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SMEM_PIPE = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr int CPAD = 8;
    static constexpr int SMEM_EPI = BM * (BN + CPAD) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0 && STAGES >= 2, "tile");
};

// per-thread chunk loader (NG9B's): chunk q < NA = A rows r0 + q RS, q >= NA = B rows r0 + (q - NA) RS; offsets in units
template <class C>
struct Loader {
    static constexpr int RS = C::THREADS / C::CH;
    static constexpr int NA = (C::BM + RS - 1) / RS;
    static constexpr int NB = (C::BN + RS - 1) / RS;
    static constexpr int NQ = NA + NB;
    const U* pa;
    const U* pb;
    uint32_t offa[NA], offb[NB];
    uint32_t dst0;
    uint32_t amask;
    int r0;
    // K, lda, ldb in units; n1 >= 0: SwiGLU interleave (gate rows from n0, up rows from n1)
    DEVI void init(const U* A, const U* B, int m0, int n0, int n1, int M, int lda, int ldb) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A;
        pb = B;
        amask = 0;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int r = r0 + q * RS;
            const int gm = m0 + r;
            const bool ok = gm < M;
            offa[q] = (uint32_t)((ok ? gm : 0) * lda + c * 8);
            amask |= (ok ? 1u : 0u) << q;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int r = r0 + q * RS;
            int wr;
            if (C::EPI == 1 && n1 >= 0) {
                const int wn = r / C::WTN, h = (r % C::WTN) >= C::WTN / 2, cc = r % (C::WTN / 2);
                wr = (h ? n1 : n0) + wn * (C::WTN / 2) + cc;
            } else {
                wr = n0 + r;
            }
            offb[q] = (uint32_t)(wr * ldb + c * 8);
        }
    }
    DEVI void issue_all(uint32_t sA, uint32_t sB, int k0) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (q < NA) {
                if (C::BM % RS == 0 || r0 + q * RS < C::BM)
                    cp_async16(sA + dst0 + q * RS * C::BK * 2, pa + offa[q] + k0, (amask >> q) & 1);
            } else {
                const int qb = q - NA;
                if (C::BN % RS == 0 || r0 + qb * RS < C::BN)
                    cp_async16(sB + dst0 + qb * RS * C::BK * 2, pb + offb[qb] + k0, true);
            }
        }
    }
};

template <class C>
struct Frag {
    uint32_t a_off[C::KS], b_off[C::KS];
    uint32_t af[2][C::MT][4];
    uint32_t bf[2][C::NT][2];
    DEVI void init(int wm, int wn, int lane) {
        constexpr int BK = C::BK;
        const int a_r = wm * C::WTM + (lane % 16);
        const int a_f = swz_f<BK>(lane % 16);
        const int a_kc = lane / 16;
        const int b_r = wn * C::WTN + (lane / 16) * 8 + (lane % 8);
        const int b_f = swz_f<BK>((lane / 16) * 8 + (lane % 8));
        const int b_kc = (lane / 8) % 2;
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            a_off[ks] = (a_r * BK + (((ks * 2 + a_kc) ^ a_f) * 8)) * 2;
            b_off[ks] = (b_r * BK + (((ks * 2 + b_kc) ^ b_f) * 8)) * 2;
        }
    }
    DEVI void load(int buf, uint32_t sA, uint32_t sB, int ks) {
        const uint32_t a = sA + a_off[ks];
        const uint32_t b = sB + b_off[ks];
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
            ldsm_x4(a + i * 16 * C::BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < C::NT; j += 2)
            ldsm_x4(b + j * 8 * C::BK * 2, bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    }
};

// acc (+)= A[m0:m0+BM, :Ku] . B-tile[:, :Ku] (Ku units); T = int (int8 data, m16n8k32 s8) or float (bf16, m16n8k16)
template <class C, class T, bool ZERO>
DEVI void mainloop(const U* __restrict__ A, const U* __restrict__ B, int M, int Ku, int lda, int ldb, int m0, int n0,
                   int n1, U* smem, T (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
    if constexpr (ZERO) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = T(0);
    }
    Loader<C> ld;
    ld.init(A, B, m0, n0, n1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa(t), sb(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa(tt % S), sb(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

// the global column of accumulator (j, e) of this thread: plain tiles n0 + wn WTN + j 8 + 2 t4 + e; SwiGLU tiles
// (o0 = the tile's first output column, H = N / 2): gate (j < NT / 2) o0 + .., up H + o0 + ..
template <class C>
DEVI int col_of(int j, int e, int nbase, int H) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wn = warp % C::WN, t4 = lane % 4;
    if constexpr (C::EPI == 1) {
        constexpr int NH = C::NT / 2;
        const int jj = j < NH ? j : j - NH;
        return (j < NH ? 0 : H) + nbase + wn * (C::WTN / 2) + jj * 8 + t4 * 2 + e;
    } else {
        return nbase + wn * C::WTN + j * 8 + t4 * 2 + e;
    }
}
template <class C>
DEVI int row_of(int i, int h, int m0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN;
    return m0 + wm * C::WTM + i * 16 + lane / 4 + h * 8;
}

// int32 sums -> fp32: sa[r] sw[n] acc + bias[n] (bias may be null)
template <class C>
DEVI void dequant(const int (&ia)[C::MT][C::NT][4], float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                  const float* __restrict__ sw, const float* __restrict__ bias, int M, int m0, int nbase, int H) {
    float rs[C::MT][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            rs[i][h] = r < M ? sa[r] : 0.f;
        }
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
        const int c0 = col_of<C>(j, 0, nbase, H);
        const float2 w = *reinterpret_cast<const float2*>(sw + c0);
        const float2 b = bias ? *reinterpret_cast<const float2*>(bias + c0) : make_float2(0.f, 0.f);
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float s = rs[i][q >> 1] * ((q & 1) ? w.y : w.x);
                fa[i][j][q] = fmaf((float)ia[i][j][q], s, (q & 1) ? b.y : b.x);
            }
    }
}

// plain bf16 epilogue: stmatrix.x4 -> padded smem tile -> 16-byte stores of rows < M (NG9B's store_tile)
template <class C>
DEVI void store_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Cm, int M, int ldc, int m0, int n0,
                     U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    constexpr int LD = C::BN + C::CPAD;
    const int mi = lane / 8, rr = lane % 8;
    const uint32_t base = smem_u32(smem) +
        (uint32_t)(((wm * C::WTM + (mi & 1) * 8 + rr) * LD + wn * C::WTN + (mi >> 1) * 8) * 2);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; j += 2) {
            const uint32_t a = base + (uint32_t)((i * 16 * LD + j * 8) * 2);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(a),
                         "r"(pack_bf2(v[i][j][0], v[i][j][1])), "r"(pack_bf2(v[i][j][2], v[i][j][3])),
                         "r"(pack_bf2(v[i][j + 1][0], v[i][j + 1][1])), "r"(pack_bf2(v[i][j + 1][2], v[i][j + 1][3]))
                         : "memory");
        }
    named_sync(1, C::THREADS);
    constexpr int CPR = C::BN / 8;
    const __nv_bfloat16* sm = reinterpret_cast<const __nv_bfloat16*>(smem);
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = i % CPR;
        int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Cm + (size_t)gm * ldc + n0 + c * 8) = *reinterpret_cast<const uint4*>(sm + r * LD + c * 8);
    }
}

// SwiGLU register epilogue (NG9B's store_swiglu): n-tile j < NT/2 = gate columns, j + NT/2 the matching up columns
template <class C>
DEVI void store_swiglu(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Out, int M, int ldo, int m0,
                       int o0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int NH = C::NT / 2;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
            if (r >= M) continue;
            __nv_bfloat16* row = Out + (size_t)r * ldo + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const __nv_bfloat162 gg = __floats2bfloat162_rn(v[i][j][2 * h], v[i][j][2 * h + 1]);
                const __nv_bfloat162 uu = __floats2bfloat162_rn(v[i][j + NH][2 * h], v[i][j + NH][2 * h + 1]);
                __nv_bfloat162 o;
                o.x = swiglu1(gg.x, uu.x);
                o.y = swiglu1(gg.y, uu.y);
                *reinterpret_cast<__nv_bfloat162*>(row + j * 8) = o;
            }
        }
}

// fp32 partial epilogue: float2 stores of rows < M
template <class C>
DEVI void store_part(const float (&v)[C::MT][C::NT][4], float* __restrict__ P, int M, int N, int m0, int n0) {
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            if (r >= M) continue;
            float* row = P + (size_t)r * N;
#pragma unroll
            for (int j = 0; j < C::NT; ++j)
                *reinterpret_cast<float2*>(row + col_of<C>(j, 0, n0, 0)) = make_float2(v[i][j][2 * h], v[i][j][2 * h + 1]);
        }
}

struct Args {
    const int8_t* A;        // [M, lda] int8
    const int8_t* B;        // [N, ldb] int8
    const __nv_bfloat16* AH;   // [M, KH] bf16 (KH = 0: none)
    const __nv_bfloat16* BH;   // [N, KH]
    const float* sa;        // [P][M]
    const float* sw;        // [P][N]
    const float* bias;      // [N] or null
    void* out;              // bf16 [M, ldo] (q8h / q8w) or fp32 [P][M][N] (q8p)
    int M, N, K, KH, lda, ldb, ldo, mb;
};

// one tile: EPI 2 + PART false = bf16 plain, EPI 1 = SwiGLU, PART = fp32 partial of group blockIdx.y
template <class C, bool PART>
DEVI void tile(const Args& a, int m0, int nt, U* smem) {
    const int p = PART ? blockIdx.y : 0;
    const int H = C::EPI == 1 ? a.N / 2 : 0;
    const int nbase = C::EPI == 1 ? nt * (C::BN / 2) : nt * C::BN;
    const int n1 = C::EPI == 1 ? H + nbase : -1;
    int ia[C::MT][C::NT][4];
    const U* A = reinterpret_cast<const U*>(a.A + (size_t)p * a.K);
    const U* B = reinterpret_cast<const U*>(a.B + (size_t)p * a.K);
    mainloop<C, int, true>(A, B, a.M, a.K / 2, a.lda / 2, a.ldb / 2, m0, nbase, n1, smem, ia);
    float fa[C::MT][C::NT][4];
    dequant<C>(ia, fa, a.sa + (size_t)p * a.M, a.sw + (size_t)p * a.N, p == 0 ? a.bias : nullptr, a.M, m0, nbase, H);
    if (a.KH > 0 && p == 0)
        mainloop<C, float, false>(reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.KH, a.KH,
                                  a.KH, m0, nbase, n1, smem, fa);
    if constexpr (PART) {
        store_part<C>(fa, reinterpret_cast<float*>(a.out) + (size_t)p * a.M * a.N, a.M, a.N, m0, nbase);
    } else if constexpr (C::EPI == 1) {
        store_swiglu<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase);
    } else {
        store_tile<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase, smem);
    }
}

// rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), rows [mb, M) small tiles
template <class CB, class CS, bool PART>
DEVI void hetero(const Args& a, U* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int ncols = CB::EPI == 1 ? a.N / 2 : a.N;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (ncols / (CB::EPI == 1 ? CB::BN / 2 : CB::BN));
    int b = blockIdx.x;
    if (b < tb) {
        tile<CB, PART>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile<CS, PART>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// FINE (DeepSeek-style fine-grained scaling): the int8 k-tile of 128 columns = one quantisation group g with its own
// per-row activation scale sa[g M + r] and per-column weight scale sw[g N + n] (wfull: sw[n] for every group); the
// int32 sums of each k-tile are dequantised into the fp32 accumulators at its last k-step:
// fa += (sa sw) * acc_kt (scales loaded at the tile's first k-step, used three k-steps later)
template <class C>
DEVI void mainloop_fine(const U* __restrict__ A, const U* __restrict__ B, int M, int N, int Ku, int lda, int ldb, int m0,
                        int n0, U* smem, float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                        const float* __restrict__ sw, int wfull) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    static_assert(BK == 64, "one k-tile = one 128-column group");
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa_ = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb_ = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) fa[i][j][q] = 0.f;
    const int r0 = row_of<C>(0, 0, m0);    // rows r0 + 16 i + 8 h
    const int c0 = col_of<C>(0, 0, n0, 0);  // columns c0 + 8 j (+ 1)
    float wa[NT][2];
    if (wfull) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float2 w = *reinterpret_cast<const float2*>(sw + c0 + 8 * j);
            wa[j][0] = w.x;
            wa[j][1] = w.y;
        }
    }
    Loader<C> ld;
    ld.init(A, B, m0, n0, -1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa_(t), sb_(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa_(0), sb_(0), 0);
    int s = 0;
    int ia[MT][NT][4];
    float ra[MT][2];
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == 0) {   // this group's scales (consumed at the last k-step)
                const float* sr = sa + (size_t)kt * M;
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = r0 + 16 * i + 8 * h;
                        ra[i][h] = r < M ? sr[r] : 0.f;
                    }
                if (!wfull) {
                    const float* sc = sw + (size_t)kt * N + c0;
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const float2 w = *reinterpret_cast<const float2*>(sc + 8 * j);
                        wa[j][0] = w.x;
                        wa[j][1] = w.y;
                    }
                }
            }
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa_(tt % S), sb_(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa_(ns), sb_(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    if (ks == 0) mma_s8z(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                    else mma_s8(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
            if (ks == KS - 1) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const float w0 = wa[j][0], w1 = wa[j][1];
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            fa[i][j][q] = fmaf((float)ia[i][j][q], ra[i][q >> 1] * ((q & 1) ? w1 : w0), fa[i][j][q]);
                }
            }
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

struct ArgsF {
    const int8_t* A;     // [M, lda] int8 (all groups)
    const int8_t* B;     // [N, ldb] int8
    const float* sa;     // [G][M]
    const float* sw;     // [G][N] (wfull: [N])
    const float* bias;   // [N] or null (added by split 0)
    float* out;          // fp32 partials [S][M][N]
    int M, N, Ks, lda, ldb, mb, wfull, pad;   // Ks = int8 columns per split (a multiple of 128)
};

// one fine split-K tile: split blockIdx.y covers int8 columns [y Ks, (y + 1) Ks) = groups y Ks / 128 ..
template <class C>
DEVI void tile_f(const ArgsF& a, int m0, int nt, U* smem) {
    const int y = blockIdx.y;
    const int n0 = nt * C::BN;
    const int g0 = y * (a.Ks / 128);
    float fa[C::MT][C::NT][4];
    mainloop_fine<C>(reinterpret_cast<const U*>(a.A + (size_t)y * a.Ks), reinterpret_cast<const U*>(a.B + (size_t)y * a.Ks),
                     a.M, a.N, a.Ks / 2, a.lda / 2, a.ldb / 2, m0, n0, smem, fa, a.sa + (size_t)g0 * a.M,
                     a.wfull ? a.sw : a.sw + (size_t)g0 * a.N, a.wfull);
    if (y == 0 && a.bias != nullptr) {
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            const float2 b = *reinterpret_cast<const float2*>(a.bias + col_of<C>(j, 0, n0, 0));
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) fa[i][j][q] += (q & 1) ? b.y : b.x;
        }
    }
    store_part<C>(fa, a.out + (size_t)y * a.M * a.N, a.M, a.N, m0, n0);
}
template <class CB, class CS>
DEVI void hetero_f(const ArgsF& a, U* smem) {
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tile_f<CB>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_f<CS>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// gate|up + SwiGLU with an int8 hidden output (the down projection's fine input): the tile's 128 output columns are
// one group g = o0 / 128: v = (swiglu - mu[c]) * is[c] (null: identity), row amax over the 128 columns (quad shuffles,
// then the 4 warp columns through smem), hq[r, c] = rn(v 127 / amax), hs[g M + r] = amax / 127
struct ArgsG {
    const int8_t* A;
    const int8_t* B;
    const __nv_bfloat16* AH;
    const __nv_bfloat16* BH;
    const float* sa;      // [M]
    const float* sw;      // [N]
    const float* bias;    // [N] or null
    int8_t* hq;           // [M, N / 2]
    float* hs;            // [(N / 2) / 128][M]
    const float* mu;      // [N / 2] or null
    const float* is;      // [N / 2] or null
    int M, N, K, KH, lda, ldb, mb, pad;
};
// ROT epilogue: the tile's 128 SwiGLU outputs per row (centred) go through smem; one warp per row rotates them (4
// columns per lane), takes the row amax and writes the int8 hidden (4 bytes per lane) and hs = amax / 127 / sqrt(128)
template <class C>
DEVI void epi_g_rot(const ArgsG& a, const float (&fa)[C::MT][C::NT][4], int m0, int o0, int H, U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2, LDS = 132;
    float* S = reinterpret_cast<float*>(smem);   // [BM][LDS] (smem free after the loops)
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x = __bfloat162float(swiglu1(g, u));
                    const int cl = wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[o0 + cl]) * a.is[o0 + cl];
                    S[rl * LDS + cl] = x;
                }
        }
    __syncthreads();
    const int g = o0 / 128;
    for (int rl = warp; rl < C::BM; rl += C::CW) {
        const float4 y4 = *reinterpret_cast<const float4*>(S + rl * LDS + lane * 4);
        float y[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = rot_sign(lane * 4 + e, y[e]);
        fwht_reg<4>(y);
        fwht_lanes<4, 16>(y, lane);
        float m = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const int r = m0 + rl;
        if (r >= a.M) continue;
        const float inv = 127.f / m;
        if (lane == 0) a.hs[(size_t)g * a.M + r] = m / 127.f * ROT_RS;
        uint32_t q4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) q4 |= (uint32_t)(__float2int_rn(y[e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint32_t*>(a.hq + (size_t)r * H + o0 + lane * 4) = q4;
    }
}
template <class C, bool ROT>
DEVI void tile_g(const ArgsG& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    int ia[C::MT][C::NT][4];
    mainloop<C, int, true>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B), a.M, a.K / 2, a.lda / 2,
                           a.ldb / 2, m0, o0, H + o0, smem, ia);
    float fa[C::MT][C::NT][4];
    dequant<C>(ia, fa, a.sa, a.sw, a.bias, a.M, m0, o0, H);
    if (a.KH > 0)
        mainloop<C, float, false>(reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.KH, a.KH,
                                  a.KH, m0, o0, H + o0, smem, fa);
    if constexpr (ROT) {
        epi_g_rot<C>(a, fa, m0, o0, H, smem);
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* red = reinterpret_cast<float*>(smem);   // [BM][WN] row maxima of the warp columns (smem free after the loops)
    float v[C::MT][2][NH][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x = __bfloat162float(swiglu1(g, u));
                    const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[c]) * a.is[c];
                    v[i][h][j][e] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (t4 == 0) red[(wm * C::WTM + i * 16 + lane / 4 + h * 8) * C::WN + wn] = m;
        }
    __syncthreads();
    const int g = o0 / 128;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int r = m0 + rl;
            float m = 1e-12f;
#pragma unroll
            for (int w = 0; w < C::WN; ++w) m = fmaxf(m, red[rl * C::WN + w]);
            const float inv = 127.f / m;
            if (r >= a.M) continue;
            if (wn == 0 && t4 == 0) a.hs[(size_t)g * a.M + r] = m / 127.f;
            int8_t* row = a.hq + (size_t)r * H + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const uint32_t q0 = (uint32_t)(__float2int_rn(v[i][h][j][0] * inv) & 0xff);
                const uint32_t q1 = (uint32_t)(__float2int_rn(v[i][h][j][1] * inv) & 0xff);
                *reinterpret_cast<uint16_t*>(row + j * 8) = (uint16_t)(q0 | (q1 << 8));
            }
        }
}
template <class CB, class CS, bool ROT>
DEVI void hetero_g(const ArgsG& a, U* smem) {
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_g<CB, ROT>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_g<CS, ROT>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

}  // namespace q8

#define Q8F_KERNEL(NAME, CB, CS)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsF a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::hetero_f<CB, CS>(a, reinterpret_cast<q8::U*>(smem_raw));                                               \
    }
#define Q8G_KERNEL(NAME, CB, CS)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsG a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::hetero_g<CB, CS, false>(a, reinterpret_cast<q8::U*>(smem_raw));                                        \
    }
#define Q8GR_KERNEL(NAME, CB, CS)                                                                                  \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsG a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::hetero_g<CB, CS, true>(a, reinterpret_cast<q8::U*>(smem_raw));                                         \
    }

#define Q8_KERNEL(NAME, CB, CS, PART)                                                                              \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::Args a) {                                \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::hetero<CB, CS, PART>(a, reinterpret_cast<q8::U*>(smem_raw));                                           \
    }

// (BM, BN, BK units, warps M, warps N, stages, epilogue)
using P_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 2>;
using P_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 2>;
using P_S32x256 = q8::Cfg<32, 256, 64, 1, 8, 2, 2>;
using P_B128x128 = q8::Cfg<128, 128, 64, 2, 4, 3, 2>;
using P_S64x128 = q8::Cfg<64, 128, 64, 2, 4, 3, 2>;
using P_S32x128 = q8::Cfg<32, 128, 64, 1, 8, 3, 2>;
using W_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 1>;
using W_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 1>;
using W_S32x256 = q8::Cfg<32, 256, 64, 1, 8, 2, 1>;
using W_B128x128 = q8::Cfg<128, 128, 64, 2, 4, 3, 1>;
using W_S64x128 = q8::Cfg<64, 128, 64, 2, 4, 3, 1>;
using W_S32x128 = q8::Cfg<32, 128, 64, 1, 8, 3, 1>;

Q8_KERNEL(q8h_b128x256_s64x256, P_B128x256, P_S64x256, false)
Q8_KERNEL(q8h_b128x256_s32x256, P_B128x256, P_S32x256, false)
Q8_KERNEL(q8h_b128x128_s64x128, P_B128x128, P_S64x128, false)
Q8_KERNEL(q8h_b128x128_s32x128, P_B128x128, P_S32x128, false)
Q8_KERNEL(q8w_b128x256_s64x256, W_B128x256, W_S64x256, false)
Q8_KERNEL(q8w_b128x256_s32x256, W_B128x256, W_S32x256, false)
Q8_KERNEL(q8w_b128x128_s64x128, W_B128x128, W_S64x128, false)
Q8_KERNEL(q8w_b128x128_s32x128, W_B128x128, W_S32x128, false)
Q8_KERNEL(q8p_b128x256_s64x256, P_B128x256, P_S64x256, true)
Q8_KERNEL(q8p_b128x256_s32x256, P_B128x256, P_S32x256, true)
Q8_KERNEL(q8p_b128x128_s64x128, P_B128x128, P_S64x128, true)
Q8_KERNEL(q8p_b128x128_s32x128, P_B128x128, P_S32x128, true)
Q8F_KERNEL(q8f_b128x128_s64x128, P_B128x128, P_S64x128)
Q8F_KERNEL(q8f_b128x128_s32x128, P_B128x128, P_S32x128)
Q8G_KERNEL(q8g_b128x256_s64x256, W_B128x256, W_S64x256)
Q8G_KERNEL(q8g_b128x256_s32x256, W_B128x256, W_S32x256)
Q8GR_KERNEL(q8gr_b128x256_s64x256, W_B128x256, W_S64x256)
Q8GR_KERNEL(q8gr_b128x256_s32x256, W_B128x256, W_S32x256)

// out[M, N] bf16 = sum_p part[p][M][N] (fp32, p ascending); 4 elements per thread
extern "C" __global__ void q8r_sum(const float* __restrict__ part, __nv_bfloat16* __restrict__ out, int MN, int P) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= MN) return;
    float4 s = *reinterpret_cast<const float4*>(part + i);
    for (int p = 1; p < P; ++p) {
        const float4 t = *reinterpret_cast<const float4*>(part + (size_t)p * MN + i);
        s.x += t.x;
        s.y += t.y;
        s.z += t.z;
        s.w += t.w;
    }
    uint2 o;
    o.x = q8::pack_bf2(s.x, s.y);
    o.y = q8::pack_bf2(s.z, s.w);
    *reinterpret_cast<uint2*>(out + i) = o;
}

// activation prepare, grid (M, P), 128 threads: group p = int8 columns [p G, (p + 1) G) of xq (G % 512 == 0 not needed:
// 4 consecutive columns per thread step, G % 4 == 0); lo / mu / is indexed by the int8 column; hi gathered by p == 0
extern "C" __global__ void __launch_bounds__(128) q8x_prep(const __nv_bfloat16* __restrict__ x, int ldx,
                                                           const int* __restrict__ lo, const float* __restrict__ mu,
                                                           const float* __restrict__ is, int8_t* __restrict__ xq,
                                                           int ldq, float* __restrict__ sa, int M, int G,
                                                           const int* __restrict__ hi, int KH,
                                                           __nv_bfloat16* __restrict__ xh) {
    const int r = blockIdx.x, p = blockIdx.y;
    const __nv_bfloat16* xr = x + (size_t)r * ldx;
    const int c0 = p * G;
    __shared__ float red[4];
    float amax = 0.f;
    for (int c = c0 + threadIdx.x * 4; c < c0 + G; c += 512) {
        const int4 ix = *reinterpret_cast<const int4*>(lo + c);
        const float4 m = *reinterpret_cast<const float4*>(mu + c);
        const float4 s = *reinterpret_cast<const float4*>(is + c);
        amax = fmaxf(amax, fabsf((__bfloat162float(xr[ix.x]) - m.x) * s.x));
        amax = fmaxf(amax, fabsf((__bfloat162float(xr[ix.y]) - m.y) * s.y));
        amax = fmaxf(amax, fabsf((__bfloat162float(xr[ix.z]) - m.z) * s.z));
        amax = fmaxf(amax, fabsf((__bfloat162float(xr[ix.w]) - m.w) * s.w));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    amax = fmaxf(amax, 1e-12f);
    const float inv = 127.f / amax;
    if (threadIdx.x == 0) sa[(size_t)p * M + r] = amax / 127.f;
    int8_t* qr = xq + (size_t)r * ldq;
    for (int c = c0 + threadIdx.x * 4; c < c0 + G; c += 512) {
        const int4 ix = *reinterpret_cast<const int4*>(lo + c);
        const float4 m = *reinterpret_cast<const float4*>(mu + c);
        const float4 s = *reinterpret_cast<const float4*>(is + c);
        const int q0 = __float2int_rn((__bfloat162float(xr[ix.x]) - m.x) * s.x * inv);
        const int q1 = __float2int_rn((__bfloat162float(xr[ix.y]) - m.y) * s.y * inv);
        const int q2 = __float2int_rn((__bfloat162float(xr[ix.z]) - m.z) * s.z * inv);
        const int q3 = __float2int_rn((__bfloat162float(xr[ix.w]) - m.w) * s.w * inv);
        const uint32_t pk = (uint32_t)(q0 & 0xff) | ((uint32_t)(q1 & 0xff) << 8) | ((uint32_t)(q2 & 0xff) << 16) |
                            ((uint32_t)(q3 & 0xff) << 24);
        *reinterpret_cast<uint32_t*>(qr + c) = pk;
    }
    if (p == 0 && KH > 0) {
        __nv_bfloat16* hr = xh + (size_t)r * KH;
        for (int j = threadIdx.x; j < KH; j += 128) hr[j] = xr[hi[j]];
    }
}

// activation prepare for contiguous columns (lo = identity: the down / out_proj inputs), grid (M, P), 256 threads:
// group p = columns [p G, (p + 1) G) (G % 8 == 0), 16-byte loads; v = (x - mu) * is, per-(group, row) amax -> scale
extern "C" __global__ void __launch_bounds__(256) q8x_prepv(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const float* __restrict__ mu, const float* __restrict__ is,
                                                            int8_t* __restrict__ xq, int ldq, float* __restrict__ sa,
                                                            int M, int G) {
    const int r = blockIdx.x, p = blockIdx.y;
    const __nv_bfloat16* xr = x + (size_t)r * ldx;
    const int c0 = p * G;
    __shared__ float red[8];
    auto vals = [&](int c, float (&v)[8]) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + c);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        const float4 m0 = *reinterpret_cast<const float4*>(mu + c), m1 = *reinterpret_cast<const float4*>(mu + c + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(is + c), s1 = *reinterpret_cast<const float4*>(is + c + 4);
        const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const float ss[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = (__bfloat162float(b[e]) - mm[e]) * ss[e];
    };
    float amax = 0.f;
    for (int c = c0 + threadIdx.x * 8; c < c0 + G; c += 2048) {
        float v[8];
        vals(c, v);
#pragma unroll
        for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
    amax = fmaxf(amax, 1e-12f);
    const float inv = 127.f / amax;
    if (threadIdx.x == 0) sa[(size_t)p * M + r] = amax / 127.f;
    int8_t* qr = xq + (size_t)r * ldq;
    for (int c = c0 + threadIdx.x * 8; c < c0 + G; c += 2048) {
        float v[8];
        vals(c, v);
        uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(v[e] * inv) & 0xff) << (8 * e);
#pragma unroll
        for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(v[4 + e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint2*>(qr + c) = make_uint2(lo4, hi4);
    }
}

// fused add + RMSNorm (+ the next Q8 family's prepare), one CTA per row, 320 threads x 8 columns = 2560:
//   y = bf16(sum_p part[p][r]) (P >= 1 fp32 partials, p ascending) or the bf16 row y[r] (P == 0)
//   hn = bf16(h + y); x = bf16((hn * rsqrt(mean(hn^2) + eps)) * w1p)   (fused.add_rmsnorm's arithmetic, any sum order)
//   Kl > 0: xq[r, c] = rn((x[lo[c]] - mu[c]) * is[c] * 127 / amax), sa[r] = amax / 127, xh[r, j] = x[hi[j]] (KH > 0)
extern "C" __global__ void __launch_bounds__(320) q8n_addnorm(
    const float* __restrict__ part, int P, const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ w1p, float eps, __nv_bfloat16* __restrict__ hn, __nv_bfloat16* __restrict__ x, int M,
    int8_t* __restrict__ xq, float* __restrict__ sa, __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo,
    const float* __restrict__ mu, const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH) {
    constexpr int D = 2560;
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
        *reinterpret_cast<uint4*>(x + (size_t)r * D + c) = xo;
        *reinterpret_cast<uint4*>(xs + c) = xo;
    }
    if (Kl <= 0) return;
    __syncthreads();
    float q[8];
    float amax = 0.f;
    const bool act = c < Kl;
    if (act) {
        const int4 i0 = *reinterpret_cast<const int4*>(lo + c), i1 = *reinterpret_cast<const int4*>(lo + c + 4);
        const int ii[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
        const float4 m0 = *reinterpret_cast<const float4*>(mu + c), m1 = *reinterpret_cast<const float4*>(mu + c + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(is + c), s1 = *reinterpret_cast<const float4*>(is + c + 4);
        const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            q[e] = (__bfloat162float(xs[ii[e]]) - mm[e]) * sv[e];
            amax = fmaxf(amax, fabsf(q[e]));
        }
    }
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
    if (t == 0) sa[r] = amax / 127.f;
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

// fine prepare (the out_proj input), one CTA per row, 256 threads x 8 columns per step: group g = 128 columns (16
// threads); v = (x - mu) * is (null: identity), xq int8, sa[g M + r] = amax_g / 127 (K % 128 == 0)
extern "C" __global__ void __launch_bounds__(256) q8x_prepf(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const float* __restrict__ mu, const float* __restrict__ is,
                                                            int8_t* __restrict__ xq, float* __restrict__ sa, int M,
                                                            int K) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (size_t)r * ldx;
    int8_t* qr = xq + (size_t)r * K;
    for (int c = threadIdx.x * 8; c < K; c += 2048) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + c);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float v[8];
        float m = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = __bfloat162float(b[e]);
            if (mu != nullptr) v[e] = (v[e] - mu[c + e]) * is[c + e];
            m = fmaxf(m, fabsf(v[e]));
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const float inv = 127.f / m;
        if ((threadIdx.x & 15) == 0) sa[(size_t)(c / 128) * M + r] = m / 127.f;
        uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(v[e] * inv) & 0xff) << (8 * e);
#pragma unroll
        for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(v[4 + e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint2*>(qr + c) = make_uint2(lo4, hi4);
    }
}

// ROT: q8x_prepf with each 128-column group rotated (q8::rot_sign + fwht) before its amax / int8 rounding
extern "C" __global__ void __launch_bounds__(256) q8x_prepfr(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const float* __restrict__ mu, const float* __restrict__ is,
                                                            int8_t* __restrict__ xq, float* __restrict__ sa, int M,
                                                            int K) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (size_t)r * ldx;
    int8_t* qr = xq + (size_t)r * K;
    for (int c = threadIdx.x * 8; c < K; c += 2048) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + c);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float v[8];
        float m = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = __bfloat162float(b[e]);
            if (mu != nullptr) v[e] = (v[e] - mu[c + e]) * is[c + e];
        }
        const int i0 = (threadIdx.x & 15) * 8;   // the columns' index in their 128-column group
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = q8::rot_sign(i0 + e, v[e]);
        q8::fwht_reg<8>(v);
        q8::fwht_lanes<8, 8>(v, threadIdx.x & 31);
#pragma unroll
        for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(v[e]));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const float inv = 127.f / m;
        if ((threadIdx.x & 15) == 0) sa[(size_t)(c / 128) * M + r] = m / 127.f * q8::ROT_RS;
        uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(v[e] * inv) & 0xff) << (8 * e);
#pragma unroll
        for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(v[4 + e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint2*>(qr + c) = make_uint2(lo4, hi4);
    }
}

#ifdef Q8K
// =====================================================================================================================
// Q8K (ruj r8q2, 24 Oct 2026; G9X port onto rgh's g9r6 incl. ROT, ruj g9x): bit-identical twins of rgh's Q8 kernels
// above. Compiled only into q8k.cubin
// (q8k.cu = "#define Q8K 1" + "#include "q8.cu""); without Q8K this file builds rgh's q8.cubin byte for byte (nothing
// above this block changed). A twin <base name>_k<LV> has the lever bits LV:
//   1 LUT  (q8g, q8gr) the SiLU of the bf16 gate g comes from q8k_silu[bits(g)], a 65,536-entry bf16 table filled by
//          q8k_lut_fill with swiglu1's own expression (bf16(silu_f32(g))); the product bf16(s * u), the centring, the
//          row amax and the int8 rounding are the base epilogue's code
//   2 RING (q8h, q8g, q8gr) one cp.async ring over the 19 int8 k-tiles and the 2 protected bf16 k-tiles: the protected
//          tiles are issued while the last int8 tiles compute, the dequant runs between the two MMA kinds (the ring
//          neither drains nor restarts); the MMAs, their order and the accumulators are the base kernel's
//   4 WTL  (q8h, q8g, q8gr, q8f) the int8 weights (q8h / q8g / q8gr: also the protected bf16 columns) in pjs's WTL layout, each
//          (32-row band, 64-unit k-tile) one contiguous 4 KB: unit (n, k) at ((n / 32) (ld / 64) + k / 64) 2048 +
//          (n % 32) 64 + k % 64; only the global source address of each B cp.async changes (same 16 B, same smem chunk)
//   tiles s96x256 (q8h, q8g, q8gr; any LV, LV 0 included): ONE 96-row tail tile row where the base runs two 64-row ones
//          (R % 128 in (64, 96]); every output element is computed by the same per-element code (int32 sums exact,
//          dequant / protected HMMA chain / epilogue per element, the q8g row amax an order-free max; q8gr: each row's
//          128-column rotation is one warp's (rgh's epi_g_rot), whichever warp and tile height)
//   ROT (q8gr = rgh's ROT SwiGLU kernels: the 128 SwiGLU outputs of a row rotated by q8::rot_sign + fwht before the
//          row amax / int8 rounding): the twins q8gr_*_k<LV> run rgh's epi_g_rot code (epi_g_rotK: the same, with
//          the LUT option on its SwiGLU line) after the same in_passes as the q8g twins
// Credit: rgh (Q8 and ROT: everything reused from above; the s96 geometry: COV9b / NG7T), pjs (WTL, NMAJ), ruj (these
// twins).
extern "C" {
__device__ uint16_t q8k_silu[65536];   // bf16 bits of bf16(silu_f32(g)) for every bf16 bit pattern g
}

namespace q8k {
using namespace q8;

template <class C, int LV>
struct LoaderK {   // q8::Loader with the WTL B addressing (LV & 4)
    static constexpr int RS = C::THREADS / C::CH;
    static constexpr int NA = (C::BM + RS - 1) / RS;
    static constexpr int NB = (C::BN + RS - 1) / RS;
    static constexpr int NQ = NA + NB;
    static constexpr bool W = (LV & 4) != 0;
    const U* pa;
    const U* pb;
    uint32_t offa[NA], offb[NB];
    uint32_t dst0;
    uint32_t amask;
    int r0;
    DEVI void init(const U* A, const U* B, int m0, int n0, int n1, int M, int lda, int ldb) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A;
        pb = B;
        amask = 0;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int r = r0 + q * RS;
            const int gm = m0 + r;
            const bool ok = gm < M;
            offa[q] = (uint32_t)((ok ? gm : 0) * lda + c * 8);
            amask |= (ok ? 1u : 0u) << q;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int r = r0 + q * RS;
            int wr;
            if (C::EPI == 1 && n1 >= 0) {
                const int wn = r / C::WTN, h = (r % C::WTN) >= C::WTN / 2, cc = r % (C::WTN / 2);
                wr = (h ? n1 : n0) + wn * (C::WTN / 2) + cc;
            } else {
                wr = n0 + r;
            }
            if constexpr (W) offb[q] = (uint32_t)((wr >> 5) * (ldb >> 6) * 2048 + (wr & 31) * 64 + c * 8);
            else offb[q] = (uint32_t)(wr * ldb + c * 8);
        }
    }
    DEVI void issue_all(uint32_t sA, uint32_t sB, int k0) {
        const uint32_t kb = W ? (uint32_t)((k0 >> 6) * 2048 + (k0 & 63)) : (uint32_t)k0;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (q < NA) {
                if (C::BM % RS == 0 || r0 + q * RS < C::BM)
                    cp_async16(sA + dst0 + q * RS * C::BK * 2, pa + offa[q] + k0, (amask >> q) & 1);
            } else {
                const int qb = q - NA;
                if (C::BN % RS == 0 || r0 + qb * RS < C::BN)
                    cp_async16(sB + dst0 + qb * RS * C::BK * 2, pb + offb[qb] + kb, true);
            }
        }
    }
};

// q8::mainloop with LoaderK (LV & 4: WTL B addresses); everything else is the base loop
template <class C, class T, bool ZERO, int LV>
DEVI void mainloopK(const U* __restrict__ A, const U* __restrict__ B, int M, int Ku, int lda, int ldb, int m0, int n0,
                    int n1, U* smem, T (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
    if constexpr (ZERO) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = T(0);
    }
    LoaderK<C, LV> ld;
    ld.init(A, B, m0, n0, n1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa(t), sb(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa(tt % S), sb(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

// RING (LV & 2): ia = A[:, :Ku] . B[:, :Ku] (int8, from zero), dq() (ia -> fa), fa += AH[:, :KHu] . BH[:, :KHu] (bf16)
// through ONE cp.async ring: k-tiles 0 .. KT8 - 1 are the int8 ones, KT8 .. KTT - 1 the protected ones (the loader
// switches to AH / BH when it issues tile KT8; tiles are issued in ascending order, so every int8 tile was issued
// before). Per k-tile: the base loop's waits, barrier, refill point and fragment double-buffering; the MMAs of each
// k-tile are the base kernel's in the base order (int8 tiles: IMMA into ia; protected tiles: HMMA into fa).
template <class C, int LV, class DQ>
DEVI void mainloop_cr(const U* __restrict__ A, const U* __restrict__ B, const U* __restrict__ AH,
                      const U* __restrict__ BH, int M, int Ku, int KHu, int lda, int ldb, int m0, int n0, int n1,
                      U* smem, int (&ia)[C::MT][C::NT][4], float (&fa)[C::MT][C::NT][4], DQ dq) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT8 = Ku / BK, KTT = KT8 + KHu / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) ia[i][j][q] = 0;
    LoaderK<C, LV> ld;
    ld.init(A, B, m0, n0, n1, M, lda, ldb);
    auto issue = [&](int tt) {
        if (tt == KT8) ld.init(AH, BH, m0, n0, n1, M, KHu, KHu);
        ld.issue_all(sa(tt % S), sb(tt % S), (tt < KT8 ? tt : tt - KT8) * BK);
    };
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KTT) issue(t);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;
    auto ktile = [&](int kt, auto& acc) {
        using T = typename std::remove_reference<decltype(acc[0][0][0])>::type;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KTT) issue(tt);
            }
            if (ks < KS - 1 || kt + 1 < KTT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    };
    for (int kt = 0; kt < KT8; ++kt) ktile(kt, ia);
    dq();
    for (int kt = KT8; kt < KTT; ++kt) ktile(kt, fa);
    cp_wait<0>();
    __syncthreads();
}

// the int8 pass, the dequant and the protected pass of an "in" tile: base order (LV & 2 = 0) or one ring (LV & 2)
template <class C, int LV>
DEVI void in_passes(const int8_t* A8, const int8_t* B8, const __nv_bfloat16* AH, const __nv_bfloat16* BH,
                    const float* sa, const float* sw, const float* bias, int M, int K, int KH, int lda, int ldb,
                    int m0, int nbase, int n1, int H, U* smem, float (&fa)[C::MT][C::NT][4]) {
    int ia[C::MT][C::NT][4];
    const U* A = reinterpret_cast<const U*>(A8);
    const U* B = reinterpret_cast<const U*>(B8);
    if constexpr ((LV & 2) != 0) {
        mainloop_cr<C, LV>(A, B, reinterpret_cast<const U*>(AH), reinterpret_cast<const U*>(BH), M, K / 2,
                           KH > 0 ? KH : 0, lda / 2, ldb / 2, m0, nbase, n1, smem, ia, fa,
                           [&]() { dequant<C>(ia, fa, sa, sw, bias, M, m0, nbase, H); });
    } else {
        mainloopK<C, int, true, LV>(A, B, M, K / 2, lda / 2, ldb / 2, m0, nbase, n1, smem, ia);
        dequant<C>(ia, fa, sa, sw, bias, M, m0, nbase, H);
        if (KH > 0)
            mainloopK<C, float, false, LV>(reinterpret_cast<const U*>(AH), reinterpret_cast<const U*>(BH), M, KH, KH,
                                           KH, m0, nbase, n1, smem, fa);
    }
}

// q8h twin tile (EPI 2, bf16 out; q8::tile with PART false)
template <class C, int LV>
DEVI void tileK(const Args& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 2, "plain tiles");
    const int nbase = nt * C::BN;
    float fa[C::MT][C::NT][4];
    in_passes<C, LV>(a.A, a.B, a.AH, a.BH, a.sa, a.sw, a.bias, a.M, a.K, a.KH, a.lda, a.ldb, m0, nbase, -1, 0, smem,
                     fa);
    store_tile<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase, smem);
}
template <class CB, class CS, int LV>
DEVI void heteroK(const Args& a, U* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tileK<CB, LV>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tileK<CS, LV>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// q8gr twin epilogue (ruj g9x) = rgh's q8::epi_g_rot; LV & 1: the SiLU from q8k_silu (the q8g twins' LUT line)
template <class C, int LV>
DEVI void epi_g_rotK(const ArgsG& a, const float (&fa)[C::MT][C::NT][4], int m0, int o0, int H, U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2, LDS = 132;
    float* S = reinterpret_cast<float*>(smem);   // [BM][LDS] (smem free after the loops)
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
#pragma unroll
            for (int j = 0; j < NH; ++j)
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
                    const int cl = wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[o0 + cl]) * a.is[o0 + cl];
                    S[rl * LDS + cl] = x;
                }
        }
    __syncthreads();
    const int g = o0 / 128;
    for (int rl = warp; rl < C::BM; rl += C::CW) {
        const float4 y4 = *reinterpret_cast<const float4*>(S + rl * LDS + lane * 4);
        float y[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = rot_sign(lane * 4 + e, y[e]);
        fwht_reg<4>(y);
        fwht_lanes<4, 16>(y, lane);
        float m = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const int r = m0 + rl;
        if (r >= a.M) continue;
        const float inv = 127.f / m;
        if (lane == 0) a.hs[(size_t)g * a.M + r] = m / 127.f * ROT_RS;
        uint32_t q4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) q4 |= (uint32_t)(__float2int_rn(y[e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint32_t*>(a.hq + (size_t)r * H + o0 + lane * 4) = q4;
    }
}

// q8g twin tile (q8::tile_g; LV & 1: the SiLU from q8k_silu; ROT: q8gr, rgh's rotating epilogue)
template <class C, int LV, bool ROT>
DEVI void tile_gK(const ArgsG& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    float fa[C::MT][C::NT][4];
    in_passes<C, LV>(a.A, a.B, a.AH, a.BH, a.sa, a.sw, a.bias, a.M, a.K, a.KH, a.lda, a.ldb, m0, o0, H + o0, H, smem,
                     fa);
    if constexpr (ROT) {
        epi_g_rotK<C, LV>(a, fa, m0, o0, H, smem);
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* red = reinterpret_cast<float*>(smem);   // [BM][WN] row maxima of the warp columns (smem free after the loops)
    float v[C::MT][2][NH][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < NH; ++j)
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
                    const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[c]) * a.is[c];
                    v[i][h][j][e] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (t4 == 0) red[(wm * C::WTM + i * 16 + lane / 4 + h * 8) * C::WN + wn] = m;
        }
    __syncthreads();
    const int g = o0 / 128;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int r = m0 + rl;
            float m = 1e-12f;
#pragma unroll
            for (int w = 0; w < C::WN; ++w) m = fmaxf(m, red[rl * C::WN + w]);
            const float inv = 127.f / m;
            if (r >= a.M) continue;
            if (wn == 0 && t4 == 0) a.hs[(size_t)g * a.M + r] = m / 127.f;
            int8_t* row = a.hq + (size_t)r * H + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const uint32_t q0 = (uint32_t)(__float2int_rn(v[i][h][j][0] * inv) & 0xff);
                const uint32_t q1 = (uint32_t)(__float2int_rn(v[i][h][j][1] * inv) & 0xff);
                *reinterpret_cast<uint16_t*>(row + j * 8) = (uint16_t)(q0 | (q1 << 8));
            }
        }
}
template <class CB, class CS, int LV, bool ROT>
DEVI void hetero_gK(const ArgsG& a, U* smem) {
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_gK<CB, LV, ROT>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_gK<CS, LV, ROT>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// q8::mainloop_fine with LoaderK (LV & 4: WTL B addresses)
template <class C, int LV>
DEVI void mainloop_fineK(const U* __restrict__ A, const U* __restrict__ B, int M, int N, int Ku, int lda, int ldb,
                         int m0, int n0, U* smem, float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                         const float* __restrict__ sw, int wfull) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    static_assert(BK == 64, "one k-tile = one 128-column group");
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa_ = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb_ = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) fa[i][j][q] = 0.f;
    const int r0 = row_of<C>(0, 0, m0);    // rows r0 + 16 i + 8 h
    const int c0 = col_of<C>(0, 0, n0, 0);  // columns c0 + 8 j (+ 1)
    float wa[NT][2];
    if (wfull) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float2 w = *reinterpret_cast<const float2*>(sw + c0 + 8 * j);
            wa[j][0] = w.x;
            wa[j][1] = w.y;
        }
    }
    LoaderK<C, LV> ld;
    ld.init(A, B, m0, n0, -1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa_(t), sb_(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa_(0), sb_(0), 0);
    int s = 0;
    int ia[MT][NT][4];
    float ra[MT][2];
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == 0) {   // this group's scales (consumed at the last k-step)
                const float* sr = sa + (size_t)kt * M;
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = r0 + 16 * i + 8 * h;
                        ra[i][h] = r < M ? sr[r] : 0.f;
                    }
                if (!wfull) {
                    const float* sc = sw + (size_t)kt * N + c0;
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const float2 w = *reinterpret_cast<const float2*>(sc + 8 * j);
                        wa[j][0] = w.x;
                        wa[j][1] = w.y;
                    }
                }
            }
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa_(tt % S), sb_(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa_(ns), sb_(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    if (ks == 0) mma_s8z(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                    else mma_s8(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
            if (ks == KS - 1) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const float w0 = wa[j][0], w1 = wa[j][1];
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            fa[i][j][q] = fmaf((float)ia[i][j][q], ra[i][q >> 1] * ((q & 1) ? w1 : w0), fa[i][j][q]);
                }
            }
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}
// q8f twin tile (q8::tile_f); WTL: split y starts at k-tile g0 of every 32-row band (g0 2048 units)
template <class C, int LV>
DEVI void tile_fK(const ArgsF& a, int m0, int nt, U* smem) {
    const int y = blockIdx.y;
    const int n0 = nt * C::BN;
    const int g0 = y * (a.Ks / 128);
    const U* Bp = (LV & 4) ? reinterpret_cast<const U*>(a.B) + (size_t)g0 * 2048
                           : reinterpret_cast<const U*>(a.B + (size_t)y * a.Ks);
    float fa[C::MT][C::NT][4];
    mainloop_fineK<C, LV>(reinterpret_cast<const U*>(a.A + (size_t)y * a.Ks), Bp, a.M, a.N, a.Ks / 2, a.lda / 2,
                          a.ldb / 2, m0, n0, smem, fa, a.sa + (size_t)g0 * a.M, a.wfull ? a.sw : a.sw + (size_t)g0 * a.N,
                          a.wfull);
    if (y == 0 && a.bias != nullptr) {
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            const float2 b = *reinterpret_cast<const float2*>(a.bias + col_of<C>(j, 0, n0, 0));
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) fa[i][j][q] += (q & 1) ? b.y : b.x;
        }
    }
    store_part<C>(fa, a.out + (size_t)y * a.M * a.N, a.M, a.N, m0, n0);
}
template <class CB, class CS, int LV>
DEVI void hetero_fK(const ArgsF& a, U* smem) {
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tile_fK<CB, LV>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_fK<CS, LV>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

}  // namespace q8k

// the SiLU table: q8k_silu[g] = bf16(silu_f32(g)) for the 65,536 bf16 bit patterns (grid 256 x 256), and its read-out
extern "C" __global__ void __launch_bounds__(256) q8k_lut_fill() {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const __nv_bfloat16 g = __ushort_as_bfloat16((unsigned short)i);
    q8k_silu[i] = __bfloat16_as_ushort(__float2bfloat16_rn(q8::silu_f32(__bfloat162float(g))));
}
extern "C" __global__ void __launch_bounds__(256) q8k_lut_get(uint16_t* __restrict__ out) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    out[i] = q8k_silu[i];
}

#define Q8K_H(NAME, CB, CS, LV)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::Args a) {                                \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8k::heteroK<CB, CS, LV>(a, reinterpret_cast<q8::U*>(smem_raw));                                          \
    }
#define Q8K_G(NAME, CB, CS, LV)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsG a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8k::hetero_gK<CB, CS, LV, false>(a, reinterpret_cast<q8::U*>(smem_raw));                                 \
    }
#define Q8K_GR(NAME, CB, CS, LV)                                                                                  \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsG a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8k::hetero_gK<CB, CS, LV, true>(a, reinterpret_cast<q8::U*>(smem_raw));                                  \
    }
#define Q8K_F(NAME, CB, CS, LV)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsF a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8k::hetero_fK<CB, CS, LV>(a, reinterpret_cast<q8::U*>(smem_raw));                                        \
    }

using P_S96x256 = q8::Cfg<96, 256, 64, 2, 4, 2, 2>;
using W_S96x256 = q8::Cfg<96, 256, 64, 2, 4, 2, 1>;

// q8h (LV 2 / 4 / 6; s96 also LV 0)
Q8K_H(q8h_b128x256_s64x256_k2, P_B128x256, P_S64x256, 2)
Q8K_H(q8h_b128x256_s64x256_k4, P_B128x256, P_S64x256, 4)
Q8K_H(q8h_b128x256_s64x256_k6, P_B128x256, P_S64x256, 6)
Q8K_H(q8h_b128x256_s32x256_k2, P_B128x256, P_S32x256, 2)
Q8K_H(q8h_b128x256_s32x256_k4, P_B128x256, P_S32x256, 4)
Q8K_H(q8h_b128x256_s32x256_k6, P_B128x256, P_S32x256, 6)
Q8K_H(q8h_b128x256_s96x256_k0, P_B128x256, P_S96x256, 0)
Q8K_H(q8h_b128x256_s96x256_k2, P_B128x256, P_S96x256, 2)
Q8K_H(q8h_b128x256_s96x256_k4, P_B128x256, P_S96x256, 4)
Q8K_H(q8h_b128x256_s96x256_k6, P_B128x256, P_S96x256, 6)
// q8g (LV 1 .. 7; s96 also LV 0)
#define Q8K_G8(S, CS)                                                                                             \
    Q8K_G(q8g_b128x256_##S##_k1, W_B128x256, CS, 1)                                                                \
    Q8K_G(q8g_b128x256_##S##_k2, W_B128x256, CS, 2)                                                                \
    Q8K_G(q8g_b128x256_##S##_k3, W_B128x256, CS, 3)                                                                \
    Q8K_G(q8g_b128x256_##S##_k4, W_B128x256, CS, 4)                                                                \
    Q8K_G(q8g_b128x256_##S##_k5, W_B128x256, CS, 5)                                                                \
    Q8K_G(q8g_b128x256_##S##_k6, W_B128x256, CS, 6)                                                                \
    Q8K_G(q8g_b128x256_##S##_k7, W_B128x256, CS, 7)
Q8K_G8(s64x256, W_S64x256)
Q8K_G8(s32x256, W_S32x256)
Q8K_G8(s96x256, W_S96x256)
Q8K_G(q8g_b128x256_s96x256_k0, W_B128x256, W_S96x256, 0)
// q8gr (ruj g9x: twins of rgh's ROT SwiGLU kernels q8gr_*; LV 1 .. 7; s96 also LV 0)
#define Q8K_GR8(S, CS)                                                                                            \
    Q8K_GR(q8gr_b128x256_##S##_k1, W_B128x256, CS, 1)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k2, W_B128x256, CS, 2)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k3, W_B128x256, CS, 3)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k4, W_B128x256, CS, 4)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k5, W_B128x256, CS, 5)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k6, W_B128x256, CS, 6)                                                              \
    Q8K_GR(q8gr_b128x256_##S##_k7, W_B128x256, CS, 7)
Q8K_GR8(s64x256, W_S64x256)
Q8K_GR8(s32x256, W_S32x256)
Q8K_GR8(s96x256, W_S96x256)
Q8K_GR(q8gr_b128x256_s96x256_k0, W_B128x256, W_S96x256, 0)
// q8f (LV 4). No s96 tails for q8f: an s96x128 small tile moves ptxas's q8f spill into the BIG tile's k-loop (3 LDL.64
// per k-tile instead of the base's 1; ptxas -v 24 B spill), so q8f keeps two s64 rows at R % 128 = 96
Q8K_F(q8f_b128x128_s64x128_k4, P_B128x128, P_S64x128, 4)
Q8K_F(q8f_b128x128_s32x128_k4, P_B128x128, P_S32x128, 4)
#endif  // Q8K
