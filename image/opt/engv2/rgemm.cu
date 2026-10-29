#include "bf16g7.cu"
template <int BM, int BN, int BK, int WM, int WN, int ST>
__device__ void run_rg(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K, int gm) {
    using G = Gemm<BM, BN, BK, WM, WN, ST, false>;
    extern __shared__ __align__(128) uint8_t smem[];
    G g(A, B, M, N, K, smem);
    const int npm = (M + BM - 1) / BM, npn = N / BN, T = npm * npn;
    const int pid = blockIdx.x;
    if (pid >= T) return;
    int tm, tn;
    tile_of(pid, npm, npn, gm, tm, tn);
    float acc[G::MT][G::NT][4];
    g.prologue(tm * BM, tn * BN);
    g.mainloop(acc, tm * BM, tn * BN, false, 0, 0);
    const int gq = g.lane / 4, t4 = g.lane % 4;
#pragma unroll
    for (int i = 0; i < G::MT; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int m = tm * BM + g.wm * G::WTM + i * 16 + gq + h * 8;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < G::NT; j++) {
                const int n = tn * BN + g.wn * G::WTN + j * 8 + t4 * 2;
                *reinterpret_cast<float2*>(C + (size_t)m * N + n) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
            }
        }
}
#define RG(BM, BN, BK, WM, WN, ST, MINB)                                                                               \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB)                                                   \
        rg_##BM##x##BN##x##BK##_w##WM##x##WN##_s##ST##_b##MINB(const uint8_t* A, const uint8_t* B, float* C, int M,    \
                                                               int N, int K, int gm) {                                 \
        run_rg<BM, BN, BK, WM, WN, ST>(A, B, C, M, N, K, gm);                                                          \
    }
RG(16, 32, 128, 1, 2, 4, 4)
RG(16, 32, 64, 1, 2, 6, 4)
RG(16, 64, 128, 1, 4, 4, 4)
RG(16, 128, 128, 1, 8, 3, 2)
RG(32, 32, 128, 2, 2, 4, 4)
RG(32, 32, 64, 2, 2, 6, 4)
RG(32, 64, 128, 2, 4, 4, 2)
RG(64, 32, 128, 4, 2, 4, 2)
RG(32, 128, 128, 2, 8, 3, 1)
