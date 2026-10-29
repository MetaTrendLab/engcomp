// reax_lt: bf16 Linear (y = x @ w^T, fp32 accumulate) through cuBLASLt with an algorithm pinned by the caller, so the
// kernel (tile, stages, no split-K) is the same for every M - campaign experiments/s1pro-d0 (s1-pro, bit-exact gate).
//   algos(M, N, K, n)  -> list of (algo bytes, [id, tile, splitk, reduction, swizzle, custom, stages, inner, cluster])
//                         from the cuBLASLt heuristic for that problem, split-K reductions masked out
//   mm(x, w, algo)     -> x @ w^T with exactly that algorithm (raises if it does not support the problem)
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <vector>
#include <cstring>

#define CK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cublasLt ", #x, " status ", (int)s_); } while (0)

static cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) CK(cublasLtCreate(&h));
  return h;
}
static constexpr size_t kWs = 32u << 20;
static torch::Tensor& workspace() {
  static torch::Tensor ws;
  if (!ws.defined()) ws = torch::empty({(long)kWs}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  return ws;
}

struct Desc {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  Desc(int64_t M, int64_t N, int64_t K) {
    // column-major view: C[N, M] = W^T(op T on [K, N]) * X(op N on [K, M])
    CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
    CK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, K, N, K));
    CK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, M, K));
    CK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, N));
  }
  ~Desc() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

static std::vector<int64_t> cfg(const cublasLtMatmulAlgo_t& al) {
  std::vector<int64_t> v;
  for (int attr = 0; attr <= 8; ++attr) {
    uint64_t val = 0; size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&al, (cublasLtMatmulAlgoConfigAttributes_t)attr, &val, sizeof(val), &w) != CUBLAS_STATUS_SUCCESS) val = (uint64_t)-1;
    else if (w == 4) val &= 0xffffffffu; else if (w == 2) val &= 0xffffu;
    v.push_back((int64_t)val);
  }
  return v;
}

std::vector<std::pair<torch::Tensor, std::vector<int64_t>>> algos(int64_t M, int64_t N, int64_t K, int64_t n) {
  Desc d(M, N, K);
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = kWs;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(n);
  int got = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(handle(), d.op, d.a, d.b, d.c, d.c, pref, (int)n, res.data(), &got));
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<std::pair<torch::Tensor, std::vector<int64_t>>> out;
  for (int i = 0; i < got; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    auto t = torch::empty({(long)sizeof(cublasLtMatmulAlgo_t)}, torch::kUInt8);
    std::memcpy(t.data_ptr(), &res[i].algo, sizeof(cublasLtMatmulAlgo_t));
    out.emplace_back(t, cfg(res[i].algo));
  }
  return out;
}

torch::Tensor mm(torch::Tensor x, torch::Tensor w, torch::Tensor algo) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && x.stride(1) == 1 && w.is_contiguous());
  TORCH_CHECK(x.stride(0) == x.size(1), "x rows must be dense");
  TORCH_CHECK(algo.numel() == (long)sizeof(cublasLtMatmulAlgo_t));
  const at::cuda::CUDAGuard g(x.device());
  int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = torch::empty({M, N}, x.options());
  if (M == 0) return y;
  Desc d(M, N, K);
  cublasLtMatmulAlgo_t al;
  std::memcpy(&al, algo.data_ptr(), sizeof(al));
  float alpha = 1.f, beta = 0.f;
  auto& ws = workspace();
  CK(cublasLtMatmul(handle(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta, y.data_ptr(), d.c,
                    y.data_ptr(), d.c, &al, ws.data_ptr(), kWs, c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("algos", &algos);
  m.def("mm", &mm);
}
