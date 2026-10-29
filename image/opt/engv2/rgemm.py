from __future__ import annotations

import ctypes as C
import os
import re
import sys
import threading

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_NAME = re.compile(rb"(rg_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")


class RGemm:
    def __init__(self, cubin=None):
        cubin = cubin or os.environ.get("REAX_RGEMM_CUBIN") or os.path.join(_DIR, "rgemm.cubin")
        self.cu = C.CDLL("libcuda.so.1")
        self.cul = C.PyDLL("libcuda.so.1") if os.environ.get("REAX_PYDLL", "0") == "1" else self.cu
        torch.cuda.init()
        torch.empty(1, device="cuda")
        img = open(cubin, "rb").read()
        self._img = C.create_string_buffer(img)
        self.mod = C.c_void_p()
        rc = self.cu.cuModuleLoadData(C.byref(self.mod), self._img)
        if rc:
            raise RuntimeError(f"cuModuleLoadData: CUDA driver error {rc}")
        self.fn = {}
        for m in sorted(set(_NAME.findall(img))):
            name = m[0].decode()
            BM, BN, BK, WM, WN, ST, B = (int(x) for x in m[1:])
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) != 0:
                continue
            smem = ST * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                continue
            self.fn[name] = (f, BM, BN, BK, WM * WN * 32, smem)
        self._tl = threading.local()

    def _args(self):
        a = getattr(self._tl, "a", None)
        if a is None:
            v = [C.c_void_p() for _ in range(3)] + [C.c_int() for _ in range(4)]
            a = self._tl.a = (v, (C.c_void_p * len(v))(*[C.cast(C.pointer(x), C.c_void_p) for x in v]))
        return a

    def supports(self, cfg, N, K):
        _, BM, BN, BK, *_ = self.fn[cfg]
        return N % BN == 0 and (2 * K) % BK == 0

    def __call__(self, x, w, cfg, gm=8, out=None):
        f, BM, BN, BK, th, smem = self.fn[cfg]
        M, K = x.shape
        N = w.shape[0]
        c = out if out is not None else torch.empty(M, N, device=x.device, dtype=torch.float32)
        T = ((M + BM - 1) // BM) * (N // BN)
        v, prm = self._args()
        v[0].value, v[1].value, v[2].value = x.data_ptr(), w.data_ptr(), c.data_ptr()
        v[3].value, v[4].value, v[5].value, v[6].value = M, N, 2 * K, gm
        rc = self.cul.cuLaunchKernel(f, T, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                     prm, None)
        if rc:
            raise RuntimeError(f"launch {cfg}: CUDA driver error {rc}")
        return c


def _test(ms):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    g = RGemm()
    N, K = 128, 2816
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(N, K, device="cuda", generator=gen) * 0.02).bfloat16()

    def t_us(fn, reps=21, launches=4):
        fn()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(launches):
                fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / launches)
        ts.sort()
        return ts[len(ts) // 2]

    for M in ms:
        x = torch.randn(M, K, device="cuda", generator=gen).bfloat16()
        ref = torch.mm(x, w.T, out_dtype=torch.float32)
        t_ref = t_us(lambda: torch.mm(x, w.T, out_dtype=torch.float32))
        res, bad = [], []
        for cfg in g.fn:
            if not g.supports(cfg, N, K):
                continue
            for gm in (8, 16):
                out = g(x, w, cfg, gm)
                if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
                    bad.append(cfg)
                    continue
                res.append((t_us(lambda cfg=cfg, gm=gm: g(x, w, cfg, gm)), cfg, gm))
        res.sort()
        best = f"{res[0][1]}/{res[0][2]} {res[0][0]:6.1f} us -> {t_ref / res[0][0]:.2f}x" if res else "none exact"
        print(f"M {M:5d}: cuBLAS {t_ref:6.1f} us | best {best} | exact {len(res)}, differ {sorted(set(bad))}",
              flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    _test([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else
          [1, 7, 16, 33, 64, 100, 128, 256, 511, 640, 800, 884, 900, 1024, 1100, 1500, 2048, 2560])
