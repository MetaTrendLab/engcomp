from __future__ import annotations

import ctypes as C
import os

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kern")
CFGS = {
    "f8_128x128x128_w2x2_s3": (128, 128, 128, 128, 3),
    "f8_128x128x128_w2x4_s3": (128, 128, 128, 256, 3),
    "f8_128x128x128_w4x2_s3": (128, 128, 128, 256, 3),
    "f8_128x256x128_w2x4_s2": (128, 256, 128, 256, 2),
    "f8_256x128x128_w4x2_s2": (256, 128, 128, 256, 2),
    "f8_64x256x128_w1x4_s2": (64, 256, 128, 128, 2),
    "f8_128x256x64_w2x4_s4": (128, 256, 64, 256, 4),
    "f8_256x128x64_w4x2_s4": (256, 128, 64, 256, 4),
    "f8_128x256x64_w2x4_s3": (128, 256, 64, 256, 3),
    "f8_128x128x64_w2x2_s4": (128, 128, 64, 128, 4),
    "f8_128x128x64_w2x4_s5": (128, 128, 64, 256, 5),
    "f8_128x128x64_w2x2_s3_b2": (128, 128, 64, 128, 3),
    "f8_128x128x64_w2x4_s3_b2": (128, 128, 64, 256, 3),
    "f8_128x128x128_w2x4_s2_b2": (128, 128, 128, 256, 2),
    "gu_256x128x64_w4x2_s4": (256, 128, 64, 256, 4),
    "gu_128x256x128_w2x4_s2": (128, 256, 128, 256, 2),
    "gu_256x128x128_w4x2_s2": (256, 128, 128, 256, 2),
}


class F8Gemm:
    def __init__(self, cubin=os.path.join(_DIR, "f8gemm.cubin")):
        self.cu = C.CDLL("libcuda.so.1")
        torch.cuda.init()
        torch.empty(1, device="cuda")
        self.mod = C.c_void_p()
        img = open(cubin, "rb").read()
        self._img = C.create_string_buffer(img)
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name, (BM, BN, BK, th, st) in CFGS.items():
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            smem = st * (BM + BN) * BK
            self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attr")
            self.fn[name] = (f, BM, BN, th, smem)

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: CUDA driver error {rc}")

    def __call__(self, a8, b8, sa=None, sb=None, cfg="f8_128x256x128_w2x4_s2", gm=8, out=None):
        f, BM, BN, th, smem = self.fn[cfg]
        M, K = a8.shape
        N = b8.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a8.device, dtype=torch.bfloat16)
        grid = ((M + BM - 1) // BM) * (N // BN)
        args = [C.c_void_p(a8.data_ptr()), C.c_void_p(b8.data_ptr()), C.c_void_p(c.data_ptr()),
                C.c_void_p(sa.data_ptr() if sa is not None else 0), C.c_void_p(sb.data_ptr() if sb is not None else 0),
                C.c_int(M), C.c_int(N), C.c_int(K), C.c_int(gm)]
        prm = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._chk(self.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, stream, prm, None), "launch")
        return c

    def gu(self, a8, b8i, xs, wsi, cfg="gu_256x128x64_w4x2_s4", gm=8):
        f, BM, BN, th, smem = self.fn[cfg]
        M, K = a8.shape
        N = b8i.shape[0]
        y = torch.empty(M, N // 2, device=a8.device, dtype=torch.bfloat16)
        amax = torch.zeros(M, device=a8.device, dtype=torch.float32)
        grid = ((M + BM - 1) // BM) * (N // BN)
        args = [C.c_void_p(a8.data_ptr()), C.c_void_p(b8i.data_ptr()), C.c_void_p(y.data_ptr()), C.c_void_p(xs.data_ptr()),
                C.c_void_p(wsi.data_ptr()), C.c_int(M), C.c_int(N), C.c_int(K), C.c_int(gm), C.c_void_p(amax.data_ptr())]
        prm = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._chk(self.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, stream, prm, None), "launch gu")
        return y, amax
