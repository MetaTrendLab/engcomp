from __future__ import annotations

import ctypes as C
import os
import re
import threading

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kern")
_NAME = re.compile(rb"((?:g[6789]|b7)p?_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")
_GUNAME = re.compile(rb"(g7gu_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")


class F8G6:
    def __init__(self, cubin=os.path.join(_DIR, "f8g6.cubin")):
        self.cu = C.PyDLL("libcuda.so.1") if os.environ.get("Q35_G6_CDLL") != "1" else C.CDLL("libcuda.so.1")
        torch.cuda.init()
        torch.empty(1, device="cuda")
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self._tl = threading.local()
        img = open(cubin, "rb").read()
        self._img = C.create_string_buffer(img)
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for m in sorted(set(_NAME.findall(img)) | set(_GUNAME.findall(img))):
            name = m[0].decode()
            BM, BN, BK, WM, WN, ST, B = (int(x) for x in m[1:])
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) != 0:
                continue
            smem = ST * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                continue
            self.fn[name] = (f, BM, BN, BK, WM * WN * 32, smem, B, name[2] == "p")

    def _args(self):
        a = getattr(self._tl, "a", None)
        if a is None:
            v = [C.c_void_p() for _ in range(5)] + [C.c_int() for _ in range(4)] + [C.c_void_p()]
            a = self._tl.a = (v, (C.c_void_p * len(v))(*[C.cast(C.pointer(x), C.c_void_p) for x in v]))
        return a

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: CUDA driver error {rc}")

    def fits(self, cfg, N, K):
        _, BM, BN, BK, *_ = self.fn[cfg]
        return N % BN == 0 and K % BK == 0

    def __call__(self, a8, b8, sa=None, sb=None, cfg="g6_128x256x128_w2x4_s2_b1", gm=8, out=None):
        f, BM, BN, BK, th, smem, B, persist = self.fn[cfg]
        M, K = a8.shape
        N = b8.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a8.device, dtype=torch.bfloat16)
        T = ((M + BM - 1) // BM) * (N // BN)
        grid = min(T, self.nsm * B) if persist else T
        v, prm = self._args()
        v[0].value, v[1].value, v[2].value = a8.data_ptr(), b8.data_ptr(), c.data_ptr()
        v[3].value = sa.data_ptr() if sa is not None else None
        v[4].value = sb.data_ptr() if sb is not None else None
        v[5].value, v[6].value, v[7].value, v[8].value = M, N, K, gm
        rc = self.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                    prm, None)
        if rc:
            self._chk(rc, "launch " + cfg)
        return c

    def gu(self, a8, b8i, xs, wsi, cfg="g7gu_256x128x64_w4x2_s4_b1", gm=16):
        f, BM, BN, BK, th, smem, B, _ = self.fn[cfg]
        M, K = a8.shape
        N = b8i.shape[0]
        y = torch.empty(M, N // 2, device=a8.device, dtype=torch.bfloat16)
        amax = torch.zeros(M, device=a8.device, dtype=torch.float32)
        grid = ((M + BM - 1) // BM) * (N // BN)
        v, prm = self._args()
        v[0].value, v[1].value, v[2].value = a8.data_ptr(), b8i.data_ptr(), y.data_ptr()
        v[3].value, v[4].value = xs.data_ptr(), wsi.data_ptr()
        v[5].value, v[6].value, v[7].value, v[8].value = M, N, K, gm
        v[9].value = amax.data_ptr()
        rc = self.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                    prm, None)
        if rc:
            self._chk(rc, "launch " + cfg)
        return y, amax
