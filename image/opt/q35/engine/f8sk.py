from __future__ import annotations

import ctypes as C
import os
import re
import threading

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kern")
_NAME = re.compile(rb"((?:s[789]|s7gu)_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")


class F8SK:
    def __init__(self, cubin=os.path.join(_DIR, "f8sk.cubin")):
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
        for m in sorted(set(_NAME.findall(img))):
            name = m[0].decode()
            BM, BN, BK, WM, WN, ST, B = (int(x) for x in m[1:])
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) != 0:
                continue
            smem = ST * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                continue
            self.fn[name] = (f, BM, BN, BK, WM * WN * 32, smem, B)
        n = max([B * BM * BN for _, BM, BN, _, _, _, B in self.fn.values()] or [1]) * self.nsm
        self._ws = (torch.empty(n, device="cuda", dtype=torch.float32), torch.zeros(2 * self.nsm, device="cuda",
                                                                                   dtype=torch.int32))
        torch.cuda.synchronize()

    def _args(self):
        a = getattr(self._tl, "a", None)
        if a is None:
            v = [C.c_void_p() for _ in range(5)] + [C.c_int() for _ in range(4)] + [C.c_void_p() for _ in range(3)]
            a = self._tl.a = (v, (C.c_void_p * len(v))(*[C.cast(C.pointer(x), C.c_void_p) for x in v]))
        return a

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: CUDA driver error {rc}")

    def fits(self, cfg, N, K):
        _, BM, BN, BK, *_ = self.fn[cfg]
        return N % BN == 0 and K % BK == 0

    def _launch(self, cfg, a8, b8, out, sa, sb, M, N, K, gm, amax):
        f, BM, BN, BK, th, smem, B = self.fn[cfg]
        T = ((M + BM - 1) // BM) * (N // BN)
        grid = min(self.nsm * B, T * (K // BK))
        s = torch.cuda.current_stream()
        ws, flags = self._ws
        v, prm = self._args()
        v[0].value, v[1].value, v[2].value = a8.data_ptr(), b8.data_ptr(), out.data_ptr()
        v[3].value = sa.data_ptr() if sa is not None else None
        v[4].value = sb.data_ptr() if sb is not None else None
        v[5].value, v[6].value, v[7].value, v[8].value = M, N, K, gm
        v[9].value = amax.data_ptr() if amax is not None else None
        v[10].value, v[11].value = ws.data_ptr(), flags.data_ptr()
        rc = self.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(s.cuda_stream), prm, None)
        if rc:
            self._chk(rc, "launch " + cfg)

    def __call__(self, a8, b8, sa=None, sb=None, cfg="s7_128x256x128_w2x4_s2_b1", gm=8, out=None):
        M, K = a8.shape
        N = b8.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a8.device, dtype=torch.bfloat16)
        self._launch(cfg, a8, b8, c, sa, sb, M, N, K, gm, None)
        return c

    def gu(self, a8, b8i, xs, wsi, cfg="s7gu_128x256x128_w2x4_s2_b1", gm=16):
        M, K = a8.shape
        N = b8i.shape[0]
        y = torch.empty(M, N // 2, device=a8.device, dtype=torch.bfloat16)
        amax = torch.zeros(M, device=a8.device, dtype=torch.float32)
        self._launch(cfg, a8, b8i, y, xs, wsi, M, N, K, gm, amax)
        return y, amax


class G6SK:

    def __init__(self, g6_cubin=None, sk_cubin=None):
        from f8g6 import F8G6
        self.g6 = F8G6(g6_cubin) if g6_cubin else F8G6()
        self.sk = F8SK(sk_cubin) if sk_cubin else F8SK()
        self.fn = {**self.g6.fn, **self.sk.fn}
        self.nsm = self.g6.nsm

    def _of(self, cfg):
        return self.sk if cfg[0] == "s" else self.g6

    def fits(self, cfg, N, K):
        return self._of(cfg).fits(cfg, N, K)

    def __call__(self, a8, b8, sa=None, sb=None, cfg="g6_128x256x128_w2x4_s2_b1", gm=8, out=None):
        return self._of(cfg)(a8, b8, sa, sb, cfg=cfg, gm=gm, out=out)

    def gu(self, a8, b8i, xs, wsi, cfg="g7gu_256x128x64_w4x2_s4_b1", gm=16):
        return self._of(cfg).gu(a8, b8i, xs, wsi, cfg=cfg, gm=gm)
