from __future__ import annotations

import ctypes as C
import os
import re
import sys
import threading

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CFGS = {
    "bg_128x128x128_w2x2_s3": (128, 128, 128, 128, 3, 0),
    "bg_128x128x128_w2x4_s3": (128, 128, 128, 256, 3, 0),
    "bg_128x128x128_w4x2_s3": (128, 128, 128, 256, 3, 0),
    "bg_128x256x128_w2x4_s2": (128, 256, 128, 256, 2, 0),
    "bg_256x128x128_w4x2_s2": (256, 128, 128, 256, 2, 0),
    "bg_64x128x128_w1x4_s4": (64, 128, 128, 128, 4, 0),
    "bg_64x256x128_w1x4_s2": (64, 256, 128, 128, 2, 0),
    "bg_64x128x128_w2x2_s4": (64, 128, 128, 128, 4, 0),
    "bg_128x64x128_w2x2_s4": (128, 64, 128, 128, 4, 0),
    "bg_128x128x64_w2x4_s5": (128, 128, 64, 256, 5, 0),
    "bg_128x128x128_w2x2_s3_b2": (128, 128, 128, 128, 3, 0),
    "bg_128x128x64_w2x2_s3_b2": (128, 128, 64, 128, 3, 0),
    "bg_64x128x128_w2x2_s2_b2": (64, 128, 128, 128, 2, 0),
    "bg_64x128x64_w2x2_s4_b2": (64, 128, 64, 128, 4, 0),
    "bg_128x64x128_w2x2_s2_b2": (128, 64, 128, 128, 2, 0),
    "bg_128x64x64_w2x2_s4_b2": (128, 64, 64, 128, 4, 0),
    "hy_128_32x128x64_w2x2_b2": (128, 128, 64, 128, 3, 32),
    "hy_128_64x128x64_w2x2_b2": (128, 128, 64, 128, 3, 64),
    "hy_128_32x128x128_w2x2": (128, 128, 128, 128, 3, 32),
    "hy_128_64x128x128_w2x2": (128, 128, 128, 128, 3, 64),
    "hy_128_32x256x128_w2x4": (128, 256, 128, 256, 2, 32),
    "hy_256_64x128x128_w4x2": (256, 128, 128, 256, 2, 64),
    "hy_128_32x128x128_w2x4": (128, 128, 128, 256, 3, 32),
}


_B7NAME = re.compile(rb"(b7(p?)_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")
_B7SNAME = re.compile(rb"(b7s_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)_b(\d+))\x00")


class BF16Gemm:
    def __init__(self, cubin=None):
        cubin = cubin or os.environ.get("REAX_BF16G_CUBIN") or os.path.join(_DIR, "bf16gemm.cubin")
        self.cu = C.CDLL("libcuda.so.1")
        self.cul = C.PyDLL("libcuda.so.1") if os.environ.get("REAX_PYDLL", "0") == "1" else self.cu
        torch.cuda.init()
        torch.empty(1, device="cuda")
        self.mod = C.c_void_p()
        self._img = C.create_string_buffer(open(cubin, "rb").read())
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name, (BM, BN, BK, th, st, bms) in CFGS.items():
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            smem = st * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                print(f"bf16g: {name} skipped ({smem} B smem)", flush=True)
                continue
            self.fn[name] = (f, BM, BN, th, smem, bms)
        self.b7 = {}
        if os.environ.get("REAX_BF16G7_CUBIN"):
            self._load_b7(os.environ["REAX_BF16G7_CUBIN"])

    def _load_b7(self, path):
        only = None
        if os.environ.get("REAX_BF16G7_ONLY"):
            only = {}
            for part in os.environ["REAX_BF16G7_ONLY"].split(";"):
                if part.strip():
                    nk, cfgs = part.split(":")
                    n, k = (int(v) for v in nk.split("x"))
                    only[(n, k)] = {c.strip() for c in cfgs.split(",") if c.strip()}
        self._b7_only = only
        self.nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._tl = threading.local()
        img = open(path, "rb").read()
        self._img7 = C.create_string_buffer(img)
        self.mod7 = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod7), self._img7), "cuModuleLoadData b7")
        wanted = None if only is None else set().union(*only.values())
        for m in sorted(set(_B7NAME.findall(img))):
            name = m[0].decode()
            if wanted is not None and name not in wanted:
                continue
            BM, BN, BK, WM, WN, ST, B = (int(x) for x in m[2:])
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod7, name.encode()) != 0:
                continue
            smem = ST * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                print(f"bf16g: {name} skipped ({smem} B smem)", flush=True)
                continue
            self.b7[name] = (f, BM, BN, BK, WM * WN * 32, smem, B, m[1] == b"p")
            self.fn[name] = self.b7[name]
        print(f"bf16g: b7 {len(self.b7)} kernels from {path}" + (" (allow-list)" if only is not None else ""), flush=True)
        self.b7s = {}
        if os.environ.get("REAX_BF16SK_CUBIN"):
            self._load_b7s(os.environ["REAX_BF16SK_CUBIN"])

    @staticmethod
    def _only(var):
        if not os.environ.get(var):
            return None
        only = {}
        for part in os.environ[var].split(";"):
            if part.strip():
                nk, cfgs = part.split(":")
                n, k = (int(v) for v in nk.split("x"))
                only[(n, k)] = {c.strip() for c in cfgs.split(",") if c.strip()}
        return only

    def _load_b7s(self, path):
        self._b7s_only = self._only("REAX_BF16SK_ONLY")
        wanted = None if self._b7s_only is None else set().union(*self._b7s_only.values())
        img = open(path, "rb").read()
        self._img7s = C.create_string_buffer(img)
        self.mod7s = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod7s), self._img7s), "cuModuleLoadData b7s")
        for m in sorted(set(_B7SNAME.findall(img))):
            name = m[0].decode()
            if wanted is not None and name not in wanted:
                continue
            BM, BN, BK, WM, WN, ST, B = (int(x) for x in m[1:])
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod7s, name.encode()) != 0:
                continue
            smem = ST * (BM + BN) * BK
            if self.cu.cuFuncSetAttribute(f, 8, smem) != 0:
                print(f"bf16g: {name} skipped ({smem} B smem)", flush=True)
                continue
            self.b7s[name] = (f, BM, BN, BK, WM * WN * 32, smem, B)
            self.fn[name] = self.b7s[name]
        n = max([B * BM * BN for _, BM, BN, _, _, _, B in self.b7s.values()] or [1]) * self.nsm
        self._ws = (torch.empty(n, device="cuda", dtype=torch.float32),
                    torch.zeros(2 * self.nsm, device="cuda", dtype=torch.int32))
        torch.cuda.synchronize()
        print(f"bf16g: b7s {len(self.b7s)} serial stream-K kernels from {path}", flush=True)

    def _b7s_args(self):
        a = getattr(self._tl, "s", None)
        if a is None:
            v = [C.c_void_p() for _ in range(5)] + [C.c_int() for _ in range(4)] + [C.c_void_p() for _ in range(3)]
            a = self._tl.s = (v, (C.c_void_p * len(v))(*[C.cast(C.pointer(x), C.c_void_p) for x in v]))
        return a

    def _b7s_call(self, a, w, cfg, gm, out):
        f, BM, BN, BK, th, smem, B = self.b7s[cfg]
        M, K = a.shape
        N = w.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
        T = ((M + BM - 1) // BM) * (N // BN)
        grid = min(self.nsm * B, T * (2 * K // BK))
        ws, flags = self._ws
        v, prm = self._b7s_args()
        v[0].value, v[1].value, v[2].value, v[3].value, v[4].value = a.data_ptr(), w.data_ptr(), c.data_ptr(), None, None
        v[5].value, v[6].value, v[7].value, v[8].value = M, N, 2 * K, gm
        v[9].value, v[10].value, v[11].value = None, ws.data_ptr(), flags.data_ptr()
        rc = self.cul.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                     prm, None)
        if rc:
            self._chk(rc, "launch " + cfg)
        return c

    def _b7_args(self):
        a = getattr(self._tl, "a", None)
        if a is None:
            v = [C.c_void_p() for _ in range(5)] + [C.c_int() for _ in range(4)]
            a = self._tl.a = (v, (C.c_void_p * len(v))(*[C.cast(C.pointer(x), C.c_void_p) for x in v]))
        return a

    def _b7_call(self, a, w, cfg, gm, out):
        f, BM, BN, BK, th, smem, B, persist = self.b7[cfg]
        M, K = a.shape
        N = w.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
        T = ((M + BM - 1) // BM) * (N // BN)
        grid = min(T, self.nsm * B) if persist else T
        v, prm = self._b7_args()
        v[0].value, v[1].value, v[2].value, v[3].value, v[4].value = a.data_ptr(), w.data_ptr(), c.data_ptr(), None, None
        v[5].value, v[6].value, v[7].value, v[8].value = M, N, 2 * K, gm
        rc = self.cul.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                     prm, None)
        if rc:
            self._chk(rc, "launch " + cfg)
        return c

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: CUDA driver error {rc}")

    def supports(self, cfg, N, K):
        if cfg in getattr(self, "b7s", ()):
            _, BM, BN, BK, *_ = self.b7s[cfg]
            only = self._b7s_only
            return N % BN == 0 and (2 * K) % BK == 0 and (only is None or cfg in only.get((N, K), ()))
        if cfg in self.b7:
            _, BM, BN, BK, *_ = self.b7[cfg]
            only = self._b7_only
            return N % BN == 0 and (2 * K) % BK == 0 and (only is None or cfg in only.get((N, K), ()))
        if cfg not in self.fn:
            return False
        _, BM, BN, _, _, _ = self.fn[cfg]
        return N % BN == 0 and (2 * K) % CFGS[cfg][2] == 0

    def __call__(self, a, w, cfg, gm=8, out=None):
        if cfg in self.b7:
            return self._b7_call(a, w, cfg, gm, out)
        if cfg in getattr(self, "b7s", ()):
            return self._b7s_call(a, w, cfg, gm, out)
        f, BM, BN, th, smem, bms = self.fn[cfg]
        M, K = a.shape
        N = w.shape[0]
        c = out if out is not None else torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
        args = [C.c_void_p(a.data_ptr()), C.c_void_p(w.data_ptr()), C.c_void_p(c.data_ptr()), C.c_int(M), C.c_int(N),
                C.c_int(2 * K), C.c_int(gm)]
        if bms:
            MB = (M // BM) * BM
            grid = (MB // BM) * (N // BN) + ((M - MB + bms - 1) // bms) * (N // BN)
            args.append(C.c_int(MB))
        else:
            grid = ((M + BM - 1) // BM) * (N // BN)
        prm = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._chk(self.cul.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, stream, prm, None), f"launch {cfg}")
        return c


SHAPES = [(8192, 2816), (10240, 2816), (2816, 8192), (2816, 4096), (4224, 2816), (2816, 2112)]


def test(ms_time):
    import vllm.model_executor.determinism.batch_invariant as bi
    import reax_lt

    flush = torch.empty(256 << 20, dtype=torch.int8, device="cuda")

    def t_ms(fn, reps=11):
        fn()
        ts = []
        for _ in range(reps):
            flush.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return ts[len(ts) // 2]

    def ref(x, w):
        bi._REAX_BYPASS[0] = True
        try:
            return bi.matmul_persistent(x, w.t())
        finally:
            bi._REAX_BYPASS[0] = False

    def lt_best(x, w, r):
        N, K = w.shape
        M = x.shape[0]
        seen, best = set(), float("inf")
        for hM in (M, 130, 512, 2048):
            for algo, _ in reax_lt.algos(hM, N, K, 64):
                key = bytes(algo.numpy().tobytes())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if torch.equal(reax_lt.mm(x, w, algo), r):
                        best = min(best, t_ms(lambda: reax_lt.mm(x, w, algo), reps=7))
                except RuntimeError:
                    pass
        return best

    g = BF16Gemm()
    gen = torch.Generator(device="cuda").manual_seed(5)
    check_ms = (1, 2, 7, 16, 31, 64, 100, 127, 128, 129, 300, 511, 645, 777, 880, 900, 1014, 1023, 1100, 1650, 3001)
    for (N, K) in SHAPES:
        w = (torch.randn(N, K, device="cuda", generator=gen) * 0.02).bfloat16()
        X = torch.randn(max(check_ms + tuple(ms_time)), K, device="cuda", generator=gen).bfloat16()
        cfgs = [c for c in g.fn if g.supports(c, N, K)]
        bad = {}
        for M in check_ms:
            r = ref(X[:M], w)
            for c in cfgs:
                if c in bad:
                    continue
                if not torch.equal(g(X[:M].contiguous(), w, c), r):
                    bad[c] = M
        ok = [c for c in cfgs if c not in bad]
        print(f"{N}x{K}: {len(ok)}/{len(cfgs)} variants byte-identical at {len(check_ms)} M"
              + (f"; MISMATCH {bad}" if bad else ""), flush=True)
        for M in ms_time:
            x = X[:M].contiguous()
            r = ref(x, w)
            fl = 2.0 * M * N * K / 1e9
            tl = lt_best(x, w, r)
            tc = t_ms(lambda: torch.mm(x, w.t()))
            res = sorted((t_ms(lambda: g(x, w, c), reps=9), c) for c in ok)
            top = " | ".join(f"{c} {t * 1e3:.1f}" for t, c in res[:3])
            print(f"  M {M:>5}: LT pin {tl * 1e3:6.1f} us ({fl / tl:3.0f} TF) cuBLAS {tc * 1e3:6.1f} | ours best "
                  f"{res[0][0] * 1e3:6.1f} us ({fl / res[0][0]:3.0f} TF, {100 * (res[0][0] / tl - 1):+.1f} % vs LT) | {top}",
                  flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [645, 704, 800, 880, 900, 960, 1014, 1100])
