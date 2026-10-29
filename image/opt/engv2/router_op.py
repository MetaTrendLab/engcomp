import sys

OLD = '''        # Tier 4: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
'''
NEW = '''        # engv2 (rgemm/router_op.py): tier 4 through rgemm where it is byte-identical to the torch.mm below
        if (_REAX_RGEMM and self.allow_cublas_router_gemm and x.dtype == torch.bfloat16 and x.dim() == 2
                and tuple(self.weight.shape) == (128, 2816) and self.weight.dtype == torch.bfloat16):
            return torch.ops.reax.router_mm(x, self.weight), None

''' + OLD
ADD = '''

# ---------------------------------------------------------------- engv2 Blink: exact router GEMM (rgemm/router_op.py)
import os as _reax_rg_os

from vllm.logger import init_logger as _reax_rg_init_logger

logger = _reax_rg_init_logger(__name__)
_REAX_RGEMM = _reax_rg_os.environ.get("REAX_RGEMM", "0") == "1"
_REAX_RG = {"tab": None, "g": None}


def _reax_rg_cfg(M):
    # fastest exact tiles on the 5090 (rgemm.py test): 32x32x64 up to M 511, 32x32x128 to 1099, then 64x32x128
    if M <= 511:
        return "rg_32x32x64_w2x2_s6_b4"
    return "rg_32x32x128_w2x2_s4_b4" if M < 1100 else "rg_64x32x128_w4x2_s4_b2"


def _reax_rg_derive(w):
    """Bool table over M: rgemm == torch.mm(out_dtype=fp32) byte for byte (this process's cuBLAS state), or False."""
    import sys as _sys
    if "/opt/engv2" not in _sys.path:
        _sys.path.insert(0, "/opt/engv2")
    from rgemm import RGemm
    g = RGemm(_reax_rg_os.environ.get("REAX_RGEMM_CUBIN", "/opt/engv2/rgemm.cubin"))
    mmax = int(_reax_rg_os.environ.get("REAX_RGEMM_MMAX", "16384"))
    gen = torch.Generator(device=w.device).manual_seed(2816)
    X = (torch.randn(mmax, w.shape[1], device=w.device, generator=gen) * 2).bfloat16()
    bad = torch.zeros(mmax + 1, dtype=torch.bool, device=w.device)
    bad[0] = True
    for M in range(1, mmax + 1):
        ref = torch.mm(X[:M], w.T, out_dtype=torch.float32)
        new = g(X[:M], w, _reax_rg_cfg(M))
        bad[M] = (ref.view(torch.int32) != new.view(torch.int32)).any()
    tab = (~bad).cpu().tolist()
    del X
    _REAX_RG["g"] = g
    return tab


@torch.library.custom_op("reax::router_mm", mutates_args=())
def _reax_router_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    tab = _REAX_RG["tab"]
    if tab is None:
        if torch.cuda.is_current_stream_capturing():
            return torch.mm(x, w.T, out_dtype=torch.float32)
        try:
            tab = _reax_rg_derive(w)
            n = sum(tab)
            logger.info("reax rgemm: router GEMM through rgemm at %d of %d M (byte-identical to torch.mm)%s", n,
                        len(tab) - 1, "" if n == len(tab) - 1 else
                        f"; torch.mm at M {[m for m, ok in enumerate(tab) if not ok and m][:12]}")
        except Exception as e:  # noqa: BLE001 - keep torch.mm
            logger.warning("reax rgemm: off (%r)", e)
            tab = False
        _REAX_RG["tab"] = tab
    M = x.shape[0]
    if tab is False or M >= len(tab) or not tab[M] or not x.is_contiguous() or not w.is_contiguous():
        return torch.mm(x, w.T, out_dtype=torch.float32)
    return _REAX_RG["g"](x, w, _reax_rg_cfg(M))


@_reax_router_mm.register_fake
def _(x, w):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)
'''


def install(path):
    src = open(path).read()
    assert src.count(OLD) == 1, f"gate_linear.py: tier-4 anchor found {src.count(OLD)} times"
    open(path, "w").write(src.replace(OLD, NEW) + ADD)
    print("router_op: installed")


def test():
    import os
    import time
    os.environ["REAX_RGEMM"] = "1"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ.setdefault("REAX_RGEMM_MMAX", "16384")
    import torch
    from vllm.model_executor.determinism import batch_invariant as bi
    bi.init_batch_invariance()
    import vllm.model_executor.layers.fused_moe.router.gate_linear as gl
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(128, 2816, device="cuda", generator=gen) * 0.02).bfloat16()
    t0 = time.time()
    tab = gl._reax_rg_derive(w)
    bad = [m for m, ok in enumerate(tab) if not ok and m]
    print(f"derive: {len(tab) - 1 - len(bad)} of {len(tab) - 1} M byte-identical in {time.time() - t0:.1f} s; "
          f"differ at {bad[:20]}{' ...' if len(bad) > 20 else ''}")
    g = gl._REAX_RG["g"]
    from triton.testing import do_bench_cudagraph
    for M in (64, 300, 600, 887, 1000, 1500, 3000):
        x = torch.randn(M, 2816, device="cuda", generator=gen).bfloat16()
        t_ref = do_bench_cudagraph(lambda: torch.mm(x, w.T, out_dtype=torch.float32), rep=100)
        t_new = do_bench_cudagraph(lambda: g(x, w, gl._reax_rg_cfg(M)), rep=100)
        print(f"M {M:5d}: torch.mm {t_ref * 1e3:5.1f} us -> rgemm {t_new * 1e3:5.1f} us ({gl._reax_rg_cfg(M)})")
    print("PASS" if not bad else "PARTIAL (torch.mm kept at the M listed)")


if __name__ == "__main__":
    if sys.argv[1] == "install":
        install(sys.argv[2])
    elif sys.argv[1] == "test":
        test()
