import sys

OLD = '        algos = torch.load(os.environ.get("REAX_LT_ALGOS", "/opt/rune/lt_algos.pt"))\n'
NEW = '        algos = _reax_lt_derive(reax_lt)\n'
DERIVE = '''

def _reax_lt_derive(reax_lt):
    """(N, K) -> first heuristic cuBLASLt algorithm (no split-K) bit-identical to the Triton kernel at every probe M."""
    g = torch.Generator(device="cuda").manual_seed(4321)
    out = {}
    for (N, K) in _REAX_TD_CFG:
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
        xs = [torch.randn(M, K, device="cuda", generator=g).bfloat16() for M in (1024, 1537, 4099, 11170)]
        refs = [_reax_td_triton(x, w) for x in xs]
        for algo, _cfg in reax_lt.algos(4096, N, K, 24):
            try:
                if all(torch.equal(reax_lt.mm(x, w, algo), r) for x, r in zip(xs, refs)):
                    out[(N, K)] = algo
                    break
            except RuntimeError:
                continue
    return out
'''

path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "d1 dense hunk not found"
src = src.replace(OLD, NEW) + DERIVE
open(path, "w").write(src)
print("lt_derive: start-up derivation installed")
