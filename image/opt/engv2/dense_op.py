import sys

OLD_OK = '''    M, K = a.shape
    N = b.shape[1]
    return M >= 1024 and (N, K) in _REAX_TD_CFG and a.is_contiguous() and b.t().is_contiguous()
'''
NEW_OK = '''    if _REAX_BYPASS[0]:
        return False
    K = a.shape[1]
    N = b.shape[1]
    return (N, K) in _REAX_TD_CFG and a.is_contiguous() and b.t().is_contiguous()
'''
OLD_TD = '''def _reax_td(a, b):
    w = b.t()
'''
NEW_TD = '''def _reax_td(a, b):
    return torch.ops.reax.dense_mm(a, b.t())


def _reax_td_eager(a, b):
    w = b.t()
'''
OP = '''

_REAX_BYPASS = [False]


@torch.library.custom_op("reax::dense_mm", mutates_args=())
def _reax_dense_mm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if a.shape[0] >= 1024:
        return _reax_td_eager(a, w.t())
    _REAX_BYPASS[0] = True
    try:
        return matmul_persistent(a, w.t())
    finally:
        _REAX_BYPASS[0] = False


@_reax_dense_mm.register_fake
def _reax_dense_mm_fake(a, w):
    return a.new_empty((a.shape[0], w.shape[0]))
'''

path = sys.argv[1]
src = open(path).read()
for old, new in ((OLD_OK, NEW_OK), (OLD_TD, NEW_TD)):
    assert src.count(old) == 1, f"hunk not found: {old[:40]!r}"
    src = src.replace(old, new)
open(path, "w").write(src + OP)
print("dense_op: reax::dense_mm installed")
