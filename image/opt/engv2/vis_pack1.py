import sys

OLD = '''    B, L, C = x.shape
    lens = mask.sum(1)
    # only tail padding (the processor pads at the end), and some padding: without any, HF builds no mask and SDPA
    # runs another kernel
    if not torch.equal(mask, torch.arange(L, device=x.device)[None] < lens[:, None]):
        return None
    lens_l = lens.tolist()
    if min(lens_l) == 0 or min(lens_l) == L:
        return None
    idx = mask.reshape(-1).nonzero().squeeze(1)
'''
NEW = '''    B, L, C = x.shape
    if _REAX_VPACK_SYNC1:  # engv2 vis_pack1.py
        if not st.get("sync1"):
            st["sync1"] = True
            print("reax vis_pack: one device-to-host read per encoder call (REAX_VIS_PACK_SYNC1)", flush=True)
        _r = _reax_vpack_index1(mask, L)
        if _r is None:
            return None
        lens, lens_l, idx = _r
    else:
''' + "".join("    " + ln + "\n" for ln in OLD.splitlines()[1:])
FLAG_ANCHOR = '_REAX_VPACK = {"on": None, "own": False, "var": 0, "lib": None}\n'
HELPER = '''

def _reax_vpack_index1(mask, L):
    """engv2 vis_pack1.py: (lens, host lengths, packed row indices) of a tail-padded [B, L] mask with one
    device-to-host read, or None (no padding / an empty image / padding not at the tail: the reference path)."""
    B = mask.shape[0]
    lens = mask.sum(1)
    ok = (mask == (torch.arange(L, device=mask.device)[None] < lens[:, None])).all()
    info = torch.cat([lens, ok.view(1).to(lens.dtype)]).tolist()
    lens_l = info[:B]
    if not info[B] or min(lens_l) == 0 or min(lens_l) == L:
        return None
    T = sum(lens_l)
    # packed row j of image b (cu[b] <= j < cu[b + 1]) is row b * L + (j - cu[b]) of the padded batch
    start = torch.arange(B, device=mask.device) * L - F.pad(lens.cumsum(0), (1, 0))[:B]
    return lens, lens_l, torch.arange(T, device=mask.device) + torch.repeat_interleave(start, lens, output_size=T)
'''


def patch(path):
    s = open(path).read()
    if "engv2 vis_pack1.py" in s:
        print("vis_pack1: already patched")
        return
    assert s.count(OLD) == 1, "vis_pack1: vis_pack's encoder block not found (apply after vis_pack.py)"
    assert s.count(FLAG_ANCHOR) == 1, "vis_pack1: vis_pack's state anchor not found"
    s = s.replace(OLD, NEW).replace(FLAG_ANCHOR, FLAG_ANCHOR + '_REAX_VPACK_SYNC1 = _reax_vpack_os.environ.get('
                                    '"REAX_VIS_PACK_SYNC1", "0") == "1"  # engv2 vis_pack1.py\n') + HELPER
    compile(s, path, "exec")
    open(path, "w").write(s)
    print("vis_pack1: installed")


def test():
    import warnings
    import torch
    import torch.nn.functional as F
    ns = {"torch": torch, "F": F}
    exec(HELPER, ns)
    new = ns["_reax_vpack_index1"]

    def old(mask, L):
        lens = mask.sum(1)
        if not torch.equal(mask, torch.arange(L, device=mask.device)[None] < lens[:, None]):
            return None
        lens_l = lens.tolist()
        if min(lens_l) == 0 or min(lens_l) == L:
            return None
        return lens, lens_l, mask.reshape(-1).nonzero().squeeze(1)

    def reads(fn, *a):
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = fn(*a)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return r, sum("synchroniz" in str(x.message) for x in w)

    g = torch.Generator().manual_seed(5150)
    L = 2520
    cases = [[2394], [2394, 2340], [2304, 2520, 2376], [2340, 2394, 2304, 2394], [1, 2520], [130, 2394], [2520],
             [2520, 2520], [0, 2394], [2394, 0]]
    cases += [torch.randint(1, L + 1, (int(torch.randint(1, 9, (1,), generator=g)),), generator=g).tolist()
              for _ in range(200)]
    n = fall = 0
    rd_old = rd_new = 0
    for valid in cases:
        for holes in (False, True):
            mask = torch.zeros(len(valid), L, dtype=torch.bool)
            for i, m in enumerate(valid):
                mask[i, :m] = True
            if holes:
                if max(valid) < 3:
                    continue
                i = max(range(len(valid)), key=lambda k: valid[k])
                mask[i, 1] = False
            mask = mask.cuda()
            a, ra = reads(old, mask, L)
            b, rb = reads(new, mask, L)
            n += 1
            if a is None or b is None:
                assert a is None and b is None, (valid, holes, a is None, b is None)
                fall += 1
                continue
            rd_old, rd_new = max(rd_old, ra), max(rd_new, rb)
            assert a[1] == b[1] and torch.equal(a[0], b[0]) and a[2].dtype == b[2].dtype and torch.equal(a[2], b[2]), \
                (valid, holes)
    print(f"vis_pack1: {n} masks, {fall} fall back in both, the rest identical (lens, host lengths, indices); "
          f"device-to-host reads per call: vis_pack {rd_old}, vis_pack1 {rd_new}")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test()
    else:
        patch(sys.argv[1])
