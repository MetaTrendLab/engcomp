import os
import sys

TF = "transformers/models/gemma4/modeling_gemma4.py"
VL = "vllm/model_executor/models/gemma4_mm.py"

ENC_OLD = "        _r = _reax_vpack_index1(mask, L)\n"
ENC_NEW = ("        _r = (_reax_vpack_index1p(enc, mask, L, pos, st) if _REAX_VPOOL1  # engv2 vis_pool1.py\n"
           "              else _reax_vpack_index1(mask, L))\n")
FLAG_ANCHOR = '_REAX_VPACK_SYNC1 = _reax_vpack_os.environ.get("REAX_VIS_PACK_SYNC1", "0") == "1"  # engv2 vis_pack1.py\n'
FLAG = '_REAX_VPOOL1 = _reax_vpack_os.environ.get("REAX_VIS_POOL1", "0") == "1"  # engv2 vis_pool1.py\n'
HELPER = '''

def _reax_vpool_counts(enc, pos, L):
    """engv2 vis_pool1.py: per image of a [B, L, 2] position tensor, (number of pooler slots hit, slots are a prefix)
    as device tensors, with HF Gemma4VisionPooler._avg_pool_by_positions' slot arithmetic; None if L does not pool."""
    k2 = enc.config.pooling_kernel_size ** 2
    length = L // k2
    if length < 1:
        return None
    k = int((L // length) ** 0.5)
    if k * k * length != L:
        return None
    cl = pos.clamp(min=0)
    max_x = cl[..., 0].max(dim=-1, keepdim=True)[0] + 1
    ki = torch.div(cl, k, rounding_mode="floor")
    ki = ki[..., 0] + (max_x // k) * ki[..., 1]
    inr = (ki >= 0) & (ki < length)  # out of range: HF's one_hot raises; that image keeps the boolean index
    hits = torch.zeros(pos.shape[0], length, dtype=torch.int32, device=pos.device)
    hits.scatter_add_(1, ki.clamp(0, length - 1).long(), inr.int())
    hit = hits > 0
    c = hit.sum(1)
    last = torch.where(hit, torch.arange(length, device=pos.device)[None], -1).amax(1)
    return c, (last + 1 == c) & inr.all(1)


def _reax_vpack_index1p(enc, mask, L, pos, st):
    """engv2 vis_pool1.py: _reax_vpack_index1 plus the pooler's per-image valid slot counts in the same read, stashed
    in st["pool"] = (position tensor pointer, shape, [count or None per image]) for vLLM's pooling loop."""
    B = mask.shape[0]
    lens = mask.sum(1)
    ok = (mask == (torch.arange(L, device=mask.device)[None] < lens[:, None])).all()
    pc = _reax_vpool_counts(enc, pos, L) if pos is not None and pos.dim() == 3 else None
    parts = [lens, ok.view(1).to(lens.dtype)]
    if pc is not None:
        parts += [pc[0].to(lens.dtype), pc[1].to(lens.dtype)]
    info = torch.cat(parts).tolist()
    if pc is not None:
        st["pool"] = (pos.data_ptr(), tuple(pos.shape),
                      [info[B + 1 + i] if info[2 * B + 1 + i] else None for i in range(B)])
    lens_l = info[:B]
    if not info[B] or min(lens_l) == 0 or min(lens_l) == L:
        return None
    T = sum(lens_l)
    start = torch.arange(B, device=mask.device) * L - F.pad(lens.cumsum(0), (1, 0))[:B]
    return lens, lens_l, torch.arange(T, device=mask.device) + torch.repeat_interleave(start, lens, output_size=T)
'''

VL_MAP_OLD = "        last_hidden_states_map: dict[int, torch.Tensor] = {}\n"
VL_MAP_NEW = VL_MAP_OLD + "        _reax_pool_c: dict[int, int] = {}  # engv2 vis_pool1.py\n"
VL_ENC_OLD = '''                with gpu_sync_allowed():
                    encoder_outputs = vt.encoder(
                        inputs_embeds=inputs_embeds,
                        attention_mask=~pad_tensor,
                        pixel_position_ids=pp_tensor,
                    )
                hidden_states = encoder_outputs.last_hidden_state
'''
VL_ENC_NEW = ('''                _reax_vpool_reset()  # engv2 vis_pool1.py
''' + VL_ENC_OLD + '''                _reax_vpool_take(pp_tensor, chunk_items, _reax_pool_c)
''')
VL_IDX_OLD = "                valid_states = pooled_states[valid_mask]\n"
VL_IDX_NEW = ('''                if orig_idx in _reax_pool_c:  # engv2 vis_pool1.py: the same rows, count read with the encoder's
                    valid_states = pooled_states[0, : _reax_pool_c[orig_idx]]
                else:
                    valid_states = pooled_states[valid_mask]
''')
VL_ADD = '''

# ---------------------------------------------------------------- engv2 vis_pool1.py: pooler count from the encoder's read
import os as _reax_vpool_os

_REAX_VPOOL1 = _reax_vpool_os.environ.get("REAX_VIS_POOL1", "0") == "1"
_REAX_VPOOL_SEEN = [False]


def _reax_vpool_state():
    import transformers.models.gemma4.modeling_gemma4 as _m
    return getattr(_m, "_REAX_VPACK", None)


def _reax_vpool_reset():
    if _REAX_VPOOL1:
        st = _reax_vpool_state()
        if st is not None:
            st["pool"] = None


def _reax_vpool_take(pp, items, out):
    """Per-image pooler counts stashed by the encoder call on this exact position tensor (else nothing: the boolean
    index stays)."""
    if not _REAX_VPOOL1:
        return
    st = _reax_vpool_state()
    p = st.get("pool") if st is not None else None
    if p is None or p[0] != pp.data_ptr() or p[1] != tuple(pp.shape) or len(p[2]) != len(items):
        return
    for i, (orig_idx, _, _) in enumerate(items):
        if p[2][i] is not None:
            out[orig_idx] = p[2][i]
    if not _REAX_VPOOL_SEEN[0] and out:
        _REAX_VPOOL_SEEN[0] = True
        print("reax vis_pool1: pooler counts from the encoder's read (REAX_VIS_POOL1)", flush=True)
'''


def patch(sp):
    tf, vl = os.path.join(sp, TF), os.path.join(sp, VL)
    s = open(tf).read()
    if "engv2 vis_pool1.py" not in s:
        assert s.count(ENC_OLD) == 1, "vis_pool1: vis_pack1's index call not found (apply after vis_pack1.py)"
        assert s.count(FLAG_ANCHOR) == 1, "vis_pool1: vis_pack1's flag not found"
        s = s.replace(ENC_OLD, ENC_NEW).replace(FLAG_ANCHOR, FLAG_ANCHOR + FLAG) + HELPER
        compile(s, tf, "exec")
        open(tf, "w").write(s)
    v = open(vl).read()
    if "engv2 vis_pool1.py" not in v:
        i = v.index("    def _process_image_input(")
        j = v.index("\n    def ", i + 1)
        body = v[i:j]
        for old in (VL_MAP_OLD, VL_ENC_OLD, VL_IDX_OLD):
            assert body.count(old) == 1, "vis_pool1: gemma4_mm hunk not found: " + old.strip().splitlines()[0]
        body = body.replace(VL_MAP_OLD, VL_MAP_NEW).replace(VL_ENC_OLD, VL_ENC_NEW).replace(VL_IDX_OLD, VL_IDX_NEW)
        v = v[:i] + body + v[j:] + VL_ADD
        compile(v, vl, "exec")
        open(vl, "w").write(v)
    print("vis_pool1: installed")


def test():
    import torch
    import torch.nn.functional as F
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler
    ns = {"torch": torch, "F": F}
    exec(HELPER, ns)
    counts = ns["_reax_vpool_counts"]

    class _Enc:
        config = Gemma4VisionConfig()

    cfg = _Enc.config
    pool = Gemma4VisionPooler(cfg).cuda()
    k = cfg.pooling_kernel_size
    L = 2520
    g = torch.Generator().manual_seed(4242)
    grids = [(w, h) for w in range(k, L + 1, k) for h in range(k, L + 1, k) if w * h <= L]
    grids = [grids[i] for i in torch.randperm(len(grids), generator=g)[:300].tolist()] + [(57, 42), (42, 57), (48, 48),
                                                                                           (60, 39), (3, 3)]
    n = pref = fall = 0
    for w, h in grids:
        for odd in (False, True):
            ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            pos = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
            if odd:
                if w < 2 * k:
                    continue
                pos = pos[pos[:, 0] < w - 1]
            valid = pos.shape[0]
            pos = torch.cat([pos, torch.full((L - valid, 2), -1)]).long()[None].cuda()
            hidden = torch.randn(1, L, cfg.hidden_size, generator=g).cuda()
            padding = (pos == -1).all(dim=-1)
            out, mask = pool(hidden_states=hidden, pixel_position_ids=pos, padding_positions=padding, output_length=L // (k * k))
            ref = out[mask]
            c, is_pref = counts(_Enc(), pos, L)
            c, is_pref = int(c[0]), bool(is_pref[0])
            n += 1
            assert c == int(mask.sum()), (w, h, odd)
            if is_pref:
                pref += 1
                assert torch.equal(out[0, :c], ref), (w, h, odd)
            else:
                fall += 1
                assert not torch.equal(mask[0, :c], torch.ones_like(mask[0, :c])) or bool(mask[0, c:].any()), (w, h, odd)
    print(f"vis_pool1: {n} position grids: counts equal to HF's mask everywhere; {pref} prefixes with identical rows, "
          f"{fall} non-prefix (boolean index kept)")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test()
    else:
        patch(sys.argv[1])
