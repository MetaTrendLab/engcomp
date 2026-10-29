import sys

OLD_ENC = '''        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # embed positions
        hidden_states = inputs_embeds
'''
NEW_ENC = '''        _packed = _reax_vpack_encoder(self, inputs_embeds, attention_mask, pixel_position_ids, kwargs)
        if _packed is not None:
            return BaseModelOutputWithPast(last_hidden_state=_packed)
''' + OLD_ENC
OLD_ATT = '''        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
'''
NEW_ATT = OLD_ATT + '''        if kwargs.get("reax_cu") is not None:
            attention_interface = _reax_vpack_attention
'''
ADD = '''

# ---------------------------------------------------------------- engv2 vision: tower on valid patches only, exact (vis_pack.py)
import os as _reax_vpack_os

_REAX_VPACK = {"on": None, "own": False, "var": 0, "lib": None}


def _reax_vattn_lib():
    st = _REAX_VPACK
    if st["lib"] is None:
        import ctypes as c
        lib = (c.PyDLL if _reax_vpack_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(_reax_vpack_os.environ.get("REAX_VIS_ATTN_LIB", "/opt/engv2/libreax_vattn.so"))
        lib.reax_vattn_fwd_v.restype = c.c_int
        lib.reax_vattn_fwd_v.argtypes = [c.c_int] + [c.c_void_p] * 5 + [c.c_int] * 4 + [c.c_longlong, c.c_float,
                                                                                         c.c_void_p]
        lib.reax_vattn_variants.restype = c.c_int
        lib.reax_vattn_variant_name.restype = c.c_char_p
        lib.reax_vattn_variant_name.argtypes = [c.c_int]
        st["lib"] = lib
    return st["lib"]


def _reax_vattn_variants():
    lib = _reax_vattn_lib()
    return [lib.reax_vattn_variant_name(i).decode() for i in range(lib.reax_vattn_variants())]


def _reax_vattn(q, k, v, cu, max_len, scale, var=None):
    """vis_attn.cu: the mem-efficient kernel's arithmetic with FA2 tiles; q, k, v [1, T, heads, 72] contiguous."""
    out = torch.empty_like(q)
    rc = _reax_vattn_lib().reax_vattn_fwd_v(_REAX_VPACK["var"] if var is None else var, out.data_ptr(), q.data_ptr(),
                                            k.data_ptr(), v.data_ptr(), cu.data_ptr(), cu.numel() - 1, max_len,
                                            q.shape[2], q.shape[3], q.stride(1), scale,
                                            torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_vattn_fwd: error {rc}")
    return out


def _reax_vpack_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, reax_cu=None,
                          reax_max=None, **kwargs):
    """The reference's SDPA kernel (mem-efficient, scale applied before the softmax) on the packed images, no bias;
    vis_attn.cu (same arithmetic) when the start-up check said so."""
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # [1, T, heads, dim]
    if (_REAX_VPACK["own"] and scaling is not None and q.shape[-1] == 72 and q.dtype == torch.bfloat16
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and q.shape == k.shape == v.shape):
        return _reax_vattn(q, k, v, reax_cu, reax_max, scaling), None
    out = torch.ops.aten._efficient_attention_forward(q, k, v, None, reax_cu, reax_cu, reax_max, reax_max, 0.0, 0, False,
                                                      scale=scaling)[0]
    return out, None


def _reax_vattn_check():
    """vis_attn.cu vs aten._efficient_attention_forward (packed, no bias) bit for bit: image counts and lengths of the
    vision tower, lengths around the 128-key blocks, wide magnitudes and very peaked rows; every variant. Returns the
    exact ones."""
    g = torch.Generator(device="cuda").manual_seed(7272)
    H, D = 16, 72
    names = _reax_vattn_variants()
    exact = set(range(len(names)))
    n = 0
    for valid, mag, kind in (((2394,), 1.0, 0), ((2394, 2340), 1.0, 0), ((2304, 2520, 2376), 3.0, 0),
                             ((1, 130, 127, 128, 129, 64), 1.0, 0), ((2394, 2304), 8.0, 0), ((2394,), 30.0, 0),
                             ((2340, 2394), 1.0, 1), ((2394,), 1.0, 2)):
        T = sum(valid)
        q, k, v = (torch.randn(1, T, H, D, device="cuda", generator=g) * mag for _ in range(3))
        if kind == 1:  # exponents 2^-8 .. 2^5 per element
            q = q * torch.exp2(torch.randint(-8, 6, (1, T, H, D), device="cuda", generator=g).float())
        if kind == 2:  # a few keys dominate
            k[:, ::97] *= 20
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        cu = F.pad(torch.tensor(valid, device="cuda").cumsum(0), (1, 0)).int()
        ref = torch.ops.aten._efficient_attention_forward(q, k, v, None, cu, cu, max(valid), max(valid), 0.0, 0, False,
                                                          scale=1.0)[0]
        for var in sorted(exact):
            out = _reax_vattn(q, k, v, cu, max(valid), 1.0, var)
            bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
            if bad:
                exact.discard(var)
                print(f"reax vis_attn: {names[var]} differs at valid {valid} mag {mag} kind {kind}: {bad} elements",
                      flush=True)
        n += 1
    print(f"reax vis_attn: {len(exact)} of {len(names)} variants bit-identical in {n} cases", flush=True)
    return sorted(exact)


def _reax_vattn_tune(exact, verbose=False):
    """Time every exact variant (L2-flushed, best of 2) on 1 and 2 images of the processor's sizes; fastest total."""
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(11)
    names = _reax_vattn_variants()
    cases = []
    for valid in ((2394,), (2394, 2340)):
        T = sum(valid)
        q, k, v = (torch.randn(1, T, 16, 72, device="cuda", generator=g).bfloat16() for _ in range(3))
        cases.append((q, k, v, F.pad(torch.tensor(valid, device="cuda").cumsum(0), (1, 0)).int(), max(valid)))
    t_ref = [min(do_bench(lambda: torch.ops.aten._efficient_attention_forward(
        q, k, v, None, cu, cu, mx, mx, 0.0, 0, False, scale=1.0)) for _ in range(2)) for q, k, v, cu, mx in cases]
    best, best_t = None, None
    for var in exact:
        tv = [min(do_bench(lambda: _reax_vattn(q, k, v, cu, mx, 1.0, var)) for _ in range(2)) for q, k, v, cu, mx in cases]
        if verbose:
            print(f"  {names[var]:8s} 1 image {tv[0] * 1e3:6.1f} us (aten {t_ref[0] * 1e3:6.1f}), 2 images "
                  f"{tv[1] * 1e3:6.1f} us (aten {t_ref[1] * 1e3:6.1f})", flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = var, tv
    return best, best_t, t_ref


def _reax_vpack_checks(verbose=False):
    """(pack exact, own attention kernel exact and faster, its fastest exact variant)."""
    ok = _reax_vpack_check()
    own, var = False, 0
    if ok and _reax_vpack_os.environ.get("REAX_VIS_ATTN", "1") == "1":
        try:
            exact = _reax_vattn_check()
            if exact:
                var, tv, tr = _reax_vattn_tune(exact, verbose)
                own = sum(tv) < sum(tr) / 1.02
                print(f"reax vis_attn: fastest exact {_reax_vattn_variants()[var]}: {tv[0] * 1e3:.1f} / "
                      f"{tv[1] * 1e3:.1f} us vs aten {tr[0] * 1e3:.1f} / {tr[1] * 1e3:.1f}", flush=True)
        except Exception as e:  # noqa: BLE001 - keep aten's kernel
            print(f"reax vis_attn: check failed ({e!r})", flush=True)
    return ok, own, var


def _reax_vpack_check():
    """aten._efficient_attention_forward packed (no bias) vs the HF SDPA call of the vision attention (padding mask),
    bit for bit, at the reference's 2520 patches and valid counts of the processor's aspect ratios."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cfg = Gemma4VisionConfig()
    cfg._attn_implementation = "sdpa"

    class _Mod:
        is_causal = False
        num_key_value_groups = 1
        training = False

    g = torch.Generator(device="cuda").manual_seed(9192)
    L, H, D = 2520, 16, 72
    n = 0
    for valid, mag in (((2394,), 1.0), ((2394, 2340), 1.0), ((2304, 2520, 2376), 3.0), ((2340, 2394, 2304, 2394), 0.3),
                       ((1, 2520), 1.0), ((130, 2394), 8.0)):
        B = len(valid)
        q, k, v = (torch.randn(B, L, H, D, device="cuda", generator=g).mul(mag).bfloat16() for _ in range(3))
        pad = torch.zeros(B, L, dtype=torch.bool, device="cuda")
        for i, m in enumerate(valid):
            pad[i, m:] = True
        mask = create_bidirectional_mask(config=cfg, inputs_embeds=q.new_empty(B, L, 8), attention_mask=~pad)
        ref = sdpa_attention_forward(_Mod(), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask,
                                     dropout=0.0, scaling=1.0)[0]
        idx = (~pad).reshape(-1).nonzero().squeeze(1)
        qp, kp, vp = (x.reshape(B * L, H, D).index_select(0, idx)[None].transpose(1, 2) for x in (q, k, v))
        cu = F.pad(torch.tensor(valid, device="cuda").cumsum(0), (1, 0)).int()
        out = _reax_vpack_attention(_Mod(), qp, kp, vp, None, scaling=1.0, reax_cu=cu, reax_max=max(valid))[0][0]
        for i, m in enumerate(valid):
            a, b = out[cu[i]:cu[i + 1]], ref[i, :m]
            if not torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)):
                print(f"reax vis_pack: differs at valid {valid} mag {mag} image {i}: "
                      f"{int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())} elements",
                      flush=True)
                return False
        n += 1
    print(f"reax vis_pack: bit-identical in {n} cases", flush=True)
    return True


def _reax_vpack_encoder(enc, x, mask, pos, kwargs):
    """Gemma4VisionEncoder.forward on the valid patches; None -> the reference path."""
    st = _REAX_VPACK
    if st["on"] is None:
        if not x.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        st["on"] = False
        if _reax_vpack_os.environ.get("REAX_VIS_PACK", "1") == "1":
            path = _reax_vpack_os.environ.get("REAX_VIS_PACK_FILE", "")
            try:
                if path and _reax_vpack_os.path.exists(path):
                    flags = open(path).read().split()
                    st["on"], st["own"] = flags[:1] == ["1"], flags[1:2] == ["1"]
                    st["var"] = int(flags[2]) if len(flags) > 2 else 0
                    how = f"loaded from {path}"
                else:
                    st["on"], st["own"], st["var"] = _reax_vpack_checks()
                    how = "checked in-process"
            except Exception as e:  # noqa: BLE001 - keep the reference path
                st["on"] = st["own"] = False
                how = f"disabled ({e!r})"
            st["own"] = st["on"] and st["own"] and _reax_vpack_os.environ.get("REAX_VIS_ATTN", "1") == "1"
            print(f"reax vis_pack: {'on' if st['on'] else 'off'}, attention "
                  f"{'vis_attn variant %d' % st['var'] if st['own'] else 'aten'} ({how})", flush=True)
    if (not st["on"] or kwargs or torch.cuda.is_current_stream_capturing() or x.dim() != 3 or pos is None
            or mask is None or mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:2])):
        return None
    B, L, C = x.shape
    lens = mask.sum(1)
    # only tail padding (the processor pads at the end), and some padding: without any, HF builds no mask and SDPA
    # runs another kernel
    if not torch.equal(mask, torch.arange(L, device=x.device)[None] < lens[:, None]):
        return None
    lens_l = lens.tolist()
    if min(lens_l) == 0 or min(lens_l) == L:
        return None
    idx = mask.reshape(-1).nonzero().squeeze(1)
    h = x.reshape(B * L, C).index_select(0, idx)[None]
    p = pos.reshape(B * L, pos.shape[-1]).index_select(0, idx)[None]
    cu = F.pad(lens.cumsum(0), (1, 0)).int()
    pe = enc.rotary_emb(h, p)
    for layer in enc.layers[: enc.config.num_hidden_layers]:
        h = layer(h, attention_mask=None, position_embeddings=pe, position_ids=p, reax_cu=cu, reax_max=max(lens_l))
    out = x.new_zeros(B * L, C)
    out.index_copy_(0, idx, h[0])
    return out.view(B, L, C)
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as mg
    if sys.argv[1] == "derive":
        ok = own = False
        var = 0
        try:
            ok, own, var = mg._reax_vpack_checks()
        except Exception as e:
            print(f"reax vis_pack: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write(f"{int(ok)} {int(own)} {var}\n")
        sys.exit(0)
    print("checks (pack, vis_attn, variant)", mg._reax_vpack_checks(verbose=True), flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("class Gemma4VisionEncoder(")
j = src.index(OLD_ENC, i)
assert j < src.index("\nclass ", i + 1), "Gemma4VisionEncoder.forward body not found"
src = src[:j] + NEW_ENC + src[j + len(OLD_ENC):]
i = src.index("class Gemma4VisionAttention(")
j = src.index(OLD_ATT, i)
assert j < src.index("\nclass ", i + 1), "Gemma4VisionAttention attention interface not found"
src = src[:j] + NEW_ATT + src[j + len(OLD_ATT):] + ADD
open(path, "w").write(src)
print("vis_pack: installed")
