import sys

OLD_INIT = '''        self.head_dtype = model_config.head_dtype if model_config is not None else None
'''
NEW_INIT = OLD_INIT + '''        self._reax_tok = getattr(model_config, "tokenizer", None) if model_config is not None else None
'''
OLD_FWD = '''        if self.logits_as_input:
            logits = hidden_states
        else:
'''
NEW_FWD = '''        if _REAX_HEAD_SUB and not self.logits_as_input and embedding_bias is None and not skip_gather:
            _sub = _reax_head_sub(self, lm_head, hidden_states)
            if _sub is not None:
                return _sub
''' + OLD_FWD
ADD = '''

# ---------------------------------------------------------------- engv2 Blink: lm_head on the option-code rows (head_sub.py)
import os as _reax_os

# unset = auto: Blink's head only (262144 x 2816; the code set is the Blink/Rune codebook, not e.g. Jeffan's); "1" forces
_REAX_HEAD_SUB_ENV = _reax_os.environ.get("REAX_HEAD_SUB", "auto")
_REAX_HEAD_SUB = _REAX_HEAD_SUB_ENV in ("1", "auto")
_REAX_HEAD_CHECKS = 3


def _reax_code_ids(tok_path):
    sys_path = __import__("sys").path
    if "/opt/reax" not in sys_path:
        sys_path.insert(0, "/opt/reax")
    from engine.backends.vllm_readout import codebook
    from transformers import AutoTokenizer
    return codebook(AutoTokenizer.from_pretrained(str(tok_path)), 255)[1]


def _reax_head_sub_init(self, lm_head, h):
    try:
        if _REAX_HEAD_SUB_ENV == "auto" and tuple(lm_head.weight.shape) != (262144, 2816):
            raise RuntimeError(f"auto: not the Blink head {tuple(lm_head.weight.shape)}")
        if (lm_head.tp_size != 1 or self.scale != 1.0 or self.head_dtype not in (None, h.dtype)
                or not isinstance(getattr(lm_head, "weight", None), torch.Tensor) or self._reax_tok is None):
            raise RuntimeError("unsupported head (tp, scale, head_dtype or no tokenizer)")
        ids = sorted(set(_reax_code_ids(self._reax_tok)))
        if not ids or max(ids) >= self.org_vocab_size:
            raise RuntimeError(f"bad code ids {ids[:4]}")
        w = lm_head.weight
        pad = (-len(ids)) % 16
        idx = torch.tensor(ids + [ids[-1]] * pad, device=w.device)
        st = {"ids": torch.tensor(ids, device=w.device), "w": w.index_select(0, idx).contiguous(), "n": len(ids),
              "checked": set()}
        logger.info("reax head_sub: lm_head on %d option-code rows of %d (ids %s ...)", len(ids), w.shape[0], ids[:3])
        return st
    except Exception as e:  # noqa: BLE001
        logger.warning("reax head_sub: off (%s)", e)
        return False


def _reax_head_rows(self, lm_head, h, st):
    stub = type("_ReaxHead", (), {})()
    stub.weight = st["w"]
    sub = lm_head.quant_method.apply(stub, h, bias=None)[..., : st["n"]]
    if self.soft_cap is not None:
        sub = sub / self.soft_cap
        sub = torch.tanh(sub)
        sub = sub * self.soft_cap
    return sub


def _reax_head_sub(self, lm_head, h):
    st = getattr(self, "_reax_hs", None)
    if st is None:
        st = self._reax_hs = _reax_head_sub_init(self, lm_head, h)
    if st is False:
        return None
    sub = _reax_head_rows(self, lm_head, h, st)
    M = h.shape[0] if h.dim() == 2 else -1
    if len(st["checked"]) < _REAX_HEAD_CHECKS and M not in st["checked"]:
        st["checked"].add(M)
        globals()["_REAX_HEAD_SUB"] = False  # the reference path, once
        try:
            full = self.forward(lm_head, h)
        finally:
            globals()["_REAX_HEAD_SUB"] = True
        ref = full.index_select(-1, st["ids"])
        if not torch.equal(ref, sub):
            logger.warning("reax head_sub: code columns differ from the full head at M %d: off", M)
            self._reax_hs = False
            return full
        logger.info("reax head_sub: M %d code columns bit-identical to the full head", M)
    out = torch.full((*h.shape[:-1], self.org_vocab_size), float("-inf"), dtype=sub.dtype, device=sub.device)
    out.index_copy_(-1, st["ids"], sub)
    return out
'''


def install(path):
    src = open(path).read()
    for old in (OLD_INIT, OLD_FWD):
        assert src.count(old) == 1, f"logits_processor.py: anchor not found: {old[:60]!r}"
    src = src.replace(OLD_INIT, NEW_INIT).replace(OLD_FWD, NEW_FWD) + ADD
    open(path, "w").write(src)
    print("head_sub: installed")


def test():
    import torch
    from vllm.model_executor.determinism import batch_invariant as bi
    g = torch.Generator(device="cuda").manual_seed(4325)
    V, K = 262144, 2816
    w = (torch.randn(V, K, device="cuda", generator=g) * 0.02).bfloat16()
    ids = sorted(torch.randperm(V, generator=torch.Generator().manual_seed(1))[:255].tolist())
    pad = (-len(ids)) % 16
    idx = torch.tensor(ids + [ids[-1]] * pad, device="cuda")
    ws = w.index_select(0, idx).contiguous()
    it = torch.tensor(ids, device="cuda")
    X = torch.randn(300, K, device="cuda", generator=g).bfloat16() * 3
    bad = []
    for M in list(range(1, 65)) + [65, 77, 128, 255, 300]:
        bi._REAX_BYPASS[0] = True
        try:
            full = bi.matmul_persistent(X[:M], w.t())
        finally:
            bi._REAX_BYPASS[0] = False
        head = bi.matmul_persistent(X[:M], w.t())
        sub = bi.matmul_persistent(X[:M], ws.t())[:, : len(ids)]
        ok = torch.equal(full.index_select(1, it), sub) and torch.equal(head.index_select(1, it), sub)
        if not ok:
            bad.append(M)
    print(f"head_sub test: {'PASS' if not bad else 'FAIL at M ' + str(bad[:8])} (255 code rows of {V}, M 1..64 + 5 larger)")
    return not bad


if __name__ == "__main__":
    if sys.argv[1] == "install":
        install(sys.argv[2])
    elif sys.argv[1] == "test":
        sys.exit(0 if test() else 1)
