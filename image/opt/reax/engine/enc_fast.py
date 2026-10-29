"""REAX_ENC_FAST=1 (engv2 Blink, default off; installed by engv2_stream.make_scheduler): the rune-family text encode
(vllm_readout ReadoutEncoder._encode_rune) without the per-decision Jinja render and the transformers tokenizer wrapper;
same prompt string, same ids.

For [system, user] with a string user turn the chat template emits a fixed prefix (bos, the system turn with the
constant system text, the user turn header), the user content through `| trim` (Jinja's trim is str.strip()) and a
fixed suffix (end of turn, the model turn header, the empty thought channel). The reference's own _encode_rune and
build_rune_prompt run unchanged against a tokenizer shim: apply_chat_template returns prefix + user.strip() + suffix (one
prefix / suffix pair per system text, cut from a real render at install), and the call returns the fast tokenizer's
Rust encode, which is what tok() runs for one string with add_special_tokens=False when the tokenizer has no truncation
or padding. Anything else (another system text, other arguments, a non-string turn) goes to the real tokenizer.
Install compares the shim with the real template on edge-case user strings (whitespace at both ends, Jinja delimiters,
non-ASCII, line separators) and the fast encode with the stock one on probe questions of every type; any difference
leaves the stock encode in place.
"""
import functools
import sys

_KW = {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}  # what build_rune_prompt passes
_EDGE = ("", "a", " \n lead and trail \t\r\n", "{{ x }} {% if y %} {# z #} <|turn>", "đ ñ 中文 é  ", "\n\nx\n\n")


class _Tok:
    def __init__(self, tok, raw, parts):
        self.tok, self.raw, self.parts = tok, raw, parts

    def apply_chat_template(self, messages, **kw):
        if kw == _KW and len(messages) == 2:
            s, u = messages
            pair = self.parts.get(s.get("content")) if s.get("role") == "system" else None
            if pair is not None and u.get("role") == "user" and isinstance(u.get("content"), str):
                return pair[0] + u["content"].strip() + pair[1]
        return self.tok.apply_chat_template(messages, **kw)

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, str):
            return {"input_ids": self.raw.encode(text, add_special_tokens=add_special_tokens).ids}
        return self.tok(text, add_special_tokens=add_special_tokens)


class _Enc:
    """What ReadoutEncoder._encode_rune reads from self."""

    def __init__(self, enc, tok):
        self.cfg, self.codes, self._item, self.tok = enc.cfg, enc.codes, enc._item, tok


def _probes(n_max):
    st = {"ref": "0a1b2c3d", "text": "  spaced  ", "nested": {"k": [1, 2.5, None, True], "u": "đ中"}}
    many = n_max + 4  # refused with the same error (Blink: max_options 26)
    return [
        (st, {"type": "choice", "instructions": "Pick one.  ", "criteria": {"a": "first", "b": " second ", "c": "{{x}}"}}),
        (st, {"type": "score", "instructions": {"x": 1}, "criteria": ["low", "mid", "high", "very high", "max"]}),
        (st, {"type": "noul", "instructions": "Is it?", "criteria": {"true": "Yes it is", "false": "No"}}),
        (st, {"type": "noul", "instructions": "\n", "criteria": None}),
        ("raw string state\n", {"type": "choice", "instructions": "", "criteria": {str(i): f"o{i}" for i in range(n_max)}}),
        (st, {"type": "choice", "instructions": "x", "criteria": {str(i): f"o{i}" for i in range(many)}}),
        (st, {"type": "bogus"}),
    ]


def _run(fn, state, q):
    try:
        e = fn(state, q)
        return e.payload, e.n_tokens, e.n_options, e.aux
    except Exception as exc:  # noqa: BLE001 - the stock encode's refusals must match too
        return type(exc).__name__, str(exc)


def install(backend):
    enc = getattr(backend, "encoder", None)
    cfg = getattr(enc, "cfg", None)
    if cfg is None or cfg.family in ("quyet", "jeffan") or not hasattr(enc, "_encode_rune"):
        print("reax enc_fast: not a rune-family text encoder, stock encode kept", flush=True)
        return False
    vr = sys.modules[type(enc).__module__]
    tok = enc.tok
    raw = getattr(tok, "_tokenizer", None)
    if raw is None or raw.truncation is not None or raw.padding is not None:
        print("reax enc_fast: no plain fast tokenizer, stock encode kept", flush=True)
        return False
    mark = "REAX"
    parts = {}
    for system in (vr.RUNE_SYSTEM, vr.RUNE_SYSTEM_EXT):
        out = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": mark}], **_KW)
        if out.count(mark) != 1:
            print("reax enc_fast: template does not place the user text once, stock encode kept", flush=True)
            return False
        parts[system] = tuple(out.split(mark))
    shim = _Tok(tok, raw, parts)
    for system in parts:
        for user in _EDGE:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            if shim.apply_chat_template(msgs, **_KW) != tok.apply_chat_template(msgs, **_KW):
                print(f"reax enc_fast: template differs on {user!r}, stock encode kept", flush=True)
                return False
    fast = functools.partial(type(enc)._encode_rune, _Enc(enc, shim))
    for state, q in _probes(min(cfg.max_options, len(enc.codes))):
        if _run(enc.encode, state, q) != _run(fast, state, q):
            print(f"reax enc_fast: encode differs on a {q.get('type')} probe, stock encode kept", flush=True)
            return False
    enc._encode_rune = fast
    print("reax enc_fast: rune text encode without the Jinja render (probes identical)", flush=True)
    return True
