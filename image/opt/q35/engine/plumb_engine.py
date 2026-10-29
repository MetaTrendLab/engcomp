from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("FLA_TENSOR_CACHE_SIZE", "16")
from pplx_engine import PplxEngine

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LETTERS)
GRAPH_SHAPES = ([(1, n) for n in (128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)]
                + [(2, n) for n in (128, 192, 256, 384, 512)] + [(4, n) for n in (128, 192, 256, 384)]
                + [(8, n) for n in (128, 192, 256)])
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
PREFIX = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def decision_options(question: dict) -> list[tuple[str, str]]:
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def prompt_text(state, criterion, options: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return PREFIX + json.dumps(payload, ensure_ascii=False).strip() + SUFFIX


def pin_fla_autotune():
    import triton
    from fla.modules import l2norm
    k = l2norm.l2norm_fwd_kernel
    k.configs = [triton.Config({"BT": 16}, num_warps=4)]
    k.cache.clear()


class PlumbEngine(PplxEngine):
    HOST_CU_DEFAULT = "1"

    def __init__(self, model_dir, exact=None, **kw):
        self.exact = (os.environ.get("PLUMB_MODE", "exact") == "exact") if exact is None else exact
        if self.exact:
            kw.setdefault("fuse", False)
            kw.setdefault("conv_impl", "torch")
            kw.setdefault("attn_impl", "sdpa_cmask")
            kw.setdefault("prune_last", False)
            kw["compile"] = False
        pin_fla_autotune()
        if not self.exact:
            kw.setdefault("keep_full_last", True)
        kw.setdefault("causal", True)
        kw.setdefault("gva_native", os.environ.get("PLUMB_GVA", "1"))
        self.letter_ids = None
        self.graphs = {}
        self.runner = None
        self.graph_waste = float(os.environ.get("PLUMB_GRAPH_WASTE", "1.5"))
        super().__init__(model_dir, **kw)

    def _read_weights(self, device):
        from safetensors import safe_open
        from tokenizers import Tokenizer
        self.temperature = float(json.loads((self.dir / "jevk5_config.json").read_text())["temperature"])
        tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        slots = [tok.encode(c, add_special_tokens=False).ids for c in LETTERS]
        assert all(len(s) == 1 for s in slots), "every answer letter must be one token"
        self.letter_ids = [s[0] for s in slots]
        pre = "model.language_model."
        W = {}
        with safe_open(str(self.dir / "model.safetensors"), framework="pt", device=device) as sf:
            for k in sf.keys():
                if k.startswith(pre):
                    W[k[len(pre):]] = sf.get_tensor(k)
        self.embed = W.pop("embed_tokens.weight")
        self.norm_w = W.pop("norm.weight")
        self.readout = self.embed[torch.tensor(self.letter_ids, device=device)].to(self.dtype).contiguous()
        return W

    @torch.inference_mode()
    def letter_logits(self, ids: torch.Tensor, lens: list[int]) -> torch.Tensor:
        return self.forward(ids, lens)

    @torch.inference_mode()
    def logits(self, arrays: list[np.ndarray]) -> np.ndarray:
        if self.runner is not None:
            return self.runner.logits(arrays)
        lens = [len(a) for a in arrays]
        g = self._pick_graph(lens)
        if g is not None:
            return g.run(arrays, lens)
        ids = torch.from_numpy(np.concatenate(arrays)).to(self.device, non_blocking=True)
        return self.letter_logits(ids, lens).cpu().numpy()

    def _pick_graph(self, lens):
        if not self.graphs:
            return None
        B, n, tot = len(lens), max(lens), sum(lens)
        best = None
        for (gb, gn), g in self.graphs.items():
            if gb >= B and gn >= n and gb * gn <= self.graph_waste * tot + 64 and (best is None or gb * gn < best[0]):
                best = (gb * gn, g)
        return best[1] if best else None

    def start_runner(self, n_streams=None, buckets=None):
        r = ExactRunner(self, n_streams or int(os.environ.get("PLUMB_STREAMS", "4")))
        r.capture(buckets)
        self.runner = r

    @torch.inference_mode()
    def capture_graphs(self, shapes):
        pool = torch.cuda.graph_pool_handle()
        for B, n in shapes:
            if (B, n) not in self.graphs:
                self.graphs[(B, n)] = _Graph(self, B, n, pool)


def _fla_cache_entries(*keys):
    import collections
    import sys
    found = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("fla") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            for cell in getattr(obj, "__closure__", None) or ():
                try:
                    c = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(c, collections.deque):
                    found += [e for e in c if isinstance(e, tuple) and any(a is k for a in (*e[0], *e[1].values()) for k in keys)]
    return found


class _Graph:

    def __init__(self, eng: PlumbEngine, B: int, n: int, pool, stream=None):
        self.eng, self.B, self.n = eng, B, n
        dev = eng.device
        self.ids = torch.zeros(B * n, dtype=torch.int64, device=dev)
        self.last = torch.arange(B, device=dev, dtype=torch.int64) * n + (n - 1)
        cu = torch.arange(B + 1, device=dev, dtype=torch.int32) * n
        offs = list(range(0, (B + 1) * n, n))
        cos, sin = eng._rope_tables([n] * B, dev)
        self.host_ids = torch.zeros(B * n, dtype=torch.int64).pin_memory()
        self.host_last = torch.zeros(B, dtype=torch.int64).pin_memory()

        cu64 = cu.long()
        cu_cpu = torch.tensor(offs, dtype=torch.int64)

        def fwd():
            return eng._drive([eng._core_gen(self.ids, cu, cu64, offs, n, self.last, cos, sin, cu_cpu, prune=False)], [None])[0]
        stream = stream or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fwd()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._keep = (cu, cu64, cu_cpu, cos, sin, offs, _fla_cache_entries(cu64, cu_cpu))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.out = fwd()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def run(self, arrays, lens):
        n, hi, hl = self.n, self.host_ids.numpy(), self.host_last.numpy()
        hi[:] = 0
        for b, a in enumerate(arrays):
            hi[b * n: b * n + len(a)] = a
            hl[b] = b * n + len(a) - 1
        for b in range(len(arrays), self.B):
            hl[b] = b * n + n - 1
        self.ids.copy_(self.host_ids, non_blocking=True)
        self.last.copy_(self.host_last, non_blocking=True)
        self.graph.replay()
        return self.out[: len(arrays)].cpu().numpy()


REF_BUCKETS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)


class ExactRunner:

    def __init__(self, eng: PlumbEngine, n_streams: int, buckets=REF_BUCKETS, max_items=256):
        self.eng, self.buckets = eng, tuple(buckets)
        self.streams = [torch.cuda.Stream() for _ in range(n_streams)]
        self.pools = [torch.cuda.graph_pool_handle() for _ in range(n_streams)]
        self.graphs: dict = {}
        self.max_items = max_items
        self.stage_ids = torch.zeros(max_items, self.buckets[-1], dtype=torch.int64).pin_memory()
        self.stage_last = torch.zeros(max_items, dtype=torch.int64).pin_memory()
        self.stage_out = torch.zeros(max_items, MAX_OPTIONS, dtype=torch.float32).pin_memory()

    @torch.inference_mode()
    def capture(self, buckets=None):
        for si, st in enumerate(self.streams):
            for n in buckets or self.buckets:
                if (si, n) not in self.graphs:
                    self.graphs[(si, n)] = _Graph(self.eng, 1, n, self.pools[si], stream=st)

    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if b >= n:
                return b
        raise ValueError("input too long")

    @torch.inference_mode()
    def logits(self, arrays):
        N = len(arrays)
        out = np.empty((N, MAX_OPTIONS), dtype=np.float32)
        for lo in range(0, N, self.max_items):
            self._chunk(arrays[lo: lo + self.max_items], out[lo: lo + self.max_items])
        return out

    def _chunk(self, arrays, out):
        S = len(self.streams)
        load = [0] * S
        used = [False] * S
        ids_np, last_np = self.stage_ids.numpy(), self.stage_last.numpy()
        order = sorted(range(len(arrays)), key=lambda i: -len(arrays[i]))
        for k, i in enumerate(order):
            a = arrays[i]
            n = self.bucket(len(a))
            si = min(range(S), key=load.__getitem__)
            load[si] += n
            used[si] = True
            g = self.graphs[(si, n)]
            ids_np[k, : len(a)] = a
            ids_np[k, len(a): n] = 0
            last_np[k] = len(a) - 1
            with torch.cuda.stream(self.streams[si]):
                g.ids.copy_(self.stage_ids[k, :n], non_blocking=True)
                g.last.copy_(self.stage_last[k: k + 1], non_blocking=True)
                g.graph.replay()
                self.stage_out[k].copy_(g.out[0], non_blocking=True)
        for si, st in enumerate(self.streams):
            if used[si]:
                st.synchronize()
        for k, i in enumerate(order):
            out[i] = self.stage_out[k].numpy()


class PlumbBackend:

    name = "plumb-4b"
    kind = "native-gpu"
    encode_inline = True

    def __init__(self, model_dir=None, max_input_tokens=4096, max_batch_tokens=None, device="cuda", **engine_kw):
        from backends.base import Encoded
        from tokenizers import Tokenizer
        self._Encoded = Encoded
        self.dir = Path(model_dir or os.environ.get("PLUMB_MODEL_DIR", "/model"))
        self.max_input_tokens = max_input_tokens
        self.max_batch_tokens = int(max_batch_tokens or os.environ.get("PLUMB_BATCH_TOKENS", 16384))
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.engine = PlumbEngine(self.dir, device=device, **engine_kw)
        self.temperature = self.engine.temperature
        if not 2.06 < self.temperature < 2.08:
            raise SystemExit("Plumb calibration temperature does not match the pinned release")
        self.device = device

    def options(self, question: dict) -> list[str]:
        return [k for k, _ in decision_options(question)]

    def encode(self, state, question: dict):
        if question.get("type") not in ("choice", "noul", "score"):
            raise ValueError("bad question type")
        try:
            opts = decision_options(question)
        except (AttributeError, TypeError, KeyError) as exc:
            raise ValueError(f"bad criteria: {type(exc).__name__}") from None
        if not 1 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"a question takes 1 to {MAX_OPTIONS} options")
        text = prompt_text(state, question.get("instructions", ""), [t for _, t in opts])
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"input exceeds {self.max_input_tokens} model tokens")
        return self._Encoded(payload=np.asarray(ids, dtype=np.int64), n_tokens=len(ids), n_options=len(opts))

    def run(self, batch):
        logits = self.engine.logits([e.payload for e in batch])
        out = []
        for row, e in zip(logits, batch):
            r = row[: e.n_options] / self.temperature
            x = np.exp(r - r.max())
            out.append((x / x.sum()).tolist())
        return out

    def warmup(self, log=print):
        import random
        rng = random.Random(7)
        E = self._Encoded

        def item(n):
            return E(payload=np.ones(n, dtype=np.int64), n_tokens=n, n_options=rng.randint(2, 8))
        if self.engine.exact:
            self.engine.start_runner(buckets=[b for b in REF_BUCKETS if b <= self.max_input_tokens])
        elif os.environ.get("PLUMB_GRAPHS", "1") == "1":
            self.engine.capture_graphs(GRAPH_SHAPES)
        shapes = [[n] for n in (17, 64, 192, 300, 700, 1500, 3000, 4096)]
        shapes += [[200] * k for k in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)]
        for lens in shapes:
            for _ in range(2):
                self.run([item(n) for n in lens])
        for _ in range(int(os.environ.get("PLUMB_WARM_STEPS", "60"))):
            n = rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
            lens = [int(min(4096, max(16, rng.lognormvariate(5.2, 0.6)))) for _ in range(n)]
            while sum(lens) > self.max_batch_tokens:
                lens.pop()
            self.run([item(m) for m in lens or [64]])
        torch.cuda.synchronize()
