from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gguf_readouts as readouts
from .base import Encoded


@dataclass(frozen=True)
class GGUFSpec:
    name: str
    hf_repo: str
    revision: str
    file: str
    sha256: str
    readout: str
    temperature: Any
    licence: str
    max_input_tokens: int = 4096
    min_options: int = 2
    max_options: int = 16
    max_state_bytes: int = 16384
    extra: dict = field(default_factory=dict)


SPECS: dict[str, GGUFSpec] = {
    "gutsy-0.8b": GGUFSpec(
        name="gutsy-0.8b", hf_repo="kouhxp/gutsy", revision="f9995fe26a337f1a4ea3f1a5ed38c9799e5a8f63",
        file="gutsy-0.8b-v04-q8_0.gguf", sha256="d71d7a3da4bb05fd7e26802c294fa08c937fb57c0d0fc75ea91ccc7231e708ee",
        readout="gutsy", licence="Apache-2.0",
        temperature={"choice": 1.0627611560799335, "score": 1.0, "yes_no": 0.765182131376029}),
    "plumb-4b-q8": GGUFSpec(
        name="plumb-4b-q8", hf_repo="crh225/plumb-4b-GGUF", revision="71c1c57e523219124f7be9c486d475d19891185f",
        file="plumb-4b-v5-Q8_0.gguf", sha256="a0d2974e04541b79e167d8acabfb7a7e4b4f93f3b47ac4a006d196d240f68177",
        readout="plumb", licence="see model card", temperature=2.07),
}


def physical_cores() -> int:
    try:
        seen = set()
        for cpu in sorted(os.sched_getaffinity(0)):
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            seen.add(base.read_text().strip() if base.exists() else str(cpu))
        return max(1, len(seen))
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


class GGUFCPUBackend:
    kind = "gguf-cpu"

    def __init__(self, model_dir: str, spec: GGUFSpec | str = "gutsy-0.8b", contexts: int | None = None,
                 threads: int | None = None, n_batch: int | None = None, n_ubatch: int | None = None):
        import llama_cpp
        from llama_cpp import Llama

        self.spec = SPECS[spec] if isinstance(spec, str) else spec
        self.name = self.spec.name
        self.max_input_tokens = self.spec.max_input_tokens
        self.contexts = contexts or int(os.environ.get("GGUF_CONTEXTS", "1"))
        self.threads = threads or int(os.environ.get("GGUF_THREADS", "0")) or max(1, physical_cores() // self.contexts)
        n_batch = n_batch or int(os.environ.get("GGUF_N_BATCH", "512"))
        n_ubatch = n_ubatch or int(os.environ.get("GGUF_N_UBATCH", "0")) or None
        self.max_batch_tokens = self.max_input_tokens * self.contexts
        path = str(Path(model_dir) / self.spec.file)
        self._lib = llama_cpp
        self._vocab = Llama(model_path=path, vocab_only=True, verbose=False)
        self._n_vocab = self._vocab.n_vocab()
        self.letter_ids: dict[str, int] = {}
        for letter in readouts.LETTERS + readouts.GUTSY_REJECT_LETTER:
            ids = self._vocab.tokenize(letter.encode(), add_bos=False, special=False)
            if len(ids) != 1:
                raise RuntimeError(f"answer letter {letter!r} is not one token")
            self.letter_ids[letter] = ids[0]
        kw = dict(model_path=path, n_ctx=self.max_input_tokens + 64, n_batch=n_batch,
                  n_threads=self.threads, n_threads_batch=self.threads, n_gpu_layers=0, use_mmap=True,
                  seed=0, logits_all=False, verbose=False)
        if n_ubatch:
            kw["n_ubatch"] = n_ubatch
        self._pool: queue.SimpleQueue = queue.SimpleQueue()
        for _ in range(self.contexts):
            self._pool.put(Llama(**kw))
        self._exec = ThreadPoolExecutor(max_workers=self.contexts, thread_name_prefix="gguf")
        self._tok_lock = threading.Lock()

    def options(self, question: dict) -> list[str]:
        return readouts.build(self.spec.readout, "", question)[1]

    def encode(self, state, question: dict) -> Encoded:
        kind = question.get("type")
        if kind not in ("choice", "noul", "score"):
            raise ValueError("bad question type")
        question = dict(question)
        question.setdefault("instructions", "")
        try:
            prompt, keys, letters = readouts.build(self.spec.readout, state, question)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"bad question: {type(exc).__name__}") from exc
        if not self.spec.min_options <= len(keys) <= self.spec.max_options:
            raise ValueError(f"{len(keys)} options (allowed {self.spec.min_options}-{self.spec.max_options})")
        with self._tok_lock:
            ids = list(self._vocab.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"input exceeds {self.max_input_tokens} model tokens")
        temp = readouts.temperature({"temperature": self.spec.temperature}, kind)
        return Encoded(payload=(ids, keys, letters, temp), n_tokens=len(ids), n_options=len(keys))

    def run(self, batch: list[Encoded]) -> list[list[float]]:
        if len(batch) == 1:
            return [self._one(batch[0].payload)]
        return list(self._exec.map(lambda e: self._one(e.payload), batch))

    def warmup(self) -> None:
        q = {"type": "noul", "instructions": "Is it a test?"}
        enc = self.encode("warm-up", q)
        for _ in range(self.contexts):
            self.run([enc] * self.contexts)

    def _one(self, payload) -> list[float]:
        ids, keys, letters, temp = payload
        llm = self._pool.get()
        try:
            logits = self._letter_logits(llm, ids, letters)
        finally:
            self._pool.put(llm)
        probs = readouts.probabilities(logits, keys, temp)
        return [probs[k] for k in keys]

    def _letter_logits(self, llm, ids: list[int], letters: list[str]) -> list[float]:
        import numpy as np

        llm.reset()
        llm.eval(ids)
        ctx = getattr(llm, "ctx", None)
        ctx = ctx if ctx is not None else llm._ctx.ctx
        ptr = self._lib.llama_get_logits_ith(ctx, -1)
        row = np.ctypeslib.as_array(ptr, shape=(self._n_vocab,))
        return [float(row[self.letter_ids[x]]) for x in letters]

    def answer(self, question: dict, probs: list[float]) -> dict:
        keys = self.options(question)
        return readouts.answer(question["type"], dict(zip(keys, probs)))


def load(name: str, model_dir: str, **kw) -> GGUFCPUBackend:
    return GGUFCPUBackend(model_dir, name, **kw)
