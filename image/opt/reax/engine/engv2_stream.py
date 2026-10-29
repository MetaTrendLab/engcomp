from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _deliver(fut, val, err):
    if not fut.done():
        if err is not None:
            fut.set_exception(err)
        else:
            fut.set_result(val)


class StreamScheduler:
    def __init__(self, backend, workers: int):
        self.backend = backend
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="vllm-req")
        self.steps = self.decisions = self.tokens = 0
        self._on_ready = None
        self.thread = threading.Thread(target=self._warm, name="warm-up", daemon=True)

    def start(self, on_ready=None):
        self._on_ready = on_ready
        self.thread.start()

    def _warm(self):
        if self._on_ready:
            self._on_ready()

    def submit(self, enc, loop, fut):
        self.steps += 1
        self.decisions += 1
        self.tokens += enc.n_tokens

        def done(f):
            err = f.exception()
            loop.call_soon_threadsafe(_deliver, fut, None if err else f.result(), err)

        self.pool.submit(self.backend._one, enc).add_done_callback(done)


class WaveScheduler(StreamScheduler):
    def __init__(self, backend, workers: int, hold_ms: float):
        super().__init__(backend, workers)
        self.hold = hold_ms / 1000.0
        self.softmax_t = sys.modules[type(backend).__module__].softmax_t
        self.cv = threading.Condition()
        self.pending = []
        self.inflight = self.wave_n = self.expect = 0
        self.t_free = 0.0
        self.stats = [0, 0, 0, 0.0, 0.0]
        self.log_every = int(os.environ.get("REAX_WAVE_LOG_EVERY", "256"))
        self.t_sent = 0.0
        self.trace = open(os.environ["REAX_WAVE_TRACE"], "a", buffering=1) if os.environ.get("REAX_WAVE_TRACE") else None
        self.dumped = 0
        self.gid, self.group_tag = 0, os.environ.get("REAX_GROUP_TAG", "0") == "1"
        self.one_req = os.environ.get("REAX_ONE_REQ", "0") == "1"
        self.disp = threading.Thread(target=self._dispatch, name="wave", daemon=True)

    def start(self, on_ready=None):
        super().start(on_ready)
        self.disp.start()

    def submit(self, enc, loop, fut):
        if self.trace:
            enc._t_sub = time.perf_counter()
        with self.cv:
            self.pending.append((enc, loop, fut))
            self.cv.notify()

    def _ready(self):
        if not self.pending or self.inflight:
            return False, False, None
        if len(self.pending) >= self.expect:
            return True, False, None
        left = self.t_free + self.hold - time.perf_counter()
        return (True, True, None) if left <= 0 else (False, False, left)

    def _take(self):
        wave, self.pending = self.pending, []
        return wave

    def _dispatch(self):
        while True:
            with self.cv:
                while True:
                    go, timed_out, wait = self._ready()
                    if go:
                        break
                    self.cv.wait(wait)
                wave = self._take()
                self.inflight += len(wave)
                self.wave_n = len(wave)
            st = self.stats
            now = time.perf_counter()
            st[0] += 1
            st[1] += len(wave)
            st[2] += timed_out
            if self.t_free:
                st[3] += now - max(self.t_free, self.t_sent)
            self.t_sent = now
            if st[0] % self.log_every == 0:
                print(f"engv2 waves: {st[0]} waves, {st[1] / st[0]:.2f} decisions each, {st[2]} sent on the hold timeout, "
                      f"mean {st[3] / st[0] * 1e3:.1f} ms held after the previous wave, {st[4] / st[0] * 1e3:.1f} ms in flight",
                      file=sys.stderr, flush=True)
            self.steps += 1
            self.decisions += len(wave)
            groups = {}
            self.gid = (self.gid + 1) % (1 << 31)
            for w in wave:
                it = w[0].payload
                plain = (hasattr(it, "letter_ids") and hasattr(it, "ids") and getattr(it, "messages", None) is None
                         and getattr(it, "reread_ids", None) is None)
                key = (("plain",) if self.one_req else tuple(it.letter_ids)) if plain else ("one", id(w))
                groups.setdefault(key, []).append(w)
            tag = (self.gid, sum(len(v) for k, v in groups.items() if k[:1] != ("one",))) if self.group_tag else None
            for key, items in groups.items():
                self.tokens += sum(w[0].n_tokens for w in items)
                if self.trace:
                    items[0][0]._t_disp = now
                    items[0][0]._dbg = [getattr(self, "burst", 0), getattr(self, "peak", 0), len(wave), int(timed_out)]
                self.pool.submit(self._group, items, None if key[:1] == ("one",) else tag).add_done_callback(
                    lambda f, items=items: self._done(items, f))

    def _group(self, items, tag=None):
        if len(items) == 1 and tag is None:
            return [self.backend._one(items[0][0])]
        its = [w[0].payload for w in items]
        sets = [list(it.letter_ids) for it in its]
        letters = sets[0]
        mixed = any(s != letters for s in sets)
        if mixed:
            letters = sorted(set().union(*sets))
        payload = {"model": self.backend.client.model, "prompt": [list(it.ids) for it in its], "max_tokens": 1,
                   "temperature": 0, "logprobs": len(letters), "allowed_token_ids": letters,
                   "return_tokens_as_token_ids": True}
        if tag is not None:
            payload["vllm_xargs"] = {"reax_g": tag[0], "reax_n": tag[1]}
        if mixed:
            payload.setdefault("vllm_xargs", {})["reax_allowed"] = ";".join(",".join(map(str, x)) for x in sets)
        if self.trace and self.dumped < 256:
            self.dumped += 1
            with open(os.environ["REAX_WAVE_TRACE"] + ".payloads", "a") as f:
                f.write(json.dumps(payload) + "\n")
        out = self.backend.client._post("/v1/completions", payload)
        res = [None] * len(its)
        for ch in out["choices"]:
            top = ch["logprobs"]["top_logprobs"][0]
            got = {int(k[9:]): v for k, v in top.items() if k.startswith("token_id:")}
            mine = sets[ch["index"]]
            missing = [i for i in mine if i not in got]
            if missing:
                raise RuntimeError(f"vLLM did not return values for option tokens {missing[:3]}")
            if mixed and len(got) != len(mine):
                raise RuntimeError("vLLM answered with the union of the letter sets (one_req.py not installed)")
            res[ch["index"]] = self.softmax_t([got[i] for i in mine], its[ch["index"]].temperature)
        return res

    def _done(self, items, f):
        err = f.exception()
        res = None if err else f.result()
        if self.trace:
            sub = [w[0]._t_sub for w in items]
            self.trace.write(f'{{"n": {len(items)}, "sub0": {min(sub)}, "sub1": {max(sub)}, '
                             f'"disp": {items[0][0]._t_disp}, "done": {time.perf_counter()}, '
                             f'"dbg": {json.dumps(getattr(items[0][0], "_dbg", None))}}}\n')
        for i, (_, loop, fut) in enumerate(items):
            loop.call_soon_threadsafe(_deliver, fut, None if err else res[i], err)
        with self.cv:
            self.inflight -= len(items)
            if self.inflight == 0:
                self.t_free = time.perf_counter()
                self.stats[4] += self.t_free - self.t_sent
                self.expect = self.wave_n + len(self.pending)
                self.cv.notify()


class HalfScheduler(WaveScheduler):

    def __init__(self, backend, workers: int, hold_ms: float = 0.0, quiet_ms: float = 0.0, peak_s: float = 2.0,
                 cap: bool = False, space_ms: float = 0.0, idle_hold_ms: float = 0.0, idle_ms: float = 50.0):
        super().__init__(backend, workers, hold_ms)
        self.quiet, self.peak_s, self.cap, self.space = quiet_ms / 1000.0, peak_s, cap, space_ms / 1000.0
        self.idle_hold, self.idle = idle_hold_ms / 1000.0, idle_ms / 1000.0
        self.peak, self.t_peak = 0, 0.0
        self.t_first = self.t_last = 0.0
        self.burst = self.burst_peak = 0
        self.held = False

    def submit(self, enc, loop, fut):
        if self.trace:
            enc._t_sub = time.perf_counter()
        with self.cv:
            now = time.perf_counter()
            if not self.pending:
                self.t_first = now
                self.held = bool(self.idle_hold and not self.inflight and self.t_free and now - self.t_free >= self.idle)
                if self.held:
                    self.burst, self.burst_peak = self.burst_peak, 0
            self.t_last = now
            self.pending.append((enc, loop, fut))
            self.cv.notify()

    def _ready(self):
        if not self.pending:
            return False, False, None
        n, now = len(self.pending) + self.inflight, time.perf_counter()
        self.burst_peak = max(self.burst_peak, n)
        if self.held and 2 * len(self.pending) < self.burst:
            left = self.t_first + self.idle_hold - now
            if left > 0:
                return False, False, left
        if self.held:
            self.peak, self.t_peak = max(n, self.burst), now
        self.held = False
        if n >= self.peak or now - self.t_peak > self.peak_s:
            self.peak, self.t_peak = n, now
        if 2 * len(self.pending) >= self.peak:
            early = self.t_sent + self.space - now if self.inflight else 0.0
            return (True, False, None) if early <= 0 else (False, False, early)
        if self.inflight:
            return False, False, None
        if self.quiet <= 0 and self.hold <= 0:
            return True, False, None
        left = []
        if self.quiet > 0:
            left.append(self.t_last + self.quiet - now)
        if self.hold > 0:
            left.append(max(self.t_first, self.t_free) + self.hold - now)
        wait = min(left)
        return (True, True, None) if wait <= 0 else (False, False, wait)

    def _take(self):
        k = (self.peak + 1) // 2
        if self.cap and self.peak >= 4 and len(self.pending) > k:
            wave, self.pending = self.pending[:k], self.pending[k:]
            self.t_first = time.perf_counter()
            return wave
        return super()._take()


def make_scheduler(backend, max_wait_ms, scheduler_cls):
    if os.environ.get("REAX_ENC_INLINE", "0") == "1":
        backend.encode_inline = True
    if os.environ.get("REAX_ENC_FAST", "0") == "1":
        from engine import enc_fast
        enc_fast.install(backend)
    mode = os.environ.get("REAX_STREAM", "1")
    if mode in ("1", "wave", "half") and hasattr(backend, "_one") and hasattr(backend, "client"):
        workers = int(os.environ.get("REAX_STREAM_WORKERS", "64"))
        if mode == "half":
            env = os.environ.get
            return HalfScheduler(backend, workers, float(env("REAX_HALF_HOLD_MS", "0")),
                                 float(env("REAX_HALF_QUIET_MS", "0")), float(env("REAX_HALF_PEAK_S", "2")),
                                 env("REAX_HALF_CAP", "0") == "1", float(env("REAX_HALF_SPACE_MS", "0")),
                                 float(env("REAX_HALF_IDLE_HOLD_MS", "0")), float(env("REAX_HALF_IDLE_MS", "50")))
        if mode == "wave":
            return WaveScheduler(backend, workers, float(os.environ.get("REAX_WAVE_HOLD_MS", "10")))
        return StreamScheduler(backend, workers)
    return scheduler_cls(backend, max_wait_ms)
