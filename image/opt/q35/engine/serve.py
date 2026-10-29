#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import hmac
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from backends.base import sidecar_answer
from backends.common import s1_answer

LONG_TEXT = 20000


_STEPLOG = os.environ.get("Q35_STEPLOG") == "1"
_REQLOG = os.environ.get("Q35_REQLOG") == "1"


def _gc_after_warmup():
    if os.environ.get("Q35_GC_LOG") == "1":
        t0 = {}

        def cb(phase, info):
            if phase == "start":
                t0["t"] = time.perf_counter()
            elif "t" in t0:
                ms = (time.perf_counter() - t0.pop("t")) * 1e3
                if ms >= 20:
                    print(f"GC gen{info.get('generation')} {ms:.1f} ms collected {info.get('collected')} t={time.perf_counter():.4f}",
                          flush=True)
        gc.callbacks.append(cb)
    if os.environ.get("Q35_GC_FREEZE") == "1":
        gc.collect()
        gc.freeze()
        print(f"q35 gc.freeze: {gc.get_freeze_count()} objects frozen", flush=True)


class Scheduler:
    def __init__(self, backend, max_wait_ms: float = 0.0, max_items: int = 256):
        self.backend = backend
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.max_wait = max_wait_ms / 1000.0
        self.max_items = max_items
        self.steps = self.decisions = self.tokens = 0
        self.outstanding = 0
        import collections
        self._peaks = collections.deque()
        self.hold = float(os.environ.get("Q35_HOLD_MS", "0")) / 1000.0
        self.hold_cap = int(os.environ.get("Q35_HOLD_CAP", "16"))
        self.early = os.environ.get("Q35_HOLD_EARLY", "0") == "1"
        self._last = (1 << 30, 0.0)
        self._on_ready = None
        self.thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)

    def start(self, on_ready=None):
        self._on_ready = on_ready
        self.thread.start()

    def count(self, k: int = 1):
        self.outstanding += k
        now = time.perf_counter()
        self._peaks.append((now, self.outstanding))
        while self._peaks and self._peaks[0][0] < now - 2.0:
            self._peaks.popleft()

    def submit(self, enc, loop, fut):
        self.count()
        self.q.put((enc, loop, fut))

    def concurrency(self) -> int:
        now = time.perf_counter()
        return max((o for t, o in list(self._peaks) if t >= now - 2.0), default=1)

    def _full(self, k: int) -> bool:
        n_last, t_last = self._last
        return (2 <= n_last <= k and time.perf_counter() - t_last < 1.0
                and n_last >= min(self.hold_cap, self.concurrency()))

    def _loop(self):
        be, q = self.backend, self.q
        if self._on_ready:
            self._on_ready()
        _gc_after_warmup()
        limit = be.max_batch_tokens
        carry = None
        while True:
            first = carry or q.get()
            t_first = time.perf_counter()
            carry = None
            batch, n = [first], first[0].n_tokens
            if limit == 0:
                pass
            else:
                t0 = time.perf_counter()
                deadline = t0 + self.max_wait if self.max_wait else 0.0
                hold_end = t0 + self.hold
                while len(batch) < self.max_items:
                    try:
                        if deadline and not (self.early and self._full(len(batch))):
                            item = q.get(timeout=max(0.0, deadline - time.perf_counter()))
                        else:
                            item = q.get_nowait()
                    except queue.Empty:
                        if (self.hold and len(batch) < min(self.hold_cap, self.concurrency())
                                and time.perf_counter() < hold_end):
                            try:
                                item = q.get(timeout=max(0.0, min(hold_end - time.perf_counter(), 0.002)))
                            except queue.Empty:
                                continue
                        else:
                            break
                    if n + item[0].n_tokens > limit:
                        carry = item
                        break
                    batch.append(item)
                    n += item[0].n_tokens
            t_run = time.perf_counter()
            try:
                res = be.run([b[0] for b in batch])
                err = None
            except Exception as e:
                res, err = None, e
            if _STEPLOG:
                print(f"STEP n={len(batch)} tok={n} t={time.perf_counter():.4f} first={t_first:.4f} run={t_run:.4f}", flush=True)
            self.steps += 1
            self.decisions += len(batch)
            self.tokens += n
            self._last = (len(batch), time.perf_counter())
            by_loop: dict = {}
            for i, b in enumerate(batch):
                by_loop.setdefault(b[1], []).append((b[2], None if err else res[i]))
            for loop, items in by_loop.items():
                loop.call_soon_threadsafe(_deliver, items, err, self)




class LaneScheduler(Scheduler):

    def __init__(self, backend, max_wait_ms: float = 0.0, lanes: int = 2, cap: int = 16):
        super().__init__(backend, max_wait_ms)
        self.lanes, self.cap = lanes, cap
        self._take = threading.Lock()
        self.thread = threading.Thread(target=self._boot, name="scheduler-boot", daemon=True)

    def _boot(self):
        if self._on_ready:
            self._on_ready()
        for i in range(self.lanes):
            threading.Thread(target=self._lane, args=(i,), name=f"lane{i}", daemon=True).start()

    def _lane(self, i):
        import torch
        be, q, limit = self.backend, self.q, self.backend.max_batch_tokens
        stream = torch.cuda.Stream()
        carry = None
        with torch.cuda.stream(stream):
            while True:
                with self._take:
                    first = carry or q.get()
                    carry = None
                    batch, n = [first], first[0].n_tokens
                    deadline = time.perf_counter() + self.max_wait if self.max_wait else 0.0
                    while len(batch) < self.cap:
                        try:
                            item = q.get(timeout=max(0.0, deadline - time.perf_counter())) if deadline else q.get_nowait()
                        except queue.Empty:
                            break
                        if limit and n + item[0].n_tokens > limit:
                            carry = item
                            break
                        batch.append(item)
                        n += item[0].n_tokens
                try:
                    res = be.run([b[0] for b in batch])
                    err = None
                except Exception as e:
                    res, err = None, e
                if _STEPLOG:
                    print(f"STEP lane={i} n={len(batch)} tok={n} t={time.perf_counter():.4f}", flush=True)
                self.steps += 1
                self.decisions += len(batch)
                self.tokens += n
                by_loop: dict = {}
                for j, b in enumerate(batch):
                    by_loop.setdefault(b[1], []).append((b[2], None if err else res[j]))
                for loop, items in by_loop.items():
                    loop.call_soon_threadsafe(_deliver, items, err)

def _enc_failed(fut, exc, sched):
    sched.outstanding -= 1
    if not fut.done():
        fut.set_exception(exc)


def _deliver(items, err, sched=None):
    if sched is not None:
        sched.outstanding -= len(items)
    for fut, val in items:
        if not fut.done():
            if err is not None:
                fut.set_exception(err)
            else:
                fut.set_result(val)


def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("serve: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE; RUNTIME_ALLOW_NO_AUTH=1 to disable)")
    return tok


class App:
    def __init__(self, backend, sched: Scheduler, token: str, model_name: str):
        self.be, self.sched, self.token, self.model = backend, sched, token.encode(), model_name
        self.ready = False
        self.loop_hdr = b"x-s1m-internal-token"
        self._inline = getattr(backend, "encode_inline", False)
        if os.environ.get("Q35_ENC_INLINE") in ("0", "1"):
            self._inline = os.environ["Q35_ENC_INLINE"] == "1"
        self._encq = None
        for i in range(int(os.environ.get("Q35_ENC_THREAD", "0") or 0)):
            if self._encq is None:
                self._encq = queue.SimpleQueue()
            threading.Thread(target=self._enc_loop, name=f"encode{i}", daemon=True).start()
        own = getattr(backend, "answer", None)
        if backend.kind == "gguf-cpu" or getattr(backend, "family", None) == "mammoth":
            s1 = own
        elif backend.name == "plumb-4b":
            s1 = lambda q, p: sidecar_answer(q, backend.options(q), p)
        else:
            s1 = lambda q, p: s1_answer(q["type"], backend.options(q), p)
        self._shape = {"/internal/v1/decide": s1, "/v1/systemone": own or s1}
        self.aliases = ()
        self.classifier = getattr(backend, "model_type", "") == "classifier_heads"

    def _enc_loop(self):
        q, sched, encode = self._encq, self.sched, self.be.encode
        while True:
            state, question, loop, fut, box = q.get()
            t0 = time.perf_counter()
            try:
                enc = encode(state, question)
            except Exception as exc:
                loop.call_soon_threadsafe(_enc_failed, fut, exc, sched)
                continue
            box += (enc, t0, time.perf_counter())
            sched.q.put((enc, loop, fut))

    async def decide(self, state, questions: dict, path: str, long_body: bool = False, images=None, marks=None):
        loop = asyncio.get_running_loop()
        if self._encq is not None and not images:
            futs, boxes = {}, {}
            for qid, q in questions.items():
                futs[qid], boxes[qid] = loop.create_future(), []
                self.sched.count()
                self._encq.put((state, q, loop, futs[qid], boxes[qid]))
            probs = {qid: await f for qid, f in futs.items()}
            if marks is not None:
                b0 = next(iter(boxes.values()))
                marks += [b0[1], b0[2], max(b[2] for b in boxes.values()), time.perf_counter()]
            shape = self._shape[path]
            return ({qid: shape(q, probs[qid]) for qid, q in questions.items()},
                    sum(b[0].n_tokens for b in boxes.values()))
        encode = (lambda st, q: self.be.encode_images(st, q, images)) if images else self.be.encode
        if marks is not None:
            enc0 = encode

            def encode(st, q, _f=enc0):
                marks.append(time.perf_counter())
                r = _f(st, q)
                marks.append(time.perf_counter())
                return r
        if long_body or not self._inline:
            encs = dict(zip(questions, await asyncio.gather(
                *(loop.run_in_executor(None, encode, state, q) for q in questions.values()))))
        else:
            encs = {qid: encode(state, q) for qid, q in questions.items()}
        futs = {}
        for qid, enc in encs.items():
            futs[qid] = loop.create_future()
            self.sched.submit(enc, loop, futs[qid])
        if marks is not None:
            marks.append(time.perf_counter())
        probs = {qid: await f for qid, f in futs.items()}
        if marks is not None:
            marks.append(time.perf_counter())
        shape = self._shape[path]
        answers = {qid: shape(q, probs[qid]) for qid, q in questions.items()}
        return answers, sum(e.n_tokens for e in encs.values())

    async def _probs(self, state, long_body: bool = False):
        loop = asyncio.get_running_loop()
        enc = await loop.run_in_executor(None, self.be.encode, state) if long_body else self.be.encode(state)
        fut = loop.create_future()
        self.sched.submit(enc, loop, fut)
        return await fut, enc.n_tokens

    async def classify(self, body: dict, long_body: bool = False):
        want = body.get("model")
        if want is not None and want != self.be.name and want not in self.aliases:
            raise ValueError(f"model must be {self.be.name}")
        if "request" not in body:
            raise ValueError("request is required")
        probs, n = await self._probs({"request": body["request"], "context": body.get("context")}, long_body)
        return {"model": self.be.name, "heads": self.be.heads_answer(probs),
                "usage": {"input_tokens": n, "output_tokens": 0}}

    async def decide_fixed(self, state, questions: dict, long_body: bool = False):
        self.be.check_questions(questions)
        probs, n = await self._probs(state, long_body)
        return self.be.sidecar_answers(questions, probs), n

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        if path == "/health" and method == "GET":
            if not self.ready:
                return await self._json(send, 503, {"ok": False, "status": "warming up"})
            info = {"ok": True, "model": self.model, "kind": self.be.kind, "steps": self.sched.steps, "decisions": self.sched.decisions}
            if hasattr(self.be, "rereads"):
                info.update(rereads=self.be.rereads, reread_tokens=self.be.reread_tokens)
            return await self._json(send, 200, info)
        routes = ("/internal/v1/decide", "/v1/systemone") + (("/v1/classify",) if self.classifier else ())
        if method != "POST" or path not in routes:
            return await self._json(send, 404, {"error": {"message": "not found"}})
        if self.token:
            got = b""
            for k, v in scope["headers"]:
                if k == self.loop_hdr:
                    got = v
                    break
            if not hmac.compare_digest(got, self.token):
                return await self._json(send, 401, {"error": {"message": "unauthorized"}})
        if not self.ready:
            return await self._json(send, 503, {"error": {"message": "warming up"}})
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        t_recv = time.perf_counter() if _REQLOG else 0.0
        if path == "/v1/classify":
            try:
                body = json.loads(b"".join(chunks))
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                out = await self.classify(body, sum(map(len, chunks)) > LONG_TEXT)
            except ValueError as exc:
                return await self._json(send, 400, {"error": {"message": f"bad request: {str(exc)[:200]}"}})
            except Exception as exc:
                return await self._json(send, 500, {"error": {"message": f"{type(exc).__name__}"}})
            return await self._json(send, 200, out)
        try:
            body = json.loads(b"".join(chunks))
            questions = body["questions"]
            state = body["state"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a non-empty object")
            if path == "/internal/v1/decide" and len(questions) != 1:
                raise ValueError("exactly one question per call")
            for q in questions.values():
                if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
                    raise ValueError("bad question type")
        except Exception as exc:
            return await self._json(send, 400, {"error": {"message": f"bad request: {type(exc).__name__}"}})
        images = body.get("images")
        if images is not None and images != []:
            if not getattr(self.be, "supports_images", False):
                return await self._json(send, 422, {"error": {"message": "this model takes text only"},
                                                    "code": "unsupported_images", "fallback": "author"})
            try:
                from engine.backends.vllm_readout import validate_images
                validate_images(images)
                if len(questions) != 1:
                    raise ValueError("an image request carries exactly one question")
            except ValueError as exc:
                return await self._json(send, 422, {"error": {"message": str(exc)}, "code": "invalid_images"})
        try:
            if self.classifier:
                answers, n = await self.decide_fixed(state, questions, sum(map(len, chunks)) > LONG_TEXT)
            else:
                marks = [] if _REQLOG else None
                answers, n = await self.decide(state, questions, path, sum(map(len, chunks)) > LONG_TEXT, images=images,
                                               marks=marks)
        except ValueError as exc:
            code = 422 if "maximum context length" in str(exc) else 400
            return await self._json(send, code, {"error": {"message": str(exc)}})
        except (KeyError, TypeError) as exc:
            return await self._json(send, 400, {"error": {"message": f"bad request: {type(exc).__name__}"}})
        except Exception as exc:
            return await self._json(send, 500, {"error": {"message": f"{type(exc).__name__}"}})
        await self._json(send, 200, {"model": self.model, "answers": answers,
                                     "usage": {"input_tokens": n, "output_tokens": 0}})
        if _REQLOG and len(marks) == 4:
            print(f"REQ n={n} recv={t_recv:.4f} x0={marks[0]:.4f} x1={marks[1]:.4f} enc={marks[2]:.4f} res={marks[3]:.4f} "
                  f"sent={time.perf_counter():.4f}", flush=True)

    @staticmethod
    async def _json(send, code, obj):
        raw = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_weights(model_dir: str, manifest_path: str) -> None:
    files = json.loads(Path(manifest_path).read_text())["weights"]["files"]
    out: dict = {}

    def one(name, digest):
        p = Path(model_dir) / name
        out[name] = p.is_file() and _sha(p) == digest
    ts = [threading.Thread(target=one, args=kv) for kv in files.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    bad = [n for n, ok in out.items() if not ok]
    if bad:
        sys.exit(f"weights mismatch or missing: {', '.join(sorted(bad))}")


def parse_opts(pairs) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        try:
            out[k.replace("-", "_")] = json.loads(v)
        except json.JSONDecodeError:
            out[k.replace("-", "_")] = v
    return out


def load_backend(name: str, a):
    from engine import registry
    try:
        spec = registry.get(name)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0])) from None
    opts = parse_opts(a.opt)
    profile = os.environ.get("REAX_PROFILE", "")
    if profile not in ("", "reference"):
        raise SystemExit(f"REAX_PROFILE must be 'reference' or unset, not {profile!r}")
    if profile == "reference":
        opts = {**spec.production_opts, **opts}
    if spec.id == "plumb-4b":
        from plumb_engine import PlumbBackend
        kw = dict(opts)
        if a.batch_tokens:
            kw["max_batch_tokens"] = a.batch_tokens
        return PlumbBackend(a.model_dir, compile=a.compile, **kw)
    if spec.backend_kind == "gguf-cpu" and not a.no_verify:
        from engine.backends.gguf_cpu import SPECS
        f = SPECS[spec.id]
        if not (Path(a.model_dir) / f.file).is_file() or _sha(Path(a.model_dir) / f.file) != f.sha256:
            sys.exit(f"weights mismatch or missing: {f.file}")
    try:
        return registry.build(spec.id, a.model_dir, **opts)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("REAX_MODEL") or "plumb-4b",
                    help="registry id or alias (python -m engine.registry); default $REAX_MODEL, else plumb-4b "
                         "(the engine competition v2 starts one image per model with REAX_MODEL=<id>)")
    ap.add_argument("--model-dir", default=next((os.environ[v] for v in ("PLUMB_MODEL_DIR", "REAX_MODEL_DIR", "GGUF_MODEL_DIR")
                                                 if os.environ.get(v)), "/model"))
    ap.add_argument("--host", default=None, help="default $RUNTIME_HOST or 127.0.0.1 ($RUNTIME_UDS wins over host and port)")
    ap.add_argument("--port", type=int, default=None, help="default $RUNTIME_PORT or 8101")
    ap.add_argument("--opt", action="append", default=[], help="backend option key=value (JSON values), repeatable")
    ap.add_argument("--manifest", default=os.environ.get("REAX_MANIFEST", ""), help="release manifest to verify weights against")
    ap.add_argument("--no-verify", action="store_true", help="skip the weight checks (manifest, pinned GGUF file)")
    ap.add_argument("--compile", default=os.environ.get("REAX_COMPILE", "1"))
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--max-wait-ms", type=float, default=float(os.environ.get("REAX_MAX_WAIT_MS", "0")))
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)

    token = _token()
    if a.manifest and not a.no_verify and Path(a.manifest).is_file():
        verify_weights(a.model_dir, a.manifest)
    backend = load_backend(a.model, a)
    lanes = int(os.environ.get("REAX_LANES", "1"))
    sched = (LaneScheduler(backend, a.max_wait_ms, lanes, int(os.environ.get("REAX_LANE_CAP", "16"))) if lanes > 1
             else Scheduler(backend, a.max_wait_ms))
    app = App(backend, sched, token, "s1-fast" if backend.name == "plumb-4b" else backend.name)
    app.aliases = ("llm-auto-router",) if getattr(backend, "model_type", "") == "classifier_heads" else ()

    def warm():
        t0 = time.time()
        try:
            if not a.no_warmup and getattr(backend, "serve_warmup", True) and hasattr(backend, "warmup"):
                backend.warmup()
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        app.ready = True
        print(f"ready: {backend.name} ({backend.kind}) warm-up {time.time() - t0:.0f}s", flush=True)

    import uvicorn
    kw = dict(log_level="warning", access_log=False, lifespan="on", timeout_keep_alive=75)
    if os.environ.get("RUNTIME_UDS"):
        kw["uds"] = os.environ["RUNTIME_UDS"]
    else:
        kw.update(host=a.host or os.environ.get("RUNTIME_HOST", "127.0.0.1"),
                  port=a.port or int(os.environ.get("RUNTIME_PORT", "8101")))
    if os.environ.get("Q35_SWITCH_US"):
        sys.setswitchinterval(float(os.environ["Q35_SWITCH_US"]) / 1e6)
    sched.start(on_ready=warm)
    uvicorn.run(app, **kw)


if __name__ == "__main__":
    main()
