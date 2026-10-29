import functools
import gc
import importlib
import inspect
import json
import re
import threading
import time

POOL = re.compile(r"^(.*?)_\d+$|^(.*?)-?\d+$")
TARGETS = (
    "json:loads", "json:dumps",
    "engine.backends.vllm_readout:VllmReadoutBackend.encode", "engine.backends.vllm_readout:build_rune_prompt",
    "engine.backends.vllm_readout:rune_options",
    "transformers.tokenization_utils_base:PreTrainedTokenizerBase.__call__",
    "transformers.tokenization_utils_base:PreTrainedTokenizerBase.apply_chat_template",
    "transformers.tokenization_utils_fast:PreTrainedTokenizerFast._batch_encode_plus",
    "transformers.tokenization_utils_fast:PreTrainedTokenizerFast._convert_encoding",
    "jinja2.environment:Template.render", "engine.enc_fast:_Tok.__call__", "engine.enc_fast:_Tok.apply_chat_template",
    "engine.engv2_stream:HalfScheduler.submit", "engine.engv2_stream:HalfScheduler._group",
    "engine.engv2_stream:HalfScheduler._done", "engine.engv2_stream:_deliver",
    "vllm.entrypoints.openai.completion.protocol:CompletionRequest.to_sampling_params",
    "vllm.entrypoints.openai.completion.serving:OpenAIServingCompletion.request_output_to_completion_response",
    "vllm.v1.engine.input_processor:InputProcessor.process_inputs",
    "vllm.v1.engine.output_processor:OutputProcessor.add_request",
    "vllm.v1.engine.output_processor:OutputProcessor.process_outputs",
    "vllm.v1.engine.logprobs:LogprobsProcessor.update_from_output",
    "vllm.v1.engine.core_client:AsyncMPClient._send_input",
    "vllm.v1.serial_utils:MsgpackEncoder.encode", "vllm.v1.serial_utils:MsgpackDecoder.decode",
)
_acc = {}
_names = {}


def _wrap(spec):
    mod, _, path = spec.partition(":")
    owner = importlib.import_module(mod)
    *outer, attr = path.split(".")
    for p in outer:
        owner = getattr(owner, p)
    raw = inspect.getattr_static(owner, attr)
    kind = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
    f = raw.__func__ if kind else raw
    label = path if outer else f"{mod}.{path}"
    tt, ident = time.thread_time_ns, threading.get_ident

    @functools.wraps(f)
    def w(*a, **k):
        t0 = tt()
        try:
            return f(*a, **k)
        finally:
            slot = _acc.get((ident(), label))
            if slot is None:
                _names[ident()] = threading.current_thread().name
                slot = _acc[(ident(), label)] = [0, 0]
            slot[0] += tt() - t0
            slot[1] += 1

    setattr(owner, attr, kind(w) if kind else w)


def _gc_hook():
    tt, ident, t0 = time.thread_time_ns, threading.get_ident, {}

    def cb(phase, info):
        me = ident()
        if phase == "start":
            t0[me] = tt()
            return
        if me in t0:
            key = (me, f"gc.gen{info.get('generation')}")
            slot = _acc.get(key)
            if slot is None:
                _names[me] = threading.current_thread().name
                slot = _acc[key] = [0, 0]
            slot[0] += tt() - t0.pop(me)
            slot[1] += 1

    gc.callbacks.append(cb)


def start(path, period_s=0.25):
    _gc_hook()
    for spec in TARGETS:
        try:
            _wrap(spec)
        except (ImportError, AttributeError) as exc:
            print(f"reax fn_cpu: {spec} not wrapped ({exc!r})", flush=True)

    def name(ident):
        n = _names.get(ident, str(ident))
        m = POOL.match(n)
        return (m.group(1) or m.group(2)) if m and (m.group(1) or m.group(2)) else n

    def loop():
        prev = {}
        with open(path, "a") as out:
            while True:
                time.sleep(period_s)
                cpu, calls = {}, {}
                for key, (ns, n) in list(_acc.items()):
                    p_ns, p_n = prev.get(key, (0, 0))
                    prev[key] = (ns, n)
                    if n > p_n:
                        k = f"{name(key[0])}|{key[1]}"
                        cpu[k] = round(cpu.get(k, 0.0) + (ns - p_ns) * 1e-9, 6)
                        calls[k] = calls.get(k, 0) + n - p_n
                out.write(json.dumps({"t": round(time.time(), 3), "pc": round(time.perf_counter(), 4), "cpu": cpu,
                                      "n": calls}) + "\n")
                out.flush()

    threading.Thread(target=loop, name="fn-cpu", daemon=True).start()
