from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import threading
import time


class InprocClient:

    def __init__(self, argv, env, model="reax", completion_temperature=0, name="text", spawn_env=None):
        from engine.backends.vllm_readout import VllmClient
        self.model, self.completion_temperature, self.name = model, completion_temperature, name
        self.option_values = VllmClient.option_values.__get__(self)
        self.image_option_values = VllmClient.image_option_values.__get__(self)
        base = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "OPENAI_API_KEY": "", **env}
        if spawn_env is None:
            os.environ.update(base)
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            self.spawn_env = None
        else:
            self.spawn_env = {**spawn_env, **base}
            self.spawn_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        assert argv[:2] == ["vllm", "serve"], argv[:2]
        self.argv = argv
        try:
            import uvloop
            self.loop = uvloop.new_event_loop()
        except ImportError:
            self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="vllm-inproc", daemon=True).start()
        t0 = time.time()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result()
        print(f"reax inproc: {name} vLLM in-process, ready in {time.time() - t0:.0f} s", flush=True)
        threading.Thread(target=self._watch, name="vllm-inproc-watch", daemon=True).start()

    def _parse(self):
        import vllm.entrypoints.cli.serve as serve_cli
        from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        parser = FlexibleArgumentParser(description="vLLM CLI",
                                        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"))
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        (cmd,) = [c for c in serve_cli.cmd_init() if c.name == "serve"]
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        args = parser.parse_args(self.argv[1:])
        cmd.validate(args)
        if getattr(args, "model_tag", None) is not None:
            args.model = args.model_tag
        if getattr(args, "headless", False) or (args.api_server_count or 1) != 1 or getattr(args, "grpc", False):
            raise SystemExit("reax inproc: only the single API server form of `vllm serve` can run in-process")
        args.api_server_count = None
        return args

    async def _start(self):
        from vllm.entrypoints.launchers.api_server.app_state import init_app_state
        from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client
        from vllm.entrypoints.launchers.app import build_app
        args = self._parse()
        self._engine_cm = build_async_engine_client(args)
        with self._environ():
            self.engine = await self._engine_cm.__aenter__()
        tasks = await self.engine.get_supported_tasks()
        app = build_app(args, tasks, self.engine.model_config)
        await init_app_state(self.engine, app.state, args, tasks)
        self._life = app.router.lifespan_context(app)
        await self._life.__aenter__()
        self.app = app
        self._handlers = {"/v1/completions": app.state.openai_serving_completion,
                          "/v1/chat/completions": app.state.openai_serving_chat}

    @contextlib.contextmanager
    def _environ(self):
        if self.spawn_env is None:
            yield
            return
        old = {k: os.environ.get(k) for k, v in self.spawn_env.items() if os.environ.get(k) != v}
        print(f"reax inproc: {self.name} EngineCore env: " + " ".join(f"{k}={self.spawn_env[k]}" for k in sorted(old)),
              flush=True)
        os.environ.update({k: self.spawn_env[k] for k in old})
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _watch(self):
        while True:
            time.sleep(1)
            if self.engine.errored:
                print(f"reax inproc: the {self.name} EngineCore died ({self.engine.dead_error!r}); stopping", flush=True)
                sys.stdout.flush()
                os._exit(1)

    async def _call(self, path, body):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        if path == "/v1/completions":
            return await self._handlers[path].create_completion(CompletionRequest.model_validate(body), None)
        return await self._handlers[path].create_chat_completion(ChatCompletionRequest.model_validate(body), None)

    def _post(self, path, body):
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        if path not in ("/v1/completions", "/v1/chat/completions"):
            raise RuntimeError(f"reax inproc: no handler for {path}")
        try:
            res = asyncio.run_coroutine_threadsafe(self._call(path, body), self.loop).result()
        except Exception as exc:
            if path == "/v1/chat/completions" and isinstance(exc, ValueError):
                raise ValueError("maximum context length exceeded or invalid image prompt") from exc
            raise RuntimeError(f"vLLM in-process: {exc!r}"[:400]) from exc
        if isinstance(res, ErrorResponse):
            if path == "/v1/chat/completions" and res.error.code == 400:
                raise ValueError("maximum context length exceeded or invalid image prompt")
            raise RuntimeError(f"vLLM HTTP {res.error.code}: {res.error.message[:300]!r}")
        return res.model_dump()

    def healthy(self):
        return not self.engine.errored


def install(backend, model_dir):
    from engine.backends.vllm_readout import preset_for, vllm_argv
    if os.environ.get("REAX_FN_CPU"):
        from engine import fn_cpu
        fn_cpu.start(os.environ["REAX_FN_CPU"])
    preset = preset_for(backend.family, os.environ.get("REAX_INPROC_PRESET", "ref-prod"))
    extra = os.environ.get("REAX_VLLM_EXTRA", "").split()
    if os.environ.get("REAX_NOSTATS", "0") == "1":
        extra.append("--disable-log-stats")
    argv = vllm_argv(preset, model_dir, port=int(os.environ.get("REAX_VLLM_PORT", "8000")), extra=extra)
    print(f"reax inproc: vllm serve flags {' '.join(a for a in argv[3:] if a.startswith('--'))}", flush=True)
    old = backend.client
    backend.client = InprocClient(argv, preset.env, model=old.model, completion_temperature=old.completion_temperature)
    if getattr(backend, "image_client", None) is old:
        backend.image_client = backend.client
    ready = os.environ.get("REAX_INPROC_READY", "/tmp/reax_inproc_ready")
    with open(ready, "w") as f:
        f.write(f"{os.getpid()}\n")
    img = getattr(backend, "image_client", None)
    env_file = os.environ.get("REAX_INPROC_IMAGE_ENV")
    if env_file and img is not None and img is not backend.client:
        go, t0 = os.environ.get("REAX_INPROC_IMAGE_GO", "/tmp/reax_inproc_image_go"), time.time()
        while not os.path.exists(go):
            if time.time() - t0 > 600:
                raise SystemExit("reax inproc: entry_dual.sh never started the image instance")
            time.sleep(0.5)
        with open(env_file) as f:
            spawn_env = json.load(f)
        argv = vllm_argv(preset, model_dir, port=int(os.environ.get("REAX_VLLM_IMAGE_PORT", "8001")),
                         extra=os.environ["REAX_INPROC_IMAGE_EXTRA"].split())
        print(f"reax inproc: image vllm serve flags {' '.join(a for a in argv[3:] if a.startswith('--'))}", flush=True)
        backend.image_client = InprocClient(argv, preset.env, model=img.model,
                                            completion_temperature=img.completion_temperature, name="image",
                                            spawn_env=spawn_env)
    elif img is not None and img is not backend.client:
        t0 = time.time()
        while not img.healthy():
            if time.time() - t0 > float(os.environ.get("REAX_INPROC_IMAGE_TIMEOUT", "1800")):
                raise SystemExit("reax inproc: the image instance did not become healthy")
            time.sleep(2)
        print(f"reax inproc: image instance healthy after {time.time() - t0:.0f} s", flush=True)
    if os.environ.get("REAX_THREAD_CPU"):
        from engine import thread_cpu
        thread_cpu.start(os.environ["REAX_THREAD_CPU"])
    if os.environ.get("REAX_SPIN", "0") == "1":
        from engine import spin_keep
        spin_keep.start()
