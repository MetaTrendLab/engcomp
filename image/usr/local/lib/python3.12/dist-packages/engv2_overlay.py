import importlib.util
import os
import sys

OVERLAY = os.environ.get("REAX_BLINK_OVERLAY_DIR", "/opt/engv2/overlay")
MODULES = {
    "vllm.model_executor.layers.logits_processor": "logits_processor.py",
    "vllm.model_executor.models.gemma4": "gemma4.py",
    "vllm.v1.worker.gpu.model_runner": "model_runner.py",
}


class _Finder:
    def find_spec(self, name, path=None, target=None):
        f = MODULES.get(name)
        if f is None:
            return None
        return importlib.util.spec_from_file_location(name, os.path.join(OVERLAY, f))


def build(site, overlay):
    import shutil
    import sysconfig

    os.makedirs(overlay, exist_ok=True)
    for name, f in MODULES.items():
        shutil.copy(os.path.join(site, *name.split(".")) + ".py", os.path.join(overlay, f))
    shutil.copy(os.path.abspath(__file__), os.path.join(site, "engv2_overlay.py"))
    spec = importlib.util.find_spec("sitecustomize")
    sc = spec.origin if spec and spec.origin else os.path.join(sysconfig.get_paths()["stdlib"], "sitecustomize.py")
    with open(sc, "a") as fh:
        fh.write("\n# engv2 blink_overlay.py: Blink-text-only vLLM modules when REAX_BLINK_OVERLAY=1\n# (other interpreters, e.g. the q35 venv, do not have the module: skip quietly)\ntry:\n    import engv2_overlay\nexcept ModuleNotFoundError as e:\n    if e.name != \"engv2_overlay\":\n        raise\n")
    print(f"blink_overlay: stock files copied to {overlay}, hook imported from {sc}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
elif os.environ.get("REAX_BLINK_OVERLAY") == "1":
    sys.meta_path.insert(0, _Finder())
