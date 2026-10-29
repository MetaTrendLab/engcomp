# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

# engv2 autotune_hidden.py install: Inductor autotune log / hidden pins hook
# (other interpreters, e.g. the q35 venv, do not have the module: skip quietly)
try:
    import engv2_autotune
except ModuleNotFoundError as e:
    if e.name != "engv2_autotune":
        raise

# engv2 blink_overlay.py: Blink-text-only vLLM modules when REAX_BLINK_OVERLAY=1
# (other interpreters, e.g. the q35 venv, do not have the module: skip quietly)
try:
    import engv2_overlay
except ModuleNotFoundError as e:
    if e.name != "engv2_overlay":
        raise

import os as _o
if _o.environ.get('REAX_JEFFAN_F8') == '1':
    import engv2_jeffan_f8  # noqa: F401  (engv2 Jeffan FP8)
