#!/usr/bin/env python3
import importlib.util
import os
import shutil
import sysconfig

here = os.path.dirname(os.path.abspath(__file__))
sp = sysconfig.get_paths()["purelib"]
shutil.copy(os.path.join(here, "engv2_jeffan_f8.py"), os.path.join(sp, "engv2_jeffan_f8.py"))
spec = importlib.util.find_spec("sitecustomize")
sc = spec.origin if spec and spec.origin else os.path.join(sp, "sitecustomize.py")
line = "\nimport os as _o\nif _o.environ.get('REAX_JEFFAN_F8') == '1':\n    import engv2_jeffan_f8  # noqa: F401  (engv2 Jeffan FP8)\n"
txt = open(sc).read() if os.path.exists(sc) else ""
if "engv2_jeffan_f8" not in txt:
    open(sc, "a").write(line)
print("engv2_jeffan_f8 installed:", sp, "sitecustomize:", sc)
