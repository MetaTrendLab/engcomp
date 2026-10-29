import json
import os
import sys

ROOT = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor")
STAMP = os.path.join(ROOT, ".engv2_pins_written")

PINS = {}


def table():
    extra = os.environ.get("REAX_PINS_FILE")
    if extra:
        return {"file": json.load(open(extra))}
    return PINS


def install():
    n = 0
    for tag, entries in table().items():
        for rel, cfg in entries.items():
            p = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(cfg, f)
            n += 1
    os.makedirs(ROOT, exist_ok=True)
    with open(STAMP, "w") as f:
        f.write(str(n))
    print(f"engv2 pins: {n} autotune choices written to {ROOT}", flush=True)


def check():
    pinned = {rel: cfg for entries in table().values() for rel, cfg in entries.items()}
    t0 = os.path.getmtime(STAMP) if os.path.exists(STAMP) else 0
    fresh, overridden = [], []
    for d, _, files in os.walk(ROOT):
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, ROOT)
            if not f.endswith(".best_config"):
                continue
            if rel not in pinned:
                if os.path.getmtime(p) >= t0:
                    fresh.append(rel)
                continue
            try:
                got = json.load(open(p))
            except (OSError, ValueError):
                got = None
            keep = lambda c: {k: v for k, v in (c or {}).items() if k not in ("time_taken_ms", "found_by_coordesc",
                                                                                "triton_cache_hash")}
            if keep(got) != keep(pinned[rel]):
                overridden.append(rel)
    print(f"engv2 pins: {len(fresh)} kernels autotuned outside the pin table" + (f": {fresh[:6]}" if fresh else "")
          + f"; {len(overridden)} pins overridden" + (f": {overridden[:6]}" if overridden else ""), flush=True)


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else install()
