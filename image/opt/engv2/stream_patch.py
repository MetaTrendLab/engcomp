import sys

path = sys.argv[1]
src = open(path).read()
old = "    sched = Scheduler(backend, a.max_wait_ms)\n"
assert src.count(old) == 1, "serve.py scheduler line not found"
src = src.replace(old, "    from engine.engv2_stream import make_scheduler\n    sched = make_scheduler(backend, a.max_wait_ms, Scheduler)\n")
open(path, "w").write(src)
print("stream_patch: installed")
