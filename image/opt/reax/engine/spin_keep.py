import os
import subprocess
import sys
import threading
import time

SPIN = r"""
import ctypes, os, signal, sys
ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))  # PR_SET_PDEATHSIG: die with the REAX server
os.sched_setaffinity(0, {int(sys.argv[1])})
os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
while True:
    pass
"""


def _read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def _cpu_of(pid, tid):
    stat = _read(f"/proc/{pid}/task/{tid}/stat")
    try:
        return int(stat[stat.rindex(")") + 2:].split()[36])
    except (ValueError, IndexError):
        return None


def _children(pid):
    kids = set()
    for tid in os.listdir(f"/proc/{pid}/task"):
        kids.update(int(k) for k in _read(f"/proc/{pid}/task/{tid}/children").split())
    return sorted(kids)


def _core(cpu):
    return _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") or str(cpu)


def _freq(cpu):
    f = _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
    return int(f) if f.isdigit() else 0


_used, _procs, _lock = set(), [], threading.Lock()


def _pin(label, pid, tid):
    with _lock:
        free = [c for c in sorted(os.sched_getaffinity(0)) if _core(c) not in _used]
        if not free:
            return
        top = max(_freq(c) for c in free)
        best = [c for c in free if _freq(c) == top]
        last = _cpu_of(pid, tid)
        cpu = last if last in best else best[0]
        try:
            os.sched_setaffinity(tid, {cpu})
        except OSError as exc:
            print(f"reax spin: could not pin {label} ({exc!r})", flush=True)
            return
        _used.add(_core(cpu))
        _procs.append(subprocess.Popen([sys.executable, "-I", "-S", "-c", SPIN, str(cpu)], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f"reax spin: {label} pinned to CPU {cpu} + a SCHED_IDLE spinner", flush=True)


def _spin_alone():
    allowed = sorted(os.sched_getaffinity(0))
    cpu = min(allowed, key=lambda c: (_freq(c), len(_core(c).replace("-", ",").split(",")), -c))
    _procs.append(subprocess.Popen([sys.executable, "-I", "-S", "-c", SPIN, str(cpu)], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f"reax spin: one SCHED_IDLE spinner on CPU {cpu}, nothing pinned", flush=True)


def start():
    me = os.getpid()
    want = set(os.environ.get("REAX_SPIN_SET", "one").split(","))
    if "one" in want:
        _spin_alone()
        return
    if "loop" in want:
        for t in threading.enumerate():
            if t.name == "vllm-inproc":
                _pin(f"loop:{t.native_id}", me, t.native_id)
    if "core" in want:
        for k in _children(me):
            if _read(f"/proc/{k}/comm").startswith("VLLM::EngineCor"):
                _pin(f"core:{k}", k, k)
    if "main" in want:
        def later():
            t0 = time.time()
            while not any(t.name == "wave" for t in threading.enumerate()) and time.time() - t0 < 1800:
                time.sleep(0.5)
            time.sleep(1.0)
            _pin("main", me, me)
            threading.Event().wait()

        threading.Thread(target=later, name="spin-keep", daemon=True).start()
