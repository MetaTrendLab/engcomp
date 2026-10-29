import json
import os
import re
import threading
import time

TICK = os.sysconf("SC_CLK_TCK")
POOL = re.compile(r"^(.*?)_\d+$|^(.*?)-?\d+$")


def _tasks(pid):
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            stat = open(f"/proc/{pid}/task/{tid}/stat").read()
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            f = stat[stat.rindex(")") + 2:].split()
            out[int(tid)] = (comm, (int(f[11]) + int(f[12])) / TICK)
        except (OSError, ValueError):
            continue
    return out


def _children(pid):
    kids = set()
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return []
    for tid in tids:
        try:
            kids.update(int(k) for k in open(f"/proc/{pid}/task/{tid}/children").read().split())
        except (OSError, ValueError):
            continue
    return sorted(kids)


def start(path, period_s=0.25):
    me = os.getpid()
    prev = {}

    def label(pid, tid, comm, names):
        if pid == me:
            name = names.get(tid, comm)
            m = POOL.match(name)
            return "reax/" + ((m.group(1) or m.group(2)) if m and (m.group(1) or m.group(2)) else name)
        return f"child{pid}/" + comm

    def loop():
        kids, last_kids, first = [], 0.0, True
        with open(path, "a") as out:
            while True:
                time.sleep(period_s)
                now = time.time()
                if now - last_kids > 5:
                    kids, last_kids = _children(me), now
                names = {t.native_id: t.name for t in threading.enumerate()}
                cpu = {}
                for pid in [me, *kids]:
                    for tid, (comm, s) in _tasks(pid).items():
                        d = s - prev.get((pid, tid), 0.0)
                        prev[(pid, tid)] = s
                        if d > 0:
                            k = label(pid, tid, comm, names)
                            cpu[k] = round(cpu.get(k, 0.0) + d, 3)
                if first:
                    first = False
                    continue
                out.write(json.dumps({"t": round(now, 3), "pc": round(time.perf_counter(), 4), "cpu": cpu}) + "\n")
                out.flush()

    threading.Thread(target=loop, name="thread-cpu", daemon=True).start()
