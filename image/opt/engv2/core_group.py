import sys

path = sys.argv[1]
src = open(path).read()
old_gather = '''            n, t_end = len(sch.waiting), time.monotonic() + _REAX_GATHER_MAX
            while True:
                left = t_end - time.monotonic()
                if left <= 0:
                    break
                try:
                    req = self.input_queue.get(timeout=min(_REAX_GATHER_GAP, left))
                except queue.Empty:
                    break
'''
new_gather = '''            n, t0 = len(sch.waiting), time.monotonic()
            while True:
                grp = _reax_group_state(sch)  # engv2 core_group.py: None untagged, True a group incomplete, False complete
                if grp is False:
                    break
                left = t0 + (_REAX_GROUP_MAX if grp else _REAX_GATHER_MAX) - time.monotonic()
                if left <= 0:
                    break
                try:
                    req = self.input_queue.get(timeout=left if grp else min(_REAX_GATHER_GAP, left))
                except queue.Empty:
                    break
'''
old_poll = '''            if got:
                wait = min(_REAX_POLL_GAP, t_end - time.monotonic())
                if wait <= 0:
                    return True
            else:
                wait = _REAX_POLL_S
            try:
                req = core.input_queue.get(timeout=wait)
            except queue.Empty:
                if got:
                    return True
                continue
            core._handle_client_request(*req)
            if not got:
                t_end = time.monotonic() + _REAX_POLL_MAX
            got += 1
'''
new_poll = '''            if got:
                grp = _reax_group_state(core.scheduler)  # engv2 core_group.py
                if grp is False:
                    return True
                left = t_end - time.monotonic() + (_REAX_GROUP_MAX - _REAX_POLL_MAX if grp else 0.0)
                wait = min(_REAX_POLL_S, left) if grp else min(_REAX_POLL_GAP, left)
                if left <= 0:
                    return True
            else:
                wait = _REAX_POLL_S
            try:
                req = core.input_queue.get(timeout=wait)
            except queue.Empty:
                if got and not grp:
                    return True
                continue
            core._handle_client_request(*req)
            if not got:
                t_end = time.monotonic() + _REAX_POLL_MAX
            got += 1
'''
helper = '''
_REAX_GROUP_MAX = float(os.environ.get("REAX_CORE_GROUP_MAX_MS", "8")) / 1e3


def _reax_group_state(sch):
    """engv2 core_group.py: None when no waiting request carries a REAX group tag, True when a tagged group has only
    part of its requests waiting, False when every tagged group is complete."""
    cnt, tot = {}, {}
    for q in (sch.waiting, getattr(sch, "skipped_waiting", ())):
        for r in q:
            xa = r.sampling_params.extra_args if r.sampling_params is not None else None
            if xa and "reax_g" in xa:
                g = xa["reax_g"]
                cnt[g] = cnt.get(g, 0) + 1
                tot[g] = xa.get("reax_n", 1)
    if not cnt:
        return None
    return any(cnt[g] < tot[g] for g in cnt)
'''
for old, new in ((old_gather, new_gather), (old_poll, new_poll)):
    assert src.count(old) == 1, "core.py hunk not found: " + old.splitlines()[0]
    src = src.replace(old, new)
anchor = '_REAX_POLL_LOG = int(os.environ.get("REAX_CORE_POLL_LOG_EVERY", "256"))\n'
assert src.count(anchor) == 1, "core_poll constants not found"
src = src.replace(anchor, anchor + helper)
open(path, "w").write(src)
print("core_group: installed")
