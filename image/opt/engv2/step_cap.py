import sys

OLD = '''                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break
'''
NEW = '''                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break
                if _REAX_STEP_NEW_MAX and len(scheduled_new_reqs) >= _REAX_STEP_NEW_MAX:
                    break  # engv2 step_cap.py: two queued REAX groups run as two steps, not one
'''
HEAD = '''import os as _reax_sc_os

_REAX_STEP_NEW_MAX = int(_reax_sc_os.environ.get("REAX_STEP_NEW_MAX", "0") or 0)
'''

path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "waiting loop not found"
src = src.replace(OLD, NEW)
anchor = "logger = init_logger(__name__)\n"
assert src.count(anchor) == 1, "logger anchor not found"
src = src.replace(anchor, anchor + "\n" + HEAD)
open(path, "w").write(src)
print("step_cap: installed")
