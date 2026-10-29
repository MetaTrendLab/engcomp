import sys

path = sys.argv[1]
src = open(path).read()
old_tail = '''        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
'''
new_tail = '''        if waited:
            logger.debug("EngineCore loop active.")

        sch = self.scheduler
        if _REAX_GATHER_GAP > 0 and not getattr(sch, "running", True) and len(getattr(sch, "waiting", ())) > 0:
            n, t_end = len(sch.waiting), time.monotonic() + _REAX_GATHER_MAX
            while True:
                left = t_end - time.monotonic()
                if left <= 0:
                    break
                try:
                    req = self.input_queue.get(timeout=min(_REAX_GATHER_GAP, left))
                except queue.Empty:
                    break
                self._handle_client_request(*req)
                n += 1
            _REAX_GATHER_STATS[0] += 1
            _REAX_GATHER_STATS[1] += n
            if _REAX_GATHER_STATS[0] % _REAX_GATHER_LOG == 0:
                logger.info("engv2 core gather: %d wave starts, %.2f requests each", _REAX_GATHER_STATS[0],
                            _REAX_GATHER_STATS[1] / _REAX_GATHER_STATS[0])

        # Handle any more client requests.
'''
assert src.count(old_tail) == 1, "core.py hunk not found"
src = src.replace(old_tail, new_tail)
src = src.replace("\nimport msgspec\n", """
import msgspec

_REAX_GATHER_GAP = float(os.environ.get("REAX_CORE_GATHER_US", "400")) / 1e6
_REAX_GATHER_MAX = float(os.environ.get("REAX_CORE_GATHER_MAX_MS", "4")) / 1e3
_REAX_GATHER_STATS = [0, 0]
_REAX_GATHER_LOG = int(os.environ.get("REAX_CORE_GATHER_LOG_EVERY", "256"))
""", 1)
open(path, "w").write(src)
print("core_gather: installed")
