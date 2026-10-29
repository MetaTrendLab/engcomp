import sys

path = sys.argv[1]
src = open(path).read()
old_exec = '''            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
'''
new_exec = old_exec + '''            _reax_settled = False
            if (_REAX_POLL_S > 0 and scheduler_output.total_num_scheduled_tokens == 0 and len(batch_queue) == 1
                    and not self.is_pooling_model and exec_future.done()):
                # engv2 core poll: settle the empty step now, then wait for the step in flight or new requests
                self._process_aborts_queue()
                for _out in self.scheduler.update_from_output(scheduler_output, exec_future.result()).items():
                    self.output_queue.put_nowait(_out)
                _reax_settled = True
                if _reax_poll(self, batch_queue[-1][0]):
                    return None, False
'''
old_queue = '''            if not deferred_scheduler_output:
                # Add this step's future to the queue.
'''
new_queue = '''            if not deferred_scheduler_output and not _reax_settled:
                # Add this step's future to the queue.
'''
for old, new in ((old_exec, new_exec), (old_queue, new_queue)):
    assert src.count(old) == 1, "core.py hunk not found: " + old.splitlines()[0]
    src = src.replace(old, new)
src = src.replace("\nimport msgspec\n", '''
import msgspec

_REAX_POLL_S = float(os.environ.get("REAX_CORE_POLL_US", "200")) / 1e6
_REAX_POLL_GAP = float(os.environ.get("REAX_CORE_POLL_GATHER_US", "400")) / 1e6
_REAX_POLL_MAX = float(os.environ.get("REAX_CORE_POLL_GATHER_MAX_MS", "4")) / 1e3
_REAX_POLL_STATS = [0, 0, 0]  # polls, polls that took in requests, requests taken in
_REAX_POLL_LOG = int(os.environ.get("REAX_CORE_POLL_LOG_EVERY", "256"))


def _reax_ready(fut):
    """The step in flight has finished: uniproc AsyncOutputFuture -> the async output's device-to-host copy event
    (model runner v2 `copy_event`, v1 `async_copy_ready_event`); a plain future -> done(). An AsyncOutputFuture is done()
    only once result() was called, so one without a known event counts as finished (vLLM's blocking wait)."""
    ao = getattr(fut, "async_output", None)
    if ao is None:
        return fut.done()
    ev = getattr(ao, "copy_event", None) or getattr(ao, "async_copy_ready_event", None)
    return ev.query() if ev is not None else True


def _reax_poll(core, fut):
    """True when requests were taken in (the caller returns so the busy loop schedules them behind the step in
    flight), False when the step in flight finished first (the caller collects it; requests taken in are scheduled on
    the next pass)."""
    st = _REAX_POLL_STATS
    st[0] += 1
    got, t_end = 0, 0.0
    try:
        while True:
            if _reax_ready(fut):
                return False
            if got:
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
    finally:
        st[1] += got > 0
        st[2] += got
        if st[0] % _REAX_POLL_LOG == 0:
            logger.info("engv2 core poll: %d polls, %d took in requests (%.2f each)", st[0], st[1],
                        st[2] / max(st[1], 1))
''', 1)
assert "_REAX_POLL_S = " in src, "import msgspec anchor not found"
open(path, "w").write(src)
print("core_poll: installed")
