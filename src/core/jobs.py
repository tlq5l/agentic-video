"""Background jobs of the stdio server: each job thread is tracked here, so the server can wait for them when its stdin closes.

A job checks `cancelled()` at its checkpoints; `finish()` sets the flag, waits for every tracked thread up to a bound, and reports the ones still alive.
"""
import threading
import time
from typing import List

# Longest a closed stdin may hold the server open. It must cover one job's longest FFmpeg wait: visual.py's sample_chunk
# call times out after 600 s, and the embedding of its chunk comes after that wait.
SHUTDOWN_BOUND_SEC = 660.0

_CANCEL = threading.Event()
_LOCK = threading.Lock()
_THREADS: List[threading.Thread] = []


def _is_job(thread) -> bool:
    # A test may swap threading.Thread for a stand-in without is_alive; such a stand-in is not a job, so it is never tracked or waited for.
    return hasattr(thread, "is_alive")


def _alive(thread) -> bool:
    return _is_job(thread) and bool(thread.is_alive())


def _started(thread) -> bool:
    # ident is set once the thread has started running.
    return getattr(thread, "ident", None) is not None


def track(thread: threading.Thread) -> threading.Thread:
    """Register a job thread (not started yet) so finish() waits for it. Returns the thread."""
    with _LOCK:
        # Drop only threads that ran and ended: a thread another caller tracked but has not started yet stays.
        _THREADS[:] = [item for item in _THREADS if not (_started(item) and not _alive(item))]
        if _is_job(thread):
            _THREADS.append(thread)
    return thread


def untrack(thread: threading.Thread) -> None:
    """Drop a tracked thread whose start() failed: it never ran, so finish() must not wait for it."""
    with _LOCK:
        _THREADS[:] = [item for item in _THREADS if item is not thread]


def cancelled() -> bool:
    """True once finish() has begun: a job should stop at its next checkpoint and start no new unit of work."""
    return _CANCEL.is_set()


def finish(bound_sec: float = SHUTDOWN_BOUND_SEC) -> List[str]:
    """Ask the jobs to stop, then wait for every tracked thread up to bound_sec. Returns the names of the threads still alive
    (empty when every job finished)."""
    _CANCEL.set()
    try:
        deadline = time.monotonic() + bound_sec
        # Each round takes a fresh snapshot: a job that starts another job as it ends tracks and starts it before it ends,
        # so the next snapshot holds it. A thread not started yet cannot be joined (Python raises), so it is only waited
        # for: a short sleep, then a new snapshot, until it starts or the bound passes.
        while True:
            with _LOCK:
                pending = [thread for thread in _THREADS if _alive(thread) or not _started(thread)]
            if not pending or time.monotonic() >= deadline:
                break
            running = [thread for thread in pending if _started(thread)]
            if running:
                for thread in running:
                    thread.join(max(0.0, deadline - time.monotonic()))
            else:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        with _LOCK:
            return [thread.name for thread in _THREADS if _alive(thread)]
    finally:
        # Cleared on return: a stdio run that happens in-process (tests) must not leave later jobs cancelled.
        _CANCEL.clear()
