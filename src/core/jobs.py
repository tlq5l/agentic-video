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


def track(thread: threading.Thread) -> threading.Thread:
    """Register a job thread (not started yet) so finish() waits for it. Returns the thread."""
    with _LOCK:
        _THREADS[:] = [item for item in _THREADS if item.is_alive()]
        _THREADS.append(thread)
    return thread


def cancelled() -> bool:
    """True once finish() has begun: a job should stop at its next checkpoint and start no new unit of work."""
    return _CANCEL.is_set()


def finish(bound_sec: float = SHUTDOWN_BOUND_SEC) -> List[str]:
    """Ask the jobs to stop, then wait for every tracked thread up to bound_sec. Returns the names of the threads still alive
    (empty when every job finished)."""
    _CANCEL.set()
    deadline = time.monotonic() + bound_sec
    with _LOCK:
        threads = list(_THREADS)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [thread.name for thread in threads if thread.is_alive()]
