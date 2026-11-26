"""Advisory locks on an open file that work on POSIX and Windows.

fcntl.flock on POSIX; msvcrt.locking on Windows, which locks one byte at offset 0
(the file position is saved and restored). Both are advisory: they coordinate our
own processes, not arbitrary writers.
"""
import contextlib
import errno
import os
import time
from typing import IO, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

# Errnos that mean "another process holds the lock", not a bad call.
_BUSY_ERRNOS = {errno.EACCES, errno.EAGAIN}
_BUSY_ERRNOS.update(
    value for value in (
        getattr(errno, name, None) for name in ("EWOULDBLOCK", "EDEADLK", "EDEADLOCK")
    ) if value is not None
)

_RETRY_SEC = 0.05


class LockBusy(BlockingIOError):
    """A non-blocking acquire found the file already locked."""


def acquire(file: IO[bytes], *, blocking: bool = True) -> None:
    """Take an exclusive advisory lock on an open binary file.

    blocking=True waits until the lock is free. blocking=False raises LockBusy
    immediately when another process holds it; other OSErrors propagate.
    """
    if os.name == "nt":
        _acquire_windows(file, blocking)
        return
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(file.fileno(), operation)
    except OSError as exc:
        if not blocking and exc.errno in _BUSY_ERRNOS:
            raise LockBusy(exc.errno, os.strerror(exc.errno)) from exc
        raise


def _acquire_windows(file: IO[bytes], blocking: bool) -> None:
    # LK_LOCK retries for about ten seconds internally, then fails; the loop makes
    # a blocking acquire wait like flock does. LK_NBLCK fails on the first try.
    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
    position = file.tell()
    try:
        file.seek(0)
        while True:
            try:
                msvcrt.locking(file.fileno(), mode, 1)
                return
            except OSError as exc:
                if exc.errno not in _BUSY_ERRNOS:
                    raise
                if not blocking:
                    raise LockBusy(exc.errno, os.strerror(exc.errno)) from exc
                time.sleep(_RETRY_SEC)
    finally:
        file.seek(position)


def release(file: IO[bytes]) -> None:
    """Release the lock taken by acquire()."""
    if os.name == "nt":
        position = file.tell()
        try:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            file.seek(position)
        return
    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(file: IO[bytes], *, blocking: bool = True) -> Iterator[None]:
    """Hold an exclusive advisory lock on `file` for the duration of the block."""
    acquire(file, blocking=blocking)
    try:
        yield
    finally:
        release(file)
