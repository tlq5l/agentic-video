"""src.core.filelock holds the same lock contract on fcntl.flock (POSIX) and msvcrt.locking (Windows)."""
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.core import filelock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _child_code(path: str, blocking: bool) -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.core import filelock\n"
        f"with open({path!r}, 'a+b') as handle:\n"
        "    try:\n"
        f"        filelock.acquire(handle, blocking={blocking!r})\n"
        "    except filelock.LockBusy:\n"
        "        sys.exit(3)\n"
    )


def _child_acquire(path: Path, blocking: bool) -> int:
    """Exit code of a separate process's acquire: 0 on success, 3 on LockBusy."""
    return subprocess.run(
        [sys.executable, "-c", _child_code(str(path), blocking)],
        cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    ).returncode


def test_lock_blocks_other_process_until_release(tmp_path):
    lock_path = tmp_path / "held.lock"
    with open(lock_path, "a+b") as handle:
        with filelock.locked(handle):
            assert _child_acquire(lock_path, blocking=False) == 3
        assert _child_acquire(lock_path, blocking=False) == 0


def test_blocking_acquire_waits_for_release(tmp_path):
    lock_path = tmp_path / "blocking.lock"
    child = subprocess.Popen(
        [sys.executable, "-c", _child_code(str(lock_path), True)],
        cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)  # let the child reach its blocking acquire
        with open(lock_path, "a+b") as handle:
            filelock.acquire(handle)
            time.sleep(0.5)
            filelock.release(handle)
        assert child.wait(timeout=60) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_second_handle_same_process_cannot_lock(tmp_path):
    lock_path = tmp_path / "same.lock"
    with open(lock_path, "a+b") as first, open(lock_path, "a+b") as second:
        filelock.acquire(first)
        with pytest.raises(filelock.LockBusy):
            filelock.acquire(second, blocking=False)
        filelock.release(first)
        filelock.acquire(second, blocking=False)
        filelock.release(second)


def test_locked_releases_on_exception(tmp_path):
    lock_path = tmp_path / "raise.lock"
    with open(lock_path, "a+b") as handle:
        with pytest.raises(RuntimeError):
            with filelock.locked(handle):
                raise RuntimeError("boom")
    assert _child_acquire(lock_path, blocking=False) == 0
