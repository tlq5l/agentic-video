"""Fast local speech-to-text using faster-whisper with subtitle fallback."""
import hashlib
import errno
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from src.core.config import Config
from src.core.errors import ASRModelError, MediaNotFoundError, SubprocessTimeoutError
from src.core.types import TranscriptSegment, TranscriptResult, VideoMetadata
from src.core.cache import MediaCache
from src.media.probe import probe_video
from src.transcript.repeats import flag_repeated_segments

def parse_srt_or_vtt(subtitle_path: Path) -> List[TranscriptSegment]:
    """Parse SRT or WebVTT file into TranscriptSegments."""
    content = subtitle_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    segments = []

    def parse_timestamp(raw: str) -> Optional[float]:
        """Parse SRT HH:MM:SS,mmm and WebVTT HH:MM:SS.mmm/MM:SS.mmm."""
        token = raw.strip().split(maxsplit=1)[0] if raw.strip() else ""
        parts = re.split(r"[:,.]", token)
        if len(parts) == 4:
            h, m, s, fraction = parts
        elif len(parts) == 3:
            h, m, s = "0", parts[0], parts[1]
            fraction = parts[2]
        else:
            return None
        if not all(part.isdigit() for part in (h, m, s, fraction)):
            return None
        hours, minutes, seconds = int(h), int(m), int(s)
        if minutes >= 60 or seconds >= 60 or len(fraction) > 3:
            return None
        milliseconds = int(fraction.ljust(3, "0"))
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def parse_timing_line(line: str) -> Optional[tuple[float, float]]:
        if "-->" not in line:
            return None
        start_raw, end_raw = line.split("-->", 1)
        start = parse_timestamp(start_raw)
        end = parse_timestamp(end_raw)
        if start is None or end is None or start < 0 or end < start:
            return None
        return start, end
    
    idx = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        timing = parse_timing_line(line)
        if timing:
            start, end = timing
            
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append(TranscriptSegment(id=idx, start_sec=round(start, 3), end_sec=round(end, 3), text=text))
                idx += 1
        else:
            i += 1
    return segments


def _find_sidecar(video_path: Path, subtitle_override: Optional[str]) -> Optional[Path]:
    """Resolve the subtitle source before consulting the transcript cache."""
    if subtitle_override and Path(subtitle_override).exists():
        return Path(subtitle_override).resolve()
    for ext in [".vtt", ".srt"]:
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            return candidate.resolve()
    return None


def _subtitle_fingerprint(subtitle_path: Optional[Path]) -> Optional[str]:
    if subtitle_path is None:
        return None
    try:
        return hashlib.sha256(subtitle_path.read_bytes()).hexdigest()[:20]
    except OSError:
        return None


def _transcript_cache_payload(
    result: TranscriptResult,
    media_fingerprint: str,
    subtitle_fingerprint: Optional[str],
    model_size: Optional[str] = None,
) -> dict:
    payload = result.to_dict()
    # Bind a transcript to the exact media and sidecar bytes that produced it.
    # These private cache fields are not part of the public transcript wire shape.
    payload["_media_fingerprint"] = media_fingerprint
    payload["_subtitle_fingerprint"] = subtitle_fingerprint
    payload["_model_size"] = model_size if subtitle_fingerprint is None else None
    return payload


def _with_repeat_flags(result: TranscriptResult) -> TranscriptResult:
    """Flag repeated ASR lines on the way out; the cache keeps the unflagged transcript."""
    flagged = flag_repeated_segments(result.segments)
    if flagged:
        result.notice = f"{result.notice} {flagged}"
    return result


def _cached_result(
    cache: MediaCache,
    p: Path,
    media_fingerprint: str,
    subtitle_fingerprint: Optional[str],
    model_size: Optional[str],
    duration_sec: float,
) -> Optional[TranscriptResult]:
    """Load and validate one transcript cache record."""
    cache_name = (
        f"transcript_subtitle_{subtitle_fingerprint}"
        if subtitle_fingerprint is not None
        else f"transcript_asr_{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_size or '')}"
    )
    cached = cache.load_json(media_fingerprint, cache_name)
    if not (
        isinstance(cached, dict)
        and cached.get("_media_fingerprint") == media_fingerprint
        and cached.get("_subtitle_fingerprint") == subtitle_fingerprint
        and (
            subtitle_fingerprint is not None
            or cached.get("_model_size") == model_size
        )
    ):
        return None
    try:
        segments = [TranscriptSegment(**s) for s in cached.get("segments", [])]
        result = TranscriptResult(
            source_path=str(p),
            language=cached.get("language", "unknown"),
            duration_sec=float(cached.get("duration_sec", duration_sec)),
            segment_count=len(segments),
            segments=segments,
            has_speech=bool(cached.get("has_speech", bool(segments))),
            notice=cached.get(
                "notice",
                "Note: Non-speech audio (music, sound effects, alarms) is NOT transcribed.",
            ),
            confidence=cached.get("confidence"),
            uncertainty=cached.get("uncertainty", ""),
        )
    except (TypeError, ValueError, KeyError):
        # Treat malformed cache entries as misses; never serve partial metadata.
        return None
    # Subtitle lines come from a file a person wrote, so only ASR lines are flagged.
    return result if subtitle_fingerprint is not None else _with_repeat_flags(result)


def find_cached_transcript(
    source_path: str,
    model_size: str,
    subtitle_override: Optional[str] = None,
    media_metadata: Optional[VideoMetadata] = None,
) -> Optional[Tuple[TranscriptResult, Optional[str]]]:
    """Return the requested transcript or the best cached ASR model result.

    The model value is ``None`` for subtitle transcripts, whose cache is shared
    across model sizes. A cache hit is accepted only when bound to the current
    media and subtitle fingerprints.
    """
    p = Path(source_path).resolve()
    if not p.exists():
        raise MediaNotFoundError(f"Source media not found: {source_path}")

    meta = media_metadata or probe_video(str(p))
    subtitle_path = _find_sidecar(p, subtitle_override)
    subtitle_fp = _subtitle_fingerprint(subtitle_path)
    cache = MediaCache()
    if subtitle_fp is not None:
        result = _cached_result(
            cache, p, meta.fingerprint, subtitle_fp, None, meta.duration_sec
        )
        return (result, None) if result is not None else None

    model_priority = ["turbo", "small", "base", "tiny"]
    if model_size in model_priority:
        model_priority.remove(model_size)
    model_priority.insert(0, model_size)
    for cached_model in model_priority:
        result = _cached_result(
            cache, p, meta.fingerprint, None, cached_model, meta.duration_sec
        )
        if result is not None:
            return result, cached_model
    return None


class _BackgroundTranscriptionSlot:
    """One non-queued ASR worker slot, shared by server processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[Tuple[str, str]] = None
        self._failures: Dict[Tuple[str, str], str] = {}

    def start(self, source_path: str, model_size: str) -> Tuple[str, Optional[str]]:
        key = (str(Path(source_path).resolve()), model_size)
        with self._lock:
            if self._active is not None:
                return ("in_progress" if self._active[0] == key[0] else "busy"), None
            if key in self._failures:
                error = self._failures.pop(key)
                return "failed", error
            process_lock, owner, lock_error = _acquire_background_process_lock(
                Config.BASE_CACHE_DIR, key[0], key[1]
            )
            if process_lock is None:
                if lock_error:
                    return "start_failed", lock_error
                same_video = owner and owner.get("source_path") == key[0]
                return ("in_progress" if same_video else "busy"), None
            self._active = key

        def run() -> None:
            _ASR_WORKER_LOCK.lock_file = process_lock
            try:
                result = transcribe_video(
                    key[0], model_size=key[1], allow_long_run=True
                )
                if result.uncertainty == ASR_WORKER_FAILURE_UNCERTAINTY:
                    raise ASRModelError(
                        ASR_WORKER_FAILURE_NOTICE,
                        {"source_path": key[0], "model_size": key[1]},
                    )
            except Exception as exc:
                with self._lock:
                    self._failures[key] = f"{type(exc).__name__}: {exc}"
                    while len(self._failures) > 16:
                        del self._failures[next(iter(self._failures))]
            finally:
                _ASR_WORKER_LOCK.lock_file = None
                _release_background_process_lock(process_lock)
                with self._lock:
                    if self._active == key:
                        self._active = None

        worker = threading.Thread(
            target=run,
            name="agentic-video-asr",
            daemon=True,
        )
        try:
            worker.start()
        except Exception:
            _release_background_process_lock(process_lock)
            with self._lock:
                if self._active == key:
                    self._active = None
            return "start_failed", "The worker thread could not be started."
        return "started", None


def _read_background_lock_owner(lock_file: Any) -> Optional[dict]:
    try:
        lock_file.seek(1 if os.name == "nt" else 0)
        value = json.loads(lock_file.read().decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _acquire_background_process_lock(
    cache_dir: Path, source_path: str, model_size: str
) -> Tuple[Optional[Any], Optional[dict], Optional[str]]:
    """Take the machine-wide cache lock without waiting for another worker."""
    lock_file = None
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_file = open(cache_dir / "asr-worker.lock", "a+b")
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                lock_file.seek(0)
                owner = _read_background_lock_owner(lock_file)
                lock_file.close()
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    return None, owner, None
                return None, None, f"Could not lock the ASR worker slot: {exc}"
        else:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                lock_file.seek(0)
                owner = _read_background_lock_owner(lock_file)
                lock_file.close()
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    return None, owner, None
                return None, None, f"Could not lock the ASR worker slot: {exc}"

        metadata_offset = 1 if os.name == "nt" else 0
        lock_file.seek(metadata_offset)
        lock_file.truncate(metadata_offset)
        lock_file.write(json.dumps({
            "pid": os.getpid(),
            "source_path": source_path,
            "model_size": model_size,
            "started_at": time.time(),
        }).encode("utf-8"))
        lock_file.flush()
        os.fsync(lock_file.fileno())
        return lock_file, None, None
    except OSError as exc:
        if lock_file is not None:
            try:
                lock_file.close()
            except OSError:
                pass
        return None, None, f"Could not start the ASR worker: {exc}"


def _release_background_process_lock(lock_file: Any) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        lock_file.close()


_BACKGROUND_TRANSCRIPTION_SLOT = _BackgroundTranscriptionSlot()
_ASR_WORKER_LOCK = threading.local()
TRANSCRIPTION_RETRY_AFTER_SECONDS = 15
ASR_CHUNK_SECONDS = 60
# "turbo" runs whisper.cpp (large-v3-turbo) on the GPU through whisper-cli: ~50x real time on Apple silicon.
WHISPER_CPP_MODEL = Path(os.environ.get(
    "AGENTIC_VIDEO_WHISPER_CPP_MODEL", "~/.cache/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"
)).expanduser()
ASR_CHUNK_OVERLAP_SECONDS = 3
ASR_WORKER_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
# Keep the single background slot finite even if model loading or decoding hangs.
ASR_BACKGROUND_TIMEOUT_SECONDS = 6 * 60 * 60
ASR_WORKER_FAILURE_NOTICE = (
    "ASR worker exited without a transcript; speech presence and timestamps are unknown."
)
ASR_WORKER_FAILURE_UNCERTAINTY = (
    "ASR worker exited without a result; speech presence and timestamps are unknown."
)


def _process_rss_bytes(pid: int) -> Optional[int]:
    """Read one process's resident set size without adding a dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)

    if sys.platform == "darwin":
        import ctypes
        import struct

        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            proc_pid_rusage = libproc.proc_pid_rusage
            proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            proc_pid_rusage.restype = ctypes.c_int
            # RUSAGE_INFO_V2 is larger than the field we need; reserve ample
            # space for the complete native struct and read ri_resident_size.
            usage = ctypes.create_string_buffer(512)
            if proc_pid_rusage(pid, 2, usage) != 0:
                return None
            return int(struct.unpack_from("Q", usage.raw, 64)[0])
        except (AttributeError, OSError, struct.error):
            return None

    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/statm", "r", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None

    # Unknown platforms fail closed: an uncapped background worker must not
    # run when its memory use cannot be measured.
    return None


def _process_group_rss_bytes(pid: int) -> Optional[int]:
    """Read aggregate RSS for the isolated worker and its decoder children."""
    if os.name == "nt":
        # A Windows Job Object enforces the aggregate hard limit. This sampled
        # value is retained for diagnostics; it is not the enforcement path.
        return _process_rss_bytes(pid)

    if sys.platform == "darwin":
        import ctypes

        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            proc_listpids = libproc.proc_listpids
            proc_listpids.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int
            ]
            proc_listpids.restype = ctypes.c_int
            process_ids = (ctypes.c_int * 4096)()
            byte_count = proc_listpids(
                2, pid, process_ids, ctypes.sizeof(process_ids)
            )  # PROC_PGRP_ONLY; each worker starts a fresh process group.
            if byte_count <= 0 or byte_count >= ctypes.sizeof(process_ids):
                return None
            pids = [
                process_ids[i]
                for i in range(byte_count // ctypes.sizeof(ctypes.c_int))
                if process_ids[i] > 0
            ]
            if pid not in pids:
                return None
            rss_values = [_process_rss_bytes(member) for member in pids]
            if rss_values[pids.index(pid)] is None:
                return None
            return sum(value for value in rss_values if value is not None)
        except (AttributeError, OSError, ValueError):
            return None

    if sys.platform.startswith("linux"):
        total = 0
        found_worker = False
        page_size = os.sysconf("SC_PAGE_SIZE")
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
                    process_group = int(fields[2])
                    resident_pages = int(fields[21])
                except (OSError, ValueError, IndexError):
                    continue  # Process exited while the process table was read.
                if process_group == pid:
                    total += resident_pages * page_size
                    found_worker = found_worker or int(entry.name) == pid
        except OSError:
            return None
        return total if found_worker else None

    # Unknown platforms retain the previous fail-closed single-process check.
    return _process_rss_bytes(pid)


class _WindowsWorkerJob:
    """A kill-on-close Windows Job Object with an aggregate memory limit."""

    def __init__(self, memory_limit_bytes: int) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        kernel32.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID, wintypes.DWORD, ctypes.c_size_t, wintypes.LPVOID,
            ctypes.c_size_t, wintypes.LPVOID, ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        kernel32.DeleteProcThreadAttributeList.restype = None
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.LPVOID, wintypes.LPVOID,
            wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR,
            wintypes.LPVOID, wintypes.LPVOID,
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL

        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = ExtendedLimitInformation()
        # JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.LimitFlags = 0x0200 | 0x2000
        info.JobMemoryLimit = memory_limit_bytes
        if not kernel32.SetInformationJobObject(
            self._handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):  # JobObjectExtendedLimitInformation
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "SetInformationJobObject failed")

    def spawn(
        self,
        command: List[str],
        stdout_file: Any,
        stderr_file: Any,
        environment: Dict[str, str],
    ) -> "_WindowsProcess":
        """Create the worker directly in this job before its first instruction."""
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class SecurityAttributes(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        class StartupInfo(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD),
                ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD),
                ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE),
            ]

        class StartupInfoEx(ctypes.Structure):
            _fields_ = [
                ("StartupInfo", StartupInfo),
                ("lpAttributeList", wintypes.LPVOID),
            ]

        class ProcessInformation(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE),
                ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD),
            ]

        kernel32 = self._kernel32
        null_security = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), None, True
        )
        null_handle = kernel32.CreateFileW(
            "NUL", 0x80000000, 0x00000003, ctypes.byref(null_security),
            3, 0x00000080, None,
        )
        if not null_handle or null_handle == wintypes.HANDLE(-1).value:
            raise OSError(self._ctypes.get_last_error(), "Could not open NUL for ASR stdin")

        std_handles = [
            null_handle,
            msvcrt.get_osfhandle(stdout_file.fileno()),
            msvcrt.get_osfhandle(stderr_file.fileno()),
        ]
        previous_inherit = []
        process_info = ProcessInformation()
        attributes = None
        attributes_initialized = False
        try:
            for handle in std_handles[1:]:
                was_inheritable = os.get_handle_inheritable(handle)
                previous_inherit.append((handle, was_inheritable))
                if not was_inheritable:
                    os.set_handle_inheritable(handle, True)

            required_size = ctypes.c_size_t(0)
            kernel32.InitializeProcThreadAttributeList(
                None, 2, 0, ctypes.byref(required_size)
            )
            if required_size.value <= 0:
                raise OSError(
                    self._ctypes.get_last_error(),
                    "InitializeProcThreadAttributeList did not return a size",
                )
            attributes = ctypes.create_string_buffer(required_size.value)
            if not kernel32.InitializeProcThreadAttributeList(
                attributes, 2, 0, ctypes.byref(required_size)
            ):
                raise OSError(
                    self._ctypes.get_last_error(),
                    "InitializeProcThreadAttributeList failed",
                )
            attributes_initialized = True

            job_list = (wintypes.HANDLE * 1)(self._handle)
            handle_list = (wintypes.HANDLE * len(std_handles))(*std_handles)
            # PROC_THREAD_ATTRIBUTE_JOB_LIST and _HANDLE_LIST. The job is
            # applied atomically by CreateProcessW before worker code can run.
            if not kernel32.UpdateProcThreadAttribute(
                attributes, 0, 0x0002000D, ctypes.byref(job_list),
                ctypes.sizeof(job_list), None, None,
            ):
                raise OSError(
                    self._ctypes.get_last_error(),
                    "Could not attach the ASR Job Object at process creation",
                )
            if not kernel32.UpdateProcThreadAttribute(
                attributes, 0, 0x00020002, ctypes.byref(handle_list),
                ctypes.sizeof(handle_list), None, None,
            ):
                raise OSError(
                    self._ctypes.get_last_error(),
                    "Could not restrict inherited ASR worker handles",
                )

            startup = StartupInfoEx()
            startup.StartupInfo.cb = ctypes.sizeof(StartupInfoEx)
            startup.StartupInfo.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = null_handle
            startup.StartupInfo.hStdOutput = std_handles[1]
            startup.StartupInfo.hStdError = std_handles[2]
            startup.lpAttributeList = ctypes.cast(attributes, wintypes.LPVOID)
            command_line = ctypes.create_unicode_buffer(
                subprocess.list2cmdline(command)
            )
            env_values = sorted(environment.items(), key=lambda item: item[0].upper())
            environment_block = ctypes.create_unicode_buffer(
                "\0".join(f"{key}={value}" for key, value in env_values) + "\0\0"
            )
            flags = 0x00080000 | 0x00000400 | 0x08000000
            # EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW
            if not kernel32.CreateProcessW(
                command[0], command_line, None, None, True, flags,
                environment_block, None, ctypes.byref(startup),
                ctypes.byref(process_info),
            ):
                raise OSError(self._ctypes.get_last_error(), "CreateProcessW failed")
            kernel32.CloseHandle(process_info.hThread)
            process_info.hThread = None
            return _WindowsProcess(
                kernel32, process_info.hProcess, process_info.dwProcessId, command
            )
        except Exception:
            if process_info.hProcess:
                kernel32.TerminateProcess(process_info.hProcess, 1)
                kernel32.CloseHandle(process_info.hProcess)
            if process_info.hThread:
                kernel32.CloseHandle(process_info.hThread)
            raise
        finally:
            if attributes_initialized:
                kernel32.DeleteProcThreadAttributeList(attributes)
            kernel32.CloseHandle(null_handle)
            for handle, was_inheritable in previous_inherit:
                if not was_inheritable:
                    os.set_handle_inheritable(handle, False)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _WindowsProcess:
    """Small Popen-compatible wrapper for a CreateProcessW job member."""

    def __init__(self, kernel32: Any, handle: Any, pid: int, command: List[str]):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._handle = handle
        self.pid = pid
        self.args = command
        self.returncode = None
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
        ]
        kernel32.GetExitCodeProcess.restype = self._ctypes_bool_type()
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = self._ctypes_dword_type()
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.TerminateProcess.restype = self._ctypes_bool_type()
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = self._ctypes_bool_type()

    @staticmethod
    def _ctypes_dword_type():
        from ctypes import wintypes

        return wintypes.DWORD

    @staticmethod
    def _ctypes_bool_type():
        from ctypes import wintypes

        return wintypes.BOOL

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        from ctypes import wintypes

        exit_code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(
            self._handle, self._ctypes.byref(exit_code)
        ):
            raise OSError(self._ctypes.get_last_error(), "GetExitCodeProcess failed")
        if exit_code.value != 259:  # STILL_ACTIVE
            self.returncode = int(exit_code.value)
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        milliseconds = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        result = self._kernel32.WaitForSingleObject(self._handle, milliseconds)
        if result == 0x00000102:  # WAIT_TIMEOUT
            raise subprocess.TimeoutExpired(self.args, timeout)
        if result != 0:
            raise OSError(self._ctypes.get_last_error(), "WaitForSingleObject failed")
        return int(self.poll())

    def kill(self) -> None:
        if self.poll() is None:
            self._kernel32.TerminateProcess(self._handle, 1)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _stop_worker_process(process: subprocess.Popen) -> None:
    """Stop the ASR process and any FFmpeg child it started."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            if process.poll() is None:
                process.kill()
    except (ProcessLookupError, FileNotFoundError, subprocess.TimeoutExpired):
        if os.name == "nt" and process.poll() is None:
            process.kill()
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_memory_capped_worker(
    command: List[str],
    timeout: Optional[float],
    memory_limit_bytes: int = ASR_WORKER_MEMORY_LIMIT_BYTES,
) -> subprocess.CompletedProcess:
    """Run isolated ASR while enforcing an RSS ceiling from the host process."""
    worker_job = (
        _WindowsWorkerJob(memory_limit_bytes) if os.name == "nt" else None
    )
    process = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_file:
            if worker_job is not None:
                process = worker_job.spawn(
                    command,
                    stdout_file,
                    stderr_file,
                    {
                        **os.environ,
                        "AGENTIC_VIDEO_ASR_PARENT_PID": str(os.getpid()),
                    },
                )
            else:
                lock_file = getattr(_ASR_WORKER_LOCK, "lock_file", None)
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=(os.name == "posix"),
                    # The child keeps the same flock open if the MCP server
                    # dies. A replacement server cannot start a second ASR
                    # worker during the watchdog's shutdown interval.
                    pass_fds=(lock_file.fileno(),)
                    if lock_file is not None and os.name == "posix" else (),
                    env={
                        **os.environ,
                        "AGENTIC_VIDEO_ASR_PARENT_PID": str(os.getpid()),
                    },
                )

            deadline = None if timeout is None else time.monotonic() + timeout
            while process.poll() is None:
                if deadline is not None and time.monotonic() >= deadline:
                    _stop_worker_process(process)
                    raise subprocess.TimeoutExpired(command, timeout)

                rss_bytes = _process_group_rss_bytes(process.pid)
                if rss_bytes is None:
                    if process.poll() is not None:
                        break
                    _stop_worker_process(process)
                    raise ASRModelError(
                        "Unable to measure ASR worker memory; stopped the worker to preserve the memory limit.",
                        {"memory_limit_bytes": memory_limit_bytes},
                    )
                if rss_bytes > memory_limit_bytes:
                    _stop_worker_process(process)
                    raise ASRModelError(
                        f"ASR worker process group exceeded its {memory_limit_bytes // (1024 * 1024)} MiB memory limit.",
                        {
                            "memory_limit_bytes": memory_limit_bytes,
                            "observed_rss_bytes": rss_bytes,
                        },
                    )

                sleep_for = 1.0
                if deadline is not None:
                    sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
                time.sleep(sleep_for)

            return_code = process.wait()
            if return_code != 0 and os.name == "posix":
                # The worker can crash while FFmpeg is still decoding. Its child
                # shares this process group, so reap it before opening another job.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout_file.seek(0)
            stderr_file.seek(0)
            return subprocess.CompletedProcess(
                command,
                return_code,
                stdout_file.read().decode("utf-8", errors="replace"),
                stderr_file.read().decode("utf-8", errors="replace"),
            )
    finally:
        # Job Object closure kills any Windows descendants left after a crash.
        if worker_job is not None:
            worker_job.close()
        if process is not None and hasattr(process, "close"):
            process.close()


def _whisper_installed(size: str) -> bool:
    """True if the weights for this model size are installed locally."""
    if size == "turbo":
        import shutil
        return shutil.which("whisper-cli") is not None and WHISPER_CPP_MODEL.is_file()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    return any((Path(HF_HUB_CACHE) / f"models--Systran--faster-whisper-{size}" / "snapshots").glob("*/model.bin"))


def _installed_whisper_models() -> List[str]:
    """Fallback model sizes that are installed locally, best first."""
    return [size for size in ("turbo", "small", "base", "tiny") if _whisper_installed(size)]


def default_model_size() -> str:
    """Model used when an agent names none: turbo where whisper.cpp is installed, else tiny."""
    return "turbo" if _whisper_installed("turbo") else Config.DEFAULT_WHISPER_MODEL


# Rough ASR speed as a share of the audio's length (measured: turbo ~3 min for a 65 min video).
_ASR_SPEED_VS_AUDIO = {"turbo": 0.05, "tiny": 0.04, "base": 0.08, "small": 0.2}


def _transcription_progress(duration_sec: float, model_size: str) -> Dict[str, Any]:
    """Elapsed time, an estimated total and a percent for the running worker (empty if unknown)."""
    try:
        with open(Path(Config.BASE_CACHE_DIR) / "asr-worker.lock", "rb") as handle:
            owner = _read_background_lock_owner(handle) or {}
        elapsed = max(0.0, time.time() - float(owner["started_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    total = max(10.0, duration_sec * _ASR_SPEED_VS_AUDIO.get(model_size, 0.2))
    return {
        "elapsed_sec": round(elapsed),
        "estimated_total_sec": round(total),
        "estimated_percent": min(99, int(100 * elapsed / total)),
    }


def request_transcription(source_path: str, model_size: str) -> Dict[str, Any]:
    """Serve cached MCP transcripts immediately and schedule uncached ASR."""
    installed = [] if _whisper_installed(model_size) else _installed_whisper_models()
    if installed:
        # Offline machines can't fetch weights; run the best installed model and say so.
        data = request_transcription(source_path, installed[0])
        data["model_substitution"] = (
            f"The {model_size} model is not installed here; used {installed[0]} instead."
        )
        return data
    p = Path(source_path).resolve()
    meta = probe_video(str(p))
    cached = find_cached_transcript(source_path, model_size, media_metadata=meta)
    if cached is not None:
        result, cached_model = cached
        if cached_model is None or cached_model == model_size:
            # Preserve the established wire shape for model and subtitle hits.
            return result.to_dict()
    else:
        if _find_sidecar(p, None) is not None or not meta.has_audio:
            # Subtitle parsing and no-audio handling are quick and synchronous.
            return transcribe_video(str(p), model_size=model_size).to_dict()
        result = None
        cached_model = None

    state, error = _BACKGROUND_TRANSCRIPTION_SLOT.start(source_path, model_size)
    retry_after = 0 if state == "failed" else TRANSCRIPTION_RETRY_AFTER_SECONDS
    message_by_state = {
        "started": f"The {model_size} transcription started in the background.",
        "in_progress": "A transcription for this video is already running.",
        "busy": "Another video's transcription is running; this request was not queued.",
        "start_failed": "The background transcription could not be started; retry the request.",
        "failed": (
            f"The previous {model_size} transcription failed: {error or 'unknown error'}. "
            "Retry this request to start another attempt."
        ),
    }
    message = message_by_state[state]
    if result is not None:
        data = result.to_dict()
        data.update({
            "requested_model_size": model_size,
            "cached_model_size": cached_model,
            "requested_model_cached": False,
            "transcription_status": state,
            "retry_after_seconds": retry_after,
            "transcription_message": (
                f"Returning the cached {cached_model} transcript because the requested "
                f"{model_size} transcript is not cached. {message}"
            ),
        })
        return data

    running = state in {"started", "in_progress"}
    progress = _transcription_progress(meta.duration_sec, model_size) if running else {}
    if progress:
        message += (
            f" About {progress['estimated_percent']}% done (estimate): {progress['elapsed_sec']} s elapsed "
            f"of roughly {progress['estimated_total_sec']} s."
        )
    return {
        "status": "in_progress" if running else state,
        "requested_model_size": model_size,
        "requested_model_cached": False,
        "transcription_status": state,
        "retry_after_seconds": retry_after,
        "transcription_message": (
            message if state == "failed"
            else f"{message} Retry in about {retry_after} seconds."
        ),
        **progress,
    }


def _deduplicate_chunk_segments(raw_segments: List[dict]) -> List[dict]:
    """Drop textually repeated spans produced from adjacent overlapping chunks."""
    def tokens(segment: dict) -> list[str]:
        normalized = re.sub(r"[^\w\s]", "", str(segment.get("text", "")).casefold())
        return normalized.split()

    def same_utterance(left: dict, right: dict) -> bool:
        a, b = tokens(left), tokens(right)
        if not a or not b:
            return False
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        return any(
            long[i:i + len(short)] == short
            for i in range(len(long) - len(short) + 1)
        )

    kept: List[dict] = []
    for raw in raw_segments:
        current = dict(raw)
        chunk_index = current.get("chunk_index")
        duplicate = False
        if chunk_index is not None:
            for previous_index in range(len(kept) - 1, -1, -1):
                previous = kept[previous_index]
                previous_chunk = previous.get("chunk_index")
                if previous_chunk is None or previous_chunk >= chunk_index:
                    continue
                if previous_chunk < chunk_index - 1:
                    break
                current_duration = float(current["end_sec"]) - float(current["start_sec"])
                previous_duration = float(previous["end_sec"]) - float(previous["start_sec"])
                if abs(float(current["start_sec"]) - float(previous["start_sec"])) > 1.0:
                    continue
                overlap = min(float(current["end_sec"]), float(previous["end_sec"])) - max(
                    float(current["start_sec"]), float(previous["start_sec"])
                )
                shorter = min(current_duration, previous_duration)
                if shorter > 0 and overlap >= shorter * 0.5 and same_utterance(current, previous):
                    if (len(tokens(current)), current_duration) > (len(tokens(previous)), previous_duration):
                        # The later chunk may have completed a phrase clipped
                        # by the earlier chunk; keep the fuller timestamp span.
                        kept[previous_index] = current
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(current)
    kept.sort(key=lambda segment: (float(segment["start_sec"]), float(segment["end_sec"])))
    for index, segment in enumerate(kept, 1):
        segment.pop("chunk_index", None)
        segment["id"] = index
    return kept

def transcribe_video(
    source_path: str,
    model_size: str = Config.DEFAULT_WHISPER_MODEL,
    subtitle_override: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    allow_long_run: bool = False,
) -> TranscriptResult:
    """Transcribe video audio using faster-whisper or external subtitles.

    An explicit timeout requests a fresh ASR attempt so a previous cached result
    cannot make timeout/failure diagnostics meaningless. ``allow_long_run`` is
    reserved for the MCP background worker to disable its short wall timeout.
    All isolated workers are RSS-supervised and decode bounded audio chunks.
    """
    p = Path(source_path).resolve()
    if not p.exists():
        raise MediaNotFoundError(f"Source media not found: {source_path}")

    meta = probe_video(str(p))
    sub_path = _find_sidecar(p, subtitle_override)
    subtitle_fp = _subtitle_fingerprint(sub_path)
    cache = MediaCache()
    cache_name = (
        f"transcript_subtitle_{subtitle_fp}"
        if sub_path is not None and subtitle_fp
        else f"transcript_asr_{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_size)}"
    )
    # A caller-supplied timeout is an explicit fresh-run request. Cache records
    # without binding metadata are rejected once for safe migration.
    if timeout_sec is None:
        cached = _cached_result(
            cache,
            p,
            meta.fingerprint,
            subtitle_fp,
            None if subtitle_fp is not None else model_size,
            meta.duration_sec,
        )
        if cached is not None:
            return cached

    if sub_path:
        segments = parse_srt_or_vtt(sub_path)
        res = TranscriptResult(
            source_path=str(p),
            language="en",
            duration_sec=meta.duration_sec,
            segment_count=len(segments),
            segments=segments,
            has_speech=len(segments) > 0,
            confidence=1.0,
            uncertainty="Subtitle sidecar used; confidence reflects source availability, not semantic accuracy.",
        )
        cache.save_json(meta.fingerprint, cache_name, _transcript_cache_payload(res, meta.fingerprint, subtitle_fp, model_size))
        return res
        
    if not meta.has_audio:
        empty_res = TranscriptResult(
            source_path=str(p),
            language="none",
            duration_sec=meta.duration_sec,
            segment_count=0,
            segments=[],
            has_speech=False,
            notice="No audio stream is present; speech content is unavailable.",
            confidence=None,
            uncertainty="No audio stream; absence of transcript is not evidence that a speech track would be empty.",
        )
        cache.save_json(meta.fingerprint, cache_name, _transcript_cache_payload(empty_res, meta.fingerprint, subtitle_fp, model_size))
        return empty_res

    # Isolate faster-whisper from the host process and its native libraries.
    worker_script = f"""
import json, os, signal, subprocess, sys, tempfile, threading, wave
decoder = None
parent_pid = int(os.environ.get("AGENTIC_VIDEO_ASR_PARENT_PID", os.getppid()))
watchdog_stop = threading.Event()
def stop_if_parent_exited():
    if os.getppid() != parent_pid:
        if os.name == "posix":
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
        if decoder is not None:
            try:
                decoder.terminate()
                decoder.wait(timeout=2)
            except Exception:
                try:
                    decoder.kill()
                except Exception:
                    pass
        os._exit(1)
def parent_watchdog():
    while not watchdog_stop.wait(0.5):
        stop_if_parent_exited()
stop_if_parent_exited()
watchdog = threading.Thread(target=parent_watchdog, name="asr-parent-watchdog", daemon=True)
watchdog.start()
exit_code = 1
def run_whisper_cpp():
    # Decode to one 16 kHz WAV, then let whisper-cli segment it; no text context is
    # carried between windows, which stops speaker labels repeating down the transcript.
    global decoder
    ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="agentic-video-asr-") as temp_dir:
        wav = os.path.join(temp_dir, "audio.wav")
        decoder = subprocess.Popen(
            [ffmpeg, "-nostdin", "-v", "error", "-i", {repr(str(p))}, "-map", "0:a:0", "-vn",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if decoder.wait() != 0:
            raise RuntimeError("FFmpeg could not decode the audio stream")
        decoder = subprocess.Popen(
            ["whisper-cli", "-m", {repr(str(WHISPER_CPP_MODEL))}, "-t", "8", "-l", "auto", "-mc", "0",
             "-sns", "-ojf", "-of", os.path.join(temp_dir, "out"), "-f", wav],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        _, err = decoder.communicate()
        if decoder.returncode != 0:
            raise RuntimeError("whisper-cli failed: " + err.decode(errors="replace")[-300:])
        with open(os.path.join(temp_dir, "out.json")) as fh:
            data = json.load(fh)
    segments = []
    for item in data.get("transcription", []):
        t = item.get("text", "").strip()
        probs = [tok["p"] for tok in item.get("tokens", []) if not tok.get("text", "").startswith("[_")]
        if t:
            segments.append({{"id": len(segments) + 1,
                "start_sec": round(item["offsets"]["from"] / 1000, 3),
                "end_sec": round(item["offsets"]["to"] / 1000, 3), "text": t,
                "confidence": round(sum(probs) / len(probs), 3) if probs else None}})
    return segments, data.get("result", {{}}).get("language", "en")
def run_faster_whisper():
    global decoder
    from faster_whisper import WhisperModel
    model = WhisperModel({repr(model_size)}, device="cpu", compute_type={repr(Config.WHISPER_COMPUTE_TYPE)}, cpu_threads=8)
    sample_rate = 16000
    chunk_seconds = {ASR_CHUNK_SECONDS}
    overlap_seconds = {ASR_CHUNK_OVERLAP_SECONDS}
    chunk_bytes = sample_rate * 2 * chunk_seconds
    overlap_bytes = sample_rate * 2 * overlap_seconds
    ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    decoder = subprocess.Popen(
        [ffmpeg, "-nostdin", "-v", "error", "-i", {repr(str(p))},
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-f", "s16le", "pipe:1"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    segments = []
    language = "en"
    offset_sec = 0.0
    chunk_index = 0
    carry = b""
    with tempfile.TemporaryDirectory(prefix="agentic-video-asr-") as temp_dir:
        while True:
            # Re-decode the last three seconds so phrases crossing a seam keep
            # context; the total in-memory PCM window stays below 63 seconds.
            audio = bytearray(carry)
            new_bytes = 0
            while new_bytes < chunk_bytes:
                part = decoder.stdout.read(chunk_bytes - new_bytes)
                if not part:
                    break
                audio.extend(part)
                new_bytes += len(part)
            if not new_bytes:
                break
            chunk_path = os.path.join(temp_dir, "chunk.wav")
            with wave.open(chunk_path, "wb") as chunk_file:
                chunk_file.setnchannels(1)
                chunk_file.setsampwidth(2)
                chunk_file.setframerate(sample_rate)
                chunk_file.writeframes(audio)
            raw_segments, info = model.transcribe(
                chunk_path,
                beam_size=2,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            language = getattr(info, "language", language)
            chunk_start_sec = offset_sec - len(carry) / (sample_rate * 2)
            for seg in raw_segments:
                t = seg.text.strip()
                # VAD removes digital silence; this second guard avoids
                # low-confidence Whisper hallucinations.
                no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
                if t and no_speech_prob < 0.60:
                    segments.append({{"id": len(segments) + 1,
                        "start_sec": round(chunk_start_sec + seg.start, 3),
                        "end_sec": round(chunk_start_sec + seg.end, 3), "text": t,
                        "chunk_index": chunk_index,
                        "confidence": round(max(0.0, min(1.0, 1.0 - no_speech_prob)), 3)}})
            offset_sec += new_bytes / (sample_rate * 2)
            carry = bytes(audio[-overlap_bytes:]) if overlap_bytes else b""
            chunk_index += 1
        decoder.stdout.close()
        decoder_status = decoder.wait()
    if decoder_status != 0:
        raise RuntimeError("FFmpeg could not decode the audio stream")
    return segments, language
try:
    segments, language = run_whisper_cpp() if {repr(model_size)} == "turbo" else run_faster_whisper()
    res = {{"ok": True, "language": language, "segments": segments}}
    print(json.dumps(res))
    exit_code = 0
except Exception as e:
    if decoder is not None:
        try:
            decoder.terminate()
            decoder.wait(timeout=2)
        except subprocess.TimeoutExpired:
            decoder.kill()
            decoder.wait()
        except Exception:
            pass
    print(json.dumps({{"ok": False, "error": str(e)}}))
finally:
    watchdog_stop.set()
    watchdog.join(timeout=2)
    sys.stdout.flush()
    sys.stderr.flush()
    # CTranslate2 can abort in native static destructors after valid output.
    os._exit(exit_code)
    """
    timeout = timeout_sec if timeout_sec is not None else (
        ASR_BACKGROUND_TIMEOUT_SECONDS if allow_long_run else Config.SUBPROCESS_TIMEOUT_SEC
    )
    try:
        command = [sys.executable, "-c", worker_script]
        proc = _run_memory_capped_worker(
            command, timeout, ASR_WORKER_MEMORY_LIMIT_BYTES
        )
        stderr_tail = proc.stderr[-500:]
        failure_details = {
            "source_path": str(p), "model_size": model_size,
            "returncode": proc.returncode, "stderr_tail": stderr_tail,
        }
        failure_suffix = f"return code {proc.returncode}; stderr tail: {stderr_tail or '(empty)'}"
        data = None
        if proc.stdout.strip():
            try:
                data = json.loads(proc.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                pass
        if isinstance(data, dict):
            if data.get("ok") is True and isinstance(data.get("segments"), list):
                try:
                    if not all(
                        isinstance(segment, dict) and isinstance(segment.get("text"), str)
                        for segment in data["segments"]
                    ):
                        raise ValueError("invalid transcript segment")
                    normalized_segments = _deduplicate_chunk_segments(data["segments"])
                    segments = [TranscriptSegment(**segment) for segment in normalized_segments]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ASRModelError(
                        f"ASR worker returned invalid segments ({failure_suffix}): {exc}",
                        failure_details,
                    ) from exc
                res = TranscriptResult(
                    source_path=str(p),
                    language=data.get("language", "en"),
                    duration_sec=meta.duration_sec,
                    segment_count=len(segments),
                    segments=segments,
                    has_speech=len(segments) > 0,
                    notice=(
                        "No speech segments detected by VAD; absence is not proof of no speech."
                        if not segments
                        else "Note: Non-speech audio (music, sound effects, alarms) is NOT transcribed."
                    ),
                    confidence=(
                        round(sum(float(s.confidence or 0.0) for s in segments) / len(segments), 3)
                        if segments else None
                    ),
                    uncertainty=(
                        "Whisper confidence is a no_speech_prob proxy; transcript content and alignment remain model-dependent."
                        if segments
                        else "No speech segments detected by VAD; absence is not proof of no speech."
                    ),
                )
                cached_path = cache.save_json(
                    meta.fingerprint,
                    cache_name,
                    _transcript_cache_payload(res, meta.fingerprint, subtitle_fp, model_size),
                )
                if allow_long_run and cached_path is None:
                    raise ASRModelError(
                        "ASR completed but its transcript could not be cached.",
                        {"source_path": str(p), "model_size": model_size},
                    )
                return _with_repeat_flags(res)
            # Worker ran but reported a clean, actionable failure (e.g. invalid model
            # name): surface it instead of masking it as "no speech".
            raise ASRModelError(
                f"ASR worker failed ({failure_suffix}): {data.get('error', 'unknown error')}",
                failure_details,
            )
        raise ASRModelError(
            f"ASR worker failed without valid JSON ({failure_suffix})",
            failure_details,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubprocessTimeoutError(
            f"ASR transcription timed out after {timeout}s",
            {"timeout": timeout, "source_path": str(p)}
        ) from exc
    except ASRModelError:
        raise
    except Exception as e:
        raise ASRModelError(f"ASR execution failed: {e}") from e
