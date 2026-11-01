"""Shadow trial: send each fresh local transcription's audio to Cloudflare and log both transcripts.

Model: @cf/openai/whisper-large-v3-turbo. The request body (audio as base64, language, vad_filter) and the
reply (result.text and result.segments with start, end and text) follow Cloudflare's docs:
https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/ (schema-input.json and schema-output.json).

This SENDS AUDIO TO CLOUDFLARE. It is a trial switch for the maintainer only: it runs only when
AGENTIC_VIDEO_SHADOW_CF_ASR names this machine's hostname, and it stops by itself after 14 days or 200 uses.
The user never gets its output; the local transcript is returned unchanged.

Flow: asr.py calls start_after_fresh_run() once a fresh local run has saved its cache: after the background slot
is released for a background run, or, for a synchronous engine call, from a daemon thread (start_in_background) so
the caller gets its transcript before any credential or state work runs. That starts a detached, low-priority child
(python -m src.transcript.shadow_cf_asr <job file>). The child checks the switch again, cuts the audio into MP3
pieces, sends up to four at a time, and appends one JSON line to the log.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.media import cloudflare

try:
    import fcntl
except ImportError:  # Windows has no flock; the state lock is then skipped
    fcntl = None

MODEL = "@cf/openai/whisper-large-v3-turbo"
FEATURE = "Shadow ASR"  # names the trial in credential errors
SENDS = "audio"
SWITCH_ENV = "AGENTIC_VIDEO_SHADOW_CF_ASR"
PIECE_ENV = "AGENTIC_VIDEO_SHADOW_PIECE_SEC"  # tests shorten the piece length; the owner never sets it
PIECE_SEC = 600
PARALLEL_ENV = "AGENTIC_VIDEO_SHADOW_PARALLEL"  # tests set 1 for a fixed piece order; the owner never sets it
PARALLEL_PIECES = 4
PIECE_TIMEOUT_SEC = 120.0
TRIAL_DAYS = 14
TRIAL_MAX_USES = 200
NICE_LEVEL = 10
ROOT = Path(__file__).resolve().parents[2]


# ---- state directory, state file and log -------------------------------------------------------

def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME", "").strip() or str(Path.home() / ".local" / "state")
    return Path(base) / "agentic-video"


def log_path() -> Path:
    return state_dir() / "shadow-cf-asr.jsonl"


def _trial_path() -> Path:
    return state_dir() / "shadow-cf-asr-state.json"


def _ensure_state_dir() -> Path:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    return directory


@contextlib.contextmanager
def _state_lock() -> Iterator[None]:
    directory = _ensure_state_dir()
    fd = os.open(directory / "shadow-cf-asr.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _read_trial() -> Dict[str, Any]:
    """The trial state; {} before the first use. An unreadable file is marked broken, which ends the trial (fails closed)."""
    try:
        raw = _trial_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {"broken": True}
    try:
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise TypeError("not an object")
        started, count, ended = state.get("started_at"), state.get("count", 0), state.get("ended", False)
        if started is not None and (isinstance(started, bool) or not isinstance(started, (int, float))):
            raise TypeError("started_at")
        if not isinstance(count, int) or isinstance(count, bool) or not isinstance(ended, bool):
            raise TypeError("count or ended")
        return state
    except (ValueError, TypeError):
        return {"broken": True}


def _write_trial(state: Dict[str, Any]) -> None:
    directory = _ensure_state_dir()
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".shadow-cf-asr-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, _trial_path())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _append_log(record: Dict[str, Any]) -> None:
    _ensure_state_dir()
    fd = os.open(log_path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.chmod(log_path(), 0o600)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trial_refusal(state: Dict[str, Any], now: float, admit: bool) -> Optional[str]:
    """Why the trial may not run, or None. admit=True is the job's admission and also applies the use limit;
    admit=False is the check before each piece and retry, which must not count again."""
    started = state.get("started_at")
    if state.get("broken"):
        return "state file unreadable"
    if started is not None and now > started + TRIAL_DAYS * 86400:
        return f"more than {TRIAL_DAYS} days after the first use"
    if admit and state.get("count", 0) >= TRIAL_MAX_USES:
        return f"{TRIAL_MAX_USES} uses reached"
    return None


def _trial_open(admit: bool) -> bool:
    """Whether the trial may send now. The first refusal writes one trial_ended line and the trial stays off."""
    with _state_lock():
        state = _read_trial()
        if state.get("ended"):
            return False
        started, count = state.get("started_at"), state.get("count", 0)
        reason = _trial_refusal(state, time.time(), admit)
        if reason is not None:
            _write_trial({"started_at": started, "count": count, "ended": True, "ended_reason": reason})
            _append_log({"ts": _now_iso(), "status": "trial_ended", "reason": reason,
                         "started_at": started, "count": count})
            return False
        if admit:
            _write_trial({"started_at": started if started is not None else time.time(), "count": count + 1,
                          "ended": False})
        return True


def trial_open_now() -> bool:
    """Read-only: whether a new job would be admitted now. Nothing is counted or written."""
    state = _read_trial()
    return not state.get("ended") and _trial_refusal(state, time.time(), admit=True) is None


# ---- parent side: called from asr.py ----------------------------------------------------------

def _switch_names_this_host() -> bool:
    names = {part.strip() for part in os.environ.get(SWITCH_ENV, "").split(",")}
    names.discard("")
    return socket.gethostname() in names


def start_after_fresh_run(source_path: str, model_size: str, result: Any, fingerprint: str,
                          local_wall_sec: float) -> bool:
    """Start the shadow job for one fresh local run. Returns True when a child was started. Never raises."""
    job_path = None
    try:
        if not _switch_names_this_host():
            return False
        cloudflare.credentials(FEATURE, SENDS)  # raises when the credentials are missing; nothing is sent
        if not _trial_open(admit=False):  # the child admits the job itself, right before it sends
            return False
        job = {
            "source_path": source_path,
            "fingerprint": fingerprint,
            "duration_sec": result.duration_sec,
            "model_size": model_size,
            "engine": "whisper.cpp" if model_size == "turbo" else "faster-whisper",
            "language": result.language,
            "local_has_speech": result.has_speech,
            "local_wall_sec": round(local_wall_sec, 3),
            "local_segments": [segment.to_dict() for segment in result.segments],
        }
        handle, job_path = tempfile.mkstemp(dir=_ensure_state_dir(), prefix="job-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(job, out)
        subprocess.Popen(
            [sys.executable, "-m", "src.transcript.shadow_cf_asr", job_path], cwd=str(ROOT),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        if job_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(job_path)
        return False


def start_in_background(**fresh_run: Any) -> None:
    """Run start_after_fresh_run on a daemon thread, so a synchronous caller returns its transcript without waiting for
    credentials (a token command can take up to its 30 s timeout) or the trial state lock. The switch is read here,
    so a switch that does not name this host starts no thread. Never raises. A process that exits before the thread
    has started its child skips that one shadow run; the user's transcript is not affected."""
    try:
        if not _switch_names_this_host():
            return
        threading.Thread(target=start_after_fresh_run, kwargs=fresh_run, name="agentic-video-shadow-prepare",
                         daemon=True).start()
    except Exception:
        return


# ---- child side: one job ----------------------------------------------------------------------

def _piece_sec() -> int:
    raw = os.environ.get(PIECE_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else PIECE_SEC


def _parallel_pieces() -> int:
    raw = os.environ.get(PARALLEL_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else PARALLEL_PIECES


def _version() -> str:
    try:
        return version("agentic-video")
    except PackageNotFoundError:
        return "dev"


def _cut_pieces(source: str, directory: str, piece_sec: int) -> Optional[List[str]]:
    """16 kHz mono MP3 pieces of piece_sec seconds, no overlap, in order. None when ffmpeg fails or finds no audio."""
    ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    done = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", source, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "32k", "-f", "segment", "-segment_time", str(piece_sec),
         "-reset_timestamps", "1", os.path.join(directory, "piece_%04d.mp3")],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=2 * 3600,
    )
    if done.returncode != 0:
        return None
    return sorted(glob.glob(os.path.join(directory, "piece_*.mp3"))) or None


def _send_piece(index: int, path: str, *, account: str, token: str, language: str, piece_sec: int,
                allowed: Callable[[], bool]) -> Dict[str, Any]:
    """One piece through Workers AI. Returns its segments with offsets added, or its error (token removed).
    `allowed` is asked before each attempt, so an expiry during the job stops the rest of its pieces."""
    try:
        with open(path, "rb") as handle:
            body: Dict[str, Any] = {"audio": base64.b64encode(handle.read()).decode("ascii"), "vad_filter": False}
        if language:
            body["language"] = language
        raw = cloudflare.run_model(MODEL, account, token, json.dumps(body).encode("utf-8"), PIECE_TIMEOUT_SEC,
                                   allowed=allowed)
        reply = json.loads(raw)
        if not isinstance(reply, dict) or reply.get("success") is not True or not isinstance(reply.get("result"), dict):
            raise ValueError("the reply did not report success")
        result = reply["result"]
        offset = float(index * piece_sec)
        segments = [
            {"piece_index": index, "start_sec": round(float(seg["start"]) + offset, 3),
             "end_sec": round(float(seg["end"]) + offset, 3), "text": str(seg["text"]).strip()}
            for seg in (result.get("segments") or [])
        ]
        return {"piece_index": index, "text": str(result.get("text", "")).strip(), "segments": segments}
    except Exception as exc:  # any failure is recorded for this piece; the others still run
        return {"piece_index": index, "error": cloudflare.hide_token(f"{type(exc).__name__}: {exc}", token)}


def _language_hint(language: str) -> str:
    return language if language.isascii() and language.isalpha() and 2 <= len(language) <= 3 else ""


def run_job(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send one job's audio and append its log line. Returns the record written, or None when nothing was sent.
    The switch is checked here as well as in the parent, so a child started by hand cannot send either."""
    if not _switch_names_this_host():
        return None
    try:
        account, token = cloudflare.credentials(FEATURE, SENDS)
    except cloudflare.CloudCredentialsError:
        return None
    piece_sec = _piece_sec()
    local_text = " ".join(seg["text"].strip() for seg in job["local_segments"] if seg["text"].strip())
    base = {
        "agentic_video": _version(), "source_fingerprint": job["fingerprint"], "source_path": job["source_path"],
        "duration_sec": job["duration_sec"], "language": job["language"], "local_model": job["model_size"],
        "local_engine": job["engine"], "local_has_speech": job["local_has_speech"], "local_text": local_text,
        "local_segments": job["local_segments"], "local_wall_sec": job["local_wall_sec"], "cf_model": MODEL,
        "piece_sec": piece_sec,
    }
    with tempfile.TemporaryDirectory(prefix="agentic-video-shadow-") as temp_dir:
        pieces = _cut_pieces(job["source_path"], temp_dir, piece_sec)
        if pieces is None:
            record = {"ts": _now_iso(), "status": "decode_failed", "cf_text": "", "cf_segments": [],
                      "cf_wall_sec": 0.0, "pieces": 0, "piece_errors": [], **base}
            _append_log(record)
            return record
        if not _trial_open(admit=True):  # one admission per job, right before its first piece is sent
            return None
        began = time.monotonic()
        send = functools.partial(_send_piece, account=account, token=token,
                                 language=_language_hint(job["language"]), piece_sec=piece_sec,
                                 allowed=lambda: _trial_open(admit=False))
        with ThreadPoolExecutor(max_workers=_parallel_pieces()) as pool:
            replies = list(pool.map(send, range(len(pieces)), pieces))
        cf_wall = time.monotonic() - began
    good = [reply for reply in replies if "error" not in reply]
    errors = [{"piece_index": reply["piece_index"], "error": reply["error"]} for reply in replies if "error" in reply]
    status = "ok" if not errors else ("partial" if good else "failed")
    record = {
        "ts": _now_iso(), "status": status, **base,
        "cf_text": " ".join(reply["text"] for reply in good if reply["text"]),
        "cf_segments": [seg for reply in good for seg in reply["segments"]],
        "cf_wall_sec": round(cf_wall, 3), "pieces": len(replies), "piece_errors": errors,
    }
    _append_log(record)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    job_path = Path(args[0])
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    finally:
        with contextlib.suppress(OSError):
            os.unlink(job_path)  # the job file holds the local transcript; it is kept only while this child runs
    with contextlib.suppress(AttributeError, OSError):
        os.nice(NICE_LEVEL)
    try:
        run_job(job)
    except Exception as exc:  # an unexpected failure is logged; no transcript or token is written
        with contextlib.suppress(Exception):
            _append_log({"ts": _now_iso(), "status": "failed", "source_path": job.get("source_path"),
                         "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
