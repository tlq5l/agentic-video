"""Turn a video URL (X post, YouTube, other yt-dlp sites, direct link) into a cached local file.

Downloads run in a background thread, one per URL, so a tool call never waits
longer than ``WAIT_SEC``; the caller repeats the call until the file is ready.
"""
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.core.config import Config

# Executor drops a call at 60 s and frame extraction may use up to 40 s after the
# download, so the download wait gets what is left (see test_url_source).
WAIT_SEC = 15.0
RETRY_AFTER_SECONDS = 15
DOWNLOAD_TIMEOUT_SEC = 20 * 60
# The Executor server's PATH only has /opt/homebrew/bin; yt-dlp often lives in ~/.local/bin.
_FALLBACK_BIN_DIRS = (Path.home() / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
_TRACKING_PARAMS = {"si", "feature", "fbclid", "gclid", "igshid", "ref_src"}


class DownloadError(RuntimeError):
    """The URL could not be turned into a local video file; the message names the URL and cause."""


def is_url(source: str) -> bool:
    return source.strip().lower().startswith(("http://", "https://"))


def normalize_url(url: str) -> str:
    """Map the different spellings of one video to one string, so they share a download."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    path = parts.path
    query = parse_qsl(parts.query, keep_blank_values=True)
    for prefix in ("www.", "m.", "mobile."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    if host in {"twitter.com", "x.com", "fxtwitter.com", "vxtwitter.com", "fixupx.com"}:
        status = re.match(r"^/[^/]+/status/(\d+)", path)
        if status:  # the post id alone names the video; the handle and /video/1 suffix vary
            return f"https://x.com/i/status/{status.group(1)}"
        host = "x.com"
    video_id = None
    if host == "youtu.be":
        video_id = path.strip("/").split("/")[0]
    elif host in {"youtube.com", "music.youtube.com"}:
        shorts = re.match(r"^/(?:shorts|live|embed)/([^/?]+)", path)
        video_id = shorts.group(1) if shorts else dict(query).get("v")
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    kept = sorted((k, v) for k, v in query if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_"))
    return urlunsplit((parts.scheme.lower(), host, path or "/", urlencode(kept), ""))


def download_dir() -> Path:
    return Path(Config.BASE_CACHE_DIR) / "downloads"


def cached_path(url: str) -> Path:
    return download_dir() / f"{hashlib.sha256(normalize_url(url).encode('utf-8')).hexdigest()}.mp4"


def _find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for folder in _FALLBACK_BIN_DIRS:
        candidate = folder / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _last_error_line(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR:")]
    return (errors or lines or ["yt-dlp gave no error output"])[-1]


def _download(url: str, target: Path) -> None:
    """Download ``url`` to ``target`` with yt-dlp; a partial file never appears at ``target``."""
    yt_dlp = _find_binary("yt-dlp")
    if yt_dlp is None:
        raise DownloadError(
            f"Could not download {url}: yt-dlp is not installed (brew install yt-dlp). "
            "Download the video yourself and pass its local path as `source`.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_stem = f"{target.stem}.tmp-{uuid.uuid4().hex}"
    cmd = [yt_dlp, "--no-playlist", "--no-progress", "--no-simulate",
           "-S", "res:1080,vcodec:h264,ext:mp4:m4a", "--merge-output-format", "mp4",
           "-o", str(target.parent / f"{temp_stem}.%(ext)s"), "--print", "after_move:filepath"]
    if shutil.which("ffmpeg") is None:
        ffmpeg = _find_binary("ffmpeg")
        if ffmpeg:
            cmd += ["--ffmpeg-location", str(Path(ffmpeg).parent)]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SEC)
        printed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or not printed or not Path(printed[-1]).is_file():
            raise DownloadError(f"Could not download {url}: {_last_error_line(proc.stderr)}")
        os.replace(printed[-1], target)
    except subprocess.TimeoutExpired:
        raise DownloadError(f"Could not download {url}: yt-dlp ran over {DOWNLOAD_TIMEOUT_SEC // 60} minutes") from None
    finally:
        for leftover in target.parent.glob(f"{temp_stem}*"):
            leftover.unlink(missing_ok=True)


class _Job:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[str] = None


_JOBS: Dict[Path, _Job] = {}
_JOBS_LOCK = threading.Lock()


def _run_job(url: str, target: Path, job: _Job) -> None:
    try:
        _download(url, target)
    except DownloadError as exc:
        job.error = str(exc)
    except Exception as exc:  # report, never kill the server thread silently
        job.error = f"Could not download {url}: {type(exc).__name__}: {exc}"
    finally:
        job.done.set()


def _start_job(url: str, target: Path) -> _Job:
    with _JOBS_LOCK:
        job = _JOBS.get(target)
        if job is None:
            job = _JOBS[target] = _Job()
            threading.Thread(target=_run_job, args=(url, target, job),
                             name=f"agentic-video-download-{target.stem[:12]}", daemon=True).start()
    return job


def resolve_urls(urls: Iterable[str]) -> Dict[str, Optional[Path]]:
    """Map each URL to its local file, or to None while its download is still running.

    Starts (or joins) one background download per URL before waiting on any, then
    waits for all of them within one shared ``WAIT_SEC``, so a call with two URLs
    never waits twice. Raises DownloadError when a download failed; the next call
    starts a fresh attempt.
    """
    targets = {url: cached_path(url) for url in urls}
    jobs = {url: _start_job(url, target) for url, target in targets.items() if not target.is_file()}
    deadline = time.monotonic() + WAIT_SEC
    resolved: Dict[str, Optional[Path]] = {}
    for url, target in targets.items():
        job = jobs.get(url)
        if job is None:
            resolved[url] = target
            continue
        if not job.done.wait(max(0.0, deadline - time.monotonic())):
            resolved[url] = None
            continue
        with _JOBS_LOCK:
            if _JOBS.get(target) is job:
                del _JOBS[target]
        if job.error:
            raise DownloadError(job.error)
        resolved[url] = target
    return resolved


def pending_result(url: str) -> Dict[str, object]:
    """The ok result returned while a download runs (same status/retry fields transcribe uses)."""
    return {
        "status": "in_progress",
        "retry_after_seconds": RETRY_AFTER_SECONDS,
        "source_url": url,
        "note": f"Downloading {url}; call again with the same source in about {RETRY_AFTER_SECONDS} seconds.",
    }
