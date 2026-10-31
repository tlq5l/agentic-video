"""Decoded source-frame timing helpers.

FFmpeg's output filters can retime frames (for example, ``fps=`` emits a
regular output clock).  This module asks ffprobe for the source stream's
best-effort frame timestamps so callers can distinguish a requested seek from
the decoded frame that was actually selected.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple

from src.core.config import Config
from src.core.errors import MediaCorruptError


_SHOWINFO_PTS_RE = re.compile(r"\bpts_time:(-?(?:\d+(?:\.\d*)?|\.\d+))\b")
_SHOWINFO_DURATION_RE = re.compile(r"\bduration_time:(\d+(?:\.\d*)?|\.\d+)\b")
_SHOWINFO_TICKS_RE = re.compile(r"\bpts:\s*(-?\d+)\b")
_INTEGER_RE = re.compile(r"-?\d+")


def parse_selected_frame_timestamp(stderr: str, last: bool = False) -> Optional[float]:
    """Parse the source PTS emitted by FFmpeg's ``showinfo`` filter.

    This is intentionally separate from ``probe_decoded_frame_timestamps``:
    probing nearby source frames cannot prove which frame a seek operation
    actually wrote to disk, especially across VFR PTS gaps.  Callers that
    request one output frame should add ``showinfo`` to that same FFmpeg
    command and use this value as the selected-frame receipt.

    With ``last=True`` the receipt is the final frame shown (for commands that
    overwrite one output file with every frame and keep the last).
    """
    if not isinstance(stderr, str):
        return None
    matches = list(_SHOWINFO_PTS_RE.finditer(stderr))
    for match in (reversed(matches) if last else matches):
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            return round(value, 6)
    return None


def parse_last_frame_duration(stderr: str) -> Optional[float]:
    """Display duration of the final frame ``showinfo`` printed (same line as its
    ``pts_time``), or None when FFmpeg gave no positive duration for it."""
    if not isinstance(stderr, str):
        return None
    frame_lines = [line for line in stderr.splitlines() if _SHOWINFO_PTS_RE.search(line)]
    if not frame_lines:
        return None
    match = _SHOWINFO_DURATION_RE.search(frame_lines[-1])
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) and value > 0.0 else None


def probe_decoded_frame_timestamps(
    video_path: str | Path,
    start_sec: float,
    end_sec: Optional[float] = None,
    timeout_sec: float = 30.0,
) -> list[float]:
    """Return sorted best-effort source-frame PTS values in a time interval.

    Timing is supplementary provenance.  A missing ffprobe binary, an
    unsupported container, or a malformed probe result returns an empty list;
    the frame extraction itself remains usable and reports a null decoded
    timestamp instead of inventing precision.
    """
    try:
        start = float(start_sec)
        end = float(end_sec) if end_sec is not None else start + 1.0
    except (TypeError, ValueError):
        return []
    if not math.isfinite(start) or not math.isfinite(end):
        return []
    start = max(0.0, start)
    end = max(start + 0.001, end)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"{start:.6f}%{end:.6f}",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(Path(video_path).resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    timestamps: list[float] = []
    for frame in payload.get("frames", []):
        raw = frame.get("best_effort_timestamp_time")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            timestamps.append(value)
    return sorted(set(timestamps))


def nearest_decoded_timestamp(
    requested_sec: float,
    decoded_timestamps: Iterable[float],
) -> Optional[float]:
    """Select the decoded source PTS nearest to a requested timestamp."""
    try:
        requested = float(requested_sec)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(requested):
        return None
    values: list[float] = []
    for raw_value in decoded_timestamps:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    values.sort()
    if not values:
        return None
    index = bisect.bisect_left(values, requested)
    candidates = []
    if index < len(values):
        candidates.append(values[index])
    if index:
        candidates.append(values[index - 1])
    return min(candidates, key=lambda value: abs(value - requested))


def timestamp_provenance(
    requested_sec: float,
    decoded_timestamps: Iterable[float],
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(decoded_timestamp, decoded-minus-requested)``."""
    decoded = nearest_decoded_timestamp(requested_sec, decoded_timestamps)
    if decoded is None:
        return None, None
    return round(decoded, 6), round(decoded - float(requested_sec), 6)


def parse_selected_frame_pts(stderr: str) -> Optional[int]:
    """Integer pts, in the stream's time base, of the last frame ``showinfo`` printed: the frame
    an output file that is overwritten for each frame keeps."""
    if not isinstance(stderr, str):
        return None
    found = _SHOWINFO_TICKS_RE.findall(stderr)
    return int(found[-1]) if found else None


class StreamClock(NamedTuple):
    """Exact clock of a file's first video stream, in seconds: the container start (the time zero
    the other tools count from), the stream's time base (seconds per tick) and the container's
    duration (0 when it gives none)."""

    origin: Fraction
    time_base: Fraction
    duration: Fraction


def _exact_decimal(raw: object) -> Fraction:
    if raw is None or raw == "N/A":
        return Fraction(0)
    return Fraction(str(raw))


def read_stream_clock(video_path: str | Path) -> StreamClock:
    """The container start, the video stream's time base and the container duration, as ffprobe prints them."""
    path = Path(video_path).resolve()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=start_time,duration:stream=time_base", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False,
    )
    try:
        if proc.returncode != 0:
            raise ValueError(proc.stderr.strip())
        payload = json.loads(proc.stdout)
        num, den = payload["streams"][0]["time_base"].split("/")
        time_base = Fraction(int(num), int(den))
        if time_base <= 0:
            raise ValueError(f"time base {time_base}")
        format_info = payload["format"]
        return StreamClock(
            _exact_decimal(format_info.get("start_time")),
            time_base,
            _exact_decimal(format_info.get("duration")),
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, ZeroDivisionError) as exc:
        raise MediaCorruptError(
            f"ffprobe could not read the clock of {path}: {exc}",
            {"video": str(path), "stderr": proc.stderr},
        ) from exc


def _tick_value(raw: object) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and _INTEGER_RE.fullmatch(raw):
        return int(raw)
    return None


def probe_packet_ticks(video_path: str | Path, start_sec: float, end_sec: Optional[float]) -> List[Tuple[int, int]]:
    """(pts, duration) in ticks of the first video stream's packets, in decode order, from start_sec
    up to end_sec (to the end of the file when None). Times are on the container clock, so they may
    be negative. ffprobe starts at the keyframe before start_sec, so the list holds every packet from
    start_sec on, plus some earlier ones. A packet without a pts is left out; one without a duration
    counts as 0."""
    interval = f"{start_sec:.6f}%" if end_sec is None else f"{start_sec:.6f}%{end_sec:.6f}"
    path = Path(video_path).resolve()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", interval,
         "-show_entries", "packet=pts,duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False,
    )
    if proc.returncode != 0:
        raise MediaCorruptError(
            f"ffprobe could not list the frames of {path}: {proc.stderr.strip()}",
            {"video": str(path), "stderr": proc.stderr},
        )
    try:
        payload = json.loads(proc.stdout)
    except ValueError as exc:
        raise MediaCorruptError(
            f"ffprobe printed no packet list for {path}",
            {"video": str(path), "stderr": proc.stderr},
        ) from exc
    packets: List[Tuple[int, int]] = []
    for packet in payload.get("packets", []):
        pts = _tick_value(packet.get("pts"))
        if pts is not None:
            packets.append((pts, _tick_value(packet.get("duration")) or 0))
    return packets


def _simplest_between(low: Fraction, high: Fraction) -> Fraction:
    """The fraction with the smallest denominator in [low, high], for 0 < low <= high (exact)."""
    whole = math.floor(low)
    if whole == low:
        return Fraction(whole)
    if whole + 1 <= high:
        return Fraction(whole + 1)
    return whole + 1 / _simplest_between(1 / (high - whole), 1 / (low - whole))


def exact_request_seconds(value: float) -> Fraction:
    """The exact time a non-negative JSON number asks for: the simplest fraction that rounds to
    that float. A float cannot hold 1/24 s or 1/30 s, so a caller who sends the float for n/fps
    means the fraction n/fps; it is the simplest fraction inside the float's rounding interval.
    Frame selection then compares integers only."""
    if value == 0:
        return Fraction(0)
    exact = Fraction(value)
    below = Fraction(math.nextafter(value, -math.inf))
    above_value = math.nextafter(value, math.inf)
    above = Fraction(above_value) if math.isfinite(above_value) else 2 * exact - below
    low = (exact + below) / 2
    high = (exact + above) / 2
    return _simplest_between(low, high)
