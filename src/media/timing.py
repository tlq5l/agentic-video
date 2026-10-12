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
from pathlib import Path
from typing import Iterable, Optional


_SHOWINFO_PTS_RE = re.compile(r"\bpts_time:(-?(?:\d+(?:\.\d*)?|\.\d+))\b")


def parse_selected_frame_timestamp(stderr: str) -> Optional[float]:
    """Parse the source PTS emitted by FFmpeg's ``showinfo`` filter.

    This is intentionally separate from ``probe_decoded_frame_timestamps``:
    probing nearby source frames cannot prove which frame a seek operation
    actually wrote to disk, especially across VFR PTS gaps.  Callers that
    request one output frame should add ``showinfo`` to that same FFmpeg
    command and use this value as the selected-frame receipt.
    """
    if not isinstance(stderr, str):
        return None
    for match in _SHOWINFO_PTS_RE.finditer(stderr):
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            return round(value, 6)
    return None


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
