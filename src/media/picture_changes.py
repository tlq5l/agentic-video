"""Picture changes that the scene score misses, for the Checked section.

Hard cut inside one footage: a cut between two shots of the same game clip changes
the picture about as much as motion does, so the FFmpeg scene score cannot tell
them apart. What sets a cut apart is a single spike: the frame-to-frame change at
the cut stands far above the changes in the frames around it, and motion does not.

Bright flash: one to three frames that are much brighter than the frames around
them (a white flash on a cut, or a bright insert). Black frames are another check.

Both read small gray frames (54x96), the same grid for both.
"""
import subprocess
from typing import Any, Dict, List

import numpy as np

_GRID = (54, 96)            # width, height of the gray frames compared
_CUT_MIN_CHANGE = 8.0       # mean abs gray change (0-255) a cut must reach
_CUT_ISOLATION_MIN = 6.0    # cut change / median change of the surrounding frames
_CUT_WINDOW = 8             # frames either side for the median (the two next to the spike are skipped)
_CUT_DEDUPE_SEC = 0.3       # a flash pair or double peak at one cut counts once
_CUT_KNOWN_SEC = 0.1        # a spike this close to a scene-score cut is that cut
_FLASH_LIFT = 60.0          # mean luma (0-255) a flash frame must rise above its surroundings
_FLASH_MAX_FRAMES = 3       # longer bright runs are scenes, not flashes
_FLASH_NEAR = (3, 9)        # frames 3 to 9 away on either side set the level around a frame


def gray_frames(video_path: str, timeout_sec: float) -> np.ndarray:
    """All frames as a (count, height, width) uint8 array, scaled to the comparison grid."""
    width, height = _GRID
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", video_path, "-map", "0:v:0", "-an",
         "-vf", "scale={}:{},format=gray".format(width, height), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout_sec, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)


def same_footage_cut_times(video_path: str, fps: float, known_cut_times: List[float] = (),
                           timeout_sec: float = 900.0) -> List[float]:
    """Times (s) of hard cuts that the scene score missed; known scene cuts are not repeated.

    A spike at change index i (between frames i and i+1) counts when it is a local peak,
    reaches _CUT_MIN_CHANGE, and is at least _CUT_ISOLATION_MIN times the median change of
    the frames _CUT_WINDOW either side (skipping the two next to it). Peaks within
    _CUT_DEDUPE_SEC of each other keep only the strongest one.
    """
    if fps <= 0:
        return []
    frames = gray_frames(video_path, timeout_sec).astype(np.float32)
    if len(frames) < 2:
        return []
    changes = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    known = [float(t) for t in known_cut_times]
    peaks = []
    for i in range(_CUT_WINDOW, len(changes) - _CUT_WINDOW - 1):
        value = float(changes[i])
        if value < _CUT_MIN_CHANGE or value < changes[i - 1] or value < changes[i + 1]:
            continue
        around = np.concatenate([changes[i - _CUT_WINDOW:i - 1], changes[i + 2:i + 1 + _CUT_WINDOW]])
        ratio = value / max(float(np.median(around)), 0.5)
        if ratio < _CUT_ISOLATION_MIN:
            continue
        time = (i + 1) / float(fps)
        if any(abs(time - t) <= _CUT_KNOWN_SEC for t in known):
            continue
        peaks.append((time, ratio))
    kept: List[tuple] = []
    for time, ratio in sorted(peaks, key=lambda item: -item[1]):
        if all(abs(time - other) > _CUT_DEDUPE_SEC for other, _ in kept):
            kept.append((time, ratio))
    return [round(time, 3) for time, _ in sorted(kept)]


def bright_flash_runs(video_path: str, fps: float, timeout_sec: float = 900.0) -> List[Dict[str, Any]]:
    """Runs of 1 to _FLASH_MAX_FRAMES frames at least _FLASH_LIFT brighter than the frames around them."""
    if fps <= 0:
        return []
    frames = gray_frames(video_path, timeout_sec)
    count = len(frames)
    if count == 0:
        return []
    level = frames.mean(axis=(1, 2), dtype=np.float32)
    lo, hi = _FLASH_NEAR
    base = np.full(count, np.nan, dtype=np.float32)
    for k in range(hi, count - hi):
        base[k] = np.median(np.concatenate([level[k - hi:k - lo + 1], level[k + lo:k + hi + 1]]))
    lifted = (level - base) >= _FLASH_LIFT
    runs: List[Dict[str, Any]] = []
    i = 0
    while i < count:
        if not lifted[i]:
            i += 1
            continue
        j = i
        while j + 1 < count and lifted[j + 1]:
            j += 1
        if j - i + 1 <= _FLASH_MAX_FRAMES:
            runs.append({"start_sec": round(i / float(fps), 3), "end_sec": round((j + 1) / float(fps), 3),
                         "lift": round(float(level[i:j + 1].max() - base[i:j + 1].min()), 1)})
        i = j + 1
    return runs
