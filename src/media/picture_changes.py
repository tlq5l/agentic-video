"""Picture changes that the scene score misses, for the Checked section.

Hard cut inside one footage: a cut between two shots of the same game clip changes
the picture about as much as motion does, so the FFmpeg scene score cannot tell
them apart. What sets a cut apart is a single spike: the frame-to-frame change at
the cut stands far above the changes in the frames around it, and motion does not.

Bright flash: one to three frames that are much brighter than the frames around
them (a white flash on a cut, or a bright insert). Black frames are another check.

Both read small gray frames (54x96), the same grid for both. Times come from
FFmpeg's showinfo (the real presentation time of each decoded frame), so a
variable-frame-rate clip keeps its real timeline.
"""
import re
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np

_GRID = (54, 96)            # width, height of the gray frames compared
_CUT_MIN_CHANGE = 8.0       # mean abs gray change (0-255) a cut must reach
_CUT_ISOLATION_MIN = 6.0    # cut change / median change of the surrounding frames
_CUT_WINDOW = 8             # frames either side for the median (the two next to the spike are skipped)
_CUT_DEDUPE_SEC = 0.3       # a flash pair or double peak at one cut counts once
_CUT_KNOWN_SEC = 0.1        # a spike this close to a scene-score cut is that cut
_FLASH_LIFT = 60.0          # mean luma (0-255) a flash frame must rise above BOTH sides around it
_FLASH_MAX_FRAMES = 3       # longer bright runs are scenes, not flashes
_FLASH_NEAR = (3, 9)        # frames 3 to 9 away on either side set the level on that side


def gray_frames(video_path: str, timeout_sec: float) -> Tuple[np.ndarray, List[float]]:
    """All frames as a (count, height, width) uint8 array, and the time (s) of each frame.

    The times are the decoded frames' own presentation times (showinfo pts_time), with
    passthrough so no frame is duplicated or dropped; the same way the scene score reads them.
    """
    width, height = _GRID
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-nostats", "-v", "info", "-i", video_path, "-map", "0:v:0", "-an",
         "-vf", "showinfo,scale={}:{},format=gray".format(width, height), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec, check=True,
    )
    times = [float(t) for t in re.findall(r"pts_time:(\S+)", proc.stderr.decode(errors="replace"))]
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width)
    if len(times) != len(frames):
        raise RuntimeError("showinfo listed {} frames but {} were decoded".format(len(times), len(frames)))
    return frames, times


def same_footage_cut_times(video_path: str, fps: float, known_cut_times: List[float] = (),
                           timeout_sec: float = 900.0) -> List[float]:
    """Times (s) of hard cuts that the scene score missed; known scene cuts are not repeated.

    A spike at change index i (between frames i and i+1) counts when it is a local peak,
    reaches _CUT_MIN_CHANGE, and is at least _CUT_ISOLATION_MIN times the median change of
    the frames _CUT_WINDOW either side (skipping the two next to it). Its time is the time
    of frame i+1. Peaks within _CUT_DEDUPE_SEC of each other keep only the strongest one.
    `fps` only guards against a clip with no frame rate.
    """
    if fps <= 0:
        return []
    frames, times = gray_frames(video_path, timeout_sec)
    frames = frames.astype(np.float32)
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
        time = times[i + 1]
        if any(abs(time - t) <= _CUT_KNOWN_SEC for t in known):
            continue
        peaks.append((time, ratio))
    kept: List[tuple] = []
    for time, ratio in sorted(peaks, key=lambda item: -item[1]):
        if all(abs(time - other) > _CUT_DEDUPE_SEC for other, _ in kept):
            kept.append((time, ratio))
    return [round(time, 3) for time, _ in sorted(kept)]


def bright_flash_runs(video_path: str, fps: float, timeout_sec: float = 900.0) -> List[Dict[str, Any]]:
    """Runs of 1 to _FLASH_MAX_FRAMES frames at least _FLASH_LIFT brighter than BOTH sides.

    Each frame is compared with the median level 3 to 9 frames before it and the median
    level 3 to 9 frames after it. The lift is the smaller of the two differences, so a
    frame at the edge of a bright scene (bright on one side only) is not a flash, and a
    scene that lasts longer than _FLASH_MAX_FRAMES is never listed.
    """
    if fps <= 0:
        return []
    frames, times = gray_frames(video_path, timeout_sec)
    count = len(frames)
    if count == 0:
        return []
    level = frames.mean(axis=(1, 2), dtype=np.float32)
    lo, hi = _FLASH_NEAR
    side_max = np.full(count, np.nan, dtype=np.float32)
    for k in range(hi, count - hi):
        before = np.median(level[k - hi:k - lo + 1])
        after = np.median(level[k + lo:k + hi + 1])
        side_max[k] = max(before, after)
    lift = level - side_max
    lifted = lift >= _FLASH_LIFT
    runs: List[Dict[str, Any]] = []
    i = 0
    while i < count:
        if not lifted[i]:
            i += 1
            continue
        j = i
        while j + 1 < count and lifted[j + 1]:
            j += 1
        # The lifted frames can be only the middle of a longer bright scene (its sides reach
        # back into the scene), so measure the whole bright stretch at half the lift.
        half = float(np.nanmin(side_max[i:j + 1])) + _FLASH_LIFT / 2.0
        first, last = i, j
        while first > 0 and level[first - 1] >= half:
            first -= 1
        while last + 1 < count and level[last + 1] >= half:
            last += 1
        if last - first + 1 <= _FLASH_MAX_FRAMES:
            end = times[last + 1] if last + 1 < count else times[last] + 1.0 / float(fps)
            runs.append({"start_sec": round(times[first], 3), "end_sec": round(end, 3),
                         "lift": round(float(lift[i:j + 1].max()), 1)})
        i = j + 1
    return runs
