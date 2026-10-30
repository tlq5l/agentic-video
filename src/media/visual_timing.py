"""When on-screen text actually appears, compared with its cue.

A cue labelled ``"<sound> | <on-screen text>"`` says the text appears with
the sound. The picture is reduced to a 160x90 grey grid; each frame's
"change share" is the fraction of cells that changed by more than a few grey
levels since the previous frame. Background motion keeps that share low and
steady, while a caption, title or cut appearing makes it stand out.
"""

from __future__ import annotations

import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.errors import MediaCorruptError, MediaNotFoundError

_GRID = (160, 90)
_CELL_CHANGE = 6.0
# A frame stands out when its change share is at least this much above the
# local median, and at least double it.
_STAND_OUT = 0.01
# A displaced appearance must cover about a caption's area of the frame.
_MIN_DISPLACED_SHARE = 0.02
_WINDOW_BEATS = 1.5


def _frame_rate(path: Path) -> float:
    """The stream's r_frame_rate, the clock ``_grid_frames`` resamples the picture to."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False,
        timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    numerator, _, denominator = probe.stdout.strip().partition("/")
    try:
        rate = float(numerator) / float(denominator or 1.0)
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    if probe.returncode != 0 or not rate > 0.0:
        raise MediaCorruptError("Could not read the video frame rate", {"source": str(path)})
    return rate


# The grid frames of the last video decoded, kept (as uint8) when small enough so
# the checker's scans of the same file share one decode.
_SHARE_MAX_BYTES = 256 * 1024 * 1024
_shared: Optional[tuple] = None  # (file key, frames), replaced as one object


def _share_key(path: Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _grid_frames(path: Path, rate: float) -> Iterator[np.ndarray]:
    """Each frame as a flat 160x90 grey grid, streamed from FFmpeg; a repeat scan of one file reuses the first decode.

    The fps filter puts frame i at i / rate by timestamp, copying frames over a variable-rate file's gaps;
    FFmpeg's default constant-rate output goes by frame durations and drifted 0.2 s on screen captures.
    """
    global _shared
    width, height = _GRID
    size = width * height
    key = _share_key(path)
    shared = _shared
    if shared is not None and shared[0] == key:
        for frame in shared[1]:
            yield frame.astype(np.float32)
        return
    kept: Optional[bytearray] = bytearray()
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
         "-vf", "fps={},scale={}:{}:flags=area,format=gray".format(Fraction(rate).limit_denominator(1001000), width, height), "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(size)
            if len(chunk) < size:
                break
            if kept is not None:
                kept += chunk
                if len(kept) > _SHARE_MAX_BYTES:
                    kept = None
            yield np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
        exit_code = process.wait(timeout=max(Config.SUBPROCESS_TIMEOUT_SEC * 4, 180.0))
        if exit_code != 0:
            raise MediaCorruptError("FFmpeg could not decode the video frames", {"source": str(path), "exit_code": exit_code})
        if kept is not None:
            _shared = (key, np.frombuffer(kept, dtype=np.uint8).reshape(-1, size))
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.wait(timeout=max(Config.SUBPROCESS_TIMEOUT_SEC * 4, 180.0))


def _video_path(source: str) -> Path:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise MediaNotFoundError("Video source file not found", {"path": str(path)})
    return path


def change_shares(source: str) -> Dict[str, Any]:
    """Per-frame change share, streamed from FFmpeg at grid size."""
    path = _video_path(source)
    rate = _frame_rate(path)
    shares: List[float] = []
    previous: Optional[np.ndarray] = None
    for frame in _grid_frames(path, rate):
        shares.append(0.0 if previous is None else float(np.mean(np.abs(frame - previous) > _CELL_CHANGE)))
        previous = frame
    return {"frame_rate": rate, "shares": np.asarray(shares)}


# A frame is blank when even its brightest 1% of cells stay below this grey
# level; the frames either side of a blank flash must show a picture, already
# at (at least 75% of) the brightness it holds over the next or last 0.2 s.
_BLANK_P99 = 40.0
_PICTURE_P99 = 80.0
_SETTLED_SHARE = 0.75
_SETTLED_SEC = 0.2
_MAX_FLASH_SEC = 0.15
_SPLICE_MAX_FRAMES = 2
_SPLICE_CONTEXT_INSERT_RATIO = 0.10
_SPLICE_CONTEXT_MOTION_RATIO = 4.0
_SPLICE_CONTEXT_DELTA_NOISE = 1.0
_SPLICE_CONTEXT_SHARE_NOISE = 0.01
_SPLICE_MIN_FRAME_DELTA = 10.0
_SPLICE_MIN_FRAME_CHANGED_SHARE = 0.20


def _runs(flags: np.ndarray) -> List[tuple]:
    """(first, last) frame index of each run of True values, in time order."""
    runs: List[tuple] = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(flags) and flags[end + 1]:
            end += 1
        runs.append((index, end))
        index = end + 1
    return runs


def _brightest_percent(path: Path, rate: float) -> np.ndarray:
    """Per frame, the grey level of its brightest 1% of grid cells (the near-black test)."""
    return np.asarray([float(np.percentile(frame, 99)) for frame in _grid_frames(path, rate)])


def blank_flashes(source: str) -> Dict[str, Any]:
    """Runs of at most 0.15 s of near-black frames between two visible pictures.

    A dark scene lasts longer, and a fade brightens or darkens over several
    frames, so only a stray frame or two of black between two steady pictures
    (a gap between clips, a missing frame) counts.
    """
    path = _video_path(source)
    rate = _frame_rate(path)
    bright = _brightest_percent(path, rate)
    blank = bright < _BLANK_P99
    flashes: List[Dict[str, Any]] = []
    for index, end in _runs(blank):
        frames = end - index + 1
        span = max(1, int(round(_SETTLED_SEC * rate)))
        if (index > 0 and end + 1 < len(blank) and frames / rate <= _MAX_FLASH_SEC + 1e-6
                and bright[index - 1] >= max(_PICTURE_P99, _SETTLED_SHARE * bright[max(0, index - span):index].max())
                and bright[end + 1] >= max(_PICTURE_P99, _SETTLED_SHARE * bright[end + 1:end + 1 + span].max())):
            flashes.append({"start_sec": round(index / rate, 3), "end_sec": round((end + 1) / rate, 3),
                            "frames": frames, "brightest_before": round(float(bright[index - 1]), 1),
                            "brightest_after": round(float(bright[end + 1]), 1)})
    return {"frame_rate": rate, "flashes": flashes}


_BLACK_MIN_SEC = 0.2
_BLACK_CACHE_KEY = "black_ranges_v1"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cached_black_ranges(cache: MediaCache, fingerprint: str) -> Optional[Dict[str, Any]]:
    """The saved black scan when it has the shape black_ranges writes; anything else is a cache miss."""
    cached = cache.load_json(fingerprint, _BLACK_CACHE_KEY)
    if not isinstance(cached, dict) or not _is_number(cached.get("frame_rate")) or cached["frame_rate"] <= 0:
        return None
    ranges = cached.get("ranges")
    if not isinstance(ranges, list):
        return None
    for item in ranges:
        if not isinstance(item, dict) or not all(_is_number(item.get(key)) for key in ("start_sec", "end_sec", "duration_sec")):
            return None
        if item["end_sec"] < item["start_sec"]:
            return None
    return cached


def black_ranges(source: str, cache: Optional[MediaCache] = None) -> Dict[str, Any]:
    """Stretches of at least 0.2 s where every frame is near-black (the blank_flashes test).

    A frame is near-black when its brightest 1% of grid cells stay below
    _BLANK_P99, so a dim scene that keeps some bright cells is not listed.
    """
    path = _video_path(source)
    if cache is None:
        cache = MediaCache()
    fingerprint = cache.compute_fingerprint(str(path))
    cached = _cached_black_ranges(cache, fingerprint)
    if cached is not None:
        return cached
    rate = _frame_rate(path)
    blank = _brightest_percent(path, rate) < _BLANK_P99
    ranges: List[Dict[str, Any]] = []
    for index, end in _runs(blank):
        frames = end - index + 1
        if frames / rate + 1e-9 >= _BLACK_MIN_SEC:
            ranges.append({"start_sec": round(index / rate, 3), "end_sec": round((end + 1) / rate, 3),
                           "duration_sec": round(frames / rate, 3)})
    result = {"frame_rate": rate, "ranges": ranges}
    cache.save_json(fingerprint, _BLACK_CACHE_KEY, result)
    return result


def frame_splices(source: str) -> Dict[str, Any]:
    """Find one- or two-frame inserts between matching pictures.

    A true cut changes the pictures on both sides. A short splice instead
    stands apart from both boundaries while the surrounding frames match.
    The bounded run length and boundary comparison also keep ordinary motion,
    scrolling scenery, fades, and long cuts from looking like a stray frame.
    """
    path = _video_path(source)
    rate = _frame_rate(path)
    frames = [frame.astype(np.uint8) for frame in _grid_frames(path, rate)]
    splices: List[Dict[str, Any]] = []

    def difference(left: np.ndarray, right: np.ndarray) -> tuple:
        delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
        return float(np.mean(delta)), float(np.mean(delta > _CELL_CHANGE))

    adjacent = [difference(left, right) for left, right in zip(frames, frames[1:])]

    def context_matches_motion(index: int, end: int, span: int,
                               context_delta: float, context_share: float) -> bool:
        """Require the surrounding frames to move like this shot normally does."""
        radius = max(3 * span, int(round(rate * 0.2)))
        first = max(0, index - 1 - radius)
        last = min(len(adjacent), end + radius + 1)
        motion = [adjacent[pair] for pair in range(first, last)
                  if pair < index - 1 or pair >= end]
        if not motion:
            return False
        ordinary_delta = float(np.median([item[0] for item in motion])) * span
        ordinary_share = min(1.0, float(np.median([item[1] for item in motion])) * span)
        return (
            ordinary_delta * 0.25 - _SPLICE_CONTEXT_DELTA_NOISE <= context_delta
            <= ordinary_delta * _SPLICE_CONTEXT_MOTION_RATIO + _SPLICE_CONTEXT_DELTA_NOISE
            and ordinary_share * 0.25 - _SPLICE_CONTEXT_SHARE_NOISE <= context_share
            <= ordinary_share * _SPLICE_CONTEXT_MOTION_RATIO + _SPLICE_CONTEXT_SHARE_NOISE
        )

    index = 1
    while index < len(frames) - 1:
        found = None
        for count in range(_SPLICE_MAX_FRAMES, 0, -1):
            end = index + count
            if end >= len(frames):
                continue
            before, after = frames[index - 1], frames[end]
            context_delta, context_share = difference(before, after)
            inserts = frames[index:end]
            before_distances = [difference(frame, before) for frame in inserts]
            after_distances = [difference(frame, after) for frame in inserts]
            distances = before_distances + after_distances
            if any(delta < _SPLICE_MIN_FRAME_DELTA or share < _SPLICE_MIN_FRAME_CHANGED_SHARE
                   for delta, share in distances):
                continue
            smallest_delta = min(delta for delta, _ in distances)
            smallest_share = min(share for _, share in distances)
            if (context_delta > _SPLICE_CONTEXT_INSERT_RATIO * smallest_delta
                    or context_share > _SPLICE_CONTEXT_INSERT_RATIO * smallest_share
                    or not context_matches_motion(index, end, count + 1, context_delta, context_share)):
                continue
            found = {
                "start_sec": round(index / rate, 3),
                "end_sec": round(end / rate, 3),
                "frames": count,
                "context_delta": round(context_delta, 2),
                "context_changed_share": round(context_share, 4),
                "frame_delta_before": round(min(delta for delta, _ in before_distances), 2),
                "frame_delta_after": round(min(delta for delta, _ in after_distances), 2),
                "frame_changed_share": round(min(share for _, share in distances), 4),
            }
            break
        if found is None:
            index += 1
        else:
            splices.append(found)
            index += int(found["frames"])
    return {"frame_rate": rate, "splices": splices}


_FROZEN_MIN_SEC = 0.5
_FROZEN_MAX_MEAN_DELTA = 0.35
_FROZEN_MAX_CHANGED_SHARE = 0.002
_FROZEN_MAX_TOTAL_DELTA = 1.0
_FROZEN_CONTEXT_SEC = 0.35
_FROZEN_CONTEXT_MOVE_SHARE = 0.0035
_FROZEN_CONTEXT_ACTIVE_SHARE = 0.001
_FROZEN_CUT_SHARE = 0.20


def _near_identical(delta: np.ndarray) -> bool:
    """Whether a frame-to-frame grey difference is small enough to call the frames the same picture."""
    return (float(np.mean(delta)) <= _FROZEN_MAX_MEAN_DELTA
            and float(np.mean(delta > 2.0)) <= _FROZEN_MAX_CHANGED_SHARE)


def frozen_pictures(source: str) -> Dict[str, Any]:
    """Find held frames inside a shot that moves on both sides.

    Near-identical frame pairs must continue for at least half a second. A
    still shot, title or end card has no surrounding motion, and a cut at the
    start of a hold is treated as a new shot rather than a freeze.
    """
    path = _video_path(source)
    rate = _frame_rate(path)
    result: Dict[str, Any] = {"frame_rate": rate, "frozen_pictures": []}
    context_frames = max(1, int(round(_FROZEN_CONTEXT_SEC * rate)))
    minimum_pairs = int(math.ceil(_FROZEN_MIN_SEC * rate))
    changes: List[float] = []
    held_runs: List[tuple] = []
    previous: Optional[np.ndarray] = None
    run_start: Optional[int] = None
    run_end: Optional[int] = None
    run_start_frame: Optional[np.ndarray] = None
    run_end_frame: Optional[np.ndarray] = None

    def close_run() -> None:
        if run_start is None or run_end is None or run_start_frame is None or run_end_frame is None:
            return
        if run_end - run_start + 1 < minimum_pairs:
            return
        endpoint_delta = np.abs(run_end_frame - run_start_frame)
        gradual_change = (float(np.mean(endpoint_delta)) > _FROZEN_MAX_TOTAL_DELTA
                          or float(np.mean(endpoint_delta > 2.0)) > 0.01)
        if not gradual_change:
            held_runs.append((run_start, run_end))

    for frame_index, frame in enumerate(_grid_frames(path, rate)):
        if previous is not None:
            pair_index = frame_index - 1
            delta = np.abs(previous - frame)
            changed_share = float(np.mean(delta > 2.0))
            changes.append(changed_share)
            if _near_identical(delta):
                if run_start is None:
                    run_start = pair_index
                    run_start_frame = previous
                run_end = pair_index
                run_end_frame = frame
            elif run_start is not None:
                close_run()
                run_start = run_end = None
                run_start_frame = run_end_frame = None
        previous = frame
    if run_start is not None:
        close_run()
    if not changes:
        return result
    changes_array = np.asarray(changes, dtype=np.float32)

    def moving_context(values: np.ndarray) -> bool:
        useful = values[values < _FROZEN_CUT_SHARE]
        return (len(useful) >= max(1, context_frames // 2)
                and float(np.mean(useful)) >= _FROZEN_CONTEXT_MOVE_SHARE
                and float(np.mean(useful > _FROZEN_CONTEXT_ACTIVE_SHARE)) >= 0.8)

    for index, end in held_runs:
        pair_span = end - index + 1
        before = changes_array[max(0, index - context_frames):index]
        after = changes_array[end + 1:min(len(changes_array), end + 1 + context_frames)]
        starts_at_cut = index > 0 and changes_array[index - 1] >= _FROZEN_CUT_SHARE
        if (not starts_at_cut
                and moving_context(before) and moving_context(after)):
            motion_before = float(np.mean(before[before < _FROZEN_CUT_SHARE]))
            motion_after = float(np.mean(after[after < _FROZEN_CUT_SHARE]))
            result["frozen_pictures"].append({
                "start_sec": round(index / rate, 3),
                "end_sec": round((end + 1) / rate, 3),
                "frames": pair_span + 1,
                "duration_sec": round(pair_span / rate, 3),
                "motion_before_share": round(motion_before, 4),
                "motion_after_share": round(motion_after, 4),
            })
    return result


_CADENCE_WINDOW_SEC = 0.5
_CADENCE_MIN_DUP_SHARE = 0.25
_CADENCE_MAX_HOLD_SEC = _FROZEN_MIN_SEC
_CADENCE_MIN_RANGE_SEC = 0.5


def _rolling_sums(flags: np.ndarray, window: int) -> np.ndarray:
    """Number of set flags in every run of `window` consecutive entries."""
    running = np.concatenate(([0], np.cumsum(flags)))
    return running[window:] - running[:-window]


def frame_cadence(source: str) -> Dict[str, Any]:
    """Find time ranges where the unique-frame rate is below the container frame rate.

    A frame is a repeat when it is near-identical to the one before it. A range
    is covered by half-second windows in which at least a quarter of the frames
    are repeats, with changing frames in between. A run of repeats longer than
    half a second is a hold (still shot, title card, stall) and is left out.
    """
    path = _video_path(source)
    rate = _frame_rate(path)
    repeat: List[bool] = []
    previous: Optional[np.ndarray] = None
    for frame in _grid_frames(path, rate):
        repeat.append(previous is not None and _near_identical(np.abs(frame - previous)))
        previous = frame
    count = len(repeat)
    result: Dict[str, Any] = {"frame_rate": rate, "frames": count, "held_frames": 0, "ranges": [], "total": 0}

    # Take long runs of repeats out of the count: they are holds, and they split ranges.
    held = np.zeros(count, dtype=bool)
    min_hold = int(math.ceil(_CADENCE_MAX_HOLD_SEC * rate))  # the shortest run frozen_pictures reports
    run_start = 0
    for index in range(count + 1):
        if index < count and repeat[index]:
            continue
        if index - run_start >= min_hold:
            held[run_start:index] = True
        run_start = index + 1
    result["held_frames"] = int(held.sum())
    repeated = np.asarray(repeat, dtype=bool) & ~held

    window = max(4, int(round(_CADENCE_WINDOW_SEC * rate)))
    if count < window:
        return result
    sums = _rolling_sums(repeated, window)
    held_sums = _rolling_sums(held, window)
    flagged = (sums >= _CADENCE_MIN_DUP_SHARE * window) & (held_sums == 0)
    covered = np.zeros(count, dtype=bool)
    for start in np.flatnonzero(flagged):
        covered[start:start + window] = True

    index = 0
    while index < count:
        if not covered[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and covered[end + 1]:
            end += 1
        repeats_at = np.flatnonzero(repeated[index:end + 1]) + index
        # Start at the frame the first repeat copies, so doubled frames read as exactly half the rate.
        first, last = int(repeats_at[0]) - 1, int(repeats_at[-1])
        frames = last - first + 1
        if frames / rate >= _CADENCE_MIN_RANGE_SEC:
            duplicates = int(repeated[first + 1:last + 1].sum())
            duration = frames / rate
            result["ranges"].append({
                "start_sec": round(first / rate, 3),
                "end_sec": round((last + 1) / rate, 3),
                "frames": frames,
                "duplicate_frames": duplicates,
                "effective_fps": round((frames - duplicates) / duration, 2),
            })
        index = end + 1
    result["total"] = len(result["ranges"])
    return result


def _appearances(times: np.ndarray, shares: np.ndarray) -> List[Dict[str, float]]:
    """Runs of frames that stand out from the window's background motion."""
    base = float(np.median(shares)) if len(shares) else 0.0
    loud = (shares >= base + _STAND_OUT) & (shares >= 2.0 * base)
    runs: List[Dict[str, float]] = []
    for index in np.flatnonzero(loud):
        if runs and index - runs[-1]["_last"] == 1:
            runs[-1]["_last"] = index
            runs[-1]["peak_share"] = max(runs[-1]["peak_share"], float(shares[index]))
        else:
            runs.append({"time_sec": float(times[index]), "peak_share": float(shares[index]), "_last": index})
    for run in runs:
        run.pop("_last")
    return runs


def text_cue_timing(
    source: str, cues: Sequence[Mapping[str, Any]], bpm: Optional[float] = None,
    tolerance_ms: float = 80.0,
) -> Dict[str, Any]:
    """For each text cue, the moment its part of the picture changes.

    - ``ok``: something appears within tolerance (at least two frames).
    - ``early``/``late``: nothing stands out at the cue, and a caption-sized
      appearance lies elsewhere in the cue's window (1.5 beats, or 1 s
      without a tempo, and no closer to a neighbouring text cue).
    - ``unseen``: nothing clear either way; no timing claim is made.
    """
    text_cues = [(float(cue.get("time_sec", cue.get("t", 0.0))), str(cue.get("label", "")))
                 for cue in cues if "|" in str(cue.get("label", ""))]
    result: Dict[str, Any] = {"text_cues": []}
    if not text_cues:
        return result
    measured = change_shares(source)
    rate = float(measured["frame_rate"])
    shares = measured["shares"]
    # Share i compares frame i with frame i-1, so it belongs to frame i's time.
    times = np.arange(len(shares)) / rate
    tolerance = max(tolerance_ms / 1000.0, 2.0 / rate)
    reach = _WINDOW_BEATS * 60.0 / float(bpm) if bpm else 1.0
    starts = sorted({time for time, _ in text_cues})
    for cue_time, label in text_cues:
        before = min([reach] + [(cue_time - other) / 2.0 for other in starts if other < cue_time - 1e-6])
        after = min([reach] + [(other - cue_time) / 2.0 for other in starts if other > cue_time + 1e-6])
        mask = (times >= cue_time - before) & (times <= cue_time + after)
        runs = _appearances(times[mask], shares[mask])
        on_cue = [run for run in runs if abs(run["time_sec"] - cue_time) <= tolerance]
        displaced = [run for run in runs if abs(run["time_sec"] - cue_time) > tolerance
                     and run["peak_share"] >= _MIN_DISPLACED_SHARE]
        entry: Dict[str, Any] = {"time_sec": round(cue_time, 6), "label": label,
                                 "window_before_sec": round(before, 3), "window_after_sec": round(after, 3),
                                 "tolerance_ms": round(tolerance * 1000.0, 1)}
        if on_cue:
            best = max(on_cue, key=lambda run: run["peak_share"])
            entry.update(status="ok", appears_sec=round(best["time_sec"], 3), peak_share=round(best["peak_share"], 4))
        elif displaced:
            best = max(displaced, key=lambda run: run["peak_share"])
            offset = best["time_sec"] - cue_time
            entry.update(status="late" if offset > 0 else "early", appears_sec=round(best["time_sec"], 3),
                         offset_ms=round(offset * 1000.0, 1), peak_share=round(best["peak_share"], 4))
        else:
            entry.update(status="unseen")
        result["text_cues"].append(entry)
    return result


# Text strokes are sharp: a pixel is stroke-like when it differs from its
# 31x31 neighbourhood mean by more than this many grey levels.
_STROKE_CONTRAST = 40.0
_STROKE_RADIUS = 15
_EDGE_PX = 4
_INNER_PX = 64
_INNER_DENSITY = 0.08
# Near-edge candidates stay review-only; there is no automatic alignment confirmation.
_NEAR_EDGE_MAX_PX = 24
_NEAR_EDGE_FRACTION = 0.02
_TEXT_BAND_GAP_PX = 4
_MIN_TEXT_BAND_HEIGHT_PX = 16
# Rows (or columns, for top/bottom) where strokes run into the edge; about
# the height of one line of caption text.
_MIN_TOUCH_LINES = 24
_SETTLE_OFFSETS_SEC = (0.3, 0.6, 0.9)


def _frame_size(path: Path) -> tuple:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True, check=False,
        timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    width, _, height = probe.stdout.strip().partition("x")
    try:
        return int(width), int(height)
    except ValueError as exc:
        raise MediaCorruptError("Could not read the video frame size", {"source": str(path)}) from exc


def _grey_frame(path: Path, time_sec: float, size: tuple) -> Optional[np.ndarray]:
    width, height = size
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", "{:.3f}".format(max(0.0, time_sec)), "-i", str(path),
         "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=False, timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    if completed.returncode != 0 or len(completed.stdout) < width * height:
        return None
    return np.frombuffer(completed.stdout[:width * height], dtype=np.uint8).reshape(height, width).astype(np.float32)


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(image, radius, mode="edge")
    summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * radius + 1
    return (summed[k:, k:] - summed[:-k, k:] - summed[k:, :-k] + summed[:-k, :-k]) / float(k * k)


def _stroke_mask(image: np.ndarray) -> np.ndarray:
    return np.abs(image - _box_mean(image, _STROKE_RADIUS)) > _STROKE_CONTRAST


def _edge_lines(strokes: np.ndarray, margin_px: int = _EDGE_PX) -> Dict[str, int]:
    margin_px = max(1, int(margin_px))
    views = {
        "left": (strokes[:, :margin_px], strokes[:, margin_px:margin_px + _INNER_PX]),
        "right": (strokes[:, -margin_px:], strokes[:, -margin_px - _INNER_PX:-margin_px]),
        "top": (strokes[:margin_px, :].T, strokes[margin_px:margin_px + _INNER_PX, :].T),
        "bottom": (strokes[-margin_px:, :].T, strokes[-margin_px - _INNER_PX:-margin_px, :].T),
    }
    return {side: int(np.sum(edge.any(axis=1) & (inner.mean(axis=1) >= _INNER_DENSITY)))
            for side, (edge, inner) in views.items()}


def _text_rows(image: np.ndarray) -> np.ndarray:
    """Rows with repeated stroke runs, as in a line of text rather than scenery."""
    width = image.shape[1]
    row_runs = np.count_nonzero(np.abs(np.diff(image, axis=1)) >= _STROKE_CONTRAST, axis=1)
    active = row_runs >= max(4, width // 160)
    rows = np.zeros(len(active), dtype=bool)
    indices = np.flatnonzero(active)
    if not len(indices):
        return rows
    first = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index > previous + 1:
            if previous - first + 1 >= 3:
                rows[first:previous + 1] = True
            first = index
        previous = index
    if previous - first + 1 >= 3:
        rows[first:previous + 1] = True
    return rows


def _text_edge_lines(strokes: np.ndarray, image: np.ndarray, margin_px: int = _EDGE_PX) -> Dict[str, int]:
    """Count edge or near-edge strokes only in text-like row bands."""
    margin_px = max(1, int(margin_px))
    width = strokes.shape[1]
    text_rows = _text_rows(image)
    top_rows = text_rows[:margin_px + _INNER_PX]
    bottom_rows = text_rows[-(margin_px + _INNER_PX):]
    views = {
        "left": (strokes[:, :margin_px], strokes[:, margin_px:margin_px + _INNER_PX], text_rows),
        "right": (strokes[:, -margin_px:], strokes[:, -margin_px - _INNER_PX:-margin_px], text_rows),
        "top": (strokes[:margin_px, :].T, strokes[margin_px:margin_px + _INNER_PX, :].T,
                np.any(strokes[:margin_px + _INNER_PX, :][top_rows], axis=0)
                if np.any(top_rows) else np.zeros(width, dtype=bool)),
        "bottom": (strokes[-margin_px:, :].T, strokes[-margin_px - _INNER_PX:-margin_px, :].T,
                   np.any(strokes[-(margin_px + _INNER_PX):, :][bottom_rows], axis=0)
                   if np.any(bottom_rows) else np.zeros(width, dtype=bool)),
    }
    counts: Dict[str, int] = {}
    for side, (edge, inner, eligible) in views.items():
        touches = edge.any(axis=1) & (inner.mean(axis=1) >= _INNER_DENSITY)
        counts[side] = int(np.count_nonzero(touches & eligible))
    return counts


def _text_edge_pixels(strokes: np.ndarray, image: np.ndarray, side: str, margin_px: int = _EDGE_PX) -> bool:
    text_rows = _text_rows(image)
    if side == "left":
        return bool(np.any(strokes[:, :margin_px] & text_rows[:, None]))
    if side == "right":
        return bool(np.any(strokes[:, -margin_px:] & text_rows[:, None]))
    if side == "top":
        rows = text_rows[:margin_px]
        return bool(np.any(strokes[:margin_px, :][rows]))
    rows = text_rows[-margin_px:]
    return bool(np.any(strokes[-margin_px:, :][rows]))


def _text_bands(image: np.ndarray) -> List[tuple]:
    """Group text-like rows into line bands, merging small gaps between glyphs."""
    rows = np.flatnonzero(_text_rows(image))
    if not len(rows):
        return []
    bands = []
    first = previous = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row - previous - 1 > _TEXT_BAND_GAP_PX:
            end = previous + 1
            if end - first >= _MIN_TEXT_BAND_HEIGHT_PX:
                bands.append((first, end))
            first = row
        previous = row
    end = previous + 1
    if end - first >= _MIN_TEXT_BAND_HEIGHT_PX:
        bands.append((first, end))
    return bands


def _line_edge_margin_px(strokes: np.ndarray, side: str, frame_height: int, y_offset: int) -> int:
    ys, xs = np.nonzero(strokes)
    if not len(xs):
        return 0
    if side == "left":
        return int(xs.min() + 1)
    if side == "right":
        return int(strokes.shape[1] - xs.max())
    if side == "top":
        return int(y_offset + ys.min() + 1)
    return int(frame_height - (y_offset + ys.max()))


def _edge_glyph_run(line_strokes: np.ndarray, side: str, margin_limit_px: int) -> bool:
    """Require a near-edge mark to join a long line of closely spaced glyphs.

    Count columns with strokes across a meaningful part of the line height;
    this discards a thin hanging wire. Work inward from the edge so a nearby
    decoration cannot borrow a separate title elsewhere in the same row band.
    Letter and word spaces may be wide, but a gap comparable to the line's
    height separates objects rather than glyphs in one line.
    """
    line_height = line_strokes.shape[0]
    substantial = np.count_nonzero(line_strokes, axis=0) >= max(3, int(math.ceil(line_height * 0.08)))
    positions = np.flatnonzero(substantial)
    if not len(positions):
        return False
    positions = positions[::-1] if side == "right" else positions
    edge_distance = (line_strokes.shape[1] - positions[0]) if side == "right" else (positions[0] + 1)
    if edge_distance > margin_limit_px:
        return False
    max_gap = max(4, int(round(line_height * 0.6)))
    end = 1
    while end < len(positions) and abs(int(positions[end]) - int(positions[end - 1])) <= max_gap:
        end += 1
    return end >= 4 and abs(int(positions[end - 1]) - int(positions[0])) >= 3 * line_height


def _near_edge_text_bands(strokes: np.ndarray, before_strokes: np.ndarray, image: np.ndarray,
                          side: str, margin_limit_px: int) -> List[Dict[str, Any]]:
    """Find tall, text-like line bands whose stable stroke extent nears an edge.

    A few rows of a glyph can set a line's full extent, so proximity is measured
    per band rather than by counting how many individual rows reach the margin.
    Matching near-edge strokes in the preceding frame suppress a stationary
    border or title that was already present.
    """
    height, width = strokes.shape
    if side == "left":
        edge_region = (slice(None), slice(_EDGE_PX, margin_limit_px))
    elif side == "right":
        edge_region = (slice(None), slice(width - margin_limit_px, width - _EDGE_PX))
    elif side == "top":
        edge_region = (slice(_EDGE_PX, margin_limit_px), slice(None))
    else:
        edge_region = (slice(height - margin_limit_px, height - _EDGE_PX), slice(None))

    found = []
    for y0, y1 in _text_bands(image):
        line_strokes = strokes[y0:y1, :]
        margin = _line_edge_margin_px(line_strokes, side, height, y0)
        if not (_EDGE_PX < margin <= margin_limit_px):
            continue
        if side in {"left", "right"} and not _edge_glyph_run(line_strokes, side, margin_limit_px):
            continue

        if side in {"left", "right"}:
            region = (slice(y0, y1), edge_region[1])
        else:
            region = (slice(max(y0, edge_region[0].start), min(y1, edge_region[0].stop)), slice(None))
        near_strokes = strokes[region]
        if not near_strokes.size:
            continue
        near_positions = np.any(near_strokes, axis=1 if side in {"left", "right"} else 0)
        position_count = int(np.count_nonzero(near_positions))
        if not position_count:
            continue

        prior_strokes = before_strokes[region]
        matching_positions = np.any(near_strokes & prior_strokes,
                                    axis=1 if side in {"left", "right"} else 0)
        prior_count = int(np.count_nonzero(matching_positions))
        if prior_count * 2 >= position_count:
            continue
        found.append({
            "y_start": y0, "y_end": y1, "height_px": y1 - y0,
            "margin_px": margin, "near_edge_positions": position_count,
            "previous_comparable_positions": prior_count,
        })
    return found


def edge_touches(image: np.ndarray) -> Dict[str, int]:
    """Lines of stroke-like pixels that run into each frame edge.

    A row counts for the left/right edge (a column for top/bottom) when the
    outer 4 px hold a stroke pixel and the next 64 px inward are at least 8%
    strokes, so a lone line or a soft gradient at the edge does not count.
    """
    return _edge_lines(_stroke_mask(image))


_LOOKAHEAD_MAX = 3
_LOOKAHEAD_BYTES = 128 * 1024 * 1024
_STRETCH_FRAMES = 45
_PTS_LINE = re.compile(r"\bn:\s*\d+ pts:\s*(-?\d+) ")
_TIME_BASE_LINE = re.compile(r"config in time_base: (\d+)/(\d+)")
_START_LINE = re.compile(r"Duration: [^,]*, start: (-?\d+\.\d+)")
_MICROSECOND = Fraction(1, 1_000_000)


def _grey_frames(path: Path, times: Sequence[float], size: tuple) -> List[Optional[np.ndarray]]:
    """What `_grey_frame` returns for each time, from one decode of the stretch they span.

    `-ss` decodes from the keyframe before the time, so separate seeks decode that
    run again each time. Here the frames of the stretch come out with their
    timestamps, and each time takes the first frame at or after it, as `-ss` does.
    A time the stretch does not settle falls back to `_grey_frame`.
    """
    width, height = size
    frame_bytes = width * height
    seconds = [Fraction("{:.3f}".format(max(0.0, time_sec))) for time_sec in times]
    start = min(seconds)
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-nostats", "-v", "info", "-threads", "1", "-copyts", "-ss", "{:.3f}".format(float(start)),
         "-t", "{:.3f}".format(float(max(seconds) - start) + 0.15), "-i", str(path),
         "-vf", "showinfo,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True, check=False, timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    log = completed.stderr.decode("utf-8", errors="replace")
    time_base = _TIME_BASE_LINE.search(log)
    file_start = _START_LINE.search(log)
    # The filter also sees the frame at the `-t` limit, which is not written out.
    stamps = [int(match) for match in _PTS_LINE.findall(log)][:len(completed.stdout) // frame_bytes]
    frames: List[Optional[np.ndarray]] = [None] * len(seconds)
    if (completed.returncode == 0 and time_base and file_start and int(time_base.group(1)) > 0
            and int(time_base.group(2)) > 0 and len(stamps) * frame_bytes == len(completed.stdout)):
        unit = Fraction(int(time_base.group(1)), int(time_base.group(2)))
        offset = Fraction(file_start.group(1))
        for slot, wanted in enumerate(seconds):
            for index, stamp in enumerate(stamps):
                gap = stamp * unit - offset - wanted
                if gap < -_MICROSECOND:
                    continue
                # Within a microsecond (but not equal) FFmpeg's own rounding decides; leave that to `_grey_frame`.
                if gap > _MICROSECOND or gap == 0:
                    chunk = completed.stdout[index * frame_bytes:(index + 1) * frame_bytes]
                    frames[slot] = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width).astype(np.float32)
                break
    return [frame if frame is not None else _grey_frame(path, time_sec, size)
            for frame, time_sec in zip(frames, times)]


def _cue_frames(path: Path, size: tuple, plans: List[tuple]) -> Iterator[tuple]:
    """For each (item, appears, moments): the frame just before and the frames at the moments.

    The next cues' frames are decoded while this one is measured.
    """
    ahead = max(1, min(_LOOKAHEAD_MAX, _LOOKAHEAD_BYTES // (size[0] * size[1] * _STRETCH_FRAMES)))
    with ThreadPoolExecutor(max_workers=ahead) as pool:
        fetching: Dict[int, Any] = {}
        for position, (item, appears, moments) in enumerate(plans):
            for upcoming in range(position, min(len(plans), position + ahead)):
                if upcoming not in fetching:
                    _, upcoming_appears, upcoming_moments = plans[upcoming]
                    fetching[upcoming] = pool.submit(
                        _grey_frames, path, [upcoming_appears - 0.1] + upcoming_moments, size)
            before, *grabbed = fetching.pop(position).result()
            yield item, appears, before, [(moment, frame) for moment, frame in zip(moments, grabbed)
                                           if frame is not None]


def text_overflow(source: str, text_cues: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Text cues whose text runs off or ends suspiciously close to a frame edge.

    For each timed text cue (from `text_cue_timing`), frames 0.3, 0.6 and 0.9 s
    after the text appears (kept before the next text cue) are compared, and
    only strokes in the same place in every one of them count: a title holds
    still, while scrolling scenery at the edge does not. Edge strokes also
    need to align with a text-like row band, so a prop or solid shape that
    appears with the title is not mistaken for cropped text. Those strokes
    must run into an edge on at least 24 lines, and the frame just before the
    text appeared must show fewer than half as many, so a fixed border or
    scenery that always touches the edge does not count. A text-like line at
    least 16 px tall whose stable stroke extent ends within 2% of an edge
    (capped at 24 px), outside the outer 4 px, is also reported for review.
    This is measured per line, since only a small part of a final glyph may
    reach the near-edge margin; the preceding frame's matching strokes are
    checked to suppress lines that were already there.
    """
    path = _video_path(source)
    timed = [item for item in text_cues if item.get("status") in {"ok", "early", "late"}]
    result: Dict[str, Any] = {"overflows": []}
    if not timed:
        return result
    size = _frame_size(path)
    starts = sorted(float(item["time_sec"]) for item in text_cues)
    plans = []
    for item in timed:
        appears = float(item.get("appears_sec", item["time_sec"]))
        limit = min([other for other in starts if other > appears + 0.05] or [appears + 10.0]) - 0.1
        moments = sorted({min(appears + offset, limit) for offset in _SETTLE_OFFSETS_SEC if limit > appears + 0.05})
        plans.append((item, appears, moments))
    for item, appears, before, frames in _cue_frames(path, size, plans):
        if before is None or len(frames) < 2:
            continue
        still = np.logical_and.reduce([_stroke_mask(frame) for _, frame in frames])
        counts = _edge_lines(still)
        text_counts = _text_edge_lines(still, frames[0][1])
        before_strokes = _stroke_mask(before)
        baseline = _edge_lines(before_strokes)
        checked = [round(moment, 3) for moment, _ in frames]
        for side in ("left", "right", "top", "bottom"):
            lines = counts[side]
            if (text_counts[side] >= _MIN_TOUCH_LINES and lines >= _MIN_TOUCH_LINES
                    and baseline[side] * 2 < lines):
                result["overflows"].append({
                    "time_sec": float(item["time_sec"]), "label": str(item["label"]), "side": side,
                    "appears_sec": round(appears, 3), "lines_touching": lines, "lines_before": baseline[side],
                    "frames_checked": checked, "frames_touching": checked,
                })
                continue

            # Keep the existing edge-touch path authoritative. Near-edge rows
            # are review-only and must not share the outer 4 px band.
            if lines or text_counts[side] or _text_edge_pixels(still, frames[0][1], side):
                continue
            dimension = size[0] if side in {"left", "right"} else size[1]
            margin_limit_px = min(_NEAR_EDGE_MAX_PX, int(round(dimension * _NEAR_EDGE_FRACTION)))
            if margin_limit_px <= _EDGE_PX:
                continue
            line_bands = _near_edge_text_bands(
                still, before_strokes, frames[0][1], side, margin_limit_px,
            )
            if not line_bands:
                continue
            item_result = {
                "time_sec": float(item["time_sec"]), "label": str(item["label"]), "side": side,
                "appears_sec": round(appears, 3), "lines_touching": 0,
                "near_edge_positions": sum(line["near_edge_positions"] for line in line_bands),
                "previous_comparable_positions": sum(line["previous_comparable_positions"] for line in line_bands),
                "line_bands": line_bands,
                "margin_px": min(line["margin_px"] for line in line_bands),
                "margin_limit_px": margin_limit_px,
                "touches_edge": False, "frames_checked": checked, "frames_near": checked,
            }
            result["overflows"].append(item_result)
    return result
