"""Lightweight scene cut detection via FFmpeg filtergraph."""
import math
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, MediaNotFoundError, DependencyNotFoundError
from src.core.types import format_timestamp
from src.core.cache import MediaCache


@dataclass
class SceneCut:
    timestamp_sec: float
    score: float
    timecode: str
    kind: str = "cut"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSegment:
    scene_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_timecode: str
    end_timecode: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_scenes(
    video_path: str,
    threshold: float = 0.3,
    min_scene_len_sec: float = 0.5,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    cache: Optional[MediaCache] = None,
    use_cache: bool = True,
    timeout_sec: Optional[float] = None,
    threads: Optional[int] = None,
) -> List[SceneCut]:
    """Detect visual scene cuts using FFmpeg scene score filter.
    
    Uses `select='gt(scene,threshold)',metadata=print:key=lavfi.scene_score` with null sink.
    Runs at ~150-300+ FPS without decoding or saving video frames to disk.
    
    Recommended thresholds:
    - 0.3 - 0.4: standard live-action camera cuts
    - 0.15 - 0.25: animated UI transitions, presentation slides, screen recordings
    """
    vpath = Path(video_path).resolve()
    if not vpath.exists():
        raise MediaNotFoundError(f"Video file not found: {video_path}", {"path": str(vpath)})
        
    if cache is None:
        cache = MediaCache()
        
    fingerprint = cache.compute_fingerprint(str(vpath))
    cache_key = _scenes_key(threshold, _range_key(start_sec, duration_sec))
    
    if use_cache:
        cached = cache.load_json(fingerprint, cache_key)
        # An empty list is a result too (a video with no hard cuts).
        if cached is not None:
            return [SceneCut(**item) for item in cached]
            
    cmd = ["ffmpeg"]
    if threads is not None:
        cmd.extend(["-threads", str(threads), "-filter_threads", str(threads),
                    "-filter_complex_threads", str(threads)])
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    # Bound input decoding before scene selection. Output-side -t can scan
    # the entire silent/static input when select emits no frames.
    if duration_sec:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(["-i", str(vpath)])
        
    filter_expr = fr"select=gt(scene\,{threshold:.3f}),metadata=print:key=lavfi.scene_score"
    cmd.extend(["-filter_complex", filter_expr, "-f", "null", "-"])
    
    timeout = timeout_sec or (Config.SUBPROCESS_TIMEOUT_SEC * 2)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
    except FileNotFoundError as exc:
        raise DependencyNotFoundError(
            f"ffmpeg binary not found on system PATH: {exc}",
            {"dependency": "ffmpeg"}
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SubprocessTimeoutError(
            f"Scene detection timed out after {timeout}s",
            {"video_path": str(vpath), "threshold": threshold}
        ) from exc

    if proc.returncode != 0:
        raise MediaCorruptError(
            f"FFmpeg scene detection failed: {proc.stderr.strip()}",
            {"video_path": str(vpath), "stderr": proc.stderr}
        )

    lines = proc.stderr.splitlines()
    raw_cuts: List[SceneCut] = []
    curr_pts: Optional[float] = None
    
    for line in lines:
        pts_m = re.search(r"pts_time:([0-9.]+)", line)
        if pts_m:
            curr_pts = float(pts_m.group(1)) + start_sec
        score_m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if score_m and curr_pts is not None:
            score = float(score_m.group(1))
            raw_cuts.append(SceneCut(
                timestamp_sec=curr_pts,
                score=score,
                timecode=format_timestamp(curr_pts)
            ))
            curr_pts = None

    filtered_cuts = _debounce(raw_cuts, min_scene_len_sec, _frame_gap_sec(raw_cuts, str(vpath), cache))

    if use_cache:
        cache.save_json(fingerprint, cache_key, [c.to_dict() for c in filtered_cuts])
    return filtered_cuts


SOFT_SAMPLE_FPS = 10
SOFT_GRID = (32, 18)
# The grid of the near-black test (visual_timing.black_ranges); the cut scan can emit it in its own decode.
GRID_SIZE = (160, 90)
SOFT_LAG_SEC = 0.5
SOFT_MIN_DIFF = 0.045
SOFT_MIN_PROMINENCE = 0.03
SOFT_MIN_GAP_SEC = 1.0
SOFT_HARD_CUT_GAP_SEC = 0.7
# A small change in an otherwise still picture (two look-alike screens) counts
# when it stands almost alone and covers a good part of the frame.
SOFT_CALM_MIN_DIFF = 0.025
SOFT_CALM_MIN_PROMINENCE = 0.022
SOFT_CALM_MIN_SHARE = 0.8
SOFT_CALM_MIN_AREA = 0.25
SOFT_CELL_CHANGED = 0.04
# Motion inside a film shot moves pixels but keeps the colours; a scene change
# brings new colours. A peak needs a colour-histogram change of this size,
# unless the picture is still just before and just after it (screens, slides).
SOFT_MIN_HIST = 0.4
# Below this the histogram change must also stand out from the level around it
# (a moving camera keeps bringing some new colours in).
SOFT_SURE_HIST = 0.5
SOFT_MIN_HIST_PROMINENCE = 0.35
SOFT_STILL_LEVEL = 0.015
SOFT_STILL_MIN_DIFF = 0.035
SOFT_WEAK_DIFF = 0.06
SOFT_WEAK_MIN_BLEND = 0.1
# A weak peak that is not a dissolve is kept when the picture is steady on both sides (a fade between
# still pictures, with or without fixed bars) or when at least this share of the frame changes
# (a whole-frame fade or step). Otherwise it is motion.
SOFT_FADE_MIN_AREA = 0.9
SOFT_STILL_REACH_SEC = 1.0
# A dissolve or fade passes through the average of the pictures before and
# after it; moving objects do not. This lets dark dissolves through the gate.
SOFT_MAX_BLEND_ERROR = 0.28
# A hard cut that the scene-score filter missed (two similar pictures, or an
# old film transfer where the new picture takes a few frames to settle) still
# puts most of its change into one frame step, or two adjacent ones. A fade,
# wipe or slide spreads it over many. Measured on every frame within this
# reach of the change, counting only what rises above the usual step there.
SOFT_CUT_REACH_SEC = 0.6
SOFT_CUT_MIN_SHARE = 0.6
SOFT_CUT_MAX_PAIR_SEC = 0.07
# The one-decode scan keeps the scene score of every frame above this, so any
# hard-cut threshold from here up is answered without decoding again.
SCAN_SCORE_FLOOR = 0.05
# A scan this long is decoded as several parts at once.
SCAN_SPLIT_MIN_SEC = 240
SCAN_MIN_PART_SEC = 60
SCAN_MAX_PARTS = 8
SCAN_PAD_SEC = 1
# Below this (pixels per frame x seconds) scaling every frame to gray in the scan
# costs less than a decode per change; above it the changes are settled later.
SCAN_SMALL_PIXEL_SEC = 60_000_000


def _range_key(start_sec: float, duration_sec: Optional[float]) -> str:
    duration_key = "full" if duration_sec is None else float(duration_sec).hex()
    return f"start{float(start_sec).hex()}_dur{duration_key}"


def _changes_key(span: str) -> str:
    # v14: a weak peak with steady pictures on both sides is kept whatever share of the frame changes.
    # v13: a change on a large video may still carry its "window", settled by _settle_changes;
    # a rescued candidate carries a "rescue" marker and is kept only when it settles as a cut.
    return f"softtrans_v14_{span}"


def _scenes_key(threshold: float, span: str) -> str:
    # v2: a cut is timed at the strongest frame of its run, not the first frame over the threshold.
    return f"scenes_v2_th{float(threshold).hex()}_{span}"


def _frame_gap_sec(raw_cuts: List[SceneCut], video_path: str, cache: Optional[MediaCache]) -> float:
    """Longest time between two frames that are next to each other: 1.5 frame lengths.

    0.0 means no two frames count as adjacent. That is the answer when fewer
    than two frames are over the threshold (nothing to join, so the frame rate
    is not read) and when the frame rate cannot be read.
    """
    if len(raw_cuts) < 2:
        return 0.0
    try:
        from src.media.probe import probe_video
        return 1.5 / float(probe_video(video_path, cache=cache).fps)
    except Exception:
        return 0.0


def _debounce(raw_cuts: List[SceneCut], min_scene_len_sec: float, frame_gap_sec: float) -> List[SceneCut]:
    """One cut per run of adjacent over-threshold frames, at the run's strongest frame.

    A partly changed frame just before a cut can score over the threshold too;
    the cut is the stronger frame next to it. Frames further apart than
    frame_gap_sec are separate changes. Then drop cuts closer than
    min_scene_len_sec to the last one kept.
    """
    peaks: List[SceneCut] = []
    prev_t: Optional[float] = None
    for cut in raw_cuts:
        if prev_t is not None and cut.timestamp_sec - prev_t <= frame_gap_sec:
            if cut.score > peaks[-1].score:
                peaks[-1] = cut
        else:
            peaks.append(cut)
        prev_t = cut.timestamp_sec
    kept: List[SceneCut] = []
    last_t = -999.0
    for cut in peaks:
        if cut.timestamp_sec - last_t >= min_scene_len_sec:
            kept.append(cut)
            last_t = cut.timestamp_sec
    return kept


def _scan_plan(video_path: str, start_sec: float, duration_sec: Optional[float]) -> List[tuple]:
    """Split a long scan into parts decoded side by side: (offset, length) in seconds from start_sec.

    One ffmpeg decode speeds up little past a few threads, so a long video
    is faster as several parts at once. The last part has no length: it runs
    to the end of the scan. Parts start on whole seconds so the ten-a-second
    samples of all parts fall on one grid.
    """
    span = duration_sec
    if not span:
        try:
            from src.media.probe import probe_video
            span = float(probe_video(video_path).duration_sec) - start_sec
        except Exception:
            return [(0, None)]
    parts = int(min(SCAN_MAX_PARTS, max(1, (os.cpu_count() or 2) // 2), span // SCAN_MIN_PART_SEC))
    if span < SCAN_SPLIT_MIN_SEC or parts < 2:
        return [(0, None)]
    length = int(math.ceil(span / parts))
    return [(i * length, None if i == parts - 1 else length) for i in range(parts)]


def _scan_frames(video_path: str, start_sec: float, duration_sec: Optional[float], timeout: float, with_gray: bool = True,
                 grid_rate: Optional[float] = None, levels_out: Optional[list] = None):
    """Decode the video once for everything scene-change detection needs.

    Returns the time and ffmpeg scene score of every frame, the 32x18 colour
    samples taken ten times a second, and (with_gray) every frame as 32x18
    gray. Without with_gray the last item is None: scaling every frame down
    costs more than the rest of the scan, so _gray_frames reads only the
    few stretches that need it.

    With grid_rate, the same decode also yields the near-black levels of the
    GRID_SIZE grid at that rate (see brightest_levels), appended to levels_out.
    """
    plan = _scan_plan(video_path, start_sec, duration_sec)
    if grid_rate is not None and not float(grid_rate).is_integer():
        # Parts start on whole seconds; a fractional rate would put their frames off the grid.
        plan = [(0, None)]
    if len(plan) > 1:
        try:
            return _scan_parts(video_path, start_sec, duration_sec, timeout, plan, with_gray, grid_rate, levels_out)
        except MediaCorruptError:
            # A part that cannot be read on its own (a seek the container
            # does not support, or a timestamp gap across its seek point):
            # read the whole range in one go instead.
            pass
    return _scan_parts(video_path, start_sec, duration_sec, timeout, [(0, None)], with_gray, grid_rate, levels_out)


def brightest_levels(frames) -> Any:
    """Per frame, the grey level of its brightest 1% of cells: the near-black test of black_ranges."""
    import numpy as np

    return np.asarray([float(np.percentile(frame, 99)) for frame in frames])


def _scan_parts(video_path: str, start_sec: float, duration_sec: Optional[float], timeout: float, plan: List[tuple], with_gray: bool = True,
                grid_rate: Optional[float] = None, levels_out: Optional[list] = None):
    import numpy as np

    width, height = SOFT_GRID
    with_grid = grid_rate is not None
    labels = ["[a]", "[b]"] + (["[c]"] if with_gray else []) + (["[d]"] if with_grid else [])
    graph = (
        f"[0:v]split={len(labels)}{''.join(labels)};"
        r"[a]select=gte(scene\,0),metadata=print:key=lavfi.scene_score:file=scene.txt,nullsink;"
        f"[b]fps={SOFT_SAMPLE_FPS}:start_time=0,scale={width}:{height},format=rgb24[soft]"
        + (f";[c]scale={width}:{height},format=gray[native]" if with_gray else "")
        + (f";[d]fps={Fraction(grid_rate).limit_denominator(1001000)}:start_time=0,scale={GRID_SIZE[0]}:{GRID_SIZE[1]}:flags=area,format=gray[grid]" if with_grid else "")
    )
    outputs = ["-map", "[soft]", "-fps_mode", "passthrough", "-f", "rawvideo", "soft.raw"]
    if with_gray:
        outputs += ["-map", "[native]", "-fps_mode", "passthrough", "-f", "rawvideo", "native.raw"]
    if with_grid:
        # No -fps_mode: the same output options as the grid reader, so the frames are the same.
        outputs += ["-map", "[grid]", "-f", "rawvideo", "grid.raw"]
    threads = max(2, (os.cpu_count() or 2) // len(plan)) if len(plan) > 1 else None
    deadline = time.monotonic() + timeout
    pieces = []
    with tempfile.TemporaryDirectory(prefix="av-scene-scan-") as workdir:
        running = []
        try:
            for index, (offset, length) in enumerate(plan):
                # Each part after the first starts SCAN_PAD_SEC early, so its
                # first frame has a frame before it to be compared with, and
                # each part but the last runs that much late, so no sample at
                # the seam is missing. The overlap is cut away below.
                lead = SCAN_PAD_SEC if index > 0 else 0
                begin = start_sec + offset - lead
                if length is not None:
                    read = length + lead + SCAN_PAD_SEC
                else:
                    read = None if not duration_sec else duration_sec - offset + lead
                cmd = ["ffmpeg", "-v", "error", "-y"]
                if threads:
                    cmd.extend(["-threads", str(threads)])
                if begin > 0:
                    cmd.extend(["-ss", f"{begin:.3f}"])
                if read:
                    cmd.extend(["-t", f"{read:.3f}"])
                # passthrough: write each frame once. The default would repeat
                # or drop frames to hold a constant rate, and the gray frames
                # would no longer line up with the printed frame times.
                cmd.extend(["-i", video_path, "-filter_complex", graph, *outputs])
                part_dir = Path(workdir) / f"part{index}"
                part_dir.mkdir()
                running.append((subprocess.Popen(cmd, cwd=part_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), part_dir, begin, lead, length))
            for proc, part_dir, begin, lead, length in running:
                _, stderr = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                if proc.returncode != 0:
                    raise MediaCorruptError("FFmpeg scene-change scan failed", {"video_path": video_path, "stderr": stderr.decode(errors="replace")[-500:]})
                printed = (part_dir / "scene.txt").read_text() if (part_dir / "scene.txt").exists() else ""
                pairs = re.findall(r"pts_time:(\S+)\s+lavfi\.scene_score=(\S+)", printed)
                times = np.array([float(t) for t, _ in pairs], dtype=np.float64) + begin
                scores = np.array([float(s) for _, s in pairs], dtype=np.float32)
                rgb = np.fromfile(part_dir / "soft.raw", dtype=np.uint8)
                rgb = rgb[: (len(rgb) // (width * height * 3)) * width * height * 3].reshape(-1, height, width, 3)
                if with_gray:
                    gray = np.fromfile(part_dir / "native.raw", dtype=np.uint8)
                    gray = gray[: (len(gray) // (width * height)) * width * height].reshape(-1, height, width)
                    if len(times) != len(gray):
                        raise MediaCorruptError("FFmpeg scene-change scan lost frames", {"video_path": video_path, "times": len(times), "frames": len(gray)})
                else:
                    gray = np.zeros(len(times), dtype=np.uint8)  # placeholder, one per frame, so the slicing below stays the same
                # Keep this part's own stretch. The seam sits a little before
                # the whole second so a frame exactly on it lands on one side.
                own = times >= begin + lead - 1e-4
                skip = lead * SOFT_SAMPLE_FPS
                if length is not None:
                    own &= times < begin + lead + length - 1e-4
                    if len(rgb) < skip + length * SOFT_SAMPLE_FPS:
                        raise MediaCorruptError("FFmpeg scene-change scan came back short", {"video_path": video_path})
                    rgb = rgb[skip: skip + length * SOFT_SAMPLE_FPS]
                else:
                    rgb = rgb[skip:]
                # The grid and the soft samples start at the seek point (see their filters) and repeat the first
                # decoded frame back to it. If that frame comes after the seam, the start of the own stretch
                # would show the wrong picture, so such a part is not used.
                if lead and len(times) and float(times[0]) > begin + lead + 1e-3:
                    raise MediaCorruptError("FFmpeg scene-change scan starts after its own stretch", {"video_path": video_path})
                levels = None
                if with_grid:
                    # Row lead*rate is the first row of the part's own stretch, and it has length*rate rows.
                    area = GRID_SIZE[0] * GRID_SIZE[1]
                    grid = np.fromfile(part_dir / "grid.raw", dtype=np.uint8)
                    grid = grid[: (len(grid) // area) * area].reshape(-1, area)
                    first = int(round(lead * grid_rate))
                    last = len(grid) if length is None else first + int(round(length * grid_rate))
                    if last > len(grid):
                        raise MediaCorruptError("FFmpeg scene-change scan returned too few grid frames", {"video_path": video_path})
                    levels = brightest_levels(grid[first:last].astype(np.float32))
                pieces.append((times[own], scores[own], rgb, gray[own], levels))
        except FileNotFoundError as exc:
            raise DependencyNotFoundError(f"ffmpeg binary not found on system PATH: {exc}", {"dependency": "ffmpeg"}) from exc
        except subprocess.TimeoutExpired as exc:
            raise SubprocessTimeoutError(f"Scene-change scan timed out after {timeout}s", {"video_path": video_path}) from exc
        finally:
            for proc, *_ in running:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
    merged = tuple(np.concatenate([piece[i] for piece in pieces]) for i in range(4))
    if with_grid and levels_out is not None:
        levels_out.append(np.concatenate([piece[4] for piece in pieces]))
    return merged if with_gray else (*merged[:3], None)


def _lag_curves(rgb):
    """Compare the samples half a second before and after each sample.

    Returns the mean absolute difference, the fraction of grid cells that
    changed, the colour-histogram difference (8 levels per channel, 0-1), and
    how far the sample is from the average of the two frames (as a share of
    their difference; near 0 for a dissolve).
    """
    import numpy as np

    width, height = SOFT_GRID
    count = len(rgb)
    frames = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    bins = (rgb >> 5).astype(np.int32)
    bins = (bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]).reshape(count, -1)
    hist = np.zeros((count, 512), dtype=np.float32)
    for i in range(count):
        hist[i] = np.bincount(bins[i], minlength=512)
    hist /= float(width * height)
    lag = max(1, int(round(SOFT_LAG_SEC * SOFT_SAMPLE_FPS)))
    diff = np.zeros(count, dtype=np.float32)
    area = np.zeros(count, dtype=np.float32)
    colour = np.zeros(count, dtype=np.float32)
    blend = np.ones(count, dtype=np.float32)
    for i in range(lag, count - lag):
        delta = np.abs(frames[i - lag] - frames[i + lag])
        diff[i] = float(delta.mean())
        area[i] = float((delta > SOFT_CELL_CHANGED).mean())
        colour[i] = float(np.abs(hist[i - lag] - hist[i + lag]).sum()) / 2.0
        middle = (frames[i - lag] + frames[i + lag]) / 2.0
        blend[i] = float(np.abs(frames[i] - middle).mean()) / (float(diff[i]) + 1e-6)
    return diff, area, colour, blend


def _side_dip(diff, index: int, step: int, reach: int) -> float:
    """Lowest value on one side of a sample before the curve climbs above it."""
    value = float(diff[index])
    lowest = value
    j = index + step
    while 0 <= j < len(diff) and abs(j - index) <= reach and diff[j] <= value:
        lowest = min(lowest, float(diff[j]))
        j += step
    return lowest


def _find_transitions(diff, area, colour, blend, start_sec: float, rescue=None) -> List[tuple]:
    """Pick the scene changes out of the lag-difference curves: (time, strength) in time order."""
    import numpy as np

    count = len(diff)
    lag_edge = max(1, int(round(SOFT_LAG_SEC * SOFT_SAMPLE_FPS)))
    near = int(round(1.0 * SOFT_SAMPLE_FPS))
    far = int(round(3.0 * SOFT_SAMPLE_FPS))
    reach = int(round(SOFT_STILL_REACH_SEC * SOFT_SAMPLE_FPS))
    peaks = []
    for i in range(count):
        value = float(diff[i])
        if value < SOFT_CALM_MIN_DIFF:
            continue
        before = diff[max(0, i - far): max(0, i - near)]
        after = diff[min(count, i + near): min(count, i + far)]
        # A low percentile, not the median: a neighbouring transition can fill
        # most of one side and would otherwise hide this one.
        levels = [float(np.percentile(part, 25)) for part in (before, after) if len(part)]
        prominence = value - (max(levels) if levels else 0.0)
        if value < SOFT_MIN_DIFF:
            if (prominence >= SOFT_CALM_MIN_PROMINENCE and prominence >= SOFT_CALM_MIN_SHARE * value
                    and area[i] >= SOFT_CALM_MIN_AREA):
                peaks.append((value, i, True))
            continue
        calm_sides = prominence >= SOFT_MIN_PROMINENCE
        if prominence < SOFT_MIN_PROMINENCE:
            # A weak transition next to a strong one sits inside the strong
            # one's side window. Measure it against the dips on each side.
            prominence = value - max(_side_dip(diff, i, -1, far), _side_dip(diff, i, 1, far))
        if prominence >= SOFT_MIN_PROMINENCE:
            peaks.append((value, i, calm_sides))
    peaks.sort(reverse=True)
    # One transition gives one plateau of high values. Take the strongest
    # peak first and claim its whole plateau plus a margin, so weaker
    # peaks of the same transition are not reported again.
    claimed: List[tuple] = []
    gap = SOFT_MIN_GAP_SEC * SOFT_SAMPLE_FPS
    found = []
    for value, i, calm_sides in peaks:
        if any(lo_c - gap <= i <= hi_c + gap for lo_c, hi_c in claimed):
            continue
        lo = hi = i
        while lo > 0 and diff[lo - 1] >= 0.8 * value:
            lo -= 1
        while hi < count - 1 and diff[hi + 1] >= 0.8 * value:
            hi += 1
        # A plateau that runs into one already claimed is a second hump of
        # the same transition (a fade through white, a blur), not a new one.
        repeat = any(lo <= hi_c and hi >= lo_c for lo_c, hi_c in claimed)
        claimed.append((lo, hi))
        if repeat:
            continue
        # A stretch that runs into the first or last sample has only one
        # visible side: it may be steady motion, not a change that ends.
        if lo <= lag_edge or hi >= count - 1 - lag_edge:
            continue
        new_colours = float(colour[lo: hi + 1].max())
        if SOFT_MIN_HIST <= new_colours < SOFT_SURE_HIST:
            sides = [colour[max(0, lo - far): max(0, lo - near)], colour[min(count, hi + near): min(count, hi + far)]]
            around = [float(np.percentile(part, 25)) for part in sides if len(part)]
            if new_colours - (max(around) if around else 0.0) < SOFT_MIN_HIST_PROMINENCE:
                new_colours = 0.0
        left = diff[max(lag_edge, lo - reach): lo]
        right = diff[hi + 1: min(count - lag_edge, hi + 1 + reach)]
        # "Still" also needs calm side windows: a peak that only stood
        # out against the dips beside it sits in busy footage.
        still = calm_sides and max(float(left.min()), float(right.min())) <= SOFT_STILL_LEVEL
        if new_colours < SOFT_MIN_HIST and float(blend[lo: hi + 1].min()) > SOFT_MAX_BLEND_ERROR:
            rejected = not still or value < SOFT_STILL_MIN_DIFF
        else:
            rejected = (value < SOFT_WEAK_DIFF
                        and SOFT_WEAK_MIN_BLEND < float(blend[lo: hi + 1].min()) <= SOFT_MAX_BLEND_ERROR
                        and not still and float(area[lo: hi + 1].max()) < SOFT_FADE_MIN_AREA)
        if rejected:
            # Not a clear soft change: it is kept only if its frame step proves a hard cut.
            if rescue is not None:
                rescue.append((start_sec + ((lo + hi) / 2.0) / SOFT_SAMPLE_FPS, value))
            continue
        found.append((start_sec + ((lo + hi) / 2.0) / SOFT_SAMPLE_FPS, value))
    found.sort()
    return found


def _step_window(times, when: float) -> Optional[tuple]:
    """Slice bounds (lo, hi) of the frames _sharp_step looks at around `when`, or None when too few.

    The slice starts one frame early, so the first frame of the reach has a step into it.
    """
    import numpy as np

    first = max(1, int(np.searchsorted(times, when - SOFT_CUT_REACH_SEC)))
    last = int(np.searchsorted(times, when + SOFT_CUT_REACH_SEC, side="right"))
    return None if last - first < 6 else (first - 1, last)


def _gray_frames(video_path: str, times):
    """The 32x18 gray frames at the scan's frame times `times`, or None when they cannot be matched.

    Seeks to just before the first of them and reads only that stretch. The frame
    times ffmpeg prints are checked against the scan, so a seek that lands
    on other frames is caught instead of used.
    """
    import numpy as np

    width, height = SOFT_GRID
    count = len(times)
    begin = max(0.0, float(times[0]) - 0.01)
    # -ss in front of -i restarts the clock at `begin`; the scan's times are begin + clock.
    length = float(times[-1]) - begin + 0.005
    cmd = ["ffmpeg", "-v", "info", "-nostats", "-ss", f"{begin:.3f}", "-t", f"{length:.3f}", "-i", video_path,
           "-vf", f"showinfo,scale={width}:{height},format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    seen = [float(t) + begin for t in re.findall(r"pts_time:(\S+)", proc.stderr.decode(errors="replace"))]
    if len(seen) != count or not np.allclose(seen, times, atol=1e-3):
        return None
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    if len(frames) != count * width * height:
        return None
    return frames.reshape(count, height, width)


def _sharp_step(times, frames) -> Optional[tuple]:
    """Tell a hard cut from an animated change by how the change is spread over frames.

    `frames` are the gray frames at `times`: one frame before a _step_window,
    then the window. Looks at the step from each frame to the next. Returns
    the first and last index (into `times`) of the frames that end the step
    (one frame, or two adjacent ones) holding most of the change, or None
    when the change is spread out.
    """
    import numpy as np

    frames = frames.astype(np.float32) / 255.0
    steps = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))
    # Motion and grain give every step some size; only what rises above the
    # usual step counts as the change.
    excess = np.maximum(steps - float(np.median(steps)), 0.0)
    total = float(excess.sum())
    if total <= 0.0:
        return None
    peak = int(np.argmax(excess))
    lo = hi = peak
    left = float(excess[peak - 1]) if peak > 0 else 0.0
    right = float(excess[peak + 1]) if peak + 1 < len(excess) else 0.0
    if max(left, right) > 0.0:
        pair = (peak - 1, peak) if left >= right else (peak, peak + 1)
        # Two steps only count as one cut when they are really adjacent in
        # time (a low frame rate would turn a short slide into two steps).
        if float(times[1 + pair[1]] - times[1 + pair[0]]) <= SOFT_CUT_MAX_PAIR_SEC:
            lo, hi = pair
    if float(excess[lo: hi + 1].sum()) < SOFT_CUT_MIN_SHARE * total:
        return None
    # Step i ends at frame i + 1.
    return 1 + lo, 1 + hi


def _scan_is_small(video_path: str, start_sec: float, duration_sec: Optional[float]) -> bool:
    """True when scaling every frame is cheaper than one short decode per change."""
    try:
        from src.media.probe import probe_video
        info = probe_video(video_path)
        span = duration_sec or float(info.duration_sec) - start_sec
        return info.width * info.height * span < SCAN_SMALL_PIXEL_SEC
    except Exception:
        return False


def _scan_scene_changes(video_path: str, start_sec: float, duration_sec: Optional[float], timeout: float,
                        grid_rate: Optional[float] = None, levels_out: Optional[list] = None):
    """One decode: every frame's scene score above the floor, and the soft scene changes.

    A change comes back as a "transition" or, when it sits in one frame step,
    a "cut". On a large video the decode leaves that to _settle_changes: the
    change carries a "window" (times and scene scores of the frames around it).
    """
    with_gray = _scan_is_small(video_path, start_sec, duration_sec)
    times, scores, rgb, gray = _scan_frames(video_path, start_sec, duration_sec, timeout, with_gray=with_gray,
                                            grid_rate=grid_rate, levels_out=levels_out)
    strong = [[round(float(t), 6), round(float(s), 6)] for t, s in zip(times, scores) if s > SCAN_SCORE_FLOOR]
    changes = []
    rescued = []
    for when, value in _find_transitions(*_lag_curves(rgb), start_sec, rescued):
        entry = {"timestamp_sec": round(when, 3), "score": round(value, 6), "kind": "transition", "plateau_sec": round(when, 3)}
        window = _step_window(times, when)
        if window is not None:
            lo, hi = window
            if gray is not None:
                _settle_entry(entry, times[lo:hi], scores[lo:hi], gray[lo:hi])
            else:
                entry["window"] = {"times": [float(t) for t in times[lo:hi]], "scores": [float(x) for x in scores[lo:hi]]}
        changes.append(entry)
    for when, value in rescued:
        window = _step_window(times, when)
        if window is None:
            continue
        lo, hi = window
        entry = {"timestamp_sec": round(when, 3), "score": round(value, 6), "kind": "transition", "plateau_sec": round(when, 3), "rescue": True}
        if gray is not None:
            _settle_entry(entry, times[lo:hi], scores[lo:hi], gray[lo:hi])
            if entry["kind"] != "cut":
                continue
            del entry["rescue"]
        else:
            entry["window"] = {"times": [float(t) for t in times[lo:hi]], "scores": [float(x) for x in scores[lo:hi]]}
        changes.append(entry)
    return strong, changes


def _settle_entry(entry: Dict[str, Any], t, scores, frames) -> None:
    """Turn a change into a "cut" when it sits in one frame step (`frames` are the gray frames at times `t`)."""
    step = _sharp_step(t, frames)
    if step is not None:
        # A two-step change is timed like a hard cut: at its stronger frame.
        best = step[0] + int(scores[step[0]: step[1] + 1].argmax())
        entry.update(timestamp_sec=round(float(t[best]), 3), kind="cut", score=round(float(scores[best]), 6))


def _settle_changes(
    video_path: str,
    changes: List[Dict[str, Any]],
    start_sec: float,
    duration_sec: Optional[float],
    timeout: float,
    skip,
) -> bool:
    """Decide cut or transition for each change that still has a "window" (in place), except those `skip` names.

    A rescued candidate (marked "rescue") that does not settle as a cut is removed from `changes`.
    Reads the gray frames of each window with its own short decode, which
    costs about a third of the whole scan if done for every change; a caller
    that will throw some changes away anyway (the echoes of hard cuts) skips
    them. Returns whether anything was settled.
    """
    import numpy as np

    todo = [entry for entry in changes if "window" in entry and not skip(entry)]
    if not todo:
        return False
    stretch_times = [np.array(entry["window"]["times"], dtype=np.float64) for entry in todo]
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        jobs = [pool.submit(_gray_frames, video_path, t) for t in stretch_times]
        stretches = [job.result() for job in jobs]
    if any(frames is None for frames in stretches):
        # A stretch that cannot be read on its own: scan again with every frame.
        times, _, _, gray = _scan_frames(video_path, start_sec, duration_sec, timeout)
        stretches = []
        for t in stretch_times:
            at = int(np.searchsorted(times, t[0] - 1e-4))
            if not np.allclose(times[at: at + len(t)], t, atol=1e-4):
                raise MediaCorruptError("FFmpeg scene-change scan did not repeat", {"video_path": video_path})
            stretches.append(gray[at: at + len(t)])
    for entry, t, frames in zip(todo, stretch_times, stretches):
        _settle_entry(entry, t, np.array(entry.pop("window")["scores"], dtype=np.float32), frames)
    dropped = {id(entry) for entry in todo if entry.pop("rescue", False) and entry["kind"] == "transition"}
    if dropped:
        changes[:] = [entry for entry in changes if id(entry) not in dropped]
    return True


def _echoes_of(cut_times: Sequence[float]):
    """Test for a change that a hard cut at one of `cut_times` already explains."""
    return lambda entry: any(abs(entry["plateau_sec"] - t) < SOFT_HARD_CUT_GAP_SEC for t in cut_times)


def scan_scene_changes(
    video_path: str,
    thresholds: Sequence[float] = (0.25,),
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    cache: Optional[MediaCache] = None,
    timeout_sec: Optional[float] = None,
    defer_echoes: bool = False,
    grid_rate: Optional[float] = None,
) -> Optional[Any]:
    """Decode the video once and fill the cache that detect_scenes and detect_soft_transitions read.

    Calling it first lets a caller that wants both hard cuts and soft
    transitions pay for one decode instead of two. It only writes cache
    entries; results come from those two functions as before. A threshold
    below SCAN_SCORE_FLOOR is left for detect_scenes to measure itself.

    With defer_echoes, soft changes that sit on a hard cut of these
    thresholds are not settled now: detect_soft_transitions drops them when
    given those cuts, and settles them itself if given others.

    With grid_rate, a whole-file scan that runs here also returns the
    brightest-1% level of each GRID_SIZE grid frame at that rate (see
    brightest_levels), taken from the same decode; None when nothing was
    decoded here (the cache already held the scan) or the scan is partial.
    """
    vpath = Path(video_path).resolve()
    if not vpath.exists():
        raise MediaNotFoundError(f"Video file not found: {video_path}", {"path": str(vpath)})
    if cache is None:
        cache = MediaCache()
    fingerprint = cache.compute_fingerprint(str(vpath))
    span = _range_key(start_sec, duration_sec)
    timeout = timeout_sec or (Config.SUBPROCESS_TIMEOUT_SEC * 2)
    changes_key = _changes_key(span)
    strong = cache.load_json(fingerprint, f"scenescores_v1_{span}")
    changes = cache.load_json(fingerprint, changes_key)
    levels_out: Optional[list] = None
    if strong is None or changes is None:
        whole = float(start_sec) == 0.0 and duration_sec is None
        levels_out = [] if grid_rate is not None and whole else None
        strong, changes = _scan_scene_changes(str(vpath), start_sec, duration_sec, timeout,
                                              grid_rate=grid_rate if levels_out is not None else None,
                                              levels_out=levels_out)
        cache.save_json(fingerprint, f"scenescores_v1_{span}", strong)
        cache.save_json(fingerprint, changes_key, changes)
    cut_times: List[float] = []
    for threshold in thresholds:
        # detect_scenes hands ffmpeg the threshold with three decimals.
        cutoff = float(f"{threshold:.3f}")
        if cutoff < SCAN_SCORE_FLOOR:
            continue
        key = _scenes_key(threshold, span)
        cached = cache.load_json(fingerprint, key)
        if cached is None:
            raw = [SceneCut(timestamp_sec=t, score=s, timecode=format_timestamp(t)) for t, s in strong if s > cutoff]
            cached = [c.to_dict() for c in _debounce(raw, 0.5, _frame_gap_sec(raw, str(vpath), cache))]
            cache.save_json(fingerprint, key, cached)
        cut_times.extend(item["timestamp_sec"] for item in cached)
    skip = _echoes_of(cut_times if defer_echoes else [])
    if _settle_changes(str(vpath), changes, start_sec, duration_sec, timeout, skip):
        cache.save_json(fingerprint, changes_key, changes)
    return levels_out[0] if levels_out else None


def detect_soft_transitions(
    video_path: str,
    hard_cuts: List[SceneCut],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    cache: Optional[MediaCache] = None,
    use_cache: bool = True,
    timeout_sec: Optional[float] = None,
) -> List[SceneCut]:
    """Find the scene changes that the scene-score filter misses.

    A scene change makes the picture half a second before differ from the
    picture half a second after, and that difference falls back once both
    samples sit in the new scene. Motion inside a scene gives a steady level
    instead, so a peak must stand out from the level around it.

    Most are animated (fades, slides, wipes, zooms) and come back with kind
    "transition", timed at their middle. One that happens in a single frame
    step is a hard cut the filter scored too low: it comes back with kind
    "cut", timed at the stronger of its one or two frames, with that frame's
    scene score.
    """
    vpath = Path(video_path).resolve()
    if cache is None:
        cache = MediaCache()
    fingerprint = cache.compute_fingerprint(str(vpath))
    span = _range_key(start_sec, duration_sec)
    changes_key = _changes_key(span)
    timeout = timeout_sec or (Config.SUBPROCESS_TIMEOUT_SEC * 2)
    changes = cache.load_json(fingerprint, changes_key) if use_cache else None
    if changes is None:
        strong, changes = _scan_scene_changes(str(vpath), start_sec, duration_sec, timeout)
        if use_cache:
            cache.save_json(fingerprint, f"scenescores_v1_{span}", strong)
            cache.save_json(fingerprint, changes_key, changes)
    # A hard cut also raises the lag difference for half a second each side;
    # the plateau it leaves is that cut, not a second change.
    echo = _echoes_of([hard.timestamp_sec for hard in hard_cuts])
    if _settle_changes(str(vpath), changes, start_sec, duration_sec, timeout, echo) and use_cache:
        cache.save_json(fingerprint, changes_key, changes)
    return [SceneCut(timestamp_sec=item["timestamp_sec"], score=item["score"],
                     timecode=format_timestamp(item["timestamp_sec"]), kind=item["kind"])
            for item in changes if not echo(item)]


def partition_into_scene_segments(
    cuts: List[SceneCut],
    total_duration_sec: float
) -> List[SceneSegment]:
    """Convert discrete scene cut timestamps into contiguous SceneSegments."""
    boundaries = [0.0] + [c.timestamp_sec for c in cuts]
    if not boundaries or boundaries[-1] < total_duration_sec:
        boundaries.append(total_duration_sec)
        
    segments: List[SceneSegment] = []
    for i in range(len(boundaries) - 1):
        s_start = boundaries[i]
        s_end = boundaries[i + 1]
        segments.append(SceneSegment(
            scene_id=i + 1,
            start_sec=s_start,
            end_sec=s_end,
            duration_sec=round(s_end - s_start, 3),
            start_timecode=format_timestamp(s_start),
            end_timecode=format_timestamp(s_end)
        ))
    return segments
