"""Bounded, asynchronous video repetition counting.

Cycles are excursions that leave and return to an observed rest/turning phase.
An excursion touching either window edge is deliberately excluded.
"""
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from PIL import Image

from src.core.config import Config
from src.core.errors import as_dependency_error
from src.core.types import FrameInfo, format_timestamp
from src.media.contact_sheet import compose_contact_sheet
from src.retrieval import periodicity

ALGORITHM_VERSION = "repcount-v5"
# Real footage has cycles shorter than 0.4 s and clips under 3 s, so short windows are sampled at the fps cap.
MAX_FPS = 15.0
MAX_ANALYSIS_FRAMES = 900
# Minimum decode timeout; long windows get more, since ffmpeg decodes every source frame.
DECODE_TIMEOUT_SEC = Config.SUBPROCESS_TIMEOUT_SEC
DECODE_TIMEOUT_PER_WINDOW_SEC = 2.0
ORPHAN_TEMP_AGE_SEC = 300
NOTE = "counts are estimates; verify ambiguous cycles with inspect_window"
# Confidence is capped at this when the periodic estimate overrides an amplitude count it disagrees with.
DISAGREE_CONFIDENCE = 0.3
NO_PERIODIC_NOTE = "no periodic motion was found; the action may be absent, too faint or not repeating"
_jobs = OrderedDict()
_jobs_lock = threading.Lock()
_worker = None
_worker_key = None


def _smooth(values, fps, min_period_sec):
    size = 1 if fps * min_period_sec < 8 else max(3, int(round(fps * 0.12)) | 1)
    return np.convolve(values, np.ones(size) / size, mode="same")


def count_signal(signal, fps, start_sec=0.0, min_period_sec=0.4, max_period_sec=3.0,
                 require_coherence=True):
    """Count complete above-baseline lobes, including variable-tempo lobes and pauses."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not math.isfinite(fps) or fps <= 0 or min_period_sec <= 0 or max_period_sec <= min_period_sec:
        raise ValueError("Invalid fps or period bounds")
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Signal must contain at least three finite samples")
    raw_start, raw_end = float(x[0]), float(x[-1])
    raw_floor = float(np.percentile(x, 1))
    x = _smooth(x, fps, min_period_sec)
    baseline = float(np.percentile(x, 10))
    amplitude = float(np.percentile(x, 95) - baseline)
    empty = {"count": 0, "cycles": [], "period_sec": None, "median_period_sec": None, "regularity": 0.0,
             "confidence": 0.0, "signal_quality": "low_activity"}
    if amplitude < 0.012:
        return empty
    y = np.maximum(x - baseline, 0)
    threshold = max(amplitude * 0.15, 0.01)
    active = y > threshold
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    raw = []
    for left, right in zip(starts, ends):
        # A low sample adjacent to the edge does not prove the lobe completed:
        # threshold crossings precede the true return to baseline.
        edge_margin = max(1, int(math.ceil((right - left + 1) * 0.25)))
        touches_start = left - edge_margin <= 0 and raw_start - raw_floor > amplitude * 0.01
        touches_end = right + edge_margin >= len(x) - 1 and raw_end - raw_floor > amplitude * 0.01
        if touches_start or touches_end:
            continue
        peak = left + int(np.argmax(y[left:right + 1]))
        height = y[peak]
        width = (right - left + 1) / fps
        if height < amplitude * 0.32 or width < max(1 / fps, min_period_sec * 0.2) or width > max_period_sec * 1.3:
            continue
        # Threshold crossing is inside the true cycle. Use the adjacent low
        # sample as an observable boundary; uncertainty is one frame or more.
        raw.append((left - 1, right + 1, peak, height))
    if not raw:
        return {**empty, "signal_quality": "irregular"}
    # Repeated lobes should have a consistent shape and plausible peak spacing.
    heights = np.array([r[3] for r in raw])
    peaks = np.array([r[2] for r in raw]) / fps
    gaps = np.diff(peaks)
    plausible = gaps[(gaps >= min_period_sec * 0.75) & (gaps <= max_period_sec * 1.25)]
    if len(raw) >= 3 and len(plausible) == 0:
        return {**empty, "signal_quality": "irregular"}
    period = float(np.median(plausible)) if len(plausible) else float(np.median([(r[1]-r[0])/fps/0.7 for r in raw]))
    if require_coherence and len(raw) >= 3:
        lag = int(round(period * fps))
        slack = max(2, int(round(lag * .15)))
        correlations = []
        for offset in range(-slack, slack + 1):
            shift = lag + offset
            if 2 <= shift < len(x) // 2:
                left, right = x[:-shift], x[shift:]
                if np.std(left) > 1e-9 and np.std(right) > 1e-9:
                    correlations.append(float(np.corrcoef(left, right)[0, 1]))
        if correlations and max(correlations) < .08:
            return empty
    if len(plausible):
        spacing = float(np.median(np.abs(plausible - np.median(plausible))) / max(np.median(plausible), 1e-9))
        continuity = len(plausible) / max(len(gaps), 1)
    else:
        spacing, continuity = 0.4, 0.35 if len(raw) == 1 else 0.0
    shape = max(0.0, 1.0 - float(np.std(heights) / max(np.mean(heights), 1e-9)))
    regularity = float(np.clip((1 - spacing) * 0.55 + continuity * 0.3 + shape * 0.15, 0, 1))
    cycles = [{"index": i + 1, "start_sec": round(start_sec + a / fps, 3),
               "end_sec": round(start_sec + b / fps, 3),
               "peak_sec": round(start_sec + p / fps, 3)}
              for i, (a, b, p, _) in enumerate(raw)]
    return {"count": len(cycles), "cycles": cycles, "period_sec": round(period, 3),
            "median_period_sec": round(period, 3),
            "regularity": round(regularity, 3), "confidence": round(regularity, 3),
            "signal_quality": "good" if regularity >= 0.5 else "irregular"}


def _principal_phase(matrix):
    centered = matrix - matrix.mean(axis=0)
    row = int(np.argmax(np.sum(centered * centered, axis=1)))
    vector = centered[row].copy()
    for _ in range(8):
        vector = centered.T @ (centered @ vector)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return centered @ vector


def _count_phase(phase, fps, start_sec, min_period_sec, max_period_sec,
                 require_coherence=True):
    edge = max(1, min(len(phase) // 10, int(fps * 0.5)))
    rest = float(np.median(np.r_[phase[:edge], phase[-edge:]]))
    candidates = [count_signal(np.maximum(sign * (phase - rest), 0), fps, start_sec,
                               min_period_sec, max_period_sec, require_coherence) for sign in (1, -1)]
    # One polarity can split a full swing into two half cycles. Prefer the
    # candidate whose peak spacing is longer when both are regular.
    return max(candidates, key=lambda r: (r["regularity"] >= 0.5,
                                          r["count"] > 0,
                                          r["period_sec"] or 0,
                                          r["regularity"]))


def count_embeddings(embeddings, fps, start_sec=0.0, min_period_sec=0.4, max_period_sec=3.0):
    """Use embedding self-similarity and its dominant phase trajectory."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be a finite frame-by-feature matrix")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = matrix @ matrix.T
    n = len(matrix)
    edge = max(1, min(n // 10, int(fps * 0.5)))
    candidates = np.r_[np.arange(edge), np.arange(n-edge, n)]
    anchor = int(candidates[np.argmax(similarity[candidates].mean(axis=1))])
    phase = _principal_phase(matrix)
    result = _count_phase(phase, fps, start_sec, min_period_sec, max_period_sec)
    if result["count"] and len(result["cycles"]) > 1:
        lag = int(round((result["period_sec"] or min_period_sec) * fps))
        if lag < n:
            result["self_similarity_score"] = round(float(np.mean(np.diag(similarity, k=lag))), 3)
    result["similarity_anchor_sec"] = round(start_sec + anchor / fps, 3)
    return result


def _motion_phase(frames, energy, fps, start_sec, min_period_sec, max_period_sec):
    # Keep memory bounded by downsampling every decoded frame to 32x18.
    pixels = np.stack([frame[::4, ::4].astype(np.float32).ravel() / 255 for frame in frames])
    phase = _principal_phase(pixels)
    if float(np.ptp(phase)) < 2.0 or float(np.percentile(energy, 95) - np.percentile(energy, 10)) < 0.002:
        return count_signal(np.zeros(len(frames)), fps, start_sec, min_period_sec, max_period_sec)
    # Spatial contrast and motion-energy gates already reject the weak random
    # fluctuations for which the one-dimensional coherence gate is intended.
    return _count_phase(phase, fps, start_sec, min_period_sec, max_period_sec,
                        require_coherence=False)


def _focus_frames(frames, region):
    if region is None:
        return frames
    height, width = frames[0].shape
    x0 = int(region["x"] * width); y0 = int(region["y"] * height)
    x1 = max(x0+1, int((region["x"]+region["w"]) * width))
    y1 = max(y0+1, int((region["y"]+region["h"]) * height))
    return [frame[y0:y1, x0:x1] for frame in frames]


def _bidirectional_motion(frames, fps):
    """Detect a foreground centroid moving to both sides of its rest pose."""
    centers = []
    for frame in frames:
        mask = frame > max(90, float(np.median(frame)) + 50)
        if mask.sum() < 10:
            centers.append((np.nan, np.nan))
            continue
        y, x = np.nonzero(mask)
        centers.append((float(np.mean(x)), float(np.mean(y))))
    centers = np.asarray(centers)
    edge = max(1, min(len(frames)//10, int(fps*.5)))
    rest = np.nanmedian(np.r_[centers[:edge], centers[-edge:]], axis=0)
    for axis in range(2):
        values = centers[:, axis] - rest[axis]
        lo, hi = np.nanpercentile(values, [5, 95])
        if lo < -8 and hi > 8 and min(abs(lo), hi) >= .3 * max(abs(lo), hi):
            return True
    return False


def _exclude_active_tail(result, energy, end_sec, fps, visual=False):
    """Exclude a short trailing lobe when the window ends during active motion."""
    if not result["cycles"] or len(energy) < 3:
        return result
    period = result["period_sec"]
    if not period:
        return result
    tail_energy = float(np.median(energy[-2:]))
    usual_energy = float(np.percentile(energy, 95))
    last_cycle = result["cycles"][-1]
    span = last_cycle["end_sec"] - last_cycle["start_sec"]
    distance = end_sec - last_cycle["end_sec"]
    quiet = max(.004, usual_energy * .35)
    # The window must end close to the lobe itself, not partway into a new rep.
    if tail_energy <= quiet or not 0 <= distance <= min(span, period * .4):
        return result
    # Excursion spans are shorter than full periods, so compare like measurements.
    peer_spans = [cycle["end_sec"] - cycle["start_sec"]
                  for cycle in result["cycles"][:-1]]
    minimum_span = (.85 * float(np.median(peer_spans)) if peer_spans
                    else .6 * period)
    # A visual phase lobe can finish before the depicted action returns to rest.
    # At an active crop edge, require enough frames after its peak to see that return.
    visual_incomplete = visual and end_sec - last_cycle["peak_sec"] < .7 * period
    if span >= minimum_span and not visual_incomplete:
        return result
    trimmed = dict(result)
    trimmed["cycles"] = result["cycles"][:-1]
    trimmed["count"] = len(trimmed["cycles"])
    trimmed["confidence"] = round(max(0, result["confidence"] - .15), 3)
    trimmed["signal_quality"] = "edge_ambiguous"
    if not trimmed["cycles"]:
        trimmed["period_sec"] = trimmed["median_period_sec"] = None
    return trimmed


def _validate(start_sec, end_sec, fps, method, region, min_period_sec, max_period_sec):
    if method not in ("auto", "motion", "embedding"):
        raise ValueError(f"method must be auto, motion, or embedding (got {method!r}); omit it to use auto")
    if not all(math.isfinite(float(x)) for x in (start_sec, end_sec, fps, min_period_sec, max_period_sec)):
        raise ValueError("start_sec, end_sec, fps, min_period_sec and max_period_sec must be finite numbers")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"Need 0 <= start_sec < end_sec (got start_sec={start_sec:g}, end_sec={end_sec:g}); omit end_sec to count to the end of the video")
    if end_sec - start_sec > 180:
        raise ValueError(f"Repetition window is {end_sec - start_sec:g}s but must be within 180 seconds; split longer clips into windows "
                         "(pass start_sec/end_sec around the action, found with search_visual or contact_sheet). "
                         "A cycle that crosses a window edge can be lost, so cut windows at pauses between cycles")
    if fps <= 0 or fps > 15 or min_period_sec <= 0 or max_period_sec <= min_period_sec:
        raise ValueError(f"fps must be in (0,15] and min_period_sec must be positive and below max_period_sec "
                         f"(got fps={fps:g}, min_period_sec={min_period_sec:g}, max_period_sec={max_period_sec:g})")
    if region is not None:
        if not isinstance(region, dict) or set(region) != {"x", "y", "w", "h"}:
            raise ValueError("region requires exactly the keys x, y, w, h as fractions of the frame (w and h, not width/height), "
                             "e.g. {\"x\": 0.25, \"y\": 0.1, \"w\": 0.5, \"h\": 0.6}; omit region to use the whole frame")
        x, y, w, h = (float(region[k]) for k in ("x", "y", "w", "h"))
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1 or y+h > 1:
            raise ValueError(f"region must lie within the frame: x, y >= 0, w, h > 0, x+w <= 1 and y+h <= 1 (got x={x:g}, y={y:g}, w={w:g}, h={h:g})")


def _decode_frames(source, start, end, fps, region):
    """Read scaled grayscale rawvideo one frame at a time from ffmpeg stdout."""
    width, height = 128, 72
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-filter_threads", "2",
           "-ss", str(start), "-i", str(source),
           "-t", str(end-start), "-vf", f"fps={fps},scale={width}:{height},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-threads", "2", "pipe:1"]
    frames = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    error_tail = bytearray()

    def drain_stderr():
        try:
            while chunk := os.read(proc.stderr.fileno(), 4096):
                error_tail.extend(chunk)
                if len(error_tail) > 500:
                    del error_tail[:-500]
        except OSError:
            pass  # The process or pipe may close during timeout cleanup.

    stderr_reader = threading.Thread(target=drain_stderr, name="repetition-ffmpeg-stderr", daemon=True)
    stderr_reader.start()
    expired = threading.Event()

    def stop_on_timeout():
        if proc.poll() is None:
            expired.set()
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    timeout_sec = max(DECODE_TIMEOUT_SEC, DECODE_TIMEOUT_PER_WINDOW_SEC * (end - start))
    timer = threading.Timer(timeout_sec, stop_on_timeout)
    timer.start()
    try:
        while len(frames) <= int(math.ceil((end-start)*fps)) + 2:
            block = proc.stdout.read(width * height)
            if expired.is_set():
                raise TimeoutError(f"ffmpeg decode timed out after {timeout_sec:g}s")
            if not block:
                break
            if len(block) != width * height:
                raise RuntimeError("Truncated video frame")
            frames.append(np.frombuffer(block, dtype=np.uint8).reshape(height, width).copy())
        code = proc.wait()
        if expired.is_set():
            raise TimeoutError(f"ffmpeg decode timed out after {timeout_sec:g}s")
        stderr_reader.join(timeout=1)
        if code:
            raise RuntimeError(f"ffmpeg decode failed: {error_tail.decode(errors='replace')}")
    finally:
        timer.cancel()
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stderr_reader.join(timeout=1)
        proc.stderr.close()
    if len(frames) < 3:
        raise ValueError("Window contains fewer than three decoded frames; use a longer window or a higher fps (at most 15)")
    if region:
        x0 = int(region["x"] * width); y0 = int(region["y"] * height)
        x1 = max(x0+1, int((region["x"]+region["w"]) * width))
        y1 = max(y0+1, int((region["y"]+region["h"]) * height))
        cropped = [frame[y0:y1, x0:x1] for frame in frames]
    else:
        cropped = frames
    energy = np.zeros(len(frames), dtype=np.float64)
    for i in range(1, len(frames)):
        energy[i] = np.abs(cropped[i].astype(np.int16) - cropped[i-1].astype(np.int16)).mean() / 255
    return frames, energy


def _source_stamp(source):
    stat = Path(source).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _embed_frames(frames, temp_dir):
    from src.retrieval.visual import get_encoder
    encoder = get_encoder()
    vectors = []
    for start in range(0, len(frames), 4):
        paths = []
        for j, frame in enumerate(frames[start:start+4]):
            path = Path(temp_dir) / f"embedding_{start+j}.jpg"
            Image.fromarray(frame).convert("RGB").resize((224, 224)).save(path)
            paths.append(path)
        vectors.extend(encoder.images(paths))
        for path in paths:
            path.unlink()
    return np.asarray(vectors)


@contextmanager
def _locked_temp_dir(parent):
    temp = tempfile.TemporaryDirectory(prefix="repcount-", dir=parent)
    lock_file = None
    try:
        lock_file = open(Path(temp.name) / ".lock", "a+b")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield temp.name
    finally:
        try:
            temp.cleanup()
        finally:
            if lock_file is not None:
                lock_file.close()


def _remove_orphaned_temp_dirs(parent):
    cutoff = time.time() - ORPHAN_TEMP_AGE_SEC
    for path in parent.iterdir():
        if not path.name.startswith("repcount-"):
            continue
        try:
            if not path.is_symlink() and path.is_dir() and path.stat().st_mtime < cutoff:
                with open(path / ".lock", "a+b") as lock_file:
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    shutil.rmtree(path)
        except FileNotFoundError:
            pass


def _effective_fps(fps, duration_sec):
    """Sample short windows at the fps cap: fast cycles and 2 s clips need it, and the frame count stays bounded."""
    return min(MAX_FPS, max(fps, MAX_ANALYSIS_FRAMES / duration_sec))


def _edges_active(energy):
    """True when the window starts or ends mid-motion, so edge cycles are real cycles."""
    quiet = max(.004, float(np.percentile(energy, 95)) * .35)
    return bool(len(energy) > 6 and (np.median(energy[1:4]) > quiet or np.median(energy[-3:]) > quiet))


def _agrees(result, found, energy):
    """True when an amplitude count matches the autocorrelation period and cycle count.

    The amplitude counter drops cycles that touch a window edge. That is right for a
    clip with idle margins but loses a rep when the clip is cut mid-motion, so there
    the counts must match exactly.
    """
    if not found or not found["count"] or not result["count"] or not result["period_sec"]:
        return False
    slack = 0 if _edges_active(energy) else max(1, round(0.1 * found["count"]))
    return (abs(result["count"] - found["count"]) <= slack
            and abs(result["period_sec"] / found["period_sec"] - 1) <= 0.25)


def _note(chosen, found):
    if chosen["count"]:
        return NOTE
    if not found or not found["count"]:
        return f"{NOTE}; {NO_PERIODIC_NOTE}"
    return (f"{NOTE}; count is 0 but the periodicity check found {found['count']} cycles "
            f"about {found['period_sec']:g} s apart, so motion is repeating: count by eye")


def _periodic_result(found):
    strength = round(min(1.0, found["periodicity"]), 3)
    return {"count": found["count"], "cycles": found["cycles"], "period_sec": found["period_sec"],
            "median_period_sec": found["period_sec"], "regularity": strength,
            "confidence": min(strength, DISAGREE_CONFIDENCE), "signal_quality": "periodic"}


def _override_note(found, amplitude, motion, embedding):
    """Say that the two methods disagree, with both counts, so the agent does not read one as settled."""
    seen = f"motion {motion['count']}" + (f", embedding {embedding['count']}" if embedding else "")
    return (f"{NOTE}; the methods disagree: the periodicity check counted {found['count']} cycles "
            f"about {found['period_sec']:g} s apart and replaced the amplitude count of {amplitude['count']} "
            f"({seen}), so confidence is capped at {DISAGREE_CONFIDENCE:g}. Either count may be right: "
            f"count by eye before answering")


def count_video(engine, start_sec=0.0, end_sec=None, region=None, fps=10.0, method="auto",
                min_period_sec=0.4, max_period_sec=3.0, action_hint=None, temp_root=None):
    source_stamp = _source_stamp(engine.source_path)
    end = engine.metadata.duration_sec if end_sec is None else float(end_sec)
    _validate(float(start_sec), end, float(fps), method, region, float(min_period_sec), float(max_period_sec))
    if end > engine.metadata.duration_sec + 0.001:
        raise ValueError(f"Repetition window ends at {end:g}s but the video is {engine.metadata.duration_sec:g}s long; use end_sec <= {engine.metadata.duration_sec:g} or omit it")
    fingerprint = engine.metadata.fingerprint
    if engine._cache.compute_fingerprint(engine.source_path) != fingerprint:
        raise RuntimeError("Source changed since video probing; create a new engine and retry")
    args = {"start_sec": start_sec, "end_sec": end, "region": region, "fps": fps, "method": method,
            "min_period_sec": min_period_sec, "max_period_sec": max_period_sec, "action_hint": action_hint}
    tag = hashlib.sha256(json.dumps([ALGORITHM_VERSION, fingerprint, args], sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = engine._cache.get_entry_dir(fingerprint)
    temp_parent = Path(temp_root) if temp_root is not None else cache_dir
    _remove_orphaned_temp_dirs(temp_parent)
    result_path = cache_dir / f"{tag}.repcount.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        sheet = result.get("contact_sheet")
        if not sheet or Path(sheet["file_path"]).exists():
            return result
    with _locked_temp_dir(temp_parent) as temp_dir:
        fps = float(fps)
        frames, energy = _decode_frames(engine.source_path, float(start_sec), end, fps, region)
        focused = _focus_frames(frames, region)
        fast_fps = _effective_fps(fps, end - float(start_sec))
        if fast_fps > fps:
            fast_frames, _ = _decode_frames(engine.source_path, float(start_sec), end, fast_fps, region)
            found = periodicity.estimate(_focus_frames(fast_frames, region), fast_fps, float(start_sec))
            del fast_frames
        else:
            found = periodicity.estimate(focused, fps, float(start_sec))
        motion = _motion_phase(focused, energy, fps, float(start_sec),
                               float(min_period_sec), float(max_period_sec))
        motion = _exclude_active_tail(motion, energy, end, fps, visual=True)
        embedding = None
        skip_embedding = method == "auto" and _agrees(motion, found, energy)
        if method == "embedding" or (method == "auto" and end - float(start_sec) <= 30 and not skip_embedding):
            try:
                vectors = _embed_frames(focused, temp_dir)
                embedding = count_embeddings(vectors, fps, float(start_sec), float(min_period_sec), float(max_period_sec))
                if float(np.percentile(energy, 95) - np.percentile(energy, 10)) < 0.002:
                    embedding = count_signal(np.zeros(len(frames)), fps, float(start_sec),
                                             float(min_period_sec), float(max_period_sec))
                embedding = _exclude_active_tail(embedding, energy, end, fps, visual=True)
            except RuntimeError:
                if method == "embedding":
                    raise
        choose_embedding = method == "embedding"
        if method == "auto" and embedding:
            if not motion["count"] and embedding["count"] >= 2:
                choose_embedding = embedding["regularity"] >= 0.75
            elif motion["count"] and embedding["count"]:
                ratio = motion["count"] / embedding["count"]
                if 1.7 <= ratio <= 2.3 and _bidirectional_motion(focused, fps):
                    choose_embedding = True
                elif abs(motion["count"] - embedding["count"]) <= 1:
                    choose_embedding = embedding["regularity"] > motion["regularity"] + 0.05
        chosen = embedding if choose_embedding else motion
        used = "embedding" if chosen is embedding else "motion"
        note = None
        if method == "auto" and found and found["count"] and not _agrees(chosen, found, energy):
            note = _override_note(found, chosen, motion, embedding)
            chosen, used = _periodic_result(found), "periodic"
        result = {"status": "complete", **chosen, "method_used": used,
                  "motion_count": motion["count"], "embedding_count": embedding["count"] if embedding else None,
                  "periodic_count": found["count"] if found else None,
                  "note": note or _note(chosen, found), "contact_sheet": None}
        result["cycles"] = [{**cycle, "frame_badge": f"#{cycle['index']}"}
                            for cycle in chosen["cycles"]]
        if chosen["cycles"]:
            evidence = []
            for cycle in chosen["cycles"]:
                index = min(len(frames)-1, max(0, round((cycle["peak_sec"]-float(start_sec))*float(fps))))
                path = Path(temp_dir) / f"peak_{cycle['index']}.jpg"
                Image.fromarray(frames[index]).save(path)
                evidence.append(FrameInfo(cycle["index"], cycle["peak_sec"], format_timestamp(cycle["peak_sec"]),
                                          str(path), 128, 72))
            staged_sheet = Path(temp_dir) / "peaks.jpg"
            result["contact_sheet"] = compose_contact_sheet(evidence, staged_sheet,
                                                               title="REPETITION PEAKS").to_dict()
        if _source_stamp(engine.source_path) != source_stamp:
            raise RuntimeError("Source changed during repetition counting; retry with the current video")
        if chosen["cycles"]:
            sheet_path = engine._cache.get_sheets_dir(fingerprint) / f"repetitions_{tag}.jpg"
            staged_sheet.replace(sheet_path)
            result["contact_sheet"]["file_path"] = str(sheet_path)
        staged = Path(temp_dir) / "result.json"
        staged.write_text(json.dumps(result))
        staged.replace(result_path)
        return result


def submit(engine, **args):
    """Start or poll a deduplicated worker; no video probing occurs here."""
    global _worker, _worker_key
    start = float(args.get("start_sec", 0.0))
    end = args.get("end_sec")
    fps = float(args.get("fps", 10.0))
    # The duration bound is checked before scheduling when caller supplies end.
    if end is not None:
        _validate(start, float(end), fps, args.get("method", "auto"), args.get("region"),
                  float(args.get("min_period_sec", 0.4)), float(args.get("max_period_sec", 3.0)))
    source = Path(engine.source_path).resolve()
    stat = source.stat()
    key = (ALGORITHM_VERSION, str(engine._cache.base_dir.resolve()), str(source),
           stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
           json.dumps(args, sort_keys=True))
    with _jobs_lock:
        if key in _jobs:
            result = dict(_jobs[key])
            if result["status"] == "in_progress" and (
                _worker_key != key or _worker is None or not _worker.is_alive()
            ):
                result = {"status": "error", "message": "Repetition count worker stopped unexpectedly; retry."}
                _jobs[key] = result
            if result["status"] == "error":
                del _jobs[key]
            return result
        if _worker is not None and _worker.is_alive():
            return {"status": "in_progress", "retry_after_seconds": 2,
                    "message": "Another repetition count is active; retry shortly."}
        while len(_jobs) >= 32:
            _jobs.popitem(last=False)
        _jobs[key] = {"status": "in_progress", "retry_after_seconds": 2}
        def work():
            try:
                result = count_video(engine, **args)
            except Exception as exc:
                result = {"status": "error", "message": str(as_dependency_error(exc))}
            with _jobs_lock:
                _jobs[key] = result
        _worker = threading.Thread(target=work, name="repetition-count", daemon=True)
        _worker_key = key
        _worker.start()
        return {"status": "in_progress", "retry_after_seconds": 2}
