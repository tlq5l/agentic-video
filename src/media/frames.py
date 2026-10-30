"""High-level frame extraction interface using two-stage seeking and caching."""
import json
import math
import hashlib
import os
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, DependencyNotFoundError, ExtractionBoundsError
from src.core.types import FrameInfo, format_timestamp
from src.core.cache import MediaCache
from src.media.seeking import build_seek_arguments
from src.media.overlay import apply_overlay_to_file
from src.media.probe import probe_video
from src.media.timing import parse_last_frame_duration, parse_selected_frame_timestamp

# How far before the requested time to look for the frame still on screen when
# a seek lands after the last frame; widen only when the narrower window holds
# no frame (very low frame rates). The frame found must still be on screen at the
# requested time (its start plus its duration reaches it), so a truncated file or
# a longer audio track can't pass off an earlier frame as the requested moment.
LAST_FRAME_LOOKBACKS_SEC = (0.5, 5.0)
# frame_at decodes from this many seconds before the request; a window with no frame
# start in it (one frame lasting longer than the window) is widened, and the last try starts at zero.
FRAME_SHOWN_LOOKBACKS_SEC = (1.0, 10.0)
# frame_at compares each frame's presentation time with the request; this float rounding
# allowance is far below any frame timestamp tick, so a frame after the request never passes.
FRAME_TIME_GUARD_SEC = 1e-9
END_TOLERANCE_SEC = 0.001
# Bump when the frame_at decode or filter chain changes, so saved PNGs made by the old chain are not reused.
FRAME_AT_CACHE_VERSION = 3


def _fps_if_within_media(vpath: Path, timestamp_sec: float, cache: MediaCache) -> Optional[float]:
    """The video's frame rate when the timestamp is inside the media (not past its end), so a
    failed seek there means "after the last frame"; None otherwise."""
    try:
        meta = probe_video(str(vpath), cache=cache)
    except Exception:
        return None
    return meta.fps if timestamp_sec <= meta.duration_sec + END_TOLERANCE_SEC else None


def _shows_at(stderr: str, timestamp_sec: float, fps: float) -> bool:
    """True when the last frame showinfo printed is still on screen at timestamp_sec."""
    start = parse_selected_frame_timestamp(stderr, last=True)
    if start is None:
        return False
    duration = parse_last_frame_duration(stderr) or (1.0 / fps if fps > 0 else 0.0)
    return start + duration >= timestamp_sec - END_TOLERANCE_SEC


def _discard_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def _run_ffmpeg_command(
    cmd: List[str],
    vpath: Path,
    timestamp_sec: float,
    tmp_file: Path,
) -> "subprocess.CompletedProcess[str]":
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=Config.SUBPROCESS_TIMEOUT_SEC,
            check=False
        )
    except FileNotFoundError as exc:
        raise DependencyNotFoundError(
            f"ffmpeg binary not found on system PATH: {exc}",
            {"dependency": "ffmpeg"}
        ) from exc
    except subprocess.TimeoutExpired as exc:
        _discard_file(tmp_file)
        raise SubprocessTimeoutError(
            f"FFmpeg timed out extracting frame at {timestamp_sec:.2f}s",
            {"video": str(vpath), "timestamp": timestamp_sec}
        ) from exc


def _decode_one_frame(
    vpath: Path,
    timestamp_sec: float,
    tmp_file: Path,
    vf: str,
    output_args: List[str],
    cache: MediaCache,
    threads: Optional[int] = None,
) -> Tuple["subprocess.CompletedProcess[str]", bool]:
    """Write the frame shown at timestamp_sec to tmp_file, filtered by vf.

    The output format follows tmp_file's extension. Returns the ffmpeg run and
    whether the seek landed after the last frame, in which case the frame written
    is that last frame (its time is read from the run's showinfo output).
    """
    pre_args, post_args = build_seek_arguments(timestamp_sec, margin_sec=5.0, strategy="two-stage")

    def _run_ffmpeg(seek_args: List[str], one_frame: bool) -> "subprocess.CompletedProcess[str]":
        cmd = ["ffmpeg", "-y", "-copyts"]
        if threads is not None:
            cmd.extend(["-threads", str(threads), "-filter_threads", str(threads)])
        cmd.extend(seek_args)
        cmd.extend(["-i", str(vpath)])
        cmd.extend(post_args)
        if one_frame:
            cmd.extend(["-frames:v", "1"])
        cmd.extend(["-vf", vf, *output_args, "-update", "1"])
        if threads is not None:
            # Input-side -threads bounds decoding; this bounds the encoder.
            cmd.extend(["-threads", str(threads)])
        cmd.append(str(tmp_file))
        return _run_ffmpeg_command(cmd, vpath, timestamp_sec, tmp_file)

    proc = _run_ffmpeg(pre_args, one_frame=True)
    last_frame_fallback = False
    fps = None
    if proc.returncode != 0 or not tmp_file.exists():
        fps = _fps_if_within_media(vpath, timestamp_sec, cache)
    if fps is not None:
        # The seek landed after the last frame's start time (a video ends at its
        # duration, but its last frame starts a frame earlier), so no frame was
        # decoded. Show that last frame; decoded_timestamp/timestamp_delta below
        # report which frame it really is.
        _discard_file(tmp_file)
        for lookback_sec in LAST_FRAME_LOOKBACKS_SEC:
            window_start = max(0.0, timestamp_sec - lookback_sec)
            window = ["-ss", f"{window_start:.3f}", "-t", f"{timestamp_sec - window_start:.3f}"]
            retry = _run_ffmpeg(window, one_frame=False)
            if retry.returncode == 0 and tmp_file.exists():
                if _shows_at(retry.stderr, timestamp_sec, fps):
                    proc = retry
                    last_frame_fallback = True
                else:
                    _discard_file(tmp_file)
                # A frame that ended before the requested time means frames are missing
                # (e.g. a truncated file): fail below rather than relabel it.
                break
            _discard_file(tmp_file)

    if proc.returncode != 0 or not tmp_file.exists():
        _discard_file(tmp_file)
        raise MediaCorruptError(
            f"FFmpeg failed extracting frame at {timestamp_sec:.2f}s: {proc.stderr.strip()}",
            {"video": str(vpath), "stderr": proc.stderr}
        )
    return proc, last_frame_fallback


def _container_start_sec(vpath: Path) -> float:
    """The container's start time: the time zero that FFmpeg's default clock uses, which
    inspect_window and the other sampling tools count from. 0 when the file gives none."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "default=noprint_wrappers=1:nokey=1", str(vpath)],
        capture_output=True, text=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False,
    )
    raw = proc.stdout.strip()
    if proc.returncode == 0 and raw == "N/A":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if proc.returncode != 0 or not math.isfinite(value):
        raise MediaCorruptError(
            f"ffprobe could not read the start time of {vpath}: {proc.stderr.strip()}",
            {"video": str(vpath), "stderr": proc.stderr},
        )
    return value


def _decode_frame_shown_at(
    vpath: Path, time_sec: float, tmp_file: Path, vf: str, origin_sec: float,
) -> float:
    """Write the frame on screen at time_sec to tmp_file, filtered by vf, and return
    that frame's presentation time on the origin_sec clock (the stream-start clock that
    inspect_window counts from).

    That is the last frame whose presentation time is at or before time_sec: a frame
    exactly at time_sec counts, and no later frame is ever written. The decode keeps the
    source clock (-copyts), so each frame's time is exact; -ss counts from the same origin.
    Decoding starts before the request; a window holding no frame is widened. If no
    frame is at or before time_sec, nothing is on screen yet and the request is out of range.
    """
    last_source_sec = origin_sec + time_sec + FRAME_TIME_GUARD_SEC
    starts = (round(max(0.0, time_sec - lookback_sec), 6) for lookback_sec in (*FRAME_SHOWN_LOOKBACKS_SEC, time_sec))
    for start in dict.fromkeys(starts):
        proc = _run_ffmpeg_command(
            [
                "ffmpeg", "-y", "-copyts",
                "-ss", f"{start:.6f}",
                # Input-side -t only bounds how far ffmpeg reads; the select decides the frame.
                "-t", f"{time_sec - start + 1.0:.6f}",
                "-i", str(vpath),
                "-vf", f"select='lte(t,{last_source_sec:.9f})',{vf}",
                "-fps_mode", "passthrough", "-update", "1",
                str(tmp_file),
            ],
            vpath, time_sec, tmp_file,
        )
        if proc.returncode != 0:
            _discard_file(tmp_file)
            raise MediaCorruptError(
                f"FFmpeg failed extracting frame at {time_sec:.2f}s: {proc.stderr.strip()}",
                {"video": str(vpath), "stderr": proc.stderr}
            )
        if tmp_file.exists():
            shown = parse_selected_frame_timestamp(proc.stderr, last=True)
            if shown is None:
                _discard_file(tmp_file)
                raise MediaCorruptError(
                    f"FFmpeg did not report the time of the frame at {time_sec:.2f}s",
                    {"video": str(vpath), "stderr": proc.stderr}
                )
            return round(shown - origin_sec, 6)
    raise ExtractionBoundsError(
        f"no frame is on screen at time_sec {time_sec:g}: the first frame of this video starts later",
        {"video": str(vpath), "time_sec": time_sec},
    )


def extract_single_frame(
    video_path: Union[str, Path],
    timestamp_sec: float,
    output_path: Optional[Union[str, Path]] = None,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    overlay_timestamp: bool = True,
    burn_timestamps: Optional[bool] = None,
    cache: Optional[MediaCache] = None,
    threads: Optional[int] = None,
) -> FrameInfo:
    """Extract a single frame at timestamp_sec using two-stage seeking.
    
    If output_path is not specified, saves into the media cache under `frames/`.
    """
    if burn_timestamps is not None:
        overlay_timestamp = burn_timestamps

    if timestamp_sec < 0:
        raise ExtractionBoundsError(
            f"timestamp_sec must be >= 0 (got {timestamp_sec})",
            {"timestamp_sec": timestamp_sec}
        )

    vpath = Path(video_path).resolve()
    if cache is None:
        cache = MediaCache()
        
    fingerprint = cache.compute_fingerprint(str(vpath))
    ts_ms = int(timestamp_sec * 1000)
    
    if output_path is None:
        frames_dir = cache.get_frames_dir(fingerprint, tag="singles")
        overlay_tag = "ts" if overlay_timestamp else "raw"
        out_file = frames_dir / f"frame_{ts_ms}_{width}w_{overlay_tag}.jpg"
    else:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
    # Check if cached already exists
    if out_file.exists() and out_file.stat().st_size > 0:
        for attempt in range(4):
            try:
                from PIL import Image
                with Image.open(out_file) as im:
                    w, h = im.size
                return FrameInfo(
                    index=0,
                    timestamp_sec=timestamp_sec,
                    timecode=format_timestamp(timestamp_sec),
                    file_path=str(out_file.resolve()),
                    width=w,
                    height=h,
                    requested_timestamp=timestamp_sec,
                )
            except Exception:
                if attempt < 3:
                    time.sleep(0.015 * (attempt + 1))

    unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_file = out_file.parent / f"{out_file.stem}.{unique_suffix}.tmp.jpg"
    proc, last_frame_fallback = _decode_one_frame(
        vpath, timestamp_sec, tmp_file, f"showinfo,scale={width}:-2", ["-q:v", "2"], cache, threads
    )

    if overlay_timestamp:
        apply_overlay_to_file(tmp_file, timestamp_sec=timestamp_sec)
        
    from PIL import Image
    with Image.open(tmp_file) as im:
        w, h = im.size
        
    # Atomically replace tmp_file to out_file
    for attempt in range(5):
        try:
            os.replace(str(tmp_file), str(out_file))
            break
        except (OSError, IOError):
            if attempt < 4:
                time.sleep(0.015 * (attempt + 1))
            else:
                try:
                    import shutil
                    shutil.move(str(tmp_file), str(out_file))
                except Exception:
                    pass
        
    # ``showinfo`` was part of the exact FFmpeg command that wrote this
    # artifact, so its PTS identifies the selected frame rather than merely a
    # nearby source frame returned by a separate probe.
    decoded_timestamp = parse_selected_frame_timestamp(proc.stderr, last=last_frame_fallback)
    timestamp_delta = (
        round(decoded_timestamp - float(timestamp_sec), 6)
        if decoded_timestamp is not None
        else None
    )

    return FrameInfo(
        index=0,
        timestamp_sec=timestamp_sec,
        timecode=format_timestamp(timestamp_sec),
        file_path=str(out_file.resolve()),
        width=w,
        height=h,
        requested_timestamp=timestamp_sec,
        decoded_timestamp=decoded_timestamp,
        timestamp_delta=timestamp_delta,
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _whole_number(value: Any) -> Optional[int]:
    """The value as an int when it is a whole number (2 or 2.0), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if isinstance(value, int) else None


def _crop_box(crop: Any, frame_width: int, frame_height: int) -> Optional[Dict[str, int]]:
    if crop is None:
        return None
    minima = {"x": 0, "y": 0, "width": 1, "height": 1}
    if not isinstance(crop, dict) or set(crop) != minima.keys():
        raise ExtractionBoundsError(
            'crop must be an object with exactly x, y, width and height in source pixels, '
            'e.g. {"x": 100, "y": 50, "width": 200, "height": 100}',
            {"crop": crop},
        )
    box: Dict[str, int] = {}
    for key, minimum in minima.items():
        value = _whole_number(crop[key])
        if value is None or value < minimum:
            raise ExtractionBoundsError(
                f"crop.{key} must be a whole number >= {minimum} in source pixels (got {crop[key]!r})",
                {"crop": crop},
            )
        box[key] = value
    if box["x"] + box["width"] > frame_width:
        raise ExtractionBoundsError(
            f"crop.x + crop.width = {box['x'] + box['width']} exceeds the frame width {frame_width}; "
            f"use crop.x + crop.width <= {frame_width}",
            {"crop": box, "frame_width": frame_width},
        )
    if box["y"] + box["height"] > frame_height:
        raise ExtractionBoundsError(
            f"crop.y + crop.height = {box['y'] + box['height']} exceeds the frame height {frame_height}; "
            f"use crop.y + crop.height <= {frame_height}",
            {"crop": box, "frame_height": frame_height},
        )
    return box


def _has_no_video_stream(vpath: Path) -> bool:
    """True only when ffprobe runs and lists no video stream, so a failed probe of
    some other kind is not blamed on the file type."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(vpath)],
            capture_output=True, text=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and not proc.stdout.strip()


def extract_frame_at(
    video_path: Union[str, Path],
    time_sec: float,
    crop: Optional[Dict[str, Any]] = None,
    scale: Optional[float] = None,
    cache: Optional[MediaCache] = None,
) -> Dict[str, Any]:
    """One frame at its native resolution, optionally cropped (source pixels) and
    upscaled with Lanczos, saved as a PNG under the cache's `frame_at` folder.

    The same source, time, crop and scale reuse the saved file. Returns the
    decoded frame's own time (frame_time_sec) and the size of the saved image.
    """
    if cache is None:
        cache = MediaCache()
    vpath = Path(video_path).resolve()
    try:
        meta = probe_video(str(vpath), cache=cache)
    except MediaCorruptError as exc:
        if _has_no_video_stream(vpath):
            raise MediaCorruptError(
                f"frame_at needs a file with a video stream, but source {vpath} has none (it is audio only); "
                "pass a video file as source",
                {"source": str(vpath)},
            ) from exc
        raise

    if not _is_finite_number(time_sec) or time_sec < 0:
        raise ExtractionBoundsError(f"time_sec must be a number of seconds >= 0 (got {time_sec!r})", {"time_sec": time_sec})
    if time_sec > meta.duration_sec:
        raise ExtractionBoundsError(
            f"time_sec {time_sec:g} is outside the valid range [0, {meta.duration_sec:.3f}] s",
            {"time_sec": time_sec, "duration_sec": meta.duration_sec},
        )
    box = _crop_box(crop, meta.width, meta.height)
    factor = 1.0 if scale is None else scale
    if not _is_finite_number(factor) or not 1.0 <= factor <= 4.0:
        raise ExtractionBoundsError(
            f"scale must be a number from 1 to 4 (got {scale!r}); omit it for the frame as it is",
            {"scale": scale},
        )
    factor = float(factor)

    source_w, source_h = (box["width"], box["height"]) if box else (meta.width, meta.height)
    out_w, out_h = round(source_w * factor), round(source_h * factor)
    # Convert to RGB before cropping: on the subsampled YUV frame an odd x or y shifts
    # the chroma grid and changes the colours of the crop.
    filters = ["showinfo", "format=rgb24"]
    if box:
        filters.append(f"crop={box['width']}:{box['height']}:{box['x']}:{box['y']}")
    if factor != 1.0:
        filters.append(f"scale={out_w}:{out_h}:flags=lanczos")

    # The exact requested time is the key: a rounded time would let two requests on
    # either side of a frame boundary share one saved frame, so the result would depend on call order.
    key = hashlib.sha256(json.dumps(
        {"version": FRAME_AT_CACHE_VERSION, "time_sec": float(time_sec), "crop": box, "scale": factor},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:16]
    frames_dir = cache.get_frames_dir(meta.fingerprint, tag="frame_at")
    out_file = frames_dir / f"frame_at_{key}.png"
    sidecar = out_file.with_suffix(".json")

    if out_file.exists() and sidecar.exists():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return {**cached, "file_path": str(out_file)}
        except (OSError, ValueError):
            pass

    tmp_file = frames_dir / f"frame_at_{key}.{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp.png"
    decoded = _decode_frame_shown_at(
        vpath, float(time_sec), tmp_file, ",".join(filters), _container_start_sec(vpath),
    )
    from PIL import Image
    with Image.open(tmp_file) as im:
        width, height = im.size
    result = {
        "frame_time_sec": decoded,
        "width": width,
        "height": height,
        "source_width": meta.width,
        "source_height": meta.height,
        "crop": box,
        "scale": factor,
    }
    os.replace(str(tmp_file), str(out_file))
    sidecar_tmp = sidecar.with_suffix(f".{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp")
    sidecar_tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(str(sidecar_tmp), str(sidecar))
    return {**result, "file_path": str(out_file)}


def extract_window_frames(
    video_path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    fps: float = Config.DEFAULT_DENSE_FPS,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    overlay_timestamp: bool = True,
    burn_timestamps: Optional[bool] = None,
    cache: Optional[MediaCache] = None
) -> List[FrameInfo]:
    """Extract all frames in a temporal window [start_sec, end_sec] at specified FPS."""
    if burn_timestamps is not None:
        overlay_timestamp = burn_timestamps
        
    from src.media.sampling import sample_dense_window
    window_res = sample_dense_window(
        video_path=str(video_path),
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        width=width,
        overlay_timestamp=overlay_timestamp,
        cache=cache
    )
    return window_res.frames
