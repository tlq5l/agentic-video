"""High-level frame extraction interface using two-stage seeking and caching."""
import os
import time
import uuid
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, DependencyNotFoundError, ExtractionBoundsError
from src.core.types import FrameInfo, format_timestamp
from src.core.cache import MediaCache
from src.media.seeking import build_seek_arguments
from src.media.overlay import apply_overlay_to_file
from src.media.timing import parse_selected_frame_timestamp


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
        
    pre_args, post_args = build_seek_arguments(timestamp_sec, margin_sec=5.0, strategy="two-stage")
    
    unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_file = out_file.parent / f"{out_file.stem}.{unique_suffix}.tmp.jpg"
    
    vf = f"showinfo,scale={width}:-2"
    cmd = ["ffmpeg", "-y", "-copyts"]
    if threads is not None:
        cmd.extend(["-threads", str(threads), "-filter_threads", str(threads)])
    cmd.extend(pre_args)
    cmd.extend(["-i", str(vpath)])
    cmd.extend(post_args)
    cmd.extend(["-frames:v", "1", "-vf", vf, "-q:v", "2", "-update", "1"])
    if threads is not None:
        # Input-side -threads bounds decoding; this bounds the JPEG encoder.
        cmd.extend(["-threads", str(threads)])
    cmd.append(str(tmp_file))
    
    try:
        proc = subprocess.run(
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
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
        raise SubprocessTimeoutError(
            f"FFmpeg timed out extracting frame at {timestamp_sec:.2f}s",
            {"video": str(vpath), "timestamp": timestamp_sec}
        ) from exc

    if proc.returncode != 0 or not tmp_file.exists():
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
        raise MediaCorruptError(
            f"FFmpeg failed extracting frame at {timestamp_sec:.2f}s: {proc.stderr.strip()}",
            {"video": str(vpath), "stderr": proc.stderr}
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
    decoded_timestamp = parse_selected_frame_timestamp(proc.stderr)
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
