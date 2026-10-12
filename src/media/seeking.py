"""Seeking algorithms and frame-exact extraction engine."""
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, DependencyNotFoundError, ExtractionBoundsError
from src.core.types import FrameInfo, format_timestamp
from src.media.timing import parse_selected_frame_timestamp


def build_seek_arguments(
    target_sec: float,
    margin_sec: float = 5.0,
    strategy: str = "two-stage"
) -> Tuple[List[str], List[str]]:
    """Build FFmpeg command line arguments for fast, frame-exact seeking.
    
    Strategies:
    - 'input': `-ss target` before `-i` (demuxer fast seek, O(1), may land on previous keyframe)
    - 'output': `-ss target` after `-i` (decoder full scan from 0, O(t), slow for large t)
    - 'two-stage' (legacy name):
        Uses one source-clock-preserving input seek.  The former split
        pre/post seek could reset the output clock and select the wrong frame
        on VFR input.
    
    Returns:
        (pre_input_args, post_input_args)
    """
    target = max(0.0, target_sec)
    
    if strategy == "output":
        return [], ["-ss", f"{target:.3f}"]
    
    if strategy == "input":
        return ["-ss", f"{target:.3f}"], []
    
    # Keep the public strategy name for compatibility, but use one input
    # seek. FFmpeg's transcoding path decodes forward from the keyframe seek
    # point and -copyts preserves the source clock for provenance.
    return ["-ss", f"{target:.3f}"], []


def extract_single_frame(
    video_path: str,
    timestamp_sec: float,
    output_path: str,
    width: Optional[int] = None,
    use_two_stage: bool = True,
    timeout_sec: Optional[float] = None
) -> FrameInfo:
    """Extract a single frame at exact timestamp using two-stage seeking.
    
    Guarantees frame-exact alignment without decoding from the beginning of large files.
    """
    vpath = Path(video_path).resolve()
    outpath = Path(output_path).resolve()

    if timestamp_sec < 0:
        raise ExtractionBoundsError(
            f"timestamp_sec must be >= 0 (got {timestamp_sec})",
            {"timestamp_sec": timestamp_sec}
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    strategy = "two-stage" if use_two_stage else "input"
    pre_args, post_args = build_seek_arguments(timestamp_sec, margin_sec=5.0, strategy=strategy)
    
    vf_filters = []
    vf_filters.append("showinfo")
    if width is not None:
        # Scale width to target, keep aspect ratio, force even height (-2)
        vf_filters.append(f"scale='min({width},iw)':-2")
        
    cmd = ["ffmpeg", "-y", "-copyts"]
    cmd.extend(pre_args)
    cmd.extend(["-i", str(vpath)])
    cmd.extend(post_args)
    cmd.extend(["-frames:v", "1"])
    
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])
        
    cmd.extend(["-q:v", "2", "-update", "1", str(outpath)])
    
    timeout = timeout_sec or Config.SUBPROCESS_TIMEOUT_SEC
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
            f"FFmpeg frame extraction timed out after {timeout}s at t={timestamp_sec:.2f}s",
            {"video_path": str(vpath), "timestamp_sec": timestamp_sec}
        ) from exc

    if proc.returncode != 0 or not outpath.exists():
        raise MediaCorruptError(
            f"FFmpeg failed to extract frame at t={timestamp_sec:.2f}s: {proc.stderr.strip()}",
            {"video_path": str(vpath), "stderr": proc.stderr}
        )

    # Determine dimensions
    w, h = 0, 0
    try:
        from PIL import Image
        with Image.open(outpath) as im:
            w, h = im.size
    except Exception:
        pass

    # This PTS comes from the same single-frame command that wrote outpath;
    # a nearby ffprobe interval is not sufficient for VFR provenance.
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
        file_path=str(outpath),
        width=w,
        height=h,
        requested_timestamp=timestamp_sec,
        decoded_timestamp=decoded_timestamp,
        timestamp_delta=timestamp_delta,
    )
