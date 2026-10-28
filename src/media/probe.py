"""Media probing and metadata extraction via ffprobe."""
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.config import Config
from src.core.errors import MediaNotFoundError, MediaCorruptError, SubprocessTimeoutError, DependencyNotFoundError
from src.core.types import VideoMetadata
from src.core.cache import MediaCache


def parse_fps(fps_str: str) -> float:
    """Parse fractional frame rate string (e.g. '30000/1001' or '25/1') to float."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            denominator = float(den)
            if denominator == 0:
                return 0.0
            return float(num) / denominator
        return float(fps_str)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


METADATA_CACHE_NAME = "metadata_display_size"


def _is_quarter_turn(video_stream: Dict[str, Any]) -> bool:
    """True when the stream is rotated 90 or 270 degrees, so its width and height swap on screen."""
    angle = None
    for side_data in video_stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            angle = side_data["rotation"]
            break
    if angle is None:
        angle = (video_stream.get("tags") or {}).get("rotate")
    try:
        return abs(float(angle or 0)) % 180 == 90
    except (TypeError, ValueError):
        return False


def probe_video(file_path: str, cache: Optional[MediaCache] = None, use_cache: bool = True) -> VideoMetadata:
    """Probe video file using ffprobe and return structured VideoMetadata.

    width and height are the displayed size: FFmpeg applies the rotation when it
    decodes, so frames come out in that size.
    Reads from local MediaCache if available.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise MediaNotFoundError(f"Video file does not exist: {file_path}", {"path": str(path)})

    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise MediaNotFoundError(
            f"Video file exists but cannot be read (locked or permission denied): {file_path}",
            {"path": str(path), "reason": str(exc)}
        ) from exc

    if cache is None:
        cache = MediaCache()

    fingerprint = cache.compute_fingerprint(str(path))
    if use_cache:
        cached_data = cache.load_json(fingerprint, METADATA_CACHE_NAME)
        if cached_data:
            return VideoMetadata(**cached_data)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(path)
    ]

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
            f"ffprobe binary not found on system PATH: {exc}",
            {"dependency": "ffprobe"}
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SubprocessTimeoutError(
            f"ffprobe timed out after {Config.SUBPROCESS_TIMEOUT_SEC}s on {file_path}",
            {"file_path": str(path), "timeout": Config.SUBPROCESS_TIMEOUT_SEC}
        ) from exc

    if proc.returncode != 0:
        raise MediaCorruptError(
            f"ffprobe failed with exit code {proc.returncode}: {proc.stderr.strip()}",
            {"file_path": str(path), "stderr": proc.stderr}
        )

    try:
        probe_json = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaCorruptError(
            f"Failed to parse ffprobe JSON output for {file_path}",
            {"file_path": str(path), "raw_output": proc.stdout[:500]}
        ) from exc

    streams = probe_json.get("streams", [])
    format_info = probe_json.get("format", {})

    video_stream: Optional[Dict[str, Any]] = None
    audio_stream: Optional[Dict[str, Any]] = None

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "video" and video_stream is None:
            video_stream = s
        elif c_type == "audio" and audio_stream is None:
            audio_stream = s

    if not video_stream:
        raise MediaCorruptError(
            f"No video streams found in {file_path}",
            {"file_path": str(path), "streams_count": len(streams)}
        )

    # Resolution
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if _is_quarter_turn(video_stream):
        width, height = height, width

    # FPS: prefer avg_frame_rate if valid, as r_frame_rate can report timebase (e.g. 1000/1, 90000/1) for VFR
    avg_fps = parse_fps(video_stream.get("avg_frame_rate", "0/0"))
    r_fps = parse_fps(video_stream.get("r_frame_rate", "0/0"))
    if 0.0 < avg_fps <= 120.0:
        fps = avg_fps
    elif 0.0 < r_fps <= 120.0:
        fps = r_fps
    elif avg_fps > 0.0:
        fps = avg_fps
    elif r_fps > 0.0:
        fps = r_fps
    else:
        fps = 25.0

    # Duration
    duration_sec = 0.0
    if "duration" in video_stream:
        try:
            duration_sec = float(video_stream["duration"])
        except (ValueError, TypeError):
            pass

    if duration_sec <= 0.0 and "duration" in format_info:
        try:
            duration_sec = float(format_info["duration"])
        except (ValueError, TypeError):
            pass

    format_name = format_info.get("format_name", "unknown")
    size_bytes = int(format_info.get("size", path.stat().st_size if path.exists() else 0))

    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    video_codec = video_stream.get("codec_name")

    metadata = VideoMetadata(
        source_path=str(path),
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        format_name=format_name,
        size_bytes=size_bytes,
        has_audio=has_audio,
        audio_codec=audio_codec,
        video_codec=video_codec,
        fingerprint=fingerprint
    )

    cache.save_json(fingerprint, METADATA_CACHE_NAME, metadata.to_dict())
    return metadata
