import math
import threading
import subprocess
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict

_SAMPLING_LOCKS: Dict[str, threading.Lock] = {}
_SAMPLING_MUTEX = threading.Lock()

def _get_sampling_lock(key: str) -> threading.Lock:
    with _SAMPLING_MUTEX:
        if key not in _SAMPLING_LOCKS:
            _SAMPLING_LOCKS[key] = threading.Lock()
        return _SAMPLING_LOCKS[key]

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, DependencyNotFoundError, ExtractionBoundsError
from src.core.types import FrameInfo, InspectionWindowResult, format_timestamp
from src.core.cache import MediaCache
from src.media.probe import probe_video
from src.media.seeking import build_seek_arguments
from src.media.overlay import apply_overlay_to_file


def _request_tag(prefix: str, payload: dict) -> str:
    """Build a collision-resistant cache tag from the complete request."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _cached_frames_valid(manifest_items: List[dict]) -> bool:
    """Validate that every cached frame artifact exists and fully decodes.

    Guards against truncated/corrupt JPEG cache entries (e.g. disk-full or a
    killed extraction leaving partial files) being served to downstream
    vision models.
    """
    if not isinstance(manifest_items, list) or not manifest_items:
        return False
    from PIL import Image
    for item in manifest_items:
        p = Path(item.get("file_path", ""))
        try:
            if not p.exists() or p.stat().st_size <= 0:
                return False
            with Image.open(p) as im:
                im.load()
        except (OSError, ValueError):
            return False
    return True


def _cleanup_partial_frames(frames_dir: Path) -> None:
    """Remove partial frame artifacts left behind by a failed/killed extraction."""
    try:
        for f in frames_dir.glob("frame_*.jpg"):
            f.unlink()
    except OSError:
        pass


def _validate_window(
    start_sec: float,
    end_sec: Optional[float],
    media_duration_sec: float,
) -> tuple[float, Optional[float]]:
    """Normalize and reject invalid temporal windows before FFmpeg runs."""
    try:
        start = float(start_sec)
        end = float(end_sec) if end_sec is not None else None
        media_duration = float(media_duration_sec)
    except (TypeError, ValueError) as exc:
        raise ExtractionBoundsError(
            "Sampling timestamps must be numeric",
            {"start_sec": start_sec, "end_sec": end_sec},
        ) from exc

    if not all(math.isfinite(value) for value in (start, media_duration)) or (
        end is not None and not math.isfinite(end)
    ):
        raise ExtractionBoundsError(
            "Sampling timestamps and media duration must be finite",
            {"start_sec": start, "end_sec": end, "media_duration_sec": media_duration},
        )
    if start < 0 or (end is not None and end < 0):
        raise ExtractionBoundsError(
            "Sampling timestamps must be >= 0",
            {"start_sec": start, "end_sec": end},
        )
    if media_duration <= 0 or start >= media_duration:
        raise ExtractionBoundsError(
            "Sampling start must be before the end of the media",
            {"start_sec": start, "media_duration_sec": media_duration},
        )
    if end is not None:
        if end <= start:
            raise ExtractionBoundsError(
                "Sampling window must satisfy start_sec < end_sec",
                {"start_sec": start, "end_sec": end},
            )
        if end > media_duration + 0.001:
            raise ExtractionBoundsError(
                "Sampling end must not exceed the media duration",
                {"end_sec": end, "media_duration_sec": media_duration},
            )
    return start, end


def sample_sparse_frames(
    video_path: str,
    fps: float = Config.DEFAULT_SPARSE_FPS,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    overlay_timestamp: bool = True,
    cache: Optional[MediaCache] = None,
    output_dir: Optional[Path] = None,
    threads: Optional[int] = None,
) -> List[FrameInfo]:
    """High-performance sparse sampling across a video or segment.
    
    Default fps=0.2 (1 frame every 5 seconds).
    Slashes token consumption by up to 80-90% compared to static 1-FPS ingestion.
    """
    vpath = Path(video_path).resolve()
    if cache is None:
        cache = MediaCache()

    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ExtractionBoundsError("Requested FPS must be numeric", {"requested_fps": fps}) from exc
    if not math.isfinite(fps):
        raise ExtractionBoundsError("Requested FPS must be finite", {"requested_fps": fps})
    meta = probe_video(str(vpath), cache=cache)
    start_sec, end_sec = _validate_window(start_sec, end_sec, meta.duration_sec)
    duration = meta.duration_sec if end_sec is None else end_sec
    window_duration = max(0.1, duration - start_sec)

    cache.validate_extraction_bounds(fps=fps, duration_sec=window_duration)

    fingerprint = meta.fingerprint
    tag = _request_tag(
        "sparse",
        {
            "fps": fps,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": duration,
            "width": width,
            "overlay_timestamp": bool(overlay_timestamp),
        },
    )
    frames_dir = Path(output_dir) if output_dir is not None else cache.get_frames_dir(fingerprint, tag=tag)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cached_manifest = None if output_dir is not None else cache.load_json(fingerprint, f"{tag}_manifest")
    if cached_manifest is not None and _cached_frames_valid(cached_manifest):
        return [FrameInfo(**item) for item in cached_manifest]

    lock_key = f"{fingerprint}_{tag}"
    with _get_sampling_lock(lock_key):
        # Re-check cache inside lock in case another thread just completed extraction
        cached_manifest = None if output_dir is not None else cache.load_json(fingerprint, f"{tag}_manifest")
        if cached_manifest is not None and _cached_frames_valid(cached_manifest):
            return [FrameInfo(**item) for item in cached_manifest]
            
        # FFmpeg extraction via fps filter
        out_pattern = str(frames_dir / "frame_%04d.jpg")
        vf = f"fps={fps},scale='min({width},iw)':-2"
        
        pre_args, post_args = build_seek_arguments(start_sec, margin_sec=5.0)
        cmd = ["ffmpeg", "-y"]
        if threads is not None:
            cmd.extend(["-threads", str(threads), "-filter_threads", str(threads)])
        cmd.extend(pre_args)
        cmd.extend(["-i", str(vpath)])
        cmd.extend(post_args)
        if end_sec is not None:
            cmd.extend(["-t", f"{window_duration:.3f}"])
        cmd.extend(["-vf", vf, "-q:v", "2"])
        if threads is not None:
            # Input-side -threads bounds decoding; this bounds the JPEG encoder.
            cmd.extend(["-threads", str(threads)])
        cmd.append(out_pattern)
        
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
            _cleanup_partial_frames(frames_dir)
            raise SubprocessTimeoutError(f"Sparse sampling timed out after {Config.SUBPROCESS_TIMEOUT_SEC}s", {"video": str(vpath)}) from exc

        if proc.returncode != 0:
            _cleanup_partial_frames(frames_dir)
            raise MediaCorruptError(f"FFmpeg sparse extraction failed: {proc.stderr.strip()}", {"video": str(vpath)})

        frame_files = sorted([p for p in frames_dir.glob("frame_*.jpg")])
        step_sec = 1.0 / fps
        results: List[FrameInfo] = []
        
        for idx, fpath in enumerate(frame_files):
            t = start_sec + idx * step_sec
            if overlay_timestamp:
                apply_overlay_to_file(fpath, t, frame_index=idx + 1)
                
            from PIL import Image
            with Image.open(fpath) as im:
                w, h = im.size
            results.append(FrameInfo(
                index=idx + 1,
                timestamp_sec=round(t, 3),
                timecode=format_timestamp(t),
                file_path=str(fpath),
                width=w,
                height=h,
                requested_timestamp=round(t, 6),
                # Batch FFmpeg's fps filter retimes output frames.  Until a
                # per-output source-PTS receipt is available, null is safer
                # than claiming that a nearby source PTS was selected.
                decoded_timestamp=None,
                timestamp_delta=None,
            ))
            
        if output_dir is None:
            cache.save_json(fingerprint, f"{tag}_manifest", [f.to_dict() for f in results])
        return results


def sample_dense_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = Config.DEFAULT_DENSE_FPS,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    overlay_timestamp: bool = True,
    cache: Optional[MediaCache] = None
) -> InspectionWindowResult:
    """Dense time-window sampling (e.g. 1-10 FPS) for sub-second moment analysis.
    
    Used by agents to zoom into fast action, OCR transitions, or audio-visual anomalies.
    """
    vpath = Path(video_path).resolve()
    if cache is None:
        cache = MediaCache()

    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ExtractionBoundsError("Requested FPS must be numeric", {"requested_fps": fps}) from exc
    if not math.isfinite(fps):
        raise ExtractionBoundsError("Requested FPS must be finite", {"requested_fps": fps})
    meta = probe_video(str(vpath), cache=cache)
    start_sec, end_sec = _validate_window(start_sec, end_sec, meta.duration_sec)
    assert end_sec is not None
    duration = end_sec - start_sec
    cache.validate_extraction_bounds(fps=fps, duration_sec=duration)

    fingerprint = meta.fingerprint
    tag = _request_tag(
        "dense",
        {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "fps": fps,
            "width": width,
            "overlay_timestamp": bool(overlay_timestamp),
        },
    )
    frames_dir = cache.get_frames_dir(fingerprint, tag=tag)

    manifest_name = f"{tag}_manifest"
    cached = cache.load_json(fingerprint, manifest_name)
    if cached is not None and _cached_frames_valid(cached):
        frames = [FrameInfo(**item) for item in cached]
        return InspectionWindowResult(
            source_path=str(vpath),
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps,
            frames=frames,
            estimated_vision_tokens=len(frames) * Config.TOKENS_PER_IMAGE
        )

    lock_key = f"{fingerprint}_{tag}"
    with _get_sampling_lock(lock_key):
        # Re-check cache inside lock in case another thread just completed extraction
        cached = cache.load_json(fingerprint, manifest_name)
        if cached is not None and _cached_frames_valid(cached):
            frames = [FrameInfo(**item) for item in cached]
            return InspectionWindowResult(
                source_path=str(vpath),
                start_sec=start_sec,
                end_sec=end_sec,
                fps=fps,
                frames=frames,
                estimated_vision_tokens=len(frames) * Config.TOKENS_PER_IMAGE
            )
            
        out_pattern = str(frames_dir / "frame_%04d.jpg")
        vf = f"fps={fps},scale='min({width},iw)':-2"
        
        pre_args, post_args = build_seek_arguments(start_sec, margin_sec=5.0)
        cmd = ["ffmpeg", "-y"]
        cmd.extend(pre_args)
        cmd.extend(["-i", str(vpath)])
        cmd.extend(post_args)
        cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend(["-vf", vf, "-q:v", "2", out_pattern])
        
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
            _cleanup_partial_frames(frames_dir)
            raise SubprocessTimeoutError(f"Dense window extraction timed out after {Config.SUBPROCESS_TIMEOUT_SEC}s", {"video": str(vpath)}) from exc

        if proc.returncode != 0:
            _cleanup_partial_frames(frames_dir)
            raise MediaCorruptError(f"FFmpeg dense window extraction failed: {proc.stderr.strip()}", {"video": str(vpath)})

        frame_files = sorted([p for p in frames_dir.glob("frame_*.jpg")])
        step_sec = 1.0 / fps
        frames: List[FrameInfo] = []
        
        for idx, fpath in enumerate(frame_files):
            t = start_sec + idx * step_sec
            if overlay_timestamp:
                apply_overlay_to_file(fpath, t, frame_index=idx + 1)
                
            from PIL import Image
            with Image.open(fpath) as im:
                w, h = im.size
            frames.append(FrameInfo(
                index=idx + 1,
                timestamp_sec=round(t, 3),
                timecode=format_timestamp(t),
                file_path=str(fpath),
                width=w,
                height=h,
                requested_timestamp=round(t, 6),
                # The batch fps filter has a synthetic output clock.  Do not
                # label it as the PTS of the source frame actually selected.
                decoded_timestamp=None,
                timestamp_delta=None,
            ))
            
        cache.save_json(fingerprint, manifest_name, [f.to_dict() for f in frames])
    
    return InspectionWindowResult(
        source_path=str(vpath),
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        frames=frames,
        estimated_vision_tokens=len(frames) * Config.TOKENS_PER_IMAGE
    )
