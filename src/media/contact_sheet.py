"""Contact sheet generation via Pillow canvas composition and FFmpeg tile filter."""
import os
import time
import uuid
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Optional, Sequence, Union
from PIL import Image, ImageDraw, ImageFont

from src.core.config import Config
from src.core.errors import SubprocessTimeoutError, MediaCorruptError, DependencyNotFoundError, ExtractionBoundsError
from src.core.types import FrameInfo, ContactSheetResult, format_timestamp
from src.core.cache import MediaCache
from src.media.probe import probe_video

# One decode thread per ffmpeg: a one-frame job gains nothing from frame
# threads, and running more of them side by side finishes sooner.
SHEET_EXTRACT_WORKERS = 8


def default_sheet_path(
    cache: MediaCache, video_path: Union[str, Path], timestamps: Sequence[float], cols: int
) -> Path:
    """Cache path for a sheet of these timestamps; same name however the frames were made."""
    fingerprint = cache.compute_fingerprint(str(Path(video_path).resolve()))
    t_hash = hash(tuple(round(t, 2) for t in timestamps)) & 0xFFFFFF
    return cache.get_sheets_dir(fingerprint) / f"sheet_{len(timestamps)}samples_{cols}cols_{t_hash:x}.jpg"


def compose_contact_sheet(
    frames: List[FrameInfo],
    output_path: Union[str, Path],
    title: str = "VIDEO OVERVIEW",
    cols: int = 4,
    pad: int = 4,
    header_height: int = 36,
    bg_color: tuple = (20, 22, 28),
    header_text_color: tuple = (220, 225, 235),
    border_color: tuple = (45, 48, 58)
) -> ContactSheetResult:
    """Compose an annotated contact sheet grid from individual frames using Pillow.
    
    Benefits over FFmpeg tile filter:
    - Supports arbitrary frame counts without empty black padding tiles.
    - Annotates each tile with its exact timestamp and sequence index.
    - Adds a high-visibility header banner with video metadata and sampling context.
    - Yields a single high-efficiency image for vision models (~258 tokens instead of N * 258).
    """
    if not frames:
        raise ValueError("Cannot create contact sheet from empty frame list.")
        
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = len(frames)
    cols = max(1, min(cols, total))
    rows = math.ceil(total / cols)
    
    sample_im = Image.open(frames[0].file_path)
    tile_w, tile_h = sample_im.size
    sample_im.close()
    
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = header_height + rows * tile_h + (rows + 1) * pad
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_title = ImageFont.load_default(size=14)
        font_badge = ImageFont.load_default(size=12)
    except TypeError:
        font_title = ImageFont.load_default()
        font_badge = ImageFont.load_default()
        
    t_start = frames[0].timestamp_sec
    t_end = frames[-1].timestamp_sec
    header_str = f"{title} | {total} Frames | [{format_timestamp(t_start)} -> {format_timestamp(t_end)}] | Grid: {cols}x{rows}"
    draw.text((pad + 4, 10), header_str, fill=header_text_color, font=font_title)
    
    timestamps: List[float] = []
    
    for idx, f_info in enumerate(frames):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_w + pad)
        y = header_height + pad + r * (tile_h + pad)
        
        timestamps.append(f_info.timestamp_sec)
        
        with Image.open(f_info.file_path) as tile_img:
            if tile_img.size != (tile_w, tile_h):
                tile_img = tile_img.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            canvas.paste(tile_img, (x, y))
            
        draw.rectangle([x - 1, y - 1, x + tile_w, y + tile_h], outline=border_color, width=1)
        
        badge_text = f"#{idx+1}  {format_timestamp(f_info.timestamp_sec)}"
        bbox = font_badge.getbbox(badge_text)
        bw, bh = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 4
        bx, by = x + 4, y + 4
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(10, 12, 16), outline=(0, 200, 255), width=1)
        draw.text((bx + 4, by + 2), badge_text, fill=(255, 255, 255), font=font_badge)
        
    tmp_out = out_path.parent / f"{out_path.stem}.{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp.jpg"
    canvas.save(tmp_out, quality=Config.JPEG_QUALITY)
    for attempt in range(5):
        try:
            os.replace(str(tmp_out), str(out_path))
            break
        except (OSError, IOError):
            if attempt < 4:
                time.sleep(0.015 * (attempt + 1))
            else:
                try:
                    import shutil
                    shutil.move(str(tmp_out), str(out_path))
                except Exception:
                    pass
    
    return ContactSheetResult(
        file_path=str(out_path.resolve()),
        sample_count=total,
        rows=rows,
        cols=cols,
        start_sec=t_start,
        end_sec=t_end,
        timestamps=timestamps,
        frame_width=tile_w,
        frame_height=tile_h
    )


def build_contact_sheet(
    video_path: Union[str, Path],
    timestamps: List[float],
    output_path: Optional[Union[str, Path]] = None,
    cols: int = 4,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    title: Optional[str] = None,
    cache: Optional[MediaCache] = None,
    deadline_sec: Optional[float] = None,
) -> ContactSheetResult:
    """Extract frames at specified timestamps and compose a contact sheet.

    With `deadline_sec`, frame extraction stops waiting after that long and
    raises SubprocessTimeoutError; frames already extracted stay cached, so a
    repeat call continues from there.
    """
    from src.media.frames import extract_single_frame
    if cache is None:
        cache = MediaCache()
        
    vpath = Path(video_path).resolve()

    def _extract(t: float) -> FrameInfo:
        return extract_single_frame(
            video_path=str(vpath), timestamp_sec=t, width=width, cache=cache, threads=1
        )

    # One ffmpeg process per frame; running a few at once keeps large sheets
    # well inside the MCP request timeout.
    pool = ThreadPoolExecutor(max_workers=SHEET_EXTRACT_WORKERS)
    futures = [pool.submit(_extract, t) for t in timestamps]
    done, pending = wait(futures, timeout=deadline_sec)
    if pending:
        pool.shutdown(wait=False, cancel_futures=True)
        raise SubprocessTimeoutError(
            f"Extracted {len(done)} of {len(timestamps)} frames before the {deadline_sec:g} s limit "
            "(the machine is busy or the video is slow to seek). Extracted frames are cached, so the "
            "same call again continues from there; or ask for fewer samples",
            {"extracted": len(done), "requested": len(timestamps), "deadline_sec": deadline_sec},
        )
    pool.shutdown()
    frames: List[FrameInfo] = [future.result() for future in futures]
    for idx, f in enumerate(frames):
        f.index = idx + 1

    if output_path is None:
        out_path = default_sheet_path(cache, vpath, timestamps, cols)
    else:
        out_path = Path(output_path).resolve()
        
    return compose_contact_sheet(
        frames=frames,
        output_path=out_path,
        title=title or f"CONTACT SHEET: {vpath.name}",
        cols=cols
    )


def generate_video_overview(
    video_path: Union[str, Path],
    samples: int = 8,
    cols: int = 4,
    output_path: Optional[Union[str, Path]] = None,
    width: int = Config.DEFAULT_FRAME_WIDTH,
    cache: Optional[MediaCache] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    deadline_sec: Optional[float] = None,
) -> ContactSheetResult:
    """Uniformly sample a whole video or bounded temporal range."""
    if cache is None:
        cache = MediaCache()
        
    vpath = Path(video_path).resolve()
    meta = probe_video(str(vpath), cache=cache)
    duration = meta.duration_sec

    try:
        start = float(start_sec)
        end = duration if end_sec is None else float(end_sec)
    except (TypeError, ValueError) as exc:
        raise ExtractionBoundsError(
            "Overview timestamps must be numeric",
            {"start_sec": start_sec, "end_sec": end_sec},
        ) from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise ExtractionBoundsError(
            "Overview timestamps must be finite",
            {"start_sec": start, "end_sec": end},
        )
    if start < 0 or start >= duration:
        raise ExtractionBoundsError(
            "Overview start must satisfy 0 <= start_sec < media duration (media_duration_sec in details; probe_video reports it)",
            {"start_sec": start, "media_duration_sec": duration},
        )
    # Durations people quote are often rounded up to the next second; clamp those instead of failing.
    if end <= start or end > duration + 1.0:
        raise ExtractionBoundsError(
            "Overview window must satisfy start_sec < end_sec <= media duration; omit end_sec to run to the end of the video",
            {"start_sec": start, "end_sec": end, "media_duration_sec": duration},
        )
    end = min(end, duration)
    window_duration = end - start

    if samples < 1 or samples > Config.MAX_FRAMES_PER_CALL:
        raise ExtractionBoundsError(
            f"Requested samples ({samples}) out of bounds (must be 1 <= samples <= {Config.MAX_FRAMES_PER_CALL})",
            {"requested_samples": samples, "max_samples": Config.MAX_FRAMES_PER_CALL}
        )

    if samples <= 1:
        timestamps = [0.0 if start == 0.0 and end_sec is None else start]
    else:
        step = window_duration / samples
        # Sample center of each interval
        timestamps = [round(start + (i + 0.5) * step, 3) if start == 0.0 and end_sec is None else start + (i + 0.5) * step for i in range(samples)]
        
    return build_contact_sheet(
        video_path=vpath,
        timestamps=timestamps,
        output_path=output_path,
        cols=cols,
        width=width,
        cache=cache,
        deadline_sec=deadline_sec,
    )


def generate_ffmpeg_tile_sheet(
    video_path: str,
    output_path: str,
    fps: float = 0.2,
    grid: str = "4x4",
    width: int = 320,
    timeout_sec: Optional[float] = None
) -> str:
    """Fast single-pass raw contact sheet generation via FFmpeg tile filter."""
    vpath = Path(video_path).resolve()
    outpath = Path(output_path).resolve()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    vf = f"fps={fps},scale='min({width},iw)':-2,tile={grid}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vpath),
        "-vf", vf,
        "-frames:v", "1",
        "-q:v", "2",
        "-update", "1",
        str(outpath)
    ]
    
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
        raise SubprocessTimeoutError(f"FFmpeg tile timed out after {timeout}s", {"video": str(vpath)}) from exc

    if proc.returncode != 0 or not outpath.exists():
        raise MediaCorruptError(f"FFmpeg tile failed: {proc.stderr.strip()}", {"video": str(vpath)})

    return str(outpath)
