"""Soft changes that the scene-score filter misses, through the detect_cuts tool on real lossless clips."""
import shutil
import subprocess

import numpy as np
import pytest

from src.core.config import Config
from src.executor import mcp_server
from src.media import scene

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

W, H, FPS = 160, 90, 30


def _encode(path, frames):
    raw = np.stack(frames).astype(np.uint8).tobytes()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(path)]
    subprocess.run(cmd, input=raw, check=True)
    return str(path)


def _flat(level):
    return np.full((H, W, 3), level, dtype=np.uint8)


TEXTURE = np.random.default_rng(3).integers(40, 200, size=(H, 64, 3), dtype=np.uint8)


def _fade_clip(path, start, duration, end=113.0, layout="flat"):
    """A low-contrast fade from 100 to `end` over `duration` seconds from `start`, in a 6 s clip.

    "flat": the whole picture fades. "bars": fixed black bars at the top and bottom, the picture between them
    fades. "static": a fixed textured picture on the left 40 %, the rest fades.
    """
    frames = []
    for n in range(6 * FPS):
        t = n / FPS
        level = 100.0 if t < start else end if t >= start + duration else 100.0 + (end - 100.0) * (t - start) / duration
        frame = _flat(int(round(level)))
        if layout == "bars":
            frame[:11] = 0
            frame[79:] = 0
        elif layout == "static":
            frame[:, :64] = TEXTURE
        frames.append(frame)
    return _encode(path, frames)


def _motion_then_step_clip(path):
    """A slowly moving square on a flat picture, then at 3 s a brightness step too small for the scene filter."""
    frames = []
    for n in range(6 * FPS):
        if n < 3 * FPS:
            frame = _flat(64)
            x = 10 + n
            frame[20:50, x: x + 30] = 64 + 120
        else:
            frame = _flat(64 + 12)
            frame[20:50, 100:130] = 64 + 12 + 60
        frames.append(frame)
    return _encode(path, frames)


def _slide_clip(path):
    """Two static slides: flat 100, then the left half turns 116 at 3 s."""
    frames = []
    for n in range(6 * FPS):
        frame = _flat(100)
        if n >= 3 * FPS:
            frame[:, : W // 2] = 116
        frames.append(frame)
    return _encode(path, frames)


def _detect(source, cache_dir, monkeypatch, large=False):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", cache_dir)
    if large:
        monkeypatch.setattr(scene, "SCAN_SMALL_PIXEL_SEC", 0)
    outcome = mcp_server.handle_call("detect_cuts", {"source": source})
    assert outcome["ok"]
    return [(cut["kind"], round(cut["timestamp_sec"], 2)) for cut in outcome["data"]["cuts"]]


@pytest.mark.parametrize("large", [False, True], ids=["small_video", "large_video"])
def test_low_scoring_step_after_motion_is_a_cut_on_both_scan_paths(tmp_path, monkeypatch, large):
    clip = _motion_then_step_clip(tmp_path / "motion_then_step.mkv")
    found = _detect(clip, tmp_path / ("large" if large else "small"), monkeypatch, large)
    assert [(kind, t) for kind, t in found if kind == "cut"] == [("cut", 3.0)]


@pytest.mark.parametrize("large", [False, True], ids=["small_video", "large_video"])
@pytest.mark.parametrize("duration", [0.2, 0.3, 0.4])
@pytest.mark.parametrize("start", [3.0, 3.033, 3.067])
@pytest.mark.parametrize("layout", ["flat", "bars", "static"])
def test_low_contrast_fade_is_a_transition_on_both_scan_paths(tmp_path, monkeypatch, layout, start, duration, large):
    end = 113.0 if layout == "flat" else 117.0
    clip = _fade_clip(tmp_path / f"fade_{layout}_{start}_{duration}.mkv", start, duration, end, layout)
    found = _detect(clip, tmp_path / ("large" if large else "small"), monkeypatch, large)
    assert [kind for kind, _ in found] == ["transition"]
    assert start < found[0][1] < start + duration + 0.3


def test_weak_cut_between_still_slides_is_kept(tmp_path, monkeypatch):
    clip = _slide_clip(tmp_path / "slides.mkv")
    found = _detect(clip, tmp_path / "cache", monkeypatch)
    assert found == [("cut", 3.0)]
