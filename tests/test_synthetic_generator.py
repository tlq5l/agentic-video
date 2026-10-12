"""
Checks of the synthetic fixture made by scripts/generate_synthetic_benchmark.py.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image


def test_synthetic_files_exist():
    fixtures_dir = Path("fixtures")
    assert (fixtures_dir / "synthetic_bench_60s.mp4").exists()
    assert (fixtures_dir / "synthetic_bench_60s.vtt").exists()
    assert (fixtures_dir / "synthetic_bench_60s.srt").exists()
    assert (fixtures_dir / "synthetic_ground_truth.json").exists()


def test_synthetic_video_streams():
    mp4_path = Path("fixtures/synthetic_bench_60s.mp4")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,codec_name,width,height",
        "-of", "json",
        str(mp4_path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)

    duration = float(data["format"]["duration"])
    assert duration == pytest.approx(60.0, abs=0.1)

    streams = data.get("streams", [])
    assert len(streams) >= 2

    v_stream = next(s for s in streams if s.get("codec_type") == "video")
    assert v_stream["codec_name"] == "h264"
    assert v_stream["width"] == 640
    assert v_stream["height"] == 360

    a_stream = next(s for s in streams if s.get("codec_type") == "audio")
    assert a_stream["codec_name"] == "aac"


def test_ground_truth_json_schema():
    gt_path = Path("fixtures/synthetic_ground_truth.json")
    data = json.loads(gt_path.read_text(encoding="utf-8"))

    assert data["benchmark_name"] == "synthetic_bench_60s"
    assert data["duration_s"] == 60.0

    tasks = data["tasks"]
    assert "task1_locate_single_event" in tasks
    assert "task2_temporal_sequence" in tasks
    assert "task3_transcript_verification" in tasks
    assert "task4_visual_state_change" in tasks
    assert "task5_sparse_recurring_event" in tasks
    assert "task6_negative_control" in tasks


def test_subtitles_content():
    vtt_text = Path("fixtures/synthetic_bench_60s.vtt").read_text(encoding="utf-8")
    assert "WEBVTT" in vtt_text
    assert "Project Cygnus activated" in vtt_text
    assert "Supernova optical flash" in vtt_text


VIDEO = Path("fixtures/synthetic_bench_60s.mp4")


def _frame_at(timestamp_s: float) -> Image.Image:
    """Decode one frame of the synthetic fixture at an exact timestamp."""
    out = subprocess.run(
        ["ffmpeg", "-ss", f"{timestamp_s:.3f}", "-i", str(VIDEO), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    ).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def test_visual_coherence_task1_supernova():
    # At 14.5s the centre shows a golden starburst (high R, mid G, low B).
    img = _frame_at(14.5)
    r, g, b = img.getpixel((img.width // 2, 200))
    assert r > 200 and g > 120 and b < 50, f"Expected golden starburst, got RGB=({r},{g},{b})"


def test_visual_coherence_task2_sequence():
    # Blue square, then green circle, then red triangle.
    cx, cy = 320, 200
    r, g, b = _frame_at(24.5).getpixel((cx, cy))
    assert b > 200 and r < 100, f"Expected blue square, got RGB=({r},{g},{b})"
    r, g, b = _frame_at(26.5).getpixel((cx, cy))
    assert g > 150 and r < 50, f"Expected green circle, got RGB=({r},{g},{b})"
    r, g, b = _frame_at(28.5).getpixel((cx, cy))
    assert r > 200 and g < 100 and b < 100, f"Expected red triangle, got RGB=({r},{g},{b})"


def test_visual_coherence_task4_subsecond_state_change():
    # The background flips from dark slate to crimson at 39.5s.
    bg_before = _frame_at(39.0).getpixel((10, 50))
    assert bg_before[0] < 50 and bg_before[2] > bg_before[0], f"Expected slate background, got {bg_before}"
    bg_after = _frame_at(39.6).getpixel((10, 50))
    assert bg_after[0] > 110 and bg_after[1] < 40, f"Expected crimson alert, got {bg_after}"
