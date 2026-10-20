"""End-to-end: frame tools must work when a sample lands after the last frame's start time."""
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src.core.config import Config
from src.core.errors import MediaCorruptError
from src.executor import mcp_server

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_bench_60s.mp4"


def test_inspect_moments_anchor_at_video_end(tmp_path, monkeypatch):
    # 60.0 s at 25 fps: the last frame starts at 59.96 s, so a seek to 59.999 s (the clamped end of the window) decodes no frame.
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path)
    outcome = mcp_server.handle_call("inspect_moments", {
        "source": str(FIXTURE),
        "moments": [{"timestamp_sec": 59.5, "label": "end"}],
        "before_sec": 1.0, "after_sec": 1.0, "samples_per_moment": 3,
    })
    assert outcome["ok"]
    moment = outcome["data"]["moments"][0]
    assert moment["sample_count"] == 3
    assert moment["sample_timestamps"][-1] == 59.999
    with Image.open(moment["contact_sheet_path"]) as sheet:
        assert sheet.width > 0


@pytest.mark.parametrize("moment_sec", [33.0, 45.0])
def test_inspect_moments_on_truncated_file_still_fails(tmp_path, monkeypatch, moment_sec):
    # A half-downloaded file still reports 60 s, but its frames stop at 29.04 s. Samples past that
    # must fail rather than show the 29.04 s frame labelled with a later time, both a few seconds
    # past the cut (inside the look-back window) and far past it.
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    whole = tmp_path / "whole.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(FIXTURE), "-c", "copy", "-movflags", "+faststart", str(whole)], check=True)
    truncated = tmp_path / "truncated.mp4"
    data = whole.read_bytes()
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(MediaCorruptError):
        mcp_server.handle_call("inspect_moments", {
            "source": str(truncated),
            "moments": [{"timestamp_sec": moment_sec}],
            "before_sec": 1.0, "after_sec": 1.0, "samples_per_moment": 3,
        })
