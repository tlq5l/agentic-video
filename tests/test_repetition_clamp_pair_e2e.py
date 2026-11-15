"""Regression for gpt6 finding 2: clamping min_period_sec and max_period_sec one at a time turned a valid pair into
equal endpoints. Drives count_repetitions through the MCP handle_call entry point and waits for each job to settle."""
import subprocess
import time

import pytest

from src.core.config import Config
from src.executor import mcp_server


def _args(source, min_period, max_period):
    return {"source": str(source), "end_sec": 3.0, "fps": 10, "method": "motion",
            "min_period_sec": min_period, "max_period_sec": max_period}


def _count(source, min_period, max_period):
    for _ in range(240):
        data = mcp_server.handle_call("count_repetitions", _args(source, min_period, max_period))["data"]
        if data.get("status") != "in_progress":
            return data
        time.sleep(0.5)
    raise AssertionError("repetition count did not finish")


def test_clamped_period_pair_stays_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "bars.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-t", "4",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    # Both ends below 2/15 s: min 0.05, max 0.10 used to clamp to the same 2/15 s. The max end moves one frame up.
    low = _count(source, 0.05, 0.10)
    assert "count" in low and "max_period_sec set to 0.2 " in low["note"] and "min_period_sec set to 0.133" in low["note"], low
    assert "Moved one frame" in low["note"], low
    # Both ends above 180 s: min 200, max 250 used to clamp to the same 180 s. The min end moves one frame down.
    high = _count(source, 200, 250)
    assert "count" in high and "min_period_sec set to 179.933 " in high["note"] and "Moved one frame" in high["note"], high
    # One end already on the bound: min 1e-9 clamps to 2/15 s and max is 2/15 s. The on-bound max end moves.
    edge = _count(source, 1e-9, 2 / 15)
    assert "count" in edge and "max_period_sec set to 0.2 " in edge["note"] and "Moved one frame" in edge["note"], edge
    # An inverted request was never valid: the engine still refuses it, and nothing repairs it.
    with pytest.raises(ValueError, match="below max_period_sec"):
        mcp_server.handle_call("count_repetitions", _args(source, 0.10, 0.05))
