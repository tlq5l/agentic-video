"""review_video end to end: a real tools/call on the 60 s fixture clip."""

import json
import time
from pathlib import Path

from src.core.config import Config
from src.executor import mcp_server

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_bench_60s.mp4"


def test_review_video_returns_all_four_views_of_a_real_clip_within_the_time_limit(tmp_path, monkeypatch):
    # Cold run: an empty cache, so the timing below is the worst case.
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")

    started = time.monotonic()
    response = json.loads(mcp_server.tool_call_response(1, "review_video", {"source": str(FIXTURE)}))
    elapsed = time.monotonic() - started

    result = response["result"]
    assert not result.get("isError"), result
    data = result["structuredContent"]
    assert data["metadata"]["duration_sec"] == 60.0 and data["metadata"]["has_audio"] is True
    assert data["contact_sheet"]["sample_count"] == 24
    assert Path(data["contact_sheet"]["file_path"]).is_file()
    assert data["contact_sheet"]["tool_file"]["_tag"] == "ToolFile"
    assert data["cuts_total"] == len(data["cuts"]) > 0
    assert data["loudness"]["peak_sec"] > 0 and data["loudness"]["outliers"]
    assert elapsed < 45, f"review_video took {elapsed:.1f} s; Executor cuts calls at 60 s"


def test_review_video_missing_file_is_a_tool_error():
    result = json.loads(mcp_server.tool_call_response(1, "review_video", {"source": "/nonexistent/clip.mp4"}))["result"]
    assert result["isError"] is True
    assert "MediaNotFoundError" in result["content"][0]["text"]
