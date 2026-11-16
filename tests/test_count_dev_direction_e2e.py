"""End to end: count_occurrences(direction=left_to_right) on count_dev seed 6102 occ26_q1 counts a crossing split by detection dropouts once.

Round-9 box printout on 992e4a3: main counted 5 (truth 2); the crossing at 7.2-10.0 s came back as three spans (7.2-8.2 s,
9.0 s and 9.6-10.0 s), each counted. A 1-sample look-alike at 16.4 s is still counted as too brief to show a direction (kept:
occurrences seen in one sample per crossing, as on count_dev occ10_q1 and occ12_q1, are only counted that way), so 3 is accepted.
"""
import shutil

import pytest

from benchmarks.dev.count_dev import generate
from benchmarks.dev.count_dev.score_tools import crossing_option, run_tool, tool_call

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def test_split_crossing_counts_once(tmp_path, monkeypatch):
    from src.core.config import Config
    from src.executor import mcp_server
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    spec = next(s for s in generate.VIDEOS if s[0] == "occ26")
    questions, truths, render = generate.plan_video(spec, 6102)
    row = next(q for q in questions if q["id"] == "occ26_q1")
    truth = next(t for t in truths if t["id"] == "occ26_q1")
    assert row["mode"] == "left_to_right" and truth["count"] == 2
    video = tmp_path / "occ26.mp4"
    generate._render(video, render["duration"], render["events"], render["texts"], render["cuts"], render["seed"],
                     render["motion_blur"], render["life_color"], render["noise"], render["tint"])
    _tool, args = tool_call(mcp_server, row, video, crossing_option(mcp_server))
    assert args["direction"] == "left_to_right"
    result = run_tool(mcp_server, "count_occurrences", args)
    assert "error" not in result, result
    spans = result["spans"]
    assert result["count"] in (2, 3), (result["count"], spans, result.get("excluded_spans"))
    assert not any(9.0 <= s["start_sec"] <= 10.2 for s in spans), spans
    assert any(s["start_sec"] <= 7.4 and s["end_sec"] >= 10.0 for s in spans), spans
