"""Round-11: motion blur makes the outlined triangle fail the convex and outlined-ink tests; main counts 0 crossings in this clip, truth 2 at 2.0-3.79 s and 4.9-7.17 s.

The video is a pinned clip, the first 8 s of the generator's render (seed 5117, occ10) made on the test box, cut by stream copy.
Rendering here instead makes the answer depend on the machine.
"""
import shutil
from pathlib import Path

import pytest

from benchmarks.dev.count_dev import generate
from benchmarks.dev.count_dev.score_tools import crossing_option, run_tool, tool_call

CLIP = Path(__file__).parent / "fixtures" / "count_dev" / "occ10-5117-first8s.mp4"

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def test_blurred_triangle_counts_its_crossings(tmp_path, monkeypatch):
    from src.core.config import Config
    from src.executor import mcp_server
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    spec = next(s for s in generate.VIDEOS if s[0] == "occ10")
    questions, truths, _render = generate.plan_video(spec, 5117)
    row = next(q for q in questions if q["id"] == "occ10_q1")
    truth = next(t for t in truths if t["id"] == "occ10_q1")
    assert row["mode"] == "left_to_right" and truth["count"] == 2
    _tool, args = tool_call(mcp_server, row, CLIP, crossing_option(mcp_server))
    assert args["direction"] == "left_to_right"
    result = run_tool(mcp_server, "count_occurrences", args)
    assert "error" not in result, result
    spans = result["spans"]
    assert result["count"] == 2, (result["count"], spans, result.get("excluded_spans"))
    for a, b in [(2.0, 3.79), (4.9, 7.17)]:
        assert any(s["start_sec"] <= b and s["end_sec"] >= a for s in spans), spans
