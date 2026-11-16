"""End to end: count_occurrences(direction=left_to_right) on count_dev seed 6102 occ26_q1 counts a crossing split by detection dropouts once.

Round-9 box printout on 992e4a3: main counted 5 (truth 2); the crossing at 7.2-10.0 s came back as three spans (7.2-8.2 s,
9.0 s and 9.6-10.0 s), each counted.

The video is a pinned clip, the first 17 s of the generator's render made on the test box (ffmpeg 5.1.9, DejaVu Sans), cut by
stream copy. Rendering here instead makes the answer depend on the machine: the Mac's own render (ffmpeg 9.0.2, Apple Symbols)
breaks the 2.4-4.0 s crossing into one-sample spans, while decoding the box render on the Mac gives the box's spans.

Round-11 box printout on 3ce7927: the full render counted 3; the extra was one sample at 16.4 s, the blurred fringe of a cyan circle decoy (strength-colour pixels in its box: blue 278, cyan 419, purple 512). A blob now counts for a colour only when that colour is at least 35% of those pixels.
"""
import shutil
from pathlib import Path

import pytest

from benchmarks.dev.count_dev import generate
from benchmarks.dev.count_dev.score_tools import crossing_option, run_tool, tool_call

CLIP = Path(__file__).parent / "fixtures" / "count_dev" / "occ26-6102-first17s.mp4"

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def test_split_crossing_counts_once(tmp_path, monkeypatch):
    from src.core.config import Config
    from src.executor import mcp_server
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    spec = next(s for s in generate.VIDEOS if s[0] == "occ26")
    questions, truths, _render = generate.plan_video(spec, 6102)
    row = next(q for q in questions if q["id"] == "occ26_q1")
    truth = next(t for t in truths if t["id"] == "occ26_q1")
    assert row["mode"] == "left_to_right" and truth["count"] == 2
    _tool, args = tool_call(mcp_server, row, CLIP, crossing_option(mcp_server))
    assert args["direction"] == "left_to_right"
    result = run_tool(mcp_server, "count_occurrences", args)
    assert "error" not in result, result
    spans = result["spans"]
    assert result["count"] == 2, (result["count"], spans, result.get("excluded_spans"))
    assert not any(16.2 <= s["start_sec"] <= 16.8 for s in spans), spans
    assert not any(9.0 <= s["start_sec"] <= 10.2 for s in spans), spans
    assert any(s["start_sec"] <= 7.4 and s["end_sec"] >= 10.0 for s in spans), spans
