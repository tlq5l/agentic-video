"""search_visual's rank-1 interval on real CLIP: a 3 s shot inside a 40 s clip gets a short window that overlaps it."""
import shutil
import subprocess
import time

import pytest

from src.core.cache import MediaCache
from src.executor import mcp_server
from src.retrieval import visual
from src.retrieval.engine import AgenticVideoEngine

SHOT_START, SHOT_END = 18.0, 21.0
QUERY = "colourful test pattern with numbers"


def _make_spliced_clip(path):
    """Lossless 40 s clip: 18 s of gradients, a 3 s test pattern, then 19 s of gradients."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for visual-index integration tests")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-t", "18", "-f", "lavfi", "-i", "gradients=s=320x180:r=10:seed=3",
        "-t", "3", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10",
        "-t", "19", "-f", "lavfi", "-i", "gradients=s=320x180:r=10:seed=4",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
        "-threads", "1", str(path),
    ], check=True, timeout=120)
    return path


def _poll(name, args, done, timeout_sec=600):
    deadline = time.monotonic() + timeout_sec
    while True:
        response = mcp_server.handle_call(name, args)
        assert response["ok"], response
        if done(response["data"]):
            return response["data"]
        assert time.monotonic() < deadline, f"{name} did not finish: {response['data']}"
        time.sleep(0.5)


def test_search_visual_rank_one_window_stays_short_and_covers_shot(tmp_path, monkeypatch):
    try:
        visual.get_encoder()
    except RuntimeError as exc:
        pytest.skip(f"real CLIP needs requirements-visual.txt and cached weights: {exc}")
    cache = MediaCache(tmp_path / "cache")
    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", lambda source: AgenticVideoEngine(source, cache=cache))
    source = str(_make_spliced_clip(tmp_path / "spliced.mp4"))
    built = _poll("build_visual_index", {"source": source},
                  lambda data: data.get("status") in ("complete", "error"))
    assert built["status"] == "complete", built
    data = _poll("search_visual", {"source": source, "query": QUERY, "top_k": 5},
                 lambda found: "intervals" in found or found.get("status") == "error")
    assert "intervals" in data, data
    top = next(item for item in data["intervals"] if 1 in item["ranks"])
    assert top["end_sec"] - top["start_sec"] < 15.0
    assert top["start_sec"] < SHOT_END and top["end_sec"] > SHOT_START
