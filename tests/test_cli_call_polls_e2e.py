"""The one-line CLI call finishes a background visual job (build, then search) in its own process.

Needs the CLIP weights from the README's visual extra; the test skips when they are absent. Each `call` is a new process,
so the CLI itself must keep asking within that one process until the job is no longer queued or running.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.model, pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg makes the test clip")]


def _status(stdout):
    result = json.loads(stdout)
    status = (result.get("structuredContent") or {}).get("status")
    if status is None:
        status = json.loads("".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")).get("status")
    return status


def test_cli_call_builds_and_searches_visual_index_in_one_process(tmp_path):
    from src.retrieval.visual import clip_missing_message, clip_weights_missing_message

    if clip_missing_message() or clip_weights_missing_message():
        pytest.skip("CLIP weights are not installed (README: optional visual extra)")
    clip = tmp_path / "two_scenes.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x90:r=2:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=2:d=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", str(clip),
    ], check=True, timeout=60)
    env = dict(os.environ, AGENTIC_VIDEO_CACHE_DIR=str(tmp_path / "cache"))

    def cli(tool, arguments):
        return subprocess.run(
            [sys.executable, "-m", "src.executor.mcp_server", "call", tool, json.dumps(arguments)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
        )

    build = cli("build_visual_index", {"source": str(clip)})
    assert build.returncode == 0, build.stderr[-500:]
    assert _status(build.stdout) == "complete", build.stdout[:300]

    search = cli("search_visual", {"source": str(clip), "query": "red", "top_k": 3})
    assert search.returncode == 0, search.stderr[-500:]
    assert _status(search.stdout) == "complete", search.stdout[:300]
