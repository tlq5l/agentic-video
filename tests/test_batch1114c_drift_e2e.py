"""End-to-end proofs for the batch 2026-11-14c drift findings on agentic-video (one test per finding).

Each test states what the docs or the switch rule promise and drives the public entry point the way a user does.
"""

import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from src.core.config import Config
from src.executor.mcp_server import handle_call
from src.media import cloudflare, motion_probe
from src.transcript import shadow_cf_asr

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "features"


def test_d3_expired_trial_with_switch_off_writes_nothing(tmp_path, monkeypatch):
    """With Workers AI switched off, a fresh run must be refused before the trial check writes trial_ended."""
    state = tmp_path / "state" / "agentic-video"
    state.mkdir(parents=True)
    expired = {"started_at": time.time() - 15 * 86400, "count": 3, "ended": False}
    (state / "shadow-cf-asr-state.json").write_text(json.dumps(expired), encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, socket.gethostname())
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t" * 20)
    monkeypatch.setattr(cloudflare, "WORKERS_AI_ON", False)
    result = SimpleNamespace(duration_sec=20.0, language="en", has_speech=True, text="", segments=[])

    started = shadow_cf_asr.start_after_fresh_run(str(tmp_path / "clip.m4a"), "tiny", result, "fp", 1.0)

    assert started is False
    assert json.loads((state / "shadow-cf-asr-state.json").read_text(encoding="utf-8")) == expired
    assert not (state / "shadow-cf-asr.jsonl").exists()


def test_d5_ffmpeg_path_reports_truncated_from_240_frames(tmp_path, monkeypatch):
    """Video files decode through ffmpeg, which cannot count frames up front, so truncated is true from exactly 240 (docs/features/motion-probe.md)."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    clip = tmp_path / "clip240.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24",
                    "-frames:v", "240", "-pix_fmt", "yuv420p", str(clip)], check=True)

    response = handle_call("motion_stats", {"source": str(clip)})

    assert response["ok"], response
    data = response.get("data", response)
    assert data["frames"] == 240, data
    assert data["truncated"] is True, data


def test_d4_motion_probe_doc_states_the_real_test_count(tmp_path):
    """docs/features/motion-probe.md must give the number of tests the file really has."""
    words = {"seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    doc = (DOCS / "motion-probe.md").read_text(encoding="utf-8")
    stated = words[re.search(r"\((\w+) end-to-end tests", doc).group(1)]
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_motion_probe_e2e.py"],
                         cwd=REPO, capture_output=True, text=True).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert stated == collected, f"doc says {stated}, file has {collected}"


def test_d2_boil_formula_is_written_where_the_pointers_say(tmp_path):
    """The tool text and motion-probe.md point to 'the README formula' for boil; README must hold it."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "f_{t-1}" in readme and "f_{t+1}" in readme


def test_d7_documented_check_commands_say_their_feature_tests_skip():
    """While Workers AI is off, the feature tests named in each 'How to check it works' line skip; the doc must say so."""
    for name in ["watch.md", "shadow-cf-asr.md", "label-sounds.md"]:
        lines = [l for l in (DOCS / name).read_text(encoding="utf-8").splitlines() if "How to check it works" in l]
        assert lines and "skip" in lines[0].lower(), name
