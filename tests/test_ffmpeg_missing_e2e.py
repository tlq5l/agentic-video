"""End to end: with ffmpeg and ffprobe off PATH, each entry point says ffmpeg is missing in one line, with the install command per OS, and fails without a traceback.

Each test runs the real command in a subprocess whose PATH is an empty folder, the way a user without ffmpeg would run it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_bench_60s.mp4"
INSTALLS = ("brew install ffmpeg", "sudo apt install ffmpeg", "winget install Gyan.FFmpeg")

# The config adds these folders to PATH on macOS when they hold both binaries, so a stripped PATH cannot hide them.
_FALLBACK_FOLDERS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="the stripped-PATH subprocess needs a POSIX layout"),
    pytest.mark.skipif(any((folder / "ffmpeg").exists() and (folder / "ffprobe").exists() for folder in _FALLBACK_FOLDERS),
                       reason="a fallback folder holds ffmpeg, which the config adds back to PATH"),
]


def _run_without_ffmpeg(args, tmp_path):
    empty = tmp_path / "empty-path"
    empty.mkdir(exist_ok=True)
    env = {"PATH": str(empty), "HOME": str(tmp_path / "home")}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_cli_commands_name_the_missing_ffmpeg_and_exit_non_zero(tmp_path):
    for command in ("cadence", "check"):
        proc = _run_without_ffmpeg(["-m", "src.loop", command, str(FIXTURE)], tmp_path)
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert "Traceback" not in proc.stderr, proc.stderr
        lines = [line for line in proc.stderr.splitlines() if "ffmpeg is missing from PATH" in line]
        assert len(lines) == 1, proc.stderr
        assert all(install in lines[0] for install in INSTALLS), lines[0]


def test_mcp_tool_call_names_the_missing_ffmpeg(tmp_path):
    proc = _run_without_ffmpeg(["-m", "src.executor.mcp_server", "call", "probe_video", json.dumps({"source": str(FIXTURE)})], tmp_path)
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    text = "".join(item.get("text", "") for item in reply.get("content", []))
    assert proc.returncode == 1 and reply.get("isError") is True, proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "ffmpeg is missing from PATH" in text and all(install in text for install in INSTALLS), text
