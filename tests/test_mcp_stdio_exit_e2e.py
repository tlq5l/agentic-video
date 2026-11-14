"""End to end over the stdio MCP server: the client closes stdin while a background build runs.

The server stops the build at a checkpoint and exits with code 0 (no -6 abort), a reply sent before the close is delivered,
and a motion cache is either complete (meta.json present) or absent, with no temp files left behind.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.cache import MediaCache
from src.retrieval.visual import clip_missing_message, clip_weights_missing_message, hf_hub_cache

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
PING_ID = 999


def _make_clip(path, size, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=300)
    return str(path)


def _payload(reply):
    result = reply.get("result", {})
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    return json.loads("".join(item.get("text", "") for item in result.get("content", [])) or "{}")


class _Session:
    """One stdio server process: JSON-RPC lines in, replies out, as an MCP client drives it."""

    def __init__(self, home):
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "HF_HOME": str(home / "hf"), "LANG": "C.UTF-8"}
        if clip_weights_missing_message() is None:
            env["HF_HUB_CACHE"] = str(hf_hub_cache())
        self.proc = subprocess.Popen([sys.executable, "-m", "src.executor.mcp_server"], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.send(INITIALIZE)
        self.recv()
        self.send(INITIALIZED)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def recv(self):
        line = self.proc.stdout.readline()
        assert line, self.proc.stderr.read()[-2000:]
        return json.loads(line)

    def call(self, request_id, name, arguments):
        self.send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        return self.recv()

    def close_stdin(self):
        """Ask for a reply, then close stdin the way a client that is done does. Returns (exit code, stderr, replies read after the close)."""
        self.send({"jsonrpc": "2.0", "id": PING_ID, "method": "ping"})
        self.proc.stdin.close()
        code = self.proc.wait(timeout=180)
        replies = [json.loads(line) for line in self.proc.stdout.read().splitlines() if line.strip()]
        return code, self.proc.stderr.read(), replies


@pytest.mark.skipif(bool(clip_missing_message() or clip_weights_missing_message()),
                    reason="needs the [visual] extra and the CLIP weights")
def test_stdin_close_during_visual_build_exits_clean(tmp_path):
    clip = _make_clip(tmp_path / "visual.mp4", "320x240", 60)
    home = tmp_path / "home"
    home.mkdir()
    session = _Session(home)
    session.call(2, "build_visual_index", {"source": clip})
    reply = session.call(3, "build_visual_index", {"source": clip})
    for attempt in range(120):  # poll until the build has committed its first frames, so it is in flight at the close
        if (_payload(reply).get("frames_indexed") or 0) > 0 or _payload(reply).get("status") not in ("queued", "running", "in_progress"):
            break
        reply = session.call(4 + attempt, "build_visual_index", {"source": clip})
    code, err, replies = session.close_stdin()
    assert code == 0, err[-2000:]
    assert "terminate called" not in err and "Fatal Python error" not in err, err[-2000:]
    assert any(item.get("id") == PING_ID for item in replies), replies


def test_stdin_close_during_motion_build_leaves_complete_or_no_cache(tmp_path):
    from src.media.motion_probe import VERSION

    clip = _make_clip(tmp_path / "motion.mp4", "1920x1080", 20)
    home = tmp_path / "home"
    home.mkdir()
    session = _Session(home)
    session.call(2, "motion_open", {"source": clip})
    in_flight = _payload(session.call(3, "motion_open", {"source": clip})).get("status")  # usually still building here
    code, err, replies = session.close_stdin()
    assert code == 0 and "terminate called" not in err, (in_flight, err[-2000:])
    assert any(item.get("id") == PING_ID for item in replies), replies

    fingerprint = MediaCache.compute_fingerprint(clip)
    entries = [path for path in home.rglob(fingerprint) if path.is_dir()]
    entry = entries[0] / VERSION if entries else None
    names = sorted(p.name for p in entry.iterdir()) if entry is not None and entry.exists() else []
    assert not [name for name in names if name.startswith(".")], names  # no temp files
    if "meta.json" in names:
        meta = json.loads((entry / "meta.json").read_text())
        assert meta.get("fingerprint") == fingerprint and meta.get("version") == VERSION, meta
        assert {"rgb.npy", "meta.json"} <= set(names), names
    else:
        assert names == [], names  # no marker: the cache must be absent, not partly written
