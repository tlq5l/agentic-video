"""End to end over the stdio MCP server: a subprocess reads JSON-RPC lines on stdin and answers on stdout, as an MCP client drives it.

The every-tool test calls each tool once with valid small arguments on a 5 s clip and checks that each call gets its own JSON-RPC reply.
The protocol test checks two stdio rules: a notification (no id) gets no reply, and ping gets an empty result.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_pip_user_matrix_e2e import _good_args

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
VISUAL_TOOLS = ("build_visual_index", "search_visual", "count_occurrences")
VISUAL_HINT = "python -m pip install 'agentic-video[visual]'"
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("stdio") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=5",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True, timeout=120)
    return str(path)


def _talk(lines, tmp_path):
    """Send the lines to one stdio server process; return its reply lines in order."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "HF_HOME": str(home / "hf"), "LANG": "C.UTF-8"}
    payload = "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines)
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=payload, capture_output=True,
                          text=True, cwd=ROOT, env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Traceback" not in done.stderr, done.stderr[-2000:]
    return [json.loads(line) for line in done.stdout.splitlines()]


def test_every_tool_answers_over_stdio(clip, tmp_path):
    args = _good_args(clip)
    args["count_repetitions"] = {**args["count_repetitions"], "end_sec": 5}  # this clip is 5 s; the shared matrix clip is 6 s
    listing_id = 2
    calls = {name: 3 + index for index, name in enumerate(args)}
    lines = [INITIALIZE, INITIALIZED, {"jsonrpc": "2.0", "id": listing_id, "method": "tools/list"}]
    lines += [{"jsonrpc": "2.0", "id": calls[name], "method": "tools/call", "params": {"name": name, "arguments": args[name]}}
              for name in args]
    replies = {reply.get("id"): reply for reply in _talk(lines, tmp_path)}

    listed = {tool["name"] for tool in replies[listing_id]["result"]["tools"]}
    assert listed == set(args), f"tools without valid arguments here: {sorted(listed ^ set(args))}"
    for name, request_id in calls.items():
        reply = replies.get(request_id)
        assert reply is not None and "result" in reply, (name, reply)
        text = "".join(item.get("text", "") for item in reply["result"]["content"])
        assert "Traceback" not in text, (name, text[:500])
        if reply["result"].get("isError") and name in VISUAL_TOOLS:
            assert VISUAL_HINT in text, (name, text[:500])  # the documented message when the [visual] extra is missing


def test_notifications_get_no_reply_and_ping_gets_an_empty_result(tmp_path):
    lines = [INITIALIZE, INITIALIZED,
             {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 99, "reason": "test"}},
             {"jsonrpc": "2.0", "id": 2, "method": "ping"},
             {"jsonrpc": "2.0", "id": 3, "method": "no/such/method"},
             "not json"]
    replies = _talk(lines, tmp_path)
    assert [reply.get("id") for reply in replies] == [1, 2, 3, None], replies
    assert replies[1]["result"] == {}, replies[1]
    assert replies[2]["error"]["code"] == -32601, replies[2]
    assert replies[3]["error"]["code"] == -32700, replies[3]
