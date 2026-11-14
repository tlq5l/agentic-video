"""End to end over the stdio MCP server: a subprocess reads JSON-RPC lines on stdin and answers on stdout, as an MCP client drives it.

The every-tool test calls each tool once with valid small arguments on a 5 s clip and checks that each call gets its own JSON-RPC reply.
The protocol test checks two stdio rules: a notification (no id) gets no reply, and ping gets an empty result.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.retrieval.visual import clip_missing_message, clip_weights_missing_message, hf_hub_cache
from test_pip_user_matrix_e2e import _good_args

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
VISUAL_TOOLS = ("build_visual_index", "search_visual", "count_occurrences")
VISUAL_HINT = "python -m pip install 'agentic-video[visual]'"
WEIGHTS_HINT = "needs the CLIP model weights"
# Statuses a visual tool may return without an error on a first call (visual.py and the job record it reads from).
VISUAL_LIVE_STATUSES = ("queued", "running", "in_progress", "busy", "partial", "complete", "index_required")
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


def _server_env(home):
    """The stdio server's environment. Its CLIP weights come from the test host's own Hugging Face cache when that holds them,
    else from an empty cache, so the server sees the same weights state the test decides on (see _visual_state)."""
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "HF_HOME": str(home / "hf"), "LANG": "C.UTF-8"}
    if clip_weights_missing_message() is None:
        env["HF_HUB_CACHE"] = str(hf_hub_cache())
    return env


def _visual_state():
    """What the visual tools must answer: the [visual] extra is missing, the CLIP weights are missing, or both are present."""
    if clip_missing_message():
        return "extra_missing"
    if clip_weights_missing_message():
        return "weights_missing"
    return "weights_present"


def _assert_visual_reply(name, reply, text, state):
    result = reply["result"]
    status = (result.get("structuredContent") or {}).get("status")
    if state == "extra_missing":
        assert result.get("isError") and VISUAL_HINT in text, (name, text[:500])
    elif state == "weights_missing":
        if name == "count_occurrences" and not result.get("isError"):
            assert status == "in_progress", (name, text[:500])  # a description starts a job; the weights gap comes on a later poll
        else:
            assert result.get("isError") and WEIGHTS_HINT in text and shlex.quote(sys.executable) in text, (name, text[:500])
    else:
        assert not result.get("isError") and status in VISUAL_LIVE_STATUSES, (name, text[:500])


def _talk(lines, tmp_path):
    """Send the lines to one stdio server process; return its reply lines in order."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = _server_env(home)
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
    state = _visual_state()

    listed = {tool["name"] for tool in replies[listing_id]["result"]["tools"]}
    assert listed == set(args), f"tools without valid arguments here: {sorted(listed ^ set(args))}"
    for name, request_id in calls.items():
        reply = replies.get(request_id)
        assert reply is not None and "result" in reply, (name, reply)
        text = "".join(item.get("text", "") for item in reply["result"]["content"])
        assert "Traceback" not in text, (name, text[:500])
        if name in VISUAL_TOOLS:
            _assert_visual_reply(name, reply, text, state)


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


def test_initialize_echoes_supported_version_else_newest(tmp_path):
    # The server supports only 2024-11-05: 2025-03-26 requires servers to accept JSON-RPC batches, and this one answers
    # an array with Invalid Request. So 2024-11-05 is also the newest supported version here.
    client = {"capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", **client}},
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2099-01-01", **client}},
        {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"protocolVersion": ["2024-11-05"]}},
        {"jsonrpc": "2.0", "id": 4, "method": "initialize", "params": "not an object"},
        {"jsonrpc": "2.0", "id": 5, "method": "ping"},
    ]
    replies = {reply.get("id"): reply for reply in _talk(lines, tmp_path)}
    for request_id in (1, 2, 3, 4):
        assert replies[request_id]["result"]["protocolVersion"] == "2024-11-05", replies[request_id]
    assert replies[5]["result"] == {}, replies[5]


def test_count_repetitions_over_stdio_notes_period_bounds(clip, tmp_path):
    lines = [INITIALIZE, INITIALIZED, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "count_repetitions", "arguments": {"source": clip, "end_sec": 5, "min_period_sec": 1e-9, "max_period_sec": 1e300}}}]
    reply = {item.get("id"): item for item in _talk(lines, tmp_path)}[2]
    text = "".join(item.get("text", "") for item in reply["result"]["content"])
    assert "min_period_sec set to" in text and "max_period_sec set to" in text, text[:500]
