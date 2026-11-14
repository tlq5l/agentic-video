"""End to end over the stdio MCP server: a subprocess reads JSON-RPC lines on stdin and answers on stdout, as an MCP client drives it.

The every-tool test calls each tool once with valid small arguments on a 5 s clip, lets the jobs those calls start settle, and checks that each call gets its own JSON-RPC reply.
The protocol test checks two stdio rules: a notification (no id) gets no reply, and ping gets an empty result.
"""

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.retrieval.visual import clip_missing_message, clip_weights_missing_message, hf_hub_cache
from test_pip_user_matrix_e2e import _good_args

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
VISUAL_TOOLS = ("build_visual_index", "search_visual", "count_occurrences")
VISUAL_HINT = "python -m pip install 'agentic-video[visual]'"
WEIGHTS_HINT = "needs the CLIP model weights"
# The gap a count_occurrences job ends on when the visual extra or the CLIP weights are missing: its later poll reports it as an error.
END_HINT = {"extra_missing": VISUAL_HINT, "weights_missing": WEIGHTS_HINT}
# A job still working: the same tool call polls it, as the CLI does (mcp_server._settled_call settles on these too).
JOB_STATUSES = ("queued", "running", "in_progress", "busy")
# The status a background tool reports once its job has finished (visual.py, occurrence.py, repetition.py, motion_probe.py).
DONE_STATUS = {"build_visual_index": "complete", "search_visual": "complete", "count_occurrences": "complete",
               "count_repetitions": "complete", "motion_open": "ready"}
# Statuses that end a job with a failure: the wait fails on them, with the reply printed.
FAILED_STATUSES = ("error", "failed")
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
        if name == "count_occurrences" and not result.get("isError"):
            assert status == "in_progress", (name, text[:500])  # a description starts a job; the missing-extra error comes on a later poll
        else:
            assert result.get("isError") and VISUAL_HINT in text, (name, text[:500])
    elif state == "weights_missing":
        if name == "count_occurrences" and not result.get("isError"):
            assert status == "in_progress", (name, text[:500])  # a description starts a job; the weights gap comes on a later poll
        else:
            assert result.get("isError") and WEIGHTS_HINT in text and shlex.quote(sys.executable) in text, (name, text[:500])
    else:
        # weights present: a queued or running reply only starts the job; _assert_visual_done checks the finished reply after the wait
        assert not result.get("isError") and status not in FAILED_STATUSES, (name, text[:500])


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


def _status(result):
    status = (result.get("structuredContent") or {}).get("status")
    if status is None:
        try:
            status = json.loads(result["content"][0]["text"]).get("status")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            status = None
    return status


class _Server:
    """One stdio server process kept open: requests go in as the test makes them, replies come back by id, and stdin closes only on close()."""

    def __init__(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        self.stderr_path = tmp_path / "server-stderr.txt"
        self.stderr = self.stderr_path.open("w")
        self.proc = subprocess.Popen([sys.executable, "-m", "src.executor.mcp_server"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, cwd=ROOT, env=_server_env(home))
        self.replies = queue.Queue()
        self.by_id = {}
        threading.Thread(target=self._read_replies, daemon=True).start()

    def _read_replies(self):
        for line in self.proc.stdout:
            self.replies.put(json.loads(line))
        self.replies.put(None)  # stdout closed

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def reply(self, request_id, timeout_sec=600):
        deadline = time.monotonic() + timeout_sec
        while request_id not in self.by_id:
            try:
                item = self.replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise AssertionError(f"no reply to request {request_id} within {timeout_sec} s") from None
            assert item is not None, "the server closed stdout before replying"
            self.by_id[item.get("id")] = item
        return self.by_id[request_id]

    def close(self, timeout_sec=600):
        """Close stdin, the server's end of file; return its exit code and its stderr text."""
        self.proc.stdin.close()
        returncode = self.proc.wait(timeout=timeout_sec)
        self.stderr.close()
        return returncode, self.stderr_path.read_text()

    def abort(self):
        """Stop the server if a failed assertion left it running."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.stderr.close()


def _payload(result):
    """A tool reply's fields: its structuredContent, else the JSON in its text."""
    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    try:
        return json.loads(result["content"][0]["text"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {}


# Tools that finish with a result of their own shape and no status (asr.py, search.py): the field that proves the finished result.
DONE_SHAPE = {"transcribe": "segments", "search_transcript": "hits"}


def _has_done_rule(name):
    return name in DONE_STATUS or name in DONE_SHAPE


def _is_done(name, result):
    """True when a reply is the tool's finished answer: its done status, or its finished shape with no status."""
    if name in DONE_STATUS:
        return _status(result) == DONE_STATUS[name]
    return _status(result) is None and isinstance(_payload(result).get(DONE_SHAPE[name]), list)


def _still_running(name, result, state):
    """True when a call's reply starts or keeps a job, so the wait polls it until it is done."""
    if _status(result) in JOB_STATUSES:
        return True
    return state == "weights_present" and name in VISUAL_TOOLS and not result.get("isError") and not _is_done(name, result)


def _assert_visual_done(name, result, text):
    """With weights present, a visual tool's finished reply: its done status and the fields of its finished shape."""
    payload = _payload(result)
    assert not result.get("isError") and _is_done(name, result), (name, text[:500])
    if name == "build_visual_index":
        assert payload.get("frames_indexed", 0) > 0, (name, text[:500])
    elif name == "search_visual":
        assert isinstance(payload.get("hits"), list) and payload.get("index_status") == "complete", (name, text[:500])
    elif name == "count_occurrences":
        assert "count" in payload and isinstance(payload.get("spans"), list), (name, text[:500])


def _settle_jobs(server, pending, next_id, state, timeout_sec=600):
    """Repeat each pending tool call (name -> request) until its job reports its done status. The same call polls the same job.
    Returns the finished reply of each call by name. Only a done status counts: an error, a missing status, a tool with no
    done rule, or a timeout fails the test, with the reply printed. Other statuses (queued, partial, index_required...) keep polling."""
    for name in pending:
        assert _has_done_rule(name), f"no done rule for {name}"
    deadline = time.monotonic() + timeout_sec
    finished = {}
    while pending:
        assert time.monotonic() < deadline, f"jobs not done after {timeout_sec} s: {sorted(pending)}"
        time.sleep(1)
        for name, request in list(pending.items()):
            server.send({**request, "id": next_id})
            reply = server.reply(next_id)
            next_id += 1
            assert "result" in reply, (name, reply)
            result = reply["result"]
            text = "".join(item.get("text", "") for item in result.get("content", []))
            if _is_done(name, result):
                finished[name] = result
                del pending[name]
            elif state in END_HINT and name == "count_occurrences" and END_HINT[state] in text:
                # The one expected end without the extra or the weights: the later poll reports that gap as an error.
                assert result.get("isError") or _status(result) in FAILED_STATUSES, (name, text[:500])
                finished[name] = result
                del pending[name]
            else:
                assert not result.get("isError") and _status(result) is not None and _status(result) not in FAILED_STATUSES, (name, text[:500])
    return finished


def test_every_tool_answers_over_stdio(clip, tmp_path):
    args = _good_args(clip)
    args["count_repetitions"] = {**args["count_repetitions"], "end_sec": 5}  # this clip is 5 s; the shared matrix clip is 6 s
    listing_id = 2
    calls = {name: 3 + index for index, name in enumerate(args)}
    requests = {name: {"jsonrpc": "2.0", "id": calls[name], "method": "tools/call", "params": {"name": name, "arguments": args[name]}}
                for name in args}
    server = _Server(tmp_path)
    try:
        for message in (INITIALIZE, INITIALIZED, {"jsonrpc": "2.0", "id": listing_id, "method": "tools/list"}, *requests.values()):
            server.send(message)
        replies = {request_id: server.reply(request_id) for request_id in (listing_id, *calls.values())}
        state = _visual_state()

        listed = {tool["name"] for tool in replies[listing_id]["result"]["tools"]}
        assert listed == set(args), f"tools without valid arguments here: {sorted(listed ^ set(args))}"
        for name, request_id in calls.items():
            reply = replies[request_id]
            assert "result" in reply, (name, reply)
            text = "".join(item.get("text", "") for item in reply["result"]["content"])
            assert "Traceback" not in text, (name, text[:500])
            if name in VISUAL_TOOLS:
                _assert_visual_reply(name, reply, text, state)

        # Stdin closes only after the jobs the calls started have settled, so this checks that each tool answers, not what the server does at shutdown.
        pending = {name: requests[name] for name, request_id in calls.items() if _still_running(name, replies[request_id]["result"], state)}
        finished = _settle_jobs(server, pending, max(calls.values()) + 1, state)
        if state == "weights_present":
            for name in VISUAL_TOOLS:
                result = finished.get(name, replies[calls[name]]["result"])
                _assert_visual_done(name, result, "".join(item.get("text", "") for item in result.get("content", [])))
        returncode, stderr = server.close()
    finally:
        server.abort()
    assert returncode == 0, stderr[-2000:]
    assert "Traceback" not in stderr, stderr[-2000:]

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
