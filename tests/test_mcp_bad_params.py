"""A tools/call whose params is null (or any non-object) must not kill the stdio server.

The bad call gets a JSON-RPC -32602 error and the server keeps answering later
requests, whether they arrive as separate lines or inside one batch array.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CALL_NULL_PARAMS = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None}
PING = {"jsonrpc": "2.0", "id": 2, "method": "ping"}


def _run(lines, tmp_path):
    """Send the lines to one stdio server process; return its stdout lines."""
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "LANG": "C.UTF-8"}
    payload = "".join(json.dumps(line) + "\n" for line in lines)
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=payload, capture_output=True,
                          text=True, cwd=ROOT, env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Traceback" not in done.stderr, done.stderr[-2000:]
    return done.stdout.splitlines()


def test_null_params_call_gets_error_and_server_keeps_answering(tmp_path):
    out = _run([CALL_NULL_PARAMS, PING], tmp_path)
    assert len(out) == 2, out
    first, second = (json.loads(line) for line in out)
    assert first["id"] == 1 and first["error"]["code"] == -32602, first
    assert second["id"] == 2 and second["result"] == {}, second


def test_batch_with_null_params_call_answers_every_id(tmp_path):
    out = _run([[CALL_NULL_PARAMS, PING]], tmp_path)
    assert len(out) == 1, out
    replies = json.loads(out[0])
    assert isinstance(replies, list), replies
    by_id = {reply.get("id"): reply for reply in replies}
    assert by_id[1]["error"]["code"] == -32602, replies
    assert by_id[2]["result"] == {}, replies
