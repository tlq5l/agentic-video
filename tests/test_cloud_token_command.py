"""End-to-end: watch_for and label_sounds share one Cloudflare token in one MCP server process.

A local stand-in answers for Cloudflare Workers AI (AGENTIC_VIDEO_CF_API_BASE). The token command is a script that
prints a banner line, then the token, and appends one line to a counter file each time it runs.
"""

import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLOUD_VARS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "AGENTIC_VIDEO_CF_TOKEN_COMMAND")
TOKEN = "fake-token-abc"


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("cloud_token") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=6:r=10",
         "-f", "lavfi", "-i", "sine=f=440:d=6:r=16000", "-shortest", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def stand_in():
    """Answers both tools' requests; refuses (401) any token other than TOKEN. Yields (base url, requests seen)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(self.headers["Authorization"])
            if self.headers["Authorization"] != f"Bearer {TOKEN}":
                self.send_response(401)
                self.end_headers()
                return
            if "images" in body:
                answers = {key: {"noul": 0.01} for key in body["questions"]}
            else:
                criteria = body["questions"]["sound"]["criteria"]
                answers = {"sound": {"probabilities": {key: 1.0 / len(criteria) for key in criteria}}}
            reply = json.dumps({"success": True, "result": {"answers": answers, "usage": {"input_tokens": 1}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()


def _token_command(tmp_path, printed):
    """Script printing a banner line, then `printed`, and logging one line per run. Returns (command, counter file)."""
    counter = tmp_path / "runs.log"
    script = tmp_path / "cf-token"
    script.write_text(f"#!/bin/sh\necho run >> '{counter}'\necho 'banner line'\necho '{printed}'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script), counter


def _server(calls, base, token_command=None, token=None):
    """Run one stdio server on (tool, arguments) calls in order. Returns (results in order, return code)."""
    env = {key: value for key, value in os.environ.items() if key not in CLOUD_VARS}
    env.update({"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "AGENTIC_VIDEO_CF_API_BASE": base})
    if token_command:
        env["AGENTIC_VIDEO_CF_TOKEN_COMMAND"] = token_command
    if token:
        env["CLOUDFLARE_API_TOKEN"] = token
    lines = "".join(
        json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}) + "\n"
        for i, (name, args) in enumerate(calls, 1)
    )
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=lines, capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=180, check=False)
    replies = {json.loads(line)["id"]: json.loads(line)["result"] for line in done.stdout.splitlines()}
    return [replies[i] for i in range(1, len(calls) + 1)], done.returncode


def test_both_tools_share_one_token_run_and_use_its_last_line(clip, stand_in, tmp_path):
    base, seen = stand_in
    command, counter = _token_command(tmp_path, TOKEN)
    watch = ("watch_for", {"source": str(clip), "conditions": ["Is a dog visible?"], "interval_sec": 2})
    label = ("label_sounds", {"source": str(clip), "start_sec": 0, "end_sec": 3, "labels": ["siren"]})
    results, code = _server([watch, label, watch], base, token_command=command)
    assert code == 0
    assert [bool(r.get("isError")) for r in results] == [False, False, False], results
    assert len(counter.read_text().splitlines()) == 1  # one run for three calls across both tools
    assert seen and set(seen) == {f"Bearer {TOKEN}"}
    assert "banner" not in json.dumps(results)


def test_a_token_that_cannot_go_in_a_header_is_refused_and_never_echoed(clip, stand_in):
    base, seen = stand_in
    secret = "bad token\nsecret-line"
    label = ("label_sounds", {"source": str(clip), "start_sec": 0, "end_sec": 3, "labels": ["siren"]})
    watch = ("watch_for", {"source": str(clip), "conditions": ["Is a dog visible?"], "interval_sec": 2})
    results, _ = _server([label, watch], base, token=secret)
    for result in results:
        text = json.dumps(result)
        assert result["isError"] and "secret-line" not in text and "bad token" not in text
    assert seen == []  # nothing was sent
