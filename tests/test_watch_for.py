"""End-to-end: drive the stdio MCP server's watch_for tool on a generated video.

The server talks to a local stand-in for Cloudflare Workers AI (AGENTIC_VIDEO_CF_API_BASE) that answers
"yes" for a frame with a red centre and "no" otherwise, so the real frame extraction, wake rule, cooldown
and clean-up run with no credentials and no paid call. The live test reaches the real Clef model and runs
only when CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or AGENTIC_VIDEO_CF_TOKEN_COMMAND) are set.
"""

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLOUD_VARS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "AGENTIC_VIDEO_CF_TOKEN_COMMAND")
SECRET = "not-a-real-token-5d2e"


@pytest.fixture(scope="module")
def red_box_video(tmp_path_factory):
    """40 s blue video with a red box over the middle from 12 s to 30 s."""
    path = tmp_path_factory.mktemp("watch_for") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=40:r=10",
         "-vf", "drawbox=x=40:y=30:w=80:h=60:color=red:t=fill:enable='between(t,12,30)'", str(path)],
        check=True,
    )
    return path


def _start_stand_in(delay_for, override=None):
    """Local Workers AI stand-in; request number n (from 0) is answered after delay_for(n) seconds, or with the
    fixed (HTTP status, body bytes) `override` when one is given. Returns (server, base url, requests seen)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            number = len(seen)
            seen.append({"path": self.path, "auth": self.headers["Authorization"], "questions": body["questions"]})
            time.sleep(delay_for(number))
            if override is not None:
                status, reply = override
                self.send_response(status)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
                return
            image = Image.open(io.BytesIO(base64.b64decode(body["images"][0].split(",", 1)[1]))).convert("RGB")
            red, _, blue = image.getpixel((image.width // 2, image.height // 2))
            chance = 0.95 if red > 150 and blue < 100 else 0.02
            answers = {key: {"noul": chance if key == "q0" else 0.01} for key in body["questions"]}
            reply = json.dumps({"success": True, "result": {"answers": answers, "usage": {"input_tokens": 554}}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
            except OSError:  # the caller gave up on a slow reply
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}", seen


@pytest.fixture
def stand_in(slow_stand_in):
    """Local Workers AI stand-in that answers at once. Returns (base url, requests seen)."""
    return slow_stand_in(lambda n: 0)


@pytest.fixture
def slow_stand_in():
    """Factory for a stand-in with slow replies: slow_stand_in(delay_for, override=None) -> (base url, requests seen)."""
    servers = []

    def make(delay_for, override=None):
        server, base, seen = _start_stand_in(delay_for, override)
        servers.append(server)
        return base, seen

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def _call(arguments, env=None, drop=()):
    """One watch_for call through the stdio server. Returns (tool result, server return code)."""
    environ = {key: value for key, value in os.environ.items() if key not in drop}
    environ.update(env or {})
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "watch_for", "arguments": arguments}}
    done = subprocess.run(
        [sys.executable, "-m", "src.executor.mcp_server"], input=json.dumps(request) + "\n", capture_output=True,
        text=True, cwd=ROOT, env=environ, timeout=180, check=False,
    )
    return json.loads(done.stdout.splitlines()[-1])["result"], done.returncode


def test_tool_is_listed_with_cloud_warning():
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=json.dumps(request) + "\n",
                          capture_output=True, text=True, cwd=ROOT, timeout=60, check=False)
    tool = next(t for t in json.loads(done.stdout.splitlines()[-1])["result"]["tools"] if t["name"] == "watch_for")
    assert "THIS SENDS VIDEO FRAMES TO CLOUDFLARE" in tool["description"]


def test_wakes_once_when_the_red_box_stays_then_cooldown_holds_the_rest(red_box_video, stand_in, tmp_path):
    base, seen = stand_in
    temp = tmp_path / "temp"
    env = {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": SECRET, "AGENTIC_VIDEO_CF_API_BASE": base,
           "AGENTIC_VIDEO_TEMP_DIR": str(temp)}
    result, code = _call({"source": str(red_box_video), "conditions": ["Is a red box visible?", "Is a dog visible?"]},
                         env=env, drop=("AGENTIC_VIDEO_CF_TOKEN_COMMAND",))
    assert code == 0 and not result.get("isError"), result  # stdin closed: the server exits by itself
    data = result["structuredContent"]
    # samples every 5 s from 0 to 35 (a frame at 40 s would be past the last one): red at 15, 20, 25, 30
    assert data["samples"] == 8 and data["complete"] and data["next_start_sec"] is None and data["scanned_to_sec"] == 35.0
    assert [(w["condition_index"], w["first_seen_sec"], w["time_sec"]) for w in data["wakes"]] == [(0, 15.0, 20.0)]
    red, dog = data["conditions"]
    assert (red["wakes"], red["dropped_by_cooldown"], dog["wakes"]) == (1, 2, 0)
    assert red["max_chance"] == 0.95 and dog["max_chance"] == 0.01
    assert data["input_tokens"] == 8 * 554
    assert len(seen) == 8 and seen[0]["auth"] == f"Bearer {SECRET}" and len(seen[0]["questions"]) == 2
    assert SECRET not in json.dumps(result)
    assert not list(temp.glob("watch_*"))  # sampled frames are deleted

    # the 2 samples in a row rule, with no cooldown: every sample from 20 s on wakes
    result, _ = _call({"source": str(red_box_video), "conditions": ["Is a red box visible?"], "cooldown_sec": 0,
                       "start_sec": 10, "end_sec": 32}, env=env, drop=("AGENTIC_VIDEO_CF_TOKEN_COMMAND",))
    assert [w["time_sec"] for w in result["structuredContent"]["wakes"]] == [20.0, 25.0, 30.0]


@pytest.mark.parametrize("delay_for, label", [
    (lambda n: 15.0, "every reply takes 15 s"),
    (lambda n: 0.0 if n < 6 else 25.0, "the API hangs after 6 good replies"),
])
@pytest.mark.slow
def test_slow_cloud_returns_finished_samples_within_about_50_s(red_box_video, slow_stand_in, tmp_path, delay_for, label):
    """Real constants: no new sample after 40 s, requests cut at 48 s, so the call stays inside Executor's 60 s limit."""
    base, seen = slow_stand_in(delay_for)
    temp = tmp_path / "temp"
    env = {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": SECRET, "AGENTIC_VIDEO_CF_API_BASE": base,
           "AGENTIC_VIDEO_TEMP_DIR": str(temp)}
    began = time.monotonic()
    result, code = _call({"source": str(red_box_video), "conditions": ["Is a red box visible?"], "interval_sec": 1},
                         env=env, drop=("AGENTIC_VIDEO_CF_TOKEN_COMMAND",))
    took = time.monotonic() - began
    assert code == 0 and took < 52, (label, took)  # Executor drops a call at 60 s
    data = result["structuredContent"] if not result.get("isError") else None
    assert data and data["complete"] is False and data["samples"] >= 2, (label, result)
    assert data["next_start_sec"] is not None and data["next_start_sec"] <= data["scanned_to_sec"] and "note" in data
    assert data["input_tokens"] == data["samples"] * 554
    assert SECRET not in json.dumps(result) and not list(temp.glob("watch_*"))


@pytest.mark.parametrize("arguments, expected", [
    ({"conditions": []}, "conditions"),
    ({"conditions": ["a", "a"]}, "distinct"),
    ({"conditions": ["a"], "interval_sec": 0.2}, "'interval_sec' is below the minimum 1"),
    ({"conditions": ["a"], "cooldown_sec": -1}, "'cooldown_sec' is below the minimum 0"),
    ({"conditions": ["a"], "start_sec": 50}, "start_sec must be from 0 up to before end_sec"),
    ({"conditions": ["a"], "start_sec": 0, "end_sec": 3}, "fewer than consecutive=2"),
])
def test_bad_arguments_get_advice_not_a_cloud_call(red_box_video, arguments, expected):
    result, _ = _call({"source": str(red_box_video), **arguments}, drop=CLOUD_VARS)
    assert result["isError"] and expected in result["content"][0]["text"]


def test_refuses_without_credentials_before_sending_anything(red_box_video):
    result, _ = _call({"source": str(red_box_video), "conditions": ["Is a red box visible?"]}, drop=CLOUD_VARS)
    text = result["content"][0]["text"]
    assert result["isError"] and "CLOUDFLARE_ACCOUNT_ID" in text and "CLOUDFLARE_API_TOKEN" in text and "Nothing was sent" in text


@pytest.mark.skipif(not os.environ.get("CLOUDFLARE_ACCOUNT_ID") or not any(os.environ.get(v) for v in CLOUD_VARS[1:]),
                    reason="needs Cloudflare credentials (paid cloud call)")
def test_live_clef_wakes_on_the_red_box(red_box_video):
    result, _ = _call({"source": str(red_box_video), "conditions": ["Is there a red rectangle in the middle of the picture?"]})
    assert not result.get("isError"), result
    wakes = result["structuredContent"]["wakes"]
    assert len(wakes) == 1 and 15.0 <= wakes[0]["time_sec"] <= 25.0


@pytest.mark.parametrize("status, reply", [
    (400, json.dumps({"errors": [{"message": f"token {SECRET} rejected"}]}).encode()),
    (200, json.dumps({"result": {"answers": {"q0": {"noul": SECRET}}}}).encode()),
], ids=["http-400-body", "malformed-chance"])
def test_token_is_never_echoed_from_a_bad_reply(red_box_video, slow_stand_in, status, reply):
    base, _ = slow_stand_in(lambda n: 0, override=(status, reply))
    env = {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": SECRET, "AGENTIC_VIDEO_CF_API_BASE": base}
    result, code = _call({"source": str(red_box_video), "conditions": ["Is a red box visible?"]},
                         env=env, drop=("AGENTIC_VIDEO_CF_TOKEN_COMMAND",))
    assert code == 0 and result["isError"] and "CloudCallError" in result["content"][0]["text"]
    assert SECRET not in json.dumps(result)
