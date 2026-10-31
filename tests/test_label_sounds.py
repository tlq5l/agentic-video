"""End-to-end: drive the stdio MCP server's label_sounds tool on a generated video.

The live test reaches a paid cloud call (Cloudflare Workers AI), so it runs only when
CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or AGENTIC_VIDEO_CF_TOKEN_COMMAND) are set.
The other tests need no credentials and show the tool refuses before anything is sent.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLOUD_VARS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "AGENTIC_VIDEO_CF_TOKEN_COMMAND")
SECRET = "not-a-real-token-8c1f"


@pytest.fixture(scope="module")
def siren_video(tmp_path_factory):
    """6 s video: 3 s of a rising and falling siren-like tone, then 3 s of a steady ring tone."""
    path = tmp_path_factory.mktemp("label_sounds") / "clip.mp4"
    siren = "aevalsrc='0.5*sin(2*PI*(900+300*sin(2*PI*0.7*t))*t)':d=3:s=16000"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=6:r=10",
         "-f", "lavfi", "-i", siren, "-f", "lavfi", "-i", "sine=f=440:d=3:r=16000",
         "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a]", "-map", "0:v", "-map", "[a]", "-shortest", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def minute_tone(tmp_path_factory):
    """60 s of a steady tone (AAC) for the time-limit check."""
    path = tmp_path_factory.mktemp("label_sounds_long") / "tone.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=60:r=16000",
                    "-c:a", "aac", str(path)], check=True)
    return path


@pytest.fixture
def stand_in():
    """Local Workers AI stand-in for label_sounds. make(reply, delay_sec=0) returns its base URL;
    reply(number, labels) gives (HTTP status, body bytes) for the request with that number, counted from 0."""
    servers = []

    def make(reply, delay_sec=0.0):
        lock = threading.Lock()
        numbers = count()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    number = next(numbers)
                time.sleep(delay_sec)
                status, payload = reply(number, body["questions"]["sound"]["criteria"])
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:  # the caller gave up on a slow reply
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def well_formed(number, labels):
    """A valid Clef-omni reply: 'none' takes all the chance."""
    chances = {key: (1.0 if key == "none" else 0.0) for key in labels}
    return 200, json.dumps({"success": True, "result": {"answers": {"sound": {"probabilities": chances}},
                                                        "usage": {"input_tokens": 7}}}).encode()


def _cloud_env(base):
    return {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": SECRET, "AGENTIC_VIDEO_CF_API_BASE": base}


def _server(requests, env_overrides, drop=()):
    """Run the stdio server on the given JSON-RPC lines. Returns (replies by id, process return code)."""
    env = {key: value for key, value in os.environ.items() if key not in drop}
    env.update(env_overrides)
    lines = "".join(json.dumps(r) + "\n" for r in requests)
    done = subprocess.run(
        [sys.executable, "-m", "src.executor.mcp_server"], input=lines, capture_output=True, text=True,
        cwd=ROOT, env=env, timeout=180, check=False,
    )
    replies = {}
    for line in done.stdout.splitlines():
        reply = json.loads(line)
        replies[reply["id"]] = reply
    return replies, done.returncode


def _call(arguments, drop=(), env=None):
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "label_sounds", "arguments": arguments}}
    replies, code = _server([request], env or {}, drop)
    assert code == 0  # stdin closed: the server exits by itself, no process is left behind
    return replies[1]["result"]


def test_tool_is_listed_with_cloud_warning():
    replies, _ = _server([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], {})
    tool = next(t for t in replies[1]["result"]["tools"] if t["name"] == "label_sounds")
    assert "THIS SENDS THE AUDIO TO CLOUDFLARE" in tool["description"]


def test_refuses_without_credentials_before_sending_anything(siren_video):
    result = _call({"source": str(siren_video), "start_sec": 0, "end_sec": 6}, drop=CLOUD_VARS)
    assert result["isError"]
    text = result["content"][0]["text"]
    assert "CLOUDFLARE_ACCOUNT_ID" in text and "CLOUDFLARE_API_TOKEN" in text and "Nothing was sent" in text


@pytest.mark.parametrize("arguments, expected", [
    ({"start_sec": 0, "end_sec": 0.5}, "at least 1 s after start_sec"),
    ({"start_sec": 0, "end_sec": 6, "window_sec": 0.2}, "'window_sec' is below the minimum 1"),
    ({"start_sec": 0, "end_sec": 6, "labels": ["siren", "siren"]}, "distinct"),
    ({"start_sec": 0, "end_sec": 6, "labels": []}, "non-empty"),
    ({"start_sec": 0, "end_sec": 400}, "windows requested; at most 60"),
])
def test_bad_arguments_get_advice_not_a_cloud_call(siren_video, arguments, expected):
    result = _call({"source": str(siren_video), **arguments}, drop=CLOUD_VARS)
    assert result["isError"] and expected in result["content"][0]["text"]


def test_missing_audio_track_is_reported(tmp_path):
    silent = tmp_path / "no_audio.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x120:d=3:r=10", str(silent)], check=True)
    result = _call({"source": str(silent), "start_sec": 0, "end_sec": 3}, drop=CLOUD_VARS,
                   env={"CLOUDFLARE_ACCOUNT_ID": "x", "CLOUDFLARE_API_TOKEN": "x"})
    assert result["isError"] and "no audio track" in result["content"][0]["text"]


def test_bad_token_is_a_clear_cloud_error_and_never_echoed(siren_video):
    result = _call({"source": str(siren_video), "start_sec": 0, "end_sec": 3}, drop=CLOUD_VARS,
                   env={"CLOUDFLARE_ACCOUNT_ID": "0" * 32, "CLOUDFLARE_API_TOKEN": SECRET})
    text = result["content"][0]["text"]
    assert result["isError"] and "CloudCallError" in text and SECRET not in text


REDACTED_REPLIES = [
    pytest.param(lambda n, labels: (400, json.dumps({"errors": [{"message": f"token {SECRET} rejected"}]}).encode()),
                 id="http-400-body"),
    pytest.param(lambda n, labels: (200, json.dumps({"result": {"answers": {"sound": {"probabilities": {"o0": SECRET}}}}}).encode()),
                 id="malformed-chance"),
    pytest.param(lambda n, labels: (200, json.dumps({"result": {"answers": {"sound": {"probabilities": SECRET}}}}).encode()),
                 id="no-chances-echo"),
]


@pytest.mark.parametrize("reply", REDACTED_REPLIES)
def test_token_is_never_echoed_from_a_bad_reply(siren_video, stand_in, reply):
    base = stand_in(reply)
    result = _call({"source": str(siren_video), "start_sec": 0, "end_sec": 3}, drop=CLOUD_VARS, env=_cloud_env(base))
    assert result["isError"] and "CloudCallError" in result["content"][0]["text"]
    assert SECRET not in json.dumps(result)


@pytest.mark.slow
def test_slow_cloud_returns_finished_windows_within_the_limit(minute_tone, stand_in):
    """Every reply takes 10 s, so the 60 one-second windows cannot all finish in time; the call must stop and say where to resume."""
    base = stand_in(well_formed, delay_sec=10.0)
    began = time.monotonic()
    result = _call({"source": str(minute_tone), "start_sec": 0, "end_sec": 60, "window_sec": 1},
                   drop=CLOUD_VARS, env=_cloud_env(base))
    took = time.monotonic() - began
    data = result["structuredContent"]
    assert not result.get("isError") and took < 52, (took, result)  # Executor drops a call at 60 s
    assert data["complete"] is False and 4 <= len(data["windows"]) < 60
    assert data["next_start_sec"] == data["windows"][-1]["end_sec"] and "Stopped after" in data["note"]
    assert SECRET not in json.dumps(result)


@pytest.mark.skipif(not any(os.environ.get(v) for v in CLOUD_VARS) or not os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
                    reason="needs Cloudflare credentials (paid cloud call)")
def test_labels_two_windows_live(siren_video):
    labels = ["siren", "dog barking", "clapping", "car horn"]
    result = _call({"source": str(siren_video), "start_sec": 0, "end_sec": 6, "window_sec": 3, "labels": labels})
    assert not result.get("isError"), result
    data = result["structuredContent"]
    assert data["labels"] == labels and len(data["windows"]) == 2 and data["input_tokens"] > 0
    for window in data["windows"]:
        assert set(window["chances"]) == set(labels)
        assert abs(sum(window["chances"].values()) + window["none_chance"] - 1.0) < 0.05
        assert window["top_label"] in labels + [None]
    assert [(w["start_sec"], w["end_sec"]) for w in data["windows"]] == [(0.0, 3.0), (3.0, 6.0)]
    assert data["windows"][0]["chances"]["siren"] > data["windows"][0]["chances"]["dog barking"]
