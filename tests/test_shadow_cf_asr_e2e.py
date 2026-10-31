"""End to end: the shadow Cloudflare trial behind handle_call("transcribe") on a generated 20 s clip.

A local stand-in answers for Cloudflare Workers AI (AGENTIC_VIDEO_CF_API_BASE) and records each request, so no audio
leaves the machine here. The only test that reaches real Cloudflare is the live one at the bottom, which is skipped
without credentials.
"""

import base64
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.core.config import Config
from src.executor.mcp_server import handle_call
from src.media import cloudflare
from src.transcript import asr, shadow_cf_asr

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_bench_60s.mp4"
SECRET = "not-a-real-token-shadow-7c1e"
WHISPER_PATH = "/ai/run/@cf/openai/whisper-large-v3-turbo"
PIECE_SEC = 8  # a 20 s clip makes 3 pieces (8, 8 and 4 s), so the stitched offsets are checked
HOSTNAME = socket.gethostname()


@pytest.fixture(scope="module")
def speech_clip(tmp_path_factory):
    """A 20 s generated clip cut from the bench fixture's speech; no subtitle sidecar, so ASR runs."""
    path = tmp_path_factory.mktemp("shadow") / "speech.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "5", "-i", str(FIXTURE), "-t", "20", "-c", "copy", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def stand_in():
    """Workers AI stand-in: every Whisper request gets the same one-segment reply. Returns (base url, requests seen)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            if self.path.endswith(WHISPER_PATH):
                status = 200
                reply = {"success": True, "errors": [], "messages": [], "result": {
                    "text": " stand-in piece ", "word_count": 2,
                    "transcription_info": {"language": "en", "duration": 8.0},
                    "segments": [{"start": 0.5, "end": 1.5, "text": " stand-in piece "}],
                }}
            else:
                status = 404
                reply = {"success": False, "errors": [{"message": "unknown model"}], "messages": [], "result": None}
            data = json.dumps(reply).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def trial_env(monkeypatch, tmp_path, stand_in):
    """Fake credentials, the stand-in as Cloudflare, a short piece length, a private state dir, and one cache dir per call."""
    base, seen = stand_in
    monkeypatch.setenv("AGENTIC_VIDEO_CF_API_BASE", base)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", SECRET)
    monkeypatch.delenv("AGENTIC_VIDEO_CF_TOKEN_COMMAND", raising=False)
    monkeypatch.setenv("AGENTIC_VIDEO_SHADOW_PIECE_SEC", str(PIECE_SEC))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(shadow_cf_asr.SWITCH_ENV, raising=False)
    monkeypatch.setattr(asr, "_installed_whisper_models", lambda: [])
    caches = iter(range(100))

    def new_cache():
        monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / f"cache-{next(caches)}")

    new_cache()
    return {"seen": seen, "state": tmp_path / "state" / "agentic-video", "new_cache": new_cache}


def _transcribe(source):
    """Call transcribe until the transcript is ready (no transcription_status key), as an agent would poll."""
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        response = handle_call("transcribe", {"source": str(source), "model_size": "tiny"})
        assert response["ok"], response
        data = response["data"]
        if data.get("transcription_status") in ("failed", "start_failed", "orphaned", "unverified"):
            raise AssertionError(data)
        if "transcription_status" not in data:
            return data
        time.sleep(2)
    raise AssertionError("the transcript was not ready in time")


def _wait_for_background_worker():
    """The background thread runs the shadow hook after the cache is saved; join it so the hook has returned."""
    for thread in threading.enumerate():
        if thread.name == "agentic-video-asr":
            thread.join(timeout=60)


def _log_records(state: Path):
    path = state / "shadow-cf-asr.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wait_for_job_record(state: Path):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        records = [record for record in _log_records(state) if record["status"] != "trial_ended"]
        if records:
            return records
        time.sleep(1)
    raise AssertionError("the shadow job never wrote its log line")


def test_switch_on_sends_three_pieces_logs_both_and_the_reply_matches_switch_off(speech_clip, trial_env, monkeypatch):
    seen, state = trial_env["seen"], trial_env["state"]
    off = _transcribe(speech_clip)
    _wait_for_background_worker()
    assert seen == [] and not state.exists()  # switch unset: nothing sent, nothing logged

    trial_env["new_cache"]()  # a fresh local run, not a cache hit
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOSTNAME)
    on = _transcribe(speech_clip)
    assert on == off  # the user's reply is the same with the trial on
    _wait_for_background_worker()
    (record,) = _wait_for_job_record(state)

    assert record["status"] == "ok" and record["piece_errors"] == []
    assert record["pieces"] == 3 and record["piece_sec"] == PIECE_SEC
    assert record["local_model"] == "tiny" and record["local_engine"] == "faster-whisper"
    assert record["cf_model"] == shadow_cf_asr.MODEL and record["source_path"] == str(speech_clip.resolve())
    assert record["source_fingerprint"]
    assert record["local_text"] == " ".join(s["text"].strip() for s in off["segments"] if s["text"].strip())
    assert record["cf_text"] == "stand-in piece stand-in piece stand-in piece"
    assert [(s["piece_index"], s["start_sec"], s["end_sec"]) for s in record["cf_segments"]] == [
        (0, 0.5, 1.5), (1, 8.5, 9.5), (2, 16.5, 17.5),
    ]
    assert '"audio":' not in json.dumps(record) and SECRET not in json.dumps(record)  # no raw audio, no token
    assert (state / "shadow-cf-asr.jsonl").stat().st_mode & 0o777 == 0o600
    assert state.stat().st_mode & 0o777 == 0o700

    requests = trial_env["seen"]
    assert len(requests) == 3
    for request in requests:
        assert request["path"].endswith(WHISPER_PATH) and request["auth"] == f"Bearer {SECRET}"
        assert request["body"]["vad_filter"] is False and request["body"]["language"] == record["language"]
        audio = base64.b64decode(request["body"]["audio"])
        assert audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")  # MP3, not raw audio
    trial = json.loads((state / "shadow-cf-asr-state.json").read_text(encoding="utf-8"))
    assert trial["count"] == 1 and trial["ended"] is False and trial["started_at"]


@pytest.mark.parametrize("switch", ["1", "some-other-host"])
def test_a_switch_that_does_not_name_this_host_sends_nothing(speech_clip, trial_env, monkeypatch, switch):
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, switch)
    _transcribe(speech_clip)
    _wait_for_background_worker()
    time.sleep(3)  # a child started now would reach the stand-in within this time
    assert trial_env["seen"] == [] and _log_records(trial_env["state"]) == []


def test_an_expired_trial_sends_nothing_and_writes_trial_ended_once(speech_clip, trial_env, monkeypatch):
    state = trial_env["state"]
    state.mkdir(parents=True, exist_ok=True)
    expired_at = time.time() - (shadow_cf_asr.TRIAL_DAYS + 1) * 86400
    (state / "shadow-cf-asr-state.json").write_text(
        json.dumps({"started_at": expired_at, "count": 3, "ended": False}), encoding="utf-8")
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOSTNAME)

    _transcribe(speech_clip)
    _wait_for_background_worker()
    trial_env["new_cache"]()  # a second fresh run after the trial has ended
    _transcribe(speech_clip)
    _wait_for_background_worker()
    time.sleep(2)

    assert trial_env["seen"] == []
    records = _log_records(state)
    assert [r["status"] for r in records] == ["trial_ended"] and records[0]["reason"].startswith("more than 14 days")
    trial = json.loads((state / "shadow-cf-asr-state.json").read_text(encoding="utf-8"))
    assert trial["ended"] is True and trial["count"] == 3


def _has_live_credentials():
    return bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID")) and bool(
        os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("AGENTIC_VIDEO_CF_TOKEN_COMMAND"))


@pytest.mark.skipif(not _has_live_credentials(), reason="needs Cloudflare credentials; the test box has none")
def test_live_whisper_turbo_answers_for_a_short_clip(speech_clip, monkeypatch, tmp_path):
    """Live: sends 2 s of real audio to Cloudflare Workers AI. Skipped without credentials, so the test box never sends it."""
    monkeypatch.delenv("AGENTIC_VIDEO_CF_API_BASE", raising=False)
    clip = tmp_path / "two-seconds.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(speech_clip), "-t", "2", "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "libmp3lame", str(clip)], check=True)
    account, token = cloudflare.credentials("Shadow ASR test", "audio")
    body = json.dumps({"audio": base64.b64encode(clip.read_bytes()).decode("ascii"), "vad_filter": False}).encode()
    reply = json.loads(cloudflare.run_model(shadow_cf_asr.MODEL, account, token, body, 120.0))
    assert reply["success"] is True and isinstance(reply["result"]["text"], str)
    assert token not in json.dumps(reply)
