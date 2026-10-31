"""End to end: the shadow Cloudflare trial behind handle_call("transcribe") on a 20 s clip of real speech.

A local stand-in answers for Cloudflare Workers AI (AGENTIC_VIDEO_CF_API_BASE) and records each request, so no audio
leaves the machine here. The only test that reaches real Cloudflare is the live one at the bottom, which runs only when
credentials, this host's switch and an open trial are all present.
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
from types import SimpleNamespace

import pytest

from src.core.config import Config
from src.executor.mcp_server import handle_call
from src.media import cloudflare
from src.retrieval.engine import AgenticVideoEngine
from src.transcript import asr, shadow_cf_asr

# 20 s of real spoken English. Made once with macOS say (the same setup as test_transcribe_background's speech test),
# then padded with silence to 20 s: say -o speech.aiff "<three sentences>"; ffmpeg -i speech.aiff -af apad -t 20 ...
# The bench fixture is no use here: its audio is tones, and its speech comes from a subtitle sidecar, not from ASR.
SPEECH_CLIP = Path(__file__).resolve().parents[1] / "fixtures" / "shadow_speech_20s.m4a"
SECRET = "not-a-real-token-shadow-7c1e"
WHISPER_PATH = "/ai/run/@cf/openai/whisper-large-v3-turbo"
PIECE_SEC = 8  # a 20 s clip makes 3 pieces (8, 8 and 4 s), so the stitched offsets are checked
HOSTNAME = socket.gethostname()


@pytest.fixture(scope="module")
def speech_clip():
    """The committed speech clip; nonempty, so a missing or empty file fails here rather than as a silent empty transcript."""
    assert SPEECH_CLIP.stat().st_size > 0
    return SPEECH_CLIP


@pytest.fixture
def stand_in():
    """Workers AI stand-in: every Whisper request gets the same one-segment reply. `seen` lists the requests;
    `on_request(count)`, when set by a test, runs after each request is recorded and before it is answered."""
    stand = SimpleNamespace(base=None, seen=[], on_request=None)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            stand.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            if stand.on_request is not None:
                stand.on_request(len(stand.seen))
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
    stand.base = f"http://127.0.0.1:{server.server_port}"
    yield stand
    server.shutdown()
    server.server_close()


@pytest.fixture
def trial_env(monkeypatch, tmp_path, stand_in):
    """Fake credentials, the stand-in as Cloudflare, a short piece length, a private state dir, and one cache dir per call."""
    monkeypatch.setenv("AGENTIC_VIDEO_CF_API_BASE", stand_in.base)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", SECRET)
    monkeypatch.delenv("AGENTIC_VIDEO_CF_TOKEN_COMMAND", raising=False)
    monkeypatch.setenv("AGENTIC_VIDEO_SHADOW_PIECE_SEC", str(PIECE_SEC))
    monkeypatch.delenv(shadow_cf_asr.PARALLEL_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(shadow_cf_asr.SWITCH_ENV, raising=False)
    monkeypatch.setattr(asr, "_installed_whisper_models", lambda: [])
    caches = iter(range(100))

    def new_cache():
        monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / f"cache-{next(caches)}")

    new_cache()
    return {"seen": stand_in.seen, "stand": stand_in, "state": tmp_path / "state" / "agentic-video",
            "new_cache": new_cache}


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


def _write_trial_state(state: Path, started_at, count: int, ended: bool = False):
    state.mkdir(parents=True, exist_ok=True)
    (state / "shadow-cf-asr-state.json").write_text(
        json.dumps({"started_at": started_at, "count": count, "ended": ended}), encoding="utf-8")


def test_switch_on_sends_three_pieces_logs_both_and_the_reply_matches_switch_off(speech_clip, trial_env, monkeypatch):
    seen, state = trial_env["seen"], trial_env["state"]
    off = _transcribe(speech_clip)
    _wait_for_background_worker()
    assert seen == [] and not state.exists()  # switch unset: nothing sent, nothing logged
    assert any(seg["text"].strip() for seg in off["segments"])  # the clip has speech, so the comparison means something

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
    assert record["local_has_speech"] is True and record["local_text"].strip()
    assert record["local_text"] == " ".join(s["text"].strip() for s in off["segments"] if s["text"].strip())
    assert record["cf_text"] == "stand-in piece stand-in piece stand-in piece"
    assert [(s["piece_index"], s["start_sec"], s["end_sec"]) for s in record["cf_segments"]] == [
        (0, 0.5, 1.5), (1, 8.5, 9.5), (2, 16.5, 17.5),
    ]
    assert '"audio":' not in json.dumps(record) and SECRET not in json.dumps(record)  # no raw audio, no token
    assert (state / "shadow-cf-asr.jsonl").stat().st_mode & 0o777 == 0o600
    assert state.stat().st_mode & 0o777 == 0o700

    assert len(seen) == 3
    for request in seen:
        assert request["path"].endswith(WHISPER_PATH) and request["auth"] == f"Bearer {SECRET}"
        assert request["body"]["vad_filter"] is False and request["body"]["language"] == record["language"]
        audio = base64.b64decode(request["body"]["audio"])
        assert audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")  # MP3, not raw audio
    trial = json.loads((state / "shadow-cf-asr-state.json").read_text(encoding="utf-8"))
    assert trial["count"] == 1 and trial["ended"] is False and trial["started_at"]


def test_a_synchronous_fresh_run_is_shadowed_too(speech_clip, trial_env, monkeypatch):
    """AgenticVideoEngine.transcribe with an explicit timeout is a fresh run with no background slot; it is shadowed."""
    seen, state = trial_env["seen"], trial_env["state"]
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOSTNAME)
    result = AgenticVideoEngine(str(speech_clip)).transcribe(model_size="tiny", timeout_sec=240)
    assert result.has_speech is True
    (record,) = _wait_for_job_record(state)
    assert record["status"] == "ok" and record["pieces"] == 3
    assert record["local_text"] == " ".join(s.text.strip() for s in result.segments if s.text.strip())
    assert len(seen) == 3


def test_a_synchronous_run_returns_before_the_shadow_credentials_resolve(speech_clip, trial_env, monkeypatch, tmp_path):
    """The shadow token command waits for a gate that this test opens only after the synchronous call has returned.
    Before the fix the hook ran the command on the caller's thread, so the command had finished (`done` existed) when
    the call returned; now the call returns while the command is still waiting. The command gives up after about 25 s."""
    seen, state = trial_env["seen"], trial_env["state"]
    done, gate = tmp_path / "command-done", tmp_path / "gate"
    script = tmp_path / "token.sh"
    script.write_text(
        "#!/bin/sh\n"
        "i=0\n"
        f"while [ ! -f {gate} ] && [ $i -lt 250 ]; do sleep 0.1; i=$((i+1)); done\n"
        f"touch {done}\n"
        f"[ -f {gate} ] || exit 1\n"
        f"echo {SECRET}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    monkeypatch.setenv("AGENTIC_VIDEO_CF_TOKEN_COMMAND", str(script))
    monkeypatch.setattr(cloudflare, "_command_token", None)  # a token kept by an earlier test would skip the command
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOSTNAME)

    result = AgenticVideoEngine(str(speech_clip)).transcribe(model_size="tiny", timeout_sec=240)
    assert result.has_speech is True
    assert not done.exists()  # the transcript came back while the shadow token command was still waiting
    gate.touch()
    (record,) = _wait_for_job_record(state)
    assert record["status"] == "ok" and record["pieces"] == 3
    assert len(seen) == 3


@pytest.mark.parametrize("switch", ["1", "some-other-host"])
def test_a_switch_that_does_not_name_this_host_sends_nothing(speech_clip, trial_env, monkeypatch, switch):
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, switch)
    _transcribe(speech_clip)
    _wait_for_background_worker()
    time.sleep(3)  # a child started now would reach the stand-in within this time
    assert trial_env["seen"] == [] and _log_records(trial_env["state"]) == []


def test_an_expired_trial_sends_nothing_and_writes_trial_ended_once(speech_clip, trial_env, monkeypatch):
    state = trial_env["state"]
    expired_at = time.time() - (shadow_cf_asr.TRIAL_DAYS + 1) * 86400
    _write_trial_state(state, expired_at, count=3)
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


def test_an_expiry_during_a_job_stops_its_remaining_pieces(speech_clip, trial_env, monkeypatch):
    """Pieces run one at a time here, so the trial can expire after the first piece is sent. The later pieces and
    their retries are refused, the job still logs once, and trial_ended is written once."""
    stand, state = trial_env["stand"], trial_env["state"]

    def expire_after_the_first_request(count):
        if count == 1:
            _write_trial_state(state, time.time() - (shadow_cf_asr.TRIAL_DAYS + 1) * 86400, count=1)

    stand.on_request = expire_after_the_first_request
    monkeypatch.setenv(shadow_cf_asr.PARALLEL_ENV, "1")
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOSTNAME)
    _transcribe(speech_clip)
    _wait_for_background_worker()
    (record,) = _wait_for_job_record(state)

    assert len(trial_env["seen"]) == 1  # only the first piece reached the stand-in
    assert record["status"] == "partial" and [e["piece_index"] for e in record["piece_errors"]] == [1, 2]
    assert [r["status"] for r in _log_records(state)].count("trial_ended") == 1


def _live_allowed():
    """Live test gate: credentials, this host named by the switch, and an open trial. Read-only."""
    has_credentials = bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID")) and bool(
        os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("AGENTIC_VIDEO_CF_TOKEN_COMMAND"))
    return has_credentials and shadow_cf_asr._switch_names_this_host() and shadow_cf_asr.trial_open_now()


@pytest.mark.skipif(not _live_allowed(), reason="needs Cloudflare credentials, this host in AGENTIC_VIDEO_SHADOW_CF_ASR "
                    "and an open trial; the test box has none, so it never sends audio there")
def test_live_shadow_job_reaches_real_whisper_turbo_for_a_short_clip(speech_clip, monkeypatch, tmp_path):
    """Live: sends 2 s of real speech to Cloudflare Workers AI through run_job, the guarded path (switch, credentials,
    trial). Skipped unless all of those are set, so the test box never sends audio."""
    monkeypatch.delenv("AGENTIC_VIDEO_CF_API_BASE", raising=False)  # the real state dir, so the count is the one the gate read
    clip = tmp_path / "two-seconds.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(speech_clip), "-t", "2", "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "libmp3lame", str(clip)], check=True)
    job = {"source_path": str(clip), "fingerprint": "live-test", "duration_sec": 2.0, "language": "en",
           "model_size": "tiny", "engine": "faster-whisper", "local_has_speech": True, "local_wall_sec": 0.0,
           "local_segments": []}
    record = shadow_cf_asr.run_job(job)
    assert record is not None and record["status"] == "ok", record
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    assert not (token and token in json.dumps(record))
