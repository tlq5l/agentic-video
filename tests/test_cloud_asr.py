"""Cloud transcription (src/transcript/cloud_asr.py): the switch, the pieces, the cache, no local fallback, the end date.

No real network: a local stand-in answers for Workers AI (AGENTIC_VIDEO_CF_API_BASE) and records each request.
"""

import json
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.types import TranscriptResult, TranscriptSegment
from src.media.probe import probe_video
from src.core.errors import ASRModelError
from src.executor.mcp_server import handle_call
from src.media import cloudflare
from src.transcript import asr, cloud_asr, shadow_cf_asr

SECRET = "not-a-real-token-cloud-asr-4f2a"
ACCOUNT = "b" * 32
WHISPER_PATH = "/ai/run/@cf/openai/whisper-large-v3-turbo"
HOST = socket.gethostname()
REAL_DEFAULT_MODEL_SIZE = asr.default_model_size  # captured before conftest swaps it for each test
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _no_local(*args, **kwargs):
    raise AssertionError("local ASR ran")


@pytest.fixture
def stand_in():
    stand = SimpleNamespace(base=None, seen=[], fail=False)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            stand.seen.append({"path": self.path, "body": body})
            if stand.fail or not self.path.endswith(WHISPER_PATH):
                status = 500  # the error body echoes the Authorization header, to check the token never leaks
                reply = {"success": False, "errors": [{"message": "boom " + str(self.headers.get("Authorization"))}],
                         "messages": [], "result": None}
            else:
                status = 200
                reply = {"success": True, "errors": [], "messages": [], "result": {
                    "text": " stand-in piece ", "word_count": 2,
                    "transcription_info": {"language": "en", "duration": 2.0},
                    "segments": [{"start": 0.5, "end": 1.5, "text": " stand-in piece "}],
                }}
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


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """3 s of tone with an audio stream: with 2 s pieces it makes two pieces (2 s and 1 s)."""
    path = tmp_path_factory.mktemp("clip") / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", str(path)],
                   check=True, timeout=120)
    return path


@pytest.fixture
def cloud(monkeypatch, tmp_path, stand_in):
    """Cloud mode on this host, fake credentials, the stand-in as Workers AI, a private cache, 2 s pieces."""
    monkeypatch.setenv("AGENTIC_VIDEO_CF_API_BASE", stand_in.base)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", SECRET)
    monkeypatch.delenv("AGENTIC_VIDEO_CF_TOKEN_COMMAND", raising=False)
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, HOST)
    monkeypatch.setattr(cloud_asr, "PIECE_SEC", 2)
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(asr, "_installed_whisper_models", lambda: [])
    return SimpleNamespace(stand=stand_in, cache=tmp_path / "cache")


def test_switch_names_this_host_only(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = tmp_path / "config" / "agentic-video" / "cloud-asr.json"
    settings.parent.mkdir(parents=True)
    monkeypatch.delenv(cloud_asr.SWITCH_ENV, raising=False)
    assert cloud_asr.selected() is False
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, "1")
    assert cloud_asr.selected() is False
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, f"some-other-host,{HOST}")
    assert cloud_asr.selected() is True
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, "")
    settings.write_text(json.dumps({"hosts": ["some-other-host"]}))
    assert cloud_asr.selected() is False
    settings.write_text(json.dumps({"hosts": [HOST]}))
    assert cloud_asr.selected() is True
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, "some-other-host")  # a set env value wins; the file is not read
    assert cloud_asr.selected() is False
    monkeypatch.setenv(cloud_asr.SWITCH_ENV, "")
    settings.write_text("{not json")
    assert cloud_asr.selected() is False


def test_default_model_is_cloud_in_cloud_mode(cloud, monkeypatch):
    monkeypatch.setattr(asr, "default_model_size", REAL_DEFAULT_MODEL_SIZE)
    assert asr.default_model_size() == "cloud"


def test_cloud_run_stitches_pieces_and_caches(cloud, clip, monkeypatch):
    monkeypatch.setattr(asr, "_run_memory_capped_worker", _no_local)
    result = asr.transcribe_video(str(clip), model_size="tiny")
    assert [s.start_sec for s in result.segments] == [0.5, 2.5]  # piece 2 is offset by its 2 s start
    assert [s.id for s in result.segments] == [1, 2]
    assert result.language == "en"
    assert "Cloudflare Workers AI" in result.notice
    sent = cloud.stand.seen
    assert len(sent) == 2
    assert all(request["body"]["vad_filter"] is False and "language" not in request["body"] for request in sent)
    again = asr.transcribe_video(str(clip), model_size="tiny")  # cache hit: no new request
    assert len(cloud.stand.seen) == 2
    assert [s.start_sec for s in again.segments] == [0.5, 2.5]
    assert any("transcript_asr_cloud" in str(path) for path in cloud.cache.rglob("*") if path.is_file())


def test_cloud_failure_never_runs_a_local_model(cloud, clip, monkeypatch):
    cloud.stand.fail = True
    monkeypatch.setattr(asr, "_run_memory_capped_worker", _no_local)
    with pytest.raises(ASRModelError) as info:
        asr.transcribe_video(str(clip), model_size="turbo")
    assert "no local model ran" in str(info.value)
    assert SECRET not in str(info.value)


def test_after_the_end_nothing_is_sent(cloud, clip, monkeypatch):
    monkeypatch.setattr(cloud_asr, "ENDS_AT", PAST)
    monkeypatch.setattr(asr._BACKGROUND_TRANSCRIPTION_SLOT, "start",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a slot started")))
    with pytest.raises(ASRModelError, match="ended"):
        asr.transcribe_video(str(clip), model_size="tiny")
    with pytest.raises(ASRModelError, match="ended"):
        asr.request_transcription(str(clip), "tiny")
    assert cloud.stand.seen == []


def test_after_the_end_cached_transcript_is_still_served(cloud, clip, monkeypatch):
    asr.transcribe_video(str(clip), model_size="tiny")  # made before the end
    sent = len(cloud.stand.seen)
    monkeypatch.setattr(cloud_asr, "ENDS_AT", PAST)
    data = asr.request_transcription(str(clip), "tiny")
    assert len(data["segments"]) == 2
    assert len(cloud.stand.seen) == sent


def test_mcp_transcribe_runs_in_the_cloud(cloud, clip):
    deadline = time.monotonic() + 120
    while True:
        response = handle_call("transcribe", {"source": str(clip), "model_size": "tiny"})
        assert response["ok"], response
        data = response["data"]
        if "transcription_status" not in data:
            break
        assert data["transcription_status"] not in ("failed", "start_failed", "orphaned", "unverified"), data
        assert time.monotonic() < deadline, data
        time.sleep(1)
    for thread in threading.enumerate():
        if thread.name == "agentic-video-asr":
            thread.join(timeout=60)
    assert "Cloudflare Workers AI" in data["notice"]
    assert [s["start_sec"] for s in data["segments"]] == [0.5, 2.5]


def test_workers_ai_switch_stays_off(cloud):
    assert cloudflare.WORKERS_AI_ON is False
    with pytest.raises(cloudflare.CloudCredentialsError):
        cloudflare.credentials("watch_for", "frames")
    assert cloud.stand.seen == []


def test_shadow_trial_never_starts_for_cloud_runs(cloud, monkeypatch):
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOST)
    assert shadow_cf_asr.start_after_fresh_run(source_path="x.mp4", model_size="cloud", result=None,
                                               fingerprint="f", local_wall_sec=0.0) is False
    assert cloud.stand.seen == []


def _seed_local(clip, model):
    """Write an old local ASR transcript for the clip, as a pre-cloud run would have left it."""
    meta = probe_video(str(clip), allow_audio_only=True)
    seg = TranscriptSegment(id=1, start_sec=0.0, end_sec=1.0, text="old local line", confidence=None)
    res = TranscriptResult(
        source_path=str(clip), language="en", duration_sec=meta.duration_sec, segment_count=1,
        segments=[seg], has_speech=True,
        notice="Note: Non-speech audio (music, sound effects, alarms) is NOT transcribed.",
        confidence=None, uncertainty="",
    )
    MediaCache().save_json(meta.fingerprint, f"transcript_asr_{model}",
                           asr._transcript_cache_payload(res, meta.fingerprint, None, model))


def test_cloud_mode_transcribe_serves_old_local_transcript_after_the_end(cloud, clip, monkeypatch):
    _seed_local(clip, "turbo")
    monkeypatch.setattr(cloud_asr, "ENDS_AT", PAST)
    res = asr.transcribe_video(str(clip))
    assert res.segments[0].text == "old local line"
    assert cloud.stand.seen == []


def test_cloud_mode_request_serves_old_local_transcript_without_a_run(cloud, clip, monkeypatch):
    _seed_local(clip, "tiny")
    monkeypatch.setattr(asr._BACKGROUND_TRANSCRIPTION_SLOT, "start",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a slot started")))
    data = asr.request_transcription(str(clip), "tiny")
    assert data["segments"][0]["text"] == "old local line"
    assert cloud.stand.seen == []


def test_shadow_trial_never_starts_for_cloud_runs_with_workers_ai_on(cloud, monkeypatch):
    monkeypatch.setenv(shadow_cf_asr.SWITCH_ENV, HOST)
    monkeypatch.setattr(cloudflare, "WORKERS_AI_ON", True)
    monkeypatch.setattr(shadow_cf_asr, "_trial_open", lambda *a, **k: True)
    started = []
    monkeypatch.setattr(shadow_cf_asr.subprocess, "Popen", lambda *a, **k: started.append(a))
    fresh = SimpleNamespace(duration_sec=3.0, language="en", has_speech=True, segments=[])
    assert shadow_cf_asr.start_after_fresh_run(source_path="x.mp4", model_size="cloud", result=fresh,
                                               fingerprint="f", local_wall_sec=0.0) is False
    assert started == []
    assert cloud.stand.seen == []
