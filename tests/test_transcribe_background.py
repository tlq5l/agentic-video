"""MCP transcript cache fallback and bounded background ASR behavior."""

import json
import os
import shutil
import signal
import sys
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.errors import ASRModelError, MediaCorruptError
from src.core.types import TranscriptResult, TranscriptSegment, VideoMetadata
from src.executor.mcp_server import handle_call
from src.transcript import asr
from src.transcript.search import search_transcript
from src.adapters.model_adapter import AgenticVideoModelAdapter


FINGERPRINT = "cached-transcript-media"


def _metadata(source_path: str) -> VideoMetadata:
    return VideoMetadata(
        source_path=source_path,
        duration_sec=120.0,
        width=640,
        height=360,
        fps=30.0,
        format_name="mp4",
        size_bytes=123,
        has_audio=True,
        audio_codec="aac",
        video_codec="h264",
        fingerprint=FINGERPRINT,
    )


def _result(source_path: str, text: str = "Cached phrase") -> TranscriptResult:
    segment = TranscriptSegment(id=1, start_sec=2.0, end_sec=3.0, text=text)
    return TranscriptResult(
        source_path=source_path,
        language="en",
        duration_sec=120.0,
        segment_count=1,
        segments=[segment],
        has_speech=True,
        confidence=0.9,
        uncertainty="Model-dependent alignment.",
    )


def _seed_cache(source_path: str, model_size: str, text: str = "Cached phrase") -> TranscriptResult:
    result = _result(source_path, text)
    payload = asr._transcript_cache_payload(result, FINGERPRINT, None, model_size)
    MediaCache().save_json(FINGERPRINT, f"transcript_asr_{model_size}", payload)
    return result


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    # Pretend no Whisper weights are installed, so a machine that has turbo
    # does not swap in a different model than the one these tests request.
    monkeypatch.setattr(asr, "_installed_whisper_models", lambda: [])
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test media")
    monkeypatch.setattr(asr, "probe_video", lambda path, **kwargs: _metadata(path))
    return source


class _RecordingSlot:
    def __init__(self, state="started", error=None):
        self.state = state
        self.error = error
        self.calls = []

    def start(self, source_path, model_size):
        self.calls.append((source_path, model_size))
        return self.state, self.error


def test_mcp_model_cache_hit_keeps_existing_output_shape(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    cached = _seed_cache(str(source), "small", "Small model phrase")
    slot = _RecordingSlot()
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    response = handle_call("transcribe", {"source": str(source), "model_size": "small"})

    assert response == {"ok": True, "data": cached.to_dict()}
    assert slot.calls == []


def test_mcp_returns_cached_model_and_starts_requested_model(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    cached = _seed_cache(str(source), "tiny", "Tiny model phrase")
    slot = _RecordingSlot("started")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    response = handle_call("transcribe", {"source": str(source), "model_size": "small"})

    assert response["ok"] is True
    assert response["data"]["segments"] == cached.to_dict()["segments"]
    assert response["data"]["requested_model_size"] == "small"
    assert response["data"]["cached_model_size"] == "tiny"
    assert response["data"]["requested_model_cached"] is False
    assert response["data"]["transcription_status"] == "started"
    assert response["data"]["retry_after_seconds"] == 15
    assert "cached tiny transcript" in response["data"]["transcription_message"]
    assert slot.calls == [(str(source), "small")]


def test_mcp_cache_miss_returns_retry_status(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    slot = _RecordingSlot("started")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    response = handle_call("transcribe", {"source": str(source), "model_size": "small"})

    assert response == {
        "ok": True,
        "data": {
            "status": "in_progress",
            "requested_model_size": "small",
            "requested_model_cached": False,
            "transcription_status": "started",
            "retry_after_seconds": 15,
            "transcription_message": "The small transcription started in the background. Retry in about 15 seconds.",
        },
    }
    assert slot.calls == [(str(source), "small")]


def test_search_cache_miss_returns_status_without_running_asr_inline(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    slot = _RecordingSlot("started")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    def synchronous_asr_is_a_bug(*args, **kwargs):
        raise AssertionError("search must not run ASR in the MCP call")

    monkeypatch.setattr(asr, "transcribe_video", synchronous_asr_is_a_bug)
    response = handle_call(
        "search_transcript", {"source": str(source), "query": "cached phrase"}
    )

    assert response["ok"] is True
    assert response["data"]["status"] == "in_progress"
    assert response["data"]["transcription_status"] == "started"
    assert response["data"]["retry_after_seconds"] == 15
    assert slot.calls == [(str(source), "tiny")]


def test_search_empty_fallback_keeps_pending_status(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    empty = _result(str(source))
    empty.segments = []
    empty.segment_count = 0
    empty.has_speech = False
    MediaCache().save_json(FINGERPRINT, "transcript_asr_base",
                           asr._transcript_cache_payload(empty, FINGERPRINT, None, "base"))
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", _RecordingSlot("started"))

    result = search_transcript(str(source), "hello")

    assert result["hits"] == []
    assert result["total"] == 0
    assert result["transcription_status"] == "started"
    assert result["retry_after_seconds"] == 15
    assert search_transcript(str(source), "the")["retry_after_seconds"] == 15


def test_model_adapter_keeps_fallback_hits_and_pending_status(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_cache(str(source), "base")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", _RecordingSlot("started"))
    adapter = AgenticVideoModelAdapter(str(source))
    monkeypatch.setattr(adapter.engine, "probe", lambda: _metadata(str(source)))
    monkeypatch.setattr(adapter.engine, "overview", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("visual fallback started")))

    search = search_transcript(str(source), "cached phrase")
    plan = adapter.formulate_inspection_plan("cached phrase")
    result = adapter.execute_agentic_workflow("cached phrase")

    assert search["hits"][0]["text"] == "Cached phrase"
    assert search["transcription_status"] == "started"
    assert search["retry_after_seconds"] == 15
    for output in (plan, result):
        assert output["is_transcript_driven"] is True
        assert output["transcript_matches"] == search["hits"]
        assert output["candidate_windows"] == [(0.5, 5.0, "Transcript match: 'Cached phrase'")]
        assert output["transcription_status"] == "started"
        assert output["retry_after_seconds"] == 15
        assert output["requested_model_size"] == "tiny"
        assert output["cached_model_size"] == "base"
        assert output["requested_model_cached"] is False
        assert "cached base transcript" in output["transcription_message"]


def test_model_adapter_keeps_no_hits_pending_status(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_cache(str(source), "base")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", _RecordingSlot("started"))
    adapter = AgenticVideoModelAdapter(str(source))
    monkeypatch.setattr(adapter.engine, "probe", lambda: _metadata(str(source)))

    plan = adapter.formulate_inspection_plan("absent phrase")

    assert plan["transcript_matches"] == []
    assert plan["candidate_windows"] == []
    assert plan["transcription_status"] == "started"
    assert plan["retry_after_seconds"] == 15
    assert plan["requested_model_size"] == "tiny"
    assert "cached base transcript" in plan["transcription_message"]


def test_model_adapter_cached_final_stays_plain_hits(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_cache(str(source), "tiny")
    slot = _RecordingSlot()
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)
    adapter = AgenticVideoModelAdapter(str(source))
    monkeypatch.setattr(adapter.engine, "probe", lambda: _metadata(str(source)))

    search = search_transcript(str(source), "cached phrase")
    plan = adapter.formulate_inspection_plan("cached phrase")

    assert isinstance(search, list)
    assert plan["transcript_matches"] == [hit.to_dict() for hit in search]
    assert plan["is_transcript_driven"] is True
    assert "transcription_status" not in plan
    assert "retry_after_seconds" not in plan
    assert slot.calls == []


@pytest.mark.parametrize("state", ["started", "in_progress", "busy"])
def test_model_adapter_propagates_pending_transcript_without_visual_fallback(monkeypatch, state):
    adapter = AgenticVideoModelAdapter("unused.mp4")
    meta = _metadata("unused.mp4")
    monkeypatch.setattr(adapter.engine, "probe", lambda: meta)
    monkeypatch.setattr(adapter.engine, "search_transcript", lambda *a, **k: {
        "status": "in_progress" if state == "started" else state,
        "transcription_status": state, "retry_after_seconds": 15,
        "transcription_message": "Retry soon.",
    })
    monkeypatch.setattr(adapter.engine, "overview", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("visual fallback started")))

    plan = adapter.formulate_inspection_plan("What was said?")
    result = adapter.execute_agentic_workflow("What was said?")

    assert plan["candidate_windows"] == []
    assert result["transcription_status"] == state
    assert result["retry_after_seconds"] == 15


def test_search_joins_existing_background_transcription(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_transcribe(source_path, model_size, allow_long_run):
        calls.append((source_path, model_size))
        started.set()
        release.wait(timeout=3)
        return _result(source_path)

    monkeypatch.setattr(asr, "transcribe_video", fake_transcribe)
    slot = asr._BackgroundTranscriptionSlot()
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    try:
        transcription = handle_call(
            "transcribe", {"source": str(source), "model_size": "small"}
        )
        assert transcription["data"]["status"] == "in_progress"
        assert started.wait(timeout=1)

        search = handle_call(
            "search_transcript", {"source": str(source), "query": "cached phrase"}
        )

        assert search["ok"] is True
        assert search["data"]["status"] == "in_progress"
        assert search["data"]["transcription_status"] == "in_progress"
        assert calls == [(str(source), "small")]
    finally:
        release.set()


def test_background_lock_prevents_second_process_worker(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    started = threading.Event()
    release = threading.Event()

    def fake_transcribe(source_path, model_size, allow_long_run):
        started.set()
        release.wait(timeout=5)
        return _result(source_path)

    monkeypatch.setattr(asr, "transcribe_video", fake_transcribe)
    slot = asr._BackgroundTranscriptionSlot()
    assert slot.start(str(source), "small")[0] == "started"
    assert started.wait(timeout=1)

    child_script = r'''
import json, sys
from pathlib import Path
from src.core.config import Config
from src.core.types import VideoMetadata
from src.transcript import asr
Config.BASE_CACHE_DIR = Path(sys.argv[1])
source = str(Path(sys.argv[2]).resolve())
def unexpected_worker(*args, **kwargs):
    raise RuntimeError("the second process started an ASR worker")
asr.transcribe_video = unexpected_worker
asr.probe_video = lambda path, **kwargs: VideoMetadata(
    source_path=path, duration_sec=120.0, width=640, height=360, fps=30.0,
    format_name="mp4", size_bytes=123, has_audio=True, audio_codec="aac",
    video_codec="h264", fingerprint="cached-transcript-media")
print(json.dumps(asr.request_transcription(source, "small")))
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-c", child_script, str(tmp_path / "cache"), str(source)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        response = json.loads(completed.stdout)
        assert response["status"] == "in_progress"
        assert response["transcription_status"] == "in_progress"
    finally:
        release.set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock inheritance test")
def test_worker_keeps_lock_after_server_crash(tmp_path):
    cache_dir = tmp_path / "cache"
    marker = tmp_path / "worker-pid"
    supervisor_script = '''
import pathlib, sys
from src.transcript import asr
lock, _, error = asr._acquire_background_process_lock(
    pathlib.Path(sys.argv[1]), "video.mp4", "small")
assert lock is not None, error
asr._ASR_WORKER_LOCK.lock_file = lock
worker = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
asr._run_memory_capped_worker([sys.executable, "-c", worker, sys.argv[2]],
                              timeout=35, memory_limit_bytes=asr.ASR_WORKER_MEMORY_LIMIT_BYTES)
'''
    supervisor = subprocess.Popen(
        [sys.executable, "-c", supervisor_script, str(cache_dir), str(marker)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    worker_pid = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not marker.exists():
            if supervisor.poll() is not None:
                pytest.fail(f"supervisor exited: {supervisor.stderr.read().decode()}")
            time.sleep(0.05)
        assert marker.exists()
        worker_pid = int(marker.read_text())
        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=3)

        lock, owner, error = asr._acquire_background_process_lock(cache_dir, "other.mp4", "tiny")
        assert lock is None and error is None
        assert owner["source_path"] == "video.mp4"
    finally:
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=3)
        supervisor.stderr.close()

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        lock, _, error = asr._acquire_background_process_lock(cache_dir, "other.mp4", "tiny")
        if lock is not None:
            asr._release_background_process_lock(lock)
            break
        assert error is None
        time.sleep(0.05)
    else:
        pytest.fail("stale worker lock was not released after its process died")


def test_mcp_reports_a_failed_background_attempt(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    slot = _RecordingSlot("failed", "ASRModelError: worker exceeded its memory limit")
    monkeypatch.setattr(asr, "_BACKGROUND_TRANSCRIPTION_SLOT", slot)

    response = handle_call("transcribe", {"source": str(source), "model_size": "small"})

    assert response["data"]["status"] == "failed"
    assert response["data"]["retry_after_seconds"] == 0
    assert "worker exceeded its memory limit" in response["data"]["transcription_message"]


def test_background_transcription_slot_is_single_and_does_not_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source_a = tmp_path / "a.mp4"
    source_b = tmp_path / "b.mp4"
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_transcribe(source_path, model_size, allow_long_run):
        calls.append((source_path, model_size, allow_long_run))
        started.set()
        release.wait(timeout=3)
        return _result(source_path)

    monkeypatch.setattr(asr, "transcribe_video", fake_transcribe)
    slot = asr._BackgroundTranscriptionSlot()

    assert slot.start(str(source_a), "small")[0] == "started"
    assert started.wait(timeout=1)
    assert slot.start(str(source_a), "small")[0] == "in_progress"
    assert slot.start(str(source_a), "base")[0] == "in_progress"
    assert slot.start(str(source_b), "base")[0] == "busy"
    assert len(calls) == 1
    assert calls[0][2] is True
    release.set()


def test_background_failure_is_reported_once_and_can_be_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    calls = []
    source_a = str(tmp_path / "a.mp4")
    source_b = str(tmp_path / "b.mp4")

    def fake_transcribe(source_path, model_size, allow_long_run):
        calls.append((source_path, model_size, allow_long_run))
        if source_path == source_a:
            raise RuntimeError("synthetic ASR failure")
        return _result(source_path)

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(asr, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(asr.threading, "Thread", ImmediateThread)
    slot = asr._BackgroundTranscriptionSlot()

    assert slot.start(source_a, "small") == ("started", None)
    assert slot.start(source_b, "small") == ("started", None)
    assert slot.start(source_a, "small") == ("failed", "RuntimeError: synthetic ASR failure")
    assert slot.start(source_a, "small") == ("started", None)
    assert len(calls) == 3


def test_native_worker_crash_is_reported_instead_of_restarted_silently(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    def fake_transcribe(source_path, model_size, allow_long_run):
        result = _result(source_path)
        result.uncertainty = asr.ASR_WORKER_FAILURE_UNCERTAINTY
        return result

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(asr, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(asr.threading, "Thread", ImmediateThread)
    slot = asr._BackgroundTranscriptionSlot()

    assert slot.start(str(tmp_path / "video.mp4"), "small") == ("started", None)
    status, error = slot.start(str(tmp_path / "video.mp4"), "small")
    assert status == "failed"
    assert "exited without a transcript" in error


@pytest.mark.skipif(os.name != "posix", reason="Parent PID changes on POSIX")
def test_asr_worker_exits_when_supervisor_dies_before_worker_startup(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    seen = {}

    def capture_worker(command, timeout, memory_limit_bytes):
        seen["script"] = command[2]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "language": "en", "segments": []}),
            stderr="",
        )

    monkeypatch.setattr(asr, "_run_memory_capped_worker", capture_worker)
    asr.transcribe_video(str(source), model_size="small", allow_long_run=True)

    worker = seen["script"]
    supervisor = '''
import os, subprocess, sys
worker = "import time, sys; time.sleep(0.5); exec(sys.argv[1])"
env = os.environ.copy()
env["AGENTIC_VIDEO_ASR_PARENT_PID"] = str(os.getpid())
child = subprocess.Popen([sys.executable, "-c", worker, sys.argv[1]],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True, env=env)
print(child.pid, flush=True)
'''
    worker_pid = None
    try:
        completed = subprocess.run(
            [sys.executable, "-c", supervisor, worker],
            capture_output=True,
            text=True,
            timeout=8,
        )
        worker_pid = int(completed.stdout.strip())
        deadline = time.monotonic() + 3
        def worker_stopped():
            rss_bytes = asr._process_rss_bytes(worker_pid)
            # Linux may keep a terminated orphan as a zombie until PID 1 reaps
            # it; zombies have no resident pages but still have a /proc entry.
            return rss_bytes is None or rss_bytes == 0

        while time.monotonic() < deadline and not worker_stopped():
            time.sleep(0.1)
        assert worker_stopped()
    finally:
        if worker_pid is not None and asr._process_rss_bytes(worker_pid) is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_long_run_worker_uses_streamed_chunks_and_a_bounded_timeout(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    seen = {}

    def fake_run(command, timeout, memory_limit_bytes):
        seen["command"] = command
        seen["timeout"] = timeout
        seen["memory_limit_bytes"] = memory_limit_bytes
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "language": "en", "segments": []}),
            stderr="",
        )

    monkeypatch.setattr(asr, "_run_memory_capped_worker", fake_run)

    result = asr.transcribe_video(str(source), model_size="small", allow_long_run=True)

    worker_script = seen["command"][2]
    compile(worker_script, "<asr-worker>", "exec")
    assert seen["timeout"] == asr.ASR_BACKGROUND_TIMEOUT_SECONDS
    assert seen["memory_limit_bytes"] == 4 * 1024 * 1024 * 1024
    assert "chunk_seconds = 60" in worker_script
    assert "overlap_seconds = 3" in worker_script
    assert "audio = bytearray(carry)" in worker_script
    assert "chunk_start_sec = offset_sec - len(carry) / (sample_rate * 2)" in worker_script
    assert "parent_watchdog" in worker_script
    assert "os.killpg(os.getpgrp(), signal.SIGKILL)" in worker_script
    assert '"-f", "s16le", "pipe:1"' in worker_script
    assert "model.transcribe(\n                chunk_path" in worker_script
    assert result.segment_count == 0


@pytest.mark.skipif(os.name != "posix", reason="os.abort uses a POSIX signal")
def test_asr_accepts_complete_json_after_worker_abort(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    real_run = asr._run_memory_capped_worker
    fake_worker = '''
import json, os, sys
print(json.dumps({"ok": True, "language": "en", "segments": [
    {"id": 1, "start_sec": 2.0, "end_sec": 3.0, "text": "Recovered phrase"}
]}), flush=True)
sys.stderr.write("native teardown failed\\n")
sys.stderr.flush()
os.abort()
'''

    def run_fake(command, timeout, memory_limit_bytes):
        return real_run([sys.executable, "-c", fake_worker], timeout, memory_limit_bytes)

    monkeypatch.setattr(asr, "_run_memory_capped_worker", run_fake)
    result = asr.transcribe_video(str(source), model_size="tiny", timeout_sec=5)

    assert result.segment_count == 1
    assert result.segments[0].text == "Recovered phrase"
    assert result.has_speech is True


def test_asr_worker_failure_reports_return_code_and_stderr_tail(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    real_run = asr._run_memory_capped_worker
    fake_worker = '''
import sys
sys.stderr.write("old diagnostic " + "x" * 600 + " final diagnostic")
sys.stderr.flush()
sys.exit(7)
'''

    def run_fake(command, timeout, memory_limit_bytes):
        return real_run([sys.executable, "-c", fake_worker], timeout, memory_limit_bytes)

    monkeypatch.setattr(asr, "_run_memory_capped_worker", run_fake)
    with pytest.raises(ASRModelError) as caught:
        asr.transcribe_video(str(source), model_size="tiny", timeout_sec=5)

    assert "return code 7" in str(caught.value)
    assert "final diagnostic" in str(caught.value)
    assert "old diagnostic" not in str(caught.value)
    assert caught.value.details["returncode"] == 7
    assert caught.value.details["stderr_tail"].endswith("final diagnostic")
    assert len(caught.value.details["stderr_tail"]) == 500


def test_overlapping_chunk_segments_need_text_evidence_and_sort_by_start():
    segments = asr._deduplicate_chunk_segments([
        {"id": 1, "start_sec": 59.5, "end_sec": 60.5, "text": "crossing seam", "chunk_index": 0},
        {"id": 1, "start_sec": 59.6, "end_sec": 60.6, "text": "crossing seam", "chunk_index": 1},
        {"id": 2, "start_sec": 61.0, "end_sec": 61.2, "text": "yes", "chunk_index": 1},
        {"id": 3, "start_sec": 61.3, "end_sec": 61.5, "text": "yes", "chunk_index": 1},
        {"id": 4, "start_sec": 119.5, "end_sec": 120.0, "text": "complete", "chunk_index": 1},
        {"id": 1, "start_sec": 119.5, "end_sec": 122.0, "text": "complete phrase", "chunk_index": 2},
    ])

    assert [(s["id"], s["start_sec"], s["text"]) for s in segments] == [
        (1, 59.5, "crossing seam"),
        (2, 61.0, "yes"),
        (3, 61.3, "yes"),
        (4, 119.5, "complete phrase"),
    ]

    distinct = asr._deduplicate_chunk_segments([
        {"id": 1, "start_sec": 58.1, "end_sec": 59.4, "text": "earlier chunk", "chunk_index": 0},
        {"id": 1, "start_sec": 57.0, "end_sec": 58.3, "text": "later overlap", "chunk_index": 1},
        {"id": 2, "start_sec": 59.5, "end_sec": 60.5, "text": "turn left", "chunk_index": 0},
        {"id": 3, "start_sec": 59.6, "end_sec": 60.6, "text": "turn right", "chunk_index": 1},
    ])
    assert [(s["id"], s["start_sec"], s["text"]) for s in distinct] == [
        (1, 57.0, "later overlap"),
        (2, 58.1, "earlier chunk"),
        (3, 59.5, "turn left"),
        (4, 59.6, "turn right"),
    ]


@pytest.mark.parametrize("first, second", [
    ("I will go home", "I will not go home"),
    ("turn left", "turn right"),
])
def test_chunk_seam_keeps_distinct_utterances(first, second):
    segments = asr._deduplicate_chunk_segments([
        {"id": 1, "start_sec": 59.5, "end_sec": 60.5, "text": first, "chunk_index": 0},
        {"id": 1, "start_sec": 59.6, "end_sec": 60.6, "text": second, "chunk_index": 1},
    ])

    assert [(s["id"], s["text"]) for s in segments] == [(1, first), (2, second)]


@pytest.mark.parametrize("first, second, expected", [
    ("Going to the store!", "going to the store", "Going to the store!"),
    ("going to the", "going to the store", "going to the store"),
])
def test_chunk_seam_drops_only_repeated_or_clipped_utterances(first, second, expected):
    segments = asr._deduplicate_chunk_segments([
        {"id": 1, "start_sec": 59.5, "end_sec": 60.5, "text": first, "chunk_index": 0},
        {"id": 1, "start_sec": 59.6, "end_sec": 60.6, "text": second, "chunk_index": 1},
    ])

    assert [(s["id"], s["text"]) for s in segments] == [(1, expected)]


def test_memory_capped_worker_kills_a_process_over_its_limit(monkeypatch):
    memory_limit_bytes = 1024 * 1024 * 1024
    monkeypatch.setattr(
        asr, "_process_group_rss_bytes", lambda pid: memory_limit_bytes + 1
    )

    with pytest.raises(ASRModelError, match="exceeded its 1024 MiB memory limit"):
        asr._run_memory_capped_worker(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=None,
            memory_limit_bytes=memory_limit_bytes,
        )


def test_memory_capped_worker_kills_ffmpeg_child_after_worker_crash(tmp_path):
    marker = tmp_path / "child-finished"
    worker_script = '''
import os, subprocess, sys
child = subprocess.Popen([
    sys.executable, "-c",
    "import pathlib, sys, time; time.sleep(2.5); pathlib.Path(sys.argv[1]).write_text('alive')",
    sys.argv[1],
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
os._exit(7)
'''

    completed = asr._run_memory_capped_worker(
        [sys.executable, "-c", worker_script, str(marker)],
        timeout=5,
        memory_limit_bytes=asr.ASR_WORKER_MEMORY_LIMIT_BYTES,
    )

    assert completed.returncode == 7
    assert completed.stdout.strip().isdigit()
    time.sleep(2.8)
    assert not marker.exists()


def test_background_worker_reports_success_when_transcript_cannot_be_cached(
    monkeypatch, tmp_path
):
    source = _setup(monkeypatch, tmp_path)

    def successful_worker(command, timeout, memory_limit_bytes):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "language": "en", "segments": []}),
            stderr="",
        )

    monkeypatch.setattr(asr, "_run_memory_capped_worker", successful_worker)
    monkeypatch.setattr(MediaCache, "save_json", lambda self, fingerprint, name, data: None)

    with pytest.raises(ASRModelError, match="transcript could not be cached"):
        asr.transcribe_video(str(source), model_size="small", allow_long_run=True)


def test_memory_monitor_reads_a_live_process_rss():
    assert asr._process_rss_bytes(os.getpid()) > 0


def test_process_group_memory_monitor_reads_a_live_worker_group():
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert asr._process_rss_bytes(worker.pid) > 0
        assert asr._process_group_rss_bytes(worker.pid) > 0
    finally:
        asr._stop_worker_process(worker)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration test")
def test_windows_job_object_starts_and_stops_worker_tree_atomically(tmp_path):
    marker = tmp_path / "decoder-started"
    worker_script = '''
import subprocess, sys, time
child = subprocess.Popen([
    sys.executable, "-c",
    "import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
    sys.argv[1],
])
print(child.pid, flush=True)
time.sleep(30)
    '''
    job = asr._WindowsWorkerJob(512 * 1024 * 1024)
    process = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_file:
            process = job.spawn(
                [sys.executable, "-c", worker_script, str(marker)],
                stdout_file,
                stderr_file,
                os.environ.copy(),
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.05)
            assert marker.exists()
            stdout_file.flush()
            stdout_file.seek(0)
            child_pid = int(stdout_file.read().decode("ascii").strip())
    finally:
        job.close()
        if process is not None:
            try:
                process.wait(timeout=10)
            finally:
                process.close()

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    child_handle = kernel32.OpenProcess(0x00100000, False, child_pid)
    if child_handle:
        try:
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            assert kernel32.WaitForSingleObject(child_handle, 5000) == 0
        finally:
            kernel32.CloseHandle(child_handle)
    else:
        assert ctypes.get_last_error() in (87, 1168)


@pytest.mark.parametrize("requested, used, turbo", [
    ("medium", "medium", True), ("base", "small", False), ("base", "turbo", True), ("turbo", "small", False),
])
def test_request_transcription_substitutes_only_missing_models(monkeypatch, tmp_path, requested, used, turbo):
    import huggingface_hub.constants as hf_constants

    for size in ("medium", "small"):
        snapshot = tmp_path / f"models--Systran--faster-whisper-{size}" / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    ggml = tmp_path / "ggml-large-v3-turbo-q5_0.bin"
    if turbo:
        ggml.write_bytes(b"")
    monkeypatch.setattr(asr, "WHISPER_CPP_MODEL", ggml)
    monkeypatch.setattr("shutil.which", lambda name: "/bin/" + name)
    monkeypatch.setattr(asr, "probe_video", lambda path, **kwargs: _metadata(path))
    seen = []

    def stop(source_path, model_size, media_metadata=None):
        seen.append(model_size)
        raise LookupError("stop")

    monkeypatch.setattr(asr, "find_cached_transcript", stop)
    with pytest.raises(LookupError):
        asr.request_transcription(str(tmp_path / "video.mp4"), requested)
    assert seen == [used]


@pytest.mark.skipif(not asr._whisper_installed("turbo") or shutil.which("say") is None,
                    reason="needs whisper-cli, the ggml turbo model and macOS say")
def test_turbo_transcribes_real_speech_through_whisper_cpp(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    speech, video = tmp_path / "speech.aiff", tmp_path / "speech.mp4"
    subprocess.run(["say", "-o", str(speech), "The orange cat sat on the red mat."], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=4", "-i", str(speech),
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    try:
        asr.probe_video(str(video))
    except MediaCorruptError as exc:  # this machine's ffmpeg/say could not build the test clip
        pytest.skip(f"could not build a speech test clip here: {exc}")
    result = asr.transcribe_video(str(video), model_size="turbo", allow_long_run=True)
    text = " ".join(segment.text for segment in result.segments).lower()
    assert "orange cat" in text and "red mat" in text
    assert all(0 <= segment.confidence <= 1 for segment in result.segments)


@pytest.mark.parametrize("installed, expected", [(True, "turbo"), (False, "tiny")])
def test_default_model_is_turbo_only_where_installed(monkeypatch, tmp_path, installed, expected):
    ggml = tmp_path / "ggml-large-v3-turbo-q5_0.bin"
    if installed:
        ggml.write_bytes(b"")
    monkeypatch.setattr(asr, "WHISPER_CPP_MODEL", ggml)
    monkeypatch.setattr("shutil.which", lambda name: "/bin/" + name)
    assert asr.default_model_size() == expected
