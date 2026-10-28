"""Repeated ASR lines are flagged when a transcript is returned; the cache stays unflagged."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.types import TranscriptResult, TranscriptSegment, VideoMetadata
from src.transcript import asr

FINGERPRINT = "flag-wiring-media"
BASE = "Note: Non-speech audio (music, sound effects, alarms) is NOT transcribed."
REPEATED = [
    (1.0, 2.0, "La la la"),
    (2.0, 3.0, "La la la"),
    (3.0, 4.0, "La la la"),
    (4.0, 6.0, "Real words here"),
]
DISTINCT = [(1.0, 2.0, "Hello there"), (2.0, 3.0, "Second thought"), (3.0, 4.0, "Final remark")]


def _metadata(source_path: str) -> VideoMetadata:
    return VideoMetadata(
        source_path=source_path, duration_sec=120.0, width=640, height=360, fps=30.0,
        format_name="mp4", size_bytes=123, has_audio=True, audio_codec="aac",
        video_codec="h264", fingerprint=FINGERPRINT,
    )


def _result(lines) -> TranscriptResult:
    segments = [
        TranscriptSegment(id=i, start_sec=s, end_sec=e, text=t)
        for i, (s, e, t) in enumerate(lines, 1)
    ]
    return TranscriptResult(
        source_path="video.mp4", language="en", duration_sec=120.0,
        segment_count=len(segments), segments=segments, has_speech=True,
    )


def _seed_asr(lines) -> None:
    payload = asr._transcript_cache_payload(_result(lines), FINGERPRINT, None, "small")
    MediaCache().save_json(FINGERPRINT, "transcript_asr_small", payload)


def _setup(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test media")
    monkeypatch.setattr(asr, "probe_video", lambda path, **kwargs: _metadata(path))
    return source


def _assert_flagged(result: TranscriptResult) -> None:
    assert [segment.suspect is not None for segment in result.segments] == [True, True, True, False]
    assert result.notice.startswith(f"{BASE} Flagged 3 lines as suspect")


def _assert_disk_unflagged() -> None:
    payload = MediaCache().load_json(FINGERPRINT, "transcript_asr_small")
    assert payload is not None
    assert all("suspect" not in segment for segment in payload["segments"])
    assert "Flagged" not in payload["notice"]


def test_cached_asr_transcript_is_flagged_on_read(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_asr(REPEATED)
    _assert_flagged(asr.transcribe_video(str(source), model_size="small"))
    _assert_disk_unflagged()


def test_two_reads_do_not_repeat_the_notice(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_asr(REPEATED)
    first = asr.find_cached_transcript(str(source), "small")
    second = asr.find_cached_transcript(str(source), "small")
    assert first is not None and second is not None
    _assert_flagged(first[0])
    assert second[0].notice == first[0].notice
    assert second[0].notice.count("Flagged") == 1


def test_cached_subtitle_transcript_is_not_flagged(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    payload = asr._transcript_cache_payload(_result(REPEATED), FINGERPRINT, "subfp", None)
    MediaCache().save_json(FINGERPRINT, "transcript_subtitle_subfp", payload)
    cached = asr._cached_result(MediaCache(), source, FINGERPRINT, "subfp", None, 120.0)
    assert cached is not None
    assert all(segment.suspect is None for segment in cached.segments)
    assert cached.notice == BASE


def test_fresh_asr_run_is_flagged_but_cached_unflagged(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    segments = [
        {"id": i, "start_sec": s, "end_sec": e, "text": t}
        for i, (s, e, t) in enumerate(REPEATED, 1)
    ]
    output = json.dumps({"ok": True, "language": "en", "segments": segments})

    def fake_worker(command, timeout, memory_limit_bytes):
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(asr, "_run_memory_capped_worker", fake_worker)
    _assert_flagged(asr.transcribe_video(str(source), model_size="small", timeout_sec=5))
    _assert_disk_unflagged()


def test_transcript_without_repeats_keeps_its_notice(monkeypatch, tmp_path):
    source = _setup(monkeypatch, tmp_path)
    _seed_asr(DISTINCT)
    result = asr.transcribe_video(str(source), model_size="small")
    assert result.notice == BASE
    assert all(segment.suspect is None for segment in result.segments)
