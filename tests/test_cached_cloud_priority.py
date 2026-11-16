"""Breaker slice 1 (batch check 2026-11-16, untracked): on a host without cloud ASR, the default model's
cache lookup must serve the local turbo transcript, not a cloud one."""

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.types import TranscriptResult, TranscriptSegment, VideoMetadata
from src.transcript import asr

FINGERPRINT = "breaker-slice1-media"


def _metadata(source_path: str) -> VideoMetadata:
    return VideoMetadata(
        source_path=source_path, duration_sec=120.0, width=640, height=360, fps=30.0,
        format_name="mp4", size_bytes=123, has_audio=True, audio_codec="aac",
        video_codec="h264", fingerprint=FINGERPRINT,
    )


def _seed(model: str, text: str) -> None:
    seg = TranscriptSegment(id=1, start_sec=0.0, end_sec=1.0, text=text)
    res = TranscriptResult(
        source_path="video.mp4", language="en", duration_sec=120.0,
        segment_count=1, segments=[seg], has_speech=True,
    )
    MediaCache().save_json(FINGERPRINT, "transcript_asr_" + model, asr._transcript_cache_payload(res, FINGERPRINT, None, model))


def test_default_model_on_host_without_cloud_serves_local_turbo_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("AGENTIC_VIDEO_CLOUD_ASR", raising=False)
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test media")
    monkeypatch.setattr(asr, "probe_video", lambda path, **kwargs: _metadata(path))
    _seed("cloud", "CLOUD LINE")
    _seed("turbo", "LOCAL TURBO LINE")
    model = asr.default_model_size()
    assert model not in ("cloud", "turbo"), model
    found = asr.find_cached_transcript(str(source), model)
    assert found is not None
    result, cached_model = found
    assert cached_model == "turbo", cached_model
    assert result.segments[0].text == "LOCAL TURBO LINE", result.segments[0].text
