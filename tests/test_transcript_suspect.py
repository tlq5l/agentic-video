"""The optional `suspect` field on transcript segments and search hits.

When a line is not flagged, the key must be absent so the wire shape is unchanged.
"""

from typing import Any

import src.transcript.search as search_module
from src.core.types import SearchResult, TranscriptResult, TranscriptSegment


SUSPECT_REASON = "same line 3 times in a row; may be invented over music or silence"


def test_unflagged_segment_dict_has_no_suspect_key():
    d = TranscriptSegment(id=1, start_sec=0.0, end_sec=1.0, text="hi").to_dict()
    assert set(d) == {
        "id", "start_sec", "end_sec", "text",
        "start_timecode", "end_timecode", "confidence",
    }


def test_transcript_result_drops_suspect_only_on_unflagged_segments():
    result = TranscriptResult(
        source_path="clip.mp4",
        language="en",
        duration_sec=2.0,
        segment_count=2,
        segments=[
            TranscriptSegment(id=1, start_sec=0.0, end_sec=1.0, text="a", suspect="repeat"),
            TranscriptSegment(id=2, start_sec=1.0, end_sec=2.0, text="b"),
        ],
    )
    segments = result.to_dict()["segments"]
    assert segments[0]["suspect"] == "repeat"
    assert "suspect" not in segments[1]


def test_segment_round_trip_through_cache_dict():
    flagged = TranscriptSegment(id=1, start_sec=0.0, end_sec=1.0, text="a", suspect="repeat")
    unflagged = TranscriptSegment(id=2, start_sec=1.0, end_sec=2.0, text="b")
    assert TranscriptSegment(**flagged.to_dict()) == flagged
    assert TranscriptSegment(**unflagged.to_dict()) == unflagged


def _hit_kwargs(**extra) -> dict[str, Any]:
    return dict(
        segment_id=1, start_sec=0.0, end_sec=1.0, text="know my name", score=1.0,
        start_timecode="00:00:00.000", end_timecode="00:00:01.000", context="know my name", **extra,
    )


def test_search_result_keeps_suspect_only_when_set():
    assert "suspect" not in SearchResult(**_hit_kwargs()).to_dict()
    flagged = SearchResult(**_hit_kwargs(suspect=SUSPECT_REASON)).to_dict()
    assert flagged["suspect"] == SUSPECT_REASON
    # model_adapter rebuilds hits with SearchResult(**hit).
    assert SearchResult(**flagged).suspect == SUSPECT_REASON


def _fake_transcription(extra=None):
    segments = [
        {
            "id": 1,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "you know my name",
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:01.000",
            "confidence": None,
            "suspect": SUSPECT_REASON,
        },
        {
            "id": 2,
            "start_sec": 1.0,
            "end_sec": 2.0,
            "text": "tell me, know my name",
            "start_timecode": "00:00:01.000",
            "end_timecode": "00:00:02.000",
            "confidence": None,
        },
    ]

    def fake(source, model_size):
        payload = {"segments": segments}
        if extra:
            payload.update(extra)
        return payload

    return fake


def test_search_transcript_passes_suspect_through_list_result(monkeypatch):
    monkeypatch.setattr(search_module.asr, "request_transcription", _fake_transcription())
    hits = search_module.search_transcript("clip.mp4", "know my name")
    assert isinstance(hits, list)
    assert all(isinstance(h, SearchResult) for h in hits)
    by_id = {h.segment_id: h for h in hits}
    assert by_id[1].suspect == SUSPECT_REASON
    assert by_id[2].suspect is None


def test_search_transcript_passes_suspect_through_status_result(monkeypatch):
    monkeypatch.setattr(
        search_module.asr,
        "request_transcription",
        _fake_transcription(extra={"cached_model_size": "tiny"}),
    )
    result = search_module.search_transcript("clip.mp4", "know my name")
    assert isinstance(result, dict)
    hits = result["hits"]
    assert all(isinstance(h, dict) for h in hits)
    by_id = {h["segment_id"]: h for h in hits}
    assert by_id[1]["suspect"] == SUSPECT_REASON
    assert "suspect" not in by_id[2]
