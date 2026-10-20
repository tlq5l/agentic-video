"""Focused tests for inspect_moments."""
import math

import pytest

from src.core.config import Config
from src.core.errors import ExtractionBoundsError
from src.core.types import ContactSheetResult, VideoMetadata
from src.retrieval.engine import AgenticVideoEngine


def _engine(duration: float = 100.0) -> AgenticVideoEngine:
    engine = AgenticVideoEngine("synthetic-placeholder.mp4")
    engine._meta = VideoMetadata(
        source_path="synthetic-placeholder.mp4",
        duration_sec=duration,
        width=1280,
        height=720,
        fps=30.0,
        format_name="mp4",
        size_bytes=1,
        has_audio=False,
    )
    return engine


def _fake_sheet(timestamps, cols):
    return ContactSheetResult(
        file_path=f"moment-{timestamps[0]:.3f}-{timestamps[-1]:.3f}.jpg",
        sample_count=len(timestamps),
        rows=math.ceil(len(timestamps) / cols),
        cols=cols,
        start_sec=timestamps[0],
        end_sec=timestamps[-1],
        timestamps=list(timestamps),
        frame_width=320,
        frame_height=180,
    )


def test_inspect_moments_includes_exact_anchor_and_symmetric_context(monkeypatch):
    engine = _engine()
    calls = []

    def fake_build(video_path, timestamps, output_path=None, cols=4, width=640, title=None, cache=None):
        calls.append((list(timestamps), cols, title))
        return _fake_sheet(timestamps, cols)

    monkeypatch.setattr(
        "src.retrieval.engine.build_contact_sheets",
        lambda video_path, jobs, cache=None: [fake_build(video_path, **job) for job in jobs],
    )
    result = engine.inspect_moments(
        moments=[{"timestamp_sec": 50.0, "label": "candidate-A"}],
        before_sec=4.0,
        after_sec=4.0,
        samples_per_moment=9,
        cols=3,
        max_total_samples=9,
    )

    item = result["moments"][0]
    assert item["sample_timestamps"] == [46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0]
    assert item["anchor_sample_index"] == 5
    assert item["relative_offsets_sec"] == [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    assert item["label"] == "candidate-A"
    assert result["total_samples"] == 9
    assert result["estimated_vision_tokens"] == 9 * Config.TOKENS_PER_IMAGE
    assert "anchor #5" in calls[0][2]
    assert "candidate-A" in calls[0][2]


def test_inspect_moments_preserves_input_order_and_labels(monkeypatch):
    engine = _engine()

    def fake_build(video_path, timestamps, output_path=None, cols=4, width=640, title=None, cache=None):
        return _fake_sheet(timestamps, cols)

    monkeypatch.setattr(
        "src.retrieval.engine.build_contact_sheets",
        lambda video_path, jobs, cache=None: [fake_build(video_path, **job) for job in jobs],
    )
    result = engine.inspect_moments(
        moments=[{"timestamp_sec": 20.0, "label": "B"}, {"timestamp_sec": 10.0, "label": "A"}],
        samples_per_moment=5,
        max_total_samples=10,
    )
    assert [item["timestamp_sec"] for item in result["moments"]] == [20.0, 10.0]
    assert [item["label"] for item in result["moments"]] == ["B", "A"]
    assert result["total"] == 2
    assert result["total_samples"] == 10


def test_inspect_moments_clamps_context_to_media_bounds(monkeypatch):
    engine = _engine(duration=100.0)

    def fake_build(video_path, timestamps, output_path=None, cols=4, width=640, title=None, cache=None):
        return _fake_sheet(timestamps, cols)

    monkeypatch.setattr(
        "src.retrieval.engine.build_contact_sheets",
        lambda video_path, jobs, cache=None: [fake_build(video_path, **job) for job in jobs],
    )
    result = engine.inspect_moments(
        moments=[{"timestamp_sec": 1.0}, {"timestamp_sec": 99.0}],
        before_sec=4.0,
        after_sec=4.0,
        samples_per_moment=9,
        max_total_samples=18,
    )
    first, last = result["moments"]
    assert first["start_sec"] == 0.0
    assert first["anchor_sample_index"] == 5
    assert last["end_sec"] == pytest.approx(99.999)
    assert last["sample_timestamps"][-1] < 100.0


def test_inspect_moments_rejects_even_sample_count(monkeypatch):
    engine = _engine()
    with pytest.raises(ExtractionBoundsError, match="odd integer"):
        engine.inspect_moments(moments=[{"timestamp_sec": 10.0}], samples_per_moment=8)


def test_inspect_moments_rejects_budget_overflow(monkeypatch):
    engine = _engine()
    with pytest.raises(ExtractionBoundsError, match="max_total_samples"):
        engine.inspect_moments(
            moments=[{"timestamp_sec": 10.0}, {"timestamp_sec": 20.0}],
            samples_per_moment=9,
            max_total_samples=17,
        )


def test_inspect_moments_rejects_out_of_bounds_anchor(monkeypatch):
    engine = _engine(duration=100.0)
    with pytest.raises(ExtractionBoundsError, match="outside media bounds"):
        engine.inspect_moments(moments=[{"timestamp_sec": 100.0}])
