"""Focused tests for the evidence-producing track_region_events primitive."""

from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from src.core.errors import ExtractionBoundsError
from src.core.types import VideoMetadata
from src.retrieval.engine import AgenticVideoEngine


def _engine(duration: float = 20.0) -> AgenticVideoEngine:
    engine = AgenticVideoEngine("synthetic-placeholder.mp4")
    engine._meta = VideoMetadata(
        source_path="synthetic-placeholder.mp4",
        duration_sec=duration,
        width=100,
        height=100,
        fps=30.0,
        format_name="mp4",
        size_bytes=1,
        has_audio=False,
    )
    return engine


def _image(tmp_path, name: str, roi_value: int, outside_value: int = 0):
    path = tmp_path / name
    image = Image.new("L", (100, 100), outside_value)
    ImageDraw.Draw(image).rectangle((25, 25, 74, 74), fill=roi_value)
    image.save(path)
    return path


def _frames(paths, timestamps=None):
    if timestamps is None:
        timestamps = list(range(len(paths)))
    return [SimpleNamespace(file_path=str(path), timestamp_sec=float(timestamp)) for path, timestamp in zip(paths, timestamps)]


def _region():
    return {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}


def _run(monkeypatch, paths, timestamps=None, **kwargs):
    monkeypatch.setattr(
        "src.retrieval.engine.sample_sparse_frames",
        lambda *args, **call_kwargs: _frames(paths, timestamps),
    )
    return _engine(duration=max(kwargs.pop("duration", 20.0), 20.0)).track_region_events(
        kwargs.pop("start_sec", 0.0),
        kwargs.pop("end_sec", float((timestamps or list(range(len(paths))))[-1])),
        _region(),
        sample_fps=kwargs.pop("sample_fps", 1.0),
        frame_width=kwargs.pop("frame_width", 640),
        change_threshold=kwargs.pop("change_threshold", 0.001),
        ambiguity_margin=kwargs.pop("ambiguity_margin", 0.001),
        min_separation_sec=kwargs.pop("min_separation_sec", 0.0),
        max_events=kwargs.pop("max_events", 32),
        max_total_frames=kwargs.pop("max_total_frames", len(paths)),
        **kwargs,
    )


def test_track_region_events_reports_zero_changes(monkeypatch, tmp_path):
    stable = _image(tmp_path, "stable.png", 80)
    result = _run(monkeypatch, [stable, stable, stable, stable], end_sec=4.0, max_total_frames=4)

    assert result["event_count"] == 0
    assert result["transitions"] == []
    assert result["uncertain"] is False
    assert result["incomplete"] is False


def test_track_region_events_reports_one_transition_with_evidence(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "a0.png", 0),
        _image(tmp_path, "a1.png", 0),
        _image(tmp_path, "b0.png", 255),
        _image(tmp_path, "b1.png", 255),
    ]
    result = _run(monkeypatch, paths, end_sec=4.0, max_total_frames=4)

    assert result["event_count"] == 1
    transition = result["transitions"][0]
    assert transition["timestamp_sec"] == 2.0
    assert transition["evidence_moments"]["before"]["timestamp_sec"] == 1.0
    assert transition["evidence_moments"]["after"]["timestamp_sec"] == 2.0
    assert transition["change_score"] > 0.5


def test_track_region_events_preserves_requested_and_decoded_timing(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "p0.png", 0),
        _image(tmp_path, "p1.png", 255),
    ]
    frames = [
        SimpleNamespace(
            file_path=str(paths[0]),
            timestamp_sec=10.0,
            requested_timestamp=10.0,
            decoded_timestamp=9.967,
        ),
        SimpleNamespace(
            file_path=str(paths[1]),
            timestamp_sec=11.0,
            requested_timestamp=11.0,
            decoded_timestamp=None,
            timestamp_delta=None,
        ),
    ]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: frames)

    result = _engine().track_region_events(
        10.0,
        12.0,
        _region(),
        sample_fps=1.0,
        change_threshold=0.001,
        ambiguity_margin=0.001,
        min_separation_sec=0.0,
        max_total_frames=2,
    )

    transition = result["transitions"][0]
    assert transition["from_timestamp_provenance"] == {
        "requested_timestamp": 10.0,
        "decoded_timestamp": 9.967,
        "timestamp_delta": -0.033,
    }
    assert transition["timestamp_provenance"] == {
        "requested_timestamp": 11.0,
        "decoded_timestamp": None,
        "timestamp_delta": None,
    }
    assert transition["evidence_moments"]["before"]["timestamp_provenance"] == transition["from_timestamp_provenance"]
    assert transition["evidence_moments"]["after"]["timestamp_provenance"] == transition["timestamp_provenance"]


def test_track_region_events_does_not_infer_decoded_pts_from_requested_clock(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "unknown0.png", 0),
        _image(tmp_path, "unknown1.png", 255),
    ]
    frames = [
        SimpleNamespace(file_path=str(paths[0]), timestamp_sec=3.0),
        SimpleNamespace(file_path=str(paths[1]), timestamp_sec=4.0),
    ]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: frames)

    result = _engine().track_region_events(
        3.0,
        5.0,
        _region(),
        sample_fps=1.0,
        change_threshold=0.001,
        ambiguity_margin=0.001,
        min_separation_sec=0.0,
        max_total_frames=2,
    )

    after = result["transitions"][0]["timestamp_provenance"]
    assert after == {
        "requested_timestamp": 4.0,
        "decoded_timestamp": None,
        "timestamp_delta": None,
    }


def test_track_region_events_counts_multiple_transitions_chronologically(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "a.png", 0),
        _image(tmp_path, "b.png", 255),
        _image(tmp_path, "c.png", 128),
        _image(tmp_path, "c2.png", 128),
    ]
    result = _run(monkeypatch, paths, end_sec=4.0, max_total_frames=4)

    assert result["event_count"] == 2
    assert [item["rank"] for item in result["transitions"]] == [1, 2]
    assert [item["timestamp_sec"] for item in result["transitions"]] == [1.0, 2.0]


def test_track_region_events_counts_reversible_progression_as_two_transitions(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "a.png", 0),
        _image(tmp_path, "b.png", 255),
        _image(tmp_path, "a2.png", 0),
    ]
    result = _run(monkeypatch, paths, end_sec=3.0, max_total_frames=3)

    assert result["event_count"] == 2
    assert [item["timestamp_sec"] for item in result["transitions"]] == [1.0, 2.0]


def test_track_region_events_keeps_boundary_evidence_inside_requested_range(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "before.png", 0),
        _image(tmp_path, "inside.png", 255),
        _image(tmp_path, "after.png", 255),
    ]
    result = _run(
        monkeypatch,
        paths,
        timestamps=[10.0, 10.5, 11.0],
        start_sec=10.0,
        end_sec=11.5,
        duration=20.0,
        sample_fps=2.0,
        max_total_frames=3,
    )

    transition = result["transitions"][0]
    before = transition["evidence_moments"]["before"]["timestamp_sec"]
    after = transition["evidence_moments"]["after"]["timestamp_sec"]
    assert 10.0 <= before <= after <= 11.5


def test_track_region_events_long_range_respects_frame_budget(monkeypatch, tmp_path):
    stable = _image(tmp_path, "stable.png", 80)
    calls = []

    def fake_sample(video_path, fps, start_sec, end_sec, width, overlay_timestamp, cache):
        calls.append((start_sec, end_sec))
        timestamps = []
        cursor = start_sec
        while cursor < end_sec - 1e-9:
            timestamps.append(cursor)
            cursor += 1.0 / fps
        return _frames([stable] * len(timestamps), timestamps)

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fake_sample)
    result = _engine(duration=500.0).track_region_events(
        0.0,
        400.0,
        _region(),
        sample_fps=0.1,
        change_threshold=0.001,
        max_total_frames=50,
    )

    assert result["sample_count"] == 40
    assert result["sample_count"] <= 50
    assert result["incomplete"] is False
    assert [(item["start_sec"], item["end_sec"]) for item in result["scan_segments"]] == calls
    assert all(
        left["end_sec"] == right["start_sec"]
        for left, right in zip(result["scan_segments"], result["scan_segments"][1:])
    )


def test_track_region_events_marks_borderline_change_uncertain(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "zero.png", 0),
        _image(tmp_path, "slight.png", 4),
    ]
    result = _run(
        monkeypatch,
        paths,
        end_sec=2.0,
        change_threshold=0.001,
        ambiguity_margin=0.1,
        max_total_frames=2,
    )

    assert result["transitions"]
    assert result["uncertain"] is True
    assert result["event_count"] is None
    assert "borderline" in result["uncertainty_reason"]


def test_track_region_events_returns_incomplete_null_count_when_sampling_is_insufficient(monkeypatch, tmp_path):
    stable = _image(tmp_path, "single.png", 80)
    monkeypatch.setattr(
        "src.retrieval.engine.sample_sparse_frames",
        lambda *args, **kwargs: _frames([stable], [0.0]),
    )

    result = _engine().track_region_events(
        0.0,
        1.0,
        _region(),
        sample_fps=1.0,
        max_total_frames=2,
    )

    assert result["sample_count"] == 1
    assert result["transitions"] == []
    assert result["event_count"] is None
    assert result["incomplete"] is True
    assert "fewer frames" in result["uncertainty_reason"]


def test_track_region_events_rejects_planned_budget_overflow(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("sampling should not run")

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fail_if_called)
    with pytest.raises(ExtractionBoundsError, match="max_total_frames"):
        _engine().track_region_events(0, 10, _region(), sample_fps=2.0, max_total_frames=10)
    assert called is False
