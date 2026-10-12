"""Focused tests for track_region_changes."""
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from src.core.errors import ExtractionBoundsError, SubprocessTimeoutError
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
    im = Image.new("L", (100, 100), outside_value)
    ImageDraw.Draw(im).rectangle((25, 25, 74, 74), fill=roi_value)
    im.save(path)
    return path


def _frames(paths):
    return [SimpleNamespace(file_path=str(path), timestamp_sec=float(i)) for i, path in enumerate(paths)]


def _region():
    return {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}


def test_track_region_changes_ranks_roi_state_change(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "f0.png", 0),
        _image(tmp_path, "f1.png", 0),
        _image(tmp_path, "f2.png", 255),
        _image(tmp_path, "f3.png", 255),
    ]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames(paths))
    result = _engine().track_region_changes(
        0, 3, _region(), sample_fps=1.0, min_change_score=0.001,
        robust_sigma=0.0, min_separation_sec=0.0, max_total_frames=4,
    )
    assert result["sample_count"] == 4
    assert result["candidates"][0]["timestamp_sec"] == 2.0
    assert result["candidates"][0]["rank"] == 1
    assert result["candidates"][0]["change_score"] > 0.5


def test_track_region_changes_preserves_sampling_provenance(monkeypatch, tmp_path):
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
            timestamp_delta=-0.033,
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

    result = _engine().track_region_changes(
        10.0,
        12.0,
        _region(),
        sample_fps=1.0,
        min_change_score=0.001,
        robust_sigma=0.0,
        min_separation_sec=0.0,
        max_total_frames=3,
    )

    sample = result["samples"][0]
    assert sample["from_timestamp_provenance"]["decoded_timestamp"] == 9.967
    assert sample["timestamp_provenance"]["decoded_timestamp"] is None


def test_track_region_changes_ignores_changes_outside_roi(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "f0.png", 80, outside_value=0),
        _image(tmp_path, "f1.png", 80, outside_value=255),
        _image(tmp_path, "f2.png", 80, outside_value=0),
    ]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames(paths))
    result = _engine().track_region_changes(
        0, 2, _region(), sample_fps=1.0, min_change_score=0.001,
        robust_sigma=0.0, max_total_frames=3,
    )
    assert result["candidates"] == []
    assert all(item["change_score"] == 0 for item in result["samples"])


def test_track_region_changes_collapses_nearby_peaks(monkeypatch, tmp_path):
    paths = [
        _image(tmp_path, "f0.png", 0),
        _image(tmp_path, "f1.png", 255),
        _image(tmp_path, "f2.png", 0),
    ]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames(paths))
    result = _engine().track_region_changes(
        0, 2, _region(), sample_fps=1.0, min_change_score=0.001,
        robust_sigma=0.0, min_separation_sec=2.0, max_total_frames=3,
    )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["timestamp_sec"] == 1.0


def test_track_region_changes_rejects_budget_before_sampling(monkeypatch):
    called = False
    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("sampling should not run")
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fail_if_called)
    with pytest.raises(ExtractionBoundsError, match="max_total_frames"):
        _engine().track_region_changes(0, 10, _region(), sample_fps=2.0, max_total_frames=10)
    assert not called


def test_track_region_changes_segments_long_scans(monkeypatch, tmp_path):
    path = _image(tmp_path, "stable.png", 80)
    calls = []

    def fake_sample(video_path, fps, start_sec, end_sec, width, overlay_timestamp, cache):
        calls.append((start_sec, end_sec))
        return [SimpleNamespace(file_path=str(path), timestamp_sec=float(start_sec))]

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fake_sample)
    result = _engine(duration=1301.0).track_region_changes(
        0, 1300, _region(), sample_fps=0.001, min_change_score=0.001,
        robust_sigma=0.0, max_total_frames=8,
    )
    assert calls[0] == (0.0, 180.0)
    assert calls[-1] == (1260.0, 1300.0)
    assert all((end - start) <= 180.0 for start, end in calls)
    assert len(result["scan_segments"]) == len(calls)
    assert result["sample_count"] == len(calls)


def test_track_region_changes_recovers_timeout_by_subdividing(monkeypatch, tmp_path):
    path = _image(tmp_path, "stable.png", 80)
    calls = []

    def fake_sample(video_path, fps, start_sec, end_sec, width, overlay_timestamp, cache):
        calls.append((start_sec, end_sec))
        if end_sec - start_sec > 50.0:
            raise SubprocessTimeoutError("synthetic timeout")
        return [
            SimpleNamespace(file_path=str(path), timestamp_sec=float(start_sec)),
            SimpleNamespace(file_path=str(path), timestamp_sec=float(end_sec)),
        ]

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fake_sample)
    result = _engine(duration=100.0).track_region_changes(
        0, 100, _region(), sample_fps=0.01, min_change_score=0.001,
        robust_sigma=0.0, max_total_frames=4,
    )

    assert calls == [(0.0, 100.0), (0.0, 50.0), (50.0, 100.0)]
    assert [(item["start_sec"], item["end_sec"]) for item in result["scan_segments"]] == [
        (0.0, 50.0), (50.0, 100.0)
    ]
    assert result["sample_count"] == 3
    assert result["sample_count"] <= 4


def test_track_region_changes_propagates_non_timeout_errors(monkeypatch):
    def fail_with_non_timeout(*args, **kwargs):
        raise RuntimeError("sentinel sampling failure")

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fail_with_non_timeout)
    with pytest.raises(RuntimeError, match="sentinel sampling failure"):
        _engine().track_region_changes(0, 10, _region(), sample_fps=0.1, max_total_frames=3)


def test_track_region_changes_recovery_enforces_frame_budget(monkeypatch, tmp_path):
    path = _image(tmp_path, "stable.png", 80)

    def fake_sample(video_path, fps, start_sec, end_sec, width, overlay_timestamp, cache):
        if end_sec - start_sec > 5.0:
            raise SubprocessTimeoutError("synthetic timeout")
        return [
            SimpleNamespace(file_path=str(path), timestamp_sec=float(start_sec)),
            SimpleNamespace(file_path=str(path), timestamp_sec=float(end_sec)),
        ]

    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", fake_sample)
    with pytest.raises(ExtractionBoundsError, match="max_total_frames"):
        _engine(duration=10.0).track_region_changes(
            0, 10, _region(), sample_fps=0.1, max_total_frames=2,
        )


def test_track_region_changes_rejects_invalid_normalized_region():
    with pytest.raises(ExtractionBoundsError, match="normalized"):
        _engine().track_region_changes(
            0, 2, {"x": 0.8, "y": 0.2, "width": 0.3, "height": 0.2},
            sample_fps=1.0, max_total_frames=3,
        )


def test_track_region_changes_single_frame_error_says_how_to_fix(monkeypatch, tmp_path):
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames([_image(tmp_path, "f0.png", 0)]))
    with pytest.raises(ExtractionBoundsError, match=r"raise sample_fps to at least 1 .*or widen start_sec/end_sec") as exc:
        _engine().track_region_changes(0, 2, _region())
    assert exc.value.details["sample_count"] == 1


def test_track_region_changes_reports_steady_change_without_candidates(monkeypatch, tmp_path):
    # Every frame differs slightly from the last: no single moment stands out.
    paths = [_image(tmp_path, f"s{i}.png", 40 + i * 4) for i in range(8)]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames(paths))
    result = _engine().track_region_changes(0, 7, _region(), sample_fps=1.0, max_total_frames=8)
    assert result["candidates"] == []
    assert result["steady_change"]["detected"] is True
    assert result["steady_change"]["active_fraction"] == 1.0


def test_track_region_changes_reports_no_steady_change_for_still_region(monkeypatch, tmp_path):
    paths = [_image(tmp_path, f"q{i}.png", 0 if i < 4 else 255) for i in range(8)]
    monkeypatch.setattr("src.retrieval.engine.sample_sparse_frames", lambda *args, **kwargs: _frames(paths))
    result = _engine().track_region_changes(
        0, 7, _region(), sample_fps=1.0, min_change_score=0.001, robust_sigma=0.0, max_total_frames=8,
    )
    assert result["steady_change"]["detected"] is False
    assert result["candidates"][0]["timestamp_sec"] == 4.0
