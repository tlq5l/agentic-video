"""Focused tests for refine_temporal_beam."""

import pytest

from src.core.config import Config
from src.core.errors import ExtractionBoundsError
from src.core.types import ContactSheetResult, VideoMetadata
from src.media.scene import SceneCut
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


def _fake_sheet(start_sec: float, end_sec: float, samples: int, cols: int) -> ContactSheetResult:
    step = (end_sec - start_sec) / samples
    timestamps = [start_sec + (i + 0.5) * step for i in range(samples)]
    return ContactSheetResult(
        file_path=f"sheet-{start_sec:.3f}-{end_sec:.3f}.jpg",
        sample_count=samples,
        rows=1,
        cols=cols,
        start_sec=timestamps[0],
        end_sec=timestamps[-1],
        timestamps=timestamps,
        frame_width=320,
        frame_height=180,
    )


def test_refine_temporal_beam_partitions_parents_without_gaps(monkeypatch):
    engine = _engine()
    calls = []

    def fake_overview(samples=12, cols=4, start_sec=0.0, end_sec=None):
        calls.append((start_sec, end_sec, samples, cols))
        return _fake_sheet(start_sec, end_sec, samples, cols)

    monkeypatch.setattr(engine, "_overview_batch", lambda ranges: [fake_overview(**r) for r in ranges])
    result = engine.refine_temporal_beam(
        windows=[{"start_sec": 10.0, "end_sec": 50.0}, {"start_sec": 60.0, "end_sec": 100.0}],
        fanout=4,
        samples_per_child=2,
        cols=2,
        max_total_samples=16,
    )

    assert result["total_samples"] == 16
    assert result["estimated_vision_tokens"] == 16 * Config.TOKENS_PER_IMAGE
    assert [p["parent_index"] for p in result["parents"]] == [1, 2]
    assert [(c["start_sec"], c["end_sec"]) for c in result["parents"][0]["children"]] == [
        (10.0, 20.0),
        (20.0, 30.0),
        (30.0, 40.0),
        (40.0, 50.0),
    ]
    assert [(c["start_sec"], c["end_sec"]) for c in result["parents"][1]["children"]] == [
        (60.0, 70.0),
        (70.0, 80.0),
        (80.0, 90.0),
        (90.0, 100.0),
    ]
    assert [c["child_index"] for c in result["parents"][0]["children"]] == [1, 2, 3, 4]
    assert calls[0] == (10.0, 20.0, 2, 2)
    assert calls[-1] == (90.0, 100.0, 2, 2)
    for parent in result["parents"]:
        children = parent["children"]
        assert children[0]["start_sec"] == parent["start_sec"]
        assert children[-1]["end_sec"] == parent["end_sec"]
        assert all(a["end_sec"] == b["start_sec"] for a, b in zip(children, children[1:]))
        for child in children:
            assert child["sample_count"] == 2
            assert all(child["start_sec"] <= t < child["end_sec"] for t in child["sample_timestamps"])


def test_refine_temporal_beam_context_expands_and_clamps(monkeypatch):
    engine = _engine(duration=20.0)
    monkeypatch.setattr(
        engine,
        "_overview_batch",
        lambda ranges: [_fake_sheet(r["start_sec"], r["end_sec"], r["samples"], r["cols"]) for r in ranges],
    )

    result = engine.refine_temporal_beam(
        windows=[{"start_sec": 1.0, "end_sec": 18.0}],
        fanout=2,
        samples_per_child=1,
        context_sec=5.0,
        max_total_samples=2,
    )
    parent = result["parents"][0]
    assert parent["input_start_sec"] == 1.0
    assert parent["input_end_sec"] == 18.0
    assert parent["start_sec"] == 0.0
    assert parent["end_sec"] == 20.0
    assert [(c["start_sec"], c["end_sec"]) for c in parent["children"]] == [(0.0, 10.0), (10.0, 20.0)]


def test_refine_temporal_beam_rejects_budget_before_extraction(monkeypatch):
    engine = _engine()

    def should_not_run(*args, **kwargs):
        raise AssertionError("extraction should not run before the budget gate")

    monkeypatch.setattr(engine, "_overview_batch", should_not_run)
    monkeypatch.setattr(engine, "detect_cuts", should_not_run)
    with pytest.raises(ExtractionBoundsError, match="max_total_samples"):
        engine.refine_temporal_beam(
            windows=[{"start_sec": 0.0, "end_sec": 40.0}],
            fanout=4,
            samples_per_child=3,
            max_total_samples=11,
            align_to_cuts=True,
        )


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [{"start_sec": -1.0, "end_sec": 2.0}],
        [{"start_sec": 5.0, "end_sec": 5.0}],
        [{"start_sec": 6.0, "end_sec": 5.0}],
        [{"start_sec": 0.0, "end_sec": float("inf")}],
        [{"start_sec": 0.0, "end_sec": 101.0}],
    ],
)
def test_refine_temporal_beam_rejects_invalid_windows(windows):
    with pytest.raises(ExtractionBoundsError):
        _engine().refine_temporal_beam(windows=windows)


def test_refine_temporal_beam_cut_alignment_is_bounded_and_deterministic(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "_overview_batch",
        lambda ranges: [_fake_sheet(r["start_sec"], r["end_sec"], r["samples"], r["cols"]) for r in ranges],
    )
    seen = []

    def fake_cuts(*args, **kwargs):
        seen.append(kwargs)
        return [
            SceneCut(timestamp_sec=18.5, score=0.7, timecode="00:00:18.500"),
            SceneCut(timestamp_sec=25.0, score=0.9, timecode="00:00:25.000"),
        ]

    monkeypatch.setattr(engine, "detect_cuts", fake_cuts)
    result = engine.refine_temporal_beam(
        windows=[{"start_sec": 0.0, "end_sec": 40.0}],
        fanout=2,
        samples_per_child=1,
        max_total_samples=2,
        align_to_cuts=True,
    )
    children = result["parents"][0]["children"]
    assert [(c["start_sec"], c["end_sec"]) for c in children] == [(0.0, 18.5), (18.5, 40.0)]
    assert seen == [{"start_sec": 0.0, "end_sec": 40.0}]


def test_refine_temporal_beam_does_not_detect_cuts_by_default(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "_overview_batch",
        lambda ranges: [_fake_sheet(r["start_sec"], r["end_sec"], r["samples"], r["cols"]) for r in ranges],
    )
    monkeypatch.setattr(engine, "detect_cuts", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected cut detection")))
    result = engine.refine_temporal_beam(
        windows=[{"start_sec": 20.0, "end_sec": 60.0}],
        fanout=2,
        samples_per_child=1,
        max_total_samples=2,
    )
    assert [(c["start_sec"], c["end_sec"]) for c in result["parents"][0]["children"]] == [(20.0, 40.0), (40.0, 60.0)]


def test_refine_temporal_beam_can_add_compact_parent_sheet(monkeypatch):
    engine = _engine()
    calls = []

    def fake_overview(samples=12, cols=4, start_sec=0.0, end_sec=None):
        calls.append((start_sec, end_sec, samples, cols))
        return _fake_sheet(start_sec, end_sec, samples, cols)

    monkeypatch.setattr(engine, "_overview_batch", lambda ranges: [fake_overview(**r) for r in ranges])
    result = engine.refine_temporal_beam(
        windows=[{"start_sec": 20.0, "end_sec": 60.0}],
        fanout=2,
        samples_per_child=2,
        cols=2,
        max_total_samples=4,
        compact_parent_sheet=True,
    )

    parent = result["parents"][0]
    assert parent["compact_contact_sheet_path"] == "sheet-20.000-60.000.jpg"
    assert parent["compact_sample_count"] == 4
    assert parent["compact_sample_timestamps"] == [25.0, 35.0, 45.0, 55.0]
    assert calls[0] == (20.0, 60.0, 4, 4)
    assert len(calls) == 3
