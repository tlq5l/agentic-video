"""Generic range tests for recursive coarse-to-fine temporal foraging."""

import json
import math
from pathlib import Path

import pytest

from src.core.errors import ExtractionBoundsError
from src.core.types import ContactSheetResult, VideoMetadata
from src.media.contact_sheet import generate_video_overview
from src.retrieval.engine import AgenticVideoEngine


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_contact_sheet_samples_only_inside_requested_range():
    video = Path("fixtures/synthetic_bench_60s.mp4")
    sheet = generate_video_overview(video, samples=4, cols=2, start_sec=20.0, end_sec=30.0)

    assert sheet.timestamps == [21.25, 23.75, 26.25, 28.75]
    assert all(20.0 <= ts < 30.0 for ts in sheet.timestamps)


def test_bounded_contact_sheet_narrow_range_never_rounds_outside_half_open_interval():
    video = Path("fixtures/synthetic_bench_60s.mp4")
    sheet = generate_video_overview(video, samples=3, cols=3, start_sec=1.0001, end_sec=1.0002)

    assert len(sheet.timestamps) == 3
    assert all(1.0001 <= ts < 1.0002 for ts in sheet.timestamps)


def test_whole_video_contact_sheet_keeps_legacy_sampling():
    video = Path("fixtures/synthetic_bench_60s.mp4")
    legacy = generate_video_overview(video, samples=4, cols=2)
    explicit = generate_video_overview(video, samples=4, cols=2, start_sec=0.0, end_sec=None)

    assert explicit.timestamps == legacy.timestamps


@pytest.mark.parametrize(
    "start,end",
    [(30.0, 30.0), (31.0, 30.0), (-1.0, 3.0), (float("nan"), 3.0), (0.0, float("inf"))],
)
def test_bounded_contact_sheet_rejects_invalid_ranges(start: float, end: float):
    video = Path("fixtures/synthetic_bench_60s.mp4")
    with pytest.raises(ExtractionBoundsError):
        generate_video_overview(video, samples=4, start_sec=start, end_sec=end)


def _engine() -> AgenticVideoEngine:
    engine = AgenticVideoEngine("synthetic-placeholder.mp4")
    engine._meta = VideoMetadata(
        source_path="synthetic-placeholder.mp4",
        duration_sec=60.0,
        width=1280,
        height=720,
        fps=30.0,
        format_name="mp4",
        size_bytes=1,
        has_audio=False,
    )
    return engine


def test_detect_cuts_forwards_bounded_range_as_absolute_start_and_duration(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_detect(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return []

    monkeypatch.setattr("src.retrieval.engine.detect_scenes", fake_detect)
    monkeypatch.setattr("src.retrieval.engine.scan_scene_changes", lambda path, **kwargs: None)
    monkeypatch.setattr("src.retrieval.engine.detect_soft_transitions", lambda path, **kwargs: [])
    assert engine.detect_cuts(threshold=0.4, start_sec=20.0, end_sec=35.0) == []
    assert seen["threshold"] == 0.4
    assert seen["start_sec"] == 20.0
    assert seen["duration_sec"] == 15.0


def test_detect_cuts_with_a_raised_threshold_does_not_turn_dropped_cuts_into_transitions(monkeypatch):
    from src.media.scene import SceneCut

    engine = _engine()
    weak = SceneCut(10.0, 0.3, "00:00:10.000")
    strong = SceneCut(20.0, 0.9, "00:00:20.000")
    seen = {}

    def fake_detect(path, threshold, **kwargs):
        return [cut for cut in (weak, strong) if cut.score > threshold]

    def fake_soft(path, hard_cuts, **kwargs):
        seen["hard_cuts"] = list(hard_cuts)
        return []

    monkeypatch.setattr("src.retrieval.engine.detect_scenes", fake_detect)
    monkeypatch.setattr("src.retrieval.engine.scan_scene_changes", lambda path, **kwargs: None)
    monkeypatch.setattr("src.retrieval.engine.detect_soft_transitions", fake_soft)
    assert engine.detect_cuts(threshold=0.8) == [strong]
    assert seen["hard_cuts"] == [weak, strong]


def test_detect_cuts_whole_video_preserves_unbounded_default(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_detect(path, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("src.retrieval.engine.detect_scenes", fake_detect)
    monkeypatch.setattr("src.retrieval.engine.scan_scene_changes", lambda path, **kwargs: None)
    monkeypatch.setattr("src.retrieval.engine.detect_soft_transitions", lambda path, **kwargs: [])
    engine.detect_cuts()
    assert seen["start_sec"] == 0.0
    assert seen["duration_sec"] is None


@pytest.mark.parametrize(
    "start,end",
    [(10.0, 10.0), (20.0, 10.0), (-1.0, 3.0), (float("nan"), 3.0), (0.0, float("inf"))],
)
def test_detect_cuts_rejects_invalid_ranges(start: float, end: float):
    with pytest.raises(ExtractionBoundsError):
        _engine().detect_cuts(start_sec=start, end_sec=end)


def test_public_contract_exposes_range_bounds():
    contract = json.loads((ROOT / "src" / "adapters" / "mcp_schemas.json").read_text(encoding="utf-8"))
    for name in ("contact_sheet", "detect_cuts"):
        props = contract["tools"][name]["inputSchema"]["properties"]
        assert {"start_sec", "end_sec"} <= set(props)


def test_detect_cuts_bounded_enforces_inclusive_start_exclusive_end(monkeypatch):
    from src.media.scene import SceneCut

    engine = _engine()

    def fake_detect(path, **kwargs):
        return [
            SceneCut(19.999, 1.0, "00:00:19.999"),
            SceneCut(20.0, 1.0, "00:00:20.000"),
            SceneCut(34.999, 1.0, "00:00:34.999"),
            SceneCut(35.0, 1.0, "00:00:35.000"),
        ]

    monkeypatch.setattr("src.retrieval.engine.detect_scenes", fake_detect)
    monkeypatch.setattr("src.retrieval.engine.scan_scene_changes", lambda path, **kwargs: None)
    monkeypatch.setattr("src.retrieval.engine.detect_soft_transitions", lambda path, **kwargs: [])
    cuts = engine.detect_cuts(start_sec=20.0, end_sec=35.0)
    assert [cut.timestamp_sec for cut in cuts] == [20.0, 34.999]


def test_scene_cache_identity_distinguishes_fractional_ranges(monkeypatch, tmp_path):
    import src.media.scene as scene_mod

    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"synthetic")

    class FakeCache:
        def __init__(self):
            self.saved = []

        def compute_fingerprint(self, path):
            return "fp"

        def load_json(self, fingerprint, key):
            return None

        def save_json(self, fingerprint, key, value):
            self.saved.append(key)

    class FakeProc:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(scene_mod.subprocess, "run", lambda *args, **kwargs: FakeProc())
    cache = FakeCache()
    scene_mod.detect_scenes(str(source), threshold=0.251, start_sec=1.1000001, duration_sec=2.2, cache=cache)
    scene_mod.detect_scenes(str(source), threshold=0.251, start_sec=1.1000002, duration_sec=2.2, cache=cache)

    assert len(cache.saved) == 2
    assert cache.saved[0] != cache.saved[1]


def test_overview_chunks_partitions_full_span_without_gaps(monkeypatch):
    engine = _engine()
    seen = []

    def fake_overview(samples=12, cols=4, start_sec=0.0, end_sec=None):
        seen.append((start_sec, end_sec, samples, cols))
        return ContactSheetResult(
            file_path=f"sheet-{len(seen)}.jpg",
            sample_count=samples,
            rows=1,
            cols=cols,
            start_sec=start_sec,
            end_sec=end_sec,
            timestamps=[start_sec + (end_sec - start_sec) / 2.0],
            frame_width=320,
            frame_height=180,
        )

    monkeypatch.setattr(engine, "overview", fake_overview)
    chunks = engine.overview_chunks(
        start_sec=5.0,
        end_sec=52.0,
        chunk_sec=20.0,
        samples_per_chunk=4,
        cols=2,
    )

    assert [(c["range_start_sec"], c["range_end_sec"]) for c in chunks] == [
        (5.0, 25.0),
        (25.0, 45.0),
        (45.0, 52.0),
    ]
    assert seen == [(5.0, 25.0, 4, 2), (25.0, 45.0, 4, 2), (45.0, 52.0, 4, 2)]
    assert [c["index"] for c in chunks] == [1, 2, 3]


def test_overview_chunks_enforces_chunk_budget():
    with pytest.raises(ExtractionBoundsError, match="max_chunks"):
        _engine().overview_chunks(start_sec=0.0, end_sec=60.0, chunk_sec=10.0, max_chunks=5)


def test_public_contract_exposes_overview_chunks():
    contract = json.loads((ROOT / "src" / "adapters" / "mcp_schemas.json").read_text(encoding="utf-8"))
    props = contract["tools"]["overview_chunks"]["inputSchema"]["properties"]
    assert {"chunk_sec", "samples_per_chunk", "start_sec", "end_sec", "max_chunks"} <= set(props)


def test_contact_sheet_clamps_end_rounded_up_past_duration():
    video = Path("fixtures/synthetic_bench_60s.mp4")
    duration = AgenticVideoEngine(str(video)).metadata.duration_sec
    sheet = generate_video_overview(video, samples=2, cols=2, start_sec=duration - 10, end_sec=math.ceil(duration) + 0.5)
    assert all(ts < duration for ts in sheet.timestamps)
    with pytest.raises(ExtractionBoundsError, match="omit end_sec"):
        generate_video_overview(video, samples=2, cols=2, start_sec=0.0, end_sec=duration + 5)
