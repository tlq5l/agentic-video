"""Focused tests for the bounded multi-window inspection primitive."""

import math

import pytest

from src.core.errors import ExtractionBoundsError
from src.core.types import FrameInfo, InspectionWindowResult, VideoMetadata
from src.retrieval.engine import AgenticVideoEngine


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


def _stub_inspection(monkeypatch, engine: AgenticVideoEngine):
    seen = []

    def fake_inspect_window(start_sec, end_sec, fps=4.0, create_sheet=True):
        seen.append((start_sec, end_sec, fps))
        frame_count = math.ceil((end_sec - start_sec) * fps)
        frames = [
            FrameInfo(
                index=i + 1,
                timestamp_sec=start_sec + i / fps,
                timecode=f"t{i}",
                file_path=f"frame-{len(seen)}-{i}.jpg",
                width=320,
                height=180,
            )
            for i in range(frame_count)
        ]
        return InspectionWindowResult(
            source_path=engine.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps,
            frames=frames,
            contact_sheet_path=f"sheet-{len(seen)}.jpg",
            estimated_vision_tokens=frame_count * 258,
        )

    monkeypatch.setattr(engine, "inspect_window", fake_inspect_window)
    return seen


def test_inspect_windows_sorts_and_merges_overlaps(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    result = engine.inspect_windows(
        windows=[
            {"start_sec": 10.0, "end_sec": 12.0},
            {"start_sec": 2.0, "end_sec": 4.0},
            {"start_sec": 3.5, "end_sec": 5.0},
        ],
        fps=2.0,
    )

    assert seen == [(2.0, 5.0, 2.0), (10.0, 12.0, 2.0)]
    assert [(item["start_sec"], item["end_sec"]) for item in result["windows"]] == [(2.0, 5.0), (10.0, 12.0)]
    assert result["windows"][0]["input_indices"] == [1, 2]
    assert result["windows"][1]["input_indices"] == [0]
    assert result["total_frames"] == 10


def test_inspect_windows_context_expands_and_clamps(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    engine.inspect_windows(
        windows=[
            {"start_sec": 0.5, "end_sec": 1.0},
            {"start_sec": 59.0, "end_sec": 59.5},
        ],
        fps=1.0,
        context_sec=2.0,
        merge_overlaps=False,
    )

    assert seen == [(0.0, 3.0, 1.0), (57.0, 60.0, 1.0)]


def test_inspect_windows_merge_false_preserves_overlapping_occurrences(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    result = engine.inspect_windows(
        windows=[{"start_sec": 4.0, "end_sec": 7.0}, {"start_sec": 5.0, "end_sec": 8.0}],
        fps=1.0,
        merge_overlaps=False,
    )

    assert seen == [(4.0, 7.0, 1.0), (5.0, 8.0, 1.0)]
    assert result["total"] == 2


def test_inspect_windows_rejects_budget_before_extraction(monkeypatch):
    engine = _engine()
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("extraction should not run after budget rejection")

    monkeypatch.setattr(engine, "inspect_window", should_not_run)
    with pytest.raises(ExtractionBoundsError, match="max_total_frames"):
        engine.inspect_windows(
            windows=[{"start_sec": 0.0, "end_sec": 10.0}],
            fps=4.0,
            max_total_frames=39,
        )
    assert called is False


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [{"start_sec": 3.0, "end_sec": 3.0}],
        [{"start_sec": 4.0, "end_sec": 3.0}],
        [{"start_sec": -1.0, "end_sec": 3.0}],
        [{"start_sec": float("nan"), "end_sec": 3.0}],
        [{"start_sec": 0.0, "end_sec": float("inf")}],
        [{"start_sec": 59.0, "end_sec": 61.0}],
        [{"start_sec": 1.0}],
    ],
)
def test_inspect_windows_rejects_invalid_windows(windows):
    with pytest.raises(ExtractionBoundsError):
        _engine().inspect_windows(windows=windows)


def test_inspect_windows_rejects_too_many_input_windows():
    windows = [{"start_sec": float(i), "end_sec": float(i) + 0.5} for i in range(17)]
    with pytest.raises(ExtractionBoundsError, match="max_windows"):
        _engine().inspect_windows(windows=windows, max_windows=16)


def test_executor_dispatch_for_inspect_windows(monkeypatch):
    import src.executor.mcp_server as mcp_server

    class FakeEngine:
        def __init__(self, source):
            self.source = source

        def inspect_windows(self, **kwargs):
            assert kwargs["windows"] == [{"start_sec": 1.0, "end_sec": 2.0}]
            assert kwargs["fps"] == 2.0
            return {
                "windows": [
                    {
                        "index": 1,
                        "input_indices": [0],
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "fps": 2.0,
                        "frames": [],
                        "contact_sheet_path": None,
                        "estimated_vision_tokens": 0,
                    }
                ],
                "total": 1,
                "total_frames": 0,
                "estimated_vision_tokens": 0,
            }

    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    outcome = mcp_server.handle_call(
        "inspect_windows",
        {
            "source": "video.mp4",
            "windows": [{"start_sec": 1.0, "end_sec": 2.0}],
            "fps": 2.0,
        },
    )
    assert outcome["ok"] is True
    assert outcome["data"]["windows"][0]["input_indices"] == [0]
