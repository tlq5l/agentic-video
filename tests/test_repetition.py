"""Blind synthetic tests for completed cycle counting and background behavior."""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.core.cache import MediaCache
from src.core.types import VideoMetadata
from src.retrieval.engine import AgenticVideoEngine
from src.retrieval import periodicity, repetition


def pulses(fps=20, count=7, periods=None, lead=0.6, tail=0.7):
    periods = periods or [0.8] * count
    duration = lead + sum(periods) + tail
    t = np.arange(round(duration * fps) + 1) / fps
    x = np.zeros_like(t)
    cursor = lead
    for period in periods:
        phase = (t - cursor) / period
        inside = (phase >= 0) & (phase <= 1)
        x[inside] = np.sin(np.pi * phase[inside]) ** 2
        cursor += period
    return t, x


def test_signal_counts_complete_variable_cycles_and_pause():
    t, x = pulses(periods=[0.55, 0.65, 0.75, 0.9, 1.0, 1.15, 1.25])
    result = repetition.count_signal(x, fps=20, start_sec=0, min_period_sec=0.4, max_period_sec=1.5)
    assert result["count"] == 7
    assert len(result["cycles"]) == 7
    assert all(c["start_sec"] < c["peak_sec"] < c["end_sec"] for c in result["cycles"])
    assert all(a["end_sec"] <= b["start_sec"] + 0.15 for a, b in zip(result["cycles"], result["cycles"][1:]))
    assert result["regularity"] > 0.5


def test_partial_edges_excluded():
    t, x = pulses(count=5)
    # Slice through first and last pulses; only the middle three complete.
    x = x[int(0.9 * 20):int(4.6 * 20)]
    result = repetition.count_signal(x, fps=20, start_sec=0.9, min_period_sec=0.5, max_period_sec=1.1)
    assert result["count"] == 3


def test_three_complete_cycles_before_partial_fourth_stay_counted():
    _, full = pulses(count=3, lead=.6, tail=.05)
    phase = np.arange(1, 6) / 20
    signal = np.r_[full[:-1], np.sin(np.pi * phase / .8) ** 2]
    result = repetition.count_signal(signal, fps=20)
    energy = np.abs(np.diff(signal, prepend=signal[0]))

    assert result["count"] == 3
    assert repetition._exclude_active_tail(result, energy, (len(signal) - 1) / 20, 20)["count"] == 3


@pytest.mark.parametrize("fps", (10, 15, 20, 30))
@pytest.mark.parametrize("partial_fraction", (.25, .4, .6))
@pytest.mark.parametrize("visual", (False, True))
def test_partial_fourth_keeps_completed_cycles_across_frame_rates(fps, partial_fraction, visual):
    _, full = pulses(fps=fps, count=3, lead=.6, tail=.05)
    n = round(.8 * partial_fraction * fps)
    phase = np.arange(1, n + 1) / fps
    signal = np.r_[full[:-1], np.sin(np.pi * phase / .8) ** 2]
    result = repetition.count_signal(signal, fps=fps)
    energy = np.abs(np.diff(signal, prepend=signal[0]))

    assert result["count"] == 3
    assert repetition._exclude_active_tail(result, energy, (len(signal) - 1) / fps, fps, visual=visual)["count"] == 3


def test_immediate_partial_fourth_does_not_remove_complete_third():
    # Motion stays active across the third cycle's boundary and into rep four.
    result = {"count": 3, "cycles": [
        {"start_sec": .5, "end_sec": 1.2},
        {"start_sec": 1.3, "end_sec": 2.0},
        {"start_sec": 2.1, "end_sec": 2.8},
    ], "period_sec": .8, "confidence": .9, "signal_quality": "good"}
    energy = np.full(63, .1)

    assert repetition._exclude_active_tail(result, energy, 3.1, 20)["count"] == 3

    short_lobe = {**result, "cycles": [*result["cycles"][:-1],
                                       {"start_sec": 2.6, "end_sec": 3.0}]}
    assert repetition._exclude_active_tail(short_lobe, energy, 3.1, 20)["count"] == 2


def test_complete_cycle_at_window_start_is_counted():
    _, x = pulses(count=2, lead=0, tail=.5)
    assert repetition.count_signal(x, fps=20)["count"] == 2


def test_noise_and_static_have_no_cycles():
    rng = np.random.default_rng(3)
    for x in (np.zeros(250), rng.normal(0, 0.002, 250), rng.normal(0, 0.02, 250)):
        result = repetition.count_signal(x, fps=20, min_period_sec=0.4, max_period_sec=2)
        assert result["count"] == 0
        assert result["signal_quality"] == "low_activity"


def test_embedding_sequence_uses_similarity_for_known_cycles():
    t, x = pulses(count=6, periods=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    phase = np.arccos(1 - 2*x)
    embeddings = np.column_stack([np.cos(phase), np.sin(phase), np.ones(len(x)) * 0.1])
    result = repetition.count_embeddings(embeddings, fps=20, min_period_sec=0.5, max_period_sec=1.2)
    assert result["count"] == 6
    assert result["regularity"] > 0.5


def swinging_frames(fps, seconds, period):
    """Textured 128x72 clip with a bright block swinging side to side."""
    rng = np.random.default_rng(3)
    background = rng.integers(40, 90, (72, 128)).astype(np.uint8)
    frames = []
    for i in range(int(seconds * fps)):
        frame = background.copy()
        # Amplitude grows with the period so the block moves at most a few pixels per frame.
        x = int(round(64 + 10 * period * np.sin(2 * np.pi * i / fps / period)))
        frame[24:48, x - 12:x + 12] = 230
        frames.append(frame)
    return np.stack(frames)


@pytest.mark.parametrize("period", (0.33, 0.9, 2.0))
def test_periodicity_finds_period_and_count_including_fast_cycles(period):
    found = periodicity.estimate(swinging_frames(15, 8.0, period), 15)
    assert found is not None
    assert found["period_sec"] == pytest.approx(period, rel=0.1)
    assert abs(found["count"] - 8.0 / period) <= 1.5
    assert all(c["start_sec"] <= c["peak_sec"] <= c["end_sec"] for c in found["cycles"])


def test_periodicity_reports_nothing_for_static_or_noise_frames():
    rng = np.random.default_rng(5)
    assert periodicity.estimate(np.full((60, 72, 128), 90, np.uint8), 15) is None
    assert periodicity.estimate(rng.integers(0, 255, (60, 72, 128)).astype(np.uint8), 15) is None


def test_periodicity_skips_regions_too_small_for_a_cell():
    assert periodicity.estimate(swinging_frames(15, 4.0, 0.9)[:, :6, :6], 15) is None


def test_short_windows_are_sampled_at_the_fps_cap_and_long_ones_are_not():
    assert repetition._effective_fps(10.0, 2.0) == repetition.MAX_FPS
    assert repetition._effective_fps(10.0, 120.0) == 10.0


def test_zero_count_note_never_denies_periodic_motion_that_was_found():
    none = {"count": 0}
    assert "no periodic motion" in repetition._note(none, None)
    assert "no periodic motion" in repetition._note(none, {"count": 0, "period_sec": 1.0})
    seen = repetition._note(none, {"count": 12, "period_sec": 0.5})
    assert "no periodic motion" not in seen and "12 cycles" in seen
    assert repetition._note({"count": 3}, None) == repetition.NOTE
    assert not repetition._agrees({"count": 5, "period_sec": 1.0}, {"count": 0, "period_sec": 1.0}, np.zeros(60))


def test_amplitude_count_yields_to_periodicity_unless_idle_margins_explain_the_gap():
    found = {"count": 7, "period_sec": 1.3}
    result = {"count": 6, "period_sec": 1.3}
    idle_edges = np.r_[np.zeros(10), np.ones(40) * .05, np.zeros(10)]
    active_edges = np.ones(60) * .05
    assert repetition._agrees(result, found, idle_edges)
    assert not repetition._agrees(result, found, active_edges)
    assert not repetition._agrees({"count": 0, "period_sec": None}, found, idle_edges)
    assert not repetition._agrees({"count": 6, "period_sec": 3.0}, found, idle_edges)


def test_background_polling_and_cache(tmp_path, monkeypatch):
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_count(_engine, **kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(3)
        return {"status": "complete", "count": 2, "cycles": []}

    monkeypatch.setattr(repetition, "count_video", slow_count)
    try:
        start = time.monotonic()
        first = engine.count_repetitions(start_sec=0, end_sec=2)
        assert time.monotonic() - start < 1
        assert first["status"] == "in_progress"
        assert first["retry_after_seconds"] > 0
        assert entered.wait(1)
        assert engine.count_repetitions(start_sec=0, end_sec=2)["status"] == "in_progress"
        assert len(calls) == 1
    finally:
        release.set()
    repetition._worker.join(3)
    assert engine.count_repetitions(start_sec=0, end_sec=2)["count"] == 2
    assert len(calls) == 1


def test_dead_background_worker_returns_error_instead_of_running(tmp_path, monkeypatch):
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    monkeypatch.setattr(repetition, "_jobs", repetition.OrderedDict())
    monkeypatch.setattr(repetition, "count_video", lambda *_args, **_kwargs: {"status": "complete", "count": 1})

    assert engine.count_repetitions(start_sec=0, end_sec=2)["status"] == "in_progress"
    repetition._worker.join(3)
    key = next(iter(repetition._jobs))
    repetition._jobs[key] = {"status": "in_progress", "retry_after_seconds": 2}
    result = engine.count_repetitions(start_sec=0, end_sec=2)
    assert result["status"] == "error"
    assert "stopped unexpectedly" in result["message"]


def test_decode_drains_large_stderr_and_bounds_filter_threads(tmp_path, monkeypatch):
    script = ("import os, sys\n"
              "os.write(2, b'x' * 1048576)\n"
              "os.write(1, bytes(128 * 72 * 3))\n")
    real_popen = subprocess.Popen
    commands = []

    def fake_ffmpeg(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(repetition.subprocess, "Popen", fake_ffmpeg)
    frames, energy = repetition._decode_frames(tmp_path / "broken.mp4", 0, 1, 10, None)
    assert len(frames) == 3
    assert len(energy) == 3
    command = commands[0]
    assert command.index("-threads") < command.index("-i")
    assert command[command.index("-threads") + 1] == "2"
    assert command.index("-filter_threads") < command.index("-i")
    assert command[command.index("-filter_threads") + 1] == "2"


def test_decode_timeout_is_reported(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def fake_ffmpeg(_command, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(5)"], **kwargs)

    monkeypatch.setattr(repetition.subprocess, "Popen", fake_ffmpeg)
    monkeypatch.setattr(repetition, "DECODE_TIMEOUT_SEC", .2)
    monkeypatch.setattr(repetition, "DECODE_TIMEOUT_PER_WINDOW_SEC", 0.0)
    with pytest.raises(TimeoutError, match="ffmpeg decode timed out after 0.2s"):
        repetition._decode_frames(tmp_path / "broken.mp4", 0, 1, 10, None)

    source = tmp_path / "broken.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    engine._meta = VideoMetadata(str(source), 2.0, 128, 72, 10.0, "mp4", 5, False,
                                 fingerprint=engine._cache.compute_fingerprint(str(source)))
    assert engine.count_repetitions(start_sec=0, end_sec=2)["status"] == "in_progress"
    repetition._worker.join(3)
    result = engine.count_repetitions(start_sec=0, end_sec=2)
    assert result["status"] == "error"
    assert "ffmpeg decode timed out" in result["message"]


def test_decode_timeout_scales_with_window(tmp_path, monkeypatch):
    started = {}

    class FakeTimer:
        def __init__(self, seconds, _callback):
            started["seconds"] = seconds
        def start(self):
            raise RuntimeError("stop after timer setup")
        def cancel(self):
            pass

    real_popen = subprocess.Popen
    monkeypatch.setattr(repetition.subprocess, "Popen",
                        lambda _command, **kwargs: real_popen([sys.executable, "-c", "pass"], **kwargs))
    monkeypatch.setattr(repetition.threading, "Timer", FakeTimer)
    with pytest.raises(RuntimeError, match="stop after timer setup"):
        repetition._decode_frames(tmp_path / "long.mp4", 0, 300, 10, None)
    assert started["seconds"] == pytest.approx(600)


def test_temp_cleanup_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    def fail_decode(*_args, **_kwargs):
        temp_dirs = list(tmp_path.glob("repcount-*"))
        assert len(temp_dirs) == 1
        with (temp_dirs[0] / ".lock").open("a+b") as other_fd:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise RuntimeError("decode failed")

    monkeypatch.setattr(repetition, "_decode_frames", fail_decode)
    engine._meta = VideoMetadata(str(source), 2.0, 128, 72, 10.0, "mp4", 5, False,
                                 fingerprint=engine._cache.compute_fingerprint(str(source)))
    with pytest.raises(RuntimeError, match="decode failed"):
        repetition.count_video(engine, start_sec=0, end_sec=2, temp_root=tmp_path)
    assert not list(tmp_path.glob("repcount-*"))


def test_count_removes_only_stale_temp_dirs(tmp_path, monkeypatch):
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    engine._meta = VideoMetadata(str(source), 2.0, 128, 72, 10.0, "mp4", 5, False,
                                 fingerprint=engine._cache.compute_fingerprint(str(source)))
    cache_dir = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    stale = cache_dir / "repcount-stale"
    raced = cache_dir / "repcount-raced"
    recent = cache_dir / "repcount-recent"
    target = cache_dir / "keep-target"
    for path in (stale, raced, recent, target):
        path.mkdir()
    symlink = cache_dir / "repcount-link"
    symlink.symlink_to(target, target_is_directory=True)
    old = time.time() - repetition.ORPHAN_TEMP_AGE_SEC - 10
    os.utime(stale, (old, old))
    os.utime(raced, (old, old))
    os.utime(symlink, (old, old), follow_symlinks=False)
    real_rmtree = repetition.shutil.rmtree

    def raced_remove(path, *args, **kwargs):
        real_rmtree(path, *args, **kwargs)
        if Path(path) == raced:
            raise FileNotFoundError(path)

    monkeypatch.setattr(repetition.shutil, "rmtree", raced_remove)
    monkeypatch.setattr(repetition, "_decode_frames", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")))

    with pytest.raises(RuntimeError, match="decode failed"):
        repetition.count_video(engine, start_sec=0, end_sec=2)
    assert not stale.exists()
    assert not raced.exists()
    assert recent.is_dir()
    assert symlink.is_symlink()
    assert target.is_dir()


def test_stale_temp_dir_is_kept_while_locked_then_removed(tmp_path):
    stale = tmp_path / "repcount-active"
    stale.mkdir()
    old = time.time() - repetition.ORPHAN_TEMP_AGE_SEC - 10
    with (stale / ".lock").open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        os.utime(stale, (old, old))
        repetition._remove_orphaned_temp_dirs(tmp_path)
        assert stale.is_dir()

    repetition._remove_orphaned_temp_dirs(tmp_path)
    assert not stale.exists()


def test_generated_clip_sheet_cache_and_cleanup(tmp_path):
    from tests.fixtures.repetition_dev import generate
    case = ("integration", "ball", 3, .8, 0, -1, 0, False)
    source = tmp_path / "integration.mp4"
    generate(case, source)
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    result = repetition.count_video(engine, method="motion", temp_root=tmp_path)
    assert result["count"] == 3
    assert [cycle["frame_badge"] for cycle in result["cycles"]] == ["#1", "#2", "#3"]
    assert Path(result["contact_sheet"]["file_path"]).is_file()
    assert result == repetition.count_video(engine, method="motion", temp_root=tmp_path)
    focused = repetition.count_video(engine, method="motion", temp_root=tmp_path,
                                     region={"x": .2, "y": .05, "w": .6, "h": .9})
    assert focused["count"] == 3
    assert not list(tmp_path.glob("repcount-*"))


def test_video_three_complete_reps_then_partial_fourth(tmp_path):
    from tests.fixtures.repetition_dev import generate
    source = tmp_path / "partial_fourth.mp4"
    generate(("partial_fourth", "ball", 4, .8, 0, -1, 0, False, (0, .9)), source)
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    result = repetition.count_video(engine, method="motion")
    assert result["count"] == 3


@pytest.mark.parametrize("kind,count,period", [("ball", 6, .8), ("pendulum", 5, 1.0)])
def test_cropped_video_excludes_edge_reps(tmp_path, kind, count, period):
    from tests.fixtures.repetition_dev import generate
    case = (kind + "_edge_test", kind, count, period, 0, -1, 0, False, (.7, .7))
    source = tmp_path / (kind + ".mp4")
    truth = generate(case, source)
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    assert repetition.count_video(engine, method="motion")["count"] == truth


def test_window_budget_rejected_before_worker(tmp_path):
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    with pytest.raises(ValueError, match="split"):
        engine.count_repetitions(start_sec=0, end_sec=181)


def test_schema_and_mcp_wiring():
    from src.executor.mcp_server import TOOLS_MANIFEST
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "src/adapters/mcp_schemas.json").read_text())
    assert "count_repetitions" in schema["tools"]
    assert "count_repetitions" in {item["name"] for item in TOOLS_MANIFEST}
    for path in ("src/adapters/mcp_definitions.ts", "src/executor/types.d.ts"):
        assert "count_repetitions" in (root / path).read_text()


def test_mcp_result_attaches_peak_sheet(tmp_path, monkeypatch):
    from src.executor import mcp_server
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"jpeg")

    class FakeEngine:
        def __init__(self, source):
            assert source == "clip.mp4"

        def count_repetitions(self, **kwargs):
            assert kwargs["method"] == "motion"
            return {"status": "complete", "count": 1, "contact_sheet": {"file_path": str(sheet)}}

    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    result = mcp_server.handle_call("count_repetitions", {"source": "clip.mp4", "method": "motion"})
    assert result["data"]["contact_sheet"]["tool_file"]["_tag"] == "ToolFile"


def test_periodic_override_caps_confidence_and_states_both_counts():
    found = {"count": 14, "cycles": [], "period_sec": .6, "periodicity": .89}
    result = repetition._periodic_result(found)
    assert result["count"] == 14 and result["confidence"] == repetition.DISAGREE_CONFIDENCE
    weak = repetition._periodic_result({**found, "periodicity": .1})
    assert weak["confidence"] == .1  # a weak signal is not raised to the cap
    note = repetition._override_note(found, {"count": 4}, {"count": 4}, {"count": 5})
    assert "14 cycles" in note and "amplitude count of 4" in note and "motion 4, embedding 5" in note
    assert "confidence is capped" in note
    assert "embedding" not in repetition._override_note(found, {"count": 4}, {"count": 4}, None)
