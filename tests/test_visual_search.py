"""Visual retrieval tests use tiny local FFmpeg clips and a deterministic encoder."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from src.core.cache import MediaCache
from src.retrieval.engine import AgenticVideoEngine
from src.retrieval import visual


def _make_video(path: Path):
    """Create three solid-color scenes so expected visual matches are explicit."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for visual-index integration tests")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x90:r=2:d=4",
        "-f", "lavfi", "-i", "color=c=green:s=160x90:r=2:d=4",
        "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=2:d=4",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-threads", "1", str(path),
    ], check=True, timeout=30)
    return path


class ColorEncoder:
    """Tiny deterministic stand-in for CLIP; image features are RGB channels."""
    calls = 0

    def images(self, paths):
        from PIL import Image
        assert len(paths) <= visual.BATCH_SIZE
        self.calls += len(paths)
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            # Green's RGB value exceeds blue/red equally, so normalize raw channels.
            rows.append(rgb / max(float(np.linalg.norm(rgb)), 1e-12))
        return np.asarray(rows, dtype=np.float32)

    def text(self, query):
        color = query.lower()
        vector = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}[color]
        result = np.asarray(vector, dtype=np.float32)
        return result / np.linalg.norm(result)


@pytest.fixture
def indexed_engine(tmp_path, monkeypatch):
    path = _make_video(tmp_path / "colors.mp4")
    cache = MediaCache(tmp_path / "cache")
    engine = AgenticVideoEngine(str(path), cache=cache)
    encoder = ColorEncoder()
    encoder.calls = 0
    monkeypatch.setattr(visual, "get_encoder", lambda: encoder)
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    visual.build_index(engine)
    return engine, encoder


def test_index_build_is_chunked_resumable_and_reuses_completed_cache(tmp_path, monkeypatch):
    path = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(path), cache=MediaCache(tmp_path / "cache"))
    encoder = ColorEncoder()
    encoder.calls = 0
    monkeypatch.setattr(visual, "get_encoder", lambda: encoder)
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    real_sample = visual.sample_chunk
    boundaries = []

    def one_shot_fail_second_chunk(*args, **kwargs):
        start = kwargs["start_sec"]
        boundaries.append(start)
        frames = real_sample(*args, **kwargs)
        if start == 4:  # chunks decode ahead in parallel, so pick the chunk by start, not by call order
            raise RuntimeError("synthetic interruption")
        return frames

    monkeypatch.setattr(visual, "sample_chunk", one_shot_fail_second_chunk)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        visual.build_index(engine)
    assert {0.0, 4.0} <= set(boundaries)
    assert not list((tmp_path / "cache").rglob("*.jpg"))
    assert not list((tmp_path / "cache").rglob("visual-chunk-*"))
    # The committed first chunk survives; a later call resumes at its cursor.
    resumed = []

    def record_resume(*args, **kwargs):
        resumed.append(kwargs["start_sec"])
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(visual, "sample_chunk", record_resume)
    result = visual.build_index(engine)
    assert result["status"] == "complete"
    assert result["frames_indexed"] == 3  # one kept frame per solid-colour scene; the rest are duplicates
    assert result["progress"] == pytest.approx(1.0)
    indexed_calls = encoder.calls
    visual.build_index(engine)
    assert encoder.calls == indexed_calls
    assert resumed[0] == 4.0 and 0.0 not in resumed
    assert not list((tmp_path / "cache").rglob("*.jpg"))
    assert not list((tmp_path / "cache").rglob("visual-chunk-*"))
    with visual._connect(engine) as conn:
        assert all(row[0] == "" for row in conn.execute("SELECT path FROM frames"))


def test_build_removes_orphaned_chunks_and_stale_search_dirs_before_sampling(tmp_path, monkeypatch):
    path = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(path), cache=MediaCache(tmp_path / "cache"))
    entry = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    stale_chunk = entry / "visual-chunk-abandoned"
    recent_chunk = entry / "visual-chunk-just-killed"
    stale_search = entry / "visual-search-abandoned"
    recent_search = entry / "visual-search-active"
    unrelated = entry / "other-cache-dir"
    for directory in (stale_chunk, recent_chunk, stale_search, recent_search, unrelated):
        directory.mkdir()
        (directory / "sentinel.txt").write_text("keep or remove")
    old = time.time() - visual.ORPHAN_TEMP_AGE_SEC - 60
    for directory in (stale_chunk, stale_search, unrelated):
        os.utime(directory, (old, old))

    real_sample = visual.sample_chunk

    def check_cleanup(*args, **kwargs):
        assert not stale_chunk.exists()
        # Only the lock holder creates chunk dirs, so a recent one is still orphaned.
        assert not recent_chunk.exists()
        assert not stale_search.exists()
        assert (recent_search / "sentinel.txt").exists()
        assert (unrelated / "sentinel.txt").exists()
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(visual, "get_encoder", lambda: ColorEncoder())
    monkeypatch.setattr(visual, "sample_chunk", check_cleanup)
    assert visual.build_index(engine)["status"] == "complete"


def test_search_removes_stale_search_dirs_but_not_chunks_or_recent_searches(indexed_engine):
    engine, _ = indexed_engine
    entry = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    stale_search = entry / "visual-search-killed"
    recent_search = entry / "visual-search-concurrent"
    chunk = entry / "visual-chunk-builder"
    for directory in (stale_search, recent_search, chunk):
        directory.mkdir()
        (directory / "sentinel.txt").write_text("keep or remove")
    old = time.time() - visual.ORPHAN_TEMP_AGE_SEC - 60
    for directory in (stale_search, chunk):
        os.utime(directory, (old, old))
    assert visual.search_index(engine, "red")["hits"]
    assert not stale_search.exists()
    # Searches do not hold the build lock, so chunk dirs may belong to a live builder.
    assert (chunk / "sentinel.txt").exists()
    assert (recent_search / "sentinel.txt").exists()


def test_visual_search_ranks_color_bounds_merges_intervals_and_makes_sheet(indexed_engine):
    engine, _ = indexed_engine
    result = visual.search_index(engine, "red", top_k=20)
    assert result["status"] == "complete"
    assert result["hits"]
    assert result["hits"][0]["timestamp_sec"] < 4.1
    assert result["hits"][0]["score"] > 0.9
    assert result["intervals"]
    assert result["intervals"][0]["start_sec"] == pytest.approx(0.0, abs=0.6)
    assert Path(result["contact_sheet"]["file_path"]).is_file()

    green = visual.search_index(engine, "green", top_k=20, start_sec=4.0, end_sec=8.0)
    assert green["hits"]
    assert all(4.0 <= hit["timestamp_sec"] < 8.0 for hit in green["hits"])
    with pytest.raises(ValueError, match="range"):
        visual.search_index(engine, "blue", start_sec=8.0, end_sec=4.0)


def test_mcp_visual_tools_poll_and_attach_contact_sheet(tmp_path, monkeypatch):
    from src.executor import mcp_server

    source = _make_video(tmp_path / "colors.mp4")
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake jpeg for ToolFile")
    calls = {"build": 0, "search": 0}

    class FakeEngine:
        def __init__(self, _source):
            pass

        def build_visual_index(self):
            calls["build"] += 1
            return {"status": "queued" if calls["build"] == 1 else "complete", "progress": 1.0}

        def search_visual(self, query, top_k=5, start_sec=0.0, end_sec=None):
            calls["search"] += 1
            if calls["search"] == 1:
                return {"status": "running", "progress": 0.5}
            return {"status": "complete", "hits": [{"timestamp_sec": 1.0}], "intervals": [],
                    "contact_sheet": {"file_path": str(sheet)}}

    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    first = mcp_server.handle_call("build_visual_index", {"source": str(source)})
    second = mcp_server.handle_call("build_visual_index", {"source": str(source)})
    assert first["data"]["status"] == "queued"
    assert second["data"]["status"] == "complete"
    args = {"source": str(source), "query": "red", "top_k": 3, "start_sec": 0, "end_sec": 4}
    assert mcp_server.handle_call("search_visual", args)["data"]["status"] == "running"
    result = mcp_server.handle_call("search_visual", args)["data"]
    assert result["status"] == "complete"
    assert result["contact_sheet"]["tool_file"]["_tag"] == "ToolFile"


def test_worker_returns_promptly_deduplicates_and_reports_errors(tmp_path, monkeypatch):
    import threading
    source = tmp_path / "dummy.mp4"
    source.write_bytes(b"dummy")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_build(engine, report):
        calls.append(1)
        report({"indexed_until_sec": 10})
        entered.set()
        assert release.wait(3)
        raise RuntimeError("test interruption")

    monkeypatch.setattr(visual, "build_index", slow_build)
    try:
        assert engine.build_visual_index()["status"] == "queued"
        assert entered.wait(2)
        progress = engine.build_visual_index()
        assert progress["status"] == "running"
        assert progress["indexed_until_sec"] == 10
        assert engine.search_visual("red")["status"] == "queued"
        assert len(calls) == 1
    finally:
        release.set()
        visual._worker.join(3)
        visual._search_worker.join(3)
    error = engine.build_visual_index()
    assert error["status"] == "error"
    assert "test interruption" in error["message"]
    # Consuming an error allows a real retry, rather than leaving a stuck job.
    monkeypatch.setattr(visual, "build_index", lambda *args: {"status": "complete"})
    assert engine.build_visual_index()["status"] == "queued"
    visual._worker.join(3)
    assert engine.build_visual_index()["status"] == "complete"


@pytest.mark.parametrize("args", [
    {"query": ""}, {"query": "red", "top_k": 21},
    {"query": "red", "top_k": True}, {"query": "red", "top_k": 1.5},
    {"query": "red", "start_sec": float("nan")},
    {"query": "red", "end_sec": float("inf")},
    {"query": "red", "start_sec": -1},
])
def test_invalid_search_rejected_before_work(tmp_path, args):
    engine = AgenticVideoEngine("does-not-exist.mp4", cache=MediaCache(tmp_path / "cache"))
    with pytest.raises(ValueError):
        engine.search_visual(**args)


def test_empty_index_does_not_load_model(tmp_path, monkeypatch):
    source = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    def no_model():
        pytest.fail("empty index should not need a model")
    monkeypatch.setattr(visual, "get_encoder", no_model)
    stale_search = engine._cache.get_entry_dir(engine.metadata.fingerprint) / "visual-search-killed"
    stale_search.mkdir()
    old = time.time() - visual.ORPHAN_TEMP_AGE_SEC - 60
    os.utime(stale_search, (old, old))
    result = visual.search_index(engine, "red")
    # Early returns still clear stale dirs left by killed searches.
    assert not stale_search.exists()
    assert result["status"] == "index_required"
    assert result["index_status"] == "missing"
    assert result["indexed_until_sec"] == 0
    assert result["duration_sec"] == pytest.approx(engine.metadata.duration_sec)
    assert result["frames_indexed"] == 0
    assert result["retry_after_seconds"] > 0
    assert "build_visual_index" in result["message"]


def test_scene_cuts_supplement_uniform_samples_and_short_tail(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    source = tmp_path / "short.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x90:r=20:d=1.25",
        "-f", "lavfi", "-i", "color=blue:s=160x90:r=20:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-threads", "1", str(source),
    ], check=True, timeout=30)
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    monkeypatch.setattr(visual, "get_encoder", lambda: ColorEncoder())
    monkeypatch.setattr(visual, "CHUNK_SEC", 2.0)
    result = visual.build_index(engine)
    assert result["progress"] == 1.0
    search = visual.search_index(engine, "blue", top_k=20)
    times = [hit["timestamp_sec"] for hit in search["hits"]]
    assert any(abs(t - 1.25) < .1 for t in times)
    assert any(t >= 2 for t in times)


def test_video_ending_within_one_frame_after_a_chunk_mark_still_indexes(tmp_path, monkeypatch):
    """60 frames at 30000/1001 fps last 2.002 s but the last frame starts at 1.969 s, so the 2.0 s chunk
    mark leaves a tail window that holds no frame. The last frame must cover that tail, not fail."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    source = tmp_path / "ntsc.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30000/1001",
        "-frames:v", "60", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", str(source),
    ], check=True, timeout=30)
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    assert 2.0 < engine.metadata.duration_sec < 2.0 + 1 / 29
    monkeypatch.setattr(visual, "get_encoder", lambda: ColorEncoder())
    monkeypatch.setattr(visual, "CHUNK_SEC", 2.0)
    result = visual.build_index(engine)
    assert result["status"] == "complete"
    assert result["progress"] == 1.0
    assert result["indexed_until_sec"] == pytest.approx(engine.metadata.duration_sec)


def test_search_uses_database_thumbnail_without_persistent_frames(indexed_engine):
    engine, _ = indexed_engine
    with visual._connect(engine) as conn:
        assert conn.execute("SELECT path FROM frames ORDER BY timestamp LIMIT 1").fetchone()[0] == ""
    result = visual.search_index(engine, "red", top_k=20)
    assert result["status"] == "complete"
    assert Path(result["contact_sheet"]["file_path"]).is_file()
    assert not list(engine._cache.base_dir.rglob("visual-search-*"))
    assert not list(engine._cache.base_dir.rglob("visual-chunk-*"))


def test_search_cleans_temporary_jpeg_after_decode_error(indexed_engine):
    engine, _ = indexed_engine
    with visual._connect(engine) as conn:
        timestamp = conn.execute("SELECT timestamp FROM frames ORDER BY timestamp LIMIT 1").fetchone()[0]
        conn.execute("UPDATE frames SET thumbnail=? WHERE timestamp=?", (b"not a jpeg", timestamp))
        conn.commit()
    with pytest.raises(OSError):
        visual.search_index(engine, "red", top_k=20)
    assert not list(engine._cache.base_dir.rglob("visual-search-*"))


def test_static_scene_detection_bounds_input_decode(tmp_path, monkeypatch):
    """No output frames must not make FFmpeg scan a long silent source."""
    import re
    from src.media import scene
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    source = tmp_path / "long-static.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=black:s=32x32:r=5:d=120", "-an", "-c:v", "libx264", "-threads", "1", str(source),
    ], check=True, timeout=30)
    original_run = subprocess.run
    decoded = []

    def measured_run(cmd, **kwargs):
        assert cmd.index("-t") < cmd.index("-i")
        assert "-filter_complex_threads" not in cmd
        result = original_run([cmd[0], "-loglevel", "debug", *cmd[1:]], **kwargs)
        decoded.extend(int(value) for value in re.findall(r"(\d+) frames decoded", result.stderr))
        return result

    monkeypatch.setattr(scene.subprocess, "run", measured_run)
    cuts = scene.detect_scenes(str(source), duration_sec=2, cache=MediaCache(tmp_path / "cache"))
    assert cuts == []
    assert decoded and max(decoded) < 30  # ten requested frames, plus decoder buffering


def test_new_process_resumes_after_abrupt_worker_exit(tmp_path, monkeypatch):
    import sys
    source = _make_video(tmp_path / "colors.mp4")
    cache_path = tmp_path / "cache"
    script = '''
import os, runpy, sys
from src.core.cache import MediaCache
from src.retrieval.engine import AgenticVideoEngine
from src.retrieval import visual
helpers = runpy.run_path("tests/test_visual_search.py")
visual._encoder = helpers["ColorEncoder"]()
visual.CHUNK_SEC = 4.0
engine = AgenticVideoEngine(sys.argv[1], cache=MediaCache(sys.argv[2]))
def stop_after_commit(progress):
    if progress["indexed_until_sec"] >= 4:
        os._exit(7)
visual.build_index(engine, stop_after_commit)
'''
    stopped = subprocess.run([sys.executable, "-c", script, str(source), str(cache_path)],
                             cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert stopped.returncode == 7
    engine = AgenticVideoEngine(str(source), cache=MediaCache(cache_path))
    encoder = ColorEncoder()
    encoder.calls = 0
    monkeypatch.setattr(visual, "get_encoder", lambda: encoder)
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    progress = []
    result = visual.build_index(engine, progress.append)
    assert progress[0]["indexed_until_sec"] == 4
    assert result["status"] == "complete"
    # Only new frames (including scene samples) were embedded after restart.
    assert encoder.calls == result["frames_indexed"] - progress[0]["frames_indexed"]


def test_partial_search_reads_committed_prefix_during_slow_build(tmp_path, monkeypatch):
    import threading
    source = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    encoder = ColorEncoder()
    entered, release = threading.Event(), threading.Event()
    real_sample = visual.sample_chunk

    def slow_second_chunk(*args, **kwargs):
        if kwargs["start_sec"] >= 4:
            entered.set()
            assert release.wait(5)
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(visual, "get_encoder", lambda: encoder)
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    monkeypatch.setattr(visual, "sample_chunk", slow_second_chunk)
    worker = threading.Thread(target=visual.build_index, args=(engine,))
    worker.start()
    try:
        assert entered.wait(4)
        # The second chunk decodes while the first is embedded, so wait for the first commit.
        deadline = time.time() + 4
        result = visual.search_index(engine, "red", top_k=5)
        while result["status"] == "index_required" and time.time() < deadline:
            time.sleep(0.05)
            result = visual.search_index(engine, "red", top_k=5)
        assert result["status"] == "partial"
        assert result["index_status"] == "partial"
        assert result["indexed_until_sec"] == 4.0
        assert result["retry_after_seconds"] > 0
        assert "Later moments" in result["message"]
        assert result["hits"]
        # Intervals must not claim footage past the committed prefix.
        assert all(interval["end_sec"] <= 4.0 for interval in result["intervals"])
        stale_search = engine._cache.get_entry_dir(engine.metadata.fingerprint) / "visual-search-killed"
        stale_search.mkdir()
        old = time.time() - visual.ORPHAN_TEMP_AGE_SEC - 60
        os.utime(stale_search, (old, old))
        later = visual.search_index(engine, "blue", start_sec=8.0)
        assert later["status"] == "partial"
        assert later["hits"] == []
        # A search with no hits still cleans up after killed searches.
        assert not stale_search.exists()
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()


def test_second_process_builder_gets_busy_progress_without_sqlite_lock_error(tmp_path, monkeypatch):
    import sys
    source = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    monkeypatch.setattr(visual, "get_encoder", lambda: ColorEncoder())
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    # Establish a committed prefix and leave the index incomplete.
    def stop_after_first(progress):
        if progress["indexed_until_sec"] >= 4:
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        visual.build_index(engine, stop_after_first)
    lock_path = engine._cache.get_entry_dir(engine.metadata.fingerprint) / f"{visual.INDEX_NAME}.lock"
    script = "import fcntl,sys,time; f=open(sys.argv[1],'a+b'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(5)"
    holder = subprocess.Popen([sys.executable, "-c", script, str(lock_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with visual._connect(engine) as writer:
            writer.execute("BEGIN IMMEDIATE")
            result = visual.build_index(engine)
            assert result["status"] == "busy"
            assert result["indexed_until_sec"] == 4.0
            assert result["retry_after_seconds"] > 0
            assert visual.search_index(engine, "red")["status"] == "partial"
            writer.rollback()
    finally:
        holder.terminate()
        holder.wait(timeout=3)


def test_index_ffmpeg_calls_limit_threads(tmp_path, monkeypatch):
    from src.media import sampling, frames, scene
    source = _make_video(tmp_path / "colors.mp4")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    monkeypatch.setattr(visual, "get_encoder", lambda: ColorEncoder())
    monkeypatch.setattr(visual, "CHUNK_SEC", 4.0)
    original_run = subprocess.run
    commands = []

    def check_threads(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            commands.append(cmd)
            limit = str(visual.FFMPEG_THREADS)
            assert cmd[cmd.index("-threads") + 1] == limit
            assert cmd[cmd.index("-filter_threads") + 1] == limit
            if cmd[-1].endswith(".jpg"):
                # Output-side -threads (after -i) bounds the JPEG encoder too.
                output_args = cmd[cmd.index("-i") + 2:]
                assert output_args[output_args.index("-threads") + 1] == limit
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(sampling.subprocess, "run", check_threads)
    monkeypatch.setattr(frames.subprocess, "run", check_threads)
    monkeypatch.setattr(scene.subprocess, "run", check_threads)
    monkeypatch.setattr(visual.subprocess, "run", check_threads)
    visual.build_index(engine)
    assert len(commands) >= 3
