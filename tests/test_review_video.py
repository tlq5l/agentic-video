"""review_video end to end: a real tools/call on the 60 s fixture clip."""

import json
import subprocess
import time
from pathlib import Path

import pytest

from src.core.cache import MediaCache
from src.core.config import Config
from src.executor import mcp_server
from src.executor.mcp_server import handle_call

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_bench_60s.mp4"


def test_review_video_returns_all_four_views_of_a_real_clip_within_the_time_limit(tmp_path, monkeypatch):
    # Cold run: an empty cache, so the timing below is the worst case.
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")

    started = time.monotonic()
    response = json.loads(mcp_server.tool_call_response(1, "review_video", {"source": str(FIXTURE)}))
    elapsed = time.monotonic() - started

    result = response["result"]
    assert not result.get("isError"), result
    data = result["structuredContent"]
    assert data["metadata"]["duration_sec"] == 60.0 and data["metadata"]["has_audio"] is True
    assert data["contact_sheet"]["sample_count"] == 24
    assert Path(data["contact_sheet"]["file_path"]).is_file()
    assert data["contact_sheet"]["tool_file"]["_tag"] == "ToolFile"
    assert data["cuts_total"] == len(data["cuts"]) > 0
    assert data["loudness"]["peak_sec"] > 0 and data["loudness"]["outliers"]
    assert elapsed < 45, f"review_video took {elapsed:.1f} s; Executor cuts calls at 60 s"


def test_review_video_missing_file_is_a_tool_error():
    result = json.loads(mcp_server.tool_call_response(1, "review_video", {"source": "/nonexistent/clip.mp4"}))["result"]
    assert result["isError"] is True
    assert "MediaNotFoundError" in result["content"][0]["text"]


def _write_lossless_clip(path, segments):
    """A 30 fps FFV1 (lossless) clip joined from solid-colour segments given as (colour, seconds)."""
    inputs = []
    labels = []
    for index, (colour, seconds) in enumerate(segments):
        inputs.append(f"color=c={colour}:s=320x240:r=30:d={seconds}[s{index}]")
        labels.append(f"[s{index}]")
    graph = ";".join(inputs) + ";" + "".join(labels) + f"concat=n={len(segments)}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "ffv1", str(path)], check=True)
    return str(path)


def test_review_video_finds_a_black_stretch_and_ignores_a_dim_picture(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")
    grey = "0x3a3a3a"  # Y about 66: dim, but well above the near-black level
    black_clip = _write_lossless_clip(tmp_path / "black.mkv", [(grey, 5.0), ("black", 0.6), (grey, 6.4)])
    dim_clip = _write_lossless_clip(tmp_path / "dim.mkv", [(grey, 12.0)])

    black = handle_call("review_video", {"source": black_clip})
    assert black["ok"] is True, black
    data = black["data"]
    assert data["black_total"] == 1
    [stretch] = data["black_ranges"]
    frame = 1 / 30
    assert abs(stretch["start_sec"] - 5.0) <= frame + 1e-6
    assert abs(stretch["end_sec"] - 5.6) <= frame + 1e-6
    assert abs(stretch["duration_sec"] - 0.6) <= frame + 1e-6

    dim = handle_call("review_video", {"source": dim_clip})
    assert dim["ok"] is True, dim
    assert dim["data"]["black_ranges"] == []
    assert dim["data"]["black_total"] == 0


def test_review_video_replaces_a_damaged_black_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")
    clip = _write_lossless_clip(tmp_path / "solid.mkv", [("black", 2.0)])
    cache = MediaCache(base_dir=tmp_path / "cache")
    fingerprint = MediaCache.compute_fingerprint(clip)
    damaged_entries = [
        {},
        {"frame_rate": 30.0},
        {"frame_rate": 30.0, "ranges": [{"start_sec": "0"}]},
        {"frame_rate": 30.0, "ranges": [{"start_sec": 1.0, "end_sec": 0.5, "duration_sec": 0.5}]},
        {"frame_rate": 30.0, "ranges": [{"start_sec": 0.0, "end_sec": 2.0, "duration_sec": -2.0}]},
        {"frame_rate": 30.0, "ranges": [{"start_sec": 0.0, "end_sec": 0.1, "duration_sec": 0.1}]},
        {"frame_rate": 30.0, "ranges": [{"start_sec": 0.0, "end_sec": 0.199, "duration_sec": 0.199}]},
        {"frame_rate": 30.0, "ranges": [{"start_sec": -1.0, "end_sec": 1.0, "duration_sec": 2.0}]},
        {"frame_rate": 10 ** 400, "ranges": []},
        {"frame_rate": 30.0, "ranges": [{"start_sec": 0.0, "end_sec": 2.0, "duration_sec": 10 ** 400}]},
    ]
    for entry in damaged_entries:
        cache.save_json(fingerprint, "black_ranges_v2", entry)
        result = handle_call("review_video", {"source": clip})
        assert result["ok"] is True, result
        assert [(s["start_sec"], s["end_sec"], s["duration_sec"]) for s in result["data"]["black_ranges"]] == [(0.0, 2.0, 2.0)]
        assert cache.load_json(fingerprint, "black_ranges_v2")["ranges"] == result["data"]["black_ranges"]


def test_review_video_decoder_failure_is_an_error_and_leaves_no_black_result(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")
    clip = _write_lossless_clip(tmp_path / "solid.mkv", [("black", 2.0)])
    real_popen = subprocess.Popen

    def grid_decoder_stops_after_ten_frames(args, **kwargs):
        if any("scale=160:90:flags=area" in arg for arg in args):
            return real_popen(["sh", "-c", "head -c 144000 /dev/zero; exit 23"], **kwargs)
        return real_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", grid_decoder_stops_after_ten_frames)
    failed = json.loads(mcp_server.tool_call_response(1, "review_video", {"source": clip}))["result"]
    assert failed["isError"] is True, failed
    assert "MediaCorruptError" in failed["content"][0]["text"]
    assert MediaCache(base_dir=tmp_path / "cache").load_json(MediaCache.compute_fingerprint(clip), "black_ranges_v2") is None

    monkeypatch.setattr(subprocess, "Popen", real_popen)
    healthy = handle_call("review_video", {"source": clip})
    assert healthy["ok"] is True, healthy
    assert [(s["start_sec"], s["end_sec"]) for s in healthy["data"]["black_ranges"]] == [(0.0, 2.0)]


def test_black_stretch_across_a_scan_part_seam_is_listed_whole(tmp_path, monkeypatch):
    # The cut scan decodes a long file as parts, and the black levels come from that decode, so the
    # seams of the parts must not cut a stretch in two. The split is forced on this short clip.
    from src.media import scene

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(scene, "SCAN_SPLIT_MIN_SEC", 4)
    monkeypatch.setattr(scene, "SCAN_MIN_PART_SEC", 2)
    monkeypatch.setattr(scene.os, "cpu_count", lambda: 8)
    clip = _write_lossless_clip(tmp_path / "seam.mkv", [("0x3a3a3a", 5.8), ("black", 0.6), ("0x3a3a3a", 5.6)])
    assert len(scene._scan_plan(clip, 0.0, None)) > 1

    result = handle_call("review_video", {"source": clip})
    assert result["ok"] is True, result
    [stretch] = result["data"]["black_ranges"]
    frame = 1 / 30
    assert abs(stretch["start_sec"] - 5.8) <= frame + 1e-6
    assert abs(stretch["end_sec"] - 6.4) <= frame + 1e-6


@pytest.mark.parametrize("black_start_sec, jump_at_sec, jump_sec", [
    # The timestamps jump 0.2 s at the 4.0 s seam. The second part seeks to 3.0 s and decodes from there on,
    # so its frames cover the seam without a gap.
    (5.0, 4.0, 0.2),
    # The timestamps jump from 2.5 s to 5.2 s, across the 3.0 s seek point: the second part's first decoded
    # frame is at 5.2 s, after its own stretch starts at the 4.0 s seam. Its grid must not be used.
    (2.5, 2.5, 2.7),
], ids=["short-gap", "gap-across-the-seek-point"])
def test_black_stretch_after_a_timestamp_gap_at_a_scan_part_seam_is_timed_the_same_in_both_cache_orders(
        tmp_path, monkeypatch, black_start_sec, jump_at_sec, jump_sec):
    # The black stretch (5.2-5.8 s) lies after the jump. Cold review (split scan) and cuts-first review
    # (one decode) must both list it at its real time.
    from src.media import scene

    clip = tmp_path / "gap.mkv"
    graph = (f"color=c=0x3a3a3a:s=320x240:r=30:d={black_start_sec}[s0];color=c=black:s=320x240:r=30:d=0.6[s1];"
             "color=c=0x3a3a3a:s=320x240:r=30:d=2.4[s2];[s0][s1][s2]concat=n=3:v=1:a=0,"
             rf"setpts=PTS+({jump_sec}/TB)*gte(T\,{jump_at_sec})[out]")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "ffv1", str(clip)], check=True)
    monkeypatch.setattr(scene, "SCAN_SPLIT_MIN_SEC", 4)
    monkeypatch.setattr(scene, "SCAN_MIN_PART_SEC", 4)
    monkeypatch.setattr(scene.os, "cpu_count", lambda: 8)
    assert len(scene._scan_plan(str(clip), 0.0, None)) > 1

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cold")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")
    cold = handle_call("review_video", {"source": str(clip)})
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cuts_first")
    assert handle_call("detect_cuts", {"source": str(clip)})["ok"] is True
    cuts_first = handle_call("review_video", {"source": str(clip)})

    frame = 1 / 30
    for result in (cold, cuts_first):
        assert result["ok"] is True, result
        [stretch] = result["data"]["black_ranges"]
        assert abs(stretch["start_sec"] - 5.2) <= frame + 1e-6
        assert abs(stretch["end_sec"] - 5.8) <= frame + 1e-6
    assert cold["data"]["black_ranges"] == cuts_first["data"]["black_ranges"]
