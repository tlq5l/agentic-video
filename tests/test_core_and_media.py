from pathlib import Path
from src.core.config import Config
from src.core.cache import MediaCache
from src.media.probe import probe_video
from src.media.frames import extract_single_frame, extract_window_frames
from src.media.contact_sheet import build_contact_sheet, generate_video_overview
from src.transcript.asr import parse_srt_or_vtt
from src.transcript.search import search_transcript
from src.retrieval.engine import AgenticVideoEngine
from src.executor.mcp_server import handle_call

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_VIDEO = str(FIXTURE_DIR / "synthetic_bench_60s.mp4")
FIXTURE_SRT = str(FIXTURE_DIR / "synthetic_bench_60s.srt")

def test_cache_fingerprint():
    cache = MediaCache()
    fp = cache.compute_fingerprint(FIXTURE_VIDEO)
    assert len(fp) >= 16
    fp2 = cache.compute_fingerprint(FIXTURE_VIDEO)
    assert fp == fp2

def test_probe_video():
    meta = probe_video(FIXTURE_VIDEO)
    assert meta.duration_sec >= 59.0
    assert meta.width == 640
    assert meta.height == 360
    assert meta.fps >= 24.0
    assert meta.has_audio is True
    assert meta.video_codec is not None

def test_extract_single_frame(tmp_path):
    out_file = tmp_path / "frame_14s.jpg"
    frame = extract_single_frame(FIXTURE_VIDEO, timestamp_sec=14.0, output_path=out_file)
    assert frame.file_path == str(out_file.resolve())
    assert Path(frame.file_path).exists()
    assert frame.width in (Config.DEFAULT_FRAME_WIDTH, 640)

def test_extract_window_frames():
    frames = extract_window_frames(FIXTURE_VIDEO, start_sec=13.0, end_sec=15.0, fps=2.0)
    assert len(frames) >= 4
    for f in frames:
        assert Path(f.file_path).exists()
        assert f.timecode != ""

def test_contact_sheet():
    timestamps = [7.0, 14.0, 21.0, 26.0, 34.0, 39.5, 48.0]
    sheet = build_contact_sheet(FIXTURE_VIDEO, timestamps=timestamps, cols=4)
    assert Path(sheet.file_path).exists()
    assert sheet.sample_count == len(timestamps)
    assert sheet.rows == 2
    assert sheet.cols == 4

def test_overview_generation():
    overview = generate_video_overview(FIXTURE_VIDEO, samples=8, cols=4)
    assert Path(overview.file_path).exists()
    assert overview.sample_count == 8

def test_transcript_srt_parsing():
    segments = parse_srt_or_vtt(Path(FIXTURE_SRT))
    assert len(segments) >= 5
    cygnus_seg = next((s for s in segments if "Cygnus" in s.text), None)
    assert cygnus_seg is not None
    assert 33.0 <= cygnus_seg.start_sec <= 35.0

def test_search_transcript():
    matches = search_transcript(FIXTURE_VIDEO, query="Cygnus", top_k=3)
    assert len(matches) > 0
    assert "Cygnus" in matches[0].text
    assert 33.0 <= matches[0].start_sec <= 35.0

def test_agentic_retrieval_engine():
    engine = AgenticVideoEngine(FIXTURE_VIDEO)
    meta = engine.probe()
    assert meta.duration_sec >= 59.0
    
    # Transcript search
    hits = engine.search_transcript("Supernova")
    assert len(hits) > 0
    
    # Dense inspect around supernova
    window = engine.inspect_window(start_sec=13.5, end_sec=15.5, fps=4.0)
    assert len(window.frames) >= 8
    assert window.contact_sheet_path is not None
    assert Path(window.contact_sheet_path).exists()

def test_mcp_server_dispatch():
    # probe_video
    res = handle_call("probe_video", {"source": FIXTURE_VIDEO})
    assert res["ok"] is True
    assert res["data"]["width"] == 640
    
    # contact_sheet
    res = handle_call("contact_sheet", {"source": FIXTURE_VIDEO, "samples": 6, "cols": 3})
    assert res["ok"] is True
    assert "tool_file" in res["data"]
    assert res["data"]["tool_file"]["_tag"] == "ToolFile"
    assert "samples_clamped_from" not in res["data"]

    # contact_sheet clamps oversized sample requests instead of failing
    res = handle_call("contact_sheet", {"source": FIXTURE_VIDEO, "samples": 600, "cols": 10})
    assert res["ok"] is True
    assert res["data"]["sample_count"] == 150
    assert res["data"]["samples_clamped_from"] == 600
    assert "clamped" in res["data"]["note"]
    
    # search_transcript
    res = handle_call("search_transcript", {"source": FIXTURE_VIDEO, "query": "Cygnus"})
    assert res["ok"] is True
    assert len(res["data"]) > 0
    
    # inspect_window
    res = handle_call("inspect_window", {"source": FIXTURE_VIDEO, "start_sec": 14.0, "end_sec": 15.0, "fps": 2.0})
    assert res["ok"] is True
    assert "tool_file" in res["data"]
