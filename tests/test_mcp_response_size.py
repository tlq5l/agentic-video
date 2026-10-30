"""Responses must fit Executor's 10 MiB stdio message limit, and failures must say why."""

import base64
import io
import json
import subprocess

import numpy as np
from PIL import Image

import src.executor.mcp_server as mcp_server
from src.core.cache import MediaCache
from src.retrieval.engine import AgenticVideoEngine


def _noisy_jpeg(path, size=(1600, 1200)):
    pixels = np.random.default_rng(7).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, "JPEG", quality=95)
    return str(path)


def _patch_dispatch(monkeypatch, data_or_exc):
    def fake_dispatch(name, args):
        if isinstance(data_or_exc, Exception):
            raise data_or_exc
        return {"ok": True, "data": data_or_exc()}

    monkeypatch.setattr(mcp_server, "_dispatch", fake_dispatch)


def test_large_contact_sheet_is_downscaled_inline_and_keeps_full_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_VIDEO_INLINE_IMAGES", "1")
    sheet = _noisy_jpeg(tmp_path / "sheet.jpg")
    full_b64 = len(mcp_server.make_tool_file(sheet)["data"])
    limit = full_b64 // 3
    monkeypatch.setattr(mcp_server, "MAX_RESPONSE_BYTES", limit)
    _patch_dispatch(monkeypatch, lambda: {"file_path": sheet, "sample_count": 116, "tool_file": mcp_server.make_tool_file(sheet)})

    line = mcp_server.tool_call_response(1, "contact_sheet", {"source": "video.mp4"})

    assert len(line) <= limit
    result = json.loads(line)["result"]
    assert not result.get("isError")
    data = result["structuredContent"]
    assert data["file_path"] == sheet
    tool_file = data["tool_file"]
    with Image.open(io.BytesIO(base64.b64decode(tool_file["data"]))) as im:
        assert im.size[0] < 1600
    assert tool_file["byteLength"] == len(base64.b64decode(tool_file["data"]))
    assert "Downscaled from 1600x1200" in tool_file["note"]
    assert "downscaled" in data["note"]
    # The text copy carries no image bytes, so the image is not sent twice.
    text = result["content"][0]["text"]
    assert tool_file["data"] not in text
    assert mcp_server._TEXT_IMAGE_PLACEHOLDER in text


def test_small_images_are_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_VIDEO_INLINE_IMAGES", "1")
    sheet = _noisy_jpeg(tmp_path / "small.jpg", size=(64, 48))
    _patch_dispatch(monkeypatch, lambda: {"file_path": sheet, "tool_file": mcp_server.make_tool_file(sheet)})
    result = json.loads(mcp_server.tool_call_response(1, "contact_sheet", {"source": "video.mp4"}))["result"]
    tool_file = result["structuredContent"]["tool_file"]
    assert tool_file["name"] == "small.jpg"
    assert base64.b64decode(tool_file["data"]) == open(sheet, "rb").read()
    assert "note" not in result["structuredContent"]


def test_oversized_response_without_images_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(mcp_server, "MAX_RESPONSE_BYTES", 5_000)
    _patch_dispatch(monkeypatch, lambda: {"file_path": "x.jpg", "timestamps": list(range(5_000))})
    result = json.loads(mcp_server.tool_call_response(1, "contact_sheet", {"source": "video.mp4"}))["result"]
    assert result["isError"]
    text = result["content"][0]["text"]
    assert "MCP message limit" in text and "fewer samples" in text


def test_unexpected_exception_names_tool_type_and_next_step(monkeypatch):
    _patch_dispatch(monkeypatch, RuntimeError("pool broke"))
    result = json.loads(mcp_server.tool_call_response(1, "inspect_windows",
                                                      {"source": "v.mp4", "windows": [{"start_sec": 0, "end_sec": 1}]}))["result"]
    assert result["isError"]
    text = result["content"][0]["text"]
    assert "RuntimeError" in text and "`inspect_windows`" in text and "pool broke" in text
    assert "Retry once" in text


def test_slow_contact_sheet_stops_at_deadline_with_retry_advice(tmp_path, monkeypatch):
    import threading

    from src.core.errors import SubprocessTimeoutError
    from src.media.contact_sheet import build_contact_sheet

    release = threading.Event()

    def slow_extract(video_path, timestamp_sec, **kwargs):
        release.wait(5)
        raise AssertionError("not reached before the deadline")

    monkeypatch.setattr("src.media.frames.extract_single_frame", slow_extract)
    try:
        build_contact_sheet(tmp_path / "v.mp4", [1.0, 2.0], cache=MediaCache(tmp_path / "cache"), deadline_sec=0.2)
        raise AssertionError("expected a timeout")
    except SubprocessTimeoutError as exc:
        text = mcp_server.format_tool_error(exc)
        assert "Extracted 0 of 2 frames before the 0.2 s limit" in text
        assert "same call again continues" in text
    finally:
        release.set()


def test_inspect_window_composes_sheet_from_dense_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24:duration=3",
         "-c:v", "mpeg4", "-q:v", "5", str(video)],
        check=True, capture_output=True,
    )

    def no_per_frame_seek(*args, **kwargs):
        raise AssertionError("inspect_window must not seek each frame again with its own ffmpeg run")

    monkeypatch.setattr("src.media.frames.extract_single_frame", no_per_frame_seek)
    monkeypatch.setattr("src.retrieval.engine.build_contact_sheets", no_per_frame_seek)
    engine = AgenticVideoEngine(str(video), cache=MediaCache(tmp_path / "cache"))

    result = engine.inspect_window(0.5, 2.0, fps=4)

    assert result.frames
    with Image.open(result.contact_sheet_path) as sheet:
        assert sheet.size[0] > 160
