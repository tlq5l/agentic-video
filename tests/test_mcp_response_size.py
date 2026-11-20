"""Responses must fit Executor's 10 MiB stdio message limit, and failures must say why."""

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

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

def _run_batch(batch, tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "LANG": "C.UTF-8"}
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=json.dumps(batch) + "\n", capture_output=True,
                          text=True, cwd=Path(__file__).resolve().parents[1], env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    lines = done.stdout.splitlines()
    assert len(lines) == 1, len(lines)
    return lines[0]


def _reply_of_size(req_id, size):
    base = {"jsonrpc": "2.0", "id": req_id, "result": {"data": ""}}
    pad = size - len(json.dumps(base))
    assert pad >= 0
    reply = {"jsonrpc": "2.0", "id": req_id, "result": {"data": "a" * pad}}
    assert len(json.dumps(reply)) == size
    return reply


def _ping_reply(req_id):
    reply = {"jsonrpc": "2.0", "id": req_id, "result": {}}
    assert len(json.dumps(reply)) == 41
    return reply


def test_batch_of_big_replies_stays_under_the_message_limit_end_to_end(tmp_path):
    """A JSON-RPC batch is one stdout line. Each reply here fits under MAX_RESPONSE_BYTES, but three of them added up
    printed a 9 MB line. The batch line must stay under the limit and still answer every id."""
    limit = 8_000_000
    # An unknown tool name is echoed into its error reply, so each reply is about 3 MB on its own.
    batch = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": "x" * 3_000_000 + str(i), "arguments": {}}}
             for i in (1, 2, 3)]
    line = _run_batch(batch, tmp_path)
    assert len(line.encode("utf-8")) <= limit, len(line.encode("utf-8"))
    replies = json.loads(line)
    assert [reply["id"] for reply in replies] == [1, 2, 3], replies


def test_batch_first_reply_just_under_the_limit_keeps_it_and_answers_the_rest_end_to_end(tmp_path):
    """Breaker repro for PR #141 (FAIL F1). Reply 1 is 7,999,912 bytes, so it fits on its own. Reply 2 (3 MB) is swapped for its
    small error, and that small error was never checked against the batch total: the line went over the limit and the whole
    batch became one id-null error. Every id must still get its own reply inside the limit."""
    limit = 8_000_000
    name_1 = "x" * 7_999_200
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name_1, "arguments": {}}},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "x" * 3_000_000, "arguments": {}}}]
    line = _run_batch(batch, tmp_path)
    assert len(line.encode("utf-8")) <= limit, len(line.encode("utf-8"))
    replies = json.loads(line)
    assert isinstance(replies, list), replies
    assert [reply["id"] for reply in replies] == [1, 2], [reply.get("id") for reply in replies]
    # Reply 1 (7,999,912 bytes) cannot sit beside reply 2's small error inside the limit, so it becomes its small error;
    # reply 2 (3 MB) fits alone and stays whole.
    assert "over the 8 MB MCP message limit" in replies[0]["result"]["content"][0]["text"]
    assert "x" * 3_000_000 in replies[1]["result"]["content"][0]["text"]


def test_batch_large_reply_followed_by_ping_keeps_both_whole_end_to_end(tmp_path):
    """A 7,999,900-byte reply followed by a 41-byte ping fits under the 8 MB limit and must keep both replies whole."""
    limit = mcp_server.MAX_RESPONSE_BYTES
    overhead = len(mcp_server.tool_call_response(1, "", {}))
    name_1 = "x" * (7_999_900 - overhead)
    assert len(mcp_server.tool_call_response(1, name_1, {})) == 7_999_900
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name_1, "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ]
    line = _run_batch(batch, tmp_path)
    assert len(line.encode("utf-8")) <= limit, len(line.encode("utf-8"))
    replies = json.loads(line)
    assert isinstance(replies, list), replies
    assert [reply["id"] for reply in replies] == [1, 2], [reply.get("id") for reply in replies]
    assert name_1 in replies[0]["result"]["content"][0]["text"]
    assert "over the 8 MB MCP message limit" not in replies[0]["result"]["content"][0]["text"]
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_reply_line_reserves_smaller_form_for_later_replies():
    limit = mcp_server.MAX_RESPONSE_BYTES

    # [7,999,900 B, 41 B]: both fit whole (7,999,944 B <= 8,000,000 B).
    batch_one_ping = [_reply_of_size(1, 7_999_900), _ping_reply(2)]
    line = mcp_server._reply_line(batch_one_ping)
    assert len(line.encode("utf-8")) <= limit
    assert json.loads(line) == batch_one_ping

    # [7,999,700 B, 41 B, 41 B]: all three fit whole (7,999,786 B <= 8,000,000 B).
    batch_two_pings = [_reply_of_size(1, 7_999_700), _ping_reply(2), _ping_reply(3)]
    line = mcp_server._reply_line(batch_two_pings)
    assert len(line.encode("utf-8")) <= limit
    assert json.loads(line) == batch_two_pings

    # Exact limit boundary: 2 + 7,999,956 + 1 + 41 == 8,000,000 -> both whole.
    batch_exact = [_reply_of_size(1, limit - 44), _ping_reply(2)]
    line = mcp_server._reply_line(batch_exact)
    assert len(line.encode("utf-8")) == limit
    assert json.loads(line) == batch_exact


def test_reply_line_swaps_first_reply_when_it_truly_cannot_fit():
    limit = mcp_server.MAX_RESPONSE_BYTES

    # 1 byte past the exact boundary with a 41-byte follower: reply 1 cannot fit and becomes its small error.
    ping_2 = _ping_reply(2)
    line = mcp_server._reply_line([_reply_of_size(1, limit - 43), ping_2])
    assert len(line.encode("utf-8")) <= limit
    replies = json.loads(line)
    assert [reply["id"] for reply in replies] == [1, 2]
    assert replies[0]["result"]["isError"] is True
    assert mcp_server._BATCH_OVERFLOW_TEXT in replies[0]["result"]["content"][0]["text"]
    assert replies[1] == ping_2

    # Reply 2 is larger than its small error (1,000 B > 186 B), so reply 1 (7,999,900 B) cannot fit even beside
    # reply 2's small error (2 + 7,999,900 + 1 + 186 = 8,000,089 > 8,000,000).
    second = _reply_of_size(2, 1_000)
    line = mcp_server._reply_line([_reply_of_size(1, 7_999_900), second])
    assert len(line.encode("utf-8")) <= limit
    replies = json.loads(line)
    assert [reply["id"] for reply in replies] == [1, 2]
    assert replies[0]["result"]["isError"] is True
    assert mcp_server._BATCH_OVERFLOW_TEXT in replies[0]["result"]["content"][0]["text"]
    assert replies[1] == second


def test_reply_line_keeps_small_replies_whole_and_falls_back_when_still_over_limit(monkeypatch):
    # Four 41-byte pings total 169 bytes (2 + 4*41 + 3). Under a 200-byte limit, each reply is smaller than its
    # 186-byte small error, so all four must be kept whole rather than swapped and overflowing.
    monkeypatch.setattr(mcp_server, "MAX_RESPONSE_BYTES", 200)
    four_pings = [_ping_reply(i) for i in (1, 2, 3, 4)]
    line = mcp_server._reply_line(four_pings)
    assert len(line.encode("utf-8")) <= 200
    assert json.loads(line) == four_pings

    # A reply equal in size to its small-error form (186 B) is kept whole too (2 + 186 == 188 <= 200).
    small_len = len(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": mcp_server._error_result(mcp_server._BATCH_OVERFLOW_TEXT),
    }))
    equal_reply = _reply_of_size(1, small_len)
    line = mcp_server._reply_line([equal_reply])
    assert len(line.encode("utf-8")) <= 200
    assert json.loads(line) == [equal_reply]

    # Five 41-byte pings total 211 bytes (> 200), so the batch still overflows into the single id-null error.
    five_pings = [_ping_reply(i) for i in (1, 2, 3, 4, 5)]
    line = mcp_server._reply_line(five_pings)
    assert len(line.encode("utf-8")) <= 200
    err = json.loads(line)
    assert err["id"] is None and err["error"]["code"] == -32603
