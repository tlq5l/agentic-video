"""read_text on frame_at and inspect_window: Apple Vision reads the word on a real frame, or the call says why it cannot."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import src.executor.mcp_server as mcp_server
from src.core.config import Config
from src.retrieval import ocr

WORD = "RELEASE"
FONT = "/System/Library/Fonts/Helvetica.ttc"

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _has_vision():
    try:
        ocr.ensure_available()
        return True
    except RuntimeError:
        return False


def _word_clip(path):
    """A 2 s lossless clip: light 48 px text on a dark background, centred."""
    font = ImageFont.truetype(FONT, 48) if Path(FONT).exists() else ImageFont.load_default(size=48)
    frame = Image.new("RGB", (640, 360), (20, 20, 28))
    ImageDraw.Draw(frame).text((320, 180), WORD, font=font, fill=(240, 240, 240), anchor="mm")
    still = path.with_suffix(".png")
    frame.save(still)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(still), "-t", "2", "-r", "25",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


@needs_ffmpeg
def test_read_text_reads_the_word_on_screen_or_says_why_it_cannot(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _word_clip(tmp_path / "word.mp4")
    frame_args = {"source": clip, "time_sec": 1.0}
    window_args = {"source": clip, "start_sec": 0.5, "end_sec": 1.5, "fps": 2}

    plain = mcp_server.handle_call("frame_at", frame_args)
    assert plain["ok"] is True and "text_lines" not in plain["data"]
    plain_window = mcp_server.handle_call("inspect_window", window_args)
    assert plain_window["ok"] is True
    assert all("text_lines" not in frame for frame in plain_window["data"]["frames"])

    if _has_vision():
        framed = mcp_server.handle_call("frame_at", {**frame_args, "read_text": True})
        assert framed["ok"] is True
        assert any(WORD in line["text"] for line in framed["data"]["text_lines"])
        windows = mcp_server.handle_call("inspect_window", {**window_args, "read_text": True})
        frames = windows["data"]["frames"]
        assert frames and all("text_lines" in frame for frame in frames)
        assert any(WORD in line["text"] for frame in frames for line in frame["text_lines"])
    else:
        for name, args in (("frame_at", {**frame_args, "read_text": True}),
                           ("inspect_window", {**window_args, "read_text": True})):
            with pytest.raises(ValueError, match="read_text needs Apple Vision"):
                mcp_server.handle_call(name, args)
            result = json.loads(mcp_server.tool_call_response(1, name, args))["result"]
            assert result["isError"] is True
            assert "read_text needs Apple Vision" in result["content"][0]["text"]
