"""End to end: inspect_window(read_text=True) on the full-resolution screen recording (3244 px wide).

Calls the real engine path (AgenticVideoEngine.inspect_window -> ocr.frame_text -> Apple Vision) on a
1-second window of the repro file. Skips when the file is not on this machine or Apple Vision is not installed.
"""
from pathlib import Path

import pytest

from src.core.cache import MediaCache
from src.retrieval import ocr
from src.retrieval.engine import AgenticVideoEngine

DESKTOP = Path("/Users/green/Desktop")
REPRO_GLOB = "Screen Recording 2026-11-16 at 11.06.09*.mov"


def _repro_file():
    matches = sorted(DESKTOP.glob(REPRO_GLOB)) if DESKTOP.is_dir() else []
    return matches[0] if matches else None


def _has_vision():
    try:
        ocr.ensure_available()
        return True
    except Exception:
        return False


REPRO = _repro_file()
needs_repro = pytest.mark.skipif(REPRO is None, reason="the 2026-11-16 11.06.09 PM screen recording is not on this machine")
needs_vision = pytest.mark.skipif(not _has_vision(), reason="Apple Vision OCR (pyobjc-framework-Vision) is required")


@needs_repro
@needs_vision
def test_inspect_window_read_text_returns_text_on_fullres_screen_recording(tmp_path):
    engine = AgenticVideoEngine(str(REPRO), cache=MediaCache(tmp_path / "cache"))
    result = engine.inspect_window(start_sec=1.0, end_sec=2.0, read_text=True, create_sheet=False)
    assert result.frames, "inspect_window returned no frames"
    for frame in result.frames:
        assert "text_lines" in frame.text, frame.file_path
