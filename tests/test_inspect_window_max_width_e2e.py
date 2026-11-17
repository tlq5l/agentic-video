"""inspect_window max_width on the repro screen recording: 768 px frames by default, native width on request, cached apart."""
import subprocess
from pathlib import Path

import pytest
from PIL import Image

import src.executor.mcp_server as mcp_server
from src.core.config import Config


def _repro():
    # The file name holds a narrow no-break space before PM, so match it with a glob.
    return next(Path("/Users/green/Desktop").glob("Screen Recording 2026-11-16 at 11.06.09*.mov"), None)


def _native_width(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


REPRO = _repro()
needs_repro = pytest.mark.skipif(REPRO is None, reason="repro screen recording is not on this Mac")


@needs_repro
def test_inspect_window_max_width_default_768_full_on_request_cached_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    native = _native_width(REPRO)
    args = {"source": str(REPRO), "start_sec": 2.0, "end_sec": 3.0, "fps": 2}

    default = mcp_server.handle_call("inspect_window", args)
    assert default["ok"] is True
    assert {frame["width"] for frame in default["data"]["frames"]} == {768}

    full = mcp_server.handle_call("inspect_window", {**args, "max_width": native})
    assert full["ok"] is True
    assert {frame["width"] for frame in full["data"]["frames"]} == {native}
    sheet = full["data"]["contact_sheet_path"]
    assert sheet and Image.open(sheet).width < native

    # Same window after the full-width call: the cache must still hand back 768 px frames.
    again = mcp_server.handle_call("inspect_window", args)
    assert {frame["width"] for frame in again["data"]["frames"]} == {768}
