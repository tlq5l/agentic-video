"""End-to-end check of the motion_ tools through handle_call: the planted-fault selftest, the boil
ordering on the bundled GIF fixtures, and millisecond warm calls once a clip is cached in memory."""

import threading
import time
from pathlib import Path

from PIL import Image

import src.executor.mcp_server as mcp_server
from src.core.config import Config
from src.media import motion_probe

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "motion"
WARM_CALL_BUDGET_MS = 50


def _timed(name, args):
    start = time.perf_counter()
    result = mcp_server.handle_call(name, args)
    return result, (time.perf_counter() - start) * 1000


def test_motion_tools_hold_up_through_handle_call(tmp_path, monkeypatch):
    """Four fixes through handle_call: a broken clip reports failed; a zoom never exceeds MAX_PX and is
    refused with an error naming box; a box too small for stats returns an error naming box; two writers
    of one PNG both finish. Each check is collected, so one run reports every defect that still exists."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    failures = []

    def check(label, fn):
        try:
            fn()
        except Exception as exc:  # record and go on to the next defect
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    v1 = str(FIXTURES / "v1-256.gif")

    def broken_clip_reports_failed():
        bad = tmp_path / "broken.gif"
        bad.write_bytes(b"not an image or a video")
        status = {}
        for _ in range(100):
            reply = mcp_server.handle_call("motion_open", {"source": str(bad)})
            assert reply["ok"], reply
            status = reply["data"]
            if status["status"] != "in_progress":
                break
            time.sleep(0.05)
        assert status["status"] == "failed" and status.get("error"), status

    def small_box_gives_named_error():
        for box in ([0, 0, 0, 0], [250, 250, 10, 10]):  # zero-area; edge box clipped to 6x6
            reply = mcp_server.handle_call("motion_stats", {"source": v1, "box": box})
            assert not reply["ok"] and "box" in reply["error"]["message"], (box, reply)

    def zoom_stays_under_cap():
        # 31 px wide over 64 frames: the cap must count the gaps between frames (old output: 8314 px wide).
        reply = mcp_server.handle_call("motion_zoom", {"source": v1, "box": [0, 0, 31, 20], "frames": "0-63", "zoom": 16})
        assert reply["ok"], reply
        with Image.open(reply["data"]["file_path"]) as png:
            assert max(png.size) <= motion_probe.MAX_PX, png.size
        # A box too big for even zoom 1 is refused, naming the parameter (old output: 16762 px wide).
        reply = mcp_server.handle_call("motion_zoom", {"source": v1, "box": [0, 0, 256, 64], "frames": "0-63"})
        assert not reply["ok"] and "box" in reply["error"]["message"], reply

    def two_writers_of_one_png():
        args = {"source": v1, "box": [64, 64, 32, 32], "frames": "0-3", "zoom": 2}
        real_save = Image.Image.save
        gate = threading.Barrier(2, timeout=10)

        def gated_save(self, fp, *rest, **kw):
            if str(fp).endswith(".tmp"):
                gate.wait()  # hold both writers until each has started its write
            return real_save(self, fp, *rest, **kw)

        monkeypatch.setattr(Image.Image, "save", gated_save)
        results = [None, None]

        def writer(i):
            try:
                results[i] = mcp_server.handle_call("motion_zoom", args)
            except Exception as exc:
                results[i] = exc

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        monkeypatch.setattr(Image.Image, "save", real_save)
        for result in results:
            assert isinstance(result, dict) and result["ok"], result
        with Image.open(results[0]["data"]["file_path"]) as png:
            png.load()

    check("motion_open on a clip that cannot be decoded", broken_clip_reports_failed)
    check("motion_stats on a box under 8 px", small_box_gives_named_error)
    check("motion_zoom over the size cap", zoom_stays_under_cap)
    check("two writers of one cached PNG", two_writers_of_one_png)
    assert not failures, "\n".join(failures)


def test_motion_tools_end_to_end_through_handle_call(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()

    selftest = mcp_server.handle_call("motion_selftest", {})
    assert selftest["ok"], selftest
    assert selftest["data"]["pass"], selftest["data"]["detectors"]
    assert selftest["data"]["clean_flag_count"] == 0

    boil = {}
    for name in ("bad-256", "v0-256", "v1-256"):
        result = mcp_server.handle_call("motion_stats", {"source": str(FIXTURES / f"{name}.gif")})
        assert result["ok"], result
        boil[name] = result["data"]["boil"]
    assert boil["bad-256"] > boil["v0-256"] > boil["v1-256"], boil

    v1 = str(FIXTURES / "v1-256.gif")
    # The fixture loop above already cached v1 in memory; this call checks the warm path only.
    stats, stats_ms = _timed("motion_stats", {"source": v1})
    assert stats["ok"] and stats["data"]["origin"] == "memory", stats
    assert stats_ms < WARM_CALL_BUDGET_MS, f"warm motion_stats took {stats_ms:.1f} ms"

    # The first zoom draws and saves the PNG; the repeat call is the warm one.
    zoom_args = {"source": v1, "box": [0, 0, 64, 64], "frames": "0-7"}
    first, _ = _timed("motion_zoom", zoom_args)
    assert first["ok"], first
    zoom, zoom_ms = _timed("motion_zoom", zoom_args)
    assert zoom["data"]["file_path"] == first["data"]["file_path"], zoom
    assert zoom["ok"] and zoom["data"]["origin"] == "memory", zoom
    assert zoom_ms < WARM_CALL_BUDGET_MS, f"warm motion_zoom took {zoom_ms:.1f} ms"
