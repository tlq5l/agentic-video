"""End-to-end check of the motion_ tools through handle_call: the planted-fault selftest, the boil
ordering on the bundled GIF fixtures, and millisecond warm calls once a clip is cached in memory."""

import time
from pathlib import Path

import src.executor.mcp_server as mcp_server
from src.core.config import Config
from src.media import motion_probe

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "motion"
WARM_CALL_BUDGET_MS = 50


def _timed(name, args):
    start = time.perf_counter()
    result = mcp_server.handle_call(name, args)
    return result, (time.perf_counter() - start) * 1000


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

    zoom, zoom_ms = _timed("motion_zoom", {"source": v1, "box": [0, 0, 64, 64], "frames": "0-7"})
    assert zoom["ok"] and zoom["data"]["origin"] == "memory", zoom
    assert zoom_ms < WARM_CALL_BUDGET_MS, f"warm motion_zoom took {zoom_ms:.1f} ms"
