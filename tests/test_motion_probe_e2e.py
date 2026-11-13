"""End-to-end check of the motion_ tools through handle_call: the planted-fault selftest, the boil
ordering on the bundled GIF fixtures, and millisecond warm calls once a clip is cached in memory."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import venv
from pathlib import Path
from typing import Any, Dict

from PIL import Image

import src.executor.mcp_server as mcp_server
from src.core.config import Config
from src.media import motion_probe

REPO = Path(__file__).resolve().parents[1]
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
        results: Dict[int, Any] = {}  # writer index -> reply, or the exception it raised

        def writer(i: int) -> None:
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
        assert sorted(results) == [0, 1], results  # both writers reported back
        for result in results.values():
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


def test_selftest_defaults_run_from_installed_wheel(tmp_path):
    """Build the wheel, install it into a fresh venv outside the repo, and run motion_selftest with default
    arguments from there. A repo-only file on the default path fails here even though the repo tests pass."""
    uv = shutil.which("uv")  # uv-managed Pythons may have no pip, so prefer uv when it is on PATH
    wheel_dir = tmp_path / "dist"
    if uv:
        build = [uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(REPO)]
    elif importlib.util.find_spec("pip"):
        build = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheel_dir), str(REPO)]
    else:
        build = [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir), str(REPO)]
    subprocess.run(build, check=True, timeout=600)
    (wheel,) = wheel_dir.glob("agentic_video-*.whl")

    env_dir = tmp_path / "venv"
    bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    py = str(bin_dir / ("python.exe" if os.name == "nt" else "python"))
    if uv:
        subprocess.run([uv, "venv", "--python", sys.executable, str(env_dir)], check=True, timeout=300)
        subprocess.run([uv, "pip", "install", "--quiet", "--python", py, str(wheel)], check=True, timeout=900)
    else:
        venv.create(env_dir, with_pip=True)
        subprocess.run([py, "-m", "pip", "install", "--quiet", str(wheel)], check=True, timeout=900)

    probe = ("import json, src.executor.mcp_server as m\n"
             "reply = m.handle_call('motion_selftest', {})\n"
             "print(json.dumps({'module': m.__file__, 'reply': reply},\n"
             "                 default=lambda o: o.item() if hasattr(o, 'item') else str(o)))")
    run = subprocess.run([py, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert str(env_dir) in out["module"], f"ran from {out['module']}, not the installed wheel"
    reply = out["reply"]
    assert reply["ok"], reply
    assert reply["data"]["pass"] is True, reply["data"]["detectors"]
    assert reply["data"]["clean_flag_count"] == 0

def test_store_cap_is_a_hard_limit(tmp_path, monkeypatch):
    """With the cap below one clip's size, the warm store still holds no more bytes than the cap.
    Old code kept the single oversized clip, so the stored total went past the cap."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(motion_probe, "STORE_MAX_BYTES", 10)
    motion_probe.clear_memory_store()
    reply = mcp_server.handle_call("motion_stats", {"source": str(FIXTURES / "v1-256.gif")})
    assert reply["ok"], reply
    with motion_probe._STORE_LOCK:
        stored = sum(c.nbytes for c in motion_probe._STORE.values())
    motion_probe.clear_memory_store()
    assert stored <= 10, f"{stored} bytes stored against a 10-byte cap"


def test_box_with_origin_outside_frame_is_refused(tmp_path, monkeypatch):
    """A box whose origin is outside the frame is an error naming box in zoom and stats.
    Old code clamped it onto the frame edge and returned an image or stats for a different region."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    v1 = str(FIXTURES / "v1-256.gif")
    failures = []
    for box in ([999, 999, 10, 10], [256, 0, 10, 10]):
        for tool, extra in (("motion_zoom", {"frames": "0-3"}), ("motion_stats", {})):
            reply = mcp_server.handle_call(tool, {"source": v1, "box": box, **extra})
            if reply["ok"] or "box" not in reply["error"]["message"]:
                failures.append(f"{tool} {box}: {reply}")
    assert not failures, "\n".join(failures)


def test_compare_image_changes_when_comparison_file_is_replaced(tmp_path, monkeypatch):
    """Replacing the comparison file with different contents must give a new saved PNG.
    Old code keyed the PNG on the path, so the replaced file got the old image back."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    comparison = tmp_path / "comparison.gif"
    shutil.copyfile(FIXTURES / "v1-256.gif", comparison)
    args = {"source": str(FIXTURES / "v0-256.gif"), "compare_to": str(comparison), "frame": 0}
    first = mcp_server.handle_call("motion_compare", args)
    assert first["ok"], first
    shutil.copyfile(FIXTURES / "bad-256.gif", comparison)
    second = mcp_server.handle_call("motion_compare", args)
    assert second["ok"], second
    assert second["data"]["file_path"] != first["data"]["file_path"], "replaced file reused the saved image"


def test_motion_stats_reports_truncated_for_clip_cut_at_240_frames(tmp_path, monkeypatch):
    """A 241-frame GIF is measured only up to 240 frames, so motion_stats says truncated with the full count;
    a 64-frame fixture is not truncated."""
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    long_gif = tmp_path / "long-241.gif"
    frames = [Image.new("RGB", (32, 32), (i, 40, 90)) for i in range(241)]
    frames[0].save(long_gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
    long_reply = mcp_server.handle_call("motion_stats", {"source": str(long_gif)})
    assert long_reply["ok"], long_reply
    assert long_reply["data"]["truncated"] is True, long_reply["data"]
    assert long_reply["data"]["source_frames"] == 241, long_reply["data"]
    short_reply = mcp_server.handle_call("motion_stats", {"source": str(FIXTURES / "v0-256.gif")})
    assert short_reply["ok"], short_reply
    assert short_reply["data"]["truncated"] is False, short_reply["data"]
    motion_probe.clear_memory_store()
