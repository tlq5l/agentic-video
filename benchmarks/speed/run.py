"""Speed and memory benchmark for the most-used agentic-video tools.

Runs a fixed call mix (the 5 tools agents call most) on each video, once
with an empty cache and once warm, through the same ``handle_call`` path
Executor uses. Each video runs in its own child process so peak memory is
per video. Outputs are compared with a saved baseline: JSON fields must
match exactly and every image must be pixel-identical or within PSNR_MIN_DB.

    python benchmarks/speed/run.py --save-baseline   # record the reference
    python benchmarks/speed/run.py                   # time and check

Prints ``SCORE <cpu seconds>`` (user + system time of this process and its
ffmpeg children; lower is better, and steadier than wall time when other work
loads the machine) and ``WALL <seconds>``; exits 1 if any output differs
from the baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(os.getenv("AV_BENCH_BASELINE", Path.home() / ".cache" / "agentic-video-bench" / "baseline"))
VIDEO_DIR = Path.home() / ".cache" / "agentic-video-bench"
VIDEOS = {
    "fixture_60s": REPO / "fixtures" / "synthetic_bench_60s.mp4",
    "x_post_127s": VIDEO_DIR / "x_post_127s.mp4",
    "clip_415s": VIDEO_DIR / "clip_415s.mp4",
}
PSNR_MIN_DB = 40.0
VOLATILE_KEYS = {"file_path", "source_path", "local_path", "data", "name", "contact_sheet_path", "sheet_path"}


def call_mix(duration: float) -> list[tuple[str, dict]]:
    mid = round(duration / 2, 2)
    return [
        ("probe_video", {}),
        ("contact_sheet", {"samples": 24}),
        ("detect_cuts", {}),
        ("inspect_window", {"start_sec": mid, "end_sec": mid + 4}),
        ("inspect_windows", {"windows": [
            {"start_sec": round(duration * f, 2), "end_sec": round(duration * f, 2) + 2} for f in (0.1, 0.5, 0.85)
        ]}),
    ]


def _images(node, out):
    """Collect image file paths from a result, in a stable order."""
    if isinstance(node, dict):
        for key in sorted(node):
            value = node[key]
            if isinstance(value, str) and value.endswith((".jpg", ".png")) and os.path.isfile(value):
                out.append(value)
            else:
                _images(value, out)
    elif isinstance(node, list):
        for item in node:
            _images(item, out)
    return out


def _stable(node):
    if isinstance(node, dict):
        return {k: _stable(v) for k, v in sorted(node.items()) if k not in VOLATILE_KEYS}
    if isinstance(node, list):
        return [_stable(v) for v in node]
    if isinstance(node, float):
        return round(node, 4)
    return node


def _cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def child(video: str, out_dir: str) -> None:
    """Run the call mix cold then warm; write timings, stable JSON and images to out_dir."""
    cache = tempfile.mkdtemp(prefix="avbench-")
    os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache
    os.environ["AGENTIC_VIDEO_TEMP_DIR"] = os.path.join(cache, "tmp")
    sys.path.insert(0, str(REPO))
    from src.executor.mcp_server import handle_call

    path = str(VIDEOS[video])
    duration = handle_call("probe_video", {"source": path})["data"]["duration_sec"]
    shutil.rmtree(cache)
    os.makedirs(cache)
    timings, cpu, outputs = {}, {}, {}
    out = Path(out_dir)
    for run in ("cold", "warm"):
        for name, args in call_mix(duration):
            start, start_cpu = time.perf_counter(), _cpu()
            result = handle_call(name, {"source": path, **args})
            timings[f"{run}:{name}"] = time.perf_counter() - start
            cpu[f"{run}:{name}"] = _cpu() - start_cpu
            if run == "cold":
                if not result.get("ok"):
                    raise SystemExit(f"{video} {name} failed: {json.dumps(result)[:500]}")
                outputs[name] = _stable(result["data"])
                for i, img in enumerate(_images(result["data"], [])):
                    shutil.copy(img, out / f"{name}_{i:03d}{Path(img).suffix}")
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS reports bytes
    kids_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    (out / "result.json").write_text(json.dumps(
        {"timings": timings, "cpu": cpu, "outputs": outputs, "peak_mb": {"python": self_kb / 1024, "largest_child": kids_kb / 1024}},
        indent=1, sort_keys=True))
    shutil.rmtree(cache, ignore_errors=True)


def psnr(a: Path, b: Path) -> float:
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        return 0.0
    mse = float(((x - y) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def compare(video: str, new: Path) -> list[str]:
    base = BASELINE_DIR / video
    problems = []
    old_out = json.loads((base / "result.json").read_text())["outputs"]
    new_out = json.loads((new / "result.json").read_text())["outputs"]
    for name in old_out:
        if old_out[name] != new_out.get(name):
            problems.append(f"{video} {name}: JSON output differs")
    old_imgs = sorted(p.name for p in base.iterdir() if p.suffix in (".jpg", ".png"))
    new_imgs = sorted(p.name for p in new.iterdir() if p.suffix in (".jpg", ".png"))
    if old_imgs != new_imgs:
        problems.append(f"{video}: image list differs ({len(old_imgs)} vs {len(new_imgs)})")
    for img in set(old_imgs) & set(new_imgs):
        score = psnr(base / img, new / img)
        if score < PSNR_MIN_DB:
            problems.append(f"{video} {img}: PSNR {score:.1f} dB < {PSNR_MIN_DB}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--child", nargs=2, metavar=("VIDEO", "OUT_DIR"))
    parser.add_argument("--only", nargs="*", help="video names to run")
    opts = parser.parse_args()
    if opts.child:
        child(*opts.child)
        return 0

    work = Path(tempfile.mkdtemp(prefix="avbench-out-"))
    total_cpu, total_wall, problems, peaks = 0.0, 0.0, [], {}
    for video in opts.only or list(VIDEOS):
        out = work / video
        out.mkdir()
        subprocess.run([sys.executable, __file__, "--child", video, str(out)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = json.loads((out / "result.json").read_text())
        seconds = sum(result["cpu"].values())
        total_cpu += seconds
        total_wall += sum(result["timings"].values())
        peaks[video] = result["peak_mb"]
        slowest = sorted(result["cpu"].items(), key=lambda kv: -kv[1])[:3]
        print(f"{video:12s} {seconds:7.2f} cpu s   python peak {result['peak_mb']['python']:6.0f} MB   "
              f"slowest: " + ", ".join(f"{k} {v:.2f} s" for k, v in slowest))
        if opts.save_baseline:
            target = BASELINE_DIR / video
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(out, target)
        else:
            problems += compare(video, out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"PEAK_MB {max(p['python'] for p in peaks.values()):.0f}")
    print(f"WALL {total_wall:.2f}")
    print(f"SCORE {total_cpu:.2f}")
    for problem in problems:
        print("QUALITY_FAIL", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
