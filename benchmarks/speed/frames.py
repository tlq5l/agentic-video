"""Cold wall time and output check for the multi-item frame tools on clip_415s.

Runs overview_chunks, inspect_moments, inspect_windows, refine_temporal_beam,
track_region_changes and track_region_events through ``handle_call``, each with an
empty cache, in one child process. JSON must equal the saved baseline and every
image must be within PSNR_MIN_DB (same rules as run.py).

    python benchmarks/speed/frames.py --save-baseline
    python benchmarks/speed/frames.py [--only inspect_moments ...]
    AV_REPO=<other checkout> python benchmarks/speed/frames.py   # time other code

Prints ``METRIC <tool>.wall=<s>`` and ``METRIC <tool>.cpu=<s>``, ``SCORE <total wall s>``;
exits 1 on any output difference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402

REPO = Path(os.getenv("AV_REPO", str(base.REPO)))
VIDEO = base.VIDEO_DIR / "clip_415s.mp4"
BASELINE_DIR = Path(os.getenv("AV_FRAMES_BASELINE", base.VIDEO_DIR / "baseline_frames"))
REGION = {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.5}
CALLS = {
    "overview_chunks": {"chunk_sec": 60, "samples_per_chunk": 8},
    "inspect_moments": {"moments": [{"timestamp_sec": t, "label": f"m{t}"} for t in (20, 70, 120, 170, 220, 270, 320, 380)]},
    "inspect_windows": {"windows": [{"start_sec": s, "end_sec": s + 2} for s in (10, 60, 110, 160, 210, 260)]},
    "refine_temporal_beam": {"windows": [{"start_sec": s, "end_sec": s + 60} for s in (10, 150, 300)]},
    "track_region_changes": {"start_sec": 0, "end_sec": 400, "region": REGION},
    "track_region_events": {"start_sec": 0, "end_sec": 400, "region": REGION},
}


def child(out_dir: str, tools: list[str]) -> None:
    cache = tempfile.mkdtemp(prefix="avfbench-")
    os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache
    os.environ["AGENTIC_VIDEO_TEMP_DIR"] = os.path.join(cache, "tmp")
    sys.path.insert(0, str(REPO))
    from src.executor.mcp_server import handle_call

    out = Path(out_dir)
    wall, cpu, outputs = {}, {}, {}
    for tool in tools:
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache)
        t0, c0 = time.perf_counter(), base._cpu()
        result = handle_call(tool, {"source": str(VIDEO), **CALLS[tool]})
        wall[tool], cpu[tool] = time.perf_counter() - t0, base._cpu() - c0
        if not result.get("ok"):
            raise SystemExit(f"{tool} failed: {json.dumps(result)[:500]}")
        outputs[tool] = base._stable(result["data"])
        for i, img in enumerate(base._images(result["data"], [])):
            shutil.copy(img, out / f"{tool}_{i:03d}{Path(img).suffix}")
    (out / "result.json").write_text(json.dumps({"wall": wall, "cpu": cpu, "outputs": outputs}, indent=1, sort_keys=True))
    shutil.rmtree(cache, ignore_errors=True)


def compare(new: Path, tools: list[str]) -> list[str]:
    old_out = json.loads((BASELINE_DIR / "result.json").read_text())["outputs"]
    new_out = json.loads((new / "result.json").read_text())["outputs"]
    problems = [f"{t}: JSON output differs" for t in tools if old_out[t] != new_out[t]]
    for tool in tools:
        old = sorted(p.name for p in BASELINE_DIR.glob(f"{tool}_*") if p.suffix in (".jpg", ".png"))
        cur = sorted(p.name for p in new.glob(f"{tool}_*") if p.suffix in (".jpg", ".png"))
        if old != cur:
            problems.append(f"{tool}: image list differs ({len(old)} vs {len(cur)})")
        for name in set(old) & set(cur):
            score = base.psnr(BASELINE_DIR / name, new / name)
            if score < base.PSNR_MIN_DB:
                problems.append(f"{name}: PSNR {score:.1f} dB < {base.PSNR_MIN_DB}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--child", nargs="+", metavar=("OUT_DIR", "TOOL"))
    parser.add_argument("--only", nargs="*", choices=list(CALLS))
    opts = parser.parse_args()
    if opts.child:
        child(opts.child[0], opts.child[1:])
        return 0
    tools = opts.only or list(CALLS)
    out = Path(tempfile.mkdtemp(prefix="avfout-"))
    subprocess.run([sys.executable, __file__, "--child", str(out), *tools], check=True, env={**os.environ, "AV_REPO": str(REPO)})
    result = json.loads((out / "result.json").read_text())
    for tool in tools:
        print(f"METRIC {tool}.wall={result['wall'][tool]:.3f}")
        print(f"METRIC {tool}.cpu={result['cpu'][tool]:.3f}")
    print(f"SCORE {sum(result['wall'].values()):.3f}")
    problems = []
    if opts.save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        old = json.loads((BASELINE_DIR / "result.json").read_text())["outputs"] if (BASELINE_DIR / "result.json").exists() else {}
        for p in out.iterdir():
            if p.name != "result.json":
                shutil.copy(p, BASELINE_DIR / p.name)
        (BASELINE_DIR / "result.json").write_text(json.dumps({"outputs": {**old, **json.loads((out / "result.json").read_text())["outputs"]}}, indent=1, sort_keys=True))
    else:
        problems = compare(out, tools)
    shutil.rmtree(out, ignore_errors=True)
    for problem in problems:
        print("QUALITY_FAIL", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
