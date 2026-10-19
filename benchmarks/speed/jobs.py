"""Speed benchmark for the slow background tools: build_visual_index (+ search_visual),
count_occurrences and count_repetitions.

Same idea as run.py: each video runs in its own child process with an empty cache, through
``handle_call`` (the path Executor uses). A background tool is called, then polled every
POLL_SEC until it reports a final status; the job's wall and CPU (user+sys incl. ffmpeg
children) are what is measured. Outputs are compared with a saved baseline:

    python benchmarks/speed/jobs.py --save-baseline
    python benchmarks/speed/jobs.py [--only clip_415s] [--jobs build,occ_text]

Set AV_REPO to time another checkout of the code (e.g. a pristine worktree) with this script.
Prints per-video ``METRIC <video>.<job>_wall=`` / ``_cpu=`` lines, ``METRIC <job>_wall`` for the
largest video (clip_415s), ``PEAK_MB``, and ``QUALITY_FAIL`` lines (exit 1) on any difference.

Quality rules: count/JSON results must match exactly; search_visual hit timestamps must match
exactly and scores within 1e-3; visual-index frame timestamps must match exactly and every
embedding must have cosine >= EMBED_MIN_COS with its baseline vector (the model, sampling and
thresholds are never changed; the tolerance only allows float-level reordering differences).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402  (shared video list, cpu helper, JSON normaliser)

REPO = Path(os.getenv("AV_REPO", str(base.REPO)))
BASELINE_DIR = Path(os.getenv("AV_JOBS_BASELINE", Path.home() / ".cache" / "agentic-video-bench" / "baseline_jobs"))
POLL_SEC = 0.2
JOB_TIMEOUT_SEC = 3000
EMBED_MIN_COS = 0.9999
SCORE_TOL = 1e-3
BIG = "clip_415s"
WAITING = {"queued", "running", "in_progress", "busy"}


def job_mix(duration: float) -> list[tuple[str, str, dict]]:
    """(job name, tool, args). Windows are fractions of the video so every video gets the same shape of work."""
    mid = round(duration * 0.3, 2)
    return [
        ("build", "build_visual_index", {}),
        ("search_a", "search_visual", {"query": "a man speaking into a microphone", "top_k": 5}),
        ("search_b", "search_visual", {"query": "a dark terminal window with code", "top_k": 5}),
        ("occ_text", "count_occurrences", {"text": "claude"}),
        ("occ_clip", "count_occurrences", {"description": "a man speaking into a microphone"}),
        ("occ_shape", "count_occurrences", {"description": "purple diamond"}),
        ("rep_30s", "count_repetitions", {"start_sec": mid, "end_sec": round(min(duration, mid + 30), 2)}),
        ("rep_120s", "count_repetitions", {"start_sec": 0.0, "end_sec": round(min(duration, 120), 2)}),
    ]


def run_job(handle_call, tool: str, args: dict) -> dict:
    """Call and poll like an agent would, until the job leaves its waiting states; returns the final result."""
    deadline = time.monotonic() + JOB_TIMEOUT_SEC
    while True:
        result = handle_call(tool, args)
        if not result.get("ok") or result["data"].get("status") not in WAITING:
            return result
        if time.monotonic() > deadline:
            raise SystemExit(f"{tool} did not finish in {JOB_TIMEOUT_SEC}s")
        time.sleep(POLL_SEC)


def dump_index(cache: str, out: Path) -> None:
    import numpy as np
    files = list(Path(cache).rglob("visual-clip-*.sqlite3"))
    if len(files) != 1:
        raise SystemExit(f"expected one visual index, found {len(files)}")
    conn = sqlite3.connect(f"file:{files[0]}?mode=ro", uri=True)
    rows = conn.execute("SELECT timestamp, width, height, embedding FROM frames ORDER BY timestamp").fetchall()
    conn.close()
    np.savez(out / "index.npz", t=np.array([r[0] for r in rows]), wh=np.array([[r[1], r[2]] for r in rows]),
             emb=np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows]))


def child(video: str, out_dir: str, only: list[str] | None) -> None:
    cache = tempfile.mkdtemp(prefix="avjobs-")
    os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache
    os.environ["AGENTIC_VIDEO_TEMP_DIR"] = os.path.join(cache, "tmp")
    sys.path.insert(0, str(REPO))
    from src.executor.mcp_server import handle_call

    path = str(base.VIDEOS[video])
    duration = handle_call("probe_video", {"source": path})["data"]["duration_sec"]
    out = Path(out_dir)
    walls, cpus, outputs = {}, {}, {}
    for name, tool, args in job_mix(duration):
        if only and ("build" if name.startswith("search") else name) not in only:  # searches need the index
            continue
        start, start_cpu = time.perf_counter(), base._cpu()
        result = run_job(handle_call, tool, {"source": path, **args})
        walls[name], cpus[name] = time.perf_counter() - start, base._cpu() - start_cpu
        if not result.get("ok"):
            raise SystemExit(f"{video} {name} failed: {json.dumps(result)[:500]}")
        data = result["data"]
        if data.get("status") == "error":
            raise SystemExit(f"{video} {name} error: {json.dumps(data)[:500]}")
        if name == "build":
            dump_index(cache, out)
            data = {"status": data["status"], "frames_indexed": data["frames_indexed"]}
        elif name.startswith("search"):
            data = {"hits": [{k: h[k] for k in ("rank", "timestamp_sec", "score")} for h in data["hits"]],
                    "intervals": data["intervals"]}
        outputs[name] = base._stable(data)
    peak_py = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # bytes on macOS
    peak_kid = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 / 1024
    (out / "result.json").write_text(json.dumps(
        {"wall": walls, "cpu": cpus, "outputs": outputs, "peak_mb": {"python": peak_py, "largest_child": peak_kid}},
        indent=1, sort_keys=True))
    shutil.rmtree(cache, ignore_errors=True)


def _close(a, b, tol) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def compare(video: str, new: Path) -> list[str]:
    import numpy as np
    ref = BASELINE_DIR / video
    old = json.loads((ref / "result.json").read_text())["outputs"]
    cur = json.loads((new / "result.json").read_text())["outputs"]
    problems = []
    for name in cur:
        if name not in old:
            continue
        tol = SCORE_TOL if name.startswith("search") else 0
        if not _close(old[name], cur[name], tol):
            problems.append(f"{video} {name}: output differs")
    if (new / "index.npz").exists() and (ref / "index.npz").exists():
        a, b = np.load(ref / "index.npz"), np.load(new / "index.npz")
        if a["t"].shape != b["t"].shape or not (a["t"] == b["t"]).all() or not (a["wh"] == b["wh"]).all():
            problems.append(f"{video} index: frame timestamps/sizes differ ({len(a['t'])} vs {len(b['t'])})")
        else:
            cos = (a["emb"] * b["emb"]).sum(axis=1) / (np.linalg.norm(a["emb"], axis=1) * np.linalg.norm(b["emb"], axis=1))
            print(f"{video} index min cosine {cos.min():.8f}")
            if cos.min() < EMBED_MIN_COS:
                problems.append(f"{video} index: min cosine {cos.min():.6f} < {EMBED_MIN_COS}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--child", nargs=2, metavar=("VIDEO", "OUT_DIR"))
    parser.add_argument("--only", nargs="*", help="video names to run")
    parser.add_argument("--jobs", help="comma list of job names (default all); 'build' includes the searches")
    opts = parser.parse_args()
    jobs = opts.jobs.split(",") if opts.jobs else None
    if opts.child:
        child(*opts.child, jobs)
        return 0
    work = Path(tempfile.mkdtemp(prefix="avjobs-out-"))
    problems, peak = [], 0.0
    for video in opts.only or list(base.VIDEOS):
        out = work / video
        out.mkdir()
        cmd = [sys.executable, __file__, "--child", video, str(out)] + (["--jobs", opts.jobs] if opts.jobs else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env={**os.environ, "AV_REPO": str(REPO)})
        result = json.loads((out / "result.json").read_text())
        peak = max(peak, result["peak_mb"]["python"])
        for name in result["wall"]:
            print(f"METRIC {video}.{name}_wall={result['wall'][name]:.2f}")
            print(f"METRIC {video}.{name}_cpu={result['cpu'][name]:.2f}")
            if video == BIG:
                print(f"METRIC {name}_wall={result['wall'][name]:.2f}")
                print(f"METRIC {name}_cpu={result['cpu'][name]:.2f}")
        print(f"METRIC {video}.peak_mb={result['peak_mb']['python']:.0f}")
        if opts.save_baseline:
            target = BASELINE_DIR / video
            if opts.jobs and target.exists():  # keep other jobs' baselines when re-saving a subset
                merged = json.loads((target / "result.json").read_text())
                for k in ("wall", "cpu", "outputs"):
                    merged[k].update(result[k])
                result = merged
                (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True))
                for extra in target.glob("*.npz"):
                    if not (out / extra.name).exists():
                        shutil.copy(extra, out)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(out, target)
        else:
            problems += compare(video, out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"METRIC peak_mb={peak:.0f}")
    for problem in problems:
        print("QUALITY_FAIL", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
