"""Speed and output check for the sound tools: audio_onsets (5 bands), loudness_timeline, sync_check.

Calls the tools through ``handle_call`` (the path Executor uses) with a warm PCM cache
and no cached results, one child process per input. Result JSON is compared byte for
byte with a saved baseline.

    python benchmarks/speed/audio.py --save-baseline
    python benchmarks/speed/audio.py [--only clip_415s] [--no-1h]

Prints ``METRIC`` lines (CPU s = user+sys incl. children) per tool and input, and
``SCORE <cpu s>`` = total of all sound tools on clip_415s. Exits 1 on any output difference.
The 1 h input is the clip_415s audio looped to 3600 s (built once, kept in the bench cache);
there sync_check runs without a cut detector (it is audio-only).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path.home() / ".cache" / "agentic-video-bench"
BASELINE_DIR = Path(os.getenv("AV_AUDIO_BASELINE", VIDEO_DIR / "baseline_audio"))
LONG_1H = Path(os.getenv("AV_AUDIO_1H", VIDEO_DIR / "synthetic_1h.m4a"))
INPUTS = {
    "fixture_60s": REPO / "fixtures" / "synthetic_bench_60s.mp4",
    "x_post_127s": VIDEO_DIR / "x_post_127s.mp4",
    "clip_415s": VIDEO_DIR / "clip_415s.mp4",
    "synthetic_1h": LONG_1H,
}
BANDS = ["broad", "low_mid", "mid", "high", "air"]
NOCACHE_PREFIXES = ("audio_onsets_", "loudness_timeline_")


def _cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def ensure_1h() -> None:
    if LONG_1H.exists():
        return
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-stream_loop", "9", "-i", str(INPUTS["clip_415s"]),
                    "-vn", "-t", "3600", "-c:a", "aac", "-b:a", "64k", str(LONG_1H)], check=True)


def child(name: str, out_dir: str) -> None:
    cache = tempfile.mkdtemp(prefix="avabench-")
    os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache
    os.environ["AGENTIC_VIDEO_TEMP_DIR"] = os.path.join(cache, "tmp")
    sys.path.insert(0, str(REPO))
    from src.core.cache import MediaCache
    from src.executor.mcp_server import handle_call
    from src.media import audio

    # Result JSON for the sound tools is never served from cache; PCM and cut caches stay warm.
    original_load = MediaCache.load_json
    MediaCache.load_json = lambda self, fp, key: None if str(key).startswith(NOCACHE_PREFIXES) else original_load(self, fp, key)

    path = str(INPUTS[name])
    duration = len(audio._decode_pcm(path)[0]) / float(audio.SAMPLE_RATE)
    cues = [{"time_sec": round(duration * f, 2), "label": "hit"} for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]
    calls = []
    for band in BANDS:
        args = {} if band == "broad" else {"band_hz": band}
        calls.append((f"onsets_{band}", "audio_onsets", args))
    calls.append(("loudness_timeline", "loudness_timeline", {}))
    calls.append(("sync_check", "sync_check", {"cues": cues}))

    cpu, wall, outputs = {}, {}, {}
    for label, tool, args in calls:
        if tool == "sync_check" and name == "synthetic_1h":
            run = lambda: audio.sync_check(path, cues=cues, cut_detector=lambda: [])
        else:
            if tool == "sync_check":
                handle_call(tool, {"source": path, **args})  # warm the cut detector cache
            run = lambda: handle_call(tool, {"source": path, **args})
        t0, c0 = time.perf_counter(), _cpu()
        result = run()
        cpu[label], wall[label] = _cpu() - c0, time.perf_counter() - t0
        if isinstance(result, dict) and "ok" in result and not result["ok"]:
            raise SystemExit(f"{name} {label} failed: {json.dumps(result)[:400]}")
        outputs[label] = json.dumps(result, sort_keys=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
    Path(out_dir, "result.json").write_text(json.dumps({"cpu": cpu, "wall": wall, "peak_mb": peak}))
    for label, text in outputs.items():
        Path(out_dir, f"{label}.json").write_text(text)
    shutil.rmtree(cache, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--child", nargs=2, metavar=("NAME", "OUT_DIR"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-1h", action="store_true")
    opts = parser.parse_args()
    if opts.child:
        child(*opts.child)
        return 0
    names = opts.only or [n for n in INPUTS if not (opts.no_1h and n == "synthetic_1h")]
    if "synthetic_1h" in names:
        ensure_1h()
    problems, primary = [], None
    for name in names:
        out = Path(tempfile.mkdtemp(prefix="avaout-"))
        subprocess.run([sys.executable, __file__, "--child", name, str(out)], check=True)
        result = json.loads((out / "result.json").read_text())
        total = sum(result["cpu"].values())
        for label, value in result["cpu"].items():
            print(f"METRIC {name}.{label}={value:.3f}")
        print(f"METRIC {name}.total={total:.3f}")
        print(f"METRIC {name}.wall={sum(result['wall'].values()):.3f}")
        print(f"METRIC {name}.peak_mb={result['peak_mb']:.2f}")
        if name == "clip_415s":
            primary = total
        target = BASELINE_DIR / name
        if opts.save_baseline:
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(out, target)
        else:
            for f in sorted(out.glob("*.json")):
                if f.name == "result.json":
                    continue
                if not (target / f.name).exists() or (target / f.name).read_bytes() != f.read_bytes():
                    problems.append(f"{name} {f.stem}: output differs")
        shutil.rmtree(out, ignore_errors=True)
    if primary is not None:
        print(f"SCORE {primary:.3f}")
    for p in problems:
        print("QUALITY_FAIL", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
