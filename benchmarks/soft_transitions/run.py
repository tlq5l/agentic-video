"""Score detect_cuts (default arguments) on the soft-transition benchmark set.

A detection matches a boundary when it falls inside the boundary interval
widened by TOL seconds. Each boundary can be matched once; every other
detection is a false positive unless it falls in an ignore zone.
Prints METRIC lines. Uses a fresh cache folder so nothing is reused.

Sets are read from $AGENTIC_VIDEO_BENCH_DIR (default ~/.cache/agentic-video-bench).
Build them with build.py, build_v2.py, build_film.py and build_apollo.py.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from build import BENCH_ROOT

TOL = 0.3

cache_dir = tempfile.mkdtemp(prefix="av-bench-cache-")
os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache_dir
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrieval.engine import AgenticVideoEngine  # noqa: E402

_found = {}


def found_cuts(path):
    """detect_cuts once per clip per process; the score variants share it."""
    if path not in _found:
        _found[path] = AgenticVideoEngine(path).detect_cuts()
    return _found[path]


def load_manifest(set_name):
    """Read a set's manifest. Each clip is looked up in the set folder first, then at its recorded path."""
    folder = BENCH_ROOT / set_name
    manifest = json.loads((folder / "manifest.json").read_text())
    _locate_clips(manifest, folder)
    return manifest


def _locate_clips(node, folder):
    if isinstance(node, dict):
        if "path" in node:
            local = folder / Path(node["path"]).name
            if local.exists():
                node["path"] = str(local)
            elif not Path(node["path"]).exists():
                raise FileNotFoundError(f"clip {node['path']} is neither in {folder} nor at its recorded path")
        for value in node.values():
            _locate_clips(value, folder)
    elif isinstance(node, list):
        for value in node:
            _locate_clips(value, folder)


def times(result):
    items = result.get("cuts", []) if isinstance(result, dict) else result
    out = []
    for item in items:
        value = item["timestamp_sec"] if isinstance(item, dict) else getattr(item, "timestamp_sec")
        out.append(float(value))
    return sorted(out)


def score(clips):
    tp = fp = fn = 0
    hard = [0, 0]
    soft = [0, 0]
    fp_neg = 0
    for name, clip in clips.items():
        found = times(found_cuts(clip["path"]))
        matched = [False] * len(clip["boundaries"])
        for t in found:
            hit = None
            for i, (s, e) in enumerate(clip["boundaries"]):
                if not matched[i] and s - TOL <= t <= e + TOL:
                    hit = i
                    break
            if hit is not None:
                matched[hit] = True
                continue
            if any(s <= t <= e for s, e in clip["ignore"]):
                continue
            fp += 1
            if not clip["boundaries"]:
                fp_neg += 1
        for ok, kind in zip(matched, clip["kinds"]):
            bucket = hard if kind == "hard" else soft
            bucket[1] += 1
            bucket[0] += 1 if ok else 0
            tp += 1 if ok else 0
            fn += 0 if ok else 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp, "fp_neg": fp_neg,
            "recall_hard": hard[0] / hard[1] if hard[1] else 1.0,
            "recall_soft": soft[0] / soft[1] if soft[1] else 1.0}


def main():
    manifest = load_manifest("soft-transitions-v1")
    started = time.time()
    dev = score(manifest["dev"])
    dev_seconds = time.time() - started
    hold = score(manifest["holdout"])
    right = 0
    for case in manifest["steady"]:
        result = AgenticVideoEngine(case["path"]).track_region_changes(case["start"], case["end"], case["region"], sample_fps=10)
        right += 1 if result.get("steady_change", {}).get("detected") is case["expect"] else 0
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"METRIC dev_f1={dev['f1'] * 100:.2f}")
    print(f"METRIC dev_precision={dev['precision'] * 100:.2f}")
    print(f"METRIC dev_recall_hard={dev['recall_hard'] * 100:.2f}")
    print(f"METRIC dev_recall_soft={dev['recall_soft'] * 100:.2f}")
    print(f"METRIC dev_false_positives={dev['fp']}")
    print(f"METRIC dev_fp_in_negative_clips={dev['fp_neg']}")
    print(f"METRIC holdout_f1={hold['f1'] * 100:.2f}")
    print(f"METRIC holdout_recall_soft={hold['recall_soft'] * 100:.2f}")
    print(f"METRIC holdout_false_positives={hold['fp']}")
    print(f"METRIC steady_accuracy={right / len(manifest['steady']) * 100:.0f}")
    print(f"METRIC dev_runtime_s={dev_seconds:.2f}")


if __name__ == "__main__":
    main()
