"""Score detect_cuts on the v2 benchmark set, the film clips and the Apollo windows.

Same matching rule as run.py. Run from the repository root:

    python benchmarks/soft_transitions/run_v2.py            # climb metrics (dev splits and the v1 dev guard)
    python benchmarks/soft_transitions/run_v2.py --holdout  # held-out metrics only

Both print METRIC lines. The default mode never reads the held-out clips.

Climb mode: dev_f1 (primary) is scored over the v2 dev clips and the film dev
clip (Big Buck Bunny) together. v2_dev_f1 and the other dev_* lines are the v2
dev clips alone. film_dev_* is the film dev clip, transitions only (hard cuts
are ignore zones there). v1_dev_f1 is the v1 guard. apollo_dev_* are the
fully inspected Apollo dev windows (build_apollo.py): cuts and transitions
both count, so recall is real, and apollo_dev_kind_accuracy is the share of
found changes labelled with the right kind. v1_holdout_f1 is the v1 guard on
its holdout clips; it is not part of the held-out split.

Holdout mode prints the same metrics for the held-out halves: holdout_*,
film_holdout_* and apollo_holdout_*.
Apollo lines are printed only when the Apollo set has been built.
"""
import argparse
import shutil
import time

import run as v1


def score_with_kinds(clips):
    """Like run.score, and also checks `kind` on the changes that were found."""
    tp = fp = fn = kind_right = kind_total = 0
    soft = [0, 0]
    for clip in clips.values():
        found = sorted((float(cut.timestamp_sec), cut.kind) for cut in v1.found_cuts(clip["path"]))
        matched, extra_fp = v1.match_clip(clip, found)
        fp += extra_fp
        for kind, label in zip(matched, clip["labels"]):
            tp += 1 if kind else 0
            fn += 0 if kind else 1
            if label == "transition":
                soft[1] += 1
                soft[0] += 1 if kind else 0
            if kind and label != "either":
                kind_total += 1
                kind_right += 1 if kind == label else 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp,
            "recall_soft": soft[0] / soft[1] if soft[1] else 1.0,
            "kind_accuracy": kind_right / kind_total if kind_total else 1.0}


def pct(value):
    return f"{value * 100:.2f}"


def apollo_lines(half, got):
    return [(f"apollo_{half}_f1", pct(got["f1"])),
            (f"apollo_{half}_precision", pct(got["precision"])),
            (f"apollo_{half}_recall", pct(got["recall"])),
            (f"apollo_{half}_recall_transitions", pct(got["recall_soft"])),
            (f"apollo_{half}_false_positives", str(got["fp"])),
            (f"apollo_{half}_kind_accuracy", pct(got["kind_accuracy"]))]


def load_apollo():
    if not (v1.BENCH_ROOT / "soft-transitions-apollo" / "manifest.json").exists():
        return None
    return v1.load_manifest("soft-transitions-apollo")


def climb_lines():
    m2 = v1.load_manifest("soft-transitions-v2")
    m1 = v1.load_manifest("soft-transitions-v1")
    film = v1.load_manifest("soft-transitions-film")
    started = time.time()
    dev = v1.score(m2["dev"])
    dev_seconds = time.time() - started
    real_dev = v1.score({k: c for k, c in m2["dev"].items() if c.get("real")})
    old_dev = v1.score(m1["dev"])
    old_hold = v1.score(m1["holdout"])
    started = time.time()
    film_dev = v1.score(film["dev"])
    film_seconds = time.time() - started
    both = v1.score({**m2["dev"], **film["dev"]})
    lines = [
        ("dev_f1", pct(both["f1"])),
        ("v2_dev_f1", pct(dev["f1"])),
        ("dev_precision", pct(dev["precision"])),
        ("dev_recall_hard", pct(dev["recall_hard"])),
        ("dev_recall_soft", pct(dev["recall_soft"])),
        ("dev_false_positives", str(dev["fp"])),
        ("dev_fp_in_negative_clips", str(dev["fp_neg"])),
        ("dev_real_f1", pct(real_dev["f1"])),
        ("film_dev_fp", str(film_dev["fp"])),
        ("film_dev_recall", pct(film_dev["recall"])),
        ("v1_dev_f1", pct(old_dev["f1"])),
        ("v1_holdout_f1", pct(old_hold["f1"])),
    ]
    apollo = load_apollo()
    started = time.time()
    if apollo:
        lines += apollo_lines("dev", score_with_kinds(apollo["dev"]))
    apollo_seconds = time.time() - started
    lines += [("dev_runtime_s", f"{dev_seconds:.2f}"), ("film_runtime_s", f"{film_seconds:.2f}")]
    if apollo:
        lines.append(("apollo_runtime_s", f"{apollo_seconds:.2f}"))
    return lines


def holdout_lines():
    m2 = v1.load_manifest("soft-transitions-v2")
    film = v1.load_manifest("soft-transitions-film")
    hold = v1.score(m2["holdout"])
    real_hold = v1.score({k: c for k, c in m2["holdout"].items() if c.get("real")})
    film_hold = v1.score(film["holdout"])
    lines = [
        ("holdout_f1", pct(hold["f1"])),
        ("holdout_recall_soft", pct(hold["recall_soft"])),
        ("holdout_false_positives", str(hold["fp"])),
        ("holdout_real_f1", pct(real_hold["f1"])),
        ("film_holdout_fp", str(film_hold["fp"])),
        ("film_holdout_recall", pct(film_hold["recall"])),
    ]
    apollo = load_apollo()
    if apollo:
        started = time.time()
        lines += apollo_lines("holdout", score_with_kinds(apollo["holdout"]))
        lines.append(("apollo_runtime_s", f"{time.time() - started:.2f}"))
    return lines


def main():
    parser = argparse.ArgumentParser(description="Score detect_cuts on the soft-transition sets.")
    parser.add_argument("--holdout", action="store_true",
                        help="print held-out metrics instead of climb metrics")
    args = parser.parse_args()
    lines = holdout_lines() if args.holdout else climb_lines()
    shutil.rmtree(v1.cache_dir, ignore_errors=True)
    for name, value in lines:
        print(f"METRIC {name}={value}")


if __name__ == "__main__":
    main()
