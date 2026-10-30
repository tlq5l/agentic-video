#!/usr/bin/env python3
"""Free offline scorer for the count dev set.

Calls count_occurrences / count_repetitions through the MCP handler (`handle_call`), with the
argument names an agent would use and the plain-language target from the question, then
compares `count` with the truth. Only defaults are passed except the crossing option, which is
used for "left to right" questions when the tool schema offers one, and the `args` of a trap
question (the min_gap_sec or max_spans an agent chose in the observed failures). No model or API calls.

    python -m benchmarks.dev.count_dev.score_tools [--split train|train2|test|all|train,train2] [--only ID ...] [--out DIR]

Generate the media first (python -m benchmarks.dev.count_dev.generate; for the real repetition
clips also python -m benchmarks.dev.count_dev.fetch_real_reps, which writes real_reps.jsonl).
CLIP-described targets and repetition counting need the local model:
export HF_HOME=<cache with the CLIP weights> HF_HUB_OFFLINE=1.

METRIC occ_trap_* lines cover the trap questions: exact is the right count, safe is right or explicitly
flagged (capped, merged gaps, appearances in more than one direction, or no count from the tool), silent_wrong is a
wrong count with no flag (should be 0).
A tool error is never safe: it counts as a wrong, unflagged answer.
METRIC rep_* lines are the synthetic clips; real_rep_* lines are the real Countix-train clips
(real_rep_tol counts within 1, or within 10% when the true count is above 10).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
POLL_SEC, TIMEOUT_SEC = .5, 900
DEFAULT_MAX_SPANS = 20  # the tool's default `max_spans`; a count this large is a cap, not an answer
SPLITS = ("train", "train2", "test")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def crossing_option(handler):
    """Name and value of the tool's documented left-to-right option, or None when the tool has none."""
    properties = handler._TOOL_SCHEMAS["count_occurrences"].get("properties", {})
    if "direction" in properties:
        return "direction", "left_to_right"
    return None


def tool_call(handler, row, video, crossing):
    args = {"source": str(video)}
    if row["tool"] == "count_occurrences":
        args["text" if row["target_kind"] == "text" else "description"] = row["target"]
        if row["mode"] == "left_to_right" and crossing:
            args[crossing[0]] = crossing[1]
        args.update(row.get("args", {}))
    return row["tool"], args


def run_tool(handler, name, args):
    """Repeat the call while status=in_progress, as the tool description tells an agent to."""
    deadline = time.time() + TIMEOUT_SEC
    while True:
        try:
            outcome = handler.handle_call(name, args)
        except Exception as exc:  # the server turns these into an error result
            return {"error": str(exc)}
        if not outcome.get("ok"):
            return {"error": json.dumps(outcome.get("error"))}
        data = outcome["data"]
        if data.get("status") == "error":
            return {"error": data.get("message") or "tool returned status=error"}
        if data.get("status") != "in_progress":
            return data
        if time.time() > deadline:
            return {"error": "timed out"}
        time.sleep(POLL_SEC)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="all",
                        help="train, train2, test, all, or a comma list such as train,train2")
    parser.add_argument("--only", nargs="*", help="question ids")
    parser.add_argument("--out", help="media directory used by generate.py")
    parser.add_argument("--tool", choices=["count_occurrences", "count_repetitions"])
    parser.add_argument("--rows-json", help="write one record per scored question here (to combine runs, e.g. text rows on macOS)")
    args = parser.parse_args()
    from benchmarks.dev.count_dev.generate import data_root
    splits = set(args.split.split(","))
    if not splits <= {"train", "train2", "test", "all"}:
        parser.error(f"unknown split in {args.split!r}")
    videos = data_root(args.out) / "videos"
    with tempfile.TemporaryDirectory(prefix="count-dev-cache-") as cache:
        os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cache  # a fresh cache: results never come from an older algorithm
        os.environ["AGENTIC_VIDEO_TEMP_DIR"] = str(Path(cache) / "tmp")
        from src.executor import mcp_server as handler
        crossing = crossing_option(handler)
        questions = read_jsonl(HERE / "questions.jsonl")
        truth = {row["id"]: row for row in read_jsonl(HERE / "truth.jsonl")}
        real = read_jsonl(HERE / "real_reps.jsonl") if (HERE / "real_reps.jsonl").exists() else []
        questions += [{"id": r["id"], "tool": "count_repetitions", "target": r["class"], "video": r["video"],
                       "split": r["split"], "real": True} for r in real]
        truth.update({r["id"]: {"count": r["count"], "layout": r["class"][:9]} for r in real})
        traps = []
        results = {name: {split: [] for split in SPLITS} for name in ("count_occurrences", "count_repetitions", "real_rep")}
        print(f"crossing option: {crossing[0] + '=' + crossing[1] if crossing else 'none (tool has no crossing option)'}")
        print(f"{'id':10} {'split':5} {'layout':9} {'truth':>5} {'answer':>6}  detail")
        records = []
        for row in questions:
            if not ("all" in splits or row["split"] in splits) or (args.only and row["id"] not in args.only):
                continue
            if args.tool and row["tool"] != args.tool:
                continue
            name, call = tool_call(handler, row, videos / row["video"], crossing)
            data = run_tool(handler, name, call)
            answer = data.get("count")
            expected = truth[row["id"]]["count"]
            detail = data.get("error") or " ".join(
                f"{k}={data[k]}" for k in ("confidence", "count_reliable", "method_used", "spans_omitted", "capped", "gaps_merged",
                                   "count_without_merges", "min_gap_sec_clamped_from") if k in data)
            results["real_rep" if row.get("real") else name][row["split"]].append((expected, answer))
            flagged = ((answer is None and data.get("status") == "complete") or bool(data.get("capped") or data.get("gaps_merged"))
                       or bool(data.get("travel_counts")))
            records.append({"id": row["id"], "split": row["split"], "tool": name, "trap": "args" in row,
                            "expected": expected, "answer": answer, "flagged": flagged, "error": "error" in data})
            if "args" in row:
                traps.append((expected, answer, flagged))
            print(f"{row['id']:10} {row['split']:5} {truth[row['id']]['layout']:9} {expected:5} "
                  f"{'-' if answer is None else answer:>6}  {detail}", flush=True)
    for name, prefix in (("count_occurrences", "occ"), ("count_repetitions", "rep"), ("real_rep", "real_rep")):
        for split in SPLITS:
            rows = results[name][split]
            if not rows:
                continue
            exact = sum(a == e for e, a in rows) / len(rows)
            near = sum(a is not None and abs(a - e) <= 1 for e, a in rows) / len(rows)
            print(f"METRIC {prefix}_exact_{split}={exact:.2f}")
            print(f"METRIC {prefix}_within1_{split}={near:.2f}")
            if name == "real_rep":
                tol = sum(a is not None and abs(a - e) <= max(1, .1 * e) for e, a in rows) / len(rows)
                rel = sum(abs((a or 0) - e) / e for e, a in rows) / len(rows)
                print(f"METRIC {prefix}_tol_{split}={tol:.2f}")
                print(f"METRIC {prefix}_relerr_{split}={rel:.2f}")
    if traps:
        print(f"METRIC occ_trap_exact={sum(a == e for e, a, _ in traps) / len(traps):.2f}")
        print(f"METRIC occ_trap_safe={sum(a == e or f for e, a, f in traps) / len(traps):.2f}")
        print(f"METRIC occ_trap_silent_wrong={sum(a != e and not f for e, a, f in traps)}")
    if args.rows_json:
        Path(args.rows_json).write_text(json.dumps(records, indent=1), encoding="utf-8")
    every = [a for split in results["count_occurrences"].values() for _, a in split]
    if every:
        print(f"METRIC occ_cap_hits={sum(a is not None and a >= DEFAULT_MAX_SPANS for a in every)}")


if __name__ == "__main__":
    main()
