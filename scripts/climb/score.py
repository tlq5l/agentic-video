#!/usr/bin/env python3
"""Score the render checker against planted faults (scripts/climb/answer_key.json).

Runs the real checker (`python -m src.loop check`) on:
  - each faulted case (<base>-pristine, cues.json and INTENT.md next to it),
  - its plain sibling (<base>, or <base>/project), used only to find faults shared by both sides,
  - the Rakan clean floor clips (no cues, no intent).

Per case: caught - 2 x false alarms. A key entry is caught when a confirmed problem has one of its
kinds within 0.5 s of its time. A confirmed problem on the faulted side that matches nothing is a
false alarm, unless the plain sibling confirms the same kind within 0.5 s (shared, not scored).
Dev and held-out totals sum the per-case scores. Floor = confirmed problems on the Rakan clips (must be 0).

Usage:
  .venv/bin/python scripts/climb/score.py [--label NAME] [--compare path/to/other.json]

Scorer version 2 (2026-11-02): duplicates of a caught fault are reported, not penalised; 
Writes .climb-tmp/score/<label>.json and prints one line per case.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
KEY = HERE / "answer_key.json"
MEDIA = Path(os.path.expanduser("~/.local/share/media-loop-work"))
RAKAN = Path(os.path.expanduser("~/Code/rakan-edits/out"))
TOL_SEC = 0.5
CHECKED_FIELDS = ("duration_sec", "resolution", "loudness", "cuts", "black_frames",
                  "flash_frames", "frozen_frames", "silence_gaps", "errored_checks")
PY = REPO / ".venv" / "bin" / "python"


def base_name(key_name: str) -> str:
    return key_name.replace("-pristine", "")


def faulted_inputs(key_name: str):
    d = MEDIA / (base_name(key_name) + "-pristine")
    return d / "out.mp4", d / "cues.json", d / "INTENT.md"


def plain_inputs(key_name: str):
    d = MEDIA / base_name(key_name)
    if (d / "project").is_dir():
        d = d / "project"
    return d / "out.mp4", d / "cues.json", d / "INTENT.md"


def run_check(video: Path, cues: Path, intent: Path, out: Path, use_inputs: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), "-m", "src.loop", "check", str(video), "--out", str(out)]
    if use_inputs:
        if cues.exists():
            cmd += ["--cues", str(cues)]
        if intent.exists():
            cmd += ["--intent", str(intent)]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    report_path = out / "report.json"
    if proc.returncode not in (0, 1) or not report_path.exists():
        return {"error": (proc.stderr or proc.stdout)[-400:], "confirmed": [], "checked_ok": False}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    confirmed = [{"kind": c.get("kind"), "start_sec": c.get("start_sec"),
                  "end_sec": c.get("end_sec"), "id": c.get("id")}
                 for c in report.get("candidates", []) if c.get("confirmed")]
    checked = report.get("checked")
    available = checked if isinstance(checked, dict) else {}
    missing = [f for f in CHECKED_FIELDS if f not in available]
    return {"confirmed": confirmed, "checked_ok": not missing, "checked_missing": missing}


def match_entries(entries, confirmed):
    used = set()
    results = []
    for entry in entries:
        hit = None
        for i, c in enumerate(confirmed):
            if i in used or c["kind"] not in entry["kinds"]:
                continue
            if abs(float(c["start_sec"]) - float(entry["time"])) <= TOL_SEC:
                hit = i
                break
        if hit is not None:
            used.add(hit)
        results.append({"time": entry["time"], "kinds": entry["kinds"], "label": entry.get("label", ""),
                        "caught": hit is not None,
                        "matched_kind": confirmed[hit]["kind"] if hit is not None else None})
    leftovers = [c for i, c in enumerate(confirmed) if i not in used]
    # A second confirmed item on an entry that is already caught (same kind, within the tolerance)
    # is a duplicate of a real fault, not a false alarm.
    duplicates = []
    still_left = []
    for c in leftovers:
        if any(c["kind"] in e["kinds"] and abs(float(c["start_sec"]) - float(e["time"])) <= TOL_SEC
               and r["caught"] for e, r in zip(entries, results)):
            duplicates.append(c)
        else:
            still_left.append(c)
    return results, still_left, duplicates


def score_case(key_name: str, entries, label_dir: Path) -> dict:
    fv, fc, fi = faulted_inputs(key_name)
    pv, pc, pi = plain_inputs(key_name)
    fault = run_check(fv, fc, fi, label_dir / (key_name + "-faulted"))
    plain = run_check(pv, pc, pi, label_dir / (key_name + "-plain"))
    if "error" in fault:
        return {"case": key_name, "error": fault["error"], "score": None}
    results, leftovers, duplicates = match_entries(entries, fault["confirmed"])
    plain_items = plain.get("confirmed", [])
    false_alarms, shared = [], []
    for c in leftovers:
        twin = any(p["kind"] == c["kind"] and abs(float(p["start_sec"]) - float(c["start_sec"])) <= TOL_SEC
                   for p in plain_items)
        (shared if twin else false_alarms).append({"kind": c["kind"], "time": c["start_sec"]})
    caught = sum(r["caught"] for r in results)
    return {
        "case": key_name,
        "caught": caught,
        "total": len(entries),
        "false_alarms": false_alarms,
        "shared_not_scored": shared,
        "duplicates_not_scored": [{"kind": c["kind"], "time": c["start_sec"]} for c in duplicates],
        "missed": [r for r in results if not r["caught"]],
        "caught_list": [r for r in results if r["caught"]],
        "score": caught - 2 * len(false_alarms),
        "checked_ok": fault.get("checked_ok"),
        "checked_missing": fault.get("checked_missing"),
    }


def floor_check(label_dir: Path) -> dict:
    clips = []
    for clip in sorted(RAKAN.glob("*.mp4")):
        out = label_dir / ("floor-" + clip.stem)
        result = run_check(clip, Path("/nonexistent"), Path("/nonexistent"), out, use_inputs=False)
        if "error" in result:
            clips.append({"clip": clip.name, "error": result["error"]})
            continue
        clips.append({"clip": clip.name, "confirmed": result["confirmed"],
                      "checked_ok": result["checked_ok"]})
    total = sum(len(c.get("confirmed", [])) for c in clips)
    errored = [c["clip"] for c in clips if "error" in c]
    return {"clips": clips, "confirmed_total": total, "errored": errored}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", default=None, help="a previous score JSON to compare against")
    args = ap.parse_args()
    key = json.loads(KEY.read_text(encoding="utf-8"))
    label_dir = REPO / ".climb-tmp" / "score" / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    summary = {"label": args.label, "dev": {}, "held_out": {}, "floor": None}
    for group in ("dev", "held_out"):
        for key_name, entries in key[group].items():
            res = score_case(key_name, entries, label_dir)
            summary[group][key_name] = res
            if res.get("score") is None:
                print("{} {}: ERROR {}".format(group, key_name, res.get("error", "")[:200]))
            else:
                print("{:9s} {:20s} caught {}/{}  FA {}  shared {}  dup {}  score {}  checked_ok={}".format(
                    group, key_name, res["caught"], res["total"], len(res["false_alarms"]),
                    len(res["shared_not_scored"]), len(res["duplicates_not_scored"]), res["score"],
                    res["checked_ok"]))
    for group in ("dev", "held_out"):
        summary[group + "_total"] = sum(
            r["score"] for r in summary[group].values() if r.get("score") is not None)
    summary["floor"] = floor_check(label_dir)
    print("TOTAL dev {}  held_out {}  floor confirmed {}".format(
        summary["dev_total"], summary["held_out_total"], summary["floor"]["confirmed_total"]))
    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("COMPARE vs {}: dev {:+d}  held_out {:+d}  floor {:+d}".format(
            old.get("label"), summary["dev_total"] - old["dev_total"],
            summary["held_out_total"] - old["held_out_total"],
            summary["floor"]["confirmed_total"] - old["floor"]["confirmed_total"]))
    out_path = REPO / ".climb-tmp" / "score" / (args.label + ".json")
    out_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
