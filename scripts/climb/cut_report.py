"""Cut-count report for the four Rakan edits (separate from the detection score).

Runs the checker on each edit, then compares the cuts it lists (scene-score cuts and
same-footage cuts) with the joins measured from the edits (scripts/climb/cut_truth.json).
A join counts as found when a listed cut is within TOL_SEC of it. Usage:
    .venv/bin/python -u scripts/climb/cut_report.py [--label NAME]
Writes .climb-tmp/cuts/<label>.json (scratch, gitignored) and prints one line per clip.
"""
import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "bin" / "python"
EDITS = Path.home() / "Code" / "rakan-edits" / "out"
TOL_SEC = 0.1


def check_clip(name: str, outdir: Path) -> dict:
    subprocess.run([str(PY), "-m", "src.loop", "check", str(EDITS / f"{name}.mp4"), "--out", str(outdir)],
                   cwd=REPO, stdout=subprocess.DEVNULL, check=True)
    checked = json.loads((outdir / "report.json").read_text()).get("checked", {})
    scene = [c["time_sec"] for c in checked.get("cuts", [])]
    same = [c["time_sec"] for c in checked.get("same_footage_cuts", [])]
    return {"scene": scene, "same": same}


def score(truth: list, found: list) -> dict:
    missed = [t for t in truth if not any(abs(t - f) <= TOL_SEC for f in found)]
    extra = [f for f in found if not any(abs(f - t) <= TOL_SEC for t in truth)]
    return {"truth": len(truth), "found": len(truth) - len(missed), "missed": missed,
            "extra": sorted(set(extra))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="cuts")
    args = parser.parse_args()
    truth = json.loads((REPO / "scripts/climb/cut_truth.json").read_text())
    scratch = REPO / ".climb-tmp" / "cuts" / args.label
    result = {}
    for name, entry in truth.items():
        found = check_clip(name, scratch / name)
        union = sorted(set(found["scene"]) | set(found["same"]))
        result[name] = {"scene_only": score(entry["joins"], found["scene"]),
                        "with_same_footage": score(entry["joins"], union),
                        "same_footage_listed": found["same"]}
        r = result[name]
        print("{:12s} joins {:2d} | scene only: found {:2d} extra {} | with same-footage: found {:2d} extra {} missed {}".format(
            name, r["scene_only"]["truth"], r["scene_only"]["found"], len(r["scene_only"]["extra"]),
            r["with_same_footage"]["found"], r["with_same_footage"]["extra"], r["with_same_footage"]["missed"]))
    total = sum(v["with_same_footage"]["truth"] for v in result.values())
    print("TOTAL joins {} | scene only found {} | with same-footage found {} extra {}".format(
        total, sum(v["scene_only"]["found"] for v in result.values()),
        sum(v["with_same_footage"]["found"] for v in result.values()),
        sum(len(v["with_same_footage"]["extra"]) for v in result.values())))
    (scratch / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
