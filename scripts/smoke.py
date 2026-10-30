"""Known-answer smoke check: five tools on the 60 s synthetic clip, compared with its ground truth.

Usage: python scripts/smoke.py
Prints PASS or FAIL per check with its seconds, then exits 1 if any check failed.
The ground truth holds the duration, size, frame rate and each event's start time; the checks
compare only those, with the tolerances below. No model is loaded. Every run uses a new empty
cache in a temporary folder, so each check measures the current code, and the seconds are cold-cache times.
"""

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = tempfile.TemporaryDirectory(prefix="smoke-cache-")
os.environ["AGENTIC_VIDEO_CACHE_DIR"] = CACHE.name  # read when src.core.config is imported, so set before the import below

from src.executor.mcp_server import handle_call

CLIP = ROOT / "fixtures" / "synthetic_bench_60s.mp4"
TRUTH = json.loads((ROOT / "fixtures" / "synthetic_ground_truth.json").read_text(encoding="utf-8"))
AUDIO_EVENTS = [event for event in TRUTH["events"] if "Silent" not in event["audio_description"]]
DURATION_TOLERANCE_SEC = 0.1
FPS_TOLERANCE = 0.01
CUT_TOLERANCE_SEC = 0.5
CUE_TOLERANCE_MS = 80.0
LOUD_ABOVE_MEDIAN_DB = 6.0
CONTACT_SHEET_SAMPLES = 12


def call(name, arguments):
    outcome = handle_call(name, arguments)
    if not outcome.get("ok"):
        raise AssertionError(f"{name} returned an error: {outcome.get('error')}")
    return outcome["data"]


def check_probe_video():
    data = call("probe_video", {"source": str(CLIP)})
    problems = []
    if abs(data["duration_sec"] - TRUTH["duration_s"]) > DURATION_TOLERANCE_SEC:
        problems.append(f"duration {data['duration_sec']} s, expected {TRUTH['duration_s']} s")
    if (data["width"], data["height"]) != (TRUTH["width"], TRUTH["height"]):
        problems.append(f"size {data['width']}x{data['height']}, expected {TRUTH['width']}x{TRUTH['height']}")
    if abs(data["fps"] - TRUTH["fps"]) > FPS_TOLERANCE:
        problems.append(f"fps {data['fps']}, expected {TRUTH['fps']}")
    if problems:
        raise AssertionError("; ".join(problems))
    return f"{data['width']}x{data['height']}, {data['fps']} fps, {data['duration_sec']} s"


def check_detect_cuts():
    cuts = [cut["timestamp_sec"] for cut in call("detect_cuts", {"source": str(CLIP)})["cuts"]]
    missing = [
        event["name"] for event in TRUTH["events"]
        if not any(abs(time_sec - event["start_s"]) <= CUT_TOLERANCE_SEC for time_sec in cuts)
    ]
    if missing:
        raise AssertionError(f"no cut within {CUT_TOLERANCE_SEC} s of the start of: {', '.join(missing)}")
    return f"{len(cuts)} cuts; all {len(TRUTH['events'])} event starts matched within {CUT_TOLERANCE_SEC} s"


def check_loudness_timeline():
    data = call("loudness_timeline", {"source": str(CLIP)})
    if abs(data["end_sec"] - TRUTH["duration_s"]) > DURATION_TOLERANCE_SEC:
        raise AssertionError(f"timeline ends at {data['end_sec']} s, expected {TRUTH['duration_s']} s")
    median = statistics.median(window["rms_dbfs"] for window in data["windows"])
    quiet = []
    for event in AUDIO_EVENTS:
        inside = [w["rms_dbfs"] for w in data["windows"] if event["start_s"] <= w["start_sec"] < event["end_s"]]
        if not inside or max(inside) < median + LOUD_ABOVE_MEDIAN_DB:
            quiet.append(event["name"])
    if quiet:
        raise AssertionError(f"no window at least {LOUD_ABOVE_MEDIAN_DB} dB above the median in: {', '.join(quiet)}")
    return f"{len(AUDIO_EVENTS)} audio events each at least {LOUD_ABOVE_MEDIAN_DB} dB above the median"


def check_sync_check():
    cues = [{"time_sec": event["start_s"], "label": event["name"].lower()} for event in AUDIO_EVENTS]
    data = call("sync_check", {"source": str(CLIP), "cues": cues})
    problems = [
        f"{cue['label']} is {cue['status']}" for cue in data["cues"] if cue["status"] != "ok"
    ] + [
        f"{cue['label']} is {cue['offset_ms']} ms off" for cue in data["cues"]
        if cue.get("offset_ms") is None or abs(cue["offset_ms"]) > CUE_TOLERANCE_MS
    ]
    if len(data["cues"]) != len(cues):
        problems.append(f"{len(data['cues'])} cues reported for {len(cues)} sent")
    if problems:
        raise AssertionError("; ".join(problems))
    return f"{len(cues)} audio cues, all ok within {CUE_TOLERANCE_MS:g} ms"


def check_contact_sheet():
    data = call("contact_sheet", {"source": str(CLIP), "samples": CONTACT_SHEET_SAMPLES, "cols": 4})
    stamps = data["timestamps"]
    if data["sample_count"] != CONTACT_SHEET_SAMPLES or len(stamps) != CONTACT_SHEET_SAMPLES:
        raise AssertionError(f"{len(stamps)} tiles, expected {CONTACT_SHEET_SAMPLES}")
    if stamps != sorted(stamps) or not all(0 <= t <= TRUTH["duration_s"] for t in stamps):
        raise AssertionError("tile times are not increasing inside the clip")
    if not Path(data["file_path"]).is_file():
        raise AssertionError(f"sheet not on disk: {data['file_path']}")
    return f"{len(stamps)} tiles from {stamps[0]} s to {stamps[-1]} s, sheet on disk"


CHECKS = [
    ("probe_video", check_probe_video),
    ("detect_cuts", check_detect_cuts),
    ("loudness_timeline", check_loudness_timeline),
    ("sync_check", check_sync_check),
    ("contact_sheet", check_contact_sheet),
]


def main():
    started = time.monotonic()
    failed = []
    for name, check in CHECKS:
        begin = time.monotonic()
        try:
            detail, status = check(), "PASS"
        except Exception as exc:  # a tool error or a wrong value is a FAIL line, and the other checks still run
            detail, status = str(exc), "FAIL"
            failed.append(name)
        print(f"{status} {name} ({time.monotonic() - begin:.2f} s): {detail}")
    print(f"smoke: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed in {time.monotonic() - started:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
