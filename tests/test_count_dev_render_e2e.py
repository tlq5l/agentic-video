"""End to end: count_dev draws the back-layout target on the path its truth describes, read from the rendered frames."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from benchmarks.dev.count_dev import generate

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
W, H = 640, 360  # the crop at the end of generate._render


def _frame(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(int)


def test_back_layout_target_starts_left_turns_right_and_ends_left(tmp_path):
    spec = next(s for s in generate.VIDEOS if s[0] == "occ27")
    _questions, truths, render = generate.plan_video(spec, generate.SEED)
    truth = next(t for t in truths if t["id"] == "occ27_q1")
    assert truth["layout"] == "back"
    committed = {}
    for line in (generate.HERE / "truth.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            committed[row["id"]] = row
    assert committed["occ27_q1"]["intervals"] == truth["intervals"]
    start, end = truth["intervals"][0]
    target = next(e for e in render["events"] if e["path"] == "cross_back" and e["start"] == start)

    def make(name, events):  # same seed, background, tints and noise; only the drawn events differ
        path = tmp_path / name
        generate._render(path, int(end) + 2, events, [], render["cuts"], render["seed"], render["motion_blur"],
                         render["life_color"], render["noise"], render["tint"])
        return path

    with_target, without = make("with.mp4", [target]), make("without.mp4", [])
    xs = []
    for u in (.1, .5, .9):
        t = start + u * (end - start)
        diff = np.abs(_frame(with_target, t) - _frame(without, t)).sum(axis=2)
        cols = np.nonzero(diff > 90)[1]
        assert cols.size > 150, f"target not visible at {t:.2f}s ({cols.size} changed pixels)"
        xs.append(float(np.median(cols)))
    first, turn, last = xs
    assert first < 250 and last < 250, xs  # the path starts and ends at the left edge
    assert turn > 450, xs  # and turns at the right
