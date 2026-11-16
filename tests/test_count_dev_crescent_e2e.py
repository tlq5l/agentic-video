"""End to end: count_dev's moving crescents keep their colour under the motion blur.

Round-8 probe: seed 6102 occ28's pink crescent kept its colour in 0 of 90 frames with the blur. Round-9 box printout: drawn
after the blur at 1.5x size, 108 of 108 frames on 6102 occ28 and 31 of 31 on 6103 occ28 keep 20+ analysis-size pixels.
"""
import shutil
import subprocess

import pytest
from PIL import Image

from benchmarks.dev.count_dev import generate
from src.retrieval import occurrence

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _frame(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return Image.frombuffer("RGB", (640, 360), raw)


@pytest.mark.parametrize("seed, min_frames", [(6102, 60), (6103, 25)])
def test_moving_crescents_keep_their_colour_under_motion_blur(tmp_path, seed, min_frames):
    spec = next(s for s in generate.VIDEOS if s[0] == "occ28")
    _questions, _truths, render = generate.plan_video(spec, seed)
    assert render["motion_blur"]
    crescents = [e for e in render["events"] if e["shape"] == "crescent"]
    assert crescents
    path = tmp_path / "occ28.mp4"
    generate._render(path, render["duration"], render["events"], render["texts"], render["cuts"], render["seed"],
                     render["motion_blur"], render["life_color"], render["noise"], render["tint"])
    full, analysis = [], []
    for event in crescents:
        t = event["start"] + .1
        while t < event["end"] - .1:
            image = _frame(path, t)
            small = image.resize(occurrence._analysis_size(image))
            full.append(int(occurrence.color_mask(image, event["color"]).sum()))
            analysis.append(int(occurrence.color_mask(small, event["color"]).sum()))
            t += .2
    assert len(full) >= min_frames
    assert sum(px > 0 for px in full) >= .9 * len(full), full
    assert sum(px >= 20 for px in analysis) >= .9 * len(analysis), analysis
