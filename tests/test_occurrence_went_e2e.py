"""End to end: with a direction passed, a span counts for it when it moved at least BACK_RISE that way inside the span,
whatever its end-to-end travel, and the span's first centre counts as a low point even before a detection dropout.
From count_dev seed 6103 occ26_q1 (round 14): a cross-and-back pass first seen mid-way read right_to_left (dx -.201), and
its .656 rise read .484 because the first centre was averaged with one seen 0.8 s later."""
import shutil

import pytest

try:
    from tests.test_occurrences import BLUE, H, W, count, make_clip, shape
except ImportError:
    from test_occurrences import BLUE, H, W, count, make_clip, shape
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _circle(draw, x):
    shape(draw, "circle", int(x * W), H // 2, 16, BLUE, outline=False)


def test_bounce_seen_late_counts_its_way_out(tmp_path):
    """Right from x .30 to .92 over 1.0-2.0 s, then back to .08 by 3.0 s (fps 10): end to end it reads right_to_left
    (dx about -.21), yet it first went about .56 right, so it counts once each way."""
    def paint(draw, t):
        if .95 <= t < 2.0:
            _circle(draw, .30 + .62 * (t - 1.0))
        elif 2.0 <= t < 3.05:
            _circle(draw, .92 - .84 * (t - 2.0))
    video = make_clip(tmp_path / "late.mp4", 3.4, paint)
    assert count(video, tmp_path / "a", description="blue circle", direction="left_to_right", fps=10)["count"] == 1
    assert count(video, tmp_path / "b", description="blue circle", direction="right_to_left", fps=10)["count"] == 1


def test_first_centre_before_a_dropout_counts_as_start(tmp_path):
    """Seen once at x .12 (1.0 s), absent 0.4 s, then right from .44 at 1.6 s to .80 at 2.0 s and back to .24 by 2.8 s
    (fps 5). The lone sample joins the span; averaged with .44 its start would read .28 and the .68 rise only .45."""
    def paint(draw, t):
        if .95 <= t < 1.05:
            _circle(draw, .12)
        elif 1.55 <= t < 2.0:
            _circle(draw, .44 + .9 * (t - 1.6))
        elif 2.0 <= t < 2.85:
            _circle(draw, .80 - .7 * (t - 2.0))
    video = make_clip(tmp_path / "dropout.mp4", 3.2, paint)
    assert count(video, tmp_path, description="blue circle", direction="left_to_right")["count"] == 1


def test_one_way_crossing_counts_only_its_way(tmp_path):
    """Left from x .92 to .08 over 1.0-3.0 s (fps 10): left_to_right counts nothing."""
    def paint(draw, t):
        if .95 <= t < 3.05:
            _circle(draw, .92 - .42 * (t - 1.0))
    video = make_clip(tmp_path / "oneway.mp4", 3.4, paint)
    assert count(video, tmp_path, description="blue circle", direction="left_to_right", fps=10)["count"] == 0
