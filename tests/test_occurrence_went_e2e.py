"""End to end: with a direction passed, a span counts for it when it moved at least BACK_RISE that way inside the span,
whatever its end-to-end travel, and the span's first centre counts as a low point even before a detection dropout.
From count_dev seed 6103 occ26_q1 (round 14): a cross-and-back pass first seen mid-way read right_to_left (dx -.201), and
its .656 rise read .484 because the first centre was averaged with one seen 0.8 s later. Round 15: a first centre more
than one step (TELEPORT_JUMP) from the next is not a low point, a rise under BACK_RISE does not count, and results stored
under the key without went-v1 are not served."""
import hashlib
import json
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


def test_stray_first_centre_far_from_the_next_is_not_a_start(tmp_path):
    """A small blue circle shows at x .06 at 1.0-1.1 s and a bigger one enters at .92 at 1.1 s and goes left to .10 by
    2.5 s (fps 10). The small one keeps the span going at 1.1 s (nearest match), so the span's centres read .06, then .92
    down to .10: a one-way right-to-left crossing. Its first centre sits .86 from the next, more than one step
    (TELEPORT_JUMP), so it is no low point and left_to_right counts nothing (round-14 breaker case)."""
    def paint(draw, t):
        if .95 <= t < 1.15:
            shape(draw, "circle", int(.06 * W), H // 2, 10, BLUE, outline=False)
        if 1.05 <= t < 2.55:
            _circle(draw, .92 - .586 * (t - 1.1))
    video = make_clip(tmp_path / "stray.mp4", 3.0, paint)
    assert count(video, tmp_path / "a", description="blue circle", direction="left_to_right", fps=10)["count"] == 0
    assert count(video, tmp_path / "b", description="blue circle", direction="right_to_left", fps=10)["count"] == 1


def test_rise_under_back_rise_counts_only_its_net_way(tmp_path):
    """Right from x .46 to .92 over 1.0-1.8 s, then back to .08 by 3.2 s (fps 10): inside the span it went about .43
    right, under BACK_RISE (.5), so it counts once right_to_left and not left_to_right. Pins BACK_RISE from below: at .3
    (round-14 mutant M4) left_to_right would count it."""
    def paint(draw, t):
        if .95 <= t < 1.8:
            _circle(draw, .46 + .575 * (t - 1.0))
        elif 1.8 <= t < 3.25:
            _circle(draw, .92 - .6 * (t - 1.8))
    video = make_clip(tmp_path / "under.mp4", 3.6, paint)
    assert count(video, tmp_path / "a", description="blue circle", direction="left_to_right", fps=10)["count"] == 0
    assert count(video, tmp_path / "b", description="blue circle", direction="right_to_left", fps=10)["count"] == 1


def test_results_stored_before_went_v1_are_not_served(tmp_path, monkeypatch):
    """A result stored under the key without went-v1 (by code before round 14's back check) is not served: the call
    counts again. Planted: the stored result moved to the old key with count 99, the current key's file removed."""
    def paint(draw, t):
        if .95 <= t < 3.05:
            _circle(draw, .92 - .42 * (t - 1.0))
    video = make_clip(tmp_path / "key.mp4", 3.4, paint)
    real_sha256, payloads = hashlib.sha256, []

    def spy(data=b"", **kwargs):
        payloads.append(bytes(data))
        return real_sha256(data, **kwargs)
    monkeypatch.setattr(hashlib, "sha256", spy)
    first = count(video, tmp_path, description="blue circle", direction="left_to_right", fps=10)["count"]
    monkeypatch.setattr(hashlib, "sha256", real_sha256)
    key = next(json.loads(p) for p in payloads if b'"join-v1"' in p)
    old = [part for part in key if part != "went-v1"]
    new_tag = real_sha256(json.dumps(key).encode()).hexdigest()[:20]
    old_tag = real_sha256(json.dumps(old).encode()).hexdigest()[:20]
    [stored] = (tmp_path / "cache").rglob(f"{new_tag}.occurrences.json")
    result = json.loads(stored.read_text())
    stored.unlink()
    result["count"] = 99
    (stored.parent / f"{old_tag}.occurrences.json").write_text(json.dumps(result))
    assert count(video, tmp_path, description="blue circle", direction="left_to_right", fps=10)["count"] == first
