"""A cut is reported at the strongest frame of its change, not at the first frame over the threshold."""
import re
import shutil
import subprocess

import pytest

from src.core.cache import MediaCache
from src.media.scene import detect_scenes
from src.retrieval.engine import AgenticVideoEngine

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _grey(level):
    return f"color=c=0x{level:02x}{level:02x}{level:02x}:" + "{sr}"


def _bars(brightness=0.0):
    return "smptebars={sr}" + (f",eq=brightness={brightness}" if brightness else "")


def _clip(path, parts, rate):
    """Join (lavfi source, frame count) parts with hard cuts, encoded without loss."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    graph = []
    for i, (source, frames) in enumerate(parts):
        cmd += ["-f", "lavfi", "-i", source.format(sr=f"size=320x180:rate={rate}")]
        graph.append(f"[{i}:v]trim=end_frame={frames},setpts=PTS-STARTPTS[p{i}]")
    graph.append("".join(f"[p{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1")
    cmd += ["-filter_complex", ";".join(graph), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


def _frame_scores(path):
    """ffmpeg's scene score of every frame, by frame number."""
    shown = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vf", r"select=gte(scene\,0),metadata=print:key=lavfi.scene_score",
                            "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    return [float(value) for value in re.findall(r"lavfi\.scene_score=([0-9.]+)", shown)]


def _frames(cuts, rate):
    return [round(cut.timestamp_sec * rate) for cut in cuts]


def test_cut_after_a_partly_changed_frame_lands_on_the_strongest_frame(tmp_path):
    # Frame 48 changes a little and scores just over the threshold; frame 49 is the real cut.
    clip = _clip(tmp_path / "partial.mp4", [(_grey(40), 48), (_grey(72), 1), (_grey(190), 48)], 24)
    scores = _frame_scores(clip)
    assert 0.25 < scores[48] < scores[49]

    direct = detect_scenes(clip, threshold=0.25, cache=MediaCache(base_dir=tmp_path / "direct"))
    engine = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "scan"))
    cold = engine.detect_cuts()
    warm = engine.detect_cuts()

    assert _frames(direct, 24) == [49]
    assert direct[0].score == pytest.approx(scores[49], abs=1e-3)
    assert _frames(cold, 24) == [49]
    assert [cut.kind for cut in cold] == ["cut"]
    assert cold[0].score == pytest.approx(scores[49], abs=1e-3)
    assert warm == cold


def test_low_scoring_cut_after_a_partly_changed_frame_is_timed_at_its_strongest_frame(tmp_path):
    # Both steps score under the hard-cut threshold, so the soft scan is what reports this cut.
    clip = _clip(tmp_path / "soft.mp4", [(_bars(), 72), (_bars(-0.03), 1), (_bars(-0.12), 72)], 24)
    scores = _frame_scores(clip)
    assert 0.05 < scores[72] < scores[73] < 0.25

    found = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "scan")).detect_cuts()

    assert [cut.kind for cut in found] == ["cut"]
    assert _frames(found, 24) == [73]
    assert found[0].score == pytest.approx(scores[73], abs=1e-3)


@pytest.mark.parametrize(
    "rate, gap",
    [
        pytest.param(30, 2, id="two-frames-later-is-not-adjacent"),
        pytest.param(24, 7, id="second-cut-inside-the-half-second-minimum"),
    ],
)
def test_a_stronger_change_after_a_gap_does_not_replace_the_first_cut(tmp_path, rate, gap):
    first = 2 * rate
    second = first + gap
    clip = _clip(tmp_path / "two.mp4", [(_grey(40), first), (_grey(80), gap), (_grey(200), first)], rate)
    scores = _frame_scores(clip)
    assert 0.25 < scores[first] < scores[second]

    # Only frames next to each other are one change. The later, stronger one is a second cut, and the 0.5 s minimum drops it.
    direct = detect_scenes(clip, threshold=0.25, cache=MediaCache(base_dir=tmp_path / "direct"))
    found = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "scan")).detect_cuts()

    assert _frames(direct, rate) == [first]
    assert _frames(found, rate) == [first]


@pytest.mark.parametrize("length", [1, 2])
def test_a_flash_gives_one_cut_at_its_first_frame(tmp_path, length):
    clip = _clip(tmp_path / "flash.mp4", [(_grey(40), 48), (_grey(120), length), (_grey(40), 48)], 24)

    direct = detect_scenes(clip, threshold=0.25, cache=MediaCache(base_dir=tmp_path / "direct"))
    found = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "scan")).detect_cuts()

    assert _frames(direct, 24) == [48]
    assert _frames(found, 24) == [48]


def test_steady_motion_without_a_cut_gives_no_cut(tmp_path):
    # The picture scrolls the whole time; no frame reaches the threshold.
    clip = _clip(tmp_path / "motion.mp4", [("testsrc2={sr},scroll=h=0.01", 96)], 24)
    assert max(_frame_scores(clip)) < 0.25

    direct = detect_scenes(clip, threshold=0.25, cache=MediaCache(base_dir=tmp_path / "direct"))
    found = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "scan")).detect_cuts()

    assert direct == []
    assert found == []
