"""Large videos settle each scene change with a short decode of its frames; the small ones scale every frame in the scan."""
import shutil
import subprocess

import pytest

from src.core.cache import MediaCache
from src.media import scene
from src.retrieval.engine import AgenticVideoEngine

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip_with_a_cross_fade_and_a_low_scoring_cut(path):
    """3 s bars, a 1 s cross-fade at 2 s into plain colour, then one step too small for the hard-cut filter."""
    rate = "size=320x180:rate=24"
    sources = [f"smptebars={rate}", f"color=c=0x2060a0:{rate}", f"smptebars={rate}",
               f"smptebars={rate},eq=brightness=-0.03", f"smptebars={rate},eq=brightness=-0.12"]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for source in sources:
        cmd += ["-f", "lavfi", "-t", "3", "-i", source]
    graph = ("[0:v][1:v]xfade=transition=fade:duration=1:offset=2[faded];"
             "[2:v]trim=end_frame=72,setpts=PTS-STARTPTS[before];"
             "[3:v]trim=end_frame=1,setpts=PTS-STARTPTS[step];"
             "[4:v]trim=end_frame=72,setpts=PTS-STARTPTS[after];"
             "[faded][before][step][after]concat=n=4:v=1")
    cmd += ["-filter_complex", graph, "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


def test_lazy_settle_path_reports_the_same_changes_as_the_small_video_path(tmp_path, monkeypatch):
    clip = _clip_with_a_cross_fade_and_a_low_scoring_cut(tmp_path / "mixed.mp4")
    reads = []
    real_gray_frames = scene._gray_frames

    def counting_gray_frames(video_path, times):
        frames = real_gray_frames(video_path, times)
        reads.append(frames is not None)
        return frames

    monkeypatch.setattr(scene, "_gray_frames", counting_gray_frames)

    small = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "small")).detect_cuts()
    assert reads == []  # the default gate scales every frame in the scan, so nothing is settled later

    monkeypatch.setattr(scene, "SCAN_SMALL_PIXEL_SEC", 0)
    lazy = AgenticVideoEngine(clip, cache=MediaCache(base_dir=tmp_path / "lazy")).detect_cuts()

    assert reads, "the lazy path never read a window of frames"
    assert all(reads), "a short read failed, so settling fell back to a full rescan"
    # The fade (kind "transition"), the hard cut where it ends, and the low-scoring cut only the soft scan finds.
    assert [(cut.kind, int(cut.timestamp_sec)) for cut in small] == [("transition", 2), ("cut", 5), ("cut", 8)]
    assert small[2].score < 0.25
    assert lazy == small
