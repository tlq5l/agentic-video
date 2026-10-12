"""Comprehensive unit and integration tests for media processing engine."""
import pytest
from pathlib import Path
from src.core.cache import MediaCache
from src.core.errors import MediaNotFoundError, ExtractionBoundsError
from src.media.probe import probe_video
from src.media.seeking import build_seek_arguments, extract_single_frame
from src.media.frames import extract_single_frame as extract_frame_with_provenance
from src.media.overlay import burn_timestamp_badge, apply_overlay_to_file
from src.media.sampling import sample_sparse_frames, sample_dense_window
from src.media.timing import parse_selected_frame_timestamp
from src.media.contact_sheet import compose_contact_sheet
import shutil
import subprocess

from src.media.scene import detect_scenes, detect_soft_transitions, partition_into_scene_segments
from PIL import Image

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_bench_60s.mp4"


def test_probe_fixture(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    meta = probe_video(str(FIXTURE_PATH), cache=cache)
    
    assert meta.duration_sec == pytest.approx(60.0, abs=0.5)
    assert meta.width == 640
    assert meta.height == 360
    assert meta.fps == pytest.approx(25.0, abs=0.1)
    assert meta.has_audio is True
    assert meta.video_codec == "h264"
    assert meta.audio_codec == "aac"
    assert len(meta.fingerprint) == 20


def test_probe_nonexistent(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    with pytest.raises(MediaNotFoundError):
        probe_video("non_existent_file.mp4", cache=cache)


def test_seek_arguments_logic():
    # Close to start (<= 5s) uses single seek
    pre, post = build_seek_arguments(3.0, margin_sec=5.0, strategy="two-stage")
    assert "-ss" in pre
    assert post == []
    
    # The compatibility strategy uses one source-clock-preserving input seek;
    # split pre/post seeking selected the wrong frame on VFR input.
    pre, post = build_seek_arguments(30.0, margin_sec=5.0, strategy="two-stage")
    assert pre == ["-ss", "30.000"]
    assert post == []


def test_selected_frame_pts_parser_is_fail_closed():
    stderr = "[Parsed_showinfo_0] n: 0 pts_time:4.500 duration:1"
    assert parse_selected_frame_timestamp(stderr) == pytest.approx(4.5)
    assert parse_selected_frame_timestamp("no showinfo receipt") is None


def test_extract_single_frame(tmp_path):
    out_frame = tmp_path / "frame_10s.jpg"
    frame_info = extract_single_frame(
        str(FIXTURE_PATH),
        timestamp_sec=10.0,
        output_path=str(out_frame),
        width=512,
        use_two_stage=True
    )
    
    assert out_frame.exists()
    assert frame_info.timestamp_sec == 10.0
    assert frame_info.timecode == "00:00:10.000"
    assert frame_info.width == 512
    # Check aspect ratio preservation: 640x360 scaled to 512 -> height = 288
    assert frame_info.height == 288


def test_frame_exposes_requested_and_decoded_timestamp(tmp_path):
    frame_info = extract_frame_with_provenance(
        str(FIXTURE_PATH),
        timestamp_sec=10.0,
        output_path=tmp_path / "provenance.jpg",
        width=320,
        overlay_timestamp=False,
    )

    assert frame_info.requested_timestamp == pytest.approx(10.0)
    assert frame_info.decoded_timestamp is not None
    assert frame_info.timestamp_delta is not None
    assert frame_info.decoded_timestamp == pytest.approx(10.0, abs=0.1)
    assert frame_info.timestamp_delta == pytest.approx(
        frame_info.decoded_timestamp - frame_info.requested_timestamp,
        abs=1e-6,
    )


def test_burn_timestamp_overlay(tmp_path):
    im = Image.new("RGB", (320, 180), color=(50, 50, 50))
    annotated = burn_timestamp_badge(im, timestamp_sec=14.5, frame_index=1)
    
    test_img_path = tmp_path / "overlay_test.jpg"
    annotated.save(test_img_path)
    assert test_img_path.exists()
    
    # Test apply_overlay_to_file
    apply_overlay_to_file(test_img_path, timestamp_sec=20.0, frame_index=2)
    assert test_img_path.exists()


def test_sample_sparse_frames(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    # Sample 0.1 FPS over first 20 seconds -> ~2 frames
    frames = sample_sparse_frames(
        str(FIXTURE_PATH),
        fps=0.1,
        start_sec=0.0,
        end_sec=20.0,
        width=480,
        overlay_timestamp=True,
        cache=cache
    )
    
    assert len(frames) >= 2
    for f in frames:
        assert Path(f.file_path).exists()
        assert f.width == 480
        # Batch fps extraction has a synthetic/requested output clock. It must
        # not claim a decoded source PTS without a per-frame receipt.
        assert f.requested_timestamp == pytest.approx(f.timestamp_sec)
        assert f.decoded_timestamp is None
        assert f.timestamp_delta is None


def test_sampling_cache_identity_includes_full_request(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    first = sample_sparse_frames(
        str(FIXTURE_PATH), fps=2.001, start_sec=0.0, end_sec=2.0,
        width=320, overlay_timestamp=False, cache=cache,
    )
    second = sample_sparse_frames(
        str(FIXTURE_PATH), fps=2.009, start_sec=0.0, end_sec=2.0,
        width=320, overlay_timestamp=False, cache=cache,
    )

    assert first and second
    frame_dirs = {Path(frame.file_path).parent for frame in first + second}
    assert len(frame_dirs) == 2


def test_sample_dense_window(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    # Dense sample 2.0 FPS between 14.0s and 16.0s (supernova flash)
    result = sample_dense_window(
        str(FIXTURE_PATH),
        start_sec=14.0,
        end_sec=16.0,
        fps=2.0,
        width=512,
        overlay_timestamp=True,
        cache=cache
    )
    
    assert result.start_sec == 14.0
    assert result.end_sec == 16.0
    assert len(result.frames) >= 4
    assert result.estimated_vision_tokens == len(result.frames) * 258


def test_sampling_rejects_invalid_temporal_windows(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    with pytest.raises(ExtractionBoundsError):
        sample_dense_window(str(FIXTURE_PATH), start_sec=2.0, end_sec=1.0, cache=cache)
    with pytest.raises(ExtractionBoundsError):
        sample_dense_window(str(FIXTURE_PATH), start_sec=0.0, end_sec=0.0, cache=cache)
    with pytest.raises(ExtractionBoundsError):
        sample_dense_window(str(FIXTURE_PATH), start_sec=0.0, end_sec=61.0, cache=cache)
    with pytest.raises(ExtractionBoundsError):
        sample_sparse_frames(str(FIXTURE_PATH), start_sec=20.0, end_sec=10.0, cache=cache)


def test_compose_contact_sheet(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    frames = sample_sparse_frames(
        str(FIXTURE_PATH),
        fps=0.1,
        start_sec=0.0,
        end_sec=20.0,
        width=320,
        cache=cache
    )
    
    sheet_path = tmp_path / "contact_sheet.jpg"
    sheet_result = compose_contact_sheet(
        frames=frames,
        output_path=str(sheet_path),
        title="BENCHMARK OVERVIEW",
        cols=2
    )
    
    assert Path(sheet_result.file_path).exists()
    assert sheet_result.sample_count == len(frames)
    assert sheet_result.cols == 2
    assert sheet_result.rows >= 1
    assert len(sheet_result.timestamps) == len(frames)


def test_scene_detection_and_partition(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    # Detect cuts around the optical flash (13-17s) at threshold 0.10
    cuts = detect_scenes(
        str(FIXTURE_PATH),
        threshold=0.10,
        min_scene_len_sec=0.5,
        start_sec=13.0,
        duration_sec=5.0,
        cache=cache
    )
    
    assert len(cuts) >= 1
    # Check partition into segments
    segments = partition_into_scene_segments(cuts, total_duration_sec=60.0)
    assert len(segments) == len(cuts) + 1
    assert segments[0].start_sec == 0.0
    assert segments[-1].end_sec == 60.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_soft_transition_found_in_a_cross_fade_and_not_in_a_still_clip(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    faded = tmp_path / "faded.mp4"
    still = tmp_path / "still.mp4"
    bars = "smptebars=size=320x180:rate=30"
    plain = "color=c=0x2060a0:size=320x180:rate=30"
    encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
    # 3 s of colour bars, a 1 s cross-fade starting at 2 s, then a plain colour: 5 s in all.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", "3", "-i", bars, "-f", "lavfi", "-t", "3", "-i", plain,
                    "-filter_complex", "[0:v][1:v]xfade=transition=fade:duration=1:offset=2", *encode, str(faded)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", "5", "-i", bars, *encode, str(still)], check=True)

    found = detect_soft_transitions(str(faded), hard_cuts=[], cache=cache)
    assert len(found) == 1
    assert found[0].kind == "transition"
    assert found[0].timestamp_sec == pytest.approx(2.5, abs=0.5)
    assert found[0].to_dict()["kind"] == "transition"

    # The ordinary hard-cut detector does not see the fade, and keeps reporting kind "cut".
    assert detect_scenes(str(faded), threshold=0.25, cache=cache) == []
    assert detect_soft_transitions(str(still), hard_cuts=[], cache=cache) == []


def _two_part_clip(path, first, second, join):
    """3 s of `first` then 3 s of `second`, joined by a hard cut ("concat") or a 1 s cross-fade."""
    graph = "[0:v][1:v]concat=n=2:v=1" if join == "concat" else "[0:v][1:v]xfade=transition=fade:duration=1:offset=2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", "3", "-i", first, "-f", "lavfi", "-t", "3", "-i", second,
                    "-filter_complex", graph, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)], check=True)


BARS = "smptebars=size=320x180:rate=30"
DIMMER_BARS = BARS + ",eq=brightness=-0.07"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_hard_cut_between_similar_pictures_is_labelled_cut(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    similar = tmp_path / "similar.mp4"
    faded = tmp_path / "faded.mp4"
    _two_part_clip(similar, BARS, DIMMER_BARS, "concat")
    _two_part_clip(faded, BARS, DIMMER_BARS, "fade")

    # The scene-score filter scores this cut below the default threshold.
    assert detect_scenes(str(similar), threshold=0.25, cache=cache) == []
    found = detect_soft_transitions(str(similar), hard_cuts=[], cache=cache)
    assert [cut.kind for cut in found] == ["cut"]
    # Timed at the first new frame, not at a middle, with that frame's scene score.
    assert found[0].timestamp_sec == pytest.approx(3.0, abs=0.04)
    assert 0.05 < found[0].score < 0.25

    # The same two pictures joined by a 1 s cross-fade change over many frames: never a cut.
    assert all(cut.kind == "transition" for cut in detect_soft_transitions(str(faded), hard_cuts=[], cache=cache))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_scan_scene_changes_fills_the_cache_for_both_detectors(tmp_path, monkeypatch):
    import src.media.scene as scene

    clip = tmp_path / "similar.mp4"
    _two_part_clip(clip, BARS, DIMMER_BARS, "concat")
    cache = MediaCache(base_dir=tmp_path / "cache")
    scene.scan_scene_changes(str(clip), thresholds=(0.25, 0.1), cache=cache)

    # Everything below must come from the cache: any further ffmpeg run fails the test.
    def no_decode(*args, **kwargs):
        raise AssertionError("the video was decoded again")
    monkeypatch.setattr(scene.subprocess, "run", no_decode)
    monkeypatch.setattr(scene.subprocess, "Popen", no_decode)

    assert detect_scenes(str(clip), threshold=0.25, cache=cache) == []  # an empty list is a cached answer too
    lower = detect_scenes(str(clip), threshold=0.1, cache=cache)
    assert [cut.timestamp_sec for cut in lower] == [pytest.approx(3.0, abs=0.04)]
    assert [cut.kind for cut in detect_soft_transitions(str(clip), hard_cuts=[], cache=cache)] == ["cut"]
    # The cut found at the lower threshold removes its own echo from the soft list.
    assert detect_soft_transitions(str(clip), hard_cuts=lower, cache=cache) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_scan_split_into_parts_gives_the_same_frames_as_one_decode(tmp_path, monkeypatch):
    import numpy as np
    import src.media.scene as scene

    clip = tmp_path / "similar.mp4"
    _two_part_clip(clip, BARS, DIMMER_BARS, "concat")
    monkeypatch.setattr(scene, "SCAN_SPLIT_MIN_SEC", 4)
    monkeypatch.setattr(scene, "SCAN_MIN_PART_SEC", 2)
    monkeypatch.setattr(scene.os, "cpu_count", lambda: 8)

    plan = scene._scan_plan(str(clip), 0.0, None)
    assert len(plan) == 3 and plan[0][0] == 0 and plan[-1][1] is None
    split = scene._scan_frames(str(clip), 0.0, None, 60)
    whole = scene._scan_parts(str(clip), 0.0, None, 60, [(0, None)])
    assert len(whole[0]) == 180
    for part, one in zip(split, whole):
        assert part.shape == one.shape
    assert np.allclose(split[0], whole[0], atol=1e-6)
    assert np.allclose(split[1], whole[1], atol=1e-6)
    assert np.array_equal(split[2], whole[2])
    assert np.array_equal(split[3], whole[3])


def test_short_scan_is_not_split():
    import src.media.scene as scene

    assert scene._scan_plan("unused.mp4", 0.0, 120.0) == [(0, None)]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_detect_cuts_raised_threshold_drops_the_low_scoring_cut(tmp_path):
    from src.retrieval.engine import AgenticVideoEngine

    clip = tmp_path / "similar.mp4"
    _two_part_clip(clip, BARS, DIMMER_BARS, "concat")
    engine = AgenticVideoEngine(str(clip), cache=MediaCache(base_dir=tmp_path / "cache"))

    found = engine.detect_cuts()
    assert [(cut.kind, round(cut.timestamp_sec)) for cut in found] == [("cut", 3)]
    assert found[0].score < 0.25
    assert engine.detect_cuts(threshold=0.5) == []
