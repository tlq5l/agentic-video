"""Focused tests for the bounded multi-window inspection primitive."""

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from src.core.errors import ExtractionBoundsError
from src.core.types import FrameInfo, InspectionWindowResult, VideoMetadata
from src.retrieval.engine import AgenticVideoEngine


def _engine() -> AgenticVideoEngine:
    engine = AgenticVideoEngine("synthetic-placeholder.mp4")
    engine._meta = VideoMetadata(
        source_path="synthetic-placeholder.mp4",
        duration_sec=60.0,
        width=1280,
        height=720,
        fps=30.0,
        format_name="mp4",
        size_bytes=1,
        has_audio=False,
    )
    return engine


def _stub_inspection(monkeypatch, engine: AgenticVideoEngine):
    seen = []

    def fake_inspect_window(start_sec, end_sec, fps=4.0, create_sheet=True):
        seen.append((start_sec, end_sec, fps))
        frame_count = math.ceil((end_sec - start_sec) * fps)
        frames = [
            FrameInfo(
                index=i + 1,
                timestamp_sec=start_sec + i / fps,
                timecode=f"t{i}",
                file_path=f"frame-{len(seen)}-{i}.jpg",
                width=320,
                height=180,
            )
            for i in range(frame_count)
        ]
        return InspectionWindowResult(
            source_path=engine.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps,
            frames=frames,
            contact_sheet_path=f"sheet-{len(seen)}.jpg",
            estimated_vision_tokens=frame_count * 258,
        )

    monkeypatch.setattr(engine, "inspect_window", fake_inspect_window)
    return seen


def test_inspect_windows_sorts_and_merges_overlaps(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    result = engine.inspect_windows(
        windows=[
            {"start_sec": 10.0, "end_sec": 12.0},
            {"start_sec": 2.0, "end_sec": 4.0},
            {"start_sec": 3.5, "end_sec": 5.0},
        ],
        fps=2.0,
    )

    assert seen == [(2.0, 5.0, 2.0), (10.0, 12.0, 2.0)]
    assert [(item["start_sec"], item["end_sec"]) for item in result["windows"]] == [(2.0, 5.0), (10.0, 12.0)]
    assert result["windows"][0]["input_indices"] == [1, 2]
    assert result["windows"][1]["input_indices"] == [0]
    assert result["total_frames"] == 10


def test_inspect_windows_context_expands_and_clamps(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    engine.inspect_windows(
        windows=[
            {"start_sec": 0.5, "end_sec": 1.0},
            {"start_sec": 59.0, "end_sec": 59.5},
        ],
        fps=1.0,
        context_sec=2.0,
        merge_overlaps=False,
    )

    assert seen == [(0.0, 3.0, 1.0), (57.0, 60.0, 1.0)]


def test_inspect_windows_merge_false_preserves_overlapping_occurrences(monkeypatch):
    engine = _engine()
    seen = _stub_inspection(monkeypatch, engine)

    result = engine.inspect_windows(
        windows=[{"start_sec": 4.0, "end_sec": 7.0}, {"start_sec": 5.0, "end_sec": 8.0}],
        fps=1.0,
        merge_overlaps=False,
    )

    assert seen == [(4.0, 7.0, 1.0), (5.0, 8.0, 1.0)]
    assert result["total"] == 2


def test_inspect_windows_rejects_budget_before_extraction(monkeypatch):
    engine = _engine()
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("extraction should not run after budget rejection")

    monkeypatch.setattr(engine, "inspect_window", should_not_run)
    with pytest.raises(ExtractionBoundsError, match="max_total_frames"):
        engine.inspect_windows(
            windows=[{"start_sec": 0.0, "end_sec": 10.0}],
            fps=4.0,
            max_total_frames=39,
        )
    assert called is False


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [{"start_sec": 3.0, "end_sec": 3.0}],
        [{"start_sec": 4.0, "end_sec": 3.0}],
        [{"start_sec": -1.0, "end_sec": 3.0}],
        [{"start_sec": float("nan"), "end_sec": 3.0}],
        [{"start_sec": 0.0, "end_sec": float("inf")}],
        [{"start_sec": 59.0, "end_sec": 61.0}],
        [{"start_sec": 1.0}],
    ],
)
def test_inspect_windows_rejects_invalid_windows(windows):
    with pytest.raises(ExtractionBoundsError):
        _engine().inspect_windows(windows=windows)


def test_inspect_windows_rejects_too_many_input_windows():
    windows = [{"start_sec": float(i), "end_sec": float(i) + 0.5} for i in range(17)]
    with pytest.raises(ExtractionBoundsError, match="max_windows"):
        _engine().inspect_windows(windows=windows, max_windows=16)


def test_executor_dispatch_for_inspect_windows(monkeypatch):
    import src.executor.mcp_server as mcp_server

    class FakeEngine:
        def __init__(self, source):
            self.source = source

        def inspect_windows(self, **kwargs):
            assert kwargs["windows"] == [{"start_sec": 1.0, "end_sec": 2.0}]
            assert kwargs["fps"] == 2.0
            return {
                "windows": [
                    {
                        "index": 1,
                        "input_indices": [0],
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "fps": 2.0,
                        "frames": [],
                        "contact_sheet_path": None,
                        "estimated_vision_tokens": 0,
                    }
                ],
                "total": 1,
                "total_frames": 0,
                "estimated_vision_tokens": 0,
            }

    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    outcome = mcp_server.handle_call(
        "inspect_windows",
        {
            "source": "video.mp4",
            "windows": [{"start_sec": 1.0, "end_sec": 2.0}],
            "fps": 2.0,
        },
    )
    assert outcome["ok"] is True
    assert outcome["data"]["windows"][0]["input_indices"] == [0]


def _write_testsrc_clip(path, duration=2.0):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=1280x720:rate=25:duration={duration}",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)],
        check=True,
    )
    return str(path)


def test_frame_at_native_frame_crop_and_scale_end_to_end(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _write_testsrc_clip(tmp_path / "clip.mp4")

    full = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 1.0})
    assert full["ok"] is True
    data = full["data"]
    assert (data["source_width"], data["source_height"]) == (1280, 720)
    assert (data["width"], data["height"]) == (1280, 720)
    assert Path(data["file_path"]).suffix == ".png" and Path(data["file_path"]).exists()
    assert abs(data["frame_time_sec"] - 1.0) <= 1 / 25
    assert data["tool_file"]["mimeType"] == "image/png"

    crop = {"x": 100, "y": 50, "width": 200, "height": 100}
    cropped = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 1.0, "crop": crop, "scale": 2})
    assert cropped["ok"] is True
    cdata = cropped["data"]
    assert (cdata["width"], cdata["height"]) == (400, 200)
    assert cdata["crop"] == crop and cdata["scale"] == 2
    with Image.open(cdata["file_path"]) as im:
        assert im.format == "PNG" and im.size == (400, 200)
    assert abs(cdata["frame_time_sec"] - 1.0) <= 1 / 25

    near_end = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 1.99})
    assert near_end["ok"] is True and abs(near_end["data"]["frame_time_sec"] - 1.99) <= 1 / 25

    out_of_frame = {"x": 1200, "y": 0, "width": 200, "height": 100}
    line = mcp_server.tool_call_response(1, "frame_at", {"source": clip, "time_sec": 1.0, "crop": out_of_frame})
    result = json.loads(line)["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "crop" in text and "1280" in text
    assert "This is a bug in the tool" not in text


def _write_variable_rate_clip(path):
    """1280x720 clip with frames at 0, 0.04, ..., 0.16 s, then the next frame at 1.0 s."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=1280x720:rate=25:duration=2",
         "-vf", "setpts='if(lt(N,5),N,20+N)'",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
         "-fps_mode", "passthrough", str(path)],
        check=True,
    )
    return str(path)


def test_frame_at_picks_frame_covering_request_on_variable_rate_video(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _write_variable_rate_clip(tmp_path / "vfr.mp4")

    outcome = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 0.2})
    assert outcome["ok"] is True
    assert outcome["data"]["frame_time_sec"] == pytest.approx(0.16, abs=0.001)


def _missing_display_rotation_reason():
    probe = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-display_rotation:v:0", "90",
         "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.1", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0 or "Unrecognized option" not in probe.stderr:
        return None
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False).stdout.splitlines()[0]
    return f"{version.split(' Copyright')[0]} has no -display_rotation input option (added in FFmpeg 6.0)"


def test_frame_at_uses_displayed_size_for_rotated_video(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    reason = _missing_display_rotation_reason()
    if reason:
        pytest.skip(reason)
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = _write_testsrc_clip(tmp_path / "clip.mp4")
    rotated = str(tmp_path / "rotated.mp4")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-display_rotation:v:0", "90",
         "-i", source, "-c", "copy", rotated],
        check=True,
    )

    full = mcp_server.handle_call("frame_at", {"source": rotated, "time_sec": 1.0})
    assert full["ok"] is True
    data = full["data"]
    assert (data["source_width"], data["source_height"]) == (720, 1280)
    with Image.open(data["file_path"]) as im:
        assert im.size == (720, 1280)

    crop = {"x": 600, "y": 100, "width": 120, "height": 200}
    cropped = mcp_server.handle_call("frame_at", {"source": rotated, "time_sec": 1.0, "crop": crop, "scale": 2})
    assert cropped["ok"] is True
    with Image.open(cropped["data"]["file_path"]) as im:
        assert im.size == (240, 400)

    too_wide = {"x": 700, "y": 0, "width": 100, "height": 100}
    line = mcp_server.tool_call_response(1, "frame_at", {"source": rotated, "time_sec": 1.0, "crop": too_wide})
    result = json.loads(line)["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "crop.x + crop.width = 800 exceeds the frame width 720" in text


BURN_PTS_FILTER = "drawtext=fontsize=28:fontcolor=white:box=1:boxcolor=black@0.7:x=8:y=8:text='%{pts\\:flt}'"


def _render_burned_clip(path, duration=2.0, retime=None):
    """320x240 25 fps test pattern with each frame's presentation time (s) burned in.
    `retime` is an optional setpts expression that gives the frames irregular times."""
    chain = ",".join(part for part in (retime, BURN_PTS_FILTER) if part)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate=25:duration={duration}",
         "-vf", chain, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
         "-fps_mode", "passthrough", str(path)],
        check=True,
    )
    return str(path)


def _frames_by_pts(path):
    """Every frame of the clip as RGB bytes, keyed by its presentation time (s), from one full decode."""
    pts = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    size = len(raw) // len(pts)
    return {round(float(t), 6): raw[i * size:(i + 1) * size] for i, t in enumerate(pts)}


def _png_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB").tobytes()


def test_frame_at_never_returns_a_frame_after_the_request(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _render_burned_clip(tmp_path / "vfr.mp4", retime="setpts='if(lt(N,5),N,20+N)'")
    frames = _frames_by_pts(clip)
    assert 0.16 in frames and 1.0 in frames

    # 0.9994 is before the 1.0 s frame, so the 0.16 s frame is still on screen.
    # 0.16 and 1.0 are exact presentation times, which count as on screen.
    for time_sec, shown in [(0.9994, 0.16), (0.16, 0.16), (1.0, 1.0), (0.5, 0.16)]:
        outcome = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": time_sec})
        assert outcome["ok"] is True
        assert outcome["data"]["frame_time_sec"] == pytest.approx(shown, abs=1e-6), time_sec
        assert _png_rgb(outcome["data"]["file_path"]) == frames[shown], time_sec


def _write_lossless_clip(path, rate, frames):
    """320x240 test pattern at `rate` frames per second, lossless H.264 (qp 0)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate={rate}", "-frames:v", str(frames),
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
         "-fps_mode", "passthrough", str(path)],
        check=True,
    )
    return str(path)


def _timeline(path):
    """The first video stream's time base, the container start, the frame pts (ticks, in
    presentation order) and every frame's RGB bytes in that order, from one full decode."""
    def probe(entries):
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout

    time_base = Fraction(probe("stream=time_base").strip())
    origin = Fraction(probe("format=start_time").strip())
    pts = sorted(int(line) for line in probe("packet=pts").split())
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    size = len(raw) // len(pts)
    return time_base, origin, pts, [raw[i * size:(i + 1) * size] for i in range(len(pts))]


def test_frame_at_picks_the_frame_on_screen_at_exact_frame_times(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    for name, rate in (("fast.mp4", "2000"), ("ntsc.mp4", "30000/1001")):
        clip = _write_lossless_clip(tmp_path / name, rate, 30)
        time_base, origin, pts, pixels = _timeline(clip)
        assert origin == 0 and len(set(pts)) == len(pts) == 30

        def ask(ticks):
            # The request is a JSON number: the float nearest to the exact time.
            return mcp_server.handle_call("frame_at", {"source": clip, "time_sec": float(ticks * time_base - origin)})

        def assert_shows(ticks, index):
            outcome = ask(ticks)
            assert outcome["ok"] is True
            assert _png_rgb(outcome["data"]["file_path"]) == pixels[index], (name, float(ticks * time_base))
            assert outcome["data"]["frame_time_sec"] == pytest.approx(float(pts[index] * time_base), abs=1e-6)

        for index in (0, 1, 2, 17, 28):
            assert_shows(pts[index], index)
            assert_shows(Fraction(pts[index]) + Fraction(pts[index + 1] - pts[index], 2), index)
            if index:
                assert_shows(pts[index] - 1, index - 1)
        with pytest.raises(ValueError, match="below the minimum 0"):
            ask(pts[0] - 1)

        end = pts[-1] + (pts[-1] - pts[-2])
        assert_shows(end, len(pts) - 1)
        with pytest.raises(ExtractionBoundsError, match=r"time_sec \S+ is outside the valid range"):
            ask(end + 1)


def test_frame_at_result_does_not_depend_on_earlier_calls(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    clip = _render_burned_clip(tmp_path / "constant.mp4")
    frames = _frames_by_pts(clip)

    def call(time_sec, cache_name):
        monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / cache_name)
        outcome = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": time_sec})
        assert outcome["ok"] is True
        return outcome["data"]

    # 0.0399 is on the 0.0 s frame and 0.0401 on the 0.04 s frame; both round to 0.040 s.
    cold = call(0.0399, "cold")
    assert call(0.0401, "warm")["frame_time_sec"] == pytest.approx(0.04, abs=1e-6)
    warm = call(0.0399, "warm")
    assert cold["frame_time_sec"] == pytest.approx(0.0, abs=1e-6)
    assert warm["frame_time_sec"] == pytest.approx(0.0, abs=1e-6)
    assert _png_rgb(warm["file_path"]) == _png_rgb(cold["file_path"]) == frames[0.0]


def test_frame_at_uses_the_stream_start_as_time_zero(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    base = _render_burned_clip(tmp_path / "base.mp4")
    frames = _frames_by_pts(base)
    late = str(tmp_path / "late.ts")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-itsoffset", "0.48", "-i", base, "-c", "copy",
         "-muxdelay", "0", "-muxpreload", "0", "-f", "mpegts", late],
        check=True,
    )
    start = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", late],
        capture_output=True, text=True, check=True,
    ).stdout)
    assert start == pytest.approx(0.48, abs=0.001)  # fixture check: the stream really starts at 0.48 s

    # The other tools (inspect_window, probe_video) count time from the stream start, so
    # time_sec 1.0 is the frame whose time is 1.0 s in that clock, not the raw pts 1.0 s frame.
    outcome = mcp_server.handle_call("frame_at", {"source": late, "time_sec": 1.0})
    assert outcome["ok"] is True
    assert outcome["data"]["frame_time_sec"] == pytest.approx(1.0, abs=1e-6)
    assert _png_rgb(outcome["data"]["file_path"]) == frames[1.0]

    outcome = mcp_server.handle_call("frame_at", {"source": late, "time_sec": 0.9994})
    assert outcome["data"]["frame_time_sec"] == pytest.approx(0.96, abs=1e-6)
    assert _png_rgb(outcome["data"]["file_path"]) == frames[0.96]


def test_frame_at_bounds_time_sec_to_the_media_duration(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _render_burned_clip(tmp_path / "two.mp4", duration=2.0)
    frames = _frames_by_pts(clip)

    with pytest.raises(ExtractionBoundsError, match=r"time_sec 2\.0005 is outside the valid range \[0, 2\.000\] s"):
        mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 2.0005})

    # time_sec equal to the duration may return the last frame (time 1.96 s).
    outcome = mcp_server.handle_call("frame_at", {"source": clip, "time_sec": 2.0})
    assert outcome["data"]["frame_time_sec"] == pytest.approx(1.96, abs=1e-6)
    assert _png_rgb(outcome["data"]["file_path"]) == frames[1.96]


def test_frame_at_names_the_source_when_it_has_no_video(tmp_path, monkeypatch):
    import src.executor.mcp_server as mcp_server
    from src.core.config import Config

    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    tone = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "aac", str(tone)],
        check=True,
    )
    line = mcp_server.tool_call_response(1, "frame_at", {"source": str(tone), "time_sec": 0.5})
    result = json.loads(line)["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "frame_at needs a file with a video stream" in text
    assert "source" in text and tone.name in text
    assert "This is a bug in the tool" not in text
