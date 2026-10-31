"""Distinct-occurrence counting on synthetic clips: colour look-alikes, exact text, crossings, zero."""
import dataclasses
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.core.cache import MediaCache
from src.retrieval import occurrence, ocr
from src.retrieval.engine import AgenticVideoEngine

W, H, FPS = 320, 180, 10
YELLOW, RED, BLUE, GREEN = (255, 220, 0), (230, 30, 30), (40, 120, 255), (30, 200, 60)
FONT = "/System/Library/Fonts/Helvetica.ttc"

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _has_vision():
    try:
        ocr.ensure_available()
        return True
    except RuntimeError:
        return False


needs_vision = pytest.mark.skipif(not _has_vision() or not Path(FONT).exists(), reason="Apple Vision OCR is required")


def shape(draw, kind, cx, cy, r, color, outline=True):
    kw = {"outline": color, "width": 3} if outline else {"fill": color}
    if kind == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], **kw)
    elif kind == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], **kw)
    elif kind == "diamond":
        draw.polygon([(cx, cy - r * 1.2), (cx + r * 1.2, cy), (cx, cy + r * 1.2), (cx - r * 1.2, cy)], **kw)
    elif kind == "triangle":
        draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], **kw)


def live(t, *windows):
    return any(a <= t < b for a, b in windows)


def make_clip(path, seconds, paint):
    """Render frames with Pillow and encode them like an ordinary lossy video."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-preset", "ultrafast", "-threads", "1", str(path)], stdin=subprocess.PIPE)
    for i in range(int(seconds * FPS)):
        image = Image.new("RGB", (W, H), (30, 30, 40))
        paint(ImageDraw.Draw(image), i / FPS)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    assert proc.wait(timeout=60) == 0
    return path


def count(video, tmp_path, **args):
    engine = AgenticVideoEngine(str(video), cache=MediaCache(tmp_path / "cache"))
    return occurrence.count_video(engine, **args)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    root = tmp_path_factory.mktemp("occurrence-clips")
    yellow = [(1.0, 2.2), (2.8, 4.0), (7.0, 8.5)]  # the second starts 0.6 s after the first ends
    red, square, blue = [(0.5, 2.0), (5.0, 6.0)], [(4.2, 5.5), (8.6, 9.5)], [(6.2, 7.0)]

    def shapes(d, t):
        if live(t, *yellow):
            shape(d, "circle", 100 + 20 * math.sin(t * 3), 90, 25, YELLOW)  # moves, still one appearance
        if live(t, *red):
            shape(d, "circle", 225, 90, 25, RED)
        if live(t, *square):
            shape(d, "square", 160, 60, 25, YELLOW)
        if live(t, *blue):
            shape(d, "circle", 75, 130, 20, BLUE)

    font = ImageFont.truetype(FONT, 40) if Path(FONT).exists() else None
    words = [("RELEASE", (1.0, 2.0)), ("RELEASED", (2.6, 3.6)), ("RELEASE", (4.2, 5.2)), ("RELEASES", (5.8, 6.8)),
             ("release", (7.4, 8.4)), ("PRE-RELEASE", (9.0, 10.0)), ("RELEAS", (10.6, 11.6)), ("RELEASE", (12.2, 13.2))]

    def text(d, t):
        for word, window in words:
            if live(t, window):
                d.text((30, 70), word, font=font, fill=(240, 240, 240))

    starts = [0.5, 2.5, 4.5, 7.5]

    def crossing(t, start, seconds=1.5):
        return -20 + (W + 40) * (t - start) / seconds if start <= t < start + seconds else None

    def markers(d, t):
        for start in starts:
            if (x := crossing(t, start)) is not None:
                shape(d, "diamond", x, 50, 14, GREEN)
        for start in (1.5, 6.0):
            if (x := crossing(t, start)) is not None:
                shape(d, "diamond", x, 100, 14, RED)
        for start in (3.5, 9.0):
            if (x := crossing(t, start)) is not None:
                shape(d, "circle", x, 150, 14, GREEN)

    def wrapping(d, t):  # never leaves the frame: three sweeps that jump back to the left edge
        shape(d, "diamond", ((t / 2.0) % 1.0) * W, 90, 14, GREEN)

    def distractors(d, t):
        if live(t, (1, 2.5)):
            shape(d, "circle", 100, 90, 25, RED)
        if live(t, (3, 4.5)):
            shape(d, "square", 160, 90, 25, YELLOW)
        if live(t, (5, 6.5)):
            shape(d, "diamond", 225, 90, 25, GREEN)
        if live(t, (7, 8)) and font:
            d.text((30, 70), "RELEASED", font=font, fill=(240, 240, 240))

    return {"shapes": make_clip(root / "shapes.mp4", 10, shapes), "text": make_clip(root / "text.mp4", 14, text),
            "markers": make_clip(root / "markers.mp4", 11, markers), "wrap": make_clip(root / "wrap.mp4", 6, wrapping),
            "zero": make_clip(root / "zero.mp4", 9, distractors)}


@needs_ffmpeg
def test_same_symbol_in_other_colors_and_shapes_is_ignored(clips, tmp_path):
    result = count(clips["shapes"], tmp_path, description="yellow outlined circle")
    assert result["status"] == "complete" and result["count"] == 3
    spans = result["spans"]
    assert [s["index"] for s in spans] == [1, 2, 3]
    for span, (start, end) in zip(spans, [(1.0, 2.2), (2.8, 4.0), (7.0, 8.5)]):
        assert abs(span["start_sec"] - start) <= 0.25 and abs(span["end_sec"] - end) <= 0.25
        assert span["start_sec"] <= span["representative_sec"] < span["end_sec"]
        assert span["frame_badge"] == f"#{span['index']}"
    assert Path(result["contact_sheet"]["file_path"]).exists()
    assert count(clips["shapes"], tmp_path, description="red circle")["count"] == 2
    assert count(clips["shapes"], tmp_path, description="yellow square")["count"] == 2
    assert count(clips["shapes"], tmp_path, description="blue circle")["count"] == 1


@needs_ffmpeg
@needs_vision
def test_exact_text_rejects_near_misses(clips, tmp_path):
    exact = count(clips["text"], tmp_path, text="RELEASE", case_sensitive=True)
    assert exact["count"] == 3
    assert count(clips["text"], tmp_path, text="RELEASE")["count"] == 4  # also matches lowercase
    assert count(clips["text"], tmp_path, text="RELEASED")["count"] == 1
    assert count(clips["text"], tmp_path, text="RELEASE", start_sec=4.0, end_sec=6.0)["count"] == 1


@needs_ffmpeg
def test_marker_crossing_the_screen_counts_each_pass(clips, tmp_path):
    result = count(clips["markers"], tmp_path, description="green outlined diamond marker")
    assert result["count"] == 4
    assert {s["travel"] for s in result["spans"]} == {"left_to_right"}
    assert count(clips["markers"], tmp_path, description="red diamond")["count"] == 2
    assert count(clips["markers"], tmp_path, description="green circle")["count"] == 2


@needs_ffmpeg
def test_marker_that_wraps_around_without_leaving_counts_each_sweep(clips, tmp_path):
    assert count(clips["wrap"], tmp_path, description="green diamond")["count"] == 3


@needs_ffmpeg
def test_zero_occurrences_is_an_explicit_zero(clips, tmp_path):
    result = count(clips["zero"], tmp_path, description="yellow outlined circle")
    assert result["status"] == "complete" and result["count"] == 0 and result["spans"] == []
    assert result["contact_sheet"] is None and "Nothing matched" in result["note"] and "Count 0 only" in result["note"]
    if _has_vision():
        assert count(clips["zero"], tmp_path, text="RELEASE")["count"] == 0


@needs_ffmpeg
def test_output_is_bounded_and_reports_omitted_spans(clips, tmp_path):
    result = count(clips["markers"], tmp_path, description="green diamond", max_spans=2)
    assert result["count"] == 4 and result["spans_returned"] == 2 and result["spans_omitted"] == 2
    assert len(result["spans"]) == 2 and "omitted" in result["note"]
    assert result["contact_sheet"]["sample_count"] == 2


@needs_ffmpeg
def test_sheet_has_one_labeled_tile_per_span_in_order(clips, tmp_path, monkeypatch):
    result = count(clips["markers"], tmp_path, description="green diamond")
    spans, sheet = result["spans"], result["contact_sheet"]
    assert result["count"] == 4 and sheet["sample_count"] == len(spans) == 4
    assert sheet["labels"] == [f"#{s['index']}{'*' if s['weak'] else ''} {s['representative_sec']:.1f}s" for s in spans]
    with Image.open(sheet["file_path"]) as image:
        assert image.width > sheet["cols"] * occurrence.TILE_SIZE[0] and image.height > sheet["rows"] * occurrence.TILE_SIZE[1]
    assert "contact_sheet" in result["note"]
    monkeypatch.setattr(occurrence, "SHEET_TILES", 3)
    capped = count(clips["markers"], tmp_path, description="green diamond", max_spans=4)
    assert capped["count"] == 4 and len(capped["spans"]) == 4
    assert capped["contact_sheet"]["sample_count"] == 3 and len(capped["contact_sheet"]["labels"]) == 3
    assert capped["contact_sheet"]["labels"] == sheet["labels"][:3] and "only the first 3 of 4" in capped["note"]


@needs_ffmpeg
def test_brief_span_is_flagged_weak_but_still_counted(tmp_path):
    def paint(d, t):
        if live(t, (1.0, 2.5), (6.0, 7.5)):
            shape(d, "circle", 100, 90, 25, RED)
        if live(t, (4.0, 4.1)):  # one detection at 10 fps
            shape(d, "circle", 220, 90, 25, RED)
    result = count(make_clip(tmp_path / "flash.mp4", 9, paint), tmp_path, description="red circle", fps=10)
    assert result["count"] == 3 and [s["weak"] for s in result["spans"]] == [False, True, False]
    assert [label[:3] for label in result["contact_sheet"]["labels"]] == ["#1 ", "#2*", "#3 "]


@needs_ffmpeg
def test_fps_is_lowered_to_fit_the_frame_budget(clips, tmp_path, monkeypatch):
    monkeypatch.setattr(occurrence, "MAX_FRAMES", 20)
    result = count(clips["shapes"], tmp_path, description="yellow circle")
    assert result["fps_used"] == 2.0 and result["frames_sampled"] == 20 and "fps lowered" in result["note"]
    monkeypatch.setattr(occurrence, "MAX_FRAMES", 5)
    with pytest.raises(ValueError, match="windows of at most"):
        count(clips["shapes"], tmp_path, description="yellow circle")


@needs_ffmpeg
def test_async_call_polls_then_returns_a_cached_result(clips, tmp_path):
    engine = AgenticVideoEngine(str(clips["shapes"]), cache=MediaCache(tmp_path / "cache"))
    args = {"description": "yellow outlined circle"}
    result = engine.count_occurrences(**args)
    deadline = time.time() + 60
    while result["status"] == "in_progress" and time.time() < deadline:
        assert result["retry_after_seconds"] > 0
        time.sleep(0.5)
        result = engine.count_occurrences(**args)
    assert result["status"] == "complete" and result["count"] == 3
    assert engine.count_occurrences(**args)["count"] == 3  # served from the on-disk result


@needs_ffmpeg
@pytest.mark.model
def test_clip_described_symbol_uses_contrast_prompts(clips, tmp_path):
    from src.retrieval import visual
    try:
        visual.get_encoder()
    except RuntimeError:
        pytest.skip("CLIP weights are not cached locally")
    stars = tmp_path / "stars.mp4"

    def paint(d, t):
        for window, color in (((1, 2), YELLOW), ((3, 4), YELLOW), ((5, 6), RED)):
            if live(t, window):
                points = [(160 + (40 if i % 2 == 0 else 17) * math.cos(-math.pi / 2 + i * math.pi / 5),
                           90 + (40 if i % 2 == 0 else 17) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
                d.line(points + [points[0]], fill=color, width=3)
        if live(t, (7, 8)):
            shape(d, "circle", 160, 90, 30, YELLOW)
    make_clip(stars, 9, paint)
    assert count(stars, tmp_path, description="yellow star")["count"] == 2
    assert count(stars, tmp_path, description="red star")["count"] == 1


def test_shape_geometry_distinguishes_common_outlines():
    def mask(kind, size, outline=True, angle=0):
        image = Image.new("L", (size * 4, size * 4), 0)
        draw = ImageDraw.Draw(image)
        c, r = size * 2, size
        if kind == "circle":
            draw.ellipse([c - r, c - r, c + r, c + r], outline=255, width=3) if outline else draw.ellipse([c - r, c - r, c + r, c + r], fill=255)
        elif kind == "square":
            draw.rectangle([c - r, c - r, c + r, c + r], outline=255, width=3)
        elif kind == "diamond":
            draw.polygon([(c, c - r * 1.2), (c + r * 1.2, c), (c, c + r * 1.2), (c - r * 1.2, c)], outline=255, width=3)
        elif kind == "triangle":
            draw.polygon([(c, c - r), (c + r, c + r), (c - r, c + r)], outline=255, width=3)
        elif kind == "star":
            pts = [(c + (r if i % 2 == 0 else r * .42) * math.cos(-math.pi / 2 + i * math.pi / 5),
                    c + (r if i % 2 == 0 else r * .42) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
            draw.line(pts + [pts[0]], fill=255, width=3)
        return np.asarray(image) > 0

    kinds = ["circle", "square", "diamond", "triangle"]
    for size in (12, 25, 60):
        for kind in kinds:
            features = occurrence.shape_features(mask(kind, size))
            scores = {other: occurrence.shape_score(other, features) for other in kinds}
            assert scores[kind] > 0, (kind, size, features)
            assert all(scores[other] == 0 for other in kinds if other != kind), (kind, size, scores)
    star = occurrence.shape_features(mask("star", 40))
    assert all(occurrence.shape_score(kind, star) == 0 for kind in kinds)
    solid = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(solid).polygon([(80 + (40 if i % 2 == 0 else 17) * math.cos(-math.pi / 2 + i * math.pi / 5),
                                    80 + (40 if i % 2 == 0 else 17) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)], fill=255)
    solid_star = occurrence.shape_features(np.asarray(solid) > 0)
    assert all(occurrence.shape_score(kind, solid_star) == 0 for kind in kinds)
    filled, ring = occurrence.shape_features(mask("circle", 25, outline=False)), occurrence.shape_features(mask("circle", 25))
    assert occurrence.style_ok("filled", filled) and not occurrence.style_ok("outlined", filled)
    assert occurrence.style_ok("outlined", ring) and not occurrence.style_ok("filled", ring)


def test_description_parsing_splits_colour_shape_and_style():
    spec = occurrence.parse_description("The yellow outlined circle")
    assert (spec["color"], spec["shape"], spec["style"]) == ("yellow", "circle", "outlined")
    assert occurrence.parse_description("green diamond marker on a red background")["color"] == "green"
    assert occurrence.parse_description("a star")["color"] is None
    assert occurrence.parse_description("red star")["prompt"] == "star"
    assert occurrence.parse_description("filled blue square")["style"] == "filled"


def test_color_mask_separates_hues_and_ignores_background():
    image = Image.new("RGB", (60, 20), (30, 30, 40))
    ImageDraw.Draw(image).rectangle([0, 0, 19, 19], fill=YELLOW)
    ImageDraw.Draw(image).rectangle([20, 0, 39, 19], fill=RED)
    ImageDraw.Draw(image).rectangle([40, 0, 59, 19], fill=GREEN)
    for color, column in (("yellow", 10), ("red", 30), ("green", 50)):
        mask = occurrence.color_mask(image, color)
        assert mask[10, column] and mask.sum() == 400


def test_color_mask_ignores_dim_noise_and_keeps_orange_out_of_yellow():
    image = Image.new("RGB", (60, 20), (30, 40, 60))  # a dark bluish background with grain, like a noisy night scene
    ImageDraw.Draw(image).rectangle([0, 0, 19, 19], fill=(40, 48, 66))
    ImageDraw.Draw(image).rectangle([20, 0, 39, 19], fill=(255, 165, 0))
    ImageDraw.Draw(image).rectangle([40, 0, 59, 19], fill=BLUE)
    assert occurrence.color_mask(image, "blue")[10, 10] == 0 and occurrence.color_mask(image, "blue")[10, 50]
    assert occurrence.color_mask(image, "orange")[10, 30] and not occurrence.color_mask(image, "yellow")[10, 30]


def test_exact_phrase_matching_uses_whole_words():
    lines = lambda *texts: [{"text": t, "confidence": .9, "x": 0, "y": i * .1, "w": .2, "h": .05} for i, t in enumerate(texts)]
    pattern = ocr.phrase_pattern("RELEASE")
    for text in ("RELEASE", "release", "Now: RELEASE!", "RELEASE NOTES"):
        assert ocr.find_phrase(lines(text), pattern), text
    for text in ("RELEASED", "RELEASES", "PRE-RELEASE", "RELEAS", "UNRELEASE"):
        assert ocr.find_phrase(lines(text), pattern) is None, text
    assert ocr.find_phrase(lines("release"), ocr.phrase_pattern("RELEASE", True), True) is None
    assert ocr.find_phrase(lines("HELLO", "WORLD"), ocr.phrase_pattern("hello world"))


def test_span_builder_merges_flicker_and_splits_on_gaps_and_jumps():
    hit = lambda x=.5: {"score": 1.0, "box": (x, .5, .1, .1), "boxes": [(x, .5, .1, .1)]}
    builder = occurrence.SpanBuilder(0.2, 0.3, keep=5)
    frame = Image.new("RGB", (32, 18))
    for t, present in enumerate([1, 1, 0, 1, 1, 0, 0, 1, 1, 1]):  # one dropped sample merges, two split
        builder.add(t * 0.2, hit() if present else None, frame)
    spans = builder.finish()
    assert [(round(s["start"], 2), round(s["end"], 2)) for s in spans] == [(0.0, 1.0), (1.4, 2.0)]
    builder = occurrence.SpanBuilder(0.2, 0.3, keep=5)
    for t, x in enumerate([.1, .3, .5, .7, .1, .3]):  # wraps from right to left without a gap
        builder.add(t * 0.2, hit(x), frame)
    assert len(builder.finish()) == 2
    builder = occurrence.SpanBuilder(0.2, 0.3, keep=5)
    both = lambda x: {"score": 1.0, "box": (x, .5, .1, .1), "boxes": [(.1, .5, .1, .1), (.8, .5, .1, .1)]}
    for t, x in enumerate([.1, .8, .1, .8]):  # two look-alikes on screen at once are one appearance
        builder.add(t * 0.2, both(x), frame)
    assert len(builder.finish()) == 1


@needs_ffmpeg
def test_errors_tell_the_agent_what_to_change(clips, tmp_path):
    engine = AgenticVideoEngine(str(clips["shapes"]), cache=MediaCache(tmp_path / "cache"))
    with pytest.raises(ValueError, match="description.*and/or.*text"):
        engine.count_occurrences()
    with pytest.raises(ValueError, match="fps must be"):
        engine.count_occurrences(description="circle", fps=30)
    with pytest.raises(ValueError, match="max_spans"):
        engine.count_occurrences(description="circle", max_spans=0)
    with pytest.raises(ValueError, match="end_sec"):
        engine.count_occurrences(description="circle", start_sec=5, end_sec=2)
    with pytest.raises(ValueError, match="within the video"):
        count(clips["shapes"], tmp_path, description="circle", end_sec=999)
    long_engine = AgenticVideoEngine(str(clips["shapes"]), cache=MediaCache(tmp_path / "cache"))
    long_engine._meta = dataclasses.replace(long_engine.metadata, duration_sec=6000.0)
    with pytest.raises(ValueError, match="0-2400, 2400-4800, 4800-6000"):
        occurrence.count_video(long_engine, text="PRESIDENT")
    with pytest.raises(FileNotFoundError, match="existing local video"):
        AgenticVideoEngine(str(tmp_path / "missing.mp4"), cache=MediaCache(tmp_path / "c")).count_occurrences(description="circle")


def test_missing_ocr_fails_with_install_instructions(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "Vision", None)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    engine = AgenticVideoEngine(str(source), cache=MediaCache(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match=r"requirements-visual\.txt.*omit `text`"):
        engine.count_occurrences(text="RELEASE")


def test_schema_and_mcp_wiring(tmp_path):
    from src.executor import mcp_server
    from src.executor.mcp_server import TOOLS_MANIFEST
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "src/adapters/mcp_schemas.json").read_text())["tools"]["count_occurrences"]
    assert "count_occurrences" in {item["name"] for item in TOOLS_MANIFEST}
    for other in ("count_repetitions", "search_visual"):
        assert other in schema["description"]  # tells the agent when not to use it
    assert schema["inputSchema"]["additionalProperties"] is False and schema["inputSchema"]["required"] == ["source"]
    for path in ("src/adapters/mcp_definitions.ts", "src/executor/types.d.ts"):
        assert "count_occurrences" in (root / path).read_text()

    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"jpeg")

    class FakeEngine:
        def __init__(self, source):
            assert source == "clip.mp4"

        def count_occurrences(self, **kwargs):
            assert kwargs["text"] == "RELEASE" and kwargs["max_spans"] == 20 and kwargs["fps"] == 5.0
            return {"status": "complete", "count": 1, "contact_sheet": {"file_path": str(sheet)}}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    try:
        result = mcp_server.handle_call("count_occurrences", {"source": "clip.mp4", "text": "RELEASE"})
    finally:
        monkeypatch.undo()
    assert result["data"]["contact_sheet"]["tool_file"]["_tag"] == "ToolFile"


@needs_ffmpeg
def test_large_min_gap_reports_the_appearances_it_joined(clips, tmp_path):
    joined = count(clips["shapes"], tmp_path, description="yellow outlined circle", min_gap_sec=1.0)
    assert joined["count"] == 2 and joined["gaps_merged"] == 1 and joined["count_without_merges"] == 3
    assert "min_gap_sec=1" in joined["note"] and "count is 3" in joined["note"]
    plain = count(clips["shapes"], tmp_path, description="yellow outlined circle")
    assert plain["count"] == 3 and "gaps_merged" not in plain and "count_without_merges" not in plain


@needs_ffmpeg
def test_rerun_with_other_gap_or_span_limit_reuses_judgments_and_matches_a_fresh_run(clips, tmp_path, monkeypatch):
    video, args = clips["shapes"], {"description": "yellow outlined circle"}
    count(video, tmp_path / "warm", **args)
    monkeypatch.setattr(occurrence, "_judge_frames", lambda *a, **k: pytest.fail("a rerun must not judge frames again"))
    rerun = count(video, tmp_path / "warm", min_gap_sec=1.0, max_spans=2, **args)
    monkeypatch.undo()
    fresh = count(video, tmp_path / "fresh", min_gap_sec=1.0, max_spans=2, **args)
    sheets = [Path(r["contact_sheet"].pop("file_path")) for r in (rerun, fresh)]
    assert rerun == fresh and rerun["contact_sheet"]["sample_count"] == 2
    assert sheets[0].read_bytes() == sheets[1].read_bytes()


@needs_ffmpeg
def test_hitting_the_span_cap_returns_no_count(clips, tmp_path, monkeypatch):
    monkeypatch.setattr(occurrence, "MAX_SPANS_LIMIT", 3)
    result = count(clips["markers"], tmp_path, description="green diamond", max_spans=2)  # 4 spans found
    assert result["capped"] is True and result["count"] is None and result["spans_found"] == 4
    assert result["note"].startswith("NOT A COUNT") and "count is null" in result["note"]
    assert len(result["spans"]) == 2 and "gaps_merged" not in result
    monkeypatch.setattr(occurrence, "MAX_SPANS_LIMIT", 5)
    assert count(clips["markers"], tmp_path / "fresh", description="green diamond", max_spans=2)["count"] == 4


def test_span_builder_lists_gaps_only_the_default_would_split():
    hit = {"score": 1.0, "box": (.5, .5, .1, .1), "boxes": [(.5, .5, .1, .1)]}
    frame = Image.new("RGB", (32, 18))
    builder = occurrence.SpanBuilder(0.2, 1.0, keep=5)
    for t in (0.0, 0.2, 0.4, 1.0, 1.2, 1.4, 2.0):  # gaps of 0.2 s (flicker), 0.4 s, 0.4 s
        builder.add(t, hit, frame)
    builder.add(2.2, None, frame)
    assert len(builder.finish()) == 1 and len(builder.bridged) == 2
    assert [(round(a, 1), round(b, 1)) for a, b in builder.bridged] == [(0.6, 1.0), (1.6, 2.0)]
    default = occurrence.SpanBuilder(0.2, 0.3, keep=5)
    for t in (0.0, 0.2, 0.4, 1.0):
        default.add(t, hit, frame)
    assert len(default.finish()) == 2 and default.bridged == []


def test_batched_clip_judging_matches_one_frame_at_a_time():
    class FakeEncoder:
        def pil_images(self, crops):
            # a unit vector per crop, from its mean colour, so different tiles score differently
            means = np.array([np.asarray(c.convert("RGB")).reshape(-1, 3).mean(axis=0) for c in crops])
            return means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-9)

    judge = occurrence.FrameJudge("a person waving", None, False)
    assert judge.batchable
    judge._clip = (FakeEncoder(), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    frames = []
    for color, spot in (((200, 20, 20), (10, 10)), ((20, 200, 20), (400, 200)), ((20, 20, 200), (0, 0)), ((120, 120, 120), (300, 100))):
        image = Image.new("RGB", (640, 360), color)
        ImageDraw.Draw(image).rectangle([spot[0], spot[1], spot[0] + 200, spot[1] + 150], fill=(255, 0, 0))
        frames.append(image)
    assert judge.judge_many(frames) == [judge.judge(frame) for frame in frames]
    assert not occurrence.FrameJudge("purple diamond", None, False).batchable
    assert not occurrence.FrameJudge("a person waving", "claude", False).batchable
    assert not occurrence.FrameJudge("a red thing", None, False).batchable
