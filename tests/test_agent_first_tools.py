"""Agent-facing behavior: clamped caps, bounded lists, actionable errors, consistent definitions."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from src.core.config import Config
from src.core.errors import ExtractionBoundsError, MediaNotFoundError
from src.core.types import VideoMetadata
from src.executor import mcp_server
from src.executor.mcp_server import (
    CLAMPED_ARGS, MAX_CUTS, MAX_LOUDNESS_WINDOWS, MAX_ONSETS, MAX_SYNC_ITEMS,
    TOOLS_MANIFEST, format_tool_error, handle_call, tool_call_response,
)
from src.retrieval.engine import AgenticVideoEngine

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "src/adapters/mcp_schemas.json").read_text(encoding="utf-8"))["tools"]
ZOD_TS = (ROOT / "src/adapters/mcp_definitions.ts").read_text(encoding="utf-8")
PUBLIC_TS = (ROOT / "src/executor/types.d.ts").read_text(encoding="utf-8")

# Minimal valid arguments so a clamp test can reach the dispatcher.
REQUIRED_VALUES = {
    "source": "clip.mp4", "query": "q", "start_sec": 0, "end_sec": 5,
    "region": {"x": 0, "y": 0, "width": 0.5, "height": 0.5},
    "windows": [{"start_sec": 0, "end_sec": 1}], "moments": [{"timestamp_sec": 1}],
}


def _minimal_args(tool):
    return {key: REQUIRED_VALUES[key] for key in CONTRACT[tool]["inputSchema"].get("required", [])}


class FakeEngine:
    """Records the last call and answers with canned data."""
    calls = []
    reply = None

    def __init__(self, source):
        self.source = source

    def __getattr__(self, method):
        def call(*args, **kwargs):
            FakeEngine.calls.append((method, kwargs))
            return FakeEngine.reply
        return call


@pytest.fixture
def fake_engine(monkeypatch):
    FakeEngine.calls = []
    FakeEngine.reply = {}
    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    return FakeEngine


# ---- clamping -------------------------------------------------------------

CLAMP_CASES = [(tool, arg, cap) for tool, args in CLAMPED_ARGS.items() for arg, (cap, _) in args.items()]


@pytest.mark.parametrize("tool,arg,cap", CLAMP_CASES)
def test_over_cap_argument_is_clamped_with_a_note(monkeypatch, tool, arg, cap):
    seen = {}

    def dispatch(name, args):
        seen.update(args)
        return {"ok": True, "data": {}}

    monkeypatch.setattr(mcp_server, "_dispatch", dispatch)
    result = handle_call(tool, {**_minimal_args(tool), arg: cap * 3})
    assert seen[arg] == cap
    assert result["data"][f"{arg}_clamped_from"] == cap * 3
    assert arg in result["data"]["note"] and str(cap) in result["data"]["note"]

    seen.clear()
    result = handle_call(tool, {**_minimal_args(tool), arg: cap})
    assert seen[arg] == cap
    assert "note" not in result["data"] and f"{arg}_clamped_from" not in result["data"]


@pytest.mark.parametrize("tool,arg,cap", CLAMP_CASES)
def test_clamp_table_matches_schema_and_documents_itself(tool, arg, cap):
    schema = CONTRACT[tool]["inputSchema"]["properties"][arg]
    assert schema["maximum"] == cap
    assert "clamped" in schema["description"]
    output = CONTRACT[tool]["outputSchema"]
    if output.get("additionalProperties") is False:
        assert f"{arg}_clamped_from" in output["properties"] and "note" in output["properties"]
    assert re.search(rf"{arg}_clamped_from\?: number;", PUBLIC_TS)


def test_search_visual_clamp_points_to_count_occurrences(fake_engine):
    fake_engine.reply = {"status": "complete", "hits": []}
    result = handle_call("search_visual", {"source": "clip.mp4", "query": "red car", "top_k": 100})
    assert fake_engine.calls[-1][1]["top_k"] == 20
    assert result["data"]["top_k_clamped_from"] == 100
    assert "count_occurrences" in result["data"]["note"]


def test_valid_search_visual_call_is_unchanged(fake_engine):
    fake_engine.reply = {"status": "complete", "hits": []}
    result = handle_call("search_visual", {"source": "clip.mp4", "query": "red car", "top_k": 20})
    assert fake_engine.calls[-1][1]["top_k"] == 20
    assert result["data"] == {"status": "complete", "hits": []}


def test_search_transcript_clamps_and_reports_more_matches(fake_engine):
    class Hit:
        def to_dict(self):
            return {"segment_id": 1}

    fake_engine.reply = [Hit() for _ in range(51)]
    result = handle_call("search_transcript", {"source": "clip.mp4", "query": "x", "top_k": 100, "mode": "lexical"})
    assert fake_engine.calls[-1][1]["top_k"] == 51
    assert result["data"]["total"] == 50 and len(result["data"]["hits"]) == 50
    assert result["data"]["top_k_clamped_from"] == 100
    assert "More matches exist" in result["data"]["note"]

    fake_engine.reply = [Hit(), Hit()]
    result = handle_call("search_transcript", {"source": "clip.mp4", "query": "x", "mode": "lexical"})
    assert fake_engine.calls[-1][1]["top_k"] == 6
    assert result["data"] == {"hits": [{"segment_id": 1}] * 2, "total": 2}


def test_search_transcript_passes_transcription_status_through(fake_engine):
    fake_engine.reply = {"status": "in_progress", "retry_after_seconds": 3}
    result = handle_call("search_transcript", {"source": "clip.mp4", "query": "x", "top_k": 100})
    assert result["data"]["status"] == "in_progress" and "hits" not in result["data"]


def test_search_transcript_rejects_top_k_below_one_with_range(fake_engine):
    with pytest.raises(ValueError, match="1 to 50"):
        handle_call("search_transcript", {"source": "clip.mp4", "query": "x", "top_k": 0})


# ---- bounded lists --------------------------------------------------------

def test_long_lists_are_bounded_and_say_what_was_omitted(fake_engine):
    class Cut:
        def __init__(self, i):
            self.timestamp_sec, self.score = float(i), 0.5

    fake_engine.reply = [Cut(i) for i in range(MAX_CUTS + 25)]
    data = handle_call("detect_cuts", {"source": "clip.mp4"})["data"]
    assert len(data["cuts"]) == MAX_CUTS and data["total"] == MAX_CUTS + 25 and data["cuts_omitted"] == 25
    assert "start_sec" in data["note"]

    fake_engine.reply = {"onsets": list(range(MAX_ONSETS + 1)), "total": MAX_ONSETS + 1}
    data = handle_call("audio_onsets", {"source": "clip.mp4"})["data"]
    assert len(data["onsets"]) == MAX_ONSETS and data["onsets_omitted"] == 1 and data["total"] == MAX_ONSETS + 1

    fake_engine.reply = {"windows": list(range(MAX_LOUDNESS_WINDOWS + 2)), "outliers": [], "total": MAX_LOUDNESS_WINDOWS + 2}
    data = handle_call("loudness_timeline", {"source": "clip.mp4"})["data"]
    assert len(data["windows"]) == MAX_LOUDNESS_WINDOWS and data["windows_omitted"] == 2

    fake_engine.reply = {"cuts": list(range(MAX_SYNC_ITEMS + 3)), "unmatched_strong_onsets": [1], "total_cuts": MAX_SYNC_ITEMS + 3}
    data = handle_call("sync_check", {"source": "clip.mp4"})["data"]
    assert len(data["cuts"]) == MAX_SYNC_ITEMS and data["cuts_omitted"] == 3
    assert "unmatched_strong_onsets_omitted" not in data


@pytest.mark.parametrize("tolerance_ms,reported", [(600, 600), (9000, 5000)])
def test_sync_check_accepts_every_documented_tolerance(monkeypatch, tmp_path, tolerance_ms, reported):
    import wave
    import numpy as np

    rate = 16000
    samples = np.zeros(rate * 3, dtype=np.float32)
    samples[rate:rate + 300] = 0.8
    wav = tmp_path / "clicks.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    monkeypatch.setattr(AgenticVideoEngine, "detect_cuts", lambda self: [])

    result = handle_call("sync_check", {"source": str(wav), "cues": [{"time_sec": 1.0}], "tolerance_ms": tolerance_ms})
    assert result["ok"], result
    assert result["data"]["tolerance_ms"] == reported


def test_sync_check_errors_keep_their_order_with_the_cut_scan_in_the_background(tmp_path):
    """The cut scan runs beside the audio work; an audio error still wins, and a scan error still surfaces."""
    import wave
    import numpy as np

    from src.core.errors import MediaCorruptError

    rate = 16000
    samples = np.zeros(rate * 2, dtype=np.float32)
    samples[rate:rate + 300] = 0.8
    wav = tmp_path / "clicks.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())

    # Audio decodes fine; the cut scan, on a file with no video, fails.
    for _ in range(2):
        with pytest.raises(MediaCorruptError, match="No video streams found"):
            handle_call("sync_check", {"source": str(wav), "cues": [{"time_sec": 1.0}]})
    # Both would fail on a missing file; the audio error is the one reported, as before.
    with pytest.raises(MediaNotFoundError, match="Audio source file not found"):
        handle_call("sync_check", {"source": str(tmp_path / "missing.mp4"), "cues": [{"time_sec": 1.0}]})


def test_short_lists_are_returned_untouched(fake_engine):
    fake_engine.reply = {"onsets": [1, 2, 3], "total": 3}
    assert handle_call("audio_onsets", {"source": "clip.mp4"})["data"] == {"onsets": [1, 2, 3], "total": 3}


def test_omission_fields_are_declared_in_every_definition():
    expected = {"detect_cuts": ["cuts"], "audio_onsets": ["onsets"], "loudness_timeline": ["windows"],
                "sync_check": ["cuts", "unmatched_strong_onsets"]}
    for tool, keys in expected.items():
        properties = CONTRACT[tool]["outputSchema"]["properties"]
        for key in keys:
            assert f"{key}_omitted" in properties
            assert re.search(rf"{key}_omitted\?: number;", PUBLIC_TS)
        assert "note" in properties


# ---- call validation ------------------------------------------------------

def test_missing_required_argument_names_it_and_the_required_set():
    with pytest.raises(ValueError, match=r"search_visual is missing required argument\(s\): query. Required: source, query"):
        handle_call("search_visual", {"source": "clip.mp4"})
    with pytest.raises(ValueError, match="`source` is a local video file path or a video URL"):
        handle_call("probe_video", {})


def test_unknown_tool_lists_available_tools():
    result = handle_call("search_visuals", {"source": "clip.mp4"})
    assert result["ok"] is False and "search_visual" in result["error"]["message"]


def test_arguments_must_be_an_object():
    with pytest.raises(ValueError, match="object of named parameters"):
        handle_call("probe_video", None)


def test_unknown_argument_is_reported_not_silently_dropped(fake_engine):
    fake_engine.reply = {"status": "complete"}
    result = handle_call("search_visual", {"source": "clip.mp4", "query": "q", "topk": 5})
    assert "Ignored unknown argument(s): topk" in result["data"]["note"]
    assert "top_k" in result["data"]["note"]


def test_format_tool_error_includes_structured_details_and_source_hint():
    text = format_tool_error(ExtractionBoundsError("window is outside media bounds", {"media_duration_sec": 12.0}))
    assert "media_duration_sec" in text and "ExtractionBoundsError" in text
    assert "existing local video file" in format_tool_error(MediaNotFoundError("Video file does not exist: x"))
    assert format_tool_error(ValueError("bad")) == "Error: bad"


# ---- argument checks on a real clip ---------------------------------------
# These run the real engine on a generated 3 s clip, so each wrong argument is
# caught by the same path an agent's call takes.

@pytest.fixture
def real_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    return str(clip)


@pytest.mark.parametrize("start,end", [("00:00.500", "00:01.500"), ("0:00:00.500", "0:00:01.500"), ("0.500", "1.500")])
def test_timecode_strings_match_the_numeric_seconds(real_clip, start, end):
    numeric = handle_call("inspect_window", {"source": real_clip, "start_sec": 0.5, "end_sec": 1.5})
    timecode = handle_call("inspect_window", {"source": real_clip, "start_sec": start, "end_sec": end})
    assert timecode["ok"] is True
    assert timecode["data"] == numeric["data"]


def test_timecode_strings_convert_inside_nested_window_items(real_clip):
    numeric = handle_call("inspect_windows", {"source": real_clip, "windows": [{"start_sec": 0.5, "end_sec": 1.5}]})
    timecode = handle_call("inspect_windows", {"source": real_clip, "windows": [{"start_sec": "00:00.500", "end_sec": "00:01.500"}]})
    assert timecode["ok"] is True
    assert timecode["data"] == numeric["data"]


def test_null_argument_counts_as_omitted(real_clip):
    assert handle_call("detect_cuts", {"source": real_clip, "threshold": None})["ok"] is True


@pytest.mark.parametrize("tool,region", [
    ("count_repetitions", {"x": 0, "y": 0, "width": 1, "height": 1}),
    ("track_region_changes", {"x": 0, "y": 0, "w": 1, "h": 1}),
    ("track_region_events", {"x": 0, "y": 0, "w": 1, "h": 1}),
])
def test_region_accepts_either_key_spelling(real_clip, tool, region):
    args = {"source": real_clip, "region": region, "start_sec": 0, "end_sec": 2}
    if tool == "count_repetitions":
        args.update(method="motion")
    else:
        args.update(sample_fps=2)
    result = handle_call(tool, args)
    assert result["ok"] is True, result


def test_bad_enum_value_names_the_allowed_values(real_clip):
    with pytest.raises(ValueError, match=r"count_repetitions: argument 'method' is not one of the allowed values "
                                         r"\(got 'fast'\); expected one of: auto, motion, embedding\."):
        handle_call("count_repetitions", {"source": real_clip, "method": "fast"})


BAD_ARGUMENT_CALLS = [
    ("detect_cuts", {"threshold": "high"}, "detect_cuts: argument 'threshold' must be a number (got 'high')"),
    ("detect_cuts", {"threshold": True}, "argument 'threshold' must be a number (got True)"),
    ("inspect_window", {"start_sec": "soon", "end_sec": 1}, "argument 'start_sec' must be a number (got 'soon'); expected seconds as a number or a MM:SS.mmm timecode"),
    ("inspect_window", {"start_sec": -1, "end_sec": 1}, "argument 'start_sec' is below the minimum 0 (got -1)"),
    ("count_repetitions", {"method": "fast"}, "argument 'method' is not one of the allowed values"),
    ("count_repetitions", {"region": {"x": 0, "y": 0, "w": 1}}, "argument 'region' does not match any allowed form; expected an object with key(s) x, y, w, h; or an object with key(s) x, y, width, height."),
    ("inspect_windows", {"windows": [{"start_sec": 0}]}, "argument 'windows[0]' is missing required key(s): end_sec"),
]


@pytest.mark.parametrize("tool,args,expected", BAD_ARGUMENT_CALLS)
def test_bad_argument_error_names_the_argument_and_is_not_a_tool_bug(real_clip, tool, args, expected):
    result = json.loads(tool_call_response(1, tool, {"source": real_clip, **args}))["result"]
    text = result["content"][0]["text"]
    assert result["isError"] is True
    assert expected in text
    assert "bug in the tool" not in text


# ---- actionable engine errors --------------------------------------------

def _engine(duration=60.0):
    engine = AgenticVideoEngine("synthetic-placeholder.mp4")
    engine._meta = VideoMetadata(source_path="synthetic-placeholder.mp4", duration_sec=duration, width=1280,
                                 height=720, fps=30.0, format_name="mp4", size_bytes=1, has_audio=False)
    return engine


def test_chunk_budget_error_says_which_chunk_length_fits():
    with pytest.raises(ExtractionBoundsError, match=r"max_chunks.*chunk_sec >= 12"):
        _engine().overview_chunks(start_sec=0.0, end_sec=60.0, chunk_sec=10.0, max_chunks=5)


def test_track_region_frame_budget_error_says_which_sample_fps_fits():
    region = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    with pytest.raises(ExtractionBoundsError, match=r"lower sample_fps to at most 5\.9"):
        _engine().track_region_changes(start_sec=0.0, end_sec=60.0, region=region, sample_fps=10.0, max_total_frames=355)
    with pytest.raises(ExtractionBoundsError, match=r"lower sample_fps to at most 5\.92"):
        _engine().track_region_events(start_sec=0.0, end_sec=60.0, region=region, sample_fps=10.0, max_total_frames=355)


def test_track_region_wrong_key_names_the_right_ones():
    with pytest.raises(ExtractionBoundsError, match="width/height, not w/h"):
        _engine().track_region_changes(start_sec=0.0, end_sec=10.0, region={"x": 0, "y": 0, "w": 1, "h": 1})


def test_inspect_budget_errors_say_how_to_fit():
    with pytest.raises(ExtractionBoundsError, match="lower samples_per_moment"):
        _engine().inspect_moments(moments=[{"timestamp_sec": 1}, {"timestamp_sec": 2}], samples_per_moment=9, max_total_samples=17)
    with pytest.raises(ExtractionBoundsError, match="call again with the remaining"):
        _engine().inspect_moments(moments=[{"timestamp_sec": 1}, {"timestamp_sec": 2}], max_moments=1)
    with pytest.raises(ExtractionBoundsError, match="lower fps"):
        _engine().inspect_windows(windows=[{"start_sec": 0, "end_sec": 50}], fps=10, max_total_frames=100)
    with pytest.raises(ExtractionBoundsError, match="pass fewer windows, lower fanout"):
        _engine().refine_temporal_beam(windows=[{"start_sec": 0, "end_sec": 50}], fanout=8, samples_per_child=12, max_total_samples=90)


def test_search_and_count_validation_errors_say_what_to_try_instead():
    from src.retrieval import occurrence, repetition, visual
    with pytest.raises(ValueError, match="clamps larger values to 20.*count_occurrences"):
        visual._validate_search("cat", 30, 0.0, None)
    with pytest.raises(ValueError, match="describe the visible content"):
        visual._validate_search("", 5, 0.0, None)
    with pytest.raises(ValueError, match=r"omit it for the default of 5"):
        occurrence._validate({"description": "yellow circle", "fps": 99})
    with pytest.raises(ValueError, match="count always includes every span"):
        occurrence._validate({"description": "yellow circle", "max_spans": 500})
    with pytest.raises(ValueError, match="w and h, not width/height"):
        repetition._validate(0.0, 5.0, 10.0, "auto", {"x": 0, "y": 0, "width": 1, "height": 1}, 0.4, 3.0)
    with pytest.raises(ValueError, match="window is 200s but must be within 180 seconds.*cut windows at pauses"):
        repetition._validate(0.0, 200.0, 10.0, "auto", None, 0.4, 3.0)


def test_dense_window_frame_limit_says_which_fps_fits():
    from src.core.cache import MediaCache
    with pytest.raises(ExtractionBoundsError, match=r"lower fps to at most 14\.9 for this 10\.0s window"):
        MediaCache.validate_extraction_bounds(fps=30.0, duration_sec=10.0)
    with pytest.raises(ExtractionBoundsError, match="contact_sheet / overview_chunks"):
        MediaCache.validate_extraction_bounds(fps=1.0, duration_sec=601.0)


# ---- definition files agree ----------------------------------------------

def test_typescript_descriptions_match_the_canonical_contract():
    for name, spec in CONTRACT.items():
        match = re.search(rf'\["{name}", \{{ description: ("(?:[^"\\]|\\.)*"), inputSchema', ZOD_TS)
        assert match, name
        assert json.loads(match.group(1)) == spec["description"], name


def test_zod_bounds_and_descriptions_match_the_canonical_schema():
    def camel(name):
        return "".join(part.capitalize() for part in name.split("_"))

    for name, spec in CONTRACT.items():
        body = re.search(rf"export const {camel(name)}InputSchema = \{{\n(.*?)\n\}};", ZOD_TS, re.DOTALL)
        assert body, name
        text = body.group(1)
        starts = [(m.group(1), m.end(), m.start()) for m in re.finditer(r"^  (\w+): ", text, re.MULTILINE)]
        fields = {prop: text[end:(starts[i + 1][2] if i + 1 < len(starts) else len(text))]
                  for i, (prop, end, _) in enumerate(starts)}
        properties = spec["inputSchema"]["properties"]
        assert set(fields) == set(properties), name
        for prop, schema in properties.items():
            expr = fields[prop]
            kinds = schema.get("type", [])
            kinds = kinds if isinstance(kinds, list) else [kinds]
            if "number" in kinds or "integer" in kinds:
                for key, method in (("minimum", "min"), ("maximum", "max")):
                    if key in schema:
                        assert f".{method}({schema[key]:g})" in expr, (name, prop, key)
                    else:
                        assert f".{method}(" not in expr, (name, prop, key)
                if "exclusiveMinimum" in schema:
                    assert ".positive()" in expr, (name, prop)
            if prop != "source" and "description" in schema:
                assert f".describe({json.dumps(schema['description'], ensure_ascii=False)})" in expr, (name, prop)


def test_every_tool_says_when_to_use_its_nearest_alternative():
    pairs = {
        "search_visual": ("count_occurrences", "count_repetitions", "search_transcript"),
        "count_occurrences": ("count_repetitions", "search_visual"),
        "count_repetitions": ("count_occurrences", "search_visual"),
        "search_transcript": ("count_occurrences", "search_visual"),
        "contact_sheet": ("inspect_window", "overview_chunks", "search_visual"),
        "overview_chunks": ("contact_sheet",),
        "inspect_window": ("contact_sheet", "inspect_windows"),
        "inspect_windows": ("inspect_window", "contact_sheet", "inspect_moments"),
        "inspect_moments": ("inspect_window",),
        "refine_temporal_beam": ("inspect_windows", "contact_sheet"),
        "track_region_changes": ("track_region_events",),
        "track_region_events": ("track_region_changes", "count_repetitions"),
        "audio_onsets": ("loudness_timeline", "label_sounds"),
        "loudness_timeline": ("audio_onsets",),
        "sync_check": ("detect_cuts", "audio_onsets"),
        "watch_for": ("inspect_window", "detect_cuts", "audio_onsets"),
        "label_sounds": ("audio_onsets", "loudness_timeline", "spectrogram_window", "search_transcript"),
    }
    for tool, names in pairs.items():
        for other in names:
            assert other in CONTRACT[tool]["description"], (tool, other)


def test_manifest_still_exposes_every_tool_with_documented_caps():
    assert {tool["name"] for tool in TOOLS_MANIFEST} == set(CONTRACT)
    assert "maximum 20" in CONTRACT["search_visual"]["description"]
    assert "maximum 50" in CONTRACT["search_transcript"]["description"]


@pytest.mark.parametrize("status", ["error", "in_progress"])
def test_unfinished_count_occurrences_gets_no_nothing_matched_advice(fake_engine, status):
    fake_engine.reply = {"status": status, "message": "window too long"}
    result = handle_call("count_occurrences", {"source": "clip.mp4", "text": "PRESIDENT"})
    assert "verify" not in result["data"]
