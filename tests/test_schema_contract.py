"""Parity checks for the canonical MCP contract."""

import json
import re
from pathlib import Path

from src.executor.mcp_server import TOOLS_MANIFEST


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src" / "adapters" / "mcp_schemas.json"
PUBLIC_TS = ROOT / "src" / "executor" / "types.d.ts"
ZOD_TS = ROOT / "src" / "adapters" / "mcp_definitions.ts"

EXPECTED_TOOLS = {
    "probe_video",
    "transcribe",
    "search_transcript",
    "build_visual_index",
    "search_visual",
    "count_repetitions",
    "count_occurrences",
    "contact_sheet",
    "review_video",
    "overview_chunks",
    "refine_temporal_beam",
    "track_region_changes",
    "track_region_events",
    "inspect_moments",
    "inspect_windows",
    "inspect_window",
    "detect_cuts",
    "check_frame_cadence",
    "audio_onsets",
    "loudness_timeline",
    "spectrogram_window",
    "sync_check",
}


def test_canonical_contract_matches_runtime_manifest():
    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))
    tools = contract["tools"]
    assert set(tools) == EXPECTED_TOOLS

    runtime = {entry["name"]: entry for entry in TOOLS_MANIFEST}
    assert set(runtime) == EXPECTED_TOOLS
    for name in EXPECTED_TOOLS:
        assert runtime[name]["description"] == tools[name]["description"]
        assert runtime[name]["inputSchema"] == tools[name]["inputSchema"]
        assert runtime[name]["outputSchema"] == tools[name]["outputSchema"]


def test_public_types_and_zod_surface_have_no_retired_tool_names():
    declarations = PUBLIC_TS.read_text(encoding="utf-8")
    zod = ZOD_TS.read_text(encoding="utf-8")

    for retired in ("sample_frames", "get_clip"):
        assert retired not in declarations
        assert retired not in zod

    for live in EXPECTED_TOOLS:
        assert re.search(rf"\b{re.escape(live)}\b", declarations)
        assert re.search(rf"\b{re.escape(live)}\b", zod)


def test_transcribe_contract_documents_cached_fallback_and_progress_result():
    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))
    transcribe = contract["tools"]["transcribe"]

    assert "background" in transcribe["description"]
    assert "retry_after_seconds" in transcribe["description"]
    output = transcribe["outputSchema"]
    assert len(output["oneOf"]) == 3  # transcript, ASR progress, URL download progress
    assert "cached_model_size" in output["properties"]
    assert "requested_model_size" in output["properties"]


def test_transcribe_and_search_schemas_accept_the_model_substitution_note():
    # asr.request_transcription adds this field when the requested model is not installed;
    # search_transcript passes that status payload through unchanged while ASR runs.
    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))
    response = {
        "status": "in_progress",
        "requested_model_size": "turbo",
        "requested_model_cached": False,
        "transcription_status": "started",
        "retry_after_seconds": 5,
        "transcription_message": "Transcription started.",
        "model_substitution": "The base model is not installed here; used turbo instead.",
    }
    for tool in ("transcribe", "search_transcript"):
        output = contract["tools"][tool]["outputSchema"]
        assert set(response) <= set(output["properties"]), tool
        _assert_schema_accepts(output, response)
    assert "model_substitution" in PUBLIC_TS.read_text(encoding="utf-8")


def test_public_frame_and_asr_provenance_fields_are_declared():
    declarations = PUBLIC_TS.read_text(encoding="utf-8")
    for field in ("requested_timestamp", "decoded_timestamp", "timestamp_delta", "confidence", "uncertainty"):
        assert field in declarations


def test_region_change_and_event_contracts_require_timestamp_provenance():
    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))

    change_sample = contract["tools"]["track_region_changes"]["outputSchema"]["properties"]["samples"]["items"]
    assert {"from_timestamp_provenance", "timestamp_provenance"} <= set(change_sample["required"])

    transition = contract["tools"]["track_region_events"]["outputSchema"]["properties"]["transitions"]["items"]
    assert {"from_timestamp_provenance", "timestamp_provenance"} <= set(transition["required"])
    for side in ("before", "after"):
        evidence = transition["properties"]["evidence_moments"]["properties"][side]
        assert "timestamp_provenance" in evidence["required"]


def _assert_schema_accepts(schema, value):
    """Validate the JSON Schema subset used by the contact_sheet contract."""
    schema_type = schema.get("type")
    if schema_type == "object":
        assert isinstance(value, dict)
        properties = schema.get("properties", {})
        assert set(schema.get("required", [])) <= set(value)
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties)
        for key, item in value.items():
            if key in properties:
                _assert_schema_accepts(properties[key], item)
    elif schema_type == "array":
        assert isinstance(value, list)
        if "items" in schema:
            for item in value:
                _assert_schema_accepts(schema["items"], item)
    elif schema_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool)
    elif schema_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
    elif schema_type == "string":
        assert isinstance(value, str)
    elif schema_type == "boolean":
        assert isinstance(value, bool)
    else:
        raise AssertionError(f"Unsupported schema type: {schema_type!r}")

    if schema_type in ("integer", "number"):
        if "minimum" in schema:
            assert value >= schema["minimum"]
        if "maximum" in schema:
            assert value <= schema["maximum"]
    if schema_type == "string" and "minLength" in schema:
        assert len(value) >= schema["minLength"]


def test_contact_sheet_clamp_request_and_response_match_canonical_schema():
    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))
    tool = contract["tools"]["contact_sheet"]
    request = {"source": "example.mp4", "samples": 600}
    response = {
        "file_path": "/tmp/contact-sheet.png",
        "sample_count": 150,
        "rows": 38,
        "cols": 4,
        "start_sec": 0.0,
        "end_sec": 60.0,
        "timestamps": [0.0] * 150,
        "frame_width": 320,
        "frame_height": 180,
        "samples_clamped_from": 600,
        "note": "samples clamped from 600 to 150",
        "tool_file": {},
    }

    _assert_schema_accepts(tool["inputSchema"], request)
    _assert_schema_accepts(tool["outputSchema"], response)
    assert response["sample_count"] == min(request["samples"], 150)
    assert response["samples_clamped_from"] == request["samples"]
    ready_required = tool["outputSchema"]["anyOf"][0]["required"]  # anyOf[1] is the URL download progress
    assert set(ready_required) <= set(response)
    assert "samples_clamped_from" not in ready_required
    assert "note" not in ready_required

    definitions = ZOD_TS.read_text(encoding="utf-8")
    contact_input = re.search(
        r"export const ContactSheetInputSchema = \{(?P<body>.*?)\n\};",
        definitions,
        re.DOTALL,
    )
    assert contact_input
    assert "samples: z.number().int().min(1).describe(" in contact_input["body"]
    assert ".max(150)" not in contact_input["body"]

    declarations = PUBLIC_TS.read_text(encoding="utf-8")
    contact_output = re.search(
        r"export interface ContactSheetOutput \{(?P<body>.*?)\n\}",
        declarations,
        re.DOTALL,
    )
    assert contact_output
    assert re.search(r"samples_clamped_from\?:\s*number;", contact_output["body"])
    assert re.search(r"note\?:\s*string;", contact_output["body"])


def test_detect_cuts_contract_tells_cuts_and_transitions_apart(monkeypatch):
    from src.executor import mcp_server
    from src.media.scene import SceneCut

    contract = json.loads(CANONICAL.read_text(encoding="utf-8"))
    tool = contract["tools"]["detect_cuts"]
    item = tool["outputSchema"]["properties"]["cuts"]["items"]
    assert item["properties"]["kind"]["enum"] == ["cut", "transition"]
    assert "kind" in item["required"]
    for words in ("`kind`", "transition", "middle", "false transitions"):
        assert words in tool["description"]

    declarations = PUBLIC_TS.read_text(encoding="utf-8")
    scene_cut = re.search(r"export interface SceneCut \{(?P<body>.*?)\n\}", declarations, re.DOTALL)
    assert scene_cut and re.search(r'kind:\s*"cut"\s*\|\s*"transition";', scene_cut["body"])
    assert "`kind` tells them apart" in ZOD_TS.read_text(encoding="utf-8")

    class Engine:
        def __init__(self, source):
            pass

        def detect_cuts(self, **kwargs):
            return [SceneCut(1.0, 0.6, "00:00:01.000"), SceneCut(2.5, 0.3, "00:00:02.500", kind="transition")]

    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", Engine)
    data = mcp_server.handle_call("detect_cuts", {"source": "clip.mp4"})["data"]
    assert [cut["kind"] for cut in data["cuts"]] == ["cut", "transition"]
    _assert_schema_accepts(tool["outputSchema"], data)
