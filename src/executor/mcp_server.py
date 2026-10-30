"""Fast stdio MCP server for Agentic Video Understanding."""
import io
import json
import math
import re
import sys
import base64
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple

from src.retrieval.engine import AgenticVideoEngine
from src.core.config import Config
from src.core.errors import AgenticVideoError, MediaNotFoundError, as_dependency_error
from src.media import fetch, visual_timing
from src.transcript import asr

# Readable fallback kept for source compatibility; runtime discovery below is
# loaded from the canonical JSON contract.
_INLINE_MANIFEST_FALLBACK = [
    {
        "name": "probe_video",
        "description": "Extract container metadata, duration, fps, resolution, and codecs from a video file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Local filesystem path to video file"}
            },
            "required": ["source"]
        }
    },
    {
        "name": "transcribe",
        "description": "Return cached timestamped segments immediately. If the requested model is not cached, return another transcript with cached_model_size and requested_model_cached=false while one background job runs; with no transcript cached, return status=in_progress and retry_after_seconds. Long audio is decoded in 60-second chunks. model_size=turbo (Whisper large-v3-turbo on the GPU via whisper.cpp) is the most accurate and also the fastest where installed: about 1 minute per hour of audio, versus 5-15 minutes for small. Omitting model_size uses turbo where installed, else tiny.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Local filesystem path to video file"},
                "model_size": {"type": "string", "enum": ["tiny", "base", "small", "turbo"]}
            },
            "required": ["source"]
        }
    },
    {
        "name": "search_transcript",
        "description": "Search cached timestamped transcript segments: keyword matches first, then matches by meaning (mode=hybrid, default; mode=lexical for keyword matches only). If ASR is needed, start or join the background transcription and return status=in_progress with retry_after_seconds instead of waiting for transcription.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to video file"},
                "query": {"type": "string", "description": "Search query terms"},
                "top_k": {"type": "integer", "default": 5},
                "mode": {"type": "string", "enum": ["hybrid", "lexical"], "default": "hybrid"}
            },
            "required": ["source", "query"]
        }
    },
    {
        "name": "build_visual_index",
        "description": "Start or poll background construction of the local visual frame index. Repeat calls to check progress; frames are processed in bounded chunks.",
        "inputSchema": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}
    },
    {
        "name": "search_visual",
        "description": "Find when a visible object, person, action, or scene appears. Searches the local visual index; build it first if needed. Returns ranked timestamps, merged intervals (the one holding rank 1 is at least 15 seconds long, or the whole searched range when that is shorter; a shorter one is widened evenly around its middle and kept inside the searched range; overlapping intervals are merged into it), and a contact sheet for visual verification.",
        "inputSchema": {"type": "object", "properties": {"source": {"type": "string"}, "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}, "start_sec": {"type": "number", "minimum": 0, "default": 0}, "end_sec": {"type": ["number", "null"], "minimum": 0}}, "required": ["source", "query"]}
    },
    {
        "name": "contact_sheet",
        "description": "Generate a visual overview contact sheet for the whole video or a bounded temporal range, with burned timestamp badges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to video file"},
                "samples": {"type": "integer", "default": 12, "description": "Number of uniform samples (1-150; larger values are clamped to 150)"},
                "cols": {"type": "integer", "default": 4},
                "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0}
            },
            "required": ["source"]
        }
    },
    {
        "name": "inspect_window",
        "description": "Extract dense frames in a target temporal window [start_sec, end_sec] with timestamp overlay.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to video file"},
                "start_sec": {"type": "number", "description": "Start timestamp in seconds"},
                "end_sec": {"type": "number", "description": "End timestamp in seconds"},
                "fps": {"type": "number", "default": 4.0, "description": "Sampling rate in frames per second"}
            },
            "required": ["source", "start_sec", "end_sec"]
        }
    },
    {
        "name": "detect_cuts",
        "description": "List hard cuts and animated scene changes (kind: cut or transition) in the whole video or a bounded temporal range. A hard cut between two similar pictures is listed as a cut with a score below the default threshold; a threshold above 0.25 leaves these out.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to video file"},
                "threshold": {"type": "number", "default": 0.25},
                "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0}
            },
            "required": ["source"]
        }
    },
    {
        "name": "audio_onsets",
        "description": "Detect non-speech audio attacks with millisecond timestamps and five-band energy profiles.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0}, "min_strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.12},
                "band_hz": {"oneOf": [{"type": "string", "enum": ["sub", "low_mid", "mid", "high", "air"]}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]},
                "compare_to": {"type": "string"}
            },
            "required": ["source"]
        }
    },
    {
        "name": "loudness_timeline",
        "description": "Measure RMS, peak, band energies, spectral centroid, silence and band-energy outliers over fixed or beat-length windows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "window_sec": {"type": "number", "minimum": 0.01},
                "bpm": {"type": "number", "minimum": 20, "maximum": 300},
                "start_sec": {"type": "number", "minimum": 0}, "end_sec": {"type": "number", "minimum": 0},
                "ignore_ranges": {"type": "array", "items": {"type": "object", "properties": {"start_sec": {"type": "number", "minimum": 0}, "end_sec": {"type": "number", "minimum": 0}, "reason": {"type": "string"}}, "required": ["start_sec", "end_sec", "reason"]}}
            },
            "required": ["source"]
        }
    },
    {
        "name": "spectrogram_window",
        "description": "Render a log-frequency dB spectrogram PNG with absolute timecode badges and optional event markers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0},
                "marks": {"type": "array", "items": {"type": "object", "properties": {"time_sec": {"type": "number", "minimum": 0}, "label": {"type": "string"}}, "required": ["time_sec", "label"]}}
            },
            "required": ["source", "start_sec", "end_sec"]
        }
    },
    {
        "name": "sync_check",
        "description": "Compare visual cuts and measured audio onsets to each other and to optional intended sound cues.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "tolerance_ms": {"type": "number", "minimum": 0, "maximum": 5000, "default": 80},
                "cues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time_sec": {"type": "number", "minimum": 0},
                            "t": {"type": "number", "minimum": 0},
                            "label": {"type": "string"},
                            "band_hz": {"oneOf": [{"type": "string", "enum": ["sub", "low_mid", "mid", "high", "air"]}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]}
                        },
                        "additionalProperties": False,
                        "anyOf": [{"required": ["time_sec"]}, {"required": ["t"]}]
                    }
                }
            },
            "required": ["source"]
        }
    }
]


def _load_canonical_tool_manifest() -> List[Dict[str, Any]]:
    """Load the language-neutral MCP contract used by public clients.

    ``src/adapters/mcp_schemas.json`` is the source of truth.  Keeping the
    dispatch code below independent of the schema prevents a documentation
    edit from silently registering an unsupported tool.
    """
    contract_path = Path(__file__).resolve().parents[1] / "adapters" / "mcp_schemas.json"
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    tools = contract.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise RuntimeError(f"Invalid agentic-video MCP contract: {contract_path}")
    manifest: List[Dict[str, Any]] = []
    for key, spec in tools.items():
        if not isinstance(spec, dict) or spec.get("name") != key:
            raise RuntimeError(f"MCP contract tool key/name mismatch: {key}")
        manifest.append({
            "name": spec["name"],
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
            "outputSchema": spec.get("outputSchema"),
        })
    return manifest


# The short inline list above remains as a readable fallback for source
# inspection, but runtime discovery is always driven by the canonical JSON.
TOOLS_MANIFEST = _load_canonical_tool_manifest()
_TOOL_SCHEMAS = {tool["name"]: tool["inputSchema"] for tool in TOOLS_MANIFEST}
_TOOL_OUTPUT_PROPS = {
    tool["name"]: (tool.get("outputSchema") or {}).get("properties", {}) for tool in TOOLS_MANIFEST
}

# Arguments whose maximum is clamped, with a note in the result, instead of rejected.
# Each cap equals the schema maximum; a test keeps the two in step.
CLAMPED_ARGS: Dict[str, Dict[str, tuple]] = {
    "search_transcript": {"top_k": (50, "Make the query more specific to narrow the hits.")},
    "search_visual": {"top_k": (20, "search_visual ranks frames and cannot count; to count distinct appearances use "
                                    "count_occurrences, or split start_sec/end_sec into ranges and search each.")},
    "count_repetitions": {"fps": (15, "Frames are sampled at most 15 times per second.")},
    "count_occurrences": {
        "fps": (15, "Frames are sampled at most 15 times per second."),
        "max_spans": (100, "`count` always includes every span; to list later spans rerun with a later start_sec."),
    },
    "overview_chunks": {
        "samples_per_chunk": (150, "Use a smaller chunk_sec for denser coverage."),
        "max_chunks": (32, "Narrow start_sec/end_sec or raise chunk_sec to cover a long video."),
    },
    "refine_temporal_beam": {
        "fanout": (8, "Pass more windows, or call again on the child ranges, for finer coverage."),
        "samples_per_child": (12, "Pass more windows, or call again on the child ranges, for finer coverage."),
        "max_total_samples": (300, "Pass fewer windows or lower fanout/samples_per_child to fit the budget."),
    },
    "inspect_window": {"fps": (30, "Extraction is limited to 30 frames per second.")},
    "inspect_windows": {
        "fps": (30, "Extraction is limited to 30 frames per second."),
        "max_windows": (16, "Call again with the remaining windows."),
        "max_total_frames": (500, "Lower fps or shorten the windows to fit the budget."),
    },
    "inspect_moments": {
        "samples_per_moment": (25, "Use a smaller before_sec/after_sec for denser coverage."),
        "max_moments": (16, "Call again with the remaining moments."),
        "max_total_samples": (400, "Pass fewer moments or lower samples_per_moment to fit the budget."),
    },
    "track_region_changes": {
        "sample_fps": (10, "Split the range into shorter windows for denser coverage."),
        "frame_width": (1920, "Frames are decoded at most 1920 pixels wide."),
        "max_candidates": (128, "Raise min_change_score to keep only stronger candidates."),
        "max_total_frames": (2000, "Lower sample_fps or shorten start_sec/end_sec to fit the budget."),
    },
    "track_region_events": {
        "sample_fps": (10, "Split the range into shorter windows for denser coverage."),
        "frame_width": (1920, "Frames are decoded at most 1920 pixels wide."),
        "max_events": (128, "Raise change_threshold to keep only stronger events."),
        "max_total_frames": (2000, "Lower sample_fps or shorten start_sec/end_sec to fit the budget."),
    },
    "loudness_timeline": {"window_sec": (600, "Use start_sec/end_sec for a longer look.")},
    "sync_check": {"tolerance_ms": (5000, "Tolerances beyond 5 seconds stop being sync checks.")},
}

# Bounds for list-valued results. Each tool has a range argument to page through
# what was left out, and the result says which one to use.
MAX_CUTS = 500
MAX_ONSETS = 500
MAX_LOUDNESS_WINDOWS = 500
MAX_SYNC_ITEMS = 500
MAX_CADENCE_RANGES = 500


def format_tool_error(exc: Exception, tool: Optional[str] = None) -> str:
    """Render an exception so the caller sees the message and any structured details."""
    exc = as_dependency_error(exc)
    if isinstance(exc, AgenticVideoError):
        text = f"{type(exc).__name__}: {exc.message}"
        if isinstance(exc, MediaNotFoundError):
            text += ". Pass the path of an existing local video file, or a video URL, as `source`."
        if exc.details:
            text += " | details: " + json.dumps(exc.details, default=str)[:600]
        return text
    if isinstance(exc, ValueError):
        # handle_call raises ValueError with caller-facing advice already in it.
        return f"Error: {exc}"
    where = f" in agentic-video tool `{tool}`" if tool else " in agentic-video"
    return (
        f"Unexpected {type(exc).__name__}{where}: {exc}. "
        "This is a bug in the tool, not in your arguments. Retry once; if it fails again, "
        "try a smaller request (fewer samples, windows or a shorter range) and report the error text."
    )


def _add_note(data: Dict[str, Any], text: str) -> None:
    data["note"] = f"{data['note']} {text}" if data.get("note") else text


def _clamp_args(name: str, args: Dict[str, Any]) -> tuple:
    """Lower over-cap arguments to their cap. Returns (args, {arg: requested value})."""
    clamped: Dict[str, Any] = {}
    limited = dict(args)
    for arg, (cap, _advice) in CLAMPED_ARGS.get(name, {}).items():
        value = args.get(arg)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > cap:
            limited[arg] = cap
            clamped[arg] = value
    return limited, clamped


def _report_clamps(name: str, data: Dict[str, Any], clamped: Dict[str, Any]) -> None:
    """Say which arguments were lowered to their cap, and what to do instead."""
    for arg, requested in clamped.items():
        cap, advice = CLAMPED_ARGS[name][arg]
        data[f"{arg}_clamped_from"] = requested
        _add_note(data, f"{arg} capped at {cap} (requested {requested:g}). {advice}")


def _bound_list(data: Dict[str, Any], key: str, limit: int, advice: str) -> None:
    """Keep the first `limit` items of data[key] and say how many were left out."""
    items = data.get(key)
    if not isinstance(items, list) or len(items) <= limit:
        return
    omitted = len(items) - limit
    data[key] = items[:limit]
    data[f"{key}_omitted"] = omitted
    _add_note(data, f"Only the first {limit} of {len(items)} {key} are listed ({omitted} omitted). {advice}")


def make_tool_file(image_path: str) -> Dict[str, Any]:
    """Encode an image as an Executor ToolFile structure."""
    p = Path(image_path)
    if not p.exists():
        return {}
    raw = p.read_bytes()
    return {
        "_tag": "ToolFile",
        "name": p.name,
        "mimeType": "image/png" if p.suffix.lower() == ".png" else "image/jpeg",
        "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii"),
        "byteLength": len(raw),
    }


# Executor's MCP stdio client drops the connection on any message over 10 MiB,
# and callers then see only "Internal tool error". Keep every response under this.
MAX_RESPONSE_BYTES = 8_000_000
# Executor also gives up on a call after 60 s; stop extracting before that so
# the caller gets a message that says what to do instead of a timeout.
FRAME_EXTRACT_DEADLINE_SEC = 40.0
_TEXT_IMAGE_PLACEHOLDER = "<base64 image omitted; file_path has the image on disk (set AGENTIC_VIDEO_INLINE_IMAGES=1 to embed it in structuredContent)>"


def _is_tool_file(node: Any) -> bool:
    return isinstance(node, dict) and node.get("_tag") == "ToolFile" and isinstance(node.get("data"), str)


def _tool_files(node: Any) -> Iterator[Dict[str, Any]]:
    """Yield every ToolFile dict inside a response payload."""
    if _is_tool_file(node):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _tool_files(value)
    elif isinstance(node, list):
        for value in node:
            yield from _tool_files(value)


def _text_copy(node: Any) -> Any:
    """The payload for the text block: same JSON, without the base64 image bytes."""
    if _is_tool_file(node):
        return {**node, "data": _TEXT_IMAGE_PLACEHOLDER}
    if isinstance(node, dict):
        return {k: _text_copy(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_text_copy(v) for v in node]
    return node


def _success_result(data: Dict[str, Any]) -> Dict[str, Any]:
    # Default: no base64 anywhere. Claude Code rejects results this large (contact_sheet ~0.4-2 MB,
    # review_video ~1.4 MB); the agent opens the JPEG at file_path instead.
    # AGENTIC_VIDEO_INLINE_IMAGES=1 puts the images back in structuredContent for clients that render them.
    text_data = _text_copy(data)
    inline = os.environ.get("AGENTIC_VIDEO_INLINE_IMAGES") == "1"
    return {
        "content": [{"type": "text", "text": json.dumps(text_data, separators=(",", ":"))}],
        "structuredContent": data if inline else text_data,
    }


def _shrink_tool_file(tool_file: Dict[str, Any], max_b64: int) -> None:
    """Replace a ToolFile's image with a smaller JPEG whose base64 fits max_b64."""
    from PIL import Image

    raw = base64.b64decode(tool_file["data"])
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        scale = min(1.0, math.sqrt(max_b64 / max(1, len(tool_file["data"])))) * 0.95
        while True:
            size = (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
            buf = io.BytesIO()
            im.resize(size, Image.Resampling.LANCZOS).save(buf, "JPEG", quality=80)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            if len(encoded) <= max_b64 or min(size) <= 64:
                break
            scale *= 0.8
    stem = Path(str(tool_file.get("name") or "image")).stem
    tool_file.update({
        "name": f"{stem}_inline.jpg",
        "mimeType": "image/jpeg",
        "data": encoded,
        "byteLength": len(buf.getvalue()),
        "note": (
            f"Downscaled from {orig_w}x{orig_h} to {size[0]}x{size[1]} to fit the 10 MB MCP message limit. "
            "The full-size image is at the file path next to this tool_file."
        ),
    })


def fit_response(data: Dict[str, Any]) -> int:
    """Downscale inline images so the tools/call result fits in MAX_RESPONSE_BYTES.

    Returns how many images were downscaled. The full-size files stay on disk.
    """
    budget = MAX_RESPONSE_BYTES
    files = list(_tool_files(data))
    if not files:
        return 0
    total = len(json.dumps(_success_result(data)))
    if total <= budget:
        return 0
    other_bytes = total - sum(len(f["data"]) for f in files)
    per_image = max(20_000, (budget - other_bytes) // len(files))
    shrunk = 0
    for tool_file in files:
        if len(tool_file["data"]) > per_image:
            _shrink_tool_file(tool_file, per_image)
            shrunk += 1
    return shrunk

# Claude Code refuses tool results over 25k tokens, and timestamped JSON costs about 3 characters per token
# (a 60,000-character transcript was refused), so list at most 20,000 characters of segments.
MAX_TRANSCRIPT_INLINE_CHARS = 20_000


def _fit_transcript(data: Dict[str, Any]) -> None:
    """Keep a long transcript inside an agent client's result limit; the full one goes to a file."""
    segments = data.get("segments")
    if not isinstance(segments, list):
        return
    used, keep = 1, 0  # list brackets, then each segment plus its comma
    for segment in segments:
        used += len(json.dumps(segment, separators=(",", ":"))) + 1
        if used > MAX_TRANSCRIPT_INLINE_CHARS:
            break
        keep += 1
    if keep == len(segments):
        return
    path = Config.BASE_CACHE_DIR / "transcripts" / f"{hashlib.sha1(str(data.get('source_path')).encode()).hexdigest()[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    data["transcript_path"] = str(path)
    _bound_list(data, "segments", keep,
                f"The full transcript is in the JSON file at transcript_path; use search_transcript to find specific words. "
                f"The listed segments cover 0 to {segments[keep - 1]['end_sec']:g} s.")


_TIMECODE = re.compile(r"(?:(?:(?P<hours>[0-9]+):)?(?P<minutes>[0-9]+):)?(?P<seconds>[0-9]+(?:\.[0-9]+)?)")
# Longer timecode strings are refused (the published maxLength); this also keeps int() under Python's digit limit.
_TIMECODE_MAX_LENGTH = 32
_KIND_WORDS = {"number": "a number", "integer": "an integer", "string": "a string", "boolean": "true or false",
               "object": "an object", "array": "a list", "null": "null"}


def _parse_timecode(text: str) -> Optional[float]:
    """Seconds from "SS.mmm", "MM:SS(.mmm)" or "HH:MM:SS(.mmm)", the format search_transcript shows; None otherwise."""
    if len(text) > _TIMECODE_MAX_LENGTH:
        return None
    match = _TIMECODE.fullmatch(text)
    if match is None:
        return None
    hours, minutes = match["hours"], match["minutes"]
    # Range-check the whole-number digits: a fraction like .99999999999999999 rounds to 60.0 in float.
    whole_seconds = int(match["seconds"].split(".")[0])
    if (minutes is not None and whole_seconds >= 60) or (hours is not None and int(minutes) >= 60):
        return None
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(match["seconds"])


def _schema_types(schema: Dict[str, Any]) -> List[str]:
    kinds = schema.get("type", [])
    return kinds if isinstance(kinds, list) else [kinds]


def _accepts(schema: Dict[str, Any], value: Any) -> bool:
    """True if `value` has one of the schema's types, and a string matches the schema's pattern. The published
    *_sec pattern accepts exactly the timecodes _parse_timecode reads (tests/test_schema_contract.py checks this)."""
    if not any(_is_kind(value, kind) for kind in _schema_types(schema)):
        return False
    if not isinstance(value, str) or schema.get("pattern") is None:
        return True
    return re.fullmatch(schema["pattern"], value) is not None


def _is_kind(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "number":
        return isinstance(value, (int, float))
    if kind == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if kind == "string":
        return isinstance(value, str)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    return True


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _expected(schema: Dict[str, Any], key: str = "") -> str:
    """Plain words for what a value must be, e.g. 'a number, at least 0 and at most 1'."""
    if "enum" in schema:
        return "one of: " + ", ".join(str(item) for item in schema["enum"])
    kinds = _schema_types(schema)
    words = _KIND_WORDS.get(kinds[0], "a value") if kinds else "a value"
    if "object" in kinds and schema.get("required"):
        words = f"an object with key(s) {', '.join(schema['required'])}"
    if key.endswith("_sec") and "number" in kinds:
        words = "seconds as a number or a MM:SS.mmm timecode"
    bounds = []
    if "minimum" in schema:
        bounds.append(f"at least {schema['minimum']:g}")
    if "exclusiveMinimum" in schema:
        bounds.append(f"greater than {schema['exclusiveMinimum']:g}")
    if "maximum" in schema:
        bounds.append(f"at most {schema['maximum']:g}")
    if "minLength" in schema:
        bounds.append(f"at least {schema['minLength']} character(s)")
    if schema.get("minItems") is not None and schema.get("minItems") == schema.get("maxItems"):
        bounds.append(f"exactly {schema['minItems']} item(s)")
    else:
        if "minItems" in schema:
            bounds.append(f"at least {schema['minItems']} item(s)")
        if "maxItems" in schema:
            bounds.append(f"at most {schema['maxItems']} item(s)")
    return f"{words}, {' and '.join(bounds)}" if bounds else words


def _fields_expected(schema: Dict[str, Any]) -> str:
    """The keys an object accepts, and which of them it needs."""
    text = f"an object with fields {', '.join(schema.get('properties', {}))}"
    required = schema.get("required", [])
    return f"{text}; {', '.join(required)} required" if required else text


def _describe_alternative(schema: Dict[str, Any], key: str) -> str:
    if "enum" in schema or _schema_types(schema):
        return _expected(schema, key)
    if "required" in schema:
        return "an object with key(s) " + ", ".join(schema["required"])
    return "a value"


def _argument_problem(schema: Dict[str, Any], value: Any, key: str, check_range: bool = True) -> Optional[Tuple[str, str, str]]:
    """Return (path, problem, expected) for the first way `value` breaks `schema`, or None.

    `path` is "" for the value itself, or ".field" / "[index]" steps down to the part that is wrong.
    """
    for combinator in ("anyOf", "oneOf"):
        alternatives = schema.get(combinator)
        if not alternatives:
            continue
        problems = [_argument_problem(alt, value, key, check_range) for alt in alternatives]
        if all(problems):
            fitting = [problem for alt, problem in zip(alternatives, problems)
                       if not _schema_types(alt) or _accepts(alt, value)]
            if len(fitting) == 1:
                return fitting[0]
            return "", "does not match any allowed form", "; or ".join(_describe_alternative(alt, key) for alt in alternatives)
    # Length first, so an oversized string never reaches the pattern match or int().
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return "", f"is shorter than {schema['minLength']} character(s)", _expected(schema, key)
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return "", f"is longer than {schema['maxLength']} character(s)", _expected(schema, key)
    kinds = _schema_types(schema)
    if kinds and not _accepts(schema, value):
        return "", f"must be {_KIND_WORDS.get(kinds[0], 'a value')} (got {_show(value)})", _expected(schema, key)
    if "enum" in schema and value not in schema["enum"]:
        return "", f"is not one of the allowed values (got {_show(value)})", _expected(schema, key)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            return "", f"has {len(value)} item(s) (needs at least {schema['minItems']})", _expected(schema, key)
        if check_range and len(value) > schema.get("maxItems", len(value)):
            return "", f"has {len(value)} item(s) (allows at most {schema['maxItems']})", _expected(schema, key)
    if check_range and _is_kind(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return "", f"is below the minimum {schema['minimum']:g} (got {_show(value)})", _expected(schema, key)
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return "", f"must be greater than {schema['exclusiveMinimum']:g} (got {_show(value)})", _expected(schema, key)
        if "maximum" in schema and value > schema["maximum"]:
            return "", f"is above the maximum {schema['maximum']:g} (got {_show(value)})", _expected(schema, key)
    if isinstance(value, dict):
        missing = [item for item in schema.get("required", []) if item not in value]
        if missing:
            return "", f"is missing required key(s): {', '.join(missing)}", _expected(schema, key)
        if schema.get("additionalProperties") is False:
            unknown = [field for field in value if field not in schema.get("properties", {})]
            if unknown:
                return f".{unknown[0]}", "is not an accepted field", _fields_expected(schema)
        for field, sub in schema.get("properties", {}).items():
            if field in value:
                problem = _argument_problem(sub, value[field], field, check_range)
                if problem:
                    return (f".{field}{problem[0]}",) + problem[1:]
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            problem = _argument_problem(schema["items"], item, key, check_range)
            if problem:
                return (f"[{index}]{problem[0]}",) + problem[1:]
    return None


def _read_timecodes(schema: Dict[str, Any], value: Any, key: str = "") -> Any:
    """Turn timecode strings in *_sec number fields into seconds, at any depth the schema describes."""
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        return {field: _read_timecodes(schema["properties"][field], item, field) if field in schema["properties"] else item
                for field, item in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_read_timecodes(schema["items"], item) for item in value]
    if isinstance(value, str) and key.endswith("_sec") and "number" in _schema_types(schema):
        seconds = _parse_timecode(value)
        if seconds is not None:
            return seconds
    return value


def _region_in_tool_keys(name: str, region: Dict[str, Any]) -> Dict[str, Any]:
    """Rename w/h to width/height, or back, so the region has the keys this tool's code reads (count_repetitions reads w/h)."""
    short = name == "count_repetitions"
    out = dict(region)
    for short_key, long_key in (("w", "width"), ("h", "height")):
        wanted, other = (short_key, long_key) if short else (long_key, short_key)
        if other in out and wanted not in out:
            out[wanted] = out.pop(other)
    return out


# One example value per argument whose shape is easy to guess wrong. Arguments with a default or an enum use those.
_CALL_EXAMPLES = {
    "moments": [{"timestamp_sec": 12.5}],
    "windows": [{"start_sec": 10, "end_sec": 14}],
    "cues": [{"time_sec": 12.5, "label": "hit"}],
    "marks": [{"time_sec": 12.5, "label": "kick"}],
    "ignore_ranges": [{"start_sec": 1, "end_sec": 2, "reason": "title card"}],
    "source": "/path/to/video.mp4",
    "query": "launch date",
    "description": "yellow outlined circle",
    "text": "RELEASE",
    "time_sec": 12.5,
}
_MOMENT_TIME_KEYS = ("timestamp_sec", "time_sec", "time", "t")


def _example_value(name: str, key: str, schema: Dict[str, Any]) -> Any:
    """A value the schema accepts for `key`: its default, its first enum value, or a known example."""
    if key == "region":
        return {"x": 0.25, "y": 0.1, "w": 0.5, "h": 0.6} if name == "count_repetitions" \
            else {"x": 0.25, "y": 0.1, "width": 0.5, "height": 0.6}
    if "default" in schema:
        return schema["default"]
    if "enum" in schema:
        return schema["enum"][0]
    if key in _CALL_EXAMPLES:
        return _CALL_EXAMPLES[key]
    for combinator in ("anyOf", "oneOf"):
        if schema.get(combinator):
            return _example_value(name, key, schema[combinator][0])
    kinds = _schema_types(schema)
    if "number" in kinds or "integer" in kinds:
        value = 1
        if "maximum" in schema:
            value = min(value, schema["maximum"])
        if "minimum" in schema:
            value = max(value, schema["minimum"])
        return value
    if "string" in kinds:
        return "text"
    if "boolean" in kinds:
        return True
    if "object" in kinds:
        properties = schema.get("properties", {})
        return {field: _example_value(name, field, properties.get(field, {})) for field in schema.get("required", [])}
    if "array" in kinds:
        return [_example_value(name, key, schema.get("items", {}))]
    return None


def _argument_error(name: str, key: str, path: str, reason: str, expected: str) -> str:
    """Where the argument is wrong, what it must be, and one correct call fragment to copy."""
    example = f'"{key}": {json.dumps(_example_value(name, key, _TOOL_SCHEMAS[name].get("properties", {}).get(key, {})))}'
    return f"{name}: argument '{key}{path}' {reason}; expected {expected}. Example: {example}."


def _moment_shape(name: str, item: Any, index: int) -> Any:
    """A bare number, or an object with one time key, becomes {"timestamp_sec": t}. Two time keys are ambiguous."""
    if _is_kind(item, "number"):
        return {"timestamp_sec": item}
    if isinstance(item, dict):
        found = [key for key in _MOMENT_TIME_KEYS if key in item]
        if len(found) > 1:
            raise ValueError(_argument_error(name, "moments", f"[{index}]",
                                             f"has more than one time key ({', '.join(found)})",
                                             "exactly one of timestamp_sec, time_sec, time, t"))
        if found and found[0] != "timestamp_sec":
            rest = {key: value for key, value in item.items() if key != found[0]}
            return {"timestamp_sec": item[found[0]], **rest}
    return item


def _accepted_shapes(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite the wrong shapes whose meaning is certain: bare or time-keyed moments, and a four-number region list.
    Anything else is left for the schema check to refuse."""
    properties = _TOOL_SCHEMAS[name].get("properties", {})
    out = dict(args)
    if "moments" in properties and isinstance(out.get("moments"), list):
        out["moments"] = [_moment_shape(name, item, index) for index, item in enumerate(out["moments"])]
    region = out.get("region")
    if "region" in properties and isinstance(region, list):
        if len(region) != 4 or not all(_is_kind(value, "number") for value in region):
            keys = "x, y, w, h" if name == "count_repetitions" else "x, y, width, height"
            raise ValueError(_argument_error(name, "region", "", "is a list that is not four numbers",
                                             f"an object with {keys}, or a list of four numbers [{keys}]"))
        x, y, width, height = region
        out["region"] = {"x": x, "y": y, "width": width, "height": height}
    return out


def _checked_arguments(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Read shapes and timecodes, and reject any value that breaks the tool's schema (nulls included)."""
    schema = _TOOL_SCHEMAS[name]
    properties = schema.get("properties", {})
    kept = _read_timecodes(schema, _accepted_shapes(name, args))
    if isinstance(kept.get("region"), dict) and "region" in properties:
        kept["region"] = _region_in_tool_keys(name, kept["region"])
    clamped = CLAMPED_ARGS.get(name, {})
    for key, value in kept.items():
        if key not in properties:
            continue
        problem = _argument_problem(properties[key], value, key, check_range=key not in clamped)
        if problem:
            path, reason, expected = problem
            raise ValueError(_argument_error(name, key, path, reason, expected))
    return kept


def handle_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the call, download URL sources, clamp over-cap arguments, and dispatch to AgenticVideoEngine."""
    if name not in _TOOL_SCHEMAS:
        return {"ok": False, "error": {"code": "unknown_tool", "message": f"Tool '{name}' not found; available tools: {', '.join(sorted(_TOOL_SCHEMAS))}"}}
    if not isinstance(args, dict):
        raise ValueError(f"{name} arguments must be an object of named parameters, e.g. {{\"source\": \"/path/to/video.mp4\"}}")
    schema = _TOOL_SCHEMAS[name]
    required = schema.get("required", [])
    missing = [key for key in required if args.get(key) in (None, "")]
    if missing:
        hint = (" `source` is a local video file path or a video URL (X post, YouTube, direct link)."
                if "source" in missing else "")
        properties = schema.get("properties", {})
        problems = " ".join(_argument_error(name, key, "", "is missing or empty", _expected(properties.get(key, {}), key))
                            for key in missing)
        raise ValueError(f"{name} is missing required argument(s): {', '.join(missing)}. Required: {', '.join(required)}. "
                         f"{problems}{hint}")
    unknown = sorted(set(args) - set(schema.get("properties", {})))
    args = _checked_arguments(name, args)
    url_args = {key: args[key] for key in ("source", "compare_to")
                if isinstance(args.get(key), str) and fetch.is_url(args[key])}
    if url_args:
        try:
            local = fetch.resolve_urls(url_args.values())
        except fetch.DownloadError as exc:
            return {"ok": False, "error": {"code": "download_failed", "message": str(exc)}}
        pending = [url for url in url_args.values() if local[url] is None]
        if pending:
            return {"ok": True, "data": fetch.pending_result(pending[0])}
        args = {**args, **{key: str(local[url]) for key, url in url_args.items()}}
    source_url = url_args.get("source")
    args, clamped = _clamp_args(name, args)
    outcome = _dispatch(name, args)
    if outcome.get("ok") and isinstance(outcome.get("data"), dict):
        data = outcome["data"] = dict(outcome["data"])
        if source_url:
            data.update(source_url=source_url, local_path=args["source"])
        _report_clamps(name, data, clamped)
        if name == "transcribe":
            _fit_transcript(data)
        if unknown:
            _add_note(data, f"Ignored unknown argument(s): {', '.join(unknown)}. Valid arguments: {', '.join(schema.get('properties', {}))}.")
        shrunk = fit_response(data)
        if shrunk and "note" in _TOOL_OUTPUT_PROPS.get(name, {}):
            _add_note(data, f"{shrunk} inline image(s) were downscaled to fit the 10 MB MCP message limit; open file_path for full size.")
    return outcome


MAX_REVIEW_OUTLIERS = 10
MAX_REVIEW_SILENCE_GAPS = 20


def _loudness_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Boil a full loudness_timeline result down to what a first look needs."""
    rms = [w["rms_dbfs"] for w in data["windows"]]
    average = round(10 * math.log10(sum(10 ** (v / 10) for v in rms) / len(rms)), 3) if rms else None
    outliers = data["outliers"]
    strongest = sorted(outliers, key=lambda o: (not o["fault_level"], -abs(o["delta_db"])))[:MAX_REVIEW_OUTLIERS]
    gaps = data["silence_gaps"]
    return {
        "average_rms_dbfs": average,
        "peak_dbfs": data["peak_dbfs"],
        "peak_sec": data["peak_sec"],
        "window_sec": data["window_sec"],
        "silence_threshold_dbfs": data["silence_threshold_dbfs"],
        "silence_gaps": gaps[:MAX_REVIEW_SILENCE_GAPS],
        "silence_gaps_total": len(gaps),
        "outliers_total": len(outliers),
        "fault_outliers_total": sum(1 for o in outliers if o["fault_level"]),
        "outliers": [{key: o[key] for key in ("start_sec", "end_sec", "band", "direction", "shape", "delta_db", "fault_level")}
                     for o in strongest],
    }


def _dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    source = args.get("source", "")
    engine = AgenticVideoEngine(source)
    
    if name == "probe_video":
        meta = engine.probe(allow_audio_only=True)
        return {"ok": True, "data": meta.to_dict()}
        
    elif name == "transcribe":
        model_size = args.get("model_size") or asr.default_model_size()
        data = engine.transcribe_request(model_size=model_size)
        return {"ok": True, "data": data}
        
    elif name == "search_transcript":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, (int, float)) or top_k < 1:
            raise ValueError(f"top_k must be an integer from 1 to 50 (got {top_k!r}); omit it for the default of 5")
        top_k = int(top_k)
        mode = args.get("mode", "hybrid")
        if mode not in ("hybrid", "lexical"):
            raise ValueError(f"mode must be 'hybrid' (keyword hits, then matches by meaning) or 'lexical' (keyword hits only) (got {mode!r}); omit it for hybrid")
        hybrid = mode == "hybrid"
        # Lexical: ask for one extra hit so the result can say whether more matches exist.
        # Hybrid: the search itself notes more keyword matches.
        results = engine.search_transcript(query, top_k=top_k if hybrid else top_k + 1, hybrid=hybrid)
        if isinstance(results, dict):
            data = dict(results)
            hits = data.get("hits")
            if hits is None:  # transcription still running: pass the status through
                return {"ok": True, "data": data}
        else:
            data = {}
            hits = [r.to_dict() for r in results]
        data["hits"] = hits[:top_k]
        data["total"] = len(data["hits"])
        if len(hits) > top_k:
            _add_note(data, f"More matches exist beyond top_k={top_k}; raise top_k (maximum 50) or make the query more specific.")
        return {"ok": True, "data": data}

    elif name == "build_visual_index":
        progress = engine.build_visual_index()
        return {"ok": True, "data": progress}

    elif name == "search_visual":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        start_sec = float(args.get("start_sec", 0.0))
        end_raw = args.get("end_sec")
        end_sec = None if end_raw is None else float(end_raw)
        result = engine.search_visual(query, top_k=top_k, start_sec=start_sec, end_sec=end_sec)
        data = dict(result)
        sheet = data.get("contact_sheet")
        if isinstance(sheet, dict) and sheet.get("file_path"):
            sheet["tool_file"] = make_tool_file(sheet["file_path"])
        elif isinstance(sheet, str):
            data["contact_sheet"] = {"file_path": sheet, "tool_file": make_tool_file(sheet)}
        return {"ok": True, "data": data}

    elif name == "count_repetitions":
        data = dict(engine.count_repetitions(
            start_sec=args.get("start_sec", 0.0), end_sec=args.get("end_sec"),
            region=args.get("region"), fps=args.get("fps", 10.0),
            method=args.get("method", "auto"),
            min_period_sec=args.get("min_period_sec", 0.4),
            max_period_sec=args.get("max_period_sec", 3.0),
            action_hint=args.get("action_hint")))
        data["verify"] = ("Rough count. Before answering, count cycles by eye with inspect_window or "
                          "contact_sheet over this window; if your count differs, trust your own.")
        sheet = data.get("contact_sheet")
        if isinstance(sheet, dict) and sheet.get("file_path"):
            sheet = dict(sheet)
            data["contact_sheet"] = sheet
            sheet["tool_file"] = make_tool_file(sheet["file_path"])
        return {"ok": True, "data": data}
        
    elif name == "count_occurrences":
        data = dict(engine.count_occurrences(
            description=args.get("description"), text=args.get("text"),
            case_sensitive=args.get("case_sensitive", False),
            start_sec=args.get("start_sec", 0.0), end_sec=args.get("end_sec"),
            fps=args.get("fps", 5.0), min_gap_sec=args.get("min_gap_sec", 0.3),
            max_spans=args.get("max_spans", 20)))
        data["verify"] = ("Rough count: spans can be false, merge across scene cuts, split one appearance, or miss "
                          "crossings. Before answering, look at each tile of the contact_sheet image (one per span, "
                          "in span order, a zoomed frame of its best match, boxed) and drop any span whose tile is "
                          "not the asked-for thing; spans with weak=true (tile label *) are the likeliest false ones. "
                          "The answer is the number of tiles that match, plus any spans the sheet does not show; if "
                          "you also count by eye and differ, trust your own.")
        if data.get("capped"):
            data["verify"] = ("There is no count. Do not answer with a number from this result; narrow the "
                              "description or window and rerun, or count by eye from the same window.")
        elif data.get("status") != "complete":
            data.pop("verify", None)
        elif not data.get("spans"):
            data["verify"] = ("Nothing matched. Before answering 0, look at contact_sheet frames over the same window "
                              "and count by eye; if your count differs, trust your own.")
        elif data.get("gaps_merged"):
            data["verify"] += (" min_gap_sec joined separate appearances here; a tile can show only one of them, so "
                               "check the gaps listed in note and use count_without_merges if they were separate.")
        if data.get("travel_counts") and not data.get("capped") and "verify" in data:
            data["verify"] += (" Some appearances show net motion across the frame; travel_counts estimates it from first "
                               "and last positions, so it is not a verified crossing or direction count. The count includes "
                               "all directions: do not answer a directional question from it without checking with inspect_window.")
        sheet = data.get("contact_sheet")
        if isinstance(sheet, dict) and sheet.get("file_path"):
            sheet = dict(sheet)
            data["contact_sheet"] = sheet
            sheet["tool_file"] = make_tool_file(sheet["file_path"])
        return {"ok": True, "data": data}

    elif name == "contact_sheet":
        requested_samples = int(args.get("samples", 12))
        # Agents often ask for hundreds of frames on long videos; give them the
        # largest sheet allowed instead of an error round trip.
        samples = min(max(requested_samples, 1), Config.MAX_FRAMES_PER_CALL)
        cols = int(args.get("cols", 4))
        start_sec = float(args.get("start_sec", 0.0))
        end_raw = args.get("end_sec")
        end_sec = None if end_raw is None else float(end_raw)
        sheet = engine.overview(
            samples=samples, cols=cols, start_sec=start_sec, end_sec=end_sec,
            deadline_sec=FRAME_EXTRACT_DEADLINE_SEC,
        )
        d = sheet.to_dict()
        if samples != requested_samples:
            d["samples_clamped_from"] = requested_samples
            d["note"] = (
                f"samples clamped from {requested_samples} to {samples} "
                f"(allowed range 1-{Config.MAX_FRAMES_PER_CALL}); use start_sec/end_sec for denser coverage"
            )
        d["tool_file"] = make_tool_file(sheet.file_path)
        return {"ok": True, "data": d}
        
    elif name == "review_video":
        # Any step failing fails the whole call with that step's own error.
        meta = engine.probe().to_dict()
        sheet = _dispatch("contact_sheet", {"source": source, "samples": args.get("samples", 24)})["data"]
        cuts = _dispatch("detect_cuts", {"source": source})["data"]
        data = {"metadata": meta, "contact_sheet": sheet, "cuts": cuts["cuts"], "cuts_total": cuts["total"],
                "loudness": _loudness_summary(engine.loudness_timeline()) if meta["has_audio"] else None}
        if "cuts_omitted" in cuts:
            data["cuts_omitted"] = cuts["cuts_omitted"]
        for note in (cuts.get("note"), None if meta["has_audio"] else "No audio track, so loudness is null.", sheet.pop("note", None)):
            if note:
                _add_note(data, note)
        return {"ok": True, "data": data}

    elif name == "overview_chunks":
        chunk_sec = float(args.get("chunk_sec", 60.0))
        samples_per_chunk = int(args.get("samples_per_chunk", 8))
        cols = int(args.get("cols", 4))
        start_sec = float(args.get("start_sec", 0.0))
        end_raw = args.get("end_sec")
        end_sec = None if end_raw is None else float(end_raw)
        max_chunks = int(args.get("max_chunks", 32))
        chunks = engine.overview_chunks(
            chunk_sec=chunk_sec,
            samples_per_chunk=samples_per_chunk,
            cols=cols,
            start_sec=start_sec,
            end_sec=end_sec,
            max_chunks=max_chunks,
        )
        for chunk in chunks:
            chunk["tool_file"] = make_tool_file(chunk["file_path"])
        return {
            "ok": True,
            "data": {
                "chunks": chunks,
                "total": len(chunks),
                "start_sec": chunks[0]["range_start_sec"],
                "end_sec": chunks[-1]["range_end_sec"],
                "chunk_sec": chunk_sec,
            },
        }

    elif name == "refine_temporal_beam":
        data = engine.refine_temporal_beam(
            windows=args.get("windows", []), fanout=int(args.get("fanout", 4)),
            samples_per_child=int(args.get("samples_per_child", 3)), cols=int(args.get("cols", 3)),
            context_sec=float(args.get("context_sec", 0.0)), max_total_samples=int(args.get("max_total_samples", 96)),
            align_to_cuts=bool(args.get("align_to_cuts", False)),
            compact_parent_sheet=bool(args.get("compact_parent_sheet", False)),
        )
        for parent in data["parents"]:
            compact_path = parent.get("compact_contact_sheet_path")
            if compact_path:
                parent["compact_tool_file"] = make_tool_file(compact_path)
            for child in parent["children"]:
                child["tool_file"] = make_tool_file(child["contact_sheet_path"])
        return {"ok": True, "data": data}
    elif name == "track_region_changes":
        data = engine.track_region_changes(
            start_sec=float(args.get("start_sec", 0.0)),
            end_sec=float(args.get("end_sec")),
            region=args.get("region", {}),
            sample_fps=float(args.get("sample_fps", 0.25)),
            frame_width=int(args.get("frame_width", 640)),
            min_change_score=float(args.get("min_change_score", 0.02)),
            robust_sigma=float(args.get("robust_sigma", 6.0)),
            min_separation_sec=float(args.get("min_separation_sec", 2.0)),
            max_candidates=int(args.get("max_candidates", 32)),
            max_total_frames=int(args.get("max_total_frames", 600)),
        )
        return {"ok": True, "data": data}

    elif name == "track_region_events":
        data = engine.track_region_events(
            start_sec=float(args.get("start_sec", 0.0)),
            end_sec=float(args.get("end_sec")),
            region=args.get("region", {}),
            sample_fps=float(args.get("sample_fps", 0.25)),
            frame_width=int(args.get("frame_width", 640)),
            change_threshold=float(args.get("change_threshold", 0.02)),
            ambiguity_margin=float(args.get("ambiguity_margin", 0.01)),
            min_separation_sec=float(args.get("min_separation_sec", 2.0)),
            max_events=int(args.get("max_events", 32)),
            max_total_frames=int(args.get("max_total_frames", 600)),
        )
        return {"ok": True, "data": data}

    elif name == "inspect_moments":
        data = engine.inspect_moments(
            moments=args.get("moments", []),
            before_sec=float(args.get("before_sec", 4.0)),
            after_sec=float(args.get("after_sec", 4.0)),
            samples_per_moment=int(args.get("samples_per_moment", 9)),
            cols=int(args.get("cols", 3)),
            max_moments=int(args.get("max_moments", 16)),
            max_total_samples=int(args.get("max_total_samples", 144)),
        )
        for item in data["moments"]:
            sheet_path = item.get("contact_sheet_path")
            if sheet_path:
                item["tool_file"] = make_tool_file(sheet_path)
        return {"ok": True, "data": data}

    elif name == "inspect_windows":
        windows = args.get("windows", [])
        fps = float(args.get("fps", 4.0))
        context_sec = float(args.get("context_sec", 0.0))
        merge_overlaps = bool(args.get("merge_overlaps", True))
        max_windows = int(args.get("max_windows", 16))
        max_total_frames = int(args.get("max_total_frames", 150))
        data = engine.inspect_windows(
            windows=windows,
            fps=fps,
            context_sec=context_sec,
            merge_overlaps=merge_overlaps,
            max_windows=max_windows,
            max_total_frames=max_total_frames,
        )
        for item in data["windows"]:
            sheet_path = item.get("contact_sheet_path")
            if sheet_path:
                item["tool_file"] = make_tool_file(sheet_path)
        return {"ok": True, "data": data}

    elif name == "inspect_window":
        start_sec = float(args.get("start_sec", 0.0))
        end_sec = float(args.get("end_sec", 0.0))
        fps = float(args.get("fps", 4.0))
        res = engine.inspect_window(start_sec=start_sec, end_sec=end_sec, fps=fps)
        d = res.to_dict()
        if res.contact_sheet_path:
            d["tool_file"] = make_tool_file(res.contact_sheet_path)
        return {"ok": True, "data": d}
        
    elif name == "frame_at":
        data = engine.frame_at(time_sec=args.get("time_sec"), crop=args.get("crop"), scale=args.get("scale"))
        data["tool_file"] = make_tool_file(data["file_path"])
        return {"ok": True, "data": data}

    elif name == "detect_cuts":
        th = float(args.get("threshold", 0.25))
        start_sec = float(args.get("start_sec", 0.0))
        end_raw = args.get("end_sec")
        end_sec = None if end_raw is None else float(end_raw)
        cuts = engine.detect_cuts(threshold=th, start_sec=start_sec, end_sec=end_sec)
        cut_items = [{"timestamp_sec": c.timestamp_sec if hasattr(c, "timestamp_sec") else c[0], "score": c.score if hasattr(c, "score") else c[1],
                      "kind": getattr(c, "kind", "cut")} for c in cuts]
        data = {"cuts": cut_items, "total": len(cut_items)}
        _bound_list(data, "cuts", MAX_CUTS, "Raise threshold (up to 1) to keep only stronger cuts, or pass start_sec/end_sec to list a later range.")
        return {"ok": True, "data": data}

    elif name == "check_frame_cadence":
        engine.probe()
        data = visual_timing.frame_cadence(source)
        _bound_list(data, "ranges", MAX_CADENCE_RANGES, "Ranges are in time order, so the omitted ones are later in the file.")
        return {"ok": True, "data": data}

    elif name == "audio_onsets":
        start_raw = args.get("start_sec")
        end_raw = args.get("end_sec")
        data = engine.audio_onsets(
            start_sec=None if start_raw is None else float(start_raw),
            end_sec=None if end_raw is None else float(end_raw),
            min_strength=float(args.get("min_strength", 0.12)),
            band_hz=args.get("band_hz"),
            compare_to=args.get("compare_to"),
        )
        data = dict(data)
        _bound_list(data, "onsets", MAX_ONSETS, "Raise min_strength (up to 1) to keep only stronger onsets, or pass start_sec/end_sec to list a later range.")
        return {"ok": True, "data": data}

    elif name == "loudness_timeline":
        window_raw = args.get("window_sec")
        bpm_raw = args.get("bpm")
        start_raw = args.get("start_sec")
        end_raw = args.get("end_sec")
        data = engine.loudness_timeline(
            window_sec=None if window_raw is None else float(window_raw),
            bpm=None if bpm_raw is None else float(bpm_raw),
            start_sec=None if start_raw is None else float(start_raw),
            end_sec=None if end_raw is None else float(end_raw),
            ignore_ranges=args.get("ignore_ranges"),
        )
        data = dict(data)
        _bound_list(data, "windows", MAX_LOUDNESS_WINDOWS,
                    "`outliers` and `silence_gaps` cover the whole range; use a larger window_sec, or start_sec/end_sec, to list the rest.")
        return {"ok": True, "data": data}

    elif name == "spectrogram_window":
        data = engine.spectrogram_window(
            start_sec=float(args.get("start_sec")),
            end_sec=float(args.get("end_sec")),
            marks=args.get("marks", []),
        )
        data["tool_file"] = make_tool_file(data["file_path"])
        return {"ok": True, "data": data}

    elif name == "sync_check":
        bpm_raw = args.get("bpm")
        data = engine.sync_check(
            cues=args.get("cues"),
            tolerance_ms=float(args.get("tolerance_ms", 80.0)),
            bpm=None if bpm_raw is None else float(bpm_raw),
        )
        data = dict(data)
        for key in ("cuts", "unmatched_strong_onsets"):
            _bound_list(data, key, MAX_SYNC_ITEMS, "total_cuts and total_cues count everything; use detect_cuts or audio_onsets with start_sec/end_sec to inspect a later range.")
        return {"ok": True, "data": data}
        
    elif name == "watch_for":
        end_raw = args.get("end_sec")
        data = engine.watch_for(
            conditions=args.get("conditions"),
            start_sec=float(args.get("start_sec", 0.0)),
            end_sec=None if end_raw is None else float(end_raw),
            interval_sec=float(args.get("interval_sec", 5.0)),
            cut=float(args.get("cut", 0.5)),
            consecutive=int(args.get("consecutive", 2)),
            cooldown_sec=float(args.get("cooldown_sec", 30.0)),
        )
        return {"ok": True, "data": data}

    elif name == "label_sounds":
        window_raw = args.get("window_sec")
        data = engine.label_sounds(
            start_sec=float(args.get("start_sec")),
            end_sec=float(args.get("end_sec")),
            labels=args.get("labels"),
            window_sec=5.0 if window_raw is None else float(window_raw),
        )
        return {"ok": True, "data": data}

    else:
        return {"ok": False, "error": {"code": "unknown_tool", "message": f"Tool '{name}' not found"}}

def _error_result(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def tool_call_response(req_id: Any, name: Any, arguments: Any) -> str:
    """Run one tools/call and return the JSON-RPC response line (without newline).

    Never raises, and never returns a line over MAX_RESPONSE_BYTES: a larger
    message would close the stdio connection and hide the real cause.
    """
    try:
        outcome = handle_call(name, arguments)
        if outcome.get("ok"):
            result = _success_result(outcome["data"])
        else:
            result = _error_result(json.dumps(outcome["error"]))
        line = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    except Exception as e:
        line = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": _error_result(format_tool_error(e, name))}, default=str)
    if len(line) > MAX_RESPONSE_BYTES:
        text = (
            f"{name} produced a {len(line) / 1e6:.1f} MB response, over the {MAX_RESPONSE_BYTES / 1e6:.0f} MB "
            "MCP message limit. Ask for less: fewer samples or windows, a lower fps, or a shorter start_sec/end_sec range."
        )
        line = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": _error_result(text)})
    return line


def run_stdio_server():
    """Run JSON-RPC stdio loop for MCP clients (Executor, Claude, Cursor)."""
    # Force UTF-8 on Windows stdio
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
            
        try:
            req = json.loads(line)
        except Exception:
            # Never leave a caller waiting forever after malformed input.
            # JSON-RPC parse errors are deliberately generic so malformed input
            # cannot reflect arbitrary payloads back to the client.
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        if not isinstance(req, dict):
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue
            
        req_id = req.get("id")
        method = req.get("method")
        
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agentic-video", "version": _package_version()}
                }
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS_MANIFEST}
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            sys.stdout.write(tool_call_response(req_id, name, arguments) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass

        else:
            # Unknown methods must produce a bounded JSON-RPC response rather
            # than silently consuming the request and causing a client timeout.
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

def _package_version():
    """Installed package version; "dev" when run from a source checkout (how Executor starts it)."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("agentic-video")
    except PackageNotFoundError:
        return "dev"


def main(argv=None):
    """Console-script entry point: `agentic-video` starts the stdio MCP server."""
    args = sys.argv[1:] if argv is None else list(argv)
    if args in (["--version"], ["-V"]):
        print("agentic-video " + _package_version())
    elif args:
        print(
            "usage: agentic-video [--version]\n\n"
            "Starts the agentic-video MCP server on stdin/stdout (add it to an MCP client as: uvx agentic-video).\n"
            "Checker for rendered videos: agentic-video-check --help"
        )
        return 0 if args in (["-h"], ["--help"]) else 2
    else:
        run_stdio_server()
    return 0


if __name__ == "__main__":
    run_stdio_server()
