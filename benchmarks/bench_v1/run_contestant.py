#!/usr/bin/env python3
"""Run one bench-v1 contestant over selected questions; results are resumable."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from .common import CONTESTANTS, DEFAULT_DATA_ROOT, DEFAULT_KEY_PATH, DEFAULT_SEALED_DIR, TRACKS, data_root, default_tool_cwd, default_tool_python, key_path, question_path, read_jsonl, visual_model_environment, scratch_dir
    from .ingest import ingest_questions
except ImportError:  # direct script execution
    from common import CONTESTANTS, DEFAULT_DATA_ROOT, DEFAULT_KEY_PATH, DEFAULT_SEALED_DIR, TRACKS, data_root, default_tool_cwd, default_tool_python, key_path, question_path, read_jsonl, visual_model_environment, scratch_dir
    from ingest import ingest_questions


REPO_ROOT = Path(__file__).resolve().parents[2]
CODEX_TOOL_TIMEOUT_SEC = 600
TOOL_NAMES = [
    "probe_video", "detect_cuts", "inspect_window", "contact_sheet",
    "search_transcript", "transcribe", "build_visual_index", "search_visual",
    "count_repetitions", "count_occurrences",
    "audio_onsets", "loudness_timeline", "spectrogram_window", "sync_check",
]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _numeric_string(value: Any) -> int | float | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


_TIME_TOKEN = r"\d+(?::[0-5]\d){0,2}(?:\.\d+)?"


def _time_string_seconds(value: Any) -> int | float | None:
    """Parse '52', '52.5', 'mm:ss' or 'hh:mm:ss' strings as seconds."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not re.fullmatch(_TIME_TOKEN, text):
        return None
    number = 0.0
    for part in text.split(":"):
        number = number * 60 + float(part)
    return int(number) if number.is_integer() else number


def _interval_string(value: Any) -> list[int | float] | None:
    """Parse unambiguous interval strings such as '7-8', '01:43 - 02:03' or '[30, 31]'."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = (
        re.fullmatch(rf"(?:interval_seconds\s*:\s*)?\[\s*({_TIME_TOKEN})\s*,\s*({_TIME_TOKEN})\s*\]", text)
        or re.fullmatch(rf"({_TIME_TOKEN})\s*(?:-|\u2013|\u2014|to)\s*({_TIME_TOKEN})", text)
    )
    if not match:
        return None
    parts = [_time_string_seconds(group) for group in match.groups()]
    return parts if all(part is not None for part in parts) else None


def _normalize_answer(value: dict[str, Any], answer_format: str) -> dict[str, Any]:
    """Normalize unambiguous format variants without changing answer content."""
    normalized = dict(value)
    answer = normalized.get("answer")
    if isinstance(normalized.get("confidence"), str):
        normalized["confidence"] = _numeric_string(normalized["confidence"])
    evidence = normalized.get("evidence")
    if isinstance(evidence, (list, dict)):
        # Gemini may return timestamped evidence objects. Keep their complete
        # structure while representing it in the runner's required string field.
        normalized["evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if answer_format in {"integer", "integer_or_zero"}:
        parsed = _numeric_string(answer)
        if parsed is not None:
            normalized["answer"] = parsed
    elif answer_format in {"number_seconds", "number_seconds_or_not_found"}:
        parsed = _numeric_string(answer)
        if parsed is None:
            parsed = _time_string_seconds(answer)
        if parsed is not None:
            normalized["answer"] = parsed
    elif answer_format == "interval_seconds" and isinstance(answer, list):
        parsed = [
            (_numeric_string(part) if _numeric_string(part) is not None else _time_string_seconds(part))
            if isinstance(part, str) else part
            for part in answer
        ]
        if len(parsed) == 2 and all(part is not None for part in parsed):
            normalized["answer"] = parsed
    elif answer_format == "interval_seconds":
        parsed = _interval_string(answer)
        if parsed is not None:
            normalized["answer"] = parsed
    return normalized


def _repair_preserves_content(original: dict[str, Any] | None, repaired: dict[str, Any] | None, answer_format: str) -> bool:
    """Accept format repair only when it keeps the parsed answer and evidence unchanged."""
    if not isinstance(original, dict) or not isinstance(repaired, dict):
        return False
    before = _normalize_answer(original, answer_format)
    after = _normalize_answer(repaired, answer_format)
    return (
        "answer" in before and "answer" in after
        and "evidence" in before and "evidence" in after
        and before["answer"] == after["answer"]
        and before["evidence"] == after["evidence"]
    )


def _validate_answer(value: dict[str, Any], answer_format: str) -> tuple[bool, str | None]:
    if set(value) - {"answer", "confidence", "evidence"}:
        return False, "unexpected JSON fields"
    if "answer" not in value or "confidence" not in value or "evidence" not in value:
        return False, "required field missing"
    answer = value["answer"]
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
        return False, "confidence must be between 0 and 1"
    if not isinstance(value["evidence"], str):
        return False, "evidence must be a string"
    if isinstance(answer, str) and answer.strip().lower().replace("-", "_").replace(" ", "_") in {"not_found", "notfound", "none"}:
        return (False, "count answers must be numeric") if answer_format in {"integer", "integer_or_zero"} else (True, None)
    if answer_format in {"integer", "integer_or_zero"}:
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)) or not float(answer).is_integer():
            return False, "answer must be an integer"
    elif answer_format == "interval_seconds":
        if not isinstance(answer, list) or len(answer) != 2:
            return False, "answer must be a two-number interval"
        if any(isinstance(part, bool) or not isinstance(part, (int, float)) or not math.isfinite(float(part)) for part in answer) or float(answer[1]) < float(answer[0]):
            return False, "answer must be an ordered numeric interval"
    elif answer_format in {"number_seconds", "number_seconds_or_not_found"}:
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
            return False, "answer must be a number of seconds"
    else:
        return False, f"unknown answer format: {answer_format}"
    return True, None


def _usage_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [event, event.get("data"), event.get("turn"), event.get("result")]
    for item in candidates:
        if isinstance(item, dict) and isinstance(item.get("usage"), dict):
            return item["usage"]
    if event.get("type") in {"turn.completed", "turn.failed"}:
        details = event.get("token_usage") or event.get("tokenUsage")
        if isinstance(details, dict):
            return details
    return None


def _tool_result_error(item: dict[str, Any], event: dict[str, Any]) -> bool | None:
    result = item.get("result") or item.get("output")
    if isinstance(result, dict):
        if result.get("isError") is True or result.get("is_error") is True:
            return True
        if result.get("error"):
            return True
    if item.get("error") or event.get("error"):
        return True
    status = str(item.get("status") or event.get("status") or "").lower()
    if status in {"failed", "error", "cancelled", "canceled", "rejected"}:
        return True
    if status in {"completed", "success", "succeeded"}:
        return False
    event_type = str(event.get("type", "")).lower()
    if event_type in {"item.completed", "tool.completed", "mcp_tool_call.completed"}:
        return False
    if event_type in {"item.failed", "tool.failed", "mcp_tool_call.failed"}:
        return True
    return None


def _tool_error_detail(item: dict[str, Any], event: dict[str, Any]) -> str | None:
    result = item.get("result") or item.get("output")
    candidates = [
        item.get("error"),
        event.get("error"),
        result.get("error") if isinstance(result, dict) else None,
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()[:1000]
        if isinstance(value, dict):
            for name in ("message", "detail", "text"):
                detail = value.get(name)
                if isinstance(detail, str) and detail.strip():
                    return detail.strip()[:1000]
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            texts = [
                part.get("text", "").strip()
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
            ]
            if texts:
                return "\n".join(texts)[:1000]
    for value in (item.get("message"), event.get("message")):
        if isinstance(value, str) and value.strip():
            return value.strip()[:1000]
    return None


def _codex_mcp_config(tool_cwd: Path, tool_python: Path) -> list[str]:
    """Shared MCP settings for contestant and overhead Codex invocations."""
    visual_env = visual_model_environment(tool_python)
    return [
        f'mcp_servers.agentic_video.command={json.dumps(str(tool_python))}',
        f'mcp_servers.agentic_video.args={json.dumps(["-m", "src.executor.mcp_server"])}',
        f'mcp_servers.agentic_video.cwd={json.dumps(str(tool_cwd))}',
        f'mcp_servers.agentic_video.enabled_tools={json.dumps(TOOL_NAMES)}',
        f'mcp_servers.agentic_video.env.HF_HOME={json.dumps(visual_env["HF_HOME"])}',
        f'mcp_servers.agentic_video.env.HF_HUB_OFFLINE={json.dumps(visual_env["HF_HUB_OFFLINE"])}',
        f'mcp_servers.agentic_video.tool_timeout_sec={CODEX_TOOL_TIMEOUT_SEC}',
        'mcp_servers.agentic_video.default_tools_approval_mode="approve"',
        'mcp_servers.agentic_video.startup_timeout_sec=45',
        'approval_policy="never"',
        'sandbox_mode="read-only"',
    ]


def _codex_events(stdout: str) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []
    messages: list[str] = []
    usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if _usage_from_event(event) is not None:
            usage = _usage_from_event(event)
        item = event.get("item") if isinstance(event.get("item"), dict) else event.get("data", {}).get("item") if isinstance(event.get("data"), dict) else None
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).lower()
            if item_type in {"agent_message", "assistant_message", "message"}:
                content = item.get("text") or item.get("content")
                if isinstance(content, str):
                    messages.append(content)
            if "mcp_tool_call" in item_type or (item_type == "tool_call" and (item.get("server") or item.get("mcp_server"))):
                call_id = item.get("call_id") or item.get("callId") or item.get("id")
                tool_name = item.get("tool") or item.get("tool_name") or item.get("name")
                identity = str(call_id or f"{item.get('server') or item.get('mcp_server')}|{tool_name}|{json.dumps(item.get('arguments') or item.get('input'), sort_keys=True, default=str)}")
                call = calls_by_id.get(identity)
                if call is None:
                    call = {
                        "server": item.get("server") or item.get("mcp_server"),
                        "tool_name": tool_name,
                        "arguments": item.get("arguments") or item.get("input"),
                        "call_id": call_id,
                        "status": "failed",
                        "outcome_detail": "no successful completion event",
                    }
                    calls_by_id[identity] = call
                    tool_calls.append(call)
                else:
                    call.update({
                        "server": item.get("server") or call.get("server"),
                        "tool_name": tool_name or call.get("tool_name"),
                        "arguments": item.get("arguments") or item.get("input") or call.get("arguments"),
                    })
                error = _tool_result_error(item, event)
                if error is not None:
                    call["status"] = "failed" if error else "succeeded"
                    call["outcome_detail"] = (_tool_error_detail(item, event) or "tool error") if error else "completed"
            elif "mcp_tool_result" in item_type or "mcp_tool_output" in item_type or item_type == "tool_result":
                call_id = str(item.get("call_id") or item.get("callId") or item.get("id") or "")
                call = calls_by_id.get(call_id)
                if call is not None:
                    error = _tool_result_error(item, event)
                    if error is not None:
                        call["status"] = "failed" if error else "succeeded"
                        call["outcome_detail"] = (_tool_error_detail(item, event) or "tool error") if error else "completed"
        if event.get("type") in {"turn.completed", "turn.failed"}:
            final = event.get("last_assistant_message") or event.get("output_text")
            if isinstance(final, str):
                messages.append(final)
    return events, messages[-1] if messages else "", usage, tool_calls


def _sandbox_profile(sealed_dir: Path, data_dir: Path, video_path: Path | None, private_code_dir: Path | None = None) -> str:
    def q(path: Path) -> str:
        return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    question_file = data_dir / "questions.jsonl"
    runs_dir = data_dir / "runs"
    videos_dir = data_dir / "videos"
    rules = [
        "(version 1)",
        "(allow default)",
    ]
    # Include both locations even when the configured key is outside the
    # default sealed tree. Resolve symlinks before constructing deny rules.
    sealed_paths = {sealed_dir.resolve(), DEFAULT_KEY_PATH.resolve().parent, key_path().resolve().parent}
    for path in sorted(sealed_paths):
        rules.extend((
            f'(deny file-read* (subpath "{q(path)}"))',
            f'(deny file-write* (subpath "{q(path)}"))',
        ))
    for path in sorted({DEFAULT_KEY_PATH.resolve(), key_path().resolve()}):
        rules.extend((
            f'(deny file-read* (literal "{q(path)}"))',
            f'(deny file-write* (literal "{q(path)}"))',
        ))
    rules.extend([
        # Deny the question bank, prior answers, and all sibling videos.
        f'(deny file-read* (literal "{q(question_file)}"))',
        f'(deny file-write* (literal "{q(question_file)}"))',
        f'(deny file-read* (subpath "{q(runs_dir)}"))',
        f'(deny file-write* (subpath "{q(videos_dir)}"))',
    ])
    # Earlier runs written with --out sit beside runs/ (runs-v2, runs-fresh1) and hold prior answers too.
    if data_dir.is_dir():
        for sibling in sorted(data_dir.glob("runs*")):
            if sibling.is_dir() and sibling.resolve() != runs_dir.resolve():
                rules.append(f'(deny file-read* (subpath "{q(sibling)}"))')
    if videos_dir.is_dir():
        for sibling in videos_dir.iterdir():
            if sibling.is_file() and (video_path is None or sibling.resolve() != video_path.resolve()):
                rules.append(f'(deny file-read* (literal "{q(sibling)}"))')
    if private_code_dir is not None:
        rules.append(f'(deny file-read* (subpath "{q(private_code_dir)}"))')
    return "\n".join(rules)


def _codex_prompt(question: dict[str, Any], video_alias: str) -> str:
    format_hint = {
        "interval_seconds": '"answer": [start_seconds, end_seconds]',
        "number_seconds": '"answer": seconds',
        "number_seconds_or_not_found": '"answer": seconds OR "not_found"',
        "integer": '"answer": an integer',
        "integer_or_zero": '"answer": an integer (0 means none)',
    }.get(question.get("answer_format"), '"answer": a number, interval, or "not_found"')
    negative_hint = "For counts, return the integer 0 when none occur; do not use not_found." if question.get("track") == "count" else "For a negative result use the string \"not_found\"."
    return f"""You are answering one blind video benchmark question.

Use the video-analysis MCP tools available in this run. Choose the matching tools for the question, and base the answer on their returned evidence. You may call only these agentic-video tools: {', '.join(TOOL_NAMES)}. Do not use shell commands or any other tool to inspect files.

The workspace contains exactly a question file and a symlink named video.mp4. For tool calls, pass this source path: {video_alias}
The benchmark answer key is sealed at ~/.local/share/agentic-video-sealed and is strictly off limits. Do not attempt to read it, search for it, or infer the answer from files outside this run.

Question: {question['question']}
Expected answer shape: {{{format_hint}, "confidence": 0 to 1, "evidence": "short evidence-based explanation"}}
Return one JSON object only, with exactly the keys answer, confidence, evidence. {negative_hint} Give seconds, not a clock timestamp. Use an interval when the question asks for a span. Do not guess if the video tools do not support the answer."""


def _repair_prompt(raw: str, answer_format: str) -> str:
    hint = {
        "interval_seconds": "number | [start_seconds, end_seconds] | not_found",
        "number_seconds": "number | not_found",
        "number_seconds_or_not_found": "number | not_found",
        "integer": "integer (0 means none; do not use not_found)",
        "integer_or_zero": "integer (0 means none; do not use not_found)",
    }.get(answer_format, "number | [start_seconds, end_seconds] | not_found")
    negative_hint = "For counts, return integer 0 when none occur; do not use not_found." if answer_format in {"integer", "integer_or_zero"} else "Use the string not_found only when the event is absent."
    return (
        "Convert the following prior answer to one valid JSON object only. Preserve its answer and evidence exactly; do not re-evaluate or change the answer. "
        f"Use keys answer, confidence, evidence; answer shape: {hint}. Confidence must be a JSON number from 0 to 1. "
        f"If prior confidence is a word or otherwise nonnumeric, use 0.5. {negative_hint} Do not add markdown.\n\nPrior answer:\n" + raw[:12000]
    )


def _run_process(command: list[str], *, cwd: Path, env: dict[str, str], prompt: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(cwd), env=env, timeout=timeout_sec, check=False,
    )


def _contestant_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        upper = name.upper()
        key_location = re.search(
            r"(?:^|_)(?:ANSWER_KEY|BENCH_KEY|SEALED_KEY)(?:_|$)|(?:^|_)KEY_(?:PATH|FILE|DIR|DIRECTORY)(?:_|$)",
            upper,
        )
        if upper in {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"} or key_location:
            env.pop(name, None)
    return env


def _aggregate_usage(*usages: dict[str, Any] | None) -> dict[str, Any] | None:
    present = [u for u in usages if isinstance(u, dict)]
    if not present:
        return None
    totals: dict[str, float] = {}
    requests = []
    for usage in present:
        requests.append(usage)
        for key, value in usage.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0) + value
    input_tokens = totals.get("total_input_tokens", totals.get("input_tokens", 0))
    output_tokens = totals.get("total_output_tokens", totals.get("output_tokens", 0))
    totals["total_tokens"] = totals.get("total_tokens", input_tokens + output_tokens)
    if totals.get("usage_missing_requests"):
        # Some request returned no usage, so the total is unknown rather than a low sum.
        totals["total_tokens"] = None
        totals["billable_tokens"] = None
    totals["requests"] = requests
    return totals


def _usage_number(usage: dict[str, Any] | None, *names: str) -> float | None:
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _account_codex_tokens(usage: dict[str, Any] | None, fixed_overhead_input_tokens: int | None, invocations: int) -> dict[str, Any] | None:
    if not usage:
        return None
    value = dict(usage)
    input_tokens = _usage_number(value, "total_input_tokens", "input_tokens")
    output_tokens = _usage_number(value, "total_output_tokens", "output_tokens")
    total_tokens = _usage_number(value, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if input_tokens is not None:
        value["total_input_tokens"] = int(input_tokens)
    if total_tokens is not None:
        value["total_tokens"] = int(total_tokens)
    if fixed_overhead_input_tokens is not None:
        overhead = fixed_overhead_input_tokens * max(1, invocations)
        value["fixed_overhead_input_tokens"] = overhead
        if input_tokens is not None:
            value["adjusted_input_tokens"] = max(0, int(input_tokens - overhead))
        if total_tokens is not None:
            value["adjusted_total_tokens"] = max(0, int(total_tokens - overhead))
    return value


def _tool_call_counts(calls: list[dict[str, Any]]) -> tuple[int, int]:
    succeeded = sum(call.get("status") == "succeeded" for call in calls)
    return succeeded, len(calls) - succeeded


def _gemini_cost(usage: dict[str, Any] | None) -> float | None:
    if not usage or usage.get("usage_missing_requests"):
        return None
    input_tokens = next((usage.get(k) for k in ("total_input_tokens", "totalInputTokens", "input_tokens", "inputTokens", "prompt_tokens") if isinstance(usage.get(k), (int, float))), 0)
    output_tokens = next((usage.get(k) for k in ("total_output_tokens", "totalOutputTokens", "output_tokens", "outputTokens", "completion_tokens") if isinstance(usage.get(k), (int, float))), 0)
    tool_tokens = next((usage.get(k) for k in ("total_tool_use_tokens", "tool_use_prompt_token_count", "totalToolUseTokens") if isinstance(usage.get(k), (int, float))), 0)
    thought_tokens = next((usage.get(k) for k in ("total_thought_tokens", "thoughts_token_count", "totalThoughtTokens") if isinstance(usage.get(k), (int, float))), 0)
    if not input_tokens and not output_tokens and not tool_tokens and not thought_tokens and not usage.get("total_tokens"):
        return None
    if not input_tokens and not output_tokens and not tool_tokens and not thought_tokens:
        input_tokens = usage.get("total_tokens", 0)
    billable_input = float(input_tokens or 0) + float(tool_tokens or 0)
    billable_output = float(output_tokens or 0) + float(thought_tokens or 0)
    return (billable_input * 0.75 + billable_output * 3.75) / 1_000_000


def _read_questions(path: Path, selectors: list[str] | None) -> list[dict[str, Any]]:
    questions = read_jsonl(path)
    if not selectors:
        return questions
    ids: set[str] = set()
    tracks: set[str] = set()
    for selector in selectors:
        for item in selector.split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("track:"):
                tracks.add(item.split(":", 1)[1])
            elif item.startswith("track="):
                tracks.add(item.split("=", 1)[1])
            elif item in TRACKS:
                tracks.add(item)
            else:
                ids.add(item)
    unknown = tracks - set(TRACKS)
    if unknown:
        raise ValueError(f"unknown track selector: {', '.join(sorted(unknown))}")
    selected = [q for q in questions if q.get("id") in ids or q.get("track") in tracks]
    missing = ids - {q["id"] for q in selected}
    if missing:
        raise ValueError(f"question ids not found: {', '.join(sorted(missing))}")
    return selected


def _run_gemini(script: Path, mode: str, model: str, video_path: Path | None, prompt: str, timeout_sec: int, previous_id: str | None = None, data_dir: Path | None = None) -> tuple[dict[str, Any], str]:
    node = shutil.which("node")
    sandbox_exec = shutil.which("sandbox-exec")
    if not node:
        raise RuntimeError("node was not found on PATH")
    if not sandbox_exec:
        raise RuntimeError("sandbox-exec is required to deny access to the sealed key")
    request: dict[str, Any] = {"operation": "repair" if video_path is None else "ask", "model": model, "mode": mode, "prompt": prompt}
    if video_path is not None:
        request["video_path"] = str(video_path)
    if previous_id:
        request["previous_interaction_id"] = previous_id
    env = _contestant_env()
    # Only the Gemini adapter itself gets the API key; contestant agents never do.
    if os.environ.get("GEMINI_API_KEY"):
        env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    profile = _sandbox_profile(DEFAULT_SEALED_DIR, data_dir or data_root(), video_path)
    process = subprocess.run(
        [sandbox_exec, "-p", profile, node, str(script)], input=json.dumps(request), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec,
        cwd=str(script.parent), env=env, check=False,
    )
    try:
        payload = json.loads(process.stdout.strip())
    except json.JSONDecodeError:
        payload = {"status": "error", "error": "Gemini adapter returned no JSON result"}
    if process.returncode and payload.get("status") == "ok":
        payload["status"] = "error"
        payload["error"] = f"Gemini adapter exited with code {process.returncode}"
    if process.stderr:
        payload["adapter_stderr"] = process.stderr[-1000:]
    return payload, process.stderr


def _codex_run(question: dict[str, Any], video_path: Path, model: str, effort: str, timeout_sec: int, tool_cwd: Path, tool_python: Path, sealed_dir: Path, data_dir: Path, fixed_overhead_input_tokens: int | None = None) -> dict[str, Any]:
    codex = shutil.which("codex")
    sandbox_exec = shutil.which("sandbox-exec")
    if not codex:
        return {"status": "error", "failure_type": "codex_missing", "error": "codex executable was not found"}
    if not sandbox_exec:
        return {"status": "error", "failure_type": "sandbox_missing", "error": "sandbox-exec is required to deny access to the sealed key"}
    temporary = scratch_dir("bench-v1-contestant-")
    workspace = temporary / "case"
    workspace.mkdir()
    video_alias = workspace / "video.mp4"
    video_alias.symlink_to(video_path.resolve())
    safe_question = {
        "id": question["id"], "track": question["track"],
        "question": question["question"], "answer_format": question["answer_format"],
        "video_path": "./video.mp4",
    }
    (workspace / "question.json").write_text(json.dumps(safe_question, ensure_ascii=False, indent=2), encoding="utf-8")

    config = _codex_mcp_config(tool_cwd, tool_python) + [
        f'model_reasoning_effort={json.dumps(effort)}',
    ]
    private_code_dir = REPO_ROOT if REPO_ROOT.resolve() != tool_cwd.resolve() else None
    profile = _sandbox_profile(sealed_dir, data_dir, video_path, private_code_dir)
    command = [sandbox_exec, "-p", profile, codex, "exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only", "--model", model, "-C", str(workspace)]
    for item in config:
        command.extend(["-c", item])
    prompt = _codex_prompt(question, str(video_alias))
    started = time.monotonic()
    env = _contestant_env()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = _run_process(command, cwd=workspace, env=env, prompt=prompt, timeout_sec=timeout_sec)
        events, raw, usage, calls = _codex_events(process.stdout)
        initial_raw = raw
        retry_usage = None
        parsed = _parse_json_object(raw)
        if parsed:
            parsed = _normalize_answer(parsed, question["answer_format"])
        initial_parsed = dict(parsed) if parsed else None
        valid, validation_error = _validate_answer(parsed, question["answer_format"]) if parsed else (False, "no JSON object in final response")
        parse_retries = 0
        if not valid:
            repair_prompt = _repair_prompt(raw or "No usable final answer was returned.", question["answer_format"])
            retry_process = _run_process(command, cwd=workspace, env=env, prompt=repair_prompt, timeout_sec=timeout_sec)
            retry_events, retry_raw, retry_usage, retry_calls = _codex_events(retry_process.stdout)
            parse_retries = 1
            events.extend(retry_events)
            calls.extend(retry_calls)
            parsed = _parse_json_object(retry_raw)
            if parsed:
                parsed = _normalize_answer(parsed, question["answer_format"])
            retry_valid, retry_error = _validate_answer(parsed, question["answer_format"]) if parsed else (False, "no JSON object in repair response")
            if retry_valid and not _repair_preserves_content(initial_parsed, parsed, question["answer_format"]):
                # Same rule as the Gemini contestants: repair may fix format only.
                retry_valid, retry_error = False, "format repair changed answer or evidence"
            if retry_valid:
                raw, validation_error, valid = retry_raw, None, True
            else:
                raw, validation_error, valid = retry_raw or raw, retry_error, False
        if process.returncode:
            successes, failures = _tool_call_counts(calls)
            return {
                "status": "error", "failure_type": "codex_exit", "error": f"codex exited with code {process.returncode}",
                "answer": parsed.get("answer") if valid else None,
                "confidence": parsed.get("confidence") if valid else None,
                "evidence": parsed.get("evidence") if valid else None,
                "raw_answer_text": raw[:8000], "initial_raw_answer_text": initial_raw[:8000],
                "usage": _account_codex_tokens(_aggregate_usage(usage, retry_usage), fixed_overhead_input_tokens, 1 + parse_retries),
                "tool_calls": calls, "tool_calls_succeeded": successes, "tool_calls_failed": failures,
                "parse_retries": parse_retries, "wall_time_s": round(time.monotonic() - started, 3),
            }
        if not valid:
            successes, failures = _tool_call_counts(calls)
            return {
                "status": "error", "failure_type": "json_parse", "error": validation_error,
                "raw_answer_text": raw[:8000], "initial_raw_answer_text": initial_raw[:8000],
                "usage": _account_codex_tokens(_aggregate_usage(usage, retry_usage), fixed_overhead_input_tokens, 1 + parse_retries),
                "tool_calls": calls, "tool_calls_succeeded": successes, "tool_calls_failed": failures,
                "parse_retries": parse_retries, "wall_time_s": round(time.monotonic() - started, 3),
            }
        successes, failures = _tool_call_counts(calls)
        return {
            "status": "ok", "answer": parsed["answer"], "confidence": parsed["confidence"], "evidence": parsed["evidence"],
            "raw_answer_text": raw[:8000], "initial_raw_answer_text": initial_raw[:8000],
            "usage": _account_codex_tokens(_aggregate_usage(usage, retry_usage), fixed_overhead_input_tokens, 1 + parse_retries),
            "tool_calls": calls, "tool_calls_succeeded": successes, "tool_calls_failed": failures,
            "parse_retries": parse_retries, "wall_time_s": round(time.monotonic() - started, 3),
            "exit_code": process.returncode, "event_count": len(events),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "failure_type": "timeout", "error": f"codex exceeded {timeout_sec} seconds", "wall_time_s": round(time.monotonic() - started, 3)}
    except Exception as exc:
        return {"status": "error", "failure_type": "runner_error", "error": f"{type(exc).__name__}: {exc}", "wall_time_s": round(time.monotonic() - started, 3)}
    finally:
        trash = shutil.which("trash")
        if trash:
            subprocess.run([trash, str(temporary)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _measure_codex_overhead(model: str, effort: str, timeout_sec: int, tool_cwd: Path, tool_python: Path, sealed_dir: Path, data_dir: Path) -> int:
    """Measure Codex's input-token baseline with tools registered but no video."""
    codex = shutil.which("codex")
    sandbox_exec = shutil.which("sandbox-exec")
    if not codex or not sandbox_exec:
        raise RuntimeError("Codex and sandbox-exec are required to measure token overhead")
    temporary = scratch_dir("bench-v1-overhead-")
    workspace = temporary / "case"
    workspace.mkdir()
    config = _codex_mcp_config(tool_cwd, tool_python) + [
        f'model_reasoning_effort={json.dumps(effort)}',
    ]
    private_code_dir = REPO_ROOT if REPO_ROOT.resolve() != tool_cwd.resolve() else None
    profile = _sandbox_profile(sealed_dir, data_dir, workspace / "no-video.mp4", private_code_dir)
    command = [sandbox_exec, "-p", profile, codex, "exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only", "--model", model, "-C", str(workspace)]
    for item in config:
        command.extend(["-c", item])
    env = _contestant_env()
    try:
        process = _run_process(
            command, cwd=workspace, env=env,
            prompt="Answer the arithmetic question with the number only: what is 2 + 2? Do not call tools. There is no video.",
            timeout_sec=timeout_sec,
        )
        _, _, usage, calls = _codex_events(process.stdout)
        if process.returncode:
            raise RuntimeError(f"Codex overhead calibration exited with code {process.returncode}")
        if calls:
            raise RuntimeError("Codex overhead calibration called a video tool despite having no video")
        input_tokens = _usage_number(usage, "total_input_tokens", "input_tokens")
        if input_tokens is None:
            raise RuntimeError("Codex overhead calibration returned no input-token usage")
        return int(input_tokens)
    finally:
        trash = shutil.which("trash")
        if trash:
            subprocess.run([trash, str(temporary)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _cannot_inspect_video(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        r"(?:cannot|can't|could not|couldn't|unable to|not able to)\s+(?:inspect|access|view|watch|analy[sz]e)\s+(?:the\s+)?(?:video|clip)",
        r"(?:video|clip)\s+(?:could not|couldn't|cannot|can't|wasn't able to)\s+be\s+(?:inspected|accessed|viewed|analyzed|analysed)",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _classify_harness_failure(result: dict[str, Any], contestant: str) -> None:
    """Classify tool-access failures without replacing an existing runner error."""
    if result.get("status") != "ok":
        return
    evidence_text = "\n".join(str(result.get(name) or "") for name in ("initial_raw_answer_text", "raw_answer_text", "error"))
    if _cannot_inspect_video(evidence_text):
        result.update(status="error", failure_type="harness", error="contestant reported that it could not inspect the video")
    elif contestant in {"ours-codex", "gemini-agentic"} and result.get("tool_calls_succeeded", 0) == 0:
        result.update(status="error", failure_type="harness", error="no video-analysis tool call succeeded")


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _record_timing(result: dict[str, Any], contestant: str) -> None:
    """Record answer, preparation, and end-to-end durations on each result."""
    measured_total = _duration_seconds(result.get("wall_time_s"))
    if contestant == "ours-codex":
        answer_time = measured_total
        preparation_parts = [
            duration for duration in (
                _duration_seconds(result.get("ingest_time_s")),
                _duration_seconds(result.get("visual_index_time_s")),
            ) if duration is not None
        ]
        preparation_time = round(sum(preparation_parts), 3) if preparation_parts else None
        if answer_time is not None and preparation_time is not None:
            end_to_end = answer_time + preparation_time
        elif answer_time is None:
            # An ingest failure has preparation time but no contestant answer.
            end_to_end = preparation_time
        else:
            end_to_end = None
    else:
        preparation_time = _duration_seconds(result.get("upload_processing_s"))
        reported_answer_time = _duration_seconds(result.get("gemini_answer_time_s"))
        split_complete = result.get("gemini_timing_complete", True)
        if preparation_time is None or reported_answer_time is None or not split_complete:
            # Adapter failures of any type may happen before it can report a
            # complete preparation/answer split. Keep the outer duration as
            # end-to-end, but never infer answer time by subtraction.
            answer_time = None
        else:
            answer_time = reported_answer_time
        end_to_end = measured_total

    result["answer_time_s"] = round(answer_time, 3) if answer_time is not None else None
    result["preparation_time_s"] = round(preparation_time, 3) if preparation_time is not None else None
    result["end_to_end_time_s"] = round(end_to_end, 3) if end_to_end is not None else None
    result["wall_time_s"] = result["answer_time_s"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contestant", required=True, choices=CONTESTANTS)
    parser.add_argument("--questions", nargs="*", help="Question ids, comma-separated ids, or track selectors: moment, search, count")
    parser.add_argument("--question-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Output directory; defaults to the standard runs/<contestant> directory")
    parser.add_argument("--model", default=None, help="Codex model; default gpt-6-luna")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--gemini-model", default="gemini-3.8-flash")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--codex-overhead-file", type=Path, default=None, help="Read/write the one-time Codex input-token overhead calibration")
    parser.add_argument("--rerun", action="store_true", help="Rerun ids that already have a result file")
    args = parser.parse_args()

    root = data_root()
    questions_file = (args.question_file or question_path()).expanduser()
    questions = _read_questions(questions_file, args.questions)
    results_dir = (args.out or root / "runs" / args.contestant).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    tool_cwd = default_tool_cwd()
    tool_python = default_tool_python()
    skipped = 0
    completed = 0
    questions_to_run = []
    for question in questions:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", question["id"])
        if (results_dir / f"{safe_id}.json").exists() and not args.rerun:
            skipped += 1
        else:
            questions_to_run.append(question)

    ingest_records: dict[str, dict[str, Any]] = {}
    ingest_failed_paths: set[str] = set()
    ingest_error: str | None = None
    if args.contestant == "ours-codex" and questions_to_run:
        if not tool_cwd.is_dir() or not tool_python.exists():
            ingest_error = f"MCP server path or requested venv Python is missing: {tool_cwd} / {tool_python}"
        else:
            ingest_started = time.monotonic()
            try:
                ingest_records = ingest_questions(
                    questions_to_run, data_dir=root, tool_cwd=tool_cwd, tool_python=tool_python,
                )
            except Exception as exc:
                ingest_error = f"{type(exc).__name__}: {exc}"
                elapsed = round(time.monotonic() - ingest_started, 3)
                ingest_records = {
                    str(Path(question["video_path"]).expanduser().resolve()): {
                        "status": "error", "error": ingest_error, "last_ingest_time_s": elapsed,
                    }
                    for question in questions_to_run
                }

        for question in questions_to_run:
            video_path = Path(question["video_path"]).expanduser().resolve()
            video_key = str(video_path)
            record = ingest_records.get(video_key)
            if ingest_error is not None:
                record = dict(record or {})
                record.update(status="error", error=ingest_error)
                ingest_records[video_key] = record
            elif record is None:
                record = {"status": "error", "error": "ingest returned no record for selected video"}
                ingest_records[video_key] = record
            if record.get("status") == "ok":
                continue

            ingest_failed_paths.add(video_key)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", question["id"])
            result_path = results_dir / f"{safe_id}.json"
            failure_result: dict[str, Any] = {
                "id": question["id"], "contestant": args.contestant,
                "track": question["track"], "dataset_split": question.get("split"),
                "model": args.model or "gpt-6-luna",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "error", "failure_type": "ingest", "answer": None,
                "tool_timeout_sec": CODEX_TOOL_TIMEOUT_SEC,
                "error": str(record.get("error") or "transcript ingest failed"),
                "wall_time_s": None,
                "ingest_time_s": _duration_seconds(record.get("last_ingest_time_s")),
                "visual_index_time_s": _duration_seconds(record.get("last_visual_index_time_s")),
                "visual_index_status": record.get("visual_index_status"),
                "visual_index_error": record.get("visual_index_error"),
                "ingest_cache_hit_before": record.get("last_cache_hit_before", record.get("cache_hit_before")),
            }
            _record_timing(failure_result, args.contestant)
            result_path.write_text(json.dumps(failure_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            completed += 1
            print(f"{question['id']}: error · ingest failed; answer not run", flush=True)

        if ingest_failed_paths:
            questions_to_run = [
                question for question in questions_to_run
                if str(Path(question["video_path"]).expanduser().resolve()) not in ingest_failed_paths
            ]
        ingest_total = sum(float(record.get("last_ingest_time_s") or 0) for record in ingest_records.values())
        visual_total = sum(float(record.get("last_visual_index_time_s") or 0) for record in ingest_records.values())
        successful_count = sum(record.get("status") == "ok" for record in ingest_records.values())
        print(
            f"Transcript ingest completed for {successful_count}/{len(ingest_records)} distinct video(s) "
            f"in {ingest_total:.3f}s; visual indexing took {visual_total:.3f}s "
            f"(both reported separately from contestant wall time).",
            flush=True,
        )
    codex_overhead: int | None = None
    if args.contestant == "ours-codex" and questions_to_run:
        overhead_path = (args.codex_overhead_file or root / "runs" / "codex-overhead.json").expanduser()
        if overhead_path.exists():
            payload = json.loads(overhead_path.read_text(encoding="utf-8"))
            codex_overhead = int(payload["fixed_overhead_input_tokens"])
        else:
            codex_overhead = _measure_codex_overhead(
                args.model or "gpt-6-luna", args.effort, args.timeout_sec,
                tool_cwd, tool_python, sealed_dir=DEFAULT_SEALED_DIR, data_dir=root,
            )
            overhead_path.parent.mkdir(parents=True, exist_ok=True)
            overhead_path.write_text(json.dumps({
                "fixed_overhead_input_tokens": codex_overhead,
                "model": args.model or "gpt-6-luna", "effort": args.effort,
                "tool_timeout_sec": CODEX_TOOL_TIMEOUT_SEC,
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "question": "trivial arithmetic; no video supplied",
            }, indent=2) + "\n", encoding="utf-8")
            print(f"Measured Codex fixed input-token overhead: {codex_overhead}", flush=True)
    node_script = Path(__file__).with_name("gemini-video.mjs")
    sealed_dir = DEFAULT_SEALED_DIR

    for question in questions_to_run:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", question["id"])
        result_path = results_dir / f"{safe_id}.json"
        result: dict[str, Any] = {
            "id": question["id"], "contestant": args.contestant,
            "track": question["track"], "dataset_split": question.get("split"),
            "model": args.model if args.contestant == "ours-codex" else args.gemini_model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        video_path = Path(question["video_path"]).expanduser()
        if args.contestant == "ours-codex":
            ingest_record = ingest_records[str(video_path.resolve())]
            result.update({
                "tool_timeout_sec": CODEX_TOOL_TIMEOUT_SEC,
                "ingest_time_s": ingest_record.get("ingest_time_s"),
                "visual_index_time_s": ingest_record.get("visual_index_time_s"),
                "visual_index_status": ingest_record.get("visual_index_status"),
                "visual_index_error": ingest_record.get("visual_index_error"),
                "ingest_cache_hit_before": ingest_record.get("cache_hit_before"),
            })
        gemini_started: float | None = None
        gemini_answer_ms: list[float] = []
        gemini_cleanup_ms: list[float] = []
        gemini_timing_complete = False
        if not video_path.is_file():
            result.update(status="error", failure_type="missing_video", error="video path does not exist", wall_time_s=0.0)
        else:
            try:
                if args.contestant == "ours-codex":
                    result.update(_codex_run(
                        question, video_path, args.model or "gpt-6-luna", args.effort, args.timeout_sec,
                        tool_cwd, tool_python, sealed_dir, root, codex_overhead,
                    ))
                else:
                    mode = "agentic" if args.contestant == "gemini-agentic" else "static"
                    gemini_started = time.monotonic()
                    initial, _ = _run_gemini(node_script, mode, args.gemini_model, video_path, _repair_prompt_placeholder(question), args.timeout_sec, data_dir=root)
                    # Preserve the measured preparation time if a later format
                    # repair call times out.
                    result["upload_processing_s"] = initial.get("upload_processing_s")
                    initial_answer_ms = _duration_seconds(initial.get("answer_ms"))
                    if initial_answer_ms is not None:
                        gemini_answer_ms.append(initial_answer_ms)
                    initial_cleanup_ms = _duration_seconds(initial.get("cleanup_ms"))
                    if initial_cleanup_ms is not None:
                        gemini_cleanup_ms.append(initial_cleanup_ms)
                    gemini_timing_complete = (
                        _duration_seconds(initial.get("upload_processing_s")) is not None
                        and initial_answer_ms is not None
                    )
                    raw = initial.get("answer_text", "")
                    parsed = _parse_json_object(raw)
                    if parsed:
                        parsed = _normalize_answer(parsed, question["answer_format"])
                    initial_parsed = dict(parsed) if parsed else None
                    valid, validation_error = _validate_answer(parsed, question["answer_format"]) if parsed else (False, "no JSON object in final response")
                    usage = initial.get("usage")
                    retries = 0
                    if not valid and initial.get("status") == "ok":
                        repair, _ = _run_gemini(
                            node_script, mode, args.gemini_model, None,
                            _repair_prompt(raw or "No usable answer was returned.", question["answer_format"]),
                            args.timeout_sec, data_dir=root,
                        )
                        repair_answer_ms = _duration_seconds(repair.get("answer_ms"))
                        if repair_answer_ms is not None:
                            gemini_answer_ms.append(repair_answer_ms)
                        else:
                            gemini_timing_complete = False
                        repair_cleanup_ms = _duration_seconds(repair.get("cleanup_ms"))
                        if repair_cleanup_ms is not None:
                            gemini_cleanup_ms.append(repair_cleanup_ms)
                        retries = 1
                        raw = repair.get("answer_text", "")
                        parsed = _parse_json_object(raw)
                        if parsed:
                            parsed = _normalize_answer(parsed, question["answer_format"])
                        valid, validation_error = _validate_answer(parsed, question["answer_format"]) if parsed else (False, repair.get("error", "no JSON object in repair response"))
                        if valid and not _repair_preserves_content(initial_parsed, parsed, question["answer_format"]):
                            valid, validation_error = False, "format repair changed answer or evidence"
                        repair_usage = repair.get("usage")
                        # A repair request without usage makes the run's totals unknown.
                        usage = _aggregate_usage(usage, repair_usage if isinstance(repair_usage, dict) else {"usage_missing_requests": 1})
                    result.update({
                        "status": "ok" if valid and initial.get("status") == "ok" else "error",
                        "failure_type": (
                            None if valid and initial.get("status") == "ok"
                            else "gemini_api" if initial.get("status") != "ok" or (retries and repair.get("status") != "ok")
                            else "json_parse"
                        ),
                        "error": (
                            None if valid and initial.get("status") == "ok"
                            else repair.get("error") if retries and repair.get("status") != "ok"
                            else validation_error if initial.get("status") == "ok"
                            else initial.get("error")
                        ),
                        "answer": parsed.get("answer") if valid else None,
                        "confidence": parsed.get("confidence") if valid else None,
                        "evidence": parsed.get("evidence") if valid else None,
                        "raw_answer_text": raw[:8000],
                        "initial_raw_answer_text": initial.get("answer_text", "")[:8000],
                        "usage": usage,
                        "step_types": initial.get("step_types", []),
                        "processing_calls": initial.get("step_types", []).count("processing_call"),
                        "upload_processing_s": initial.get("upload_processing_s"),
                        "parse_retries": retries,
                        "wall_time_s": round(time.monotonic() - gemini_started, 3),
                        "gemini_answer_time_s": round(sum(gemini_answer_ms) / 1000, 3) if gemini_timing_complete else None,
                        "gemini_timing_complete": gemini_timing_complete,
                        "cleanup_time_s": round(sum(gemini_cleanup_ms) / 1000, 3) if gemini_cleanup_ms else None,
                        "estimated_cost_usd": _gemini_cost(usage),
                        "tool_calls_succeeded": int(initial.get("video_tool_calls_succeeded", 0) or 0),
                        "tool_calls_failed": int(initial.get("video_tool_calls_failed", 0) or 0),
                        "tool_results_received": int(initial.get("video_tool_results_received", 0) or 0),
                    })
            except subprocess.TimeoutExpired:
                elapsed = round(time.monotonic() - gemini_started, 3) if gemini_started is not None else float(args.timeout_sec)
                result.update(status="error", failure_type="timeout", error=f"contestant exceeded {args.timeout_sec} seconds", wall_time_s=elapsed)
            except Exception as exc:
                elapsed = round(time.monotonic() - gemini_started, 3) if gemini_started is not None else 0.0
                result.update(status="error", failure_type="runner_error", error=f"{type(exc).__name__}: {exc}", wall_time_s=elapsed)
        result.setdefault("tool_calls_succeeded", 0)
        result.setdefault("tool_calls_failed", 0)
        _classify_harness_failure(result, args.contestant)
        _record_timing(result, args.contestant)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed += 1
        print(
            f"{question['id']}: {result.get('status')} · answer {result.get('answer_time_s')}s"
            f" · end-to-end {result.get('end_to_end_time_s')}s",
            flush=True,
        )
    print(f"Completed {completed}; skipped existing {skipped}; result directory: {results_dir}")
    return 0


def _repair_prompt_placeholder(question: dict[str, Any]) -> str:
    negative_hint = "For count questions return integer 0 when none occur; do not use not_found." if question.get("track") == "count" else "For negative results use answer \"not_found\"."
    return (
        f"Answer the question using only the supplied video. Return one JSON object with exactly answer, confidence, evidence. "
        f"Answer format: {question['answer_format']}. Seconds are numeric seconds. {negative_hint}\n\n"
        f"Question: {question['question']}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
