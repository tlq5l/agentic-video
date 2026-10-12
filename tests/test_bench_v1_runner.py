import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.bench_v1 import common, run_contestant
from benchmarks.bench_v1.common import DEFAULT_KEY_PATH
from benchmarks.bench_v1.generate_synthetic import _validate_generated_questions
from benchmarks.bench_v1.run_contestant import (
    CODEX_TOOL_TIMEOUT_SEC,
    TOOL_NAMES,
    _codex_mcp_config,
    _codex_run,
    _cannot_inspect_video,
    _classify_harness_failure,
    _codex_events,
    _measure_codex_overhead,
    _normalize_answer,
    _repair_prompt,
    _repair_preserves_content,
    _record_timing,
    _tool_call_counts,
    _validate_answer,
)


def test_numeric_strings_are_normalized_without_accepting_non_numeric_content():
    interval = _normalize_answer(
        {"answer": ["1.25", "2.5"], "confidence": "0.8", "evidence": "visible"},
        "interval_seconds",
    )
    assert interval["answer"] == [1.25, 2.5]
    assert interval["confidence"] == 0.8
    assert _validate_answer(interval, "interval_seconds")[0] is True

    count = _normalize_answer({"answer": "4", "confidence": 0.9, "evidence": "counted"}, "integer")
    assert count["answer"] == 4
    assert _validate_answer(count, "integer")[0] is True

    wrong = _normalize_answer({"answer": "about four", "confidence": 0.9, "evidence": "counted"}, "integer")
    assert _validate_answer(wrong, "integer")[0] is False


def test_codex_mcp_call_records_completed_success_and_error_outcomes():
    events = [
        {"type": "item.started", "item": {"type": "mcp_tool_call", "server": "agentic_video", "tool": "probe_video", "call_id": "ok-1", "arguments": {}}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "agentic_video", "tool": "probe_video", "call_id": "ok-1", "result": {"content": [{"type": "text", "text": "duration: 10"}], "isError": False}}},
        {"type": "item.started", "item": {"type": "mcp_tool_call", "server": "agentic_video", "tool": "inspect_window", "call_id": "bad-1", "arguments": {}}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "agentic_video", "tool": "inspect_window", "call_id": "bad-1", "result": {"content": [{"type": "text", "text": "denied"}], "isError": True}}},
    ]
    _, _, _, calls = _codex_events("\n".join(json.dumps(item) for item in events))
    assert _tool_call_counts(calls) == (1, 1)
    assert calls[1]["outcome_detail"] == "denied"


def test_codex_incomplete_tool_call_counts_as_failure():
    event = {"type": "item.started", "item": {"type": "mcp_tool_call", "server": "agentic_video", "tool": "probe_video", "call_id": "pending", "arguments": {}}}
    _, _, _, calls = _codex_events(json.dumps(event))
    assert _tool_call_counts(calls) == (0, 1)


def test_codex_tool_timeout_is_present_in_contestant_and_overhead_configs(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sealed_dir = tmp_path / "sealed"
    commands = []
    environments = []
    temp_paths = []

    def fake_mkdtemp(prefix, dir=None):
        path = tmp_path / f"scratch-{len(temp_paths)}"
        path.mkdir()
        temp_paths.append(path)
        return str(path)

    def fake_run_process(command, *, prompt, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        if "arithmetic question" in prompt:
            event = {"type": "turn.completed", "usage": {"total_input_tokens": 31}, "output_text": "4"}
        else:
            event = {
                "type": "turn.completed",
                "usage": {"total_input_tokens": 31, "total_output_tokens": 4},
                "last_assistant_message": json.dumps({
                    "answer": [1.0, 2.0], "confidence": 0.8, "evidence": "transcript segment",
                }),
            }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(event), stderr="")

    monkeypatch.setattr(run_contestant.shutil, "which", lambda name: {
        "codex": "/mock/codex", "sandbox-exec": "/mock/sandbox-exec",
    }.get(name))
    monkeypatch.setattr(common.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(run_contestant, "_run_process", fake_run_process)
    hf_home = tmp_path / "custom-hf-cache"
    monkeypatch.setenv("AGENTIC_VIDEO_HF_HOME", str(hf_home))
    monkeypatch.setenv("AGENTIC_VIDEO_BENCH_KEY", str(tmp_path / "other-sealed" / "answer_key.jsonl"))
    monkeypatch.setenv("BENCH_KEY_DIR", str(tmp_path / "other-sealed"))
    question = {
        "id": "search-1", "track": "search", "question": "Find the phrase",
        "answer_format": "interval_seconds",
    }
    contestant = _codex_run(
        question, video, "gpt-6-luna", "medium", 90, tool_cwd,
        tmp_path / "python", sealed_dir, data_dir,
    )
    overhead = _measure_codex_overhead(
        "gpt-6-luna", "medium", 90, tool_cwd, tmp_path / "python",
        sealed_dir, data_dir,
    )

    assert contestant["status"] == "ok"
    assert overhead == 31
    assert len(commands) == 2
    assert all("AGENTIC_VIDEO_BENCH_KEY" not in env and "BENCH_KEY_DIR" not in env for env in environments)
    expected = f"mcp_servers.agentic_video.tool_timeout_sec={CODEX_TOOL_TIMEOUT_SEC}"
    expected_hf_home = f'mcp_servers.agentic_video.env.HF_HOME="{hf_home}"'
    expected_hf_offline = 'mcp_servers.agentic_video.env.HF_HUB_OFFLINE="1"'
    config = _codex_mcp_config(tool_cwd, tmp_path / "python")
    assert expected in config
    assert expected_hf_home in config
    assert expected_hf_offline in config
    assert all(expected in command for command in commands)
    assert all(expected_hf_home in command and expected_hf_offline in command for command in commands)
    for command in commands:
        for value in (expected, expected_hf_home, expected_hf_offline):
            assert any(item == "-c" and command[index + 1] == value for index, item in enumerate(command[:-1]))


def test_codex_allow_list_exposes_visual_search_and_count_tools():
    assert isinstance(TOOL_NAMES, list)
    assert "build_visual_index" in TOOL_NAMES
    assert "search_visual" in TOOL_NAMES
    assert "count_repetitions" in TOOL_NAMES
    schema_path = Path(__file__).resolve().parents[1] / "src" / "adapters" / "mcp_schemas.json"
    served = set(json.loads(schema_path.read_text(encoding="utf-8"))["tools"])
    assert set(TOOL_NAMES) <= served


def test_codex_hf_home_defaults_to_tool_venv(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_VIDEO_HF_HOME", raising=False)
    tool_python = tmp_path / ".venv" / "bin" / "python"
    expected = f'mcp_servers.agentic_video.env.HF_HOME="{tool_python.parent.parent / "huggingface"}"'
    assert expected in _codex_mcp_config(tmp_path, tool_python)
    assert 'mcp_servers.agentic_video.env.HF_HUB_OFFLINE="1"' in _codex_mcp_config(tmp_path, tool_python)


def test_sandbox_denies_configured_key_target_and_directory_as_well_as_default(monkeypatch, tmp_path):
    configured_dir = tmp_path / "configured-sealed"
    configured_dir.mkdir()
    configured_key = configured_dir / "answer_key.jsonl"
    configured_key.write_text("sealed fixture", encoding="utf-8")
    alias = tmp_path / "key-alias.jsonl"
    alias.symlink_to(configured_key)
    monkeypatch.setenv("AGENTIC_VIDEO_BENCH_KEY", str(alias))

    profile = run_contestant._sandbox_profile(
        tmp_path / "original-sealed", tmp_path / "data", tmp_path / "video.mp4",
    )

    for path in (configured_key, DEFAULT_KEY_PATH.resolve()):
        for action in ("file-read*", "file-write*"):
            assert f'(deny {action} (literal "{path}"))' in profile
            assert f'(deny {action} (subpath "{path.parent}"))' in profile


def test_runner_ingests_selected_videos_and_records_ingest_separately(monkeypatch, tmp_path):
    root = tmp_path / "bench-data"
    root.mkdir()
    video = tmp_path / "selected.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    tool_python = tmp_path / "python"
    tool_python.write_text("")
    out = tmp_path / "results"
    overhead = tmp_path / "overhead.json"
    overhead.write_text(json.dumps({"fixed_overhead_input_tokens": 12}))
    question = {
        "id": "search-1", "track": "search", "split": "real",
        "video_path": str(video), "question": "Find a spoken phrase",
        "answer_format": "interval_seconds",
    }
    ingest_calls = []
    run_calls = []

    def fake_ingest(questions, **kwargs):
        ingest_calls.append((questions, kwargs))
        return {
            str(video.resolve()): {
                "status": "ok", "ingest_time_s": 481.2, "visual_index_time_s": 18.8,
                "visual_index_status": "complete",
                "cache_hit_before": False, "last_ingest_time_s": 481.2,
            }
        }

    def fake_codex(*args, **kwargs):
        run_calls.append(args)
        return {
            "status": "ok", "answer": [120.0, 124.0], "confidence": 0.9,
            "evidence": "transcript search result", "wall_time_s": 33.0,
            "tool_calls_succeeded": 1, "tool_calls_failed": 0,
        }

    monkeypatch.setattr(run_contestant, "data_root", lambda: root)
    monkeypatch.setattr(run_contestant, "_read_questions", lambda path, selectors: [question])
    monkeypatch.setattr(run_contestant, "ingest_questions", fake_ingest)
    monkeypatch.setattr(run_contestant, "_codex_run", fake_codex)
    monkeypatch.setattr(run_contestant.shutil, "which", lambda name: None)
    monkeypatch.setenv("AGENTIC_VIDEO_MCP_CWD", str(tool_cwd))
    monkeypatch.setenv("AGENTIC_VIDEO_PYTHON", str(tool_python))
    monkeypatch.setattr(sys, "argv", [
        "run_contestant.py", "--contestant", "ours-codex",
        "--questions", "search-1", "--out", str(out),
        "--codex-overhead-file", str(overhead),
    ])

    assert run_contestant.main() == 0
    assert len(ingest_calls) == 1
    assert ingest_calls[0][0] == [question]
    assert len(run_calls) == 1
    result = json.loads((out / "search-1.json").read_text())
    assert result["ingest_time_s"] == 481.2
    assert result["visual_index_time_s"] == 18.8
    assert result["visual_index_status"] == "complete"
    assert result["ingest_cache_hit_before"] is False
    assert result["tool_timeout_sec"] == CODEX_TOOL_TIMEOUT_SEC
    assert result["wall_time_s"] == 33.0
    assert result["answer_time_s"] == 33.0
    assert result["preparation_time_s"] == 500.0
    assert result["end_to_end_time_s"] == 533.0


@pytest.mark.parametrize("raise_ingest_error", [False, True])
def test_ingest_failures_are_saved_for_each_selected_question(monkeypatch, tmp_path, raise_ingest_error):
    root = tmp_path / "bench-data"
    root.mkdir()
    failed_video = tmp_path / "failed.mp4"
    failed_video.write_bytes(b"failed video fixture")
    good_video = tmp_path / "good.mp4"
    good_video.write_bytes(b"good video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    tool_python = tmp_path / "python"
    tool_python.write_text("")
    out = tmp_path / "results"
    overhead = tmp_path / "overhead.json"
    overhead.write_text(json.dumps({"fixed_overhead_input_tokens": 12}))
    questions = [
        {"id": "search-1", "track": "search", "split": "real", "video_path": str(failed_video), "question": "Phrase A?", "answer_format": "interval_seconds"},
        {"id": "search-2", "track": "search", "split": "real", "video_path": str(failed_video), "question": "Phrase B?", "answer_format": "interval_seconds"},
        {"id": "search-3", "track": "search", "split": "real", "video_path": str(good_video), "question": "Phrase C?", "answer_format": "interval_seconds"},
    ]
    run_ids = []

    def fake_ingest(selected_questions, **kwargs):
        if raise_ingest_error:
            raise RuntimeError("transcript cache unavailable")
        assert selected_questions == questions
        return {
            str(failed_video.resolve()): {
                "status": "error", "error": "ASR failed", "last_ingest_time_s": 12.5,
                "last_cache_hit_before": False,
            },
            str(good_video.resolve()): {
                "status": "ok", "ingest_time_s": 4.0, "last_ingest_time_s": 4.0,
                "cache_hit_before": False,
            },
        }

    def fake_codex(question, *args, **kwargs):
        run_ids.append(question["id"])
        return {
            "status": "ok", "answer": [1.0, 2.0], "confidence": 0.9,
            "evidence": "found", "wall_time_s": 5.0,
            "tool_calls_succeeded": 1,
        }

    monkeypatch.setattr(run_contestant, "data_root", lambda: root)
    monkeypatch.setattr(run_contestant, "_read_questions", lambda path, selectors: questions)
    monkeypatch.setattr(run_contestant, "ingest_questions", fake_ingest)
    monkeypatch.setattr(run_contestant, "_codex_run", fake_codex)
    monkeypatch.setattr(run_contestant, "_measure_codex_overhead", lambda *args, **kwargs: 12)
    monkeypatch.setenv("AGENTIC_VIDEO_MCP_CWD", str(tool_cwd))
    monkeypatch.setenv("AGENTIC_VIDEO_PYTHON", str(tool_python))
    monkeypatch.setattr(sys, "argv", [
        "run_contestant.py", "--contestant", "ours-codex",
        "--questions", "search-1", "search-2", "search-3", "--out", str(out),
        "--codex-overhead-file", str(overhead),
    ])

    assert run_contestant.main() == 0
    failed_ids = ["search-1", "search-2"] if not raise_ingest_error else ["search-1", "search-2", "search-3"]
    assert all((out / f"{question_id}.json").exists() for question_id in failed_ids)
    for question_id in failed_ids:
        result = json.loads((out / f"{question_id}.json").read_text())
        assert result["status"] == "error"
        assert result["failure_type"] == "ingest"
        assert result["answer"] is None
        if not raise_ingest_error:
            assert result["preparation_time_s"] == 12.5
            assert result["answer_time_s"] is None
            assert result["end_to_end_time_s"] == 12.5
    assert run_ids == ([] if raise_ingest_error else ["search-3"])
    if not raise_ingest_error:
        assert json.loads((out / "search-3.json").read_text())["status"] == "ok"


def test_runner_records_gemini_upload_processing_time_without_api_call(monkeypatch, tmp_path):
    root = tmp_path / "bench-data"
    root.mkdir()
    video = tmp_path / "selected.mp4"
    video.write_bytes(b"video fixture")
    out = tmp_path / "results"
    question = {
        "id": "search-1", "track": "search", "split": "real",
        "video_path": str(video), "question": "Find a spoken phrase",
        "answer_format": "interval_seconds",
    }
    answer_text = json.dumps({
        "answer": [120.0, 124.0], "confidence": 0.9,
        "evidence": "transcript search result",
    })

    def fake_gemini(*args, **kwargs):
        return ({
            "status": "ok", "answer_text": answer_text, "usage": {},
            "step_types": ["processing_call", "processing_result"],
            "video_tool_calls_succeeded": 1, "video_tool_calls_failed": 0,
            "video_tool_results_received": 1, "upload_processing_s": 14.25,
            "answer_ms": 25750.0, "cleanup_ms": 400.0,
        }, "")

    monotonic_values = iter([100.0, 140.0])
    monkeypatch.setattr(run_contestant.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(run_contestant, "data_root", lambda: root)
    monkeypatch.setattr(run_contestant, "_read_questions", lambda path, selectors: [question])
    monkeypatch.setattr(run_contestant, "_run_gemini", fake_gemini)
    monkeypatch.setattr(sys, "argv", [
        "run_contestant.py", "--contestant", "gemini-agentic",
        "--questions", "search-1", "--out", str(out),
    ])

    assert run_contestant.main() == 0
    result = json.loads((out / "search-1.json").read_text())
    assert result["status"] == "ok"
    assert result["upload_processing_s"] == 14.25
    assert result["answer_time_s"] == 25.75
    assert result["preparation_time_s"] == 14.25
    assert result["end_to_end_time_s"] == 40.0
    assert result["cleanup_time_s"] == 0.4
    assert result["wall_time_s"] == 25.75


def test_harness_classifier_catches_explicit_video_access_failure():
    assert _cannot_inspect_video("I could not inspect the video because the tool call failed.")
    assert not _cannot_inspect_video("I inspected the video and found the blue sign.")


@pytest.mark.parametrize("failure_type", ["timeout", "runner_error", "json_parse"])
def test_zero_tool_calls_do_not_replace_an_existing_error(failure_type):
    result = {
        "status": "error", "failure_type": failure_type,
        "error": "existing failure", "tool_calls_succeeded": 0,
    }

    _classify_harness_failure(result, "ours-codex")

    assert result["failure_type"] == failure_type
    assert result["error"] == "existing failure"


def test_zero_successful_tool_calls_on_an_otherwise_valid_result_are_harness_failure():
    result = {"status": "ok", "answer": 3, "tool_calls_succeeded": 0}

    _classify_harness_failure(result, "gemini-agentic")

    assert result["status"] == "error"
    assert result["failure_type"] == "harness"


def test_gemini_timeout_keeps_end_to_end_time_when_answer_split_is_unknown():
    result = {"failure_type": "timeout", "wall_time_s": 60.0}

    _record_timing(result, "gemini-plain")

    assert result["answer_time_s"] is None
    assert result["preparation_time_s"] is None
    assert result["end_to_end_time_s"] == 60.0
    assert result["wall_time_s"] is None


@pytest.mark.parametrize("failure_type", ["runner_error", "gemini_api", "json_parse"])
def test_gemini_failures_without_adapter_timing_do_not_become_answer_time(failure_type):
    result = {"failure_type": failure_type, "wall_time_s": 60.0}

    _record_timing(result, "gemini-plain")

    assert result["answer_time_s"] is None
    assert result["preparation_time_s"] is None
    assert result["end_to_end_time_s"] == 60.0
    assert result["wall_time_s"] is None


def test_gemini_text_only_repair_uses_repair_operation_without_video(monkeypatch, tmp_path):
    calls = []
    environments = []

    def fake_run(command, **kwargs):
        calls.append(json.loads(kwargs["input"]))
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"ok","answer_text":"{}"}', stderr="")

    monkeypatch.setattr(run_contestant.shutil, "which", lambda _: "/usr/bin/node")
    monkeypatch.setattr(run_contestant.subprocess, "run", fake_run)
    monkeypatch.setenv("AGENTIC_VIDEO_BENCH_KEY", str(tmp_path / "answer_key.jsonl"))
    monkeypatch.setenv("ANSWER_KEY_PATH", str(tmp_path / "answer_key.jsonl"))
    result, _ = run_contestant._run_gemini(
        Path(tmp_path) / "gemini-video.mjs", "static", "gemini-3.8-flash", None,
        "Repair this prior raw answer: invalid confidence.", 30,
    )

    assert result["status"] == "ok"
    assert "AGENTIC_VIDEO_BENCH_KEY" not in environments[0]
    assert "ANSWER_KEY_PATH" not in environments[0]
    assert calls == [{
        "operation": "repair", "model": "gemini-3.8-flash", "mode": "static",
        "prompt": "Repair this prior raw answer: invalid confidence.",
    }]


def test_gemini_adapter_sandbox_denies_both_keys_and_other_videos(monkeypatch, tmp_path):
    default_key = tmp_path / "default-sealed" / "answer_key.jsonl"
    configured_key = tmp_path / "relocated-sealed" / "answer_key.jsonl"
    monkeypatch.setattr(run_contestant, "DEFAULT_KEY_PATH", default_key)
    monkeypatch.setenv("AGENTIC_VIDEO_BENCH_KEY", str(configured_key))
    data_dir = tmp_path / "data"
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(parents=True)
    selected_video = videos_dir / "selected.mp4"
    other_video = videos_dir / "other.mp4"
    selected_video.write_bytes(b"selected fixture")
    other_video.write_bytes(b"other fixture")
    script = tmp_path / "adapter" / "gemini-video.mjs"
    script.parent.mkdir()
    script.write_text("// fixture", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"ok"}', stderr="")

    monkeypatch.setattr(run_contestant.shutil, "which", lambda name: {
        "node": "/mock/node", "sandbox-exec": "/mock/sandbox-exec",
    }.get(name))
    monkeypatch.setattr(run_contestant.subprocess, "run", fake_run)

    for video_path in (selected_video, None):
        result, _ = run_contestant._run_gemini(
            script, "static", "gemini-3.8-flash", video_path, "fixture prompt", 30,
            data_dir=data_dir,
        )
        assert result["status"] == "ok"

    assert len(commands) == 2
    for command in commands:
        assert command[:2] == ["/mock/sandbox-exec", "-p"]
        assert command[3:] == ["/mock/node", str(script)]
        profile = command[2]
        assert "(allow default)" in profile
        for key in (default_key, configured_key):
            for action in ("file-read*", "file-write*"):
                assert f'(deny {action} (literal "{key}"))' in profile
                assert f'(deny {action} (subpath "{key.parent}"))' in profile
        assert f'(deny file-read* (literal "{other_video}"))' in profile
    assert f'(deny file-read* (literal "{selected_video}"))' not in commands[0][2]
    assert f'(deny file-read* (literal "{selected_video}"))' in commands[1][2]


def test_gemini_adapter_fails_closed_without_sandbox_exec(monkeypatch, tmp_path):
    monkeypatch.setattr(run_contestant.shutil, "which", lambda name: "/mock/node" if name == "node" else None)
    with pytest.raises(RuntimeError, match="sandbox-exec is required to deny access to the sealed key"):
        run_contestant._run_gemini(tmp_path / "gemini-video.mjs", "static", "gemini-3.8-flash", None, "repair", 30)


def test_format_repair_keeps_answer_and_requires_numeric_confidence():
    prompt = _repair_prompt(
        '{"answer":4,"confidence":"high","evidence":"counted"}', "integer"
    )
    assert "Preserve its answer and evidence exactly" in prompt
    assert "Confidence must be a JSON number from 0 to 1" in prompt
    assert "use 0.5" in prompt
    assert '"answer":4' in prompt


def test_format_repair_cannot_change_answer_or_evidence():
    initial = {"answer": "4", "confidence": "high", "evidence": "counted"}
    repaired = {"answer": 4, "confidence": 0.5, "evidence": "counted"}
    assert _repair_preserves_content(initial, repaired, "integer")
    assert not _repair_preserves_content({**initial, "answer": "5"}, repaired, "integer")
    assert not _repair_preserves_content(initial, {**repaired, "evidence": "guessed"}, "integer")
    assert not _repair_preserves_content(None, repaired, "integer")


def test_structured_evidence_normalizes_to_stable_text_without_changing_content():
    initial = {
        "answer": 4,
        "confidence": "0.8",
        "evidence": [{"time": 2.5, "detail": "visible event"}],
    }
    repaired = {
        "answer": 4,
        "confidence": 0.5,
        "evidence": [{"detail": "visible event", "time": 2.5}],
    }
    normalized = _normalize_answer(initial, "integer")
    assert normalized["evidence"] == '[{"detail":"visible event","time":2.5}]'
    assert _validate_answer(normalized, "integer")[0] is True
    assert _repair_preserves_content(initial, repaired, "integer")


def test_synthetic_generator_rejects_duplicate_question_text():
    question_rows = [
        {"id": "m1", "track": "moment", "question": "When does a blue sign appear?"},
        {"id": "m2", "track": "moment", "question": "  WHEN does a blue sign appear? "},
    ]
    key_rows = [
        {"id": "m1", "truth": 1.0},
        {"id": "m2", "truth": 2.0},
    ]
    with pytest.raises(RuntimeError, match="duplicate synthetic question text"):
        _validate_generated_questions(question_rows, key_rows, {"m1": 1, "m2": 1})


def test_synthetic_generator_rejects_more_target_occurrences_than_count_truth():
    question_rows = [{"id": "c1", "track": "count", "question": "How many green markers cross?"}]
    key_rows = [{"id": "c1", "truth": 2}]
    with pytest.raises(RuntimeError, match="occurs 3 times but truth allows 2"):
        _validate_generated_questions(question_rows, key_rows, {"c1": 3})


@pytest.mark.parametrize(("answer", "answer_format", "expected"), [
    ("01:43 - 02:03", "interval_seconds", [103, 123]),
    ("7-8", "interval_seconds", [7, 8]),
    ("7 to 9.5", "interval_seconds", [7, 9.5]),
    ("interval_seconds: [30, 31]", "interval_seconds", [30, 31]),
    (["00:01:22", "00:01:36"], "interval_seconds", [82, 96]),
    ("1:02:03 - 1:02:10", "interval_seconds", [3723, 3730]),
    ("00:52", "number_seconds", 52),
    ("01:30.5", "number_seconds_or_not_found", 90.5),
])
def test_unambiguous_timestamp_strings_normalize_to_seconds(answer, answer_format, expected):
    # Gemini often writes "mm:ss - mm:ss"; converting it is a format change only,
    # so the initial answer is valid and a repair that returns seconds is kept.
    initial = {"answer": answer, "confidence": 0.8, "evidence": "seen"}
    normalized = _normalize_answer(initial, answer_format)
    assert normalized["answer"] == expected
    assert _validate_answer(normalized, answer_format)[0] is True
    assert _repair_preserves_content(initial, {"answer": expected, "confidence": 0.5, "evidence": "seen"}, answer_format)
    assert not _repair_preserves_content(initial, {"answer": [0, 1], "confidence": 0.5, "evidence": "seen"}, answer_format)


@pytest.mark.parametrize(("answer", "answer_format"), [
    ("about 7", "interval_seconds"),
    ("1:5", "interval_seconds"),
    ("7-8", "number_seconds"),
    ("00:52", "integer"),
    ("-3", "interval_seconds"),
])
def test_ambiguous_strings_are_not_normalized(answer, answer_format):
    normalized = _normalize_answer({"answer": answer, "confidence": 0.8, "evidence": "seen"}, answer_format)
    assert normalized["answer"] == answer
    assert _validate_answer(normalized, answer_format)[0] is False


def test_codex_format_repair_cannot_change_answer(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    temp_paths = []

    def fake_mkdtemp(prefix, dir=None):
        path = tmp_path / f"scratch-{len(temp_paths)}"
        path.mkdir()
        temp_paths.append(path)
        return str(path)

    replies = [
        {"answer": "roughly the middle", "confidence": 0.8, "evidence": "seen"},
        {"answer": [10, 20], "confidence": 0.5, "evidence": "seen"},
    ]

    def fake_run_process(command, *, prompt, **kwargs):
        event = {"type": "turn.completed", "usage": {"total_input_tokens": 5},
                 "last_assistant_message": json.dumps(replies.pop(0))}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(event), stderr="")

    monkeypatch.setattr(run_contestant.shutil, "which", lambda name: {
        "codex": "/mock/codex", "sandbox-exec": "/mock/sandbox-exec",
    }.get(name))
    monkeypatch.setattr(common.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(run_contestant, "_run_process", fake_run_process)
    question = {"id": "m-1", "track": "moment", "question": "When?", "answer_format": "interval_seconds"}
    result = _codex_run(question, video, "gpt-6-luna", "medium", 90, tool_cwd,
                        tmp_path / "python", tmp_path / "sealed", tmp_path)
    assert result["status"] == "error"
    assert result["failure_type"] == "json_parse"
    assert result["error"] == "format repair changed answer or evidence"
    assert result["parse_retries"] == 1


def test_scratch_dir_is_under_a_noindex_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(common.tempfile, "tempdir", str(tmp_path))
    first = common.scratch_dir("bench-v1-contestant-")
    second = common.scratch_dir("bench-v1-contestant-")
    assert first.parent == second.parent == tmp_path / "agentic-video-bench.noindex"
    assert first != second
    assert first.is_dir() and first.name.startswith("bench-v1-contestant-")
