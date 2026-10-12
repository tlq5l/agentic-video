#!/usr/bin/env python3
"""Score contestant result files against the sealed bench-v1 answer key."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import CONTESTANTS, DEFAULT_DATA_ROOT, key_path, read_jsonl
except ImportError:  # direct script execution
    from common import CONTESTANTS, DEFAULT_DATA_ROOT, key_path, read_jsonl


GEMINI_INPUT_USD_PER_MILLION = 0.75
GEMINI_OUTPUT_USD_PER_MILLION = 3.75


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = _number(value[0]), _number(value[1])
    if start is None or end is None or end < start:
        return None
    return start, end


def interval_iou(predicted: Any, truth: Any) -> float | None:
    """Return temporal IoU, including defined behavior for zero-width points."""
    p, t = _interval(predicted), _interval(truth)
    if p is None or t is None:
        return None
    intersection = max(0.0, min(p[1], t[1]) - max(p[0], t[0]))
    union = max(p[1], t[1]) - min(p[0], t[0])
    if union == 0:
        return 1.0 if p[0] == t[0] else 0.0
    return intersection / union


def _is_not_found(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().replace("-", "_").replace(" ", "_") in {
        "not_found", "notfound", "none"
    }


def _normalized_cue_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s_-]+", "_", value.strip().casefold())


def score_prediction(track: str, truth: Any, prediction: Any, *, truth_windows: list[Any] | None = None) -> dict[str, float | bool | None]:
    """Pure per-answer metrics used by the CLI and unit tests."""
    if track == "count":
        expected = _number(truth)
        actual = _number(prediction)
        if expected is None or actual is None:
            return {"exact": False, "within_one": False}
        return {"exact": actual == expected, "within_one": abs(actual - expected) <= 1}

    if _is_not_found(truth):
        hit = _is_not_found(prediction)
        if track == "moment":
            return {"hit_0.2": hit, "hit_0.5": hit, "hit_1.0": hit, "start_error": 0.0 if hit else None, "iou": None}
        return {"hit_2": hit, "hit_5": hit, "distance": 0.0 if hit else None, "iou": None}

    target_interval = _interval(truth)
    valid_truth_windows = [window for raw in (truth_windows or []) if (window := _interval(raw)) is not None]
    if not valid_truth_windows and target_interval is not None:
        valid_truth_windows = [target_interval]
    if track == "moment" and truth_windows and valid_truth_windows:
        pred_interval = _interval(prediction)
        if pred_interval is None:
            return {"iou": 0.0, "hit_iou_0.5": False, "hit_iou_0.7": False,
                    "midpoint_error": None, "midpoint_within_2": False}
        ious = [(interval_iou(pred_interval, window) or 0.0, window) for window in valid_truth_windows]
        best_iou = max(score for score, _ in ious)
        best_windows = [window for score, window in ious if score == best_iou]
        predicted_midpoint = (pred_interval[0] + pred_interval[1]) / 2.0
        midpoint_error = min(abs(predicted_midpoint - (window[0] + window[1]) / 2.0) for window in best_windows)
        return {
            "iou": best_iou,
            "hit_iou_0.5": best_iou >= 0.5,
            "hit_iou_0.7": best_iou >= 0.7,
            "midpoint_error": midpoint_error,
            "midpoint_within_2": midpoint_error <= 2.0,
        }
    if target_interval is not None:
        pred_interval = _interval(prediction)
        point = _number(prediction)
        if pred_interval is not None:
            # Moment accuracy checks both boundaries when the contestant returns
            # an interval. Its IoU is reported separately as a shape/overlap score.
            start_error = abs(pred_interval[0] - target_interval[0])
            end_error = abs(pred_interval[1] - target_interval[1])
            distance = max(start_error, end_error) if track == "moment" else max(
                target_interval[0] - (pred_interval[0] + pred_interval[1]) / 2.0,
                0.0,
                (pred_interval[0] + pred_interval[1]) / 2.0 - target_interval[1],
            )
            iou = interval_iou(pred_interval, target_interval)
        elif point is not None:
            distance = max(target_interval[0] - point, 0.0, point - target_interval[1])
            iou = None
            start_error = distance
        else:
            distance, iou, start_error = None, None, None
    else:
        target_point = _number(truth)
        pred_interval = _interval(prediction)
        point = _number(prediction)
        if target_point is None:
            distance, iou, start_error = None, None, None
        elif pred_interval is not None:
            center = (pred_interval[0] + pred_interval[1]) / 2.0
            distance = abs(center - target_point)
            iou = None
            start_error = abs(pred_interval[0] - target_point)
        elif point is not None:
            distance = abs(point - target_point)
            iou = None
            start_error = distance
        else:
            distance, iou, start_error = None, None, None

    if track == "moment":
        return {
            "hit_0.2": distance is not None and distance <= 0.2,
            "hit_0.5": distance is not None and distance <= 0.5,
            "hit_1.0": distance is not None and distance <= 1.0,
            "start_error": start_error,
            "boundary_error": distance,
            "iou": iou,
        }
    return {
        "hit_2": distance is not None and distance <= 2.0,
        "hit_5": distance is not None and distance <= 5.0,
        "distance": distance,
        "iou": iou,
    }


def _usage_tokens(usage: Any, *, adjusted: bool = False) -> float | None:
    if not isinstance(usage, dict) or usage.get("usage_missing_requests"):
        return None
    if adjusted:
        value = _number(usage.get("adjusted_total_tokens"))
        return value if value is not None else _usage_tokens(usage)
    value = _number(usage.get("total_tokens"))
    if value is not None:
        return value
    candidates = ("billable_tokens", "billableTokens")
    for key in candidates:
        value = _number(usage.get(key))
        if value is not None:
            return value
    input_tokens = next((_number(usage.get(k)) for k in ("total_input_tokens", "totalInputTokens", "input_tokens", "inputTokens", "prompt_tokens") if _number(usage.get(k)) is not None), None)
    output_tokens = next((_number(usage.get(k)) for k in ("total_output_tokens", "totalOutputTokens", "output_tokens", "outputTokens", "completion_tokens") if _number(usage.get(k)) is not None), None)
    tool_tokens = next((_number(usage.get(k)) for k in ("total_tool_use_tokens", "tool_use_prompt_token_count", "totalToolUseTokens") if _number(usage.get(k)) is not None), None)
    thought_tokens = next((_number(usage.get(k)) for k in ("total_thought_tokens", "thoughts_token_count", "totalThoughtTokens") if _number(usage.get(k)) is not None), None)
    if input_tokens is not None or output_tokens is not None or tool_tokens is not None or thought_tokens is not None:
        return (input_tokens or 0.0) + (output_tokens or 0.0) + (tool_tokens or 0.0) + (thought_tokens or 0.0)
    return next((_number(usage.get(k)) for k in ("total_tokens", "totalTokens", "tokens_used") if _number(usage.get(k)) is not None), None)


def _cost_for(contestant: str, usage: Any) -> float | None:
    if not contestant.startswith("gemini-") or not isinstance(usage, dict) or usage.get("usage_missing_requests"):
        return None
    in_tok = next((_number(usage.get(k)) for k in ("total_input_tokens", "totalInputTokens", "input_tokens", "inputTokens", "prompt_tokens") if _number(usage.get(k)) is not None), 0.0)
    out_tok = next((_number(usage.get(k)) for k in ("total_output_tokens", "totalOutputTokens", "output_tokens", "outputTokens", "completion_tokens") if _number(usage.get(k)) is not None), 0.0)
    tool_tok = next((_number(usage.get(k)) for k in ("total_tool_use_tokens", "tool_use_prompt_token_count", "totalToolUseTokens") if _number(usage.get(k)) is not None), 0.0)
    thought_tok = next((_number(usage.get(k)) for k in ("total_thought_tokens", "thoughts_token_count", "totalThoughtTokens") if _number(usage.get(k)) is not None), 0.0)
    # If only an aggregate was returned, charge it as input: it is a conservative,
    # slightly low-cost bound because input is cheaper than output.
    aggregate = _number(usage.get("total_tokens"))
    if in_tok == 0 and out_tok == 0 and tool_tok == 0 and thought_tok == 0 and aggregate is not None:
        in_tok = aggregate
    billable_input = in_tok + tool_tok
    billable_output = out_tok + thought_tok
    return (billable_input * GEMINI_INPUT_USD_PER_MILLION + billable_output * GEMINI_OUTPUT_USD_PER_MILLION) / 1_000_000


def _result_paths(results_root: Path) -> list[Path]:
    """Find result JSON files, following symlinked contestant folders.

    ``Path.glob("**")`` skips symlinked directories, so a results folder that
    links to another run's contestant folder would score as all missing. Each
    real directory is visited once, which also stops symlink loops.
    """
    paths: list[Path] = []
    visited: set[tuple[int, int]] = set()
    for directory, subdirs, files in os.walk(results_root, followlinks=True):
        try:
            stat = os.stat(directory)
        except OSError:
            subdirs[:] = []
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in visited:
            subdirs[:] = []
            continue
        visited.add(identity)
        paths.extend(Path(directory) / name for name in files if name.endswith(".json"))
    return sorted(paths)


def _load_results(results_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not results_root.exists():
        return results
    for path in _result_paths(results_root):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "id" in value and "contestant" in value:
            results.append(value)
    return results


def _percentage(values: list[bool | None]) -> str:
    if not values:
        return "—"
    return f"{100 * sum(bool(value) for value in values) / len(values):.1f}%"


def _result_times(contestant: str, result: dict[str, Any]) -> tuple[float | None, float | None, float | None, bool]:
    """Return answer, preparation, end-to-end seconds, and legacy cleanup uncertainty."""
    wall = _number(result.get("wall_time_s"))
    upload_processing = _number(result.get("upload_processing_s"))
    cleanup = _number(result.get("cleanup_time_s"))
    includes_unmeasured_cleanup = False
    preparation = _number(result.get("preparation_time_s"))
    if preparation is None:
        if contestant == "ours-codex":
            preparation_parts = [
                duration for duration in (
                    _number(result.get("ingest_time_s")),
                    _number(result.get("visual_index_time_s")),
                ) if duration is not None
            ]
            preparation = sum(preparation_parts) if preparation_parts else None
        elif contestant.startswith("gemini-"):
            preparation = upload_processing

    if "answer_time_s" in result:
        answer = _number(result.get("answer_time_s"))
    elif contestant.startswith("gemini-"):
        if wall is not None and upload_processing is not None:
            # Older records used wall_time_s for upload, processing, answer,
            # and Files API cleanup. Cleanup was not always timed separately.
            answer = max(0.0, wall - upload_processing - (cleanup or 0.0))
            includes_unmeasured_cleanup = cleanup is None
        else:
            # Do not present an unsplit legacy Gemini duration as answer time.
            answer = None
    else:
        answer = wall

    end_to_end = _number(result.get("end_to_end_time_s"))
    if end_to_end is None:
        if contestant.startswith("gemini-") and "answer_time_s" not in result and wall is not None:
            end_to_end = wall
        elif answer is not None:
            end_to_end = answer + (preparation or 0.0)
        else:
            end_to_end = wall
    return answer, preparation, end_to_end, includes_unmeasured_cleanup


def score_rows(
    questions: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    contestants: list[str] | None = None,
    per_question: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize scores per track and contestant.

    When ``per_question`` is a list, one row per scored question is appended to
    it with the question id, track, contestant, result status, contestant answer,
    and hit metrics. It never carries answer-key fields.
    """
    key_by_id = {row["id"]: row for row in keys}
    q_by_id = {row["id"]: row for row in questions}
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    contestant_names = {str(result.get("contestant", "unknown")) for result in results}
    contestant_names.update(str(contestant) for contestant in (contestants or []))
    result_by_contestant_and_id: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        question_id = result.get("id")
        if question_id in q_by_id:
            contestant = str(result.get("contestant", "unknown"))
            result_by_contestant_and_id[(contestant, question_id)] = result
    for contestant in sorted(contestant_names):
        for question in questions:
            question_id = question["id"]
            result = result_by_contestant_and_id.get((contestant, question_id))
            if result is None:
                result = {
                    "id": question_id, "contestant": contestant,
                    "status": "error", "failure_type": "missing", "answer": None,
                }
            grouped[(question["track"], contestant)].append((
                question, key_by_id.get(question_id, {"id": question_id}), result,
            ))

    summary: list[dict[str, Any]] = []
    for (track, contestant), items in sorted(grouped.items()):
        metric_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        failures = 0
        harness_failures = 0
        token_values = []
        adjusted_token_values = []
        answer_values = []
        ingest_by_video: dict[str, float] = {}
        visual_index_by_video: dict[str, float] = {}
        preparation_by_video: dict[str, float] = {}
        upload_processing_values = []
        gemini_preparation_values = []
        gemini_end_to_end_values = []
        cleanup_values = []
        answer_includes_unmeasured_cleanup_count = 0
        preparation_unknown_count = 0
        missing_results = 0
        costs = []
        negative_hits: list[bool] = []
        for question, key, result in items:
            is_harness_failure = result.get("failure_type") == "harness"
            failed = result.get("status") != "ok" or is_harness_failure
            failures += int(failed)
            harness_failures += int(is_harness_failure)
            missing_results += int(result.get("failure_type") == "missing")
            usage = result.get("usage")
            tokens = _usage_tokens(usage)
            if tokens is not None:
                token_values.append(tokens)
            adjusted_tokens = _usage_tokens(usage, adjusted=True)
            if adjusted_tokens is not None:
                adjusted_token_values.append(adjusted_tokens)
            answer_time, preparation_time, end_to_end_time, includes_unmeasured_cleanup = _result_times(contestant, result)
            if answer_time is not None:
                answer_values.append(answer_time)
            answer_includes_unmeasured_cleanup_count += int(includes_unmeasured_cleanup)
            if preparation_time is None:
                preparation_unknown_count += 1
            ingest_time = _number(result.get("ingest_time_s")) if contestant == "ours-codex" else None
            visual_index_time = _number(result.get("visual_index_time_s")) if contestant == "ours-codex" else None
            if contestant == "ours-codex":
                video_key = str(question.get("video_path") or question.get("id"))
                # Before visual timing was recorded, preparation_time_s was
                # the transcript-ingest value. Keep those saved runs usable.
                if ingest_time is None and "visual_index_time_s" not in result:
                    ingest_time = _number(result.get("preparation_time_s"))
                if ingest_time is not None:
                    ingest_by_video.setdefault(video_key, ingest_time)
                if visual_index_time is not None:
                    visual_index_by_video.setdefault(video_key, visual_index_time)
            upload_processing = _number(result.get("upload_processing_s"))
            if upload_processing is not None:
                upload_processing_values.append(upload_processing)
            if contestant.startswith("gemini-"):
                if preparation_time is not None:
                    gemini_preparation_values.append(preparation_time)
                if end_to_end_time is not None:
                    gemini_end_to_end_values.append(end_to_end_time)
                cleanup_time = _number(result.get("cleanup_time_s"))
                if cleanup_time is not None:
                    cleanup_values.append(cleanup_time)
            # The missing-usage marker overrides any saved estimate from the measured requests.
            missing_usage = isinstance(usage, dict) and usage.get("usage_missing_requests")
            cost = None if missing_usage else _number(result.get("estimated_cost_usd"))
            if cost is None and not missing_usage:
                cost = _cost_for(contestant, usage)
            if cost is not None:
                costs.append(cost)
            metrics = score_prediction(
                track, key.get("truth"), result.get("answer"),
                truth_windows=(key.get("truth_windows") or [key.get("truth")])
                if question.get("split") == "real" and track == "moment" else None,
            )
            if _is_not_found(key.get("truth")) or _normalized_cue_type(question.get("cue_type")) == "negative_control":
                negative_hits.append(not failed and bool(
                    metrics.get("hit_0.5") if track == "moment" else
                    metrics.get("hit_2") if track == "search" else
                    metrics.get("exact")
                ))
            if failed:
                for name in metrics:
                    if isinstance(metrics[name], bool):
                        metrics[name] = False
                    else:
                        metrics[name] = None
            metric_rows.append((question, key, metrics))
            if per_question is not None:
                per_question.append({
                    "id": question["id"],
                    "track": track,
                    "split": question.get("split"),
                    "contestant": contestant,
                    "status": "ok" if not failed else str(result.get("failure_type") or result.get("status") or "error"),
                    "answer": result.get("answer"),
                    "metrics": dict(metrics),
                })

        avg = lambda seq: (sum(seq) / len(seq)) if seq else None
        metrics_only = [metrics for _, _, metrics in metric_rows]
        mean_iou = avg([
            metrics["iou"] if isinstance(metrics.get("iou"), (int, float)) else 0.0
            for question, key, metrics in metric_rows
            if _interval(key.get("truth")) is not None
            and not _is_not_found(key.get("truth"))
            and _normalized_cue_type(question.get("cue_type")) != "negative_control"
        ])
        preparation_video_keys = list(ingest_by_video)
        preparation_video_keys.extend(
            video for video in visual_index_by_video if video not in ingest_by_video
        )
        preparation_by_video = {
            video: ingest_by_video.get(video, 0.0) + visual_index_by_video.get(video, 0.0)
            for video in preparation_video_keys
        }
        total_visual_index = sum(visual_index_by_video.values()) if visual_index_by_video else 0.0
        if contestant == "ours-codex":
            total_preparation = sum(preparation_by_video.values())
            total_end_to_end = sum(answer_values) + total_preparation
        else:
            total_preparation = sum(gemini_preparation_values)
            total_end_to_end = sum(gemini_end_to_end_values)
        mean_end_to_end = total_end_to_end / len(items) if items else None
        synthetic_moments = [metrics for question, _, metrics in metric_rows if question.get("split") == "synthetic" and track == "moment"]
        real_moments = [
            metrics for question, key, metrics in metric_rows
            if track == "moment" and question.get("split") == "real"
            and _interval(key.get("truth")) is not None
            and not _is_not_found(key.get("truth"))
            and _normalized_cue_type(question.get("cue_type")) != "negative_control"
        ]
        summary.append({
            "track": track,
            "contestant": contestant,
            "n": len(items),
            "failures": failures,
            "harness_failures": harness_failures,
            "negative_control_hit": _percentage(negative_hits),
            "hit_0.2": _percentage([x.get("hit_0.2") for x in synthetic_moments]) if track == "moment" else None,
            "hit_0.5": _percentage([x.get("hit_0.5") for x in synthetic_moments]) if track == "moment" else None,
            "hit_1.0": _percentage([x.get("hit_1.0") for x in synthetic_moments]) if track == "moment" else None,
            "real_iou_0.5": _percentage([x.get("hit_iou_0.5") for x in real_moments]) if track == "moment" else None,
            "real_iou_0.7": _percentage([x.get("hit_iou_0.7") for x in real_moments]) if track == "moment" else None,
            "real_midpoint_2": _percentage([x.get("midpoint_within_2") for x in real_moments]) if track == "moment" else None,
            "moment_iou": mean_iou if track == "moment" else None,
            "hit_2": _percentage([x.get("hit_2") for x in metrics_only]) if track == "search" else None,
            "hit_5": _percentage([x.get("hit_5") for x in metrics_only]) if track == "search" else None,
            "search_iou": mean_iou if track == "search" else None,
            "exact": _percentage([x.get("exact") for x in metrics_only]) if track == "count" else None,
            "within_one": _percentage([x.get("within_one") for x in metrics_only]) if track == "count" else None,
            "mean_tokens_total": avg(token_values),
            "tokens_total_count": len(token_values),
            "mean_tokens_adjusted": avg(adjusted_token_values),
            "tokens_adjusted_count": len(adjusted_token_values),
            "mean_answer_s": avg(answer_values),
            "mean_answer_is_upper_bound": answer_includes_unmeasured_cleanup_count > 0,
            "answer_time_count": len(answer_values),
            "answer_includes_unmeasured_cleanup_count": answer_includes_unmeasured_cleanup_count,
            "total_preparation_s": total_preparation if (preparation_by_video or gemini_preparation_values) else None,
            "mean_preparation_s": avg(list(preparation_by_video.values())) if contestant == "ours-codex" else avg(gemini_preparation_values),
            "preparation_unknown_count": preparation_unknown_count,
            "total_end_to_end_s": total_end_to_end if answer_values or preparation_by_video or gemini_end_to_end_values else None,
            "mean_end_to_end_s": mean_end_to_end,
            "total_ingest_s": sum(ingest_by_video.values()) if ingest_by_video else None,
            "mean_ingest_s_per_video": avg(list(ingest_by_video.values())),
            "total_visual_index_s": total_visual_index if visual_index_by_video else None,
            "mean_visual_index_s_per_video": avg(list(visual_index_by_video.values())),
            "mean_upload_processing_s": avg(upload_processing_values),
            "total_cleanup_s": sum(cleanup_values) if cleanup_values else None,
            "mean_cleanup_s": avg(cleanup_values),
            "missing_results": missing_results,
            "total_gemini_cost_usd": sum(costs) if costs else None,
            "gemini_cost_count": len(costs),
        })
    return summary


def _summary_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Track | Contestant | N | Failures | Missing | Harness | Accuracy | Secondary | Interval IoU | Total tokens | Adjusted tokens | Mean answer (s; timed) | Prep total (s) | Prep unknown | Mean prep / video or run (s) | Cleanup total (s) | End-to-end total (s) | Gemini cost (USD) |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        track = row["track"]
        if track == "moment":
            accuracy = f"synth ≤0.5s {row['hit_0.5']} / real IoU≥0.5 {row['real_iou_0.5']} / negative {row['negative_control_hit']}"
            secondary = f"synth ≤0.2s {row['hit_0.2']} / real IoU≥0.7 {row['real_iou_0.7']} / midpoint≤2s {row['real_midpoint_2']}"
            iou = "—" if row["moment_iou"] is None else f"{row['moment_iou']:.3f}"
        elif track == "search":
            accuracy = f"≤2s {row['hit_2']} / ≤5s {row['hit_5']}"
            secondary = "midpoint to truth interval"
            iou = "—" if row["search_iou"] is None else f"{row['search_iou']:.3f}"
        else:
            accuracy = f"exact {row['exact']}"
            secondary = f"within ±1 {row['within_one']}"
            iou = "—"
        def measured(value: float | None, count: int, *, money: bool = False) -> str:
            displayed = "—" if value is None else ("$" if money else "") + (f"{value:.4f}" if money else f"{value:.0f}")
            return f"{displayed} ({count}/{row['n']}{'; incomplete' if count < row['n'] else ''})"

        tokens = measured(row["mean_tokens_total"], row["tokens_total_count"])
        adjusted_tokens = measured(row["mean_tokens_adjusted"], row["tokens_adjusted_count"])
        answer = "—" if row["mean_answer_s"] is None else f"{'≤' if row['mean_answer_is_upper_bound'] else ''}{row['mean_answer_s']:.1f} ({row['answer_time_count']}/{row['n']})"
        if row["answer_includes_unmeasured_cleanup_count"]:
            count = row["answer_includes_unmeasured_cleanup_count"]
            answer += f"; includes unmeasured cleanup for {count} {'run' if count == 1 else 'runs'}"
        preparation_total = "—" if row["total_preparation_s"] is None else f"{row['total_preparation_s']:.1f}"
        preparation_unknown = str(row["preparation_unknown_count"])
        preparation_mean = "—" if row["mean_preparation_s"] is None else f"{row['mean_preparation_s']:.1f}"
        cleanup_total = "—" if row["total_cleanup_s"] is None else f"{row['total_cleanup_s']:.1f}"
        end_to_end = "—" if row["total_end_to_end_s"] is None else f"{row['total_end_to_end_s']:.1f}"
        cost = measured(row["total_gemini_cost_usd"], row["gemini_cost_count"], money=True) if row["contestant"].startswith("gemini-") else "—"
        lines.append(
            f"| {track} | {row['contestant']} | {row['n']} | {row['failures']} | {row['missing_results']} | {row['harness_failures']} | "
            f"{accuracy} | {secondary} | {iou} | {tokens} | {adjusted_tokens} | {answer} | {preparation_total} | {preparation_unknown} | {preparation_mean} | {cleanup_total} | {end_to_end} | {cost} |"
        )
    if not summary:
        lines.append("No scored results found.")
    lines.extend([
        "",
        "Scoring: N is the number of questions in the supplied question set for each contestant and track. Missing result files count as failures and wrong answers; Missing reports their count. Synthetic moment, search, and count accuracy rates include negative controls, which have defined hit values. Real-moment IoU and midpoint rates and mean interval IoU use only interval truths, excluding negative controls and not-found truths; failed or missing results and point answers in that set count as misses (zero IoU). Use --questions with the selected question JSONL when scoring a partial run; the default question set scores the full benchmark.",
        "Resources: token means and Gemini cost show measured runs over N. Incomplete means are averages of measured runs only; incomplete Gemini cost is the recorded subtotal, excluding runs with unknown cost.",
        "Timing: adapter-reported Gemini answer time excludes upload, file processing, and cleanup. Legacy Gemini answer time is derived from wall time minus measured preparation and cleanup; a ≤ mean is an upper bound because one or more contributing runs include unmeasured Files API cleanup, with the affected count shown. Its mean reports the number of timed answers over N. Preparation total counts Codex transcript ingest plus visual indexing once per distinct video and Gemini upload + processing once per run. Prep unknown counts runs whose preparation/answer split was unavailable. Cleanup is reported separately and remains inside Gemini end-to-end duration. End-to-end total includes measured run durations; for Codex it adds once-only transcript ingest and visual indexing to answer times. Means divide the end-to-end total by the scored question count.",
    ])
    return "\n".join(lines)


def _hit(value: Any) -> str:
    return "hit" if value else "miss"


def _per_question_table(rows: list[dict[str, Any]]) -> str:
    """Per-question hit/miss lines. Uses only derived metrics and the contestant's answer."""
    lines = [
        "| ID | Track | Contestant | Status | Result | Contestant answer |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        metrics = row["metrics"]
        track = row["track"]
        # The label depends only on track and split, which the agent already sees, never on which
        # metric keys score_prediction returned for this key. The IoU and other number-valued
        # metrics are left out because they show whether the truth is an interval or not-found.
        if track == "count":
            result = f"exact {_hit(metrics.get('exact'))} / within ±1 {_hit(metrics.get('within_one'))}"
        elif track == "search":
            result = f"≤2s {_hit(metrics.get('hit_2'))}"
        elif track == "moment" and row.get("split") == "real":
            # Interval truths report hit_iou_0.5. Not-found truths only have the tolerance hits.
            hit = metrics["hit_iou_0.5"] if "hit_iou_0.5" in metrics else metrics.get("hit_0.5")
            result = f"real IoU≥0.5 {_hit(hit)}"
        elif track == "moment":
            result = f"≤0.5s {_hit(metrics.get('hit_0.5'))}"
        else:
            result = f"≤2s {_hit(metrics.get('hit_2'))}"
        answer = "—" if row["answer"] is None else json.dumps(row["answer"], ensure_ascii=False)
        answer = answer.replace("|", "\\|")
        lines.append(f"| {row['id']} | {track} | {row['contestant']} | {row['status']} | {result} | {answer} |")
    if not rows:
        lines.append("No scored questions.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_DATA_ROOT / "questions.jsonl")
    parser.add_argument("--key", type=Path, default=key_path())
    parser.add_argument("--results", type=Path, default=DEFAULT_DATA_ROOT / "runs")
    parser.add_argument("--contestant", action="append", help="Limit scoring to one or more contestant names")
    parser.add_argument(
        "--per-question", action="store_true",
        help="After the summary, print one line per scored question: id, track, contestant, status, "
             "hit or miss, and the contestant's answer. It never prints ground-truth answers or any "
             "other answer-key field, so the output is safe to show agents that must not learn the truth values.",
    )
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    keys = read_jsonl(args.key)
    results = _load_results(args.results)
    if args.contestant:
        wanted = set(args.contestant)
        results = [row for row in results if row.get("contestant") in wanted]
        contestant_names = sorted(wanted)
    else:
        contestant_names = sorted(set(CONTESTANTS) | {
            path.name for path in args.results.iterdir() if path.is_dir()
        }) if args.results.exists() else list(CONTESTANTS)
    per_question: list[dict[str, Any]] | None = [] if args.per_question else None
    summary = score_rows(questions, keys, results, contestants=contestant_names, per_question=per_question)
    print(_summary_table(summary))
    if per_question is not None:
        print()
        print(_per_question_table(per_question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
