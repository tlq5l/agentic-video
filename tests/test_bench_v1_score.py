import json
from benchmarks.bench_v1 import score as score_module
from benchmarks.bench_v1.common import CONTESTANTS
from benchmarks.bench_v1.score import _summary_table, interval_iou, score_prediction, score_rows
import pytest


def test_moment_point_tolerances():
    scored = score_prediction("moment", 10.0, 10.19)
    assert scored["hit_0.2"] is True
    assert scored["hit_0.5"] is True
    assert scored["hit_1.0"] is True
    assert scored["iou"] is None


def test_moment_interval_requires_both_boundaries_and_reports_iou():
    scored = score_prediction("moment", [1.0, 2.0], [1.1, 2.21])
    assert scored["hit_0.2"] is False
    assert scored["hit_0.5"] is True
    assert scored["hit_1.0"] is True
    assert scored["iou"] == interval_iou([1.1, 2.21], [1.0, 2.0])
    assert round(scored["iou"], 3) == 0.744


def test_point_prediction_inside_moment_truth_interval_is_a_hit():
    scored = score_prediction("moment", [3.0, 4.0], 3.5)
    assert scored["hit_0.2"] is True
    assert scored["iou"] is None


def test_search_tolerance_uses_distance_to_truth_interval():
    scored = score_prediction("search", [10.0, 12.0], [14.0, 15.0])
    assert scored["hit_2"] is False
    assert scored["hit_5"] is True
    assert scored["iou"] == 0.0


def test_search_interval_overlap_and_negative_control():
    hit = score_prediction("search", [10.0, 12.0], 12.0)
    negative = score_prediction("search", "not_found", "not found")
    assert hit["hit_2"] is True
    assert negative["hit_2"] is True
    assert negative["hit_5"] is True


def test_count_exact_and_within_one():
    assert score_prediction("count", 5, 5) == {"exact": True, "within_one": True}
    assert score_prediction("count", 5, 4) == {"exact": False, "within_one": True}
    assert score_prediction("count", 0, 2) == {"exact": False, "within_one": False}


def test_real_moment_uses_best_matching_qvh_window_and_iou_thresholds():
    scored = score_prediction(
        "moment", [0.0, 2.0], [20.5, 23.5],
        truth_windows=[[0.0, 2.0], [20.0, 24.0]],
    )
    assert scored["iou"] == 0.75
    assert scored["hit_iou_0.5"] is True
    assert scored["hit_iou_0.7"] is True
    assert scored["midpoint_error"] == 0.0
    assert scored["midpoint_within_2"] is True


def test_real_moment_reports_iou_05_but_not_07_when_overlap_is_coarse():
    scored = score_prediction(
        "moment", [0.0, 2.0], [19.0, 25.0],
        truth_windows=[[0.0, 2.0], [20.0, 24.0]],
    )
    assert scored["iou"] == 2 / 3
    assert scored["hit_iou_0.5"] is True
    assert scored["hit_iou_0.7"] is False
    assert scored["midpoint_within_2"] is True


def test_real_moment_point_at_window_midpoint_misses_all_positional_rates_and_mean_iou():
    scored = score_prediction(
        "moment", [10.0, 14.0], 12.0,
        truth_windows=[[10.0, 14.0]],
    )
    assert scored["iou"] == 0.0
    assert scored["hit_iou_0.5"] is False
    assert scored["hit_iou_0.7"] is False
    assert scored["midpoint_within_2"] is False

    row = score_rows(
        questions=[{"id": "m1", "track": "moment", "split": "real"}],
        keys=[{"id": "m1", "truth": [10.0, 14.0], "truth_windows": [[10.0, 14.0]]}],
        results=[{"id": "m1", "contestant": "ours-codex", "status": "ok", "answer": 12.0}],
    )[0]
    assert (row["real_iou_0.5"], row["real_iou_0.7"], row["real_midpoint_2"]) == (
        "0.0%", "0.0%", "0.0%"
    )
    assert row["moment_iou"] == 0.0


def test_harness_failures_are_reported_and_count_wrong_in_accuracy():
    summary = score_rows(
        questions=[{"id": "m1", "track": "moment", "split": "synthetic"}],
        keys=[{"id": "m1", "truth": 10.0}],
        results=[{
            "id": "m1", "contestant": "ours-codex", "status": "error",
            "failure_type": "harness", "answer": 10.0,
            "usage": {"total_tokens": 100000, "adjusted_total_tokens": 25000},
        }],
    )[0]
    assert summary["harness_failures"] == 1
    assert summary["failures"] == 1
    assert summary["hit_0.5"] == "0.0%"
    assert summary["mean_tokens_total"] == 100000
    assert summary["mean_tokens_adjusted"] == 25000


def test_harness_failure_counts_wrong_for_negative_control_accuracy():
    summary = score_rows(
        questions=[{"id": "m1", "track": "moment", "split": "real", "cue_type": "negative_control"}],
        keys=[{"id": "m1", "truth": "not_found"}],
        results=[{
            "id": "m1", "contestant": "ours-codex", "status": "error",
            "failure_type": "harness", "answer": None,
        }],
    )[0]
    assert summary["harness_failures"] == 1
    assert summary["negative_control_hit"] == "0.0%"


def test_identical_timeout_counts_wrong_for_all_contestants():
    questions = [{"id": "c1", "track": "count", "split": "synthetic"}]
    keys = [{"id": "c1", "truth": 3}]
    contestants = ["ours-codex", "gemini-agentic", "gemini-plain"]
    results = [
        {
            "id": "c1", "contestant": contestant, "status": "error",
            "failure_type": "timeout", "answer": None,
        }
        for contestant in contestants
    ]

    summary = score_rows(questions, keys, results)
    assert {row["contestant"]: row["exact"] for row in summary} == {
        contestant: "0.0%" for contestant in contestants
    }
    assert {row["contestant"]: row["failures"] for row in summary} == {
        contestant: 1 for contestant in contestants
    }


def test_missing_question_result_counts_wrong_in_question_set_denominator():
    questions = [
        {"id": "c1", "track": "count", "split": "synthetic"},
        {"id": "c2", "track": "count", "split": "synthetic"},
    ]
    summary = score_rows(
        questions,
        keys=[{"id": "c1", "truth": 3}, {"id": "c2", "truth": 7}],
        results=[{
            "id": "c1", "contestant": "ours-codex", "status": "ok", "answer": 3,
        }],
    )[0]
    table = _summary_table([summary])

    assert summary["n"] == 2
    assert summary["failures"] == 1
    assert summary["missing_results"] == 1
    assert summary["exact"] == "50.0%"
    assert "| count | ours-codex | 2 | 1 | 1 |" in table


def test_explicit_contestant_with_no_result_files_is_all_missing():
    summary = score_rows(
        questions=[{"id": "c1", "track": "count", "split": "synthetic"}],
        keys=[{"id": "c1", "truth": 3}],
        results=[],
        contestants=["gemini-plain"],
    )[0]

    assert summary["n"] == 1
    assert summary["failures"] == 1
    assert summary["missing_results"] == 1
    assert summary["exact"] == "0.0%"


def test_score_cli_reports_missing_rows_for_default_contestants_without_result_files(monkeypatch, tmp_path, capsys):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"id":"c1","track":"count","split":"synthetic"}\n')
    key_path = tmp_path / "key.jsonl"
    key_path.write_text('{"id":"c1","truth":3}\n')
    results_path = tmp_path / "no-results"
    monkeypatch.setattr("sys.argv", [
        "score.py", "--questions", str(questions_path), "--key", str(key_path),
        "--results", str(results_path),
    ])

    assert score_module.main() == 0
    output = capsys.readouterr().out
    for contestant in CONTESTANTS:
        assert f"| count | {contestant} | 1 | 1 | 1 |" in output


def test_score_cli_uses_relocated_key_when_key_flag_is_omitted(monkeypatch, tmp_path, capsys):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"id":"c1","track":"count","split":"synthetic"}\n')
    relocated_key = tmp_path / "relocated-sealed" / "answer_key.jsonl"
    relocated_key.parent.mkdir()
    relocated_key.write_text('{"id":"c1","truth":3}\n')
    monkeypatch.setenv("AGENTIC_VIDEO_BENCH_KEY", str(relocated_key))
    monkeypatch.setattr("sys.argv", [
        "score.py", "--questions", str(questions_path), "--results", str(tmp_path / "no-results"),
    ])

    assert score_module.main() == 0
    assert "| count | gemini-plain | 1 | 1 | 1 |" in capsys.readouterr().out


def test_resource_table_marks_missing_usage_and_cost_as_incomplete():
    questions = [
        {"id": "c1", "track": "count", "split": "synthetic"},
        {"id": "c2", "track": "count", "split": "synthetic"},
    ]
    keys = [{"id": "c1", "truth": 3}, {"id": "c2", "truth": 4}]
    measured_result = {
        "id": "c1", "contestant": "gemini-plain", "status": "ok", "answer": 3,
        "usage": {"total_tokens": 100}, "estimated_cost_usd": 0.25,
    }
    timed_out_result = {
        "id": "c2", "contestant": "gemini-plain", "status": "error",
        "failure_type": "timeout", "answer": None,
    }

    complete = score_rows(questions[:1], keys[:1], [measured_result])[0]
    complete_table = _summary_table([complete])
    assert "| 100 (1/1) | 100 (1/1) |" in complete_table
    assert "$0.2500 (1/1)" in complete_table
    assert "incomplete" not in complete_table.splitlines()[2]

    incomplete = score_rows(questions, keys, [measured_result, timed_out_result])[0]
    incomplete_table = _summary_table([incomplete])
    assert incomplete["tokens_total_count"] == 1
    assert incomplete["tokens_adjusted_count"] == 1
    assert incomplete["gemini_cost_count"] == 1
    assert "| 100 (1/2; incomplete) | 100 (1/2; incomplete) |" in incomplete_table
    assert "$0.2500 (1/2; incomplete)" in incomplete_table


def test_successful_answer_without_usage_is_unmeasured_not_zero():
    from benchmarks.bench_v1.run_contestant import _aggregate_usage, _gemini_cost

    questions = [{"id": "c1", "track": "count", "split": "synthetic"}]
    keys = [{"id": "c1", "truth": 3}]
    # Adapter shape when the API answered but returned no usage metadata.
    adapter_usage = {"total_tokens": None, "billable_tokens": None, "usage_missing_requests": 1, "requests": [None]}
    repaired = _aggregate_usage({"input_tokens": 50, "output_tokens": 5}, adapter_usage)
    assert repaired["total_tokens"] is None
    assert _gemini_cost(adapter_usage) is None and _gemini_cost(repaired) is None
    for usage in (adapter_usage, repaired):
        result = {"id": "c1", "contestant": "gemini-plain", "status": "ok", "answer": 3,
                  "usage": usage, "estimated_cost_usd": _gemini_cost(usage)}
        row = score_rows(questions, keys, [result])[0]
        assert row["tokens_total_count"] == 0
        assert row["gemini_cost_count"] == 0
        assert "(0/1; incomplete)" in _summary_table([row])
    stale_estimate = {"id": "c1", "contestant": "gemini-plain", "status": "ok", "answer": 3,
                      "usage": repaired, "estimated_cost_usd": 0.25}
    assert score_rows(questions, keys, [stale_estimate])[0]["gemini_cost_count"] == 0


def test_legacy_gemini_usage_without_marker_scores_as_before():
    questions = [{"id": "c1", "track": "count", "split": "synthetic"}]
    keys = [{"id": "c1", "truth": 3}]
    result = {"id": "c1", "contestant": "gemini-plain", "status": "ok", "answer": 3,
              "usage": {"billable_tokens": 100}}
    row = score_rows(questions, keys, [result])[0]
    assert row["tokens_total_count"] == 1
    assert row["gemini_cost_count"] == 1


def test_score_reports_answer_preparation_and_end_to_end_time_boundaries():
    questions = [
        {"id": "s1", "track": "search", "split": "real", "video_path": "/videos/long.mp4"},
        {"id": "s2", "track": "search", "split": "real", "video_path": "/videos/long.mp4"},
    ]
    keys = [{"id": "s1", "truth": [1.0, 2.0]}, {"id": "s2", "truth": [3.0, 4.0]}]
    results = [
        {
            "id": "s1", "contestant": "ours-codex", "status": "ok",
            "answer": [1.0, 2.0], "ingest_time_s": 90.0, "visual_index_time_s": 15.0, "wall_time_s": 30.0,
        },
        {
            "id": "s2", "contestant": "ours-codex", "status": "ok",
            "answer": [3.0, 4.0], "ingest_time_s": 90.0, "visual_index_time_s": 15.0, "wall_time_s": 32.0,
        },
        {
            "id": "s1", "contestant": "gemini-agentic", "status": "ok",
            "answer": [1.0, 2.0], "upload_processing_s": 8.0, "wall_time_s": 40.0,
            "cleanup_time_s": 2.0,
        },
        {
            "id": "s2", "contestant": "gemini-agentic", "status": "ok",
            "answer": [3.0, 4.0], "upload_processing_s": 12.0, "wall_time_s": 44.0,
            "cleanup_time_s": 3.0,
        },
    ]

    summary = score_rows(questions, keys, results)
    ours = next(row for row in summary if row["contestant"] == "ours-codex")
    gemini = next(row for row in summary if row["contestant"] == "gemini-agentic")
    table = _summary_table(summary)

    assert ours["total_ingest_s"] == 90.0
    assert ours["mean_ingest_s_per_video"] == 90.0
    assert ours["total_visual_index_s"] == 15.0
    assert ours["mean_visual_index_s_per_video"] == 15.0
    assert ours["mean_answer_s"] == 31.0
    assert ours["mean_preparation_s"] == 105.0
    assert ours["total_preparation_s"] == 105.0
    assert ours["total_end_to_end_s"] == 167.0
    assert gemini["mean_upload_processing_s"] == 10.0
    assert gemini["mean_answer_s"] == 29.5
    assert gemini["answer_includes_unmeasured_cleanup_count"] == 0
    assert gemini["total_preparation_s"] == 20.0
    assert gemini["total_end_to_end_s"] == 84.0
    assert gemini["total_cleanup_s"] == 5.0
    assert gemini["mean_cleanup_s"] == 2.5
    assert "Mean answer (s; timed)" in table
    assert "Prep total (s)" in table
    assert "Cleanup total (s)" in table
    assert "End-to-end total (s)" in table
    assert "Codex transcript ingest plus visual indexing once per distinct video" in table


def test_mixed_saved_and_new_results_combine_first_transcript_and_visual_times_per_video():
    video = "/videos/shared.mp4"
    questions = [
        {"id": "s1", "track": "search", "split": "real", "video_path": video},
        {"id": "s2", "track": "search", "split": "real", "video_path": video},
    ]
    keys = [{"id": "s1", "truth": [1.0, 2.0]}, {"id": "s2", "truth": [3.0, 4.0]}]
    # s1 is a saved result from before aa6442b: preparation was transcript
    # ingest. s2 is a fresh result with split transcript and visual timings.
    results = [
        {
            "id": "s1", "contestant": "ours-codex", "status": "ok",
            "answer": [1.0, 2.0], "wall_time_s": 2.0, "preparation_time_s": 10.0,
        },
        {
            "id": "s2", "contestant": "ours-codex", "status": "ok",
            "answer": [3.0, 4.0], "answer_time_s": 3.0,
            "ingest_time_s": 10.0, "visual_index_time_s": 20.0,
            "preparation_time_s": 30.0,
        },
    ]

    row = score_rows(questions, keys, results, contestants=["ours-codex"])[0]
    assert row["total_ingest_s"] == 10.0
    assert row["total_visual_index_s"] == 20.0
    assert row["total_preparation_s"] == 30.0
    assert row["mean_preparation_s"] == 30.0
    assert row["total_end_to_end_s"] == 35.0


def test_legacy_preparation_only_results_match_parent_commit_timing():
    # Under aa6442b^, preparation_time_s was the once-per-video ingest value.
    questions = [{"id": "s1", "track": "search", "split": "real", "video_path": "/videos/legacy.mp4"}]
    keys = [{"id": "s1", "truth": [1.0, 2.0]}]
    results = [{
        "id": "s1", "contestant": "ours-codex", "status": "ok",
        "answer": [1.0, 2.0], "wall_time_s": 2.0, "preparation_time_s": 10.0,
    }]

    row = score_rows(questions, keys, results, contestants=["ours-codex"])[0]
    assert row["total_ingest_s"] == 10.0
    assert row["mean_ingest_s_per_video"] == 10.0
    assert row["total_visual_index_s"] is None
    assert row["mean_visual_index_s_per_video"] is None
    assert row["total_preparation_s"] == 10.0
    assert row["mean_preparation_s"] == 10.0
    assert row["total_end_to_end_s"] == 12.0


def test_legacy_gemini_timing_without_upload_measurement_is_not_guessed_as_answer_time():
    summary = score_rows(
        questions=[{"id": "s1", "track": "search", "split": "real"}],
        keys=[{"id": "s1", "truth": [1.0, 2.0]}],
        results=[{
            "id": "s1", "contestant": "gemini-plain", "status": "error",
            "failure_type": "timeout", "wall_time_s": 60.0,
        }],
    )[0]

    assert summary["mean_answer_s"] is None
    assert summary["answer_time_count"] == 0
    assert summary["preparation_unknown_count"] == 1
    assert summary["total_end_to_end_s"] == 60.0


def test_legacy_gemini_answer_time_flags_unmeasured_cleanup_in_row():
    row = score_rows(
        questions=[{"id": "s1", "track": "search", "split": "real"}],
        keys=[{"id": "s1", "truth": [1.0, 2.0]}],
        results=[{
            "id": "s1", "contestant": "gemini-plain", "status": "ok",
            "answer": [1.0, 2.0], "wall_time_s": 40.0,
            "upload_processing_s": 8.0,
        }],
    )[0]

    assert row["mean_answer_s"] == 32.0
    assert row["total_end_to_end_s"] == 40.0
    assert row["mean_answer_is_upper_bound"] is True
    assert row["answer_includes_unmeasured_cleanup_count"] == 1
    assert "≤32.0 (1/1); includes unmeasured cleanup for 1 run" in _summary_table([row])


def test_mixed_gemini_answer_times_keep_legacy_run_and_mark_mean_as_upper_bound():
    row = score_rows(
        questions=[{"id": "s1", "track": "search"}, {"id": "s2", "track": "search"}],
        keys=[{"id": "s1", "truth": [1.0, 2.0]}, {"id": "s2", "truth": [1.0, 2.0]}],
        results=[
            {"id": "s1", "contestant": "gemini-plain", "status": "ok", "answer": [1.0, 2.0],
             "wall_time_s": 40.0, "upload_processing_s": 8.0},
            {"id": "s2", "contestant": "gemini-plain", "status": "ok", "answer": [1.0, 2.0],
             "answer_time_s": 20.0, "preparation_time_s": 5.0, "cleanup_time_s": 1.0},
        ],
    )[0]

    assert row["mean_answer_s"] == 26.0
    assert row["answer_time_count"] == 2
    assert row["mean_answer_is_upper_bound"] is True
    assert "≤26.0 (2/2); includes unmeasured cleanup for 1 run" in _summary_table([row])


def test_new_gemini_answer_time_is_not_reduced_by_cleanup_again():
    row = score_rows(
        questions=[{"id": "s1", "track": "search", "split": "real"}],
        keys=[{"id": "s1", "truth": [1.0, 2.0]}],
        results=[{
            "id": "s1", "contestant": "gemini-plain", "status": "ok",
            "answer": [1.0, 2.0], "answer_time_s": 30.0,
            "preparation_time_s": 8.0, "cleanup_time_s": 2.0,
            "end_to_end_time_s": 40.0,
        }],
    )[0]

    assert row["mean_answer_s"] == 30.0
    assert row["mean_answer_is_upper_bound"] is False
    assert row["answer_includes_unmeasured_cleanup_count"] == 0


def test_real_moment_not_found_does_not_lower_positional_rates_but_failure_does():
    questions = [
        {"id": "hit", "track": "moment", "split": "real"},
        {"id": "negative", "track": "moment", "split": "real", "cue_type": "NEGATIVE-control"},
        {"id": "failed", "track": "moment", "split": "real"},
        {"id": "missing", "track": "moment", "split": "real"},
    ]
    keys = [
        {"id": "hit", "truth": [1.0, 2.0], "truth_windows": [[1.0, 2.0]]},
        {"id": "negative", "truth": "not_found"},
        {"id": "failed", "truth": [1.0, 2.0], "truth_windows": [[1.0, 2.0]]},
        {"id": "missing", "truth": [1.0, 2.0]},
    ]
    results = [
        {"id": "hit", "contestant": "ours-codex", "status": "ok", "answer": [1.0, 2.0]},
        {"id": "negative", "contestant": "ours-codex", "status": "ok", "answer": "not_found"},
        {"id": "failed", "contestant": "ours-codex", "status": "error",
         "failure_type": "harness", "answer": [1.0, 2.0]},
    ]

    row = score_rows(questions[:2], keys[:2], results[:2])[0]
    assert row["negative_control_hit"] == "100.0%"
    assert (row["real_iou_0.5"], row["real_iou_0.7"], row["real_midpoint_2"]) == (
        "100.0%", "100.0%", "100.0%"
    )

    row = score_rows(questions, keys, results)[0]
    assert row["failures"] == 2
    assert row["missing_results"] == 1
    assert row["harness_failures"] == 1
    assert (row["real_iou_0.5"], row["real_iou_0.7"], row["real_midpoint_2"]) == (
        "33.3%", "33.3%", "33.3%"
    )


@pytest.mark.parametrize("split,truth,answer,extra_key,rate_fields", [
    ("synthetic", [1.0, 2.0], [1.0, 2.0], {}, ("hit_0.2", "hit_0.5", "hit_1.0")),
    ("real", [1.0, 2.0], [1.0, 2.0], {"truth_windows": [[1.0, 2.0]]},
     ("real_iou_0.5", "real_iou_0.7", "real_midpoint_2")),
])
def test_moment_accuracy_means_include_failed_and_missing_results(
    split, truth, answer, extra_key, rate_fields,
):
    questions = [{"id": f"m{i}", "track": "moment", "split": split} for i in range(3)]
    keys = [{"id": f"m{i}", "truth": truth, **extra_key} for i in range(3)]
    results = [
        {"id": "m0", "contestant": "ours-codex", "status": "ok", "answer": answer},
        {"id": "m1", "contestant": "ours-codex", "status": "error",
         "failure_type": "timeout", "answer": answer},
    ]

    row = score_rows(questions, keys, results)[0]
    assert row["failures"] == 2
    assert row["missing_results"] == 1
    assert {name: row[name] for name in rate_fields} == {name: "33.3%" for name in rate_fields}
    assert row["moment_iou"] == pytest.approx(1 / 3)


def test_search_accuracy_means_include_failed_and_missing_results():
    questions = [{"id": f"s{i}", "track": "search", "split": "real"} for i in range(3)]
    keys = [{"id": f"s{i}", "truth": [1.0, 2.0]} for i in range(3)]
    results = [
        {"id": "s0", "contestant": "ours-codex", "status": "ok", "answer": [1.0, 2.0]},
        {"id": "s1", "contestant": "ours-codex", "status": "error",
         "failure_type": "timeout", "answer": [1.0, 2.0]},
    ]

    row = score_rows(questions, keys, results)[0]
    assert row["failures"] == 2
    assert row["missing_results"] == 1
    assert row["hit_2"] == row["hit_5"] == "33.3%"
    assert row["search_iou"] == pytest.approx(1 / 3)


def test_count_accuracy_rates_include_failed_and_missing_results():
    questions = [{"id": f"c{i}", "track": "count", "split": "synthetic"} for i in range(3)]
    keys = [{"id": f"c{i}", "truth": 2} for i in range(3)]
    results = [
        {"id": "c0", "contestant": "ours-codex", "status": "ok", "answer": 2},
        {"id": "c1", "contestant": "ours-codex", "status": "error",
         "failure_type": "timeout", "answer": 2},
    ]

    row = score_rows(questions, keys, results)[0]
    assert row["failures"] == 2
    assert row["missing_results"] == 1
    assert row["exact"] == row["within_one"] == "33.3%"
    assert row["moment_iou"] is None
    assert row["search_iou"] is None


@pytest.mark.parametrize("cue_type", ["negative control", "negative_control", "NEGATIVE-CONTROL", " Negative  _control "])
def test_synthetic_count_negative_control_includes_failed_and_missing(cue_type):
    questions = [{"id": f"c{i}", "track": "count", "split": "synthetic",
                  "cue_type": cue_type} for i in range(3)]
    keys = [{"id": f"c{i}", "truth": 0} for i in range(3)]
    results = [
        {"id": "c0", "contestant": "ours-codex", "status": "ok", "answer": 0},
        {"id": "c1", "contestant": "ours-codex", "status": "error",
         "failure_type": "timeout", "answer": 0},
    ]

    row = score_rows(questions, keys, results)[0]
    assert row["negative_control_hit"] == "33.3%"


def test_undefined_interval_iou_counts_as_zero_in_track_mean():
    row = score_rows(
        questions=[{"id": "s0", "track": "search", "split": "real"},
                   {"id": "s1", "track": "search", "split": "real"}],
        keys=[{"id": "s0", "truth": [1.0, 2.0]},
              {"id": "s1", "truth": [1.0, 2.0]}],
        results=[{"id": "s0", "contestant": "ours-codex", "status": "ok", "answer": [1.0, 2.0]},
                 {"id": "s1", "contestant": "ours-codex", "status": "ok", "answer": 1.5}],
    )[0]
    assert row["hit_2"] == "100.0%"
    assert row["search_iou"] == 0.5


@pytest.mark.parametrize("track,iou_field", [("moment", "moment_iou"), ("search", "search_iou")])
def test_mean_interval_iou_excludes_negative_controls_but_counts_undefined_answers(track, iou_field):
    questions = [
        {"id": name, "track": track, "split": "synthetic", "cue_type": cue_type}
        for name, cue_type in [
            ("correct", "visual_only"), ("harness", "visual_only"),
            ("missing", "visual_only"), ("point", "visual_only"),
            ("negative_cue", " NEGATIVE - control "), ("not_found", "visual_only"),
        ]
    ]
    keys = [
        {"id": name, "truth": "not_found" if name == "not_found" else [1.0, 2.0]}
        for name in ("correct", "harness", "missing", "point", "negative_cue", "not_found")
    ]
    results = [
        {"id": "correct", "contestant": "ours-codex", "status": "ok", "answer": [1.0, 2.0]},
        {"id": "harness", "contestant": "ours-codex", "status": "error",
         "failure_type": "harness", "answer": [1.0, 2.0]},
        {"id": "point", "contestant": "ours-codex", "status": "ok", "answer": 1.5},
        {"id": "negative_cue", "contestant": "ours-codex", "status": "ok", "answer": [1.0, 2.0]},
        {"id": "not_found", "contestant": "ours-codex", "status": "ok", "answer": "not_found"},
    ]

    row = score_rows(questions, keys, results)[0]
    assert row["n"] == 6
    assert row["failures"] == 2
    assert row["harness_failures"] == 1
    assert row["missing_results"] == 1
    assert row[iou_field] == pytest.approx(0.25)


def test_score_cli_follows_symlinked_contestant_folders(monkeypatch, tmp_path, capsys):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"id":"c1","track":"count","split":"synthetic"}\n')
    key_path = tmp_path / "key.jsonl"
    key_path.write_text('{"id":"c1","truth":3}\n')
    real_folder = tmp_path / "runs" / "gemini-agentic"
    real_folder.mkdir(parents=True)
    (real_folder / "c1.json").write_text(
        '{"id":"c1","contestant":"gemini-agentic","status":"ok","answer":3}'
    )
    # A link back to an ancestor must not loop forever.
    (real_folder / "loop").symlink_to(tmp_path / "runs", target_is_directory=True)
    results_path = tmp_path / "runs-v2"
    results_path.mkdir()
    (results_path / "gemini-agentic").symlink_to(
        "../runs/gemini-agentic", target_is_directory=True
    )

    loaded = score_module._load_results(results_path)
    assert [(row["contestant"], row["id"]) for row in loaded] == [("gemini-agentic", "c1")]

    monkeypatch.setattr("sys.argv", [
        "score.py", "--questions", str(questions_path), "--key", str(key_path),
        "--results", str(results_path), "--contestant", "gemini-agentic",
    ])
    assert score_module.main() == 0
    output = capsys.readouterr().out
    assert "| count | gemini-agentic | 1 | 0 | 0 | 0 | exact 100.0% |" in output


def test_score_cli_per_question_shows_hits_and_answers_but_never_truth(monkeypatch, tmp_path, capsys):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        '{"id":"c1","track":"count","split":"synthetic"}\n'
        '{"id":"s1","track":"search","split":"real"}\n'
        '{"id":"m1","track":"moment","split":"synthetic"}\n'
    )
    key_path = tmp_path / "key.jsonl"
    key_path.write_text(
        '{"id":"c1","truth":4217,"note":"SECRET-KEY-NOTE"}\n'
        '{"id":"s1","truth":[123.25,130.75]}\n'
        '{"id":"m1","truth":[61.125,63.875]}\n'
    )
    results_path = tmp_path / "runs" / "ours-codex"
    results_path.mkdir(parents=True)
    (results_path / "c1.json").write_text(
        '{"id":"c1","contestant":"ours-codex","status":"ok","answer":4218}'
    )
    (results_path / "s1.json").write_text(
        '{"id":"s1","contestant":"ours-codex","status":"ok","answer":[124.0,129.0]}'
    )
    monkeypatch.setattr("sys.argv", [
        "score.py", "--questions", str(questions_path), "--key", str(key_path),
        "--results", str(results_path.parent), "--contestant", "ours-codex", "--per-question",
    ])

    assert score_module.main() == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    count_line = next(line for line in lines if line.startswith("| c1 |"))
    search_line = next(line for line in lines if line.startswith("| s1 |"))
    moment_line = next(line for line in lines if line.startswith("| m1 |"))
    assert count_line == "| c1 | count | ours-codex | ok | exact miss / within ±1 hit | 4218 |"
    assert search_line == "| s1 | search | ours-codex | ok | ≤2s hit | [124.0, 129.0] |"
    assert moment_line == "| m1 | moment | ours-codex | missing | ≤0.5s miss | — |"
    for secret in ("4217", "123.25", "130.75", "61.125", "63.875", "SECRET-KEY-NOTE"):
        assert secret not in output


def test_per_question_lines_do_not_depend_on_truth_type(monkeypatch, tmp_path, capsys):
    # Each group has the same track, split and contestant answer but a different key shape
    # (interval, bare point, not-found, non-numeric). Lines must match apart from the id.
    truths = {"int": [100.5, 110.5], "pt": 105.5, "nf": "not_found", "bad": "SECRET-BAD-TRUTH"}
    answers = {"hit": [100.5, 110.5], "far": [300.0, 310.0], "nf": "not_found"}
    questions, keys, results = [], [], []
    for track, split in [("search", "real"), ("moment", "real"), ("moment", "synthetic"), ("mystery", "real")]:
        for truth_name, truth in truths.items():
            for answer_name, answer in answers.items():
                for failed in (False, True):
                    qid = f"{track}-{split}-{truth_name}-{answer_name}-{'fail' if failed else 'ok'}"
                    questions.append({"id": qid, "track": track, "split": split,
                                      "cue_type": "negative_control" if truth_name == "nf" else "visual_only"})
                    keys.append({"id": qid, "truth": truth, "truth_windows": [truth], "note": "SECRET-KEY-NOTE"})
                    results.append({"id": qid, "contestant": "ours-codex", "answer": answer,
                                    "status": "error" if failed else "ok",
                                    **({"failure_type": "harness"} if failed else {})})
    tmp_path.joinpath("q.jsonl").write_text("".join(json.dumps(row) + "\n" for row in questions))
    tmp_path.joinpath("k.jsonl").write_text("".join(json.dumps(row) + "\n" for row in keys))
    results_path = tmp_path / "runs" / "ours-codex"
    results_path.mkdir(parents=True)
    for row in results:
        results_path.joinpath(f"{row['id']}.json").write_text(json.dumps({**row, "contestant": "ours-codex"}))
    monkeypatch.setattr("sys.argv", [
        "score.py", "--questions", str(tmp_path / "q.jsonl"), "--key", str(tmp_path / "k.jsonl"),
        "--results", str(results_path.parent), "--contestant", "ours-codex", "--per-question",
    ])

    assert score_module.main() == 0
    output = capsys.readouterr().out
    for secret in ("100.5", "110.5", "105.5", "SECRET-BAD-TRUTH", "SECRET-KEY-NOTE"):
        assert secret not in output.replace("[100.5, 110.5]", "")
    shapes: dict[tuple, set[str]] = {}
    for line in output.splitlines():
        if line.startswith("| search-") or line.startswith("| moment-") or line.startswith("| mystery-"):
            qid, track, _, status, result, answer = [cell.strip() for cell in line.strip("|").split(" | ")]
            _, split, _, answer_name, outcome = qid.split("-")
            label = result.rsplit(" ", 1)[0]
            shapes.setdefault((track, split, status, answer_name, outcome), set()).add(label)
    assert len(shapes) == 4 * 3 * 2
    # The label never varies with the truth, so it cannot reveal the truth type.
    assert all(len(labels) == 1 for labels in shapes.values())
    assert shapes[("search", "real", "ok", "far", "ok")] == {"≤2s"}
    assert shapes[("moment", "real", "ok", "far", "ok")] == {"real IoU≥0.5"}
    assert shapes[("moment", "synthetic", "ok", "far", "ok")] == {"≤0.5s"}
    assert shapes[("mystery", "real", "ok", "far", "ok")] == {"≤2s"}


def test_per_question_real_moment_hit_for_interval_and_not_found_truths():
    def line(truth, answer, status="ok"):
        rows = []
        score_rows(
            [{"id": "m", "track": "moment", "split": "real"}], [{"id": "m", "truth": truth, "truth_windows": [truth]}],
            [{"id": "m", "contestant": "ours-codex", "status": status, "answer": answer}], per_question=rows)
        return score_module._per_question_table(rows).splitlines()[2]

    assert line([10.0, 20.0], [10.0, 20.0]) == "| m | moment | ours-codex | ok | real IoU≥0.5 hit | [10.0, 20.0] |"
    assert line([10.0, 20.0], "not_found") == "| m | moment | ours-codex | ok | real IoU≥0.5 miss | \"not_found\" |"
    assert line("not_found", "not_found") == "| m | moment | ours-codex | ok | real IoU≥0.5 hit | \"not_found\" |"
    assert line("not_found", [10.0, 20.0]) == "| m | moment | ours-codex | ok | real IoU≥0.5 miss | [10.0, 20.0] |"
    assert line("not_found", "not_found", status="error") == "| m | moment | ours-codex | error | real IoU≥0.5 miss | \"not_found\" |"


def test_score_cli_help_says_per_question_hides_truth(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["score.py", "--help"])
    with pytest.raises(SystemExit):
        score_module.main()
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--per-question" in help_text
    assert "never prints ground-truth answers" in help_text
