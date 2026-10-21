"""Fixer guard rails, usage-limit handling, and two false-flag fixes."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.cache import MediaCache
from src.loop import (_build_candidates, _fix_brief, _fix_times, _protected_changes, _usage_limit_message,
                      main, run_loop)
from src.media.audio import _outlier_sections, _present_at_file_start, sync_check

RATE = 48000
INTENT = ("The noise riser grows from 8.750 s to 10.000 s. "
          "A muffled, low-passed bridge runs from 12.500 s through 15.000 s.")


def _config(tmp_path: Path, fixer_cmd, *, ladder=None, intent=True, max_rounds=6) -> Path:
    config = {
        "project_dir": str(tmp_path), "build_cmd": "true", "video": "render.mp4", "judge": "none",
        "max_rounds": max_rounds, "fixer_cmd": fixer_cmd, "out": str(tmp_path / "loop-output"),
    }
    if intent:
        (tmp_path / "INTENT.md").write_text(INTENT, encoding="utf-8")
        config["intent"] = "INTENT.md"
    if ladder is not None:
        config["ladder"] = ladder
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _checker(issue_path: Path):
    def check(video, **kwargs):
        return {"candidates": [
            {"id": name, "kind": "synthetic_fault", "start_sec": 0.5 + index, "end_sec": 0.5 + index,
             "confirmed": True, "numbers": {}, "evidence": {}}
            for index, name in enumerate(json.loads(issue_path.read_text(encoding="utf-8")))
        ]}
    return check


def _fixer(tmp_path: Path, body: str) -> list:
    script = tmp_path / "fake_fixer.py"
    script.write_text("import sys, json\nfrom pathlib import Path\n" + body, encoding="utf-8")
    return [sys.executable, str(script)]


# 1. The fixer is told what is deliberate, and changes to it are caught.

def test_fix_brief_carries_intent_and_protected_sections(tmp_path):
    problem = {"id": "late-whoosh", "kind": "cue_late", "start_sec": 10.0, "end_sec": 10.0,
               "numbers": {}, "rule_verdict": "fault", "confirmed": True, "evidence": {}}
    brief = _fix_brief([problem], tmp_path, intent=INTENT)
    assert "late-whoosh" in brief
    assert INTENT.split(". ")[0] in brief
    assert "8.750–10.000 s" in brief and "12.500–15.000 s" in brief
    assert "false alarm" in brief and "Do not redesign or re-level" in brief


def test_protected_changes_flag_only_audible_moves():
    original = [{"start_sec": 8.75, "end_sec": 10.0, "sentence": "riser",
                 "band_db": {"high": -20.0, "mid": -30.0, "air": -80.0}}]
    small = [dict(original[0], band_db={"high": -21.5, "mid": -29.0, "air": -70.0})]
    riser_cut = [dict(original[0], band_db={"high": -29.0, "mid": -30.5, "air": -80.0})]
    assert _protected_changes(original, small, 3.0) == []
    [problem] = _protected_changes(original, riser_cut, 3.0)
    assert problem["kind"] == "intent_section_changed" and problem["confirmed"]
    assert set(problem["numbers"]["bands"]) == {"high"}
    assert problem["numbers"]["bands"]["high"]["change_db"] == -9.0


def test_protected_changes_ignore_quiet_codec_band_but_keep_content_changes():
    before = [{"start_sec": 20.571, "end_sec": 22.857, "sentence": "quench dropout",
               "band_db": {"sub": -73.55, "low_mid": -61.57, "mid": -29.45,
                           "high": -28.41, "air": -37.27}}]
    after = [{"start_sec": 20.571, "end_sec": 22.857, "sentence": "quench dropout",
              "band_db": {"sub": -73.23, "low_mid": -57.39, "mid": -29.41,
                          "high": -28.43, "air": -37.33}}]
    # Measured from newtest7 pristine vs round 1: only the codec floor moved.
    assert _protected_changes(before, after, 3.0) == []

    groove_returns = [dict(after[0], band_db={**after[0]["band_db"], "sub": -20.0,
                                               "low_mid": -24.0})]
    [problem] = _protected_changes(before, groove_returns, 3.0)
    assert set(problem["numbers"]["bands"]) == {"sub", "low_mid"}

    sweep_removed = [dict(before[0], band_db={**before[0]["band_db"], "high": -40.0})]
    [problem] = _protected_changes(before, sweep_removed, 3.0)
    assert set(problem["numbers"]["bands"]) == {"high"}


def test_run_loop_asks_fixer_to_restore_a_changed_section(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["late-whoosh"]), encoding="utf-8")
    riser = tmp_path / "riser_db.txt"
    riser.write_text("-20", encoding="utf-8")
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    # Round 1 fixes the whoosh but also turns the riser down; round 2 restores it.
    fixer = _fixer(tmp_path, (
        "brief = sys.stdin.read()\n"
        "n = len(list(Path(sys.argv[1]).iterdir()))\n"
        "(Path(sys.argv[1]) / f'{n}.txt').write_text(brief)\n"
        "Path(sys.argv[2]).write_text('[]')\n"
        "Path(sys.argv[3]).write_text('-20' if 'intent_section_changed' in brief else '-29')\n"
    ))
    config = _config(tmp_path, fixer + [str(briefs), str(issues), str(riser)])

    def levels(video, sections):
        return [{"start_sec": s["start_sec"], "end_sec": s["end_sec"], "sentence": s["sentence"],
                 "band_db": {"high": float(riser.read_text()) if s["start_sec"] < 11 else -40.0}}
                for s in sections]

    result = run_loop(str(config), check_function=_checker(issues), level_function=levels)
    assert result["status"] == "success"
    assert result["rounds"] == 2
    second = (briefs / "1.txt").read_text()
    assert "intent_section_changed" in second and "Undo that change" in second
    assert riser.read_text() == "-20"


# 2. A usage limit is "retry later", not a failed rung.

def test_usage_limit_detection():
    assert _usage_limit_message(1, "Claude AI usage limit reached|1790000000") is not None
    assert _usage_limit_message(1, "error: You've hit your usage limit. Try again at 3:05 PM.") is not None
    assert _usage_limit_message(0, "5-hour limit reached ∙ resets 3pm") is not None
    long_report = "Fixed the late whoosh. " * 40 + "The limiter's rate limit knob was untouched."
    assert _usage_limit_message(0, long_report) is None
    assert _usage_limit_message(1, "Traceback: build.sh failed") is None


def test_run_loop_stops_for_retry_later_without_climbing(tmp_path, monkeypatch):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["stuck"]), encoding="utf-8")
    fixer = _fixer(tmp_path, "sys.stdin.read()\nprint('Claude AI usage limit reached|1790000000')\nsys.exit(1)\n")
    ladder = [{"model": "m0", "effort": "e"}, {"model": "m1", "effort": "e"}]
    config = _config(tmp_path, fixer, ladder=ladder, intent=False)
    result = run_loop(str(config), check_function=_checker(issues))
    assert result["status"] == "retry later"
    assert result["rounds"] == 0 and result["rungs_used"] == []
    assert [item["id"] for item in result["remaining"]] == ["stuck"]
    [entry] = [json.loads(line) for line in (tmp_path / "loop-output" / "loop_log.jsonl").read_text().splitlines()]
    assert entry["fixer_status"] == "usage_limit" and entry["counted"] is False
    assert entry["next_rung_index"] == 0
    summary = (tmp_path / "loop-output" / "summary.md").read_text()
    assert "Retry later" in summary and "usage limit" in summary
    fixed_section = summary.split("## Fixed problems", 1)[1].split("## Fixer verdicts on review items", 1)[0]
    assert "No confirmed problems fixed." in fixed_section
    assert "`stuck`" not in fixed_section
    import src.loop as loop_module
    monkeypatch.setattr(loop_module, "run_loop", lambda path: dict(result))
    assert main(["run", "--config", str(config)]) == loop_module.EXIT_RETRY_LATER


def test_fixer_stdout_becomes_last_message(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps(["one"]), encoding="utf-8")
    fixer = _fixer(tmp_path, "sys.stdin.read()\nPath(sys.argv[1]).write_text('[]')\nprint('Moved the whoosh.')\n")
    config = _config(tmp_path, fixer + [str(issues)], intent=False)
    assert run_loop(str(config), check_function=_checker(issues))["status"] == "success"
    assert "Moved the whoosh." in (tmp_path / "loop-output" / "last-message.txt").read_text()


# 3a. Small single-band boosts are not faults; big or broad ones are.

def _windows(values_by_band):
    count = len(next(iter(values_by_band.values())))
    bands = ("sub", "low_mid", "mid", "high", "air")
    return [{"start_sec": i * 0.625, "end_sec": (i + 1) * 0.625, "silent": False,
             "band_db": {band: values_by_band.get(band, [-40.0] * count)[i] for band in bands}}
            for i in range(count)]


def test_loudness_step_needs_minimum_size_or_breadth():
    flat = [-40.0] * 16

    def stepped(size):
        values = flat.copy()
        values[5:10] = [-40.0 + size] * 5
        return values

    small = [item for item in _outlier_sections(_windows({"low_mid": stepped(6.8)})) if item["shape"] == "step" and item["direction"] == "boost"]
    assert small and not any(item["fault_level"] for item in small)
    large = [item for item in _outlier_sections(_windows({"mid": stepped(10.0)})) if item["shape"] == "step" and item["direction"] == "boost"]
    assert large and all(item["fault_level"] for item in large)
    broad = _outlier_sections(_windows({band: stepped(6.8) for band in ("sub", "low_mid", "mid", "high")}))
    assert [item["fault_level"] for item in broad if item["shape"] == "step" and item["direction"] == "boost"] == [True] * 4


def test_range_problems_need_most_of_their_span_inside_intent():
    from src.media.audio import _mostly_inside_ignore_range
    ignored = {"start_sec": 3.125, "end_sec": 3.75, "reason": "deliberate section"}
    assert not _mostly_inside_ignore_range(3.0, 3.2, {"start_sec": 3.15, "end_sec": 4.0})
    assert _mostly_inside_ignore_range(3.0, 3.2, {"start_sec": 2.95, "end_sec": 3.15})

    def stepped(size):
        values = [-40.0] * 16
        values[5:10] = [-40.0 + size] * 5
        return values

    outliers = _outlier_sections(_windows({"mid": stepped(10.0)}), ignore_ranges=[ignored])
    [step] = [item for item in outliers if item["shape"] == "step" and item["direction"] == "boost"]
    assert not step["ignored"]
    covered = _outlier_sections(
        _windows({"mid": stepped(10.0)}),
        ignore_ranges=[{"start_sec": 3.0, "end_sec": 6.5, "reason": "deliberate section"}],
    )
    [covered_step] = [item for item in covered if item["shape"] == "step" and item["direction"] == "boost"]
    assert covered_step["ignored"]


# 3b. A cue at t=0 with its sound present is not reported missing.

def _kick(length):
    t = np.arange(length) / RATE
    return np.sin(2 * np.pi * 90 * t) * np.exp(-t * 18.0)


def _click(length):
    t = np.arange(length) / RATE
    return 0.5 * np.sin(2 * np.pi * 5000 * t) * np.exp(-t * 300.0)


def _start_mix(tmp_path, *, first_kick: bool) -> tuple:
    beat = 60.0 / 110
    duration = 16 * beat + 0.5
    samples = np.zeros(int(duration * RATE))
    t = np.arange(len(samples)) / RATE
    samples += 0.05 * np.sin(2 * np.pi * 880 * t)  # pad bed from the first sample
    kick, click = _kick(int(0.4 * RATE)), _click(int(0.05 * RATE))
    cues = [{"time_sec": 0.0, "label": "pad bed"}]
    for n in range(16):
        at = n * beat
        cues.append({"time_sec": round(at, 6), "label": "kick"})
        if n or first_kick:
            first = int(round(at * RATE))
            samples[first:first + len(kick)] += 0.6 * kick[:len(samples) - first]
        if n % 4 == 0:
            cues.append({"time_sec": round(at, 6), "label": "text pop"})
            first = int(round(at * RATE))
            samples[first:first + len(click)] += click[:len(samples) - first]
    path = tmp_path / ("start-{}.wav".format("kick" if first_kick else "nokick"))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(samples, -0.95, 0.95) * 32767).astype("<i2").tobytes())
    return path, cues


def _first_kick(result):
    return next(cue for cue in result["cues"] if cue["time_sec"] == 0.0 and cue["label"] == "kick")


def test_cue_at_start_with_usual_band_level_is_review_not_missing():
    # Numbers from a held-out mix: the downbeat kick at 0 s sat at its
    # group's usual level, but with no pre-roll no attack could be measured.
    kick = {"cue_level_db": 27.11, "typical_level_db": 27.11}
    assert _present_at_file_start(0.0, kick)
    assert not _present_at_file_start(0.5, kick)
    assert not _present_at_file_start(0.0, {"cue_level_db": 15.0, "typical_level_db": 27.11})
    assert not _present_at_file_start(0.0, {"cue_level_db": -75.0, "typical_level_db": None})
    assert _present_at_file_start(0.0, {"cue_level_db": -24.4, "typical_level_db": None})


def test_cue_at_start_is_still_missing_when_its_band_is_empty(tmp_path):
    path, cues = _start_mix(tmp_path, first_kick=False)
    result = sync_check(str(path), cues=cues, bpm=110, cache=MediaCache(tmp_path / "cache"), cut_detector=lambda: [])
    assert _first_kick(result)["status"] == "missing"


def test_protected_guard_ignores_moments_the_fixer_was_asked_to_fix():
    # A misplaced kick at 14.875 s sat inside the bridge; moving it to 15.0 s
    # is the fix, even though the bridge's sub level drops.
    def section(kick_db):
        windows = [{"start_sec": 12.5 + i * 0.25, "end_sec": 12.75 + i * 0.25,
                    "band_db": {"sub": kick_db if i == 9 else -30.0}} for i in range(10)]
        return [{"start_sec": 12.5, "end_sec": 15.0, "sentence": "bridge", "windows": windows}]

    original, fixed = section(-5.0), section(-30.0)
    assert _protected_changes(original, fixed, 3.0)  # without context it looks like damage
    kick = {"kind": "cue_early", "start_sec": 15.0, "end_sec": 15.0, "numbers": {"offset_ms": -125.0}}
    assert _protected_changes(original, fixed, 3.0, _fix_times([kick])) == []


def test_displaced_match_inside_intent_section_is_review_not_fault():
    sync = {"tolerance_ms": 80.0, "cues": [
        {"time_sec": 10.0, "label": "cut whoosh", "status": "early", "offset_ms": -280.0,
         "matched_onset_time_sec": 9.72, "audible_in_any_band": True},
        {"time_sec": 7.5, "label": "cut whoosh", "status": "late", "offset_ms": 250.0,
         "matched_onset_time_sec": 7.75, "audible_in_any_band": True},
        # A kick misplaced into the muffled bridge is still a fault: a bridge
        # does not swell into the next cue.
        {"time_sec": 15.0, "label": "downbeat kick 13", "status": "early", "offset_ms": -125.0,
         "matched_onset_time_sec": 14.875, "audible_in_any_band": True},
    ]}
    rows = {row["start_sec"]: row for row in _build_candidates(sync, {}, {}, [], intent=INTENT)}
    assert rows[10.0]["rule_verdict"] == "ambiguous" and "inside a section" in rows[10.0]["ambiguous_reason"]
    assert rows[10.0]["intent_note"].startswith("The noise riser")
    assert rows[7.5]["rule_verdict"] == "fault"
    assert rows[15.0]["rule_verdict"] == "fault" and rows[15.0]["kind"] == "cue_early"


def test_one_off_cue_on_a_section_edge_is_review_not_fault():
    def missing(time_sec, label):
        return {"time_sec": time_sec, "label": label, "status": "missing", "evidence": "no_band_beyond_companions",
                "audible_in_any_band": True}

    # The sub drop lands as the muffled bridge ends; the whoosh sits mid-scene.
    sync = {"tolerance_ms": 80.0, "cues": [missing(15.0, "sub drop"), missing(11.0, "whoosh")]}
    rows = {row["start_sec"]: row for row in _build_candidates(sync, {}, {}, [], intent=INTENT)}
    assert rows[15.0]["rule_verdict"] == "ambiguous" and "begins or ends" in rows[15.0]["ambiguous_reason"]
    assert rows[15.0]["intent_note"].startswith("A muffled")
    assert rows[11.0]["rule_verdict"] == "fault" and rows[11.0]["kind"] == "cue_missing"


def test_intent_ranges_named_by_end_point_or_after_beats():
    from src.loop import _parse_intent_sections
    notes = ("The sound rises as a two-beat noise riser from 8.750 s to the reveal at 10.000 s. "
             "A filtered noise riser grows across the two beats from 12.000 s to 13.091 s.")
    assert [(x["start_sec"], x["end_sec"]) for x in _parse_intent_sections(notes)] == [(8.75, 10.0), (12.0, 13.091)]
    assert _parse_intent_sections("Bars 9 to 12 build. Kicks 4-8 are loud.") == []


def test_missing_cue_with_band_at_usual_level_is_review():
    from src.media.audio import _band_at_usual_level
    # A held-out downbeat kick under a text pop: at its usual level, yet
    # neither matcher credited it.
    assert _band_at_usual_level({"cue_level_db": 27.22, "typical_level_db": 27.1})
    assert not _band_at_usual_level({"cue_level_db": 20.0, "typical_level_db": 27.1})
    assert not _band_at_usual_level({"cue_level_db": 27.0, "typical_level_db": None})


def test_small_repeated_group_does_not_accept_a_weak_instance_from_a_transient():
    from src.media.audio import _small_group_match_is_unsupported, _small_group_peer_level_drop_db
    cues = [{"group": "clink", "time_sec": 12.321}, {"group": "clink", "time_sec": 14.464}]
    checks = {
        0: {"dominant_band": "140-177 Hz", "cue_level_db": 18.86},
        1: {"dominant_band": "140-177 Hz", "cue_level_db": 8.18},
    }
    assert _small_group_peer_level_drop_db(1, cues, checks) == 10.68
    assert _small_group_match_is_unsupported(1, cues, checks)
    assert not _small_group_match_is_unsupported(0, cues, checks)
    three_cues = cues + [{"group": "clink", "time_sec": 16.0}]
    three_checks = dict(checks)
    three_checks[2] = {"dominant_band": "140-177 Hz", "cue_level_db": 18.0}
    assert _small_group_match_is_unsupported(1, three_cues, three_checks)


def test_repeated_group_missing_instance_is_a_robust_peer_outlier():
    from src.media.audio import _small_group_match_is_unsupported, _small_group_peer_level_drop_db

    cues = [{"group": "whoosh", "time_sec": time} for time in (4.5, 11.4, 18.3, 22.8, 27.4)]
    levels = (-9.57, -11.83, -11.08, -10.56, -14.65)
    checks = {index: {"dominant_band": "high", "cue_level_db": level}
              for index, level in enumerate(levels)}
    assert round(_small_group_peer_level_drop_db(4, cues, checks), 2) == 3.83
    assert _small_group_match_is_unsupported(4, cues, checks)

    # A modest quiet instance stays within the peer range instead of becoming a fault.
    checks[4] = {"dominant_band": "high", "cue_level_db": -13.0}
    assert not _small_group_match_is_unsupported(4, cues, checks)


def test_synthetic_soft_attack_measurement_separates_build_from_missing_neutral_cue():
    from src.media.audio import SAMPLE_RATE, _measure_soft_attack_build, _no_band_cue_outcome

    rate, cue_time, duration, beat_sec = SAMPLE_RATE, 2.0, 6.0, 0.5
    rng = np.random.default_rng(38)
    flat = rng.normal(0.0, 0.001, int(rate * duration)).astype(np.float32)
    ramped = flat.copy()
    swell_noise = rng.normal(0.0, 1.0, len(flat)).astype(np.float32)
    times = np.arange(len(flat), dtype=np.float32) / rate
    rise = np.clip((times - cue_time) / (2.0 * beat_sec), 0.0, 1.0)
    fade = np.clip((cue_time + 2.0 * beat_sec + 0.1 - times) / 0.1, 0.0, 1.0)
    ramped += (0.08 * rise * fade * swell_noise).astype(np.float32)

    def candidate(samples):
        build = _measure_soft_attack_build(samples, cue_time, None, beat_sec, [1.0, 4.0, 4.5])
        status, reason, evidence = _no_band_cue_outcome(build)
        cue = {"time_sec": cue_time, "label": "fx 1", "status": status, "evidence": evidence,
               "ambiguous_reason": reason, "audible_in_any_band": True, "soft_attack_build": build}
        return build, _build_candidates({"tolerance_ms": 80.0, "cues": [cue]}, {}, {}, [])[0]

    swell_measurement, swell_row = candidate(ramped)
    assert swell_measurement["measured"]
    assert swell_measurement["window_sec"] == 1.0
    assert swell_measurement["relative_rise_db"] >= 6.0
    assert swell_measurement["end_excess_db"] >= 3.0
    assert swell_row["kind"] == "cue_ambiguous" and swell_row["rule_verdict"] == "ambiguous"

    absent_measurement, absent_row = candidate(flat)
    assert absent_measurement["conclusive"] and not absent_measurement["measured"]
    assert absent_row["kind"] == "cue_missing" and absent_row["rule_verdict"] == "fault"


def test_soft_attack_measurement_stops_at_the_next_group_cue_or_file_end():
    from src.media.audio import SAMPLE_RATE, _measure_soft_attack_build

    samples = np.zeros(8 * SAMPLE_RATE, dtype=np.float32)
    check = _measure_soft_attack_build(
        samples, 2.0, None, 0.5, [4.5, 5.5], next_cue_sec=3.25, _span_sec=2.0,
    )
    assert check["window_sec"] == 1.25
    end_limited = _measure_soft_attack_build(
        samples, 7.5, None, 0.5, [], next_cue_sec=9.0, _span_sec=4.0,
    )
    assert end_limited["window_sec"] == 0.5


def test_soft_attack_measurement_finds_a_build_over_one_bar():
    from src.media.audio import SAMPLE_RATE, _measure_soft_attack_build

    rate, cue_time, duration, beat_sec = SAMPLE_RATE, 2.0, 8.0, 0.5
    rng = np.random.default_rng(63)
    samples = rng.normal(0.0, 0.003, int(rate * duration)).astype(np.float32)
    swell_noise = rng.normal(0.0, 1.0, len(samples)).astype(np.float32)
    times = np.arange(len(samples), dtype=np.float32) / rate
    rise = np.clip((times - cue_time) / (4.0 * beat_sec), 0.0, 1.0)
    fade = np.clip((cue_time + 4.0 * beat_sec + 0.1 - times) / 0.1, 0.0, 1.0)
    samples += (0.01 * rise * fade * swell_noise).astype(np.float32)

    check = _measure_soft_attack_build(
        samples, cue_time, None, beat_sec, [1.0, 4.0, 5.0, 6.0], next_cue_sec=duration,
    )
    assert check["window_sec"] == 2.0
    assert check["conclusive"] and check["measured"]


def test_swell_words_are_only_a_tiebreaker_when_audio_build_is_inconclusive():
    unmeasured = {"time_sec": 15.0, "label": "riser", "status": "missing",
                  "evidence": "no_band_beyond_companions", "audible_in_any_band": True,
                  "soft_attack_build": {"conclusive": False, "measured": False}}
    rows = _build_candidates({"tolerance_ms": 80.0, "cues": [unmeasured]}, {}, {}, [])
    assert rows[0]["kind"] == "cue_ambiguous" and rows[0]["rule_verdict"] == "ambiguous"

    note_only = dict(unmeasured, time_sec=5.0, label="noise")
    rows = _build_candidates({"tolerance_ms": 80.0, "cues": [note_only]}, {}, {}, [],
                             intent="The noise builds gradually before the reveal.")
    assert rows[0]["kind"] == "cue_ambiguous" and rows[0]["rule_verdict"] == "ambiguous"

    measured_absent = dict(unmeasured, soft_attack_build={"conclusive": True, "measured": False})
    rows = _build_candidates({"tolerance_ms": 80.0, "cues": [measured_absent]}, {}, {}, [])
    assert rows[0]["kind"] == "cue_missing" and rows[0]["rule_verdict"] == "fault"


def _shared_reports(pop_moves: bool, whoosh_moves: bool, whoosh_times=(3.273, 6.545, 9.818)):
    reports = []
    for time in (0.0, 3.273, 6.545, 9.818, 13.091):
        moved = pop_moves and time == 3.273
        reports.append({"time_sec": time, "label": "text pop", "cue_group": "text pop",
                        "status": "late" if moved or (whoosh_moves and time == 3.273) else "ok",
                        "matched_onset_time_sec": time + 0.25 if moved or (whoosh_moves and time == 3.273) else time})
    for time in whoosh_times:
        moved = whoosh_moves and time == 3.273
        reports.append({"time_sec": time, "label": "cut whoosh", "cue_group": "cut whoosh",
                        "status": "late" if moved else "ok",
                        "matched_onset_time_sec": time + 0.25 if moved else time})
    return reports


def _status(reports, group, time=3.273):
    return next(r["status"] for r in reports if r["cue_group"] == group and r["time_sec"] == time)


def test_shared_move_is_blamed_on_the_group_that_rides_along():
    from src.media.audio import _attribute_shared_displacement
    # Every whoosh comes with a pop; only the whoosh moved, yet both match
    # at the same late spot.
    reports = _shared_reports(pop_moves=False, whoosh_moves=True)
    _attribute_shared_displacement(reports)
    assert _status(reports, "cut whoosh") == "late"
    pop = next(r for r in reports if r["cue_group"] == "text pop" and r["time_sec"] == 3.273)
    assert pop["status"] == "ambiguous" and pop["evidence"] == "shared_displacement"
    assert "cut whoosh" in pop["ambiguous_reason"]


def test_shared_move_rule_leaves_other_cases_alone():
    from src.media.audio import _attribute_shared_displacement
    # The pop moved alone: nothing to share, it stays a fault.
    reports = _shared_reports(pop_moves=True, whoosh_moves=False)
    _attribute_shared_displacement(reports)
    assert _status(reports, "text pop") == "late"
    # Groups that do not usually coincide: both stay faults.
    reports = _shared_reports(pop_moves=False, whoosh_moves=True, whoosh_times=(3.273, 4.5, 8.0, 11.0))
    _attribute_shared_displacement(reports)
    assert _status(reports, "text pop") == "late" and _status(reports, "cut whoosh") == "late"
    # Always together with equal counts: cannot tell which moved, keep both.
    reports = _shared_reports(pop_moves=False, whoosh_moves=True, whoosh_times=(0.0, 3.273, 6.545, 9.818, 13.091))
    _attribute_shared_displacement(reports)
    assert _status(reports, "text pop") == "late" and _status(reports, "cut whoosh") == "late"


def _stack_simple(cue_level_db, attacks):
    # A chime stacked with a whoosh and a tom: its band is quiet at the cue,
    # and `attacks` are the times where the chime band really fires.
    def level_test(at, _index):
        return {"passed": any(abs(at - time) < 0.001 for time in attacks), "level_db": 15.6}
    return {"cue_level_db": cue_level_db, "typical_level_db": 15.6, "cue_index": 1,
            "onsets": [{"time_sec": time} for time in attacks], "level_test": level_test}


def test_stacked_cue_quiet_in_its_own_band_is_reported_at_its_real_attack():
    from src.media.audio import _own_band_displaced
    cues = [{"time_sec": 8.571, "group": "chime"}, {"time_sec": 11.429, "group": "chime"},
            {"time_sec": 11.429, "group": "slide swish"}, {"time_sec": 17.143, "group": "chime"}]
    slot = {"slot_before_ms": 285.7, "slot_after_ms": 285.7, "typical_attack_lag_ms": 0}
    assert _own_band_displaced(_stack_simple(1.9, [11.6]), 1, cues, slot, 0.5, 0.08) == 11.6
    # Band at its usual level: the template's on-time match stands.
    assert _own_band_displaced(_stack_simple(14.0, [11.6]), 1, cues, slot, 0.5, 0.08) is None
    # Nothing clear elsewhere in the slot: no timing claim.
    assert _own_band_displaced(_stack_simple(1.9, []), 1, cues, slot, 0.5, 0.08) is None
    # A group that always sounds ~120 ms after its cue is on time at +90 ms.
    lagging = dict(slot, typical_attack_lag_ms=120)
    assert _own_band_displaced(_stack_simple(1.9, [11.519]), 1, cues, lagging, 0.5, 0.08) is None


# 3. Checks for faults the cue matcher cannot see.

def _write_stereo(path: Path, pan_after: float, louder_after: float = 1.0, seconds: float = 16.0) -> None:
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * RATE)) / RATE
    tone = 0.2 * np.sin(2 * np.pi * 500.0 * t) + 0.02 * rng.standard_normal(len(t))
    later = t >= seconds / 2
    gain = np.where(later, louder_after, 1.0)
    left = tone * gain * np.where(later, 1.0 - pan_after, 1.0)
    right = tone * gain * np.where(later, 1.0 + pan_after, 1.0)
    frames = (np.clip(np.stack([left, right], axis=1), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(frames.tobytes())


def test_stereo_balance_shift_flags_a_repanned_stretch(tmp_path):
    from src.media.audio import stereo_balance_shifts
    path = tmp_path / "pan.wav"
    _write_stereo(path, pan_after=0.8)
    shifts = stereo_balance_shifts(str(path), bpm=120)["shifts"]
    assert shifts and shifts[0]["side"] == "right" and shifts[0]["start_sec"] == 8.0
    assert shifts[0]["delta_db"] > 6.0 and not shifts[0]["new_content"]
    # The author's notes cover the stretch: nothing to report.
    assert not stereo_balance_shifts(str(path), bpm=120, ignore_ranges=[{"start_sec": 8.0, "end_sec": 16.0}])["shifts"]
    # Centred throughout: nothing to report.
    _write_stereo(path, pan_after=0.0)
    assert not stereo_balance_shifts(str(path), bpm=120)["shifts"]


def test_stereo_balance_shift_with_much_louder_band_is_new_content(tmp_path):
    from src.media.audio import stereo_balance_shifts
    path = tmp_path / "enter.wav"
    _write_stereo(path, pan_after=0.8, louder_after=3.0)
    shifts = stereo_balance_shifts(str(path), bpm=120)["shifts"]
    assert shifts and all(item["new_content"] for item in shifts)
    rows = _build_candidates({}, {}, {}, [], balance={"shifts": shifts})
    assert rows and rows[0]["rule_verdict"] == "informational"


def test_text_cue_timing_finds_a_late_caption(monkeypatch):
    from src.media import visual_timing
    shares = np.full(300, 0.004)            # 10 s of gentle background motion at 30 fps
    shares[60:65] = 0.3                      # on time at 2.0 s
    shares[171:174] = 0.03                   # 700 ms late for the 5.0 s cue
    monkeypatch.setattr(visual_timing, "change_shares", lambda source: {"frame_rate": 30.0, "shares": shares})
    cues = [{"time_sec": 2.0, "label": "chime | ON TIME"}, {"time_sec": 5.0, "label": "chime | LATE"},
            {"time_sec": 8.0, "label": "chime | FADES IN"}, {"time_sec": 6.0, "label": "low tom"}]
    rows = {item["label"]: item for item in visual_timing.text_cue_timing("x.mp4", cues, bpm=84)["text_cues"]}
    assert rows["chime | ON TIME"]["status"] == "ok"
    assert rows["chime | LATE"]["status"] == "late" and abs(rows["chime | LATE"]["offset_ms"] - 700.0) < 1.0
    assert rows["chime | FADES IN"]["status"] == "unseen"
    assert "low tom" not in rows
    candidates = _build_candidates({}, {}, {}, cues, visual={"text_cues": list(rows.values())})
    assert [(row["kind"], row["rule_verdict"]) for row in candidates] == [("text_late", "fault")]


def test_uncued_sound_is_a_fault_unless_the_notes_cover_it():
    item = {"time_sec": 20.0, "group": "chime", "score": 0.99, "group_median_score": 1.0,
            "simple_check_level_db": 15.9, "group_typical_level_db": 15.6}
    rows = _build_candidates({"uncued_sounds": [item]}, {}, {}, [], intent="")
    assert rows[0]["kind"] == "uncued_sound" and rows[0]["rule_verdict"] == "fault"
    rows = _build_candidates({"uncued_sounds": [dict(item, time_sec=9.25)]}, {}, {}, [], intent=INTENT)
    assert rows[0]["rule_verdict"] == "informational"


def _write_wav(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    frames = (np.clip(np.stack([left, right], axis=1), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(frames.tobytes())


def test_low_end_on_one_side_for_the_whole_piece_is_off_centre(tmp_path):
    from src.media.audio import stereo_balance_shifts
    t = np.arange(16 * RATE) / RATE
    bass = 0.3 * np.sin(2 * np.pi * 120.0 * t)
    mid = 0.1 * np.sin(2 * np.pi * 2000.0 * t)
    path = tmp_path / "bass.wav"
    _write_wav(path, bass + mid, 0.05 * bass + mid)
    result = stereo_balance_shifts(str(path), bpm=120)
    assert [(row["band"], row["side"]) for row in result["off_centre"]] == [("88-177 Hz", "left")]
    assert not result["shifts"]
    rows = _build_candidates({}, {}, {}, [], balance=result)
    assert [(row["kind"], row["rule_verdict"]) for row in rows] == [("stereo_off_centre", "fault")]
    # A mid-range part panned hard is an arrangement choice, not low end.
    _write_wav(path, bass + mid, bass + 0.05 * mid)
    assert not stereo_balance_shifts(str(path), bpm=120)["off_centre"]


def test_clipping_runs_find_a_clipped_burst_but_not_a_lone_overshoot(tmp_path):
    from src.media.audio import clipping_runs
    t = np.arange(10 * RATE) / RATE
    music = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    burst = (t >= 3.0) & (t < 3.2)
    music[burst] = np.clip(3.0 * np.sin(2 * np.pi * 60.0 * t[burst]), -0.98, 0.98)
    music[int(7.0 * RATE)] = 0.99
    path = tmp_path / "clip.wav"
    _write_wav(path, music, music)
    result = clipping_runs(str(path))
    assert [(row["start_sec"], row["end_sec"]) for row in result["runs"]] == [(3.0, 3.2)]
    assert not result["throughout"]
    rows = _build_candidates({}, {}, {}, [], clipping=result)
    assert [(row["kind"], row["rule_verdict"]) for row in rows] == [("clipping", "fault")]
    # Inside a section the notes describe, it is informational.
    noted = clipping_runs(str(path), ignore_ranges=[{"start_sec": 2.5, "end_sec": 3.5}])
    assert _build_candidates({}, {}, {}, [], clipping=noted)[0]["rule_verdict"] == "informational"
    partial = clipping_runs(str(path), ignore_ranges=[{"start_sec": 3.15, "end_sec": 4.0}])
    [partial_run] = partial["runs"]
    assert not partial_run["in_ignored_range"]
    assert _build_candidates({}, {}, {}, [], clipping=partial)[0]["rule_verdict"] == "fault"
    # A master that is hot throughout is limited, not clipped at one moment.
    loud = np.clip(3.0 * np.sin(2 * np.pi * 60.0 * t), -0.98, 0.98)
    _write_wav(path, loud, loud)
    hot = clipping_runs(str(path))
    assert hot["throughout"]
    assert {row["rule_verdict"] for row in _build_candidates({}, {}, {}, [], clipping=hot)} == {"informational"}


def _caption(width: int = 1280, height: int = 720, x0: int = 96, x1: int = 1000) -> np.ndarray:
    image = np.full((height, width), 40.0, dtype=np.float32)
    for x in range(x0, min(x1, width), 12):           # letter strokes, 6 px wide
        image[200:280, x:x + 6] = 235.0
    return image


def _paint_text_band(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
    for x in range(x0, x1, 12):
        image[y0:y1, x:min(x + 6, x1)] = 235.0


def _near_edge_gap_title() -> np.ndarray:
    image = np.full((720, 1280), 40.0, dtype=np.float32)
    _paint_text_band(image, 295, 375, 100, 1220)
    # The last glyph is an L: its stem stays well inside the margin, while its
    # foot ends at x=1274. Only the foot reaches the 24 px near-edge region.
    image[295:375, 1220:1228] = 235.0
    image[366:375, 1220:1275] = 235.0
    # A subtitle is centered around x=930; it must not be used as a centered
    # line confirmation for the left-anchored title.
    _paint_text_band(image, 400, 440, 800, 1060)
    return image


def _paint_right_aligned_line(image: np.ndarray, y0: int, y1: int, x0: int, end_x: int) -> None:
    _paint_text_band(image, y0, y1, x0, end_x - 40)
    image[y0:y1, end_x - 40:end_x - 32] = 235.0
    image[y1 - 8:y1, end_x - 40:end_x] = 235.0


def test_edge_touches_counts_text_running_into_an_edge():
    from src.media.visual_timing import edge_touches
    assert edge_touches(_caption())["right"] == 0
    assert edge_touches(_caption(x1=1400))["right"] >= 60
    assert edge_touches(_caption(x1=1400))["left"] == 0


def test_text_overflow_flags_a_title_cut_off_by_the_frame(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    frames = {"before": np.full((720, 1280), 40.0, dtype=np.float32)}
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: (
        frames["before"] if at < 2.0 else _caption(x1=1400) if at < 4.0 else _caption()))
    timed = [{"time_sec": 2.0, "label": "tick | BIOLUMINESCENT", "status": "ok", "appears_sec": 2.0},
             {"time_sec": 4.0, "label": "tick | FITS", "status": "ok", "appears_sec": 4.0}]
    overflows = visual_timing.text_overflow(str(video), timed)["overflows"]
    assert [(row["label"], row["side"]) for row in overflows] == [("tick | BIOLUMINESCENT", "right")]
    rows = _build_candidates({}, {}, {}, [], visual={"overflows": overflows})
    assert [(row["kind"], row["rule_verdict"]) for row in rows] == [("text_cropped", "fault")]


def test_text_overflow_reviews_a_large_line_ending_six_pixels_before_the_edge(monkeypatch, tmp_path):
    from src.media import visual_timing
    from src.loop import _judge_question_text
    video = tmp_path / "near-edge.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 40.0, dtype=np.float32)
    after = _near_edge_gap_title()
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < 2.0 else after)
    timed = [{"time_sec": 2.0, "label": "tick | LONG TITLE", "status": "ok", "appears_sec": 2.0}]

    [overflow] = visual_timing.text_overflow(str(video), timed)["overflows"]
    assert overflow["side"] == "right"
    assert overflow["margin_px"] == 6
    assert overflow["touches_edge"] is False
    assert overflow["line_bands"][0]["height_px"] >= 75
    assert overflow["line_bands"][0]["near_edge_positions"] < 24
    [candidate] = _build_candidates({}, {}, {}, [], visual={"overflows": [overflow]})
    assert candidate["kind"] == "text_cropped"
    assert candidate["rule_verdict"] == "ambiguous"
    assert "6 px from the right edge" in candidate["ambiguous_reason"]
    question = _judge_question_text(candidate, "")
    assert "80 px tall" in question and "cut-off stroke" in question
    assert "measurements cannot distinguish" in question
    assert "Inspect the visible letters and words" not in question


def test_text_overflow_ignores_a_centred_title_with_normal_margins(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "centred.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 40.0, dtype=np.float32)
    after = _caption(x0=200, x1=1080)
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < 2.0 else after)
    timed = [{"time_sec": 2.0, "label": "tick | CENTRED TITLE", "status": "ok", "appears_sec": 2.0}]
    assert visual_timing.text_overflow(str(video), timed)["overflows"] == []


@pytest.mark.parametrize("appears", [0.033, 26.667])
def test_text_overflow_ignores_separate_round_decorations_near_edge(monkeypatch, tmp_path, appears):
    from src.media import visual_timing
    video = tmp_path / "lanterns.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 40.0, dtype=np.float32)
    after = before.copy()
    yy, xx = np.ogrid[:720, :1280]
    for centre in range(40, 1241, 120):
        after[(xx - centre) ** 2 + (yy - 135) ** 2 <= 30 ** 2] = 220.0
    _paint_text_band(after, 260, 350, 250, 1030)
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < appears else after)
    timed = [{"time_sec": appears, "label": "gong | CENTRED TITLE", "status": "ok", "appears_sec": appears}]
    assert visual_timing.text_overflow(str(video), timed)["overflows"] == []


def test_text_overflow_ignores_a_near_edge_line_already_in_the_previous_frame(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "existing-near-edge.mp4"
    video.write_bytes(b"")
    existing = _near_edge_gap_title()
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: existing)
    timed = [{"time_sec": 2.0, "label": "tick | LONG TITLE", "status": "ok", "appears_sec": 2.0}]
    assert visual_timing.text_overflow(str(video), timed)["overflows"] == []


def test_right_aligned_two_line_near_edge_block_is_review_only(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "lopsided.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 40.0, dtype=np.float32)
    after = np.full((720, 1280), 40.0, dtype=np.float32)
    _paint_right_aligned_line(after, 240, 320, 240, 1269)
    _paint_right_aligned_line(after, 340, 390, 620, 1269)
    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < 2.0 else after)
    timed = [{"time_sec": 2.0, "label": "tick | CUT TITLE", "status": "ok", "appears_sec": 2.0}]

    overflows = visual_timing.text_overflow(str(video), timed)["overflows"]
    assert len(overflows) == 1
    overflow = overflows[0]
    assert overflow["margin_px"] == 12
    rows = _build_candidates({}, {}, {}, [], visual={"overflows": [overflow]})
    assert [(row["kind"], row["rule_verdict"]) for row in rows] == [("text_cropped", "ambiguous")]


def test_text_overflow_ignores_a_static_prop_that_appears_with_the_title(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "press.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 235.0, dtype=np.float32)
    after = before.copy()
    # A solid press rod touches the top edge on the same cut as the title.
    after[:50, 600:680] = 70.0
    after[50:150, 220:1060] = 30.0
    for x in range(80, 900, 12):
        after[575:645, x:x + 6] = 20.0

    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < 2.0 else after)
    timed = [{"time_sec": 2.0, "label": "stamp | PULL THE PRESS", "status": "ok", "appears_sec": 2.0}]
    assert visual_timing.text_overflow(str(video), timed)["overflows"] == []


def test_text_overflow_still_finds_text_running_into_the_top_edge(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "top.mp4"
    video.write_bytes(b"")
    before = np.full((720, 1280), 40.0, dtype=np.float32)
    after = before.copy()
    for x in range(80, 900, 12):
        after[:80, x:x + 6] = 235.0

    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", lambda path, at, size: before if at < 2.0 else after)
    timed = [{"time_sec": 2.0, "label": "tick | CUT OFF", "status": "ok", "appears_sec": 2.0}]
    overflows = visual_timing.text_overflow(str(video), timed)["overflows"]
    assert [(row["label"], row["side"]) for row in overflows] == [("tick | CUT OFF", "top")]


def test_uncued_attack_band_skips_bands_claimed_by_other_cues():
    from src.media.audio import _SIMPLE_BANDS, _uncued_attack_band

    class Levels:
        def __init__(self, rises):
            self.rises = rises

        def cue(self, at):
            return self.before(at) + self.rises.get(round(at, 2), np.zeros(len(_SIMPLE_BANDS)))

        def before(self, at):
            return np.full(len(_SIMPLE_BANDS), -40.0)

    ping = np.zeros(len(_SIMPLE_BANDS))
    ping[[4, 13]] = 20.0                   # a thump band (4) and the ping's own band (13)
    thump_only = np.zeros(len(_SIMPLE_BANDS))
    thump_only[4] = 25.0
    thump_band = {"low_hz": _SIMPLE_BANDS[4][0], "high_hz": _SIMPLE_BANDS[4][1]}
    levels = Levels({1.0: ping, 2.0: ping, 3.0: ping, 7.2: ping, 8.4: thump_only})
    found = _uncued_attack_band(levels, 7.2, [1.0, 2.0, 3.0], [thump_band])
    assert found and found["band"] == "{:g}-{:g} Hz".format(*_SIMPLE_BANDS[13])
    # Only the thump attacks here: nothing of the ping's own.
    assert _uncued_attack_band(levels, 8.4, [1.0, 2.0, 3.0], [thump_band]) is None


def test_uncued_attack_band_ignores_a_band_the_group_only_shares():
    from src.media.audio import _SIMPLE_BANDS, _uncued_attack_band

    class Levels:
        def __init__(self, rises):
            self.rises = rises

        def cue(self, at):
            return self.before(at) + self.rises.get(round(at, 2), np.zeros(len(_SIMPLE_BANDS)))

        def before(self, at):
            return np.full(len(_SIMPLE_BANDS), -40.0)

    kick = np.zeros(len(_SIMPLE_BANDS))
    kick[2], kick[24] = 50.0, 36.0          # the kick itself, and a click that always lands with it
    click_only = np.zeros(len(_SIMPLE_BANDS))
    click_only[24] = 40.0
    levels = Levels({1.0: kick, 2.0: kick, 3.0: kick, 9.0: click_only})
    assert _uncued_attack_band(levels, 9.0, [1.0, 2.0, 3.0], []) is None


def test_text_overflow_ignores_scenery_scrolling_past_the_edge(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    blank = np.full((720, 1280), 40.0, dtype=np.float32)

    def scrolling(path, at, size):
        if at < 2.0:
            return blank
        # A stall front crossing the edge, somewhere else in every frame checked.
        image = blank.copy()
        top = 100 + int(round((at - 2.0) * 500))
        for x in range(1100, 1280, 12):
            image[top:top + 80, x:x + 6] = 235.0
        return image

    monkeypatch.setattr(visual_timing, "_frame_size", lambda path: (1280, 720))
    monkeypatch.setattr(visual_timing, "_grey_frame", scrolling)
    timed = [{"time_sec": 2.0, "label": "tick | GRILLS", "status": "ok", "appears_sec": 2.0}]
    assert visual_timing.text_overflow(str(video), timed)["overflows"] == []


def test_blank_flashes_finds_stray_black_frames_but_not_fades(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    picture, black = np.full(14400, 200.0, dtype=np.float32), np.zeros(14400, dtype=np.float32)
    fade_in = [np.full(14400, level, dtype=np.float32) for level in (80.0, 120.0, 160.0, 200.0)]
    frames = ([picture] * 30 + [black] * 2 + [picture] * 30        # two stray black frames at 1.0 s
              + [black] + fade_in + [picture] * 30                  # a cut into a scene that fades in
              + [black] * 30 + [picture] * 10)                      # a dark scene
    monkeypatch.setattr(visual_timing, "_frame_rate", lambda path: 30.0)
    monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate: iter(frames))
    flashes = visual_timing.blank_flashes(str(video))["flashes"]
    assert [(row["start_sec"], row["frames"]) for row in flashes] == [(1.0, 2)]
    rows = _build_candidates({}, {}, {}, [], visual={"flashes": flashes})
    assert [(row["kind"], row["rule_verdict"]) for row in rows] == [("blank_flash", "fault")]


def test_frame_splices_find_short_insertions_but_not_motion_cuts_or_black_runs(monkeypatch, tmp_path):
    from src.media import visual_timing

    video = tmp_path / "splice.mp4"
    video.write_bytes(b"")
    picture = np.full((90, 160), 90.0, dtype=np.float32)
    picture[15:72, 25:110] = 180.0
    splice = np.full((90, 160), 20.0, dtype=np.float32)
    splice[10:80, 48:130] = 240.0
    second_shot = np.full((90, 160), 220.0, dtype=np.float32)
    monkeypatch.setattr(visual_timing, "_frame_rate", lambda path: 30.0)

    monkeypatch.setattr(visual_timing, "_grid_frames",
                        lambda path, rate: iter([picture] * 5 + [splice] + [picture] * 6))
    [single] = visual_timing.frame_splices(str(video))["splices"]
    assert (single["start_sec"], single["frames"], single["context_delta"]) == (0.167, 1, 0.0)

    monkeypatch.setattr(visual_timing, "_grid_frames",
                        lambda path, rate: iter([picture] * 5 + [splice] * 2 + [picture] * 6))
    [double] = visual_timing.frame_splices(str(video))["splices"]
    assert (double["start_sec"], double["frames"]) == (0.167, 2)

    moving_with_insert = ([np.roll(picture, shift, axis=1) for shift in range(15)]
                          + [splice] * 2
                          + [np.roll(picture, shift, axis=1) for shift in range(17, 34)])
    monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate: iter(moving_with_insert))
    [moving_splice] = visual_timing.frame_splices(str(video))["splices"]
    assert (moving_splice["start_sec"], moving_splice["frames"]) == (0.5, 2)
    assert moving_splice["context_delta"] > 2.0

    moving = [np.roll(picture, shift, axis=1) for shift in range(12)]
    fast = [np.roll(picture, shift * 4, axis=1) for shift in range(24)]
    fade = [((1.0 - alpha) * picture + alpha * second_shot).astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, 15)]
    hard_cut = [picture] * 6 + [second_shot] * 6
    black_cut = [picture] * 6 + [np.zeros_like(picture)] * 8
    for frames in (moving, fast, fade, hard_cut, black_cut):
        monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate, frames=frames: iter(frames))
        assert visual_timing.frame_splices(str(video))["splices"] == []

    row = _build_candidates({}, {}, {}, [], visual={"frame_splices": [single]})[0]
    assert row["kind"] == "flash_frame" and row["rule_verdict"] == "fault"
    from src.loop import _judge_question_text
    assert "frame splice" in _judge_question_text(row, "")
    [intent_row] = _build_candidates({}, {}, {}, [], intent="Intentional glitch from 0.100 s to 0.200 s.",
                                     visual={"frame_splices": [single]})
    assert intent_row["rule_verdict"] == "informational"
    flash = {"start_sec": single["start_sec"], "end_sec": single["end_sec"], "frames": 1,
             "brightest_before": 168.0, "brightest_after": 168.0}
    rows = _build_candidates({}, {}, {}, [], visual={"flashes": [flash], "frame_splices": [single]})
    assert [row["kind"] for row in rows] == ["blank_flash"]


def test_frozen_pictures_need_motion_on_both_sides_and_ignore_still_cards(monkeypatch, tmp_path):
    from src.media import visual_timing
    video = tmp_path / "held.mp4"
    video.write_bytes(b"")
    still = np.zeros((90, 160), dtype=np.float32)
    still[20:60, 20:60] = 180.0
    moving_before = [np.roll(still, shift, axis=1) for shift in range(30)]
    held = [moving_before[-1]] * 30
    moving_after = [np.roll(moving_before[-1], shift, axis=1) for shift in range(1, 31)]
    monkeypatch.setattr(visual_timing, "_frame_rate", lambda path: 30.0)
    monkeypatch.setattr(visual_timing, "_grid_frames",
                        lambda path, rate: iter(moving_before + held + moving_after))
    found = visual_timing.frozen_pictures(str(video))["frozen_pictures"]
    assert len(found) == 1
    assert found[0]["duration_sec"] >= 0.5
    assert found[0]["motion_before_share"] > 0.002 and found[0]["motion_after_share"] > 0.002

    # An end card and a gradual fade are not held frames inside moving footage.
    card = np.full((90, 160), 230.0, dtype=np.float32)
    monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate: iter([card] * 80))
    assert visual_timing.frozen_pictures(str(video))["frozen_pictures"] == []
    moving_scene = [np.roll(still, shift, axis=1) for shift in range(30)]
    title_card = moving_scene[:15] + [card] * 30 + moving_scene[15:]
    monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate: iter(title_card))
    assert visual_timing.frozen_pictures(str(video))["frozen_pictures"] == []
    fade = [np.full((90, 160), value, dtype=np.float32) for value in range(20, 201, 4)]
    monkeypatch.setattr(visual_timing, "_grid_frames", lambda path, rate: iter(fade))
    assert visual_timing.frozen_pictures(str(video))["frozen_pictures"] == []


def test_frozen_picture_inside_intent_is_informational_and_judged_clearly():
    item = {"start_sec": 1.0, "end_sec": 1.8, "frames": 25, "duration_sec": 0.8,
            "motion_before_share": 0.04, "motion_after_share": 0.01}
    [row] = _build_candidates({}, {}, {}, [], intent="A deliberate hold from 1.000 s to 2.000 s.",
                              visual={"frozen_pictures": [item]})
    assert row["kind"] == "frozen_picture" and row["rule_verdict"] == "informational"
    from src.loop import _judge_question_text
    assert "picture barely changes" in _judge_question_text(row, "")
