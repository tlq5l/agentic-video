"""Cue-group sync tests for coincident layers and backing percussion."""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.cache import MediaCache
from src.loop import _build_candidates, _judge_candidates, _markdown_report
from src.media.audio import (
    _cue_group_name,
    _simple_confirms_stronger_displaced_attack,
    _soft_temporal_displacement_is_stronger,
    sync_check,
)


RATE = 48000


def _mix(path: Path, duration: float, events) -> Path:
    """Write deterministic sine-burst fixtures: (time, hz, duration, amplitude)."""
    samples = np.zeros(int(duration * RATE), dtype=np.float64)
    for start_sec, frequency, burst_sec, amplitude in events:
        first = int(round(start_sec * RATE))
        count = int(round(burst_sec * RATE))
        local_t = np.arange(count, dtype=np.float64) / RATE
        burst = amplitude * np.sin(2.0 * math.pi * frequency * local_t) * np.exp(-local_t * 24.0)
        last = min(len(samples), first + count)
        samples[first:last] += burst[:last - first]
    pcm = np.clip(samples, -0.95, 0.95)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def _sync(path: Path, cues, cache: MediaCache, bpm=120):
    return sync_check(
        str(path), cues=cues, tolerance_ms=80, search_window_ms=500,
        bpm=bpm, cache=cache, cut_detector=lambda: [],
    )


def test_free_text_labels_share_their_leading_kind():
    labels = [
        "text pop | first lyric", "text pop click 04",
        "text pop: second lyric", "text pop — third lyric",
    ]
    assert {_cue_group_name(label) for label in labels} == {"text pop"}
    assert _cue_group_name("downbeat kick 07") == "downbeat kick"
    assert _cue_group_name("cut whoosh | reveal") == "cut whoosh"


def test_always_coincident_pop_uses_kick_only_bars(tmp_path):
    kick_times = [0.5 + 2.0 * index for index in range(12)]
    pop_times = [0.5 + 4.0 * index for index in range(6)]
    events = [(time, 70, 0.13, 0.75) for time in kick_times]
    events += [(time, 1450, 0.12, 0.50) for index, time in enumerate(pop_times)
               if index not in (2, 4)]
    events.append((pop_times[4] + 0.30, 1450, 0.12, 0.50))
    media = _mix(tmp_path / "always-overlaid.wav", 24.0, events)
    cues = [
        {"time_sec": time, "label": "downbeat kick {:02d}".format(index + 1)}
        for index, time in enumerate(kick_times)
    ] + [
        {"time_sec": time, "label": label}
        for time, label in zip(pop_times, (
            "text pop | first lyric", "text pop click 02", "text pop: third lyric",
            "text pop — fourth lyric", "text pop click 05", "text pop | sixth lyric",
        ))
    ]
    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    pops = [cue for cue in result["cues"] if cue["cue_group"] == "text pop"]
    assert result["cue_signatures"]["text pop"]["method"] == "differential_bar_excess"
    assert [cue["status"] for cue in pops] == ["ok", "ok", "missing", "ok", "late", "ok"]
    assert pops[4]["offset_ms"] == pytest.approx(300.0, abs=25.0)
    assert all(cue["match_method"] in {"differential_bar_excess", "overlap_template", "temporal_group_template"} for cue in pops)
    assert sum(cue["status"] == "ambiguous" for cue in result["cues"]) <= 1


def test_overlap_templates_find_late_early_missing_and_file_start(tmp_path):
    kick_times = [2.0 * index for index in range(12)]
    whoosh_indices = {0, 2, 4, 6, 8, 10}
    click_indices = {1, 3, 5, 7, 9, 11}
    events = [(time, 70, 0.13, 0.75) for time in kick_times]
    for index in whoosh_indices:
        events.append((kick_times[index] + (0.25 if index == 4 else 0.0), 1350, 0.16, 0.55))
    for index in click_indices - {9}:
        events.append((kick_times[index] + (-0.12 if index == 7 else 0.0), 5800, 0.06, 0.50))
    media = _mix(tmp_path / "fully-overlaid.wav", 25.0, events)
    cues = [{"time_sec": time, "label": f"downbeat kick {i:02d}"} for i, time in enumerate(kick_times)]
    cues += [{"time_sec": kick_times[i], "label": f"cut whoosh {i:02d}"} for i in sorted(whoosh_indices)]
    cues += [{"time_sec": kick_times[i], "label": f"text pop click {i:02d}"} for i in sorted(click_indices)]
    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    by_label = {item["label"]: item for item in result["cues"]}
    assert by_label["downbeat kick 00"]["status"] == "ok"
    assert by_label["cut whoosh 04"]["status"] == "late"
    assert by_label["cut whoosh 04"]["offset_ms"] == pytest.approx(250, abs=40)
    assert by_label["text pop click 07"]["status"] == "early"
    assert by_label["text pop click 07"]["offset_ms"] == pytest.approx(-120, abs=40)
    assert by_label["text pop click 09"]["status"] == "missing"
    assert sum(item["status"] == "ambiguous" for item in result["cues"]) <= 1
    planted = {"cut whoosh 04", "text pop click 07", "text pop click 09"}
    assert all(item["status"] == "ok" for item in result["cues"] if item["label"] not in planted)
    assert by_label["cut whoosh 04"]["match_method"] == "temporal_group_template"
    assert by_label["cut whoosh 04"]["group_median_score"] is not None


def test_soft_temporal_displacement_needs_a_stronger_group_band_attack():
    assert _soft_temporal_displacement_is_stronger(0.474, 1.059, 0.367)
    assert not _soft_temporal_displacement_is_stronger(0.40, 1.059, 0.367)
    assert not _soft_temporal_displacement_is_stronger(0.474, 1.059, 0.41)

    simple = {
        "cue_level_db": 1.48,
        "typical_level_db": 3.13,
        "cue_index": 2,
        "onsets": [{"time_sec": 9.173}],
        "level_test": lambda at, index: {
            "rise_db": 15.95, "level_db": 7.29, "reference_db": 4.27,
        },
    }
    assert _simple_confirms_stronger_displaced_attack(simple, 8.889, 9.159, 0.08)
    assert not _simple_confirms_stronger_displaced_attack(
        dict(simple, cue_level_db=2.30), 8.889, 9.159, 0.08,
    )
    assert not _simple_confirms_stronger_displaced_attack(
        dict(simple, level_test=lambda at, index: {
            "rise_db": 2.99, "level_db": 7.29, "reference_db": 4.27,
        }), 8.889, 9.159, 0.08,
    )


def test_late_whoosh_does_not_hide_behind_on_time_kick(tmp_path):
    events = []
    events += [(time, 70, 0.13, 0.8) for time in (0.5, 2.5, 4.5, 8.5)]
    events += [(time, 1450, 0.16, 0.55) for time in (1.5, 3.5, 4.8)]
    media = _mix(tmp_path / "kick-whoosh.wav", 12.0, events)
    cues = [
        {"time_sec": 0.5, "label": "Kick 1"},
        {"time_sec": 2.5, "label": "Kick 2"},
        {"time_sec": 4.5, "label": "Kick 3"},
        {"time_sec": 8.5, "label": "Kick 4"},
        {"time_sec": 1.5, "label": "Whoosh 1"},
        {"time_sec": 3.5, "label": "Whoosh 2"},
        {"time_sec": 4.5, "label": "Whoosh 3"},
    ]

    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    by_label = {item["label"]: item for item in result["cues"]}
    assert by_label["Kick 3"]["status"] == "ok"
    assert by_label["Whoosh 3"]["status"] == "late"
    assert by_label["Whoosh 3"]["offset_ms"] == pytest.approx(300.0, abs=20.0)


def test_missing_click_is_not_satisfied_by_hats_on_its_beat(tmp_path):
    hat_times = [0.5 + 2.0 * index for index in range(8)]
    click_times = (0.5, 2.5, 6.5, 8.5)
    events = [(time, 8200, 0.06, 0.22) for time in hat_times]
    events += [(time, 2600, 0.025, 0.5) for time in click_times]
    media = _mix(tmp_path / "clicks-and-hats.wav", 16.0, events)
    cues = [
        {"time_sec": 0.5, "label": "Click 1"},
        {"time_sec": 2.5, "label": "Click 2"},
        {"time_sec": 4.5, "label": "Click 3"},  # Deliberately absent; hat remains.
        {"time_sec": 6.5, "label": "Click 4"},
        {"time_sec": 8.5, "label": "Click 5"},
    ]

    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    by_label = {item["label"]: item for item in result["cues"]}
    assert by_label["Click 3"]["status"] == "missing"
    assert by_label["Click 3"]["matched_onset_time_sec"] is None
    assert by_label["Click 3"]["offset_ms"] is None
    assert by_label["Click 1"]["status"] == "ok"
    assert by_label["Click 2"]["status"] == "ok"


def test_clean_coincident_cues_use_each_groups_signature(tmp_path):
    events = []
    events += [(time, 70, 0.13, 0.8) for time in (0.5, 2.5, 4.5, 8.5)]
    events += [(time, 1450, 0.16, 0.55) for time in (1.5, 3.5, 4.5)]
    media = _mix(tmp_path / "clean-layers.wav", 12.0, events)
    cues = [
        {"time_sec": 0.5, "label": "Kick 1"},
        {"time_sec": 2.5, "label": "Kick 2"},
        {"time_sec": 4.5, "label": "Kick 3"},
        {"time_sec": 8.5, "label": "Kick 4"},
        {"time_sec": 1.5, "label": "Whoosh 1"},
        {"time_sec": 3.5, "label": "Whoosh 2"},
        {"time_sec": 4.5, "label": "Whoosh 3"},
    ]

    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    assert all(item["status"] == "ok" for item in result["cues"])
    assert result["unmatched_strong_onsets"] == []


def test_just_outside_tolerance_is_early_not_missing(tmp_path):
    media = _mix(tmp_path / "slightly-early.wav", 2.0, [(1.51, 1700, 0.04, 0.6)])
    result = _sync(
        media, [{"time_sec": 1.6, "label": "Hit 1"}],
        MediaCache(tmp_path / "cache"), bpm=None,
    )
    cue = result["cues"][0]
    assert cue["status"] == "early"
    assert cue["offset_ms"] is not None and cue["offset_ms"] < -80.0


def test_one_click_cannot_satisfy_neighbouring_cues(tmp_path):
    cue_times = [0.5, 2.5, 4.5, 6.5, 8.0, 8.5]
    events = [(time, 2800, .04, .6) for time in cue_times[:4]]
    events.append((8.38, 2800, .04, .6))
    media = _mix(tmp_path / "neighbours.wav", 10.0, events)
    cues = [{"time_sec": time, "label": f"click {index:02d}"} for index, time in enumerate(cue_times)]
    result = _sync(media, cues, MediaCache(tmp_path / "cache"))
    by_label = {item["label"]: item for item in result["cues"]}
    assert by_label["click 04"]["status"] == "missing"
    assert by_label["click 05"]["status"] == "early"
    assert by_label["click 05"]["offset_ms"] == pytest.approx(-120, abs=35)


def test_previous_beats_weak_click_is_not_an_edge_early_match(tmp_path):
    times = [0.5 + 2.0 * index for index in range(6)]
    events = [(time, 3100, .045, .60) for time in times[:-1]]
    events.append((times[-1] - .47, 3100, .045, .20))
    media = _mix(tmp_path / "edge-rattle.wav", 12.0, events)
    cues = [{"time_sec": time, "label": f"click {index:02d}"} for index, time in enumerate(times)]
    result = _sync(media, cues, MediaCache(tmp_path / "cache"), bpm=120)
    last = next(item for item in result["cues"] if item["label"] == "click 05")
    assert last["status"] == "missing"
    assert last["matched_onset_time_sec"] is None
    assert last["slot_before_ms"] <= 200.0


def test_cue_candidate_names_exact_label_group_offset_and_score():
    sync = {"tolerance_ms": 80.0, "cues": [
        {"time_sec": 3.0, "label": "cut whoosh | reveal", "cue_group": "cut whoosh",
         "status": "late", "offset_ms": 250.0, "signature_score": .87,
         "group_median_score": 1.1, "match_method": "temporal_group_template"},
    ], "cuts": []}
    candidates = _build_candidates(sync, {"outliers": [], "windows": [], "silence_gaps": []}, {"onsets": []}, [])
    item = candidates[0]
    assert item["cue_label"] == "cut whoosh | reveal"
    assert item["cue_group"] == "cut whoosh"
    assert item["numbers"]["offset_ms"] == 250.0
    assert item["numbers"]["signature_score"] == .87
    report = _markdown_report({"video": "synthetic.mp4", "candidates": candidates})
    assert "cut whoosh | reveal" in report and "0.87" in report


def test_group_wide_lag_is_informational_and_not_confirmed():
    sync = {"tolerance_ms": 80.0, "cues": [
        {"time_sec": 0.0, "label": "cut whoosh 00", "cue_group": "cut whoosh", "status": "ok"},
    ], "cuts": [], "systematic_lags": [
        {"group": "cut whoosh", "lag_ms": 120, "method": "group_template_attack"},
    ]}
    candidates = _build_candidates(sync, {"outliers": [], "windows": [], "silence_gaps": []}, {"onsets": []}, [])
    assert len(candidates) == 1 and candidates[0]["rule_verdict"] == "informational"
    _judge_candidates(candidates, "", judge="none", client=None)
    assert candidates[0]["confirmed"] is False
    assert "Needs review" in _markdown_report({"video": "synthetic.mp4", "candidates": candidates})


def test_cut_with_nearby_onset_is_not_a_cut_without_sound_fault():
    sync = {
        "tolerance_ms": 80.0,
        "cuts": [
            {"time_sec": 1.0, "score": 0.7, "offset_ms": 16.0,
             "nearest_onset": {"time_sec": 1.016, "strength": 0.10}},
            {"time_sec": 2.0, "score": 0.7, "offset_ms": 12.0,
             "nearest_onset": {"time_sec": 2.012, "strength": 0.8}},
        ],
        "cues": [],
    }
    candidates = _build_candidates(sync, {"outliers": [], "windows": [], "silence_gaps": []}, {"onsets": []}, [])
    assert not any(item["kind"] == "cut_without_sound" for item in candidates)
