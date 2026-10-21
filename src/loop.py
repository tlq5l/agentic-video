"""Video check and iterative source-fixing commands (`python -m src.loop`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.cache import MediaCache
from src.media.audio import (audio_onsets, clipping_runs, loudness_timeline, spectrogram_window, stereo_balance_shifts,
                             sync_check)
from src.media.contact_sheet import build_contact_sheet
from src.media.probe import probe_video
from src.media.visual_timing import blank_flashes, frame_cadence, frame_splices, frozen_pictures, text_cue_timing, text_overflow
from src.jev import decide as jev_decide
from src.jev import scrub as scrub_jev


DEFAULT_LADDER = [
    {"model": "gpt-6-luna", "effort": "xhigh"},
    {"model": "gpt-6-sol", "effort": "medium"},
    {"model": "gpt-6-sol", "effort": "high"},
    {"model": "gpt-6-astra", "effort": "low"},
]
DEFAULT_FIXER_CMD = (
    "codex exec -m {model} -c model_reasoning_effort={effort} "
    "-C {project_dir} -s workspace-write -o {last_message_file} -"
)
DEFAULT_OUTPUT_DIR = ".agentic-video-check"
EXIT_RETRY_LATER = 75  # EX_TEMPFAIL: the fixer hit a usage limit; run again later.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _archive_existing(path: Path) -> None:
    """Keep prior generated output recoverable before replacing it."""
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = path.with_name(path.name + ".previous-" + stamp)
    os.replace(str(path), str(archived))


def _fresh_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(path.stem + "-" + stamp + path.suffix)


def _load_cues(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cues", [])
    if not isinstance(data, list):
        raise ValueError("cues.json must contain an array or an object with a 'cues' array")
    deduped: List[Dict[str, Any]] = []
    for item in data:
        t = float(item.get("time_sec", item.get("t", 0.0)))
        label = _label_key(item.get("label", "cue"))
        # The same label within 2 ms is one cue listed twice, not two cues.
        if any(_label_key(kept.get("label", "cue")) == label and abs(_cue_time(kept) - t) <= 0.002
               for kept in deduped):
            continue
        deduped.append(dict(item))
    return deduped


def _label_key(label: Any) -> str:
    return " ".join(str(label).split()).casefold()


def _load_cue_tolerance(path: Optional[str], default_ms: float = 80.0) -> float:
    """Honor optional timing metadata in a cues document."""
    if not path:
        return default_ms
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("tolerance_ms") is not None:
        return float(data["tolerance_ms"])
    return default_ms


def _read_intent(path: Optional[str]) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8") if path else ""


def _cue_time(cue: Mapping[str, Any]) -> float:
    return float(cue.get("time_sec", cue.get("t")))


def _problem_id(kind: str, start_sec: float, label: str = "") -> str:
    identity = "{}|{}|{}".format(kind, round(float(start_sec), 2), label.strip().lower())
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in label.strip())
    slug = "-".join(part for part in slug.split("-") if part)[:24]
    return "{}-{}-{}".format(kind.replace("_", "-"), int(round(start_sec * 1000)), slug or digest)


def _is_nearby_cue(cues: Sequence[Mapping[str, Any]], start: float, end: float, pad: float = 0.75) -> bool:
    return any(start - pad <= _cue_time(cue) <= end + pad for cue in cues)


_TIME_TOKEN = r"((?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"
_TIME_UNIT = r"(?:\s*(ms|milliseconds?|s|secs?|seconds?)\b)?"
_INTENT_RANGE_PATTERNS = (
    re.compile(r"(?i)\b(between)\s+" + _TIME_TOKEN + _TIME_UNIT + r"\s+and\s+" + _TIME_TOKEN + _TIME_UNIT),
    re.compile(r"(?i)(?:\b(from)\s+)?" + _TIME_TOKEN + _TIME_UNIT
               + r"\s*(?:to|until|till|through|thru|-|–|—)\s*" + _TIME_TOKEN + _TIME_UNIT),
    # "from 8.75 s to the reveal at 10 s": a few words may name the end point.
    re.compile(r"(?i)\b(from)\s+" + _TIME_TOKEN + _TIME_UNIT
               + r"\s+(?:up\s+)?(?:to|until|till|into)\s+(?:[A-Za-z]+\s+){1,4}?(?:at\s+)?" + _TIME_TOKEN + _TIME_UNIT),
)
_NON_TIME_WORDS = {"bar", "bars", "beat", "beats", "measure", "measures", "frame", "frames",
                   "db", "hz", "khz", "bpm", "%", "percent", "fps", "px", "x"}


def _intent_seconds(token: str, unit: Optional[str]) -> float:
    if ":" in token:
        value = 0.0
        for part in token.split(":"):
            value = value * 60.0 + float(part)
    else:
        value = float(token)
    return value / 1000.0 if unit and unit.casefold().startswith("m") else value


def _parse_intent_sections(intent: str) -> List[Dict[str, Any]]:
    """Time ranges named in the author's notes, each with the sentence naming it.

    Accepts phrasings such as "from 12.5 s to 15.0 s", "12.5–15 s",
    "between 12.5 and 15 seconds", "0:12.5 - 0:15" and "12500 ms to 15000 ms".
    A bare "12-15" counts only after "from"/"between"; numbers that are bars,
    beats, decibels or frequencies are ignored.
    """
    sections: List[Dict[str, Any]] = []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", intent) if part.strip()]
    for sentence in sentences:
        taken: List[Tuple[int, int]] = []
        for pattern in _INTENT_RANGE_PATTERNS:
            for match in pattern.finditer(sentence):
                if any(match.start() < end and start < match.end() for start, end in taken):
                    continue
                keyword, first, first_unit, second, second_unit = match.groups()
                before = re.findall(r"[A-Za-z%]+", sentence[:match.start()])
                after = re.match(r"\s*([A-Za-z%]+)", sentence[match.end():])
                # Explicit seconds units make it a time range even after a
                # word like "beats" ("across two beats from 12 s to 13 s").
                if not (first_unit or second_unit) and (
                        (before and before[-1].casefold() in _NON_TIME_WORDS)
                        or (after and after.group(1).casefold() in _NON_TIME_WORDS)):
                    continue
                clock = ":" in first or ":" in second
                if not (first_unit or second_unit or clock or keyword):
                    continue
                start_sec = _intent_seconds(first, first_unit or second_unit)
                end_sec = _intent_seconds(second, second_unit or first_unit)
                if end_sec <= start_sec:
                    continue
                taken.append((match.start(), match.end()))
                sections.append({"start_sec": round(start_sec, 6), "end_sec": round(end_sec, 6),
                                 "sentence": sentence})
    return sections


def _intent_section_at(sections: Sequence[Mapping[str, Any]], start: float, end: Optional[float] = None,
                       end_margin: float = 0.0) -> Optional[Mapping[str, Any]]:
    """The intent section covering a moment or span; sections are half-open.

    A range "from A to B" covers [A, B): at B the next section begins. A
    single moment must also lie at least `end_margin` (one cue tolerance)
    before B, so a cue on the boundary is judged normally.
    """
    if end is None:
        return next((section for section in sections
                     if section["start_sec"] - 0.010 <= start < section["end_sec"] - end_margin), None)
    return next((section for section in sections
                 if section["start_sec"] - 0.010 <= start and end <= section["end_sec"]), None)


def _intent_section_for_span(
    sections: Sequence[Mapping[str, Any]], start: float, end: float,
) -> Optional[Mapping[str, Any]]:
    """The intent section containing most of a measured range."""
    duration = end - start
    if duration <= 0.0:
        return _intent_section_at(sections, start)
    return next((section for section in sections
                 if max(0.0, min(end, float(section["end_sec"]))
                        - max(start, float(section["start_sec"]))) / duration > 0.5), None)


def _intent_section_edge(sections: Sequence[Mapping[str, Any]], moment: float,
                         margin: float) -> Optional[Mapping[str, Any]]:
    """The intent section that begins or ends within `margin` of a moment."""
    return next((section for section in sections
                 if min(abs(moment - section["start_sec"]), abs(moment - section["end_sec"])) <= margin), None)


_SWELL_WORDS = re.compile(r"(?i)\b(?:riser|rises?|rising|swells?|swelling|builds?|building|grows?|growing"
                          r"|crescendo|sweeps?|swoosh)\b")


def _has_swell_words_tiebreaker(
    cue_label: str, time_sec: float, intent: str, sections: Sequence[Mapping[str, Any]],
) -> bool:
    """Use rising-sound words only when audio could not measure the build."""
    sound = str(cue_label).split("|", 1)[0].strip()
    if _SWELL_WORDS.search(sound):
        return True
    section = _intent_section_at(sections, time_sec)
    if section is not None and _SWELL_WORDS.search(str(section.get("sentence", ""))):
        return True
    words = {word.casefold() for word in re.findall(r"[A-Za-z]{4,}", sound)}
    if not words:
        return False
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", intent) if part.strip()]
    return any(_SWELL_WORDS.search(sentence) and any(word in sentence.casefold() for word in words)
               for sentence in sentences)


def _build_candidates(
    sync: Mapping[str, Any],
    loudness: Mapping[str, Any],
    onsets: Mapping[str, Any],
    cues: Sequence[Mapping[str, Any]],
    intent: str = "",
    balance: Optional[Mapping[str, Any]] = None,
    visual: Optional[Mapping[str, Any]] = None,
    clipping: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    intent_sections = _parse_intent_sections(intent)

    for cue in sync.get("cues", []):
        status = cue.get("status")
        if status not in {"missing", "early", "late", "ambiguous"}:
            continue
        time_sec = float(cue["time_sec"])
        cue_label = str(cue.get("label", "cue")).strip()

        tolerance = float(sync.get("tolerance_ms") or 80.0) / 1000.0
        matching_intent = _intent_section_at(intent_sections, time_sec, end_margin=tolerance)
        audible_somewhere = cue.get("audible_in_any_band", True)

        if matching_intent is not None and audible_somewhere:
            # The notes say this stretch is deliberately altered, and some
            # sound is still there: report it, but do not call it a fault.
            rule_verdict = "informational"
            classification = "expected to be altered per the author's notes"
            kind = "cue_" + status
        elif status == "ambiguous":
            rule_verdict = "ambiguous"
            classification = "cue match ambiguous"
            kind = "cue_ambiguous"
        elif (cue.get("evidence") == "no_band_beyond_companions"
              and (edge := _intent_section_edge(intent_sections, time_sec, tolerance)) is not None):
            # The only proof it is missing is that no band beats the same
            # companion cues elsewhere, but the mix around a section the notes
            # describe (a dropout, a breath) differs from those moments.
            matching_intent = edge
            rule_verdict = "ambiguous"
            classification = "cue lands on the edge of a deliberate section"
            kind = "cue_ambiguous"
            cue = dict(cue, ambiguous_reason="no band stands out beyond its companion cues, but it lands where a "
                            "section the author's notes describe begins or ends, so those moments do not compare")
        elif (status == "missing" and cue.get("evidence") == "no_band_beyond_companions"
              and not (cue.get("soft_attack_build") or {}).get("conclusive", False)
              and _has_swell_words_tiebreaker(cue_label, time_sec, intent, intent_sections)):
            rule_verdict = "ambiguous"
            classification = "rising sound needs review because its build could not be measured"
            kind = "cue_ambiguous"
            cue = dict(cue, ambiguous_reason="the gradual-build measurement was inconclusive, and the cue or a "
                            "relevant author note describes a riser, swell, or build")
        elif status == "early" and cue.get("matched_onset_time_sec") is not None and _SWELL_WORDS.search(str((
                _intent_section_at(intent_sections, float(cue["matched_onset_time_sec"])) or {}).get("sentence", ""))):
            # The early "match" sits inside a riser or swell the notes
            # describe, which builds into this cue: that sound may be the
            # swell itself, so the timing cannot be trusted.
            matching_intent = _intent_section_at(intent_sections, float(cue["matched_onset_time_sec"]))
            rule_verdict = "ambiguous"
            classification = "matched sound lies inside a deliberate section"
            kind = "cue_ambiguous"
            cue = dict(cue, ambiguous_reason="the {} match at {:.3f} s lies inside a section the author's notes "
                            "describe".format(status, float(cue["matched_onset_time_sec"])))
        else:
            rule_verdict = "fault"
            classification = "cue match " + status
            kind = "cue_" + status

        candidate = {
            "id": _problem_id(kind, time_sec, cue_label),
            "kind": kind,
            "cue_label": cue_label,
            "cue_group": cue.get("cue_group"),
            "start_sec": time_sec,
            "end_sec": time_sec,
            "numbers": {
                "cue_time_sec": time_sec,
                "offset_ms": cue.get("offset_ms"),
                "matched_onset_time_sec": cue.get("matched_onset_time_sec"),
                "nearest_onset_time_sec": cue.get("nearest_onset_time_sec"),
                "signature_score": cue.get("signature_score"),
                "group_median_score": cue.get("group_median_score"),
                "on_cue_score": cue.get("on_cue_score"),
                "shifted_null_floor": cue.get("shifted_null_floor"),
                "credible_score_floor": cue.get("credible_score_floor"),
                "slot_before_ms": cue.get("slot_before_ms"),
                "slot_after_ms": cue.get("slot_after_ms"),
                "group_typical_attack_lag_ms": cue.get("group_typical_attack_lag_ms"),
                "signature_excess_ratio": cue.get("signature_excess_ratio"),
                "signature_bands": cue.get("signature_bands"),
                "tolerance_ms": sync.get("tolerance_ms"),
                "band_hz": cue.get("band_hz"),
                "dominant_band": cue.get("dominant_band"),
                "simple_check_on_time": cue.get("simple_check_on_time"),
                "simple_check_level_db": cue.get("simple_check_level_db"),
                "simple_check_reference_db": cue.get("simple_check_reference_db"),
                "template_status": cue.get("template_status"),
                "evidence": cue.get("evidence"),
                "audible_in_any_band": cue.get("audible_in_any_band"),
                "soft_attack_build": cue.get("soft_attack_build"),
            },
            "classification": classification,
            "shape": None,
            "match_method": cue.get("match_method"),
            "ambiguous_reason": cue.get("ambiguous_reason"),
            "nearby_cues": [dict(item) for item in cues if abs(_cue_time(item) - time_sec) <= 1.0],
            "rule_verdict": rule_verdict,
            "_marks": [{"time_sec": time_sec, "label": cue_label[:60]}],
        }
        if matching_intent is not None:
            candidate["intent_note"] = matching_intent["sentence"]
        candidates.append(candidate)

    for lag in sync.get("systematic_lags", []):
        group = str(lag["group"])
        first = next((cue for cue in sync.get("cues", []) if cue.get("cue_group") == group), None)
        if first is None:
            continue
        time_sec = float(first["time_sec"])
        candidates.append({
            "id": _problem_id("cue_group_systematic_lag", time_sec, group),
            "kind": "cue_group_systematic_lag", "cue_label": "group '{}'".format(group),
            "cue_group": group, "start_sec": time_sec, "end_sec": time_sec,
            "numbers": {"group_typical_attack_lag_ms": lag["lag_ms"], "method": lag["method"]},
            "classification": "informational", "shape": None,
            "rule_verdict": "informational", "_marks": [{"time_sec": time_sec, "label": group[:60]}],
            "nearby_cues": [dict(item) for item in cues if abs(_cue_time(item) - time_sec) <= 1.0],
        })

    step_groups: List[List[Mapping[str, Any]]] = []
    for item in loudness.get("outliers", []):
        if not item.get("fault_level") or item.get("ignored") or item.get("shape") != "step":
            continue
        start, end = float(item["start_sec"]), float(item["end_sec"])
        group = next((existing for existing in step_groups if
                      item.get("direction") == existing[0].get("direction")
                      and min(end, float(existing[0]["end_sec"]))
                      >= max(start, float(existing[0]["start_sec"]))), None)
        if group is None:
            step_groups.append([item])
        else:
            group.append(item)

    for group in step_groups:
        representative = max(group, key=lambda item: abs(float(item.get("delta_db", 0.0))))
        start = min(float(item["start_sec"]) for item in group)
        end = max(float(item["end_sec"]) for item in group)
        bands = sorted({str(item.get("band", "band")) for item in group})
        numbers = dict(representative)
        numbers["bands"] = bands
        numbers["per_band"] = [dict(item) for item in group]
        candidates.append({
            "id": _problem_id("loudness_step", start, "-".join(bands)),
            "kind": "loudness_step",
            "start_sec": start,
            "end_sec": end,
            "numbers": numbers,
            "classification": representative.get("classification"),
            "shape": representative.get("shape"),
            "nearby_cues": [dict(cue) for cue in cues if start - 1.0 <= _cue_time(cue) <= end + 1.0],
            "rule_verdict": "fault",
            "_marks": [
                {"time_sec": start, "label": "step begins"},
                {"time_sec": end, "label": "step ends"},
            ],
        })

    windows = loudness.get("windows", [])
    cue_fault_times = [
        float(item["start_sec"]) for item in candidates
        if item.get("kind") in {"cue_missing", "cue_early", "cue_late"}
        and item.get("rule_verdict") == "fault"
    ]
    for gap in loudness.get("silence_gaps", []):
        start, end = float(gap["start_sec"]), float(gap["end_sec"])
        duration = end - start
        if duration < 0.25:
            continue
        # The same absent/late cue can also create a short measured silence;
        # keep that as one cue fault instead of reporting the gap twice.
        if any(start - 0.15 <= cue_time <= end + 0.15 for cue_time in cue_fault_times):
            continue
        before = any(not w.get("silent") and float(w["end_sec"]) >= start - 0.75 and float(w["end_sec"]) <= start + 0.02 for w in windows)
        after = any(not w.get("silent") and float(w["start_sec"]) <= end + 0.02 and float(w["start_sec"]) >= end - 0.75 for w in windows)
        in_cue = any(start <= _cue_time(cue) <= end for cue in cues)
        if not in_cue and not (before and after and duration >= 0.4):
            continue
        section = _intent_section_for_span(intent_sections, start, end)
        candidates.append({
            "id": _problem_id("programme_silence", start),
            "kind": "programme_silence",
            "start_sec": start,
            "end_sec": end,
            "numbers": {**dict(gap), "audible_before": before, "audible_after": after},
            "classification": "silence inside programme",
            "shape": "gap",
            "nearby_cues": [dict(cue) for cue in cues if start - 1.0 <= _cue_time(cue) <= end + 1.0],
            "rule_verdict": ("informational" if section is not None
                             else "fault" if duration >= 0.4 or in_cue else "intentional"),
            "_marks": [
                {"time_sec": start, "label": "silence begins"},
                {"time_sec": end, "label": "silence ends"},
            ],
        })
        if section is not None:
            candidates[-1]["intent_note"] = section["sentence"]

    # One row per re-panned stretch, however many bands moved with it.
    shift_groups: List[List[Mapping[str, Any]]] = []
    for item in (balance or {}).get("shifts", []):
        group = next((existing for existing in shift_groups
                      if item["side"] == existing[0]["side"]
                      and min(float(item["end_sec"]), float(existing[0]["end_sec"]))
                      > max(float(item["start_sec"]), float(existing[0]["start_sec"]))), None)
        if group is None:
            shift_groups.append([item])
        else:
            group.append(item)
    for group in shift_groups:
        representative = max(group, key=lambda item: abs(float(item["delta_db"])))
        start = min(float(item["start_sec"]) for item in group)
        end = max(float(item["end_sec"]) for item in group)
        new_content = all(item.get("new_content") for item in group)
        candidates.append({
            "id": _problem_id("stereo_balance_shift", start, representative["side"]),
            "kind": "stereo_balance_shift", "start_sec": start, "end_sec": end,
            "numbers": {**dict(representative), "bands": sorted({str(item["band"]) for item in group}),
                        "per_band": [dict(item) for item in group]},
            "classification": ("new content on one side (arrangement change)" if new_content
                               else "existing sound re-panned"),
            "shape": "step",
            "nearby_cues": [dict(cue) for cue in cues if start - 1.0 <= _cue_time(cue) <= end + 1.0],
            "rule_verdict": "informational" if new_content else "fault",
            "_marks": [{"time_sec": start, "label": "balance shifts " + representative["side"]},
                       {"time_sec": end, "label": "balance returns"}],
        })

    # A low band that is one-sided for the whole piece has no "usual" to
    # shift from, so it is its own row: one per side.
    for side in ("left", "right"):
        group = [item for item in (balance or {}).get("off_centre", []) if item["side"] == side]
        if not group:
            continue
        representative = max(group, key=lambda item: abs(float(item["balance_db"])))
        start = min(float(item["start_sec"]) for item in group)
        end = max(float(item["end_sec"]) for item in group)
        candidates.append({
            "id": _problem_id("stereo_off_centre", start, side),
            "kind": "stereo_off_centre", "start_sec": start, "end_sec": end,
            "numbers": {**dict(representative), "bands": sorted({str(item["band"]) for item in group}),
                        "per_band": [dict(item) for item in group]},
            "classification": "low end panned to one side",
            "shape": "whole piece",
            "nearby_cues": [],
            "rule_verdict": "fault",
            "_marks": [{"time_sec": start, "label": "low end sits " + side}],
        })

    # A master loud throughout is limited, not clipped at one moment: that
    # is a mastering choice, so its runs are informational.
    throughout = bool((clipping or {}).get("throughout"))
    for item in (clipping or {}).get("runs", []):
        start, end = float(item["start_sec"]), float(item["end_sec"])
        section = _intent_section_for_span(intent_sections, start, end)
        in_notes = bool(item.get("in_ignored_range") or section is not None)
        candidate = {
            "id": _problem_id("clipping", start),
            "kind": "clipping", "start_sec": start, "end_sec": end,
            "numbers": {**dict(item), "hot_share": (clipping or {}).get("hot_share"),
                        "level": (clipping or {}).get("level")},
            "classification": ("expected to be altered per the author's notes" if in_notes
                               else "master loud throughout" if throughout else "clipped peaks"),
            "shape": "burst",
            "nearby_cues": [dict(cue) for cue in cues if start - 0.5 <= _cue_time(cue) <= end + 0.5],
            "rule_verdict": "informational" if in_notes or throughout else "fault",
            "_marks": [{"time_sec": start, "label": "clipping"}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    for item in (visual or {}).get("text_cues", []):
        if item.get("status") not in {"early", "late"}:
            continue
        time_sec = float(item["time_sec"])
        section = _intent_section_at(intent_sections, time_sec)
        kind = "text_" + str(item["status"])
        candidate = {
            "id": _problem_id(kind, time_sec, str(item["label"])),
            "kind": kind, "cue_label": str(item["label"]),
            "start_sec": min(time_sec, float(item["appears_sec"])), "end_sec": max(time_sec, float(item["appears_sec"])),
            "numbers": dict(item),
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else "on-screen text " + str(item["status"])),
            "shape": None,
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - time_sec) <= 1.0],
            "rule_verdict": "informational" if section is not None else "fault",
            "_marks": [{"time_sec": time_sec, "label": "cue"},
                       {"time_sec": float(item["appears_sec"]), "label": "text appears"}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    for item in (visual or {}).get("flashes", []):
        start, end = float(item["start_sec"]), float(item["end_sec"])
        section = _intent_section_at(intent_sections, start)
        candidate = {
            "id": _problem_id("blank_flash", start),
            "kind": "blank_flash", "start_sec": start, "end_sec": end,
            "numbers": dict(item),
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else "{} near-black frame(s) between two pictures".format(item["frames"])),
            "shape": None,
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - start) <= 1.0],
            "rule_verdict": "informational" if section is not None else "fault",
            "_marks": [{"time_sec": start, "label": "black frame"}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    flash_spans = [(float(item["start_sec"]), float(item["end_sec"])) for item in (visual or {}).get("flashes", [])]
    for item in (visual or {}).get("frame_splices", []):
        start, end = float(item["start_sec"]), float(item["end_sec"])
        if any(start < flash_end and flash_start < end for flash_start, flash_end in flash_spans):
            continue  # already reported as a blank flash
        section = _intent_section_for_span(intent_sections, start, end)
        candidate = {
            "id": _problem_id("flash_frame", start),
            "kind": "flash_frame", "start_sec": start, "end_sec": end,
            "numbers": dict(item),
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else "brief frame differs from matching pictures on both sides"),
            "shape": "spliced frame run",
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - start) <= 1.0],
            "rule_verdict": "informational" if section is not None else "fault",
            "_marks": [{"time_sec": start, "label": "frame splice"},
                       {"time_sec": end, "label": "picture resumes"}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    for item in (visual or {}).get("frozen_pictures", []):
        start, end = float(item["start_sec"]), float(item["end_sec"])
        section = _intent_section_for_span(intent_sections, start, end)
        candidate = {
            "id": _problem_id("frozen_picture", start),
            "kind": "frozen_picture", "start_sec": start, "end_sec": end,
            "numbers": dict(item),
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else "picture holds still while nearby frames move"),
            "shape": "held frame run",
            "nearby_cues": [dict(cue) for cue in cues if start - 1.0 <= _cue_time(cue) <= end + 1.0],
            "rule_verdict": "informational" if section is not None else "fault",
            "_marks": [{"time_sec": start, "label": "picture holds"},
                       {"time_sec": end, "label": "picture moves again"}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    for item in (visual or {}).get("overflows", []):
        time_sec = float(item["time_sec"])
        section = _intent_section_at(intent_sections, time_sec)
        touches_edge = item.get("touches_edge", True)
        frames = item.get("frames_touching") or item.get("frames_near") or item.get("frames_checked", [])
        side = str(item["side"])
        candidate = {
            "id": _problem_id("text_cropped", time_sec, str(item["label"])),
            "kind": "text_cropped", "cue_label": str(item["label"]),
            "start_sec": float(item["appears_sec"]), "end_sec": max(frames),
            "numbers": dict(item),
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else ("on-screen text runs off the " + side + " edge" if touches_edge else
                                     "on-screen text ends close to the " + side + " edge")),
            "shape": None,
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - time_sec) <= 1.0],
            "rule_verdict": ("informational" if section is not None else
                             "fault" if touches_edge else "ambiguous"),
            "_marks": [{"time_sec": float(moment),
                        "label": "text at " + side + " edge" if touches_edge else "text near " + side + " edge"}
                       for moment in frames[:1]],
        }
        if section is None and not touches_edge:
            candidate["ambiguous_reason"] = "text ends {} px from the {} edge; it may be cut off between letters".format(
                item.get("margin_px"), side,
            )
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    for item in sync.get("uncued_sounds", []):
        time_sec = float(item["time_sec"])
        section = _intent_section_at(intent_sections, time_sec)
        group = str(item["group"])
        candidate = {
            "id": _problem_id("uncued_sound", time_sec, group),
            "kind": "uncued_sound", "cue_label": "uncued '{}'".format(group), "cue_group": group,
            "start_sec": time_sec, "end_sec": time_sec,
            "numbers": {**dict(item), "tolerance_ms": sync.get("tolerance_ms")},
            "classification": ("expected to be altered per the author's notes" if section is not None
                               else "sound with no cue"),
            "shape": None,
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - time_sec) <= 1.0],
            "rule_verdict": "informational" if section is not None else "fault",
            "_marks": [{"time_sec": time_sec, "label": "uncued " + group[:50]}],
        }
        if section is not None:
            candidate["intent_note"] = section["sentence"]
        candidates.append(candidate)

    cuts = list(sync.get("cuts", []))
    cut_sound_window_ms = max(150.0, float(sync.get("tolerance_ms", 80.0)))
    onset_near_cut = [
        bool(cut.get("offset_ms") is not None and abs(float(cut["offset_ms"])) <= cut_sound_window_ms)
        for cut in cuts
    ]
    patterned_cuts = len(cuts) >= 2 and sum(onset_near_cut) >= 2 and sum(onset_near_cut) / len(cuts) >= 0.6
    for index, cut in enumerate(cuts):
        time_sec = float(cut["time_sec"])
        has_sound = onset_near_cut[index]
        cue_expected = _is_nearby_cue(cues, time_sec, time_sec, pad=0.25)
        if has_sound or not (cue_expected or patterned_cuts):
            continue
        candidates.append({
            "id": _problem_id("cut_without_sound", time_sec),
            "kind": "cut_without_sound",
            "start_sec": time_sec,
            "end_sec": time_sec,
            "numbers": {
                "cut_score": cut.get("score"),
                "nearest_onset": cut.get("nearest_onset"),
                "nearest_onset_offset_ms": cut.get("offset_ms"),
                "expected_from_cues": cue_expected,
                "expected_from_cut_pattern": patterned_cuts,
                "cuts_with_nearby_sound": sum(onset_near_cut),
                "total_cuts": len(cuts),
            },
            "classification": "visual cut without nearby sound",
            "shape": None,
            "nearby_cues": [dict(cue) for cue in cues if abs(_cue_time(cue) - time_sec) <= 1.0],
            "rule_verdict": "fault",
            "_marks": [{"time_sec": time_sec, "label": "cut"}],
        })

    candidates = _collapse_duplicate_rows(candidates)
    candidates.sort(key=lambda item: (item["start_sec"], item["kind"], item["id"]))
    return candidates


def _collapse_duplicate_rows(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (label, time): keep the most severe verdict and a unique ID."""
    severity = {"fault": 3, "ambiguous": 2, "informational": 1, "intentional": 0}
    kept: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    order: List[Tuple[str, str, float]] = []
    for item in candidates:
        family = "cue" if str(item["kind"]).startswith("cue_") else str(item["kind"])
        key = (family, _label_key(item.get("cue_label") or ""), round(float(item["start_sec"]), 2))
        if key not in kept:
            kept[key] = item
            order.append(key)
        elif severity.get(item.get("rule_verdict"), 0) > severity.get(kept[key].get("rule_verdict"), 0):
            kept[key] = item
    rows = [kept[key] for key in order]
    seen_ids: Dict[str, int] = {}
    for item in rows:
        count = seen_ids.get(item["id"], 0)
        seen_ids[item["id"]] = count + 1
        if count:
            item["id"] = "{}-{}".format(item["id"], count + 1)
    return rows


def _rule_for_jev(value: str) -> str:
    if value == "fault":
        return "real fault"
    if value == "ambiguous":
        return "measurement unreliable"
    return "intentional per the author's notes"


def _relevant_intent(intent: str, item: Mapping[str, Any]) -> str:
    if item.get("intent_note"):
        return item["intent_note"]
    notes = intent.strip()
    if not notes:
        return "No author intent notes were provided."
    start_sec = float(item.get("start_sec", item.get("time_sec", 0.0)))
    tolerance = float(item.get("numbers", {}).get("tolerance_ms") or 80.0) / 1000.0
    section = _intent_section_at(_parse_intent_sections(notes), start_sec, end_margin=tolerance)
    if section is not None:
        return section["sentence"]
    labels = [str(cue.get("label", "")).casefold() for cue in item.get("nearby_cues", [])]
    if item.get("cue_label"):
        labels.append(str(item["cue_label"]).casefold())
    terms = {word for label in labels for word in label.split() if len(word) >= 4}
    # Sentences about other time ranges do not apply here.
    ranged = {section["sentence"] for section in _parse_intent_sections(notes)}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", notes) if s.strip() and s.strip() not in ranged]
    matching = [s for s in sentences if any(t in s.casefold() for t in terms)]
    if matching:
        return "\n".join(matching)[:1200]
    general = " ".join(sentences)
    if general and len(general) <= 300:
        return general
    return "No author intent notes specifically mention this section."


def _judge_question_text(item: Mapping[str, Any], intent: str) -> str:
    numbers = item.get("numbers", {})
    kind = str(item.get("kind", "candidate"))
    start = float(item.get("start_sec", 0.0))
    offset = numbers.get("offset_ms")
    tolerance = numbers.get("tolerance_ms")
    if kind == "cue_group_systematic_lag":
        measurement = "The '{}' group reaches its typical detectable attack energy {} ms after its cue; this is informational.".format(
            item.get("cue_group", "cue"), numbers.get("group_typical_attack_lag_ms"),
        )
    elif kind.startswith("cue_"):
        labels = [str(cue.get("label", "cue")) for cue in item.get("nearby_cues", [])
                  if abs(_cue_time(cue) - start) <= 0.01]
        label = str(item.get("cue_label") or (labels[0] if labels else "the cue"))
        if kind == "cue_missing":
            measurement = "No sound matching '{}' was found near its cue at {:.3f} s (tolerance {} ms).".format(label, start, tolerance)
        elif kind == "cue_ambiguous":
            measurement = "At {:.3f} s, the measured sound could not be separated for '{}': {}.".format(
                start, label, item.get("ambiguous_reason") or "the spectral evidence is inconclusive",
            )
        else:
            direction = "before" if offset is not None and float(offset) < 0 else "after"
            measurement = "The '{}' sound was found {:.1f} ms {} its cue at {:.3f} s; tolerance is {} ms.".format(
                label, abs(float(offset or 0.0)), direction, start, tolerance,
            )
    elif kind == "uncued_sound":
        measurement = ("A '{}' sound plays at {:.3f} s (template score {} vs the group's usual {}; its band at {} dB, "
                       "usual {} dB), but no '{}' cue is near it.").format(
            item.get("cue_group"), start, numbers.get("score"), numbers.get("group_median_score"),
            numbers.get("simple_check_level_db"), numbers.get("group_typical_level_db"), item.get("cue_group"),
        )
        if numbers.get("also_matches"):
            measurement += " It also matches the '{}' pattern; these sounds always play together at cues.".format(
                "', '".join(numbers["also_matches"]))
    elif kind in {"text_early", "text_late"}:
        measurement = ("The on-screen text for '{}' (cue {:.3f} s) appears at {:.3f} s, {:.0f} ms {}; nothing in the "
                       "picture changes at the cue itself.").format(
            item.get("cue_label"), start, float(numbers.get("appears_sec", start)), abs(float(numbers.get("offset_ms") or 0.0)),
            "after its cue" if kind == "text_late" else "before its cue",
        )
    elif kind == "blank_flash":
        measurement = ("From {:.3f} to {:.3f} s, {} frame(s) are near-black (brightest 1% of the picture under 40 of 255), "
                       "between steady pictures (brightest {} before, {} after); a fade or a dark scene lasts longer.").format(
            start, float(item.get("end_sec", start)), numbers.get("frames"), numbers.get("brightest_before"),
            numbers.get("brightest_after"),
        )
    elif kind == "flash_frame":
        measurement = ("From {:.3f} to {:.3f} s, {} frame(s) differ from both surrounding pictures by at least {:.1f} "
                       "grey levels, while the picture before and after differs by only {:.1f}; this is a brief frame splice.").format(
            start, float(item.get("end_sec", start)), numbers.get("frames"),
            min(float(numbers.get("frame_delta_before", 0.0)), float(numbers.get("frame_delta_after", 0.0))),
            numbers.get("context_delta"),
        )
    elif kind == "frozen_picture":
        measurement = ("From {:.3f} to {:.3f} s, the picture barely changes for {} frames, while nearby frames move "
                       "(motion before {}, after {}). A still title or end card has no moving picture on both sides.").format(
            start, float(item.get("end_sec", start)), numbers.get("frames"),
            numbers.get("motion_before_share"), numbers.get("motion_after_share"),
        )
    elif kind == "text_cropped":
        side = numbers.get("side")
        if numbers.get("touches_edge") is False:
            frames = numbers.get("frames_near", numbers.get("frames_checked", []))
            bands = numbers.get("line_bands", [])
            band_summary = ", ".join(
                "{} px tall".format(band.get("height_px"))
                for band in bands
            ) or "a text line"
            measurement = ("After the text for '{}' appears ({:.3f} s), the stable text-like line band(s) {} end within "
                           "{} px of the {} edge across frames {}; the preceding frame had {} comparable near-edge "
                           "positions. These measurements cannot distinguish a cut-off stroke from a complete letter "
                           "ending before a small dark gap.").format(
                item.get("cue_label"), start, band_summary, numbers.get("margin_px"), side,
                ", ".join("{:.2f} s".format(float(moment)) for moment in frames),
                numbers.get("previous_comparable_positions", 0),
            )
        else:
            frames = numbers.get("frames_touching", [])
            measurement = ("After the text for '{}' appears ({:.3f} s), strokes that stay in place across frames {} run into "
                           "the {} edge of the frame on {} lines; just before it appeared, {} lines touched that edge. "
                           "These measurements indicate strokes at the edge, but do not show whether the affected "
                           "letters or words are visibly incomplete.").format(
                item.get("cue_label"), start,
                ", ".join("{:.2f} s".format(float(moment)) for moment in frames),
                side, numbers.get("lines_touching"), numbers.get("lines_before"),
            )
    elif kind == "stereo_balance_shift":
        measurement = ("From {:.3f} to {:.3f} s the {} band sits {:.1f} dB to the {} (right minus left {} dB) "
                       "against its usual {} dB elsewhere in the piece; its level changes by {} dB there.").format(
            start, float(item.get("end_sec", start)), ", ".join(numbers.get("bands", [])),
            abs(float(numbers.get("delta_db", 0.0))), numbers.get("side"), numbers.get("balance_db"),
            numbers.get("usual_balance_db"), numbers.get("level_change_db"),
        )
    elif kind == "stereo_off_centre":
        measurement = ("Across the piece ({:.3f} to {:.3f} s, {:.0%} of bars) the {} band sits {:.1f} dB to the {} "
                       "(right minus left); low end is normally centred.").format(
            start, float(item.get("end_sec", start)), float(numbers.get("share_of_windows", 0.0)),
            ", ".join(numbers.get("bands", [])), abs(float(numbers.get("balance_db", 0.0))), numbers.get("side"),
        )
    elif kind == "clipping":
        measurement = ("From {:.3f} to {:.3f} s, {} samples sit at or above {} of full scale (peak {}); "
                       "{:.0%} of the programme is that hot.").format(
            start, float(item.get("end_sec", start)), numbers.get("samples_at_ceiling"), numbers.get("level"),
            numbers.get("peak"), float(numbers.get("hot_share") or 0.0),
        )
    elif kind == "loudness_step":
        measurement = "A step-shaped level change from {:.3f} to {:.3f} s measured {:.1f} dB in band(s) {}.".format(
            start, float(item.get("end_sec", start)), float(numbers.get("delta_db", 0.0)),
            ", ".join(numbers.get("bands", [numbers.get("band", "unknown")])),
        )
    elif kind == "programme_silence":
        measurement = "The programme went silent from {:.3f} to {:.3f} s ({:.3f} s).".format(
            start, float(item.get("end_sec", start)), float(numbers.get("duration_sec", 0.0)),
        )
    elif kind == "cut_without_sound":
        measurement = "A visual cut at {:.3f} s has no sound within {:.0f} ms; nearest onset offset is {} ms.".format(
            start, float(numbers.get("tolerance_ms") or 150.0), numbers.get("nearest_onset_offset_ms"),
        )
    else:
        measurement = "At {:.3f} s, {} was measured: {}.".format(start, kind, json.dumps(numbers, sort_keys=True))
    return (
        measurement + "\nAuthor intent notes: " + _relevant_intent(intent, item)
        + "\nChoose whether this is a real unintended fault, intentional per the notes, or a measurement that cannot be trusted."
    )


def _judge_candidates(
    candidates: Sequence[Dict[str, Any]],
    intent: str,
    *,
    judge: str,
    client: Any,
) -> Optional[str]:
    if judge == "none" or not candidates:
        for item in candidates:
            item["judge_verdict"] = None
            item["confirmed"] = item["rule_verdict"] == "fault"
            item["disagreement"] = False
        return None

    questions = {}
    for index, item in enumerate(candidates):
        qid = "candidate_{}".format(index + 1)
        item["_judge_question"] = qid
        questions[qid] = {
            "type": "choice",
            "instructions": _judge_question_text(item, intent),
            "criteria": {
                "real fault": "Reliable evidence shows an unintended source fault the editor should fix.",
                "intentional per the author's notes": "The author's notes explain this event as deliberate or harmless.",
                "measurement unreliable": "The audio evidence cannot separate the sounds or does not support a reliable verdict.",
            },
        }
    state = {
        "task": "Evaluate each measured candidate independently. Do not infer a fault from an ambiguous measurement.",
        "candidates": [
            {
                "question": item["_judge_question"],
                "id": item["id"],
                "kind": item["kind"],
                "time_sec": [item["start_sec"], item["end_sec"]],
                "numbers": item["numbers"],
                "classification": item.get("classification"),
                "shape": item.get("shape"),
                "nearby_cues": item.get("nearby_cues", []),
                "rule_verdict": _rule_for_jev(item["rule_verdict"]),
                "relevant_intent_notes": _relevant_intent(intent, item),
            }
            for item in candidates
        ],
    }
    try:
        response = client(state, questions)
        answers = response.get("answers", {})
        for item in candidates:
            qid = item.pop("_judge_question")
            answer = answers.get(qid, {})
            choice = answer.get("choice")
            aliases = {
                "fault": "real fault", "intentional": "intentional per the author's notes",
                "harmless": "intentional per the author's notes",
            }
            choice = aliases.get(choice, choice)
            if choice not in {"real fault", "intentional per the author's notes", "measurement unreliable"}:
                raise ValueError("Jev did not return a valid choice for every candidate")
            confidence = answer.get("confidence")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and not math.isfinite(confidence):
                confidence = None
            if confidence is not None and not 0.0 <= confidence <= 1.0:
                confidence = None
            verdict = choice
            item["judge_verdict"] = {
                "verdict": verdict,
                "choice": choice,
                "confidence": confidence,
                "provider": response.get("provider"),
                "model": response.get("model"),
            }
            item["confirmed"] = item["rule_verdict"] == "fault" and choice == "real fault"
            item["disagreement"] = choice != _rule_for_jev(item["rule_verdict"])
        return None
    except Exception as exc:
        message = scrub_jev(exc)
        for item in candidates:
            item.pop("_judge_question", None)
            item["judge_verdict"] = {"verdict": "unavailable", "confidence": None, "error": message, "fallback": "rules"}
            item["confirmed"] = item["rule_verdict"] == "fault"
            item["disagreement"] = False
        return "Jev unavailable; rule verdicts used ({}).".format(message)


def _copy_evidence(
    candidates: Sequence[Dict[str, Any]],
    video: Path,
    duration: float,
    has_audio: bool,
    evidence_dir: Path,
    cache: MediaCache,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(candidates, start=1):
        stem = "{:03d}-{}".format(index, item["id"][:64])
        evidence: Dict[str, Any] = {"spectrogram": None, "contact_sheet": None}
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        window_start = max(0.0, start - 1.0)
        window_end = min(duration, max(end + 1.0, window_start + 0.2))
        if has_audio and window_end > window_start:
            spec = spectrogram_window(
                str(video), window_start, window_end,
                marks=item.pop("_marks", []), cache=cache,
            )
            spec_target = _fresh_path(evidence_dir / (stem + "-spectrogram.png"))
            spec_target.write_bytes(Path(spec["file_path"]).read_bytes())
            evidence["spectrogram"] = str(spec_target.relative_to(evidence_dir.parent))
        else:
            item.pop("_marks", None)
            evidence["spectrogram_unavailable"] = "video has no audio stream"

        frame_times = sorted({
            round(min(max(0.0, start - 0.45), max(0.0, duration - 0.001)), 3),
            round(min(max(0.0, (start + end) / 2.0), max(0.0, duration - 0.001)), 3),
            round(min(max(0.0, end + 0.45), max(0.0, duration - 0.001)), 3),
        })
        try:
            sheet = build_contact_sheet(
                str(video), frame_times, output_path=_fresh_path(evidence_dir / (stem + "-frames.jpg")),
                cols=min(3, len(frame_times)), title="Evidence around {:.3f}s".format(start), cache=cache,
            )
            evidence["contact_sheet"] = str(Path(sheet.file_path).relative_to(evidence_dir.parent))
        except Exception as exc:
            evidence["contact_sheet_error"] = scrub_jev(exc)[:300]
        item["evidence"] = evidence


def _markdown_report(report: Mapping[str, Any]) -> str:
    candidates = list(report.get("candidates", []))
    confirmed = [item for item in candidates if item.get("confirmed")]
    review = [item for item in candidates if not item.get("confirmed") and (
        item.get("rule_verdict") in {"ambiguous", "informational"}
        or item.get("classification") == "informational"
    )]
    other = [item for item in candidates if not item.get("confirmed") and item not in review]
    lines = ["# Video check report", "", "Video: `{}`".format(report.get("video", "")), ""]
    lines.append("Confirmed problems: **{}**".format(len(confirmed)))
    if report.get("judge_notice"):
        lines.extend(["", "Judge: {}".format(report["judge_notice"])])
    if report.get("judge_agreement"):
        agreement = report["judge_agreement"]
        lines.extend(["", "Jev agreement with rules: **{}/{} ({:.1f}%)**".format(
            agreement["agreed"], agreement["total"], 100.0 * agreement["rate"],
        )])
    if not candidates:
        lines.extend(["", "No candidate problems found."])
        return "\n".join(lines) + "\n"

    for heading, group in (("Confirmed problems", confirmed), ("Needs review", review), ("Not confirmed", other)):
        if not group:
            continue
        lines.extend(["", "## {}".format(heading), ""])
        for item in group:
            lines.append("### {} — {:.3f}s{}".format(
                item.get("kind", "problem"), item.get("start_sec", 0.0),
                "–{:.3f}s".format(item["end_sec"]) if item.get("end_sec") != item.get("start_sec") else "",
            ))
            lines.append("")
            lines.append("- ID: `{}`".format(item.get("id")))
            if item.get("cue_label"):
                lines.append("- Cue: **{}** (group `{}`); offset **{} ms**; score **{}**".format(
                    item["cue_label"], item.get("cue_group"), item.get("numbers", {}).get("offset_ms"),
                    item.get("numbers", {}).get("signature_score"),
                ))
            lines.append("- Rule: **{}**".format(item.get("rule_verdict")))
            verdict = item.get("judge_verdict")
            if verdict:
                if verdict.get("verdict") == "unavailable":
                    lines.append("- Jev: unavailable; fell back to rules")
                else:
                    lines.append("- Jev: **{}** (confidence {})".format(verdict.get("verdict"), verdict.get("confidence")))
            if item.get("disagreement"):
                lines.append("- **Rule and Jev disagree.**")
            lines.append("- Evidence: {}".format(", ".join(
                "[{}]({})".format(key.replace("_", " "), path)
                for key, path in item.get("evidence", {}).items()
                if key in {"spectrogram", "contact_sheet"} and path
            ) or "none available"))
            lines.append("- Measurements: `{}`".format(json.dumps(item.get("numbers", {}), ensure_ascii=False, sort_keys=True)))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _measurement_summary(
    sync: Mapping[str, Any], loudness: Mapping[str, Any], onsets: Mapping[str, Any],
) -> Dict[str, Any]:
    strongest = sorted(onsets.get("onsets", []), key=lambda item: -float(item.get("strength", 0.0)))[:40]
    return {
        "sync_check": {
            "tolerance_ms": sync.get("tolerance_ms"),
            "total_cues": sync.get("total_cues", len(sync.get("cues", []))),
            "cues": sync.get("cues", []),
            "total_cuts": sync.get("total_cuts", len(sync.get("cuts", []))),
        },
        "loudness_timeline": {
            "window_sec": loudness.get("window_sec"),
            "bpm": loudness.get("bpm"),
            "silence_gaps": loudness.get("silence_gaps", []),
            "outliers": loudness.get("outliers", []),
        },
        "audio_onsets": {
            "total": onsets.get("total", len(onsets.get("onsets", []))),
            "strongest_onsets_sample": strongest,
        },
    }


def _judge_agreement(candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    judged = [item for item in candidates if isinstance(item.get("judge_verdict"), dict)
              and item["judge_verdict"].get("verdict") != "unavailable"]
    if not judged:
        return None
    agreed = sum(not item.get("disagreement", False) for item in judged)
    rubber_stamp = (len(judged) >= 3
                    and all(item["judge_verdict"].get("choice") == "real fault" for item in judged)
                    and any(item.get("rule_verdict") != "fault" for item in judged))
    return {
        "agreed": agreed, "total": len(judged),
        "rate": round(agreed / len(judged), 4), "rubber_stamp": rubber_stamp,
    }


def check_video(
    video_path: str,
    *,
    cues_path: Optional[str] = None,
    bpm: Optional[float] = None,
    intent_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    judge: str = "none",
    judge_client: Any = jev_decide,
) -> Dict[str, Any]:
    """Measure one video, judge candidate problems, and write JSON/Markdown evidence."""
    video = Path(video_path).expanduser().resolve()
    output = Path(out_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cues = _load_cues(cues_path)
    intent = _read_intent(intent_path)
    intent_ranges = [
        {"start_sec": section["start_sec"], "end_sec": section["end_sec"],
         "reason": "author's notes: " + section["sentence"][:180]}
        for section in _parse_intent_sections(intent)
    ]
    cache = MediaCache()
    meta = probe_video(str(video), cache=cache)

    if meta.has_audio:
        sync = sync_check(
            str(video), cues=cues, tolerance_ms=_load_cue_tolerance(cues_path),
            cache=cache, bpm=bpm,
        )
        loudness = loudness_timeline(str(video), bpm=bpm, window_sec=None if bpm else 0.25, cache=cache,
                                     ignore_ranges=intent_ranges or None)
        onsets = audio_onsets(str(video), cache=cache)
        balance = stereo_balance_shifts(str(video), bpm=bpm, ignore_ranges=intent_ranges or None)
        clipping = clipping_runs(str(video), ignore_ranges=intent_ranges or None)
    else:
        from src.media.scene import detect_scenes
        cuts = detect_scenes(str(video), cache=cache)
        sync = {
            "cuts": [dict(timestamp_sec=cut.timestamp_sec, score=cut.score, timecode=cut.timecode,
                          nearest_onset=None, offset_ms=None) for cut in cuts],
            "cues": [dict(time_sec=_cue_time(cue), label=cue.get("label", "cue"), status="missing",
                           matched_onset_time_sec=None, offset_ms=None) for cue in cues],
            "tolerance_ms": 80.0,
        }
        loudness = {"windows": [], "outliers": [], "silence_gaps": []}
        onsets = {"onsets": [], "total": 0}
        balance = {"shifts": []}
        clipping = {"runs": []}

    try:
        visual = text_cue_timing(str(video), cues, bpm=bpm, tolerance_ms=_load_cue_tolerance(cues_path))
    except Exception as exc:
        visual = {"text_cues": [], "error": scrub_jev(exc)[:300]}
    try:
        visual["overflows"] = text_overflow(str(video), visual.get("text_cues", []))["overflows"]
    except Exception as exc:
        visual["overflow_error"] = scrub_jev(exc)[:300]
    try:
        visual["flashes"] = blank_flashes(str(video))["flashes"]
    except Exception as exc:
        visual["flash_error"] = scrub_jev(exc)[:300]
    try:
        visual["frame_splices"] = frame_splices(str(video))["splices"]
    except Exception as exc:
        visual["frame_splice_error"] = scrub_jev(exc)[:300]
    try:
        visual["frozen_pictures"] = frozen_pictures(str(video))["frozen_pictures"]
    except Exception as exc:
        visual["frozen_picture_error"] = scrub_jev(exc)[:300]

    candidates = _build_candidates(sync, loudness, onsets, cues, intent=intent, balance=balance, visual=visual,
                                   clipping=clipping)
    _copy_evidence(candidates, video, meta.duration_sec, meta.has_audio, output / "evidence", cache)
    judge_notice = _judge_candidates(candidates, intent, judge=judge, client=judge_client)
    agreement = _judge_agreement(candidates)
    if agreement and agreement["rubber_stamp"]:
        judge_notice = "Jev chose 'real fault' for every candidate, including rule-based review items."
    report = {
        "video": str(video),
        "duration_sec": meta.duration_sec,
        "measurements": _measurement_summary(sync, loudness, onsets),
        "judge": judge,
        "judge_notice": judge_notice,
        "judge_agreement": agreement,
        "candidates": candidates,
    }
    _archive_existing(output / "report.json")
    _archive_existing(output / "report.md")
    report_json = scrub_jev(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report_md = scrub_jev(_markdown_report(report))
    (output / "report.json").write_text(report_json, encoding="utf-8")
    (output / "report.md").write_text(report_md, encoding="utf-8")
    return report


def _format_problem_list(problems: Sequence[Mapping[str, Any]]) -> str:
    if not problems:
        return "- No confirmed problems."
    lines = []
    for problem in problems:
        evidence = problem.get("evidence", {})
        paths = [str((Path(problem.get("_out_dir", ".")) / value).resolve())
                 for value in evidence.values() if isinstance(value, str) and value.endswith((".png", ".jpg", ".jpeg"))]
        lines.append(
            "- `{}` {} at {:.3f}–{:.3f}s; numbers={}; evidence={}".format(
                problem.get("id"), problem.get("kind"), float(problem.get("start_sec", 0)),
                float(problem.get("end_sec", problem.get("start_sec", 0))),
                json.dumps(problem.get("numbers", {}), sort_keys=True), ", ".join(paths) or "none",
            )
        )
        if problem.get("kind") == "intent_section_changed":
            lines.append("  An earlier fix changed this deliberate section. Undo that change so it again "
                         "matches the notes: {}".format(problem.get("intent_note", "")))
    return "\n".join(lines)


def _format_review_list(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "- No review items."
    lines = []
    for item in items:
        evidence = item.get("evidence", {})
        base = Path(item.get("_out_dir", "."))
        paths = [
            "{}: {}".format(key, (base / str(evidence[key])).resolve())
            for key in ("contact_sheet", "spectrogram")
            if isinstance(evidence.get(key), str) and evidence[key]
        ]
        start = float(item.get("start_sec", 0.0))
        end = float(item.get("end_sec", start))
        lines.append(
            "- `{}` {} at {:.3f}–{:.3f}s; reason={}; numbers={}; evidence={}".format(
                item.get("id"), item.get("kind", "review item"), start, end,
                item.get("ambiguous_reason") or item.get("classification") or "measurement is inconclusive",
                json.dumps(item.get("numbers", {}), ensure_ascii=False, sort_keys=True),
                ", ".join(paths) or "none available",
            )
        )
    return "\n".join(lines)


def _review_key(item: Mapping[str, Any]) -> str:
    """Stable key for suppressing an unchanged review the fixer dismissed."""
    return json.dumps(
        {"id": item.get("id"), "kind": item.get("kind"), "numbers": item.get("numbers", {})},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )


_REVIEW_VERDICT_LINE = re.compile(
    r"^\s*REVIEW\s+(.+?)\s*:\s*(not\s+a\s+fault|fault)\s*[—–−-]\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _parse_review_verdicts(reply: str, items: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Read the fixer's one-line verdicts, accepting common dash and spacing variants."""
    allowed = {str(item.get("id")) for item in items}
    parsed: Dict[str, Dict[str, str]] = {}
    for line in reply.splitlines():
        match = _REVIEW_VERDICT_LINE.match(line)
        if not match:
            continue
        item_id = match.group(1).strip().strip("`* ")
        if item_id not in allowed:
            continue
        verdict = "not a fault" if re.sub(r"\s+", " ", match.group(2).casefold()) == "not a fault" else "fault"
        parsed[item_id] = {"id": item_id, "verdict": verdict, "reason": match.group(3).strip()}
    return [parsed[str(item.get("id"))] for item in items if str(item.get("id")) in parsed]


def _fix_brief(
    problems: Sequence[Mapping[str, Any]],
    project_dir: Path,
    build_error: Optional[str] = None,
    intent: str = "",
    review_items: Sequence[Mapping[str, Any]] = (),
) -> str:
    if build_error is not None:
        confirmed_text = "Build failed:\n{}".format(build_error)
    else:
        actionable = [problem for problem in problems
                      if problem.get("confirmed", True) and problem.get("rule_verdict", "fault") == "fault"
                      and problem.get("classification") != "informational"]
        confirmed_text = _format_problem_list(actionable)
    reviews = [item for item in review_items
               if not item.get("confirmed") and item.get("rule_verdict") == "ambiguous"]
    review_section = "\nReview items:\n{}\n".format(_format_review_list(reviews)) if reviews else ""
    review_rules = (
        "- Check each review item yourself (open the frames/evidence, measure with ffmpeg if needed). Change the source only "
        "if you confirm a real, unintended fault; otherwise leave it and say why.\n"
        "- At the very end of your reply, include one line per review item in this format: "
        "REVIEW <id>: fault|not a fault — <short reason>.\n"
    ) if reviews else ""
    sections = _parse_intent_sections(intent)
    protected = "\n".join(
        "- {:.3f}–{:.3f} s: {}".format(section["start_sec"], section["end_sec"], section["sentence"])
        for section in sections
    ) or "- None named with times; still leave every sound the notes describe as it is."
    notes = intent.strip() or "No author notes were provided."
    return (
        "Fix the source project so the generated video passes its media check.\n\n"
        "Confirmed problems:\n{}\n\n"
        "{}\n"
        "Author's notes (deliberate choices; they are not problems):\n{}\n\n"
        "Do not touch these deliberate sections (their sounds, levels, filters and timing):\n{}\n\n"
        "Rules:\n"
        "- Fix only the confirmed problems listed above, with the smallest source change that fixes each one.\n"
        "{}"
        "- Do not change any other sound's design, level, filter or timing, even if it seems related.\n"
        "- Do not redesign or re-level a sound just so the checker detects it. If the source already "
        "places a listed sound correctly, the report is a false alarm: leave that part unchanged and "
        "say so in your report.\n"
        "- Do not edit cues or intent files.\n"
        "- Report every source change you made and why.\n"
    ).format(confirmed_text, review_section, notes, protected, review_rules)


def _command_result(command: Any, cwd: Path, *, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    if isinstance(command, list):
        args: Any = [str(part) for part in command]
        shell = False
    else:
        args = str(command)
        shell = True
    child_env = os.environ.copy()
    child_env.pop("TYPESAFE_API_KEY", None)
    child_env.pop("OPENROUTER_API_KEY", None)
    try:
        return subprocess.run(args, cwd=str(cwd), shell=shell, input=input_text, capture_output=True, text=True, check=False, env=child_env)
    except OSError as exc:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))


def _run_fixer(
    template: Any,
    rung: Mapping[str, str],
    project_dir: Path,
    last_message_file: Path,
    brief: str,
) -> Tuple[int, str]:
    values = {
        "model": rung["model"],
        "effort": rung["effort"],
        "project_dir": shlex.quote(str(project_dir)),
        "last_message_file": shlex.quote(str(last_message_file)),
    }
    if isinstance(template, list):
        list_values = dict(values, project_dir=str(project_dir), last_message_file=str(last_message_file))
        argv = [str(part).format(**list_values) for part in template]
    else:
        argv = shlex.split(str(template).format(**values))
    try:
        child_env = os.environ.copy()
        child_env.pop("TYPESAFE_API_KEY", None)
        child_env.pop("OPENROUTER_API_KEY", None)
        result = subprocess.run(argv, cwd=str(project_dir), input=scrub_jev(brief), capture_output=True, text=True, check=False, env=child_env)
        output = (result.stdout or "") + (result.stderr or "")
        # Fixers that print their answer (rather than writing the file) still
        # leave a last message behind.
        if result.stdout and last_message_file.exists() and not last_message_file.read_text(encoding="utf-8").strip():
            last_message_file.write_text(scrub_jev(result.stdout), encoding="utf-8")
        return int(result.returncode), output[-12000:]
    except OSError as exc:
        return 127, str(exc)


_USAGE_LIMIT_PATTERNS = re.compile(
    r"(?i)usage limit|(?:5-hour|five-hour|weekly|daily|session) limit|limit reached|rate[- ]limit"
    r"|quota|too many requests|out of (?:credits|usage)|overloaded_error"
)


def _usage_limit_message(exit_code: int, output: str, last_message: str = "") -> Optional[str]:
    """The fixer's usage-limit line, if the run stopped because of a limit.

    A clean run with a long report that merely mentions a limit is not one:
    the match must come with a failed exit or be nearly all the fixer said.
    """
    for text in (output, last_message):
        match = _USAGE_LIMIT_PATTERNS.search(text or "")
        if match is None or not (exit_code != 0 or len(text.strip()) <= 400):
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        return text[line_start:line_end if line_end >= 0 else len(text)].strip()[:300]
    return None


def _section_levels(video: Path, sections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Band levels (dB) in 0.25 s windows across each intent section of a built video."""
    if not sections or not video.exists():
        return []
    cache = MediaCache()
    if not probe_video(str(video), cache=cache).has_audio:
        return []
    levels = []
    for section in sections:
        timeline = loudness_timeline(str(video), window_sec=0.25, start_sec=section["start_sec"],
                                     end_sec=section["end_sec"], cache=cache)
        windows = [{"start_sec": window["start_sec"], "end_sec": window["end_sec"], "band_db": window["band_db"]}
                   for window in timeline.get("windows", [])]
        levels.append({"start_sec": section["start_sec"], "end_sec": section["end_sec"],
                       "sentence": section["sentence"], "windows": windows})
    return levels


def _mean_band_db(windows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    bands: Dict[str, float] = {}
    for band in (windows[0]["band_db"] if windows else {}):
        power = [10.0 ** (float(window["band_db"][band]) / 10.0) for window in windows]
        bands[band] = round(10.0 * math.log10(max(sum(power) / len(power), 1e-12)), 2)
    return bands


def _fix_times(problems: Sequence[Mapping[str, Any]]) -> List[float]:
    """Moments a fixer was asked to change: each problem's span and its matched sound."""
    times: List[float] = []
    for problem in problems:
        if problem.get("kind") == "intent_section_changed":
            continue
        start = float(problem.get("start_sec", 0.0))
        times += [start, float(problem.get("end_sec", start))]
        numbers = problem.get("numbers") or {}
        if numbers.get("matched_onset_time_sec") is not None:
            times.append(float(numbers["matched_onset_time_sec"]))
        if numbers.get("offset_ms") is not None:
            times.append(start + float(numbers["offset_ms"]) / 1000.0)
    return times


def _protected_changes(
    original: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]], tolerance_db: float,
    fix_times: Sequence[float] = (), pad_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """A problem for each author-described section the fixer audibly changed.

    Windows within `pad_sec` of a moment the fixer was asked to fix are left
    out: moving a misplaced kick out of a section is the fix, not damage.
    """
    problems = []
    relative_band_floor_db = 24.0
    for before, after in zip(original, current):
        pairs = list(zip(
            before.get("windows") or [{"start_sec": before["start_sec"], "end_sec": before["end_sec"], "band_db": before["band_db"]}],
            after.get("windows") or [{"start_sec": after["start_sec"], "end_sec": after["end_sec"], "band_db": after["band_db"]}],
        ))
        kept = [(old, new) for old, new in pairs
                if not any(float(old["start_sec"]) - pad_sec <= t <= float(old["end_sec"]) + pad_sec for t in fix_times)]
        if not kept:
            continue
        before_db = _mean_band_db([old for old, _ in kept])
        after_db = _mean_band_db([new for _, new in kept])
        section_peak_db = max((*before_db.values(), *after_db.values()), default=-120.0)
        changed = {
            band: {"before_db": level, "after_db": after_db[band],
                   "change_db": round(after_db[band] - level, 2)}
            for band, level in before_db.items()
            if band in after_db and max(level, after_db[band]) > -60.0
            and max(level, after_db[band]) >= section_peak_db - relative_band_floor_db
            and abs(after_db[band] - level) >= tolerance_db
        }
        if not changed:
            continue
        start, end = float(before["start_sec"]), float(before["end_sec"])
        problems.append({
            "id": _problem_id("intent_section_changed", start),
            "kind": "intent_section_changed",
            "start_sec": start,
            "end_sec": end,
            "numbers": {"bands": changed, "tolerance_db": tolerance_db,
                        "windows_compared": len(kept), "windows_skipped_near_fixes": len(pairs) - len(kept)},
            "classification": "a deliberate section changed during fixing; restore it",
            "intent_note": before["sentence"],
            "rule_verdict": "fault",
            "confirmed": True,
            "nearby_cues": [],
            "evidence": {},
        })
    return problems


def _problem_keys(problems: Sequence[Mapping[str, Any]]) -> set:
    return {str(problem.get("id")) for problem in problems}


def _same_problem_remains(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> bool:
    for old in before:
        old_start = float(old.get("start_sec", 0.0))
        old_end = float(old.get("end_sec", old_start))
        for new in after:
            old_kind = str(old.get("kind", ""))
            new_kind = str(new.get("kind", ""))
            old_family = "cue" if old_kind.startswith("cue_") else old_kind
            new_family = "cue" if new_kind.startswith("cue_") else new_kind
            if old_family != new_family:
                continue
            new_start = float(new.get("start_sec", 0.0))
            new_end = float(new.get("end_sec", new_start))
            separation = max(0.0, new_start - old_end, old_start - new_end)
            if separation > 0.5:
                continue
            if old_family == "cue":
                old_labels = {cue.get("label") for cue in old.get("nearby_cues", [])}
                new_labels = {cue.get("label") for cue in new.get("nearby_cues", [])}
                if old_labels and new_labels and not old_labels.intersection(new_labels):
                    continue
            return True
    return False


def _config_path(config: Mapping[str, Any], project_dir: Path, key: str) -> Optional[str]:
    value = config.get(key)
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (project_dir / path).resolve())


def _run_build_check(
    config: Mapping[str, Any], project_dir: Path, output_root: Path, round_no: int, check_function: Any,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]], Optional[Path]]:
    build = _command_result(config["build_cmd"], project_dir)
    if build.returncode != 0:
        error = ((build.stdout or "") + (build.stderr or "")).strip()[-12000:]
        return scrub_jev("build exited with code {}:\n{}".format(build.returncode, error)), [], [], None
    video = Path(config["video"])
    if not video.is_absolute():
        video = project_dir / video
    out_dir = output_root / "round-{:02d}".format(round_no)
    _archive_existing(out_dir)
    report = check_function(
        str(video), cues_path=_config_path(config, project_dir, "cues"),
        bpm=config.get("bpm"),
        intent_path=_config_path(config, project_dir, "intent"),
        out_dir=str(out_dir), judge=config.get("judge", "none"),
    )
    report_path = out_dir / "report.json"
    problems = [dict(problem, _out_dir=str(out_dir)) for problem in report.get("candidates", [])
                if problem.get("confirmed") and problem.get("rule_verdict", "fault") == "fault"
                and problem.get("classification") != "informational"]
    review_items = [dict(item, _out_dir=str(out_dir)) for item in report.get("candidates", [])
                    if not item.get("confirmed") and item.get("rule_verdict") == "ambiguous"]
    return None, problems, review_items, report_path


def run_loop(
    config_path: str, *, check_function: Any = check_video, level_function: Any = _section_levels,
) -> Dict[str, Any]:
    """Run the build/check/fix/re-check loop and persist round logs and summary."""
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("loop config must be a JSON object")
    project_dir = Path(config["project_dir"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = (config_file.parent / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()
    output_root = Path(config.get("out", DEFAULT_OUTPUT_DIR))
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    max_rounds = int(config.get("max_rounds", 6))
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    ladder = config.get("ladder", DEFAULT_LADDER)
    if not ladder or any("model" not in rung or "effort" not in rung for rung in ladder):
        raise ValueError("ladder must be a non-empty array of {model, effort} objects")
    fixer_cmd = config.get("fixer_cmd", DEFAULT_FIXER_CMD)
    last_message_file = Path(config.get("last_message_file", output_root / "last-message.txt"))
    if not last_message_file.is_absolute():
        last_message_file = project_dir / last_message_file
    last_message_file.parent.mkdir(parents=True, exist_ok=True)
    intent_path = _config_path(config, project_dir, "intent")
    intent = _read_intent(intent_path)
    protected_sections = _parse_intent_sections(intent) if config.get("protect_intent_sections", True) else []
    protect_tolerance_db = float(config.get("protect_tolerance_db", 3.0))
    video_path = Path(config["video"]).expanduser()
    if not video_path.is_absolute():
        video_path = project_dir / video_path
    original_levels: Optional[List[Dict[str, Any]]] = None
    fix_times: List[float] = []
    usage_limit: Optional[str] = None

    def build_and_check(
        round_no: int,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]], Optional[Path]]:
        """Build and check, then flag deliberate sections that no longer sound as they first did."""
        nonlocal original_levels
        error, problems, reviews, report = _run_build_check(config, project_dir, output_root, round_no, check_function)
        if error is not None or not protected_sections:
            return error, problems, reviews, report
        levels = level_function(video_path, protected_sections)
        if original_levels is None:
            original_levels = levels
        else:
            out_dir = str(report.parent) if report else str(output_root)
            problems = problems + [dict(problem, _out_dir=out_dir) for problem in
                                   _protected_changes(original_levels, levels, protect_tolerance_db, fix_times)]
        return error, problems, reviews, report

    log_path = output_root / "loop_log.jsonl"
    _archive_existing(log_path)
    log_path.write_text("", encoding="utf-8")
    entries: List[Dict[str, Any]] = []
    rungs_used: List[str] = []
    fixed_items: Dict[str, Dict[str, Any]] = {}
    dismissed_review_keys: set = set()
    all_review_verdicts: List[Dict[str, Any]] = []
    rung_index = 0
    no_progress_uses = 0
    status = "running"
    final_problems: List[Dict[str, Any]] = []
    current_error: Optional[str] = None
    current_problems: List[Dict[str, Any]] = []
    current_reviews: List[Dict[str, Any]] = []
    current_report: Optional[Path] = None
    state_ready = False
    started = time.monotonic()

    for round_no in range(1, max_rounds + 1):
        round_start = time.monotonic()
        round_started_at = _now()
        if not state_ready:
            current_error, current_problems, current_reviews, current_report = build_and_check(round_no)
        state_ready = False
        before = current_problems
        review_before = [item for item in current_reviews if _review_key(item) not in dismissed_review_keys]
        before_error = current_error
        before_report = current_report
        if current_error is None and not before and not review_before:
            status = "success"
            final_problems = []
            entry = {
                "round": round_no, "rung": None, "rung_index": None,
                "problems_before": [], "problems_after": [], "fixer_exit_code": None,
                "build_error": None, "started_at": round_started_at, "duration_sec": round(time.monotonic() - round_start, 3),
                "check_report": str(current_report) if current_report else None,
                "review_items_before": [], "review_verdicts": [],
            }
            entries.append(entry)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(scrub_jev(json.dumps(entry, ensure_ascii=False)) + "\n")
            break

        rung = ladder[rung_index]
        used_rung_index = rung_index
        rung_label = "{} {}".format(rung["model"], rung["effort"])
        brief = _fix_brief(before, project_dir, before_error, intent=intent, review_items=review_before)
        fix_times.extend(_fix_times(before))
        _archive_existing(last_message_file)
        last_message_file.write_text("", encoding="utf-8")
        fixer_code, fixer_output = _run_fixer(fixer_cmd, rung, project_dir, last_message_file, brief)
        usage_limit = _usage_limit_message(
            fixer_code, fixer_output,
            last_message_file.read_text(encoding="utf-8") if last_message_file.exists() else "",
        )
        fixer_reply = (last_message_file.read_text(encoding="utf-8") if last_message_file.exists() else "")
        review_verdicts = _parse_review_verdicts(fixer_reply + "\n" + fixer_output, review_before)
        all_review_verdicts.extend(dict(verdict, round=round_no) for verdict in review_verdicts)
        if usage_limit is not None:
            # Out of usage is not a failed attempt: stop without climbing the
            # ladder or counting the round, so a later re-run starts fresh.
            status = "retry later"
            final_problems = before
            entry = {
                "round": round_no, "rung": rung_label, "rung_index": used_rung_index,
                "fixer_status": "usage_limit", "usage_limit_message": usage_limit, "counted": False,
                "problems_before": before if before else ([{"kind": "build_failure", "message": before_error}] if before_error else []),
                "review_items_before": review_before,
                "review_verdicts": review_verdicts,
                "problems_after": None, "build_error": before_error, "fixer_exit_code": fixer_code,
                "fixer_output": scrub_jev(fixer_output)[-4000:], "started_at": round_started_at,
                "duration_sec": round(time.monotonic() - round_start, 3),
                "check_report": str(before_report) if before_report else None, "next_rung_index": rung_index,
            }
            entries.append(entry)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(scrub_jev(json.dumps(entry, ensure_ascii=False)) + "\n")
            break
        items_by_id = {str(item.get("id")): item for item in review_before}
        for verdict in review_verdicts:
            if verdict["verdict"] == "not a fault" and verdict["id"] in items_by_id:
                dismissed_review_keys.add(_review_key(items_by_id[verdict["id"]]))
        rungs_used.append(rung_label)

        next_error, after, reviews_after, next_report = build_and_check(round_no + 1)
        same_problem_remains = _same_problem_remains(before, after) or (before_error is not None and next_error is not None)
        for problem in before:
            if next_error is None and not _same_problem_remains([problem], after):
                problem_id = str(problem.get("id"))
                fixed_items[problem_id] = dict(problem)
        if next_error is not None:
            current_error = next_error
            current_problems = []
            current_reviews = []
            final_problems = []
        else:
            current_error = None
            current_problems = after
            current_reviews = reviews_after
            final_problems = after
        current_report = next_report
        state_ready = True

        if same_problem_remains:
            no_progress_uses += 1
            if no_progress_uses >= 2:
                if rung_index + 1 < len(ladder):
                    rung_index += 1
                    no_progress_uses = 0
                else:
                    status = "needs human review"
        else:
            no_progress_uses = 0

        entry = {
            "round": round_no,
            "rung": rung_label,
            "rung_index": used_rung_index,
            "fixer_status": "ran",
            "problems_before": before if before else ([{"kind": "build_failure", "message": before_error}] if before_error else []),
            "review_items_before": review_before,
            "review_verdicts": review_verdicts,
            "problems_after": after if next_error is None else [{"kind": "build_failure", "message": next_error}],
            "review_items_after": reviews_after if next_error is None else [],
            "build_error": before_error,
            "fixer_exit_code": fixer_code,
            "fixer_output": scrub_jev(fixer_output)[-4000:],
            "started_at": round_started_at,
            "duration_sec": round(time.monotonic() - round_start, 3),
            "check_report": str(before_report) if before_report else None,
            "after_check_report": str(next_report) if next_report else None,
            "next_rung_index": rung_index,
        }
        entries.append(entry)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(scrub_jev(json.dumps(entry, ensure_ascii=False)) + "\n")

        if status == "needs human review":
            break
        if not after and next_error is None:
            status = "success"
            final_problems = []
            # This iteration already performed the clean verification; it is the final check.
            break
        if round_no == max_rounds:
            status = "needs human review"
            final_problems = after

    elapsed = round(time.monotonic() - started, 3)
    remaining = final_problems
    completed_rounds = len([entry for entry in entries
                            if entry.get("fixer_exit_code") is not None and entry.get("counted", True)])
    verified_fixed_items = fixed_items if completed_rounds else {}
    summary = {
        "status": status,
        "rounds": completed_rounds,
        "rungs_used": rungs_used,
        "fixed_ids": sorted(verified_fixed_items),
        "fixed": [verified_fixed_items[key] for key in sorted(verified_fixed_items)],
        "remaining": remaining,
        "review_verdicts": all_review_verdicts,
        "elapsed_sec": elapsed,
        "usage_limit_message": usage_limit,
    }
    summary_lines = [
        "# Media check loop summary", "", "Status: **{}**".format(status), "",
        "Rounds with fixer: {}".format(summary["rounds"]),
        "Elapsed: {:.3f}s".format(elapsed), "",
        "## Rungs used", "",
    ]
    summary_lines.extend(["- {}".format(value) for value in rungs_used] or ["- None (video already passed)"])
    summary_lines.extend(["", "## Fixed problems", ""])
    if summary["fixed"]:
        for problem in summary["fixed"]:
            problem_id = str(problem.get("id"))
            summary_lines.append("- `{}` {} at {:.3f}s".format(
                problem_id, problem.get("kind", "problem"), float(problem.get("start_sec", 0.0)),
            ))
            evidence = problem.get("evidence", {})
            base = Path(problem.get("_out_dir", ""))
            links = [
                "[{}](<{}>)".format(key.replace("_", " "), (base / value).resolve())
                for key, value in evidence.items()
                if isinstance(value, str) and key in {"spectrogram", "contact_sheet"}
            ]
            if links:
                summary_lines.append("  Evidence: {}".format(", ".join(links)))
    else:
        summary_lines.append("- No confirmed problems fixed.")
    summary_lines.extend(["", "## Fixer verdicts on review items", ""])
    if all_review_verdicts:
        for verdict in all_review_verdicts:
            summary_lines.append("- Round {} `{}`: **{}** — {}".format(
                verdict["round"], verdict["id"], verdict["verdict"], verdict["reason"] or "no reason provided",
            ))
    else:
        summary_lines.append("- None reported.")
    summary_lines.extend(["", "## Remaining problems", ""])
    if current_error:
        summary_lines.append("- Build failed: {}".format(current_error))
    elif remaining:
        for problem in remaining:
            report_link = problem.get("_out_dir", "")
            evidence = problem.get("evidence", {})
            paths = [
                "[{}](<{}>)".format(key.replace("_", " "), (Path(report_link) / value).resolve())
                for key, value in evidence.items()
                if isinstance(value, str) and key in {"spectrogram", "contact_sheet"}
            ]
            summary_lines.append("- `{}` {} at {:.3f}s; evidence: {}".format(
                problem.get("id"), problem.get("kind"), float(problem.get("start_sec", 0)),
                ", ".join(paths) or "see round report",
            ))
    else:
        summary_lines.append("- None. The final check passed.")
    if status == "retry later":
        summary_lines.extend([
            "", "Retry later: the fixer stopped at a usage limit ({}). This attempt was not counted "
            "and the ladder did not move. Re-run the loop once the limit resets; its first check "
            "will pick up any partial edits.".format(usage_limit),
        ])
    elif status != "success":
        summary_lines.extend(["", "Needs human review (Opus): the configured ladder did not clear all problems."])
    _archive_existing(output_root / "summary.md")
    (output_root / "summary.md").write_text(scrub_jev("\n".join(summary_lines) + "\n"), encoding="utf-8")
    summary["output_dir"] = str(output_root)
    return summary


def _cmd_check(args: argparse.Namespace) -> int:
    report = check_video(
        args.video, cues_path=args.cues, bpm=args.bpm, intent_path=args.intent,
        out_dir=args.out, judge=args.judge,
    )
    confirmed = [item for item in report["candidates"] if item["confirmed"]]
    print("Wrote {} ({} confirmed problem{}).".format(
        scrub_jev(str(Path(args.out).expanduser().resolve() / "report.md")),
        len(confirmed), "" if len(confirmed) == 1 else "s",
    ))
    return 1 if confirmed else 0


def _cmd_cadence(args: argparse.Namespace) -> int:
    print(json.dumps(frame_cadence(args.video), indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    summary = run_loop(args.config)
    print("{} after {} fixer round(s). Summary: {}/summary.md".format(
        summary["status"], summary["rounds"], scrub_jev(summary["output_dir"]),
    ))
    if summary["status"] == "retry later":
        return EXIT_RETRY_LATER
    return 0 if summary["status"] == "success" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.loop", description="Check and iteratively fix video source projects")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="measure a video and write a problem report")
    check.add_argument("video")
    check.add_argument("--cues")
    check.add_argument("--bpm", type=float)
    check.add_argument("--intent")
    check.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    check.add_argument("--judge", choices=("jev", "none"), default="none")
    check.set_defaults(func=_cmd_check)
    cadence = commands.add_parser("cadence", help="print time ranges where the unique-frame rate is below the file's frame rate (same result as the check_frame_cadence tool)")
    cadence.add_argument("video")
    cadence.set_defaults(func=_cmd_cadence)
    run = commands.add_parser("run", help="run the build/check/fix/re-check loop")
    run.add_argument("--config", required=True)
    run.set_defaults(func=_cmd_run)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print("error: {}".format(scrub_jev(exc)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
