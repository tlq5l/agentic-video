#!/usr/bin/env python3
"""Generate the deterministic, textured synthetic split for bench-v1."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import unicodedata
import wave
from pathlib import Path
from typing import Any

try:
    from .common import data_root, key_path, merge_by_id, question_path, read_jsonl, write_jsonl_atomic, scratch_dir
except ImportError:  # direct script execution
    from common import data_root, key_path, merge_by_id, question_path, read_jsonl, write_jsonl_atomic, scratch_dir


FPS = 24
MOMENT_SECONDS = 78
COUNT_SECONDS = 88
SEARCH_SECONDS = 600
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Apple Symbols.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

SHAPES = [
    ("yellow outlined circle", "○", "yellow"),
    ("cyan triangle", "▲", "cyan"),
    ("orange diamond", "◆", "orange"),
    ("magenta star", "★", "magenta"),
    ("white plus sign", "✚", "white"),
    ("blue square", "■", "blue"),
    ("green outlined diamond", "◇", "lime"),
    ("red arrowhead", "▶", "red"),
    ("purple crescent", "☾", "purple"),
]


def _require(command: str) -> str:
    found = shutil.which(command)
    if not found:
        raise RuntimeError(f"required executable not found on PATH: {command}")
    return found


def _font() -> str | None:
    return str(next((p for p in FONT_CANDIDATES if p.exists()), None)) if any(p.exists() for p in FONT_CANDIDATES) else None


def _ffmpeg_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _quantize(value: float, fps: int = FPS) -> float:
    return round(round(value * fps) / fps, 6)


def _run(command: list[str], *, quiet: bool = True) -> None:
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if process.returncode:
        detail = (process.stderr or process.stdout)[-1600:]
        raise RuntimeError(f"command failed ({process.returncode}): {command[0]} {detail}")


def _draw_filter(symbol: str, color: str, start: float, end: float, index: int, *, size: int = 72) -> str:
    font = _font()
    parts = []
    if font:
        parts.append(f"fontfile='{_ffmpeg_filter_path(font)}'")
    parts.extend([
        f"text='{symbol}'", f"fontcolor={color}", f"fontsize={size}",
        "borderw=3", "bordercolor=black@0.9",
        (f"x='40+560*(t-{start:.6f})/{max(0.1, end - start):.6f}'" if index >= 100 else f"x='62+{index % 4 * 136}+25*sin(t*1.7+{index * 0.61:.3f})'"),
        f"y='88+{index % 3 * 88}+17*cos(t*1.19+{index * 0.43:.3f})'",
        f"enable='gte(t,{start:.6f})*lt(t,{end:.6f})'",
    ])
    return "drawtext=" + ":".join(parts)


def _draw_text_filter(text: str, color: str, start: float, end: float, *, size: int = 48, y: int = 185) -> str:
    font = _font()
    safe_text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")
    parts = []
    if font:
        parts.append(f"fontfile='{_ffmpeg_filter_path(font)}'")
    parts.extend([
        f"text='{safe_text}'", f"fontcolor={color}", f"fontsize={size}",
        "borderw=3", "bordercolor=black@0.95", "box=1", "boxcolor=black@0.76", "boxborderw=15",
        "x='(w-text_w)/2'", f"y={y}",
        f"enable='gte(t,{start:.6f})*lt(t,{end:.6f})'",
    ])
    return "drawtext=" + ":".join(parts)


def _render_video(
    path: Path,
    duration: float,
    events: list[dict[str, Any]],
    seed: int,
    *,
    long_form: bool = False,
    text_events: list[dict[str, Any]] | None = None,
    scene_cuts: list[tuple[float, float, str]] | None = None,
    motion_blur: bool = False,
) -> None:
    ffmpeg = _require("ffmpeg")
    fps = 8 if long_form else FPS
    width, height = (700, 400) if long_form else (720, 420)
    filters = [
        f"noise=alls=7:allf=t+u:all_seed={seed}",
        "drawgrid=width=60:height=52:thickness=1:color=white@0.12",
    ]
    for start, end, color in scene_cuts or []:
        filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color={color}@0.78:t=fill:enable='gte(t,{start:.6f})*lt(t,{end:.6f})'")
    # Independently moving visual decoys make the target a semantic recognition task,
    # while the crop makes the entire scene drift as if from handheld camera motion.
    for i in range(11):
        start = (i * 7.91 + seed % 29) % max(1.0, duration - 2.0)
        end = min(duration, start + 0.27 + (i % 3) * 0.11)
        symbol, color = [("○", "deepskyblue"), ("◆", "yellow"), ("▲", "lightgreen"), ("★", "coral"), ("■", "violet")][i % 5]
        filters.append(_draw_filter(symbol, color, start, end, i + 19, size=40 + (i % 2) * 6))
    for event in events:
        filters.append(_draw_filter(event["symbol"], event["color"], event["start"], event["end"], event["index"], size=72 if not long_form else 66))
    for event in text_events or []:
        filters.append(_draw_text_filter(event["text"], event.get("color", "white"), event["start"], event["end"], size=event.get("size", 48), y=event.get("y", 185)))
    if motion_blur:
        filters.append("tmix=frames=3:weights='1 2 1'")
    filters.extend([
        f"crop=640:360:x='40+12*sin(t*0.16)':y='20+9*cos(t*0.11)'",
        "format=yuv420p",
    ])
    filter_graph = ",".join(filters)
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        f"life=s={width}x{height}:rate={fps}:ratio=0.28:rule=B36/S23:seed={seed}:life_color=0x77b7c5:death_color=0x192b3a:mold_color=0x354b5c:mold=3",
        "-t", str(duration),
        "-vf", filter_graph, "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "31" if long_form else "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ])


def _write_wave(path: Path, sample_rate: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(frames)


def _speech_bounds(wav_path: Path) -> tuple[float, float]:
    block_ms = 10
    with wave.open(str(wav_path), "rb") as audio:
        rate, width, channels = audio.getframerate(), audio.getsampwidth(), audio.getnchannels()
        chunk_frames = max(1, round(rate * block_ms / 1000))
        energies: list[float] = []
        while raw := audio.readframes(chunk_frames):
            count = len(raw) // width
            if width == 2:
                values = memoryview(raw).cast("h")
                energy = math.sqrt(sum(float(v) * float(v) for v in values) / max(1, count))
            else:
                energy = 0.0
            energies.append(energy)
    active = [i for i, energy in enumerate(energies) if energy > 80]
    if not active:
        raise RuntimeError("generated speech track was silent")
    return active[0] * block_ms / 1000, min(len(energies), active[-1] + 1) * block_ms / 1000


def _make_speech(stage: Path, phrase: str, index: int) -> tuple[Path, float, float]:
    say = _require("say")
    ffmpeg = _require("ffmpeg")
    source = stage / f"speech-{index}.aiff"
    wav_path = stage / f"speech-{index}.wav"
    voice = "Samantha"
    _run([say, "-v", voice, "-r", "175", "-o", str(source), phrase])
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(wav_path)])
    return wav_path, *_speech_bounds(wav_path)


def _mix_long_audio(video_path: Path, out_path: Path, stage: Path, search_cues: list[dict[str, Any]], duration: int, seed: int) -> list[dict[str, Any]]:
    ffmpeg = _require("ffmpeg")
    sound_items = [cue for cue in search_cues if cue.get("cue_type") == "sound_cued"]
    speech_items = [cue for cue in search_cues if cue.get("cue_type") == "speech_cued"]
    speech_paths: dict[str, tuple[Path, float, float]] = {}
    for index, cue in enumerate(speech_items):
        wav, first, last = _make_speech(stage, cue["phrase"], index)
        speech_paths[cue["id"]] = (wav, first, last)
        cue["truth"] = [_quantize(cue["play_at"] + first, 100), _quantize(cue["play_at"] + last, 100)]

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.002:sample_rate=44100:duration={duration}:seed={seed}",
    ]
    audio_items: list[tuple[str, float]] = []
    for cue in sound_items:
        cue_start = cue["play_at"]
        cue_end = cue_start
        for tone_index, frequency in enumerate(cue.get("frequencies", (680, 960, 1380))):
            note_start = cue_start + tone_index * 0.19
            note_duration = float(cue.get("note_duration", 0.16))
            command.extend([
                "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:sample_rate=44100:duration={note_duration}",
            ])
            audio_items.append(("tone", note_start))
            cue_end = note_start + note_duration
        cue["truth"] = [_quantize(cue_start, 100), _quantize(cue_end, 100)]
    for cue in speech_items:
        wav, first, last = speech_paths[cue["id"]]
        command.extend(["-i", str(wav)])
        audio_items.append(("speech", cue["play_at"]))
    # A low, unrelated audio transient acts as a decoy for onset-only strategies.
    for index, note_start in enumerate((53.5, 151.25, 278.75, 401.5, 536.0)):
        command.extend(["-f", "lavfi", "-i", f"sine=frequency={440 + index * 37}:sample_rate=44100:duration=0.12"])
        audio_items.append(("decoy", note_start))

    n_inputs = 1 + len(audio_items)
    filters = []
    labels = []
    for index in range(n_inputs):
        label = f"a{index}"
        volume = "0.22" if index == 0 else ("0.9" if audio_items[index - 1][0] != "decoy" else "0.28")
        source = f"[{index}:a]"
        if index:
            delay_ms = round(audio_items[index - 1][1] * 1000)
            source += f"adelay=delays={delay_ms}:all=1,"
        filters.append(f"{source}volume={volume}[{label}]")
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"amix=inputs={n_inputs}:duration=first:normalize=0[out]")
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]", "-t", str(duration),
        "-c:a", "aac", "-b:a", "80k", str(out_path),
    ])
    _run(command)
    return search_cues


def _mux(video_path: Path, audio_path: Path, output: Path) -> None:
    ffmpeg = _require("ffmpeg")
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(output),
    ])


def _make_question(identifier: str, track: str, video: Path, question: str, fmt: str, *, cue_type: str = "visual_only") -> dict[str, Any]:
    return {
        "id": identifier,
        "track": track,
        "split": "synthetic",
        "video_path": str(video),
        "question": question,
        "answer_format": fmt,
        "cue_type": cue_type,
    }


def _make_key(identifier: str, truth: Any, tolerance: dict[str, Any], notes: str, source: dict[str, Any]) -> dict[str, Any]:
    return {"id": identifier, "truth": truth, "tolerance": tolerance, "notes": notes, "source": source}


def _validate_generated_questions(question_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], target_occurrences: dict[str, int]) -> None:
    by_id = {row["id"]: row for row in key_rows}
    normalized_questions: dict[str, str] = {}
    question_ids = {row["id"] for row in question_rows}
    if len(question_ids) != len(question_rows):
        raise RuntimeError("synthetic question IDs are not unique")
    if question_ids != set(target_occurrences) or question_ids != set(by_id):
        raise RuntimeError("synthetic target-occurrence validation is missing a question or answer-key entry")
    for row in question_rows:
        normalized = " ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split())
        if normalized in normalized_questions:
            raise RuntimeError(f"duplicate synthetic question text: {normalized_questions[normalized]} and {row['id']}")
        normalized_questions[normalized] = row["id"]
        expected = target_occurrences[row["id"]]
        if expected < 0:
            raise RuntimeError(f"negative target occurrence count for {row['id']}")
        if row["track"] == "count":
            try:
                truth = int(by_id[row["id"]]["truth"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"count target for {row['id']} has no integer truth") from exc
            if truth != expected:
                raise RuntimeError(f"count target for {row['id']} occurs {expected} times but truth allows {truth}")
        elif expected not in {0, 1}:
            raise RuntimeError(f"non-count target for {row['id']} occurs {expected} times")
        elif expected == 0 and by_id[row["id"]]["truth"] != "not_found":
            raise RuntimeError(f"absent target for {row['id']} is not marked not_found")
        elif expected == 1 and by_id[row["id"]]["truth"] == "not_found":
            raise RuntimeError(f"present target for {row['id']} is marked not_found")


def generate(seed: int = 8448) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = data_root()
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    stage = scratch_dir("bench-v1-synth-stage-")
    question_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    target_occurrences: dict[str, int] = {}
    rng = random.Random(seed)
    source = {"dataset": "bench-v1 deterministic synthetic generator", "license": "generated benchmark media; no third-party source", "seed": seed}

    def add_question(
        identifier: str, track: str, video: Path, question: str, answer_format: str,
        truth: Any, tolerance: dict[str, Any], notes: str, *, cue_type: str, target_occurrences_count: int,
        verification_intervals: list[list[float]] | None = None,
    ) -> None:
        question_rows.append(_make_question(identifier, track, video, question, answer_format, cue_type=cue_type))
        key_row = _make_key(identifier, truth, tolerance, notes, {**source, "question_type": cue_type})
        if verification_intervals is not None:
            key_row["verification_intervals"] = verification_intervals
        key_rows.append(key_row)
        target_occurrences[identifier] = target_occurrences_count

    try:
        # Moment: object appearances, text changes, and motion-blurred crossings
        # are split across three independent videos.
        object_moment_path = videos / "synthetic-moment-objects.mp4"
        object_events = []
        object_truths = []
        for index, (label, symbol, color) in enumerate(SHAPES[:3]):
            start = _quantize(5.0 + index * 22.0 + rng.uniform(0.0, 1.0))
            end = _quantize(start + (0.55, 0.82, 1.1)[index])
            object_events.append({"symbol": symbol, "color": color, "start": start, "end": end, "index": index})
            object_truths.append((label, start, end))
        _render_video(object_moment_path, MOMENT_SECONDS, object_events, seed + 1)
        for index, (label, start, end) in enumerate(object_truths):
            identifier = f"synth_moment_{index + 1:02d}"
            question = f"At what second does the {label} first become visible?"
            add_question(identifier, "moment", object_moment_path, question, "number_seconds", start,
                         {"primary_seconds": 0.5, "reported_seconds": [0.2, 0.5, 1.0]},
                         "One frame-aligned symbol appearance on a moving textured background.", cue_type="visual moment", target_occurrences_count=1)

        text_moment_path = videos / "synthetic-moment-text-scenes.mp4"
        text_moments = [
            ("POWER RESTORED", 7.0, 8.1),
            ("CABIN PRESSURIZED", 30.0, 31.35),
            ("REACTOR SAFE", 55.0, 56.0),
        ]
        text_events = [{"text": text, "start": start, "end": end, "size": 38} for text, start, end in text_moments]
        _render_video(text_moment_path, MOMENT_SECONDS, [], seed + 11, text_events=text_events,
                      scene_cuts=[(0.0, 24.0, "navy"), (24.0, 50.0, "maroon"), (50.0, 78.0, "darkgreen")])
        for offset, (text, start, end) in enumerate(text_moments, 4):
            identifier = f"synth_moment_{offset:02d}"
            question = f"What interval in seconds is the exact on-screen phrase ‘{text}’ visible? Return start and end."
            add_question(identifier, "moment", text_moment_path, question, "interval_seconds", [start, end],
                         {"primary_seconds": 0.5, "reported_seconds": [0.2, 0.5, 1.0]},
                         "A single exact text change appears over a hard scene cut.", cue_type="visual moment", target_occurrences_count=1)

        crossing_moment_path = videos / "synthetic-moment-motion-blur.mp4"
        crossing_moments = []
        crossing_events = []
        for index, (label, symbol, color) in enumerate(SHAPES[6:9]):
            start = _quantize(8.0 + index * 22.0 + rng.uniform(0.0, 0.7))
            end = _quantize(start + 1.5)
            crossing_events.append({"symbol": symbol, "color": color, "start": start, "end": end, "index": 100 + index})
            crossing_moments.append((label, start, end))
        _render_video(crossing_moment_path, MOMENT_SECONDS, crossing_events, seed + 12, motion_blur=True,
                      scene_cuts=[(0.0, 26.0, "darkslateblue"), (26.0, 52.0, "darkolivegreen"), (52.0, 78.0, "darkslategray")])
        for offset, (label, start, end) in enumerate(crossing_moments, 7):
            identifier = f"synth_moment_{offset:02d}"
            question = f"During what interval does the {label} marker cross the display from left to right? Return start and end."
            add_question(identifier, "moment", crossing_moment_path, question, "interval_seconds", [start, end],
                         {"primary_seconds": 0.5, "reported_seconds": [0.2, 0.5, 1.0]},
                         "One motion-blurred left-to-right marker crossing.", cue_type="visual moment", target_occurrences_count=1)
        add_question("synth_moment_10", "moment", text_moment_path,
                     "At what second does the exact phrase ‘ANTARCTIC POLAR BEAR’ appear? If it never appears, answer not_found.",
                     "number_seconds_or_not_found", "not_found",
                     {"primary_seconds": 0.5, "reported_seconds": [0.2, 0.5, 1.0]},
                     "The phrase is absent from all generated text and symbol overlays.", cue_type="negative control", target_occurrences_count=0)

        # Count: three icon questions, three exact-text questions, and three
        # moving-marker questions use separate source videos.
        icon_count_path = videos / "synthetic-count-icons.mp4"
        icon_events = []
        icon_targets = []
        icon_counts = [3, 2, 4]
        for index, (label, symbol, color) in enumerate(SHAPES[:3]):
            count = icon_counts[index]
            base = 4.0 + index * 25.0 + rng.uniform(0.0, 0.5)
            intervals = []
            for rep in range(count):
                start = _quantize(base + rep * 1.7)
                end = _quantize(start + 0.9)
                icon_events.append({"symbol": symbol, "color": color, "start": start, "end": end, "index": index * 10 + rep})
                intervals.append([start, end])
            icon_targets.append((label, symbol, color, count, intervals))
            decoy_start = _quantize(base + count * 1.7 + 0.45)
            decoy_color = ("magenta", "gold", "lightgreen")[index]
            icon_events.append({"symbol": symbol, "color": decoy_color, "start": decoy_start, "end": _quantize(decoy_start + 0.8), "index": index + 30})
        _render_video(icon_count_path, COUNT_SECONDS, icon_events, seed + 21)
        for index, (label, _symbol, _color, count, intervals) in enumerate(icon_targets, 1):
            identifier = f"synth_count_{index:02d}"
            question = f"How many separate times does the {label} appear? Ignore the same symbol in other colors."
            add_question(identifier, "count", icon_count_path, question, "integer", count,
                         {"exact": True, "off_by_one": 1}, "Repeated frame-bounded target appearances with a similar-color distractor.",
                         cue_type="visual action count", target_occurrences_count=count,
                         verification_intervals=intervals)

        text_count_path = videos / "synthetic-count-text-scenes.mp4"
        text_targets = [("DOCKING", 2), ("RELEASE", 4), ("ALL CLEAR", 3)]
        text_events = []
        text_intervals = []
        for group, (phrase, count) in enumerate(text_targets):
            base = 4.0 + group * 25.0 + rng.uniform(0.0, 0.5)
            y = 50 + group * 105
            intervals = []
            for rep in range(count):
                start = _quantize(base + rep * 1.8)
                end = _quantize(start + 1.0)
                text_events.append({"text": phrase, "start": start, "end": end, "size": 42, "y": y})
                intervals.append([start, end])
            text_intervals.append(intervals)
            decoy_start = _quantize(base + count * 1.8 + 0.45)
            text_events.append({"text": phrase + "ED", "start": decoy_start, "end": _quantize(decoy_start + 1.0), "size": 40, "y": y})
        _render_video(text_count_path, COUNT_SECONDS, [], seed + 22, text_events=text_events,
                      scene_cuts=[(0.0, 29.0, "midnightblue"), (29.0, 58.0, "darkred"), (58.0, 88.0, "darkslategray")])
        for index, (phrase, count) in enumerate(text_targets, 4):
            identifier = f"synth_count_{index:02d}"
            question = f"How many separate times does the exact on-screen phrase ‘{phrase}’ appear? Ignore similar phrases with extra letters."
            add_question(identifier, "count", text_count_path, question, "integer", count,
                         {"exact": True, "off_by_one": 1}, "Exact phrase appears repeatedly; a longer look-alike phrase is a distractor.",
                         cue_type="visual action count", target_occurrences_count=count,
                         verification_intervals=text_intervals[index - 4])

        motion_count_path = videos / "synthetic-count-motion-blur.mp4"
        motion_count_events = []
        motion_targets = []
        motion_counts = [2, 3, 1]
        for group, (label, symbol, color) in enumerate(SHAPES[6:9]):
            count = motion_counts[group]
            intervals = []
            base = 5.0 + group * 24.0 + rng.uniform(0.0, 0.5)
            for rep in range(count):
                start = _quantize(base + rep * 2.0)
                end = _quantize(start + 1.2)
                motion_count_events.append({"symbol": symbol, "color": color, "start": start, "end": end, "index": 110 + group * 10 + rep})
                intervals.append([start, end])
            motion_targets.append((label, symbol, color, count, intervals))
            decoy_start = _quantize(base + count * 2.0 + 0.55)
            motion_count_events.append({"symbol": symbol, "color": ("coral", "gold", "deepskyblue")[group], "start": decoy_start, "end": _quantize(decoy_start + 1.1), "index": 150 + group})
        _render_video(motion_count_path, COUNT_SECONDS, motion_count_events, seed + 23, motion_blur=True,
                      scene_cuts=[(0.0, 29.0, "darkslateblue"), (29.0, 58.0, "darkolivegreen"), (58.0, 88.0, "darkslategray")])
        for index, (label, _symbol, _color, count, intervals) in enumerate(motion_targets, 7):
            identifier = f"synth_count_{index:02d}"
            question = f"How many times does the {label} marker travel left to right across the screen? Ignore different-colored look-alikes."
            add_question(identifier, "count", motion_count_path, question, "integer", count,
                         {"exact": True, "off_by_one": 1}, "Motion-blurred marker crossings with different-color look-alikes.",
                         cue_type="visual action count", target_occurrences_count=count,
                         verification_intervals=intervals)
        add_question("synth_count_10", "count", motion_count_path,
                     "How many times does a solid black crescent cross the display? Answer 0 if none appear.",
                     "integer_or_zero", 0, {"exact": True, "off_by_one": 1},
                     "No solid black crescent is drawn.", cue_type="negative control", target_occurrences_count=0)

        # Long search is divided across three independently rendered ten-minute
        # videos. Targets include scene text, motion, speech, and distinct sounds.
        search_paths = [
            videos / "synthetic-search-scenes-10min.mp4",
            videos / "synthetic-search-motion-audio-10min.mp4",
            videos / "synthetic-search-text-audio-10min.mp4",
        ]
        search_events: list[list[dict[str, Any]]] = [[], [], []]
        search_text_events: list[list[dict[str, Any]]] = [[], [], []]
        search_cues: list[list[dict[str, Any]]] = [[], [], []]
        visual_specs = [
            ("synth_search_01", 49.0, 0, "yellow eye-shaped ring", SHAPES[0], False),
            ("synth_search_02", 158.0, 1, "cyan warning triangle", SHAPES[1], False),
            ("synth_search_03", 288.0, 2, "TRANSFER COMPLETE", None, True),
            ("synth_search_04", 521.0, 0, "green outlined diamond", SHAPES[6], False),
        ]
        for index, (identifier, start, video_index, target, shape, is_text) in enumerate(visual_specs):
            end = start + 0.9
            if is_text:
                search_text_events[video_index].append({"text": target, "start": start, "end": end, "size": 48})
                question = f"Find the moment in this 10-minute video when the display shows the exact phrase ‘{target}’. Return [start, end] in seconds."
            else:
                assert shape is not None
                search_events[video_index].append({"symbol": shape[1], "color": shape[2], "start": start, "end": end, "index": 35 + index})
                question = f"Find the moment in this 10-minute video when the {target} is shown. Return [start, end] in seconds."
            search_cues[video_index].append({"id": identifier, "play_at": start, "truth": [start, end], "cue_type": "visual_only", "question": question,
                                             "notes": "One target visual event; nearby overlays are different targets."})
            target_occurrences[identifier] = 1

        speech_specs = [
            ("synth_search_05", 103.0, 0, "The copper umbrella is ready", SHAPES[5]),
            ("synth_search_06", 247.0, 1, "silver bicycle", SHAPES[7]),
            ("synth_search_07", 399.0, 2, "quiet station", SHAPES[3]),
        ]
        for index, (identifier, start, video_index, phrase, visual) in enumerate(speech_specs):
            search_events[video_index].append({"symbol": visual[1], "color": visual[2], "start": start + 1.0, "end": start + 1.9, "index": 45 + index})
            search_cues[video_index].append({"id": identifier, "play_at": start, "phrase": phrase, "cue_type": "speech_cued",
                                             "question": f"Find the moment in this 10-minute video when the speaker says ‘{phrase}’. Return [start, end] in seconds.",
                                             "notes": "The target phrase occurs once in the speech audio."})
            target_occurrences[identifier] = 1

        sound_specs = [
            ("synth_search_08", 184.0, 0, "three ascending electronic chime notes", (680, 960, 1380)),
            ("synth_search_09", 463.0, 1, "three descending electronic chime notes", (1380, 960, 680)),
        ]
        for index, (identifier, start, video_index, cue_name, frequencies) in enumerate(sound_specs):
            distractor = SHAPES[4 + index * 3]
            search_events[video_index].append({"symbol": distractor[1], "color": distractor[2], "start": start + 4.0, "end": start + 4.9, "index": 50 + index})
            search_cues[video_index].append({"id": identifier, "play_at": start, "frequencies": frequencies, "cue_type": "sound_cued",
                                             "question": f"Find the moment in this 10-minute video when {cue_name} sound. Return [start, end] in seconds.",
                                             "notes": "The target is audio-only; a visual distractor occurs a few seconds later."})
            target_occurrences[identifier] = 1

        for video_index, path in enumerate(search_paths):
            cuts = [(0.0, 200.0, "navy"), (200.0, 400.0, "darkred"), (400.0, 600.0, "darkgreen")]
            _render_video(path, SEARCH_SECONDS, search_events[video_index], seed + 30 + video_index, long_form=True,
                          text_events=search_text_events[video_index], scene_cuts=cuts, motion_blur=video_index == 1)
            audio_path = stage / f"synthetic-search-audio-{video_index}.m4a"
            _mix_long_audio(path, audio_path, stage, search_cues[video_index], SEARCH_SECONDS, seed + 40 + video_index)
            muxed_path = stage / f"synthetic-search-muxed-{video_index}.mp4"
            _mux(path, audio_path, muxed_path)
            os.replace(muxed_path, path)

        for cue in (cue for group in search_cues for cue in group):
            add_question(cue["id"], "search", search_paths[next(i for i, group in enumerate(search_cues) if cue in group)],
                         cue["question"], "interval_seconds", cue.get("truth", "not_found"),
                         {"primary_seconds": 2.0, "reported_seconds": [2.0, 5.0]}, cue["notes"],
                         cue_type=cue["cue_type"], target_occurrences_count=1)
        add_question("synth_search_10", "search", search_paths[2],
                     "Find the moment in this 10-minute video when a dog barks. If no bark occurs, answer not_found.",
                     "number_seconds_or_not_found", "not_found", {"primary_seconds": 2.0, "reported_seconds": [2.0, 5.0]},
                     "No dog bark is present in the generated audio or video.", cue_type="negative control", target_occurrences_count=0)

        q_counts = {track: sum(row["track"] == track for row in question_rows) for track in ("moment", "search", "count")}
        if q_counts != {"moment": 10, "search": 10, "count": 10}:
            raise RuntimeError(f"synthetic question-count invariant failed: {q_counts}")
        for track in ("moment", "search", "count"):
            paths = {row["video_path"] for row in question_rows if row["track"] == track}
            if len(paths) < 3:
                raise RuntimeError(f"{track} synthetic questions use only {len(paths)} videos; expected at least 3")
        _validate_generated_questions(question_rows, key_rows, target_occurrences)
        return question_rows, key_rows
    finally:
        trash = shutil.which("trash")
        if trash:
            subprocess.run([trash, str(stage)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8448)
    parser.add_argument("--force", action="store_true", help="Replace prior bench-v1 question/key rows with matching ids")
    args = parser.parse_args()
    questions, keys = generate(args.seed)
    q_path, k_path = question_path(), key_path()
    prior_questions = read_jsonl(q_path) if q_path.exists() else []
    prior_keys = read_jsonl(k_path) if k_path.exists() else []
    if not args.force:
        # Generation is idempotent: same IDs are replaced; all other splits survive.
        pass
    write_jsonl_atomic(q_path, merge_by_id(prior_questions, questions))
    write_jsonl_atomic(k_path, merge_by_id(prior_keys, keys), mode=0o600)
    print(f"Generated 30 synthetic questions across 9 synthetic videos under {data_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
