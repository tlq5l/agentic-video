"""End-to-end tests for the optional cues.json "sound_file" entry, through the real checker CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

RATE = 48000
REPO = Path(__file__).resolve().parents[1]
CLICK_TIMES = (1.0, 2.0)


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")


def _write_click_wav(path: Path, duration: float, click_times) -> Path:
    samples = np.zeros(int(duration * RATE), dtype=np.float32)
    count = int(0.025 * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    pulse = 0.75 * np.sin(2 * np.pi * 1750 * t) * np.exp(-t * 80.0)
    for when in click_times:
        first = int(round(when * RATE))
        samples[first:first + count] += pulse
    pcm = np.clip(samples, -0.95, 0.95)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


def _video_with_clicks(tmp_path: Path) -> Path:
    """A 3 s test pattern whose audio has a click at 1.0 s and at 2.0 s."""
    audio = _write_click_wav(tmp_path / "mix.wav", 3.0, CLICK_TIMES)
    video = tmp_path / "clicks.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24:duration=3",
        "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(video),
    ], check=True, capture_output=True, text=True)
    return video


def _run_check(video: Path, cues: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([
        sys.executable, "-m", "src.loop", "check", str(video), "--cues", str(cues),
        "--out", str(out), "--judge", "none",
    ], cwd=str(REPO), capture_output=True, text=True, check=False)


def _sync_cue(report: dict, label: str) -> dict:
    cues = report["measurements"]["sync_check"]["cues"]
    matches = [cue for cue in cues if cue.get("label") == label]
    assert len(matches) == 1, "expected one sync cue labelled {!r}, got {}".format(label, [c.get("label") for c in cues])
    return matches[0]


def test_missing_sound_file_is_a_cue_error_and_other_cues_are_still_checked(tmp_path):
    _require_ffmpeg()
    video = _video_with_clicks(tmp_path)
    cues = tmp_path / "cues.json"
    cues.write_text(json.dumps({"cues": [
        {"time_sec": 1.0, "label": "missing file hit", "sound_file": "no_such_sound.wav"},
        {"time_sec": 2.0, "label": "plain hit"},
    ]}), encoding="utf-8")
    out = tmp_path / "check"

    command = _run_check(video, cues, out)

    assert command.returncode in (0, 1), command.stderr[-800:]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    broken = _sync_cue(report, "missing file hit")
    assert broken["sound_file"]["present"] is False
    assert broken["sound_file"]["error"]
    plain = _sync_cue(report, "plain hit")
    assert "sound_file" not in plain
    assert plain.get("status") is not None


def test_present_sound_file_is_matched_in_the_mix_and_not_reported_missing(tmp_path):
    _require_ffmpeg()
    video = _video_with_clicks(tmp_path)
    sound = _write_click_wav(tmp_path / "click.wav", 0.05, [0.0])
    cues = tmp_path / "cues.json"
    cues.write_text(json.dumps({"cues": [
        {"time_sec": 1.0, "label": "sound hit", "sound_file": sound.name},
        {"time_sec": 2.0, "label": "plain hit"},
    ]}), encoding="utf-8")
    out = tmp_path / "check"

    command = _run_check(video, cues, out)

    assert command.returncode in (0, 1), command.stderr[-800:]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    matched = _sync_cue(report, "sound hit")
    assert matched["sound_file"]["present"] is True
    assert "error" not in matched["sound_file"]
    missing = [c for c in report["candidates"]
               if c.get("kind") == "cue_missing" and abs(float(c.get("start_sec", -99.0)) - 1.0) <= 0.1]
    assert missing == []
