"""Guards audio_onsets through handle_call: loud 1 kHz beeps over speech-like babble at band [900, 1100] and 0.7,
quiet starts over a steady noise bed, and clips shorter than one analysis window."""
import wave

import numpy as np
import pytest

from src.core.config import Config
from src.executor.mcp_server import handle_call

RATE = 48000
BEEPS = [0.8, 1.9, 2.7, 3.6, 4.4, 5.2]


def _babble(count, rng):
    """One speech-like voice at -22 dBFS RMS: gliding harmonic syllables in 3-8 syllable phrases over a room-noise floor."""
    signal = np.zeros(count)
    position = 0.1
    while position < count / RATE:
        for _ in range(int(rng.integers(3, 9))):
            first = int(position * RATE)
            length = int(rng.uniform(0.12, 0.30) * RATE)
            local = np.arange(length) / RATE
            f0 = rng.uniform(100, 240) * (1 + rng.uniform(-0.1, 0.1) * local / local[-1])
            phase = 2 * np.pi * np.cumsum(f0) / RATE
            formants = (rng.uniform(300, 800), rng.uniform(900, 2200), rng.uniform(2300, 3000))
            mean_f0 = f0.mean()
            syllable = sum(
                sum(np.exp(-0.5 * ((k * mean_f0 - formant) / 120.0) ** 2) for formant in formants)
                / k ** 0.6 * np.sin(k * phase) for k in range(1, int(4500 / mean_f0)))
            syllable *= np.minimum(1.0, np.minimum(local / 0.03, (local[-1] - local) / 0.06))
            signal[first:first + length] += syllable[:max(0, count - first)]
            position = (first + length) / RATE + rng.uniform(0.0, 0.06)
        position += rng.uniform(0.2, 0.6)
    signal += 0.01 * np.std(signal) * rng.standard_normal(count)
    return signal * (10 ** (-22 / 20) / np.sqrt(np.mean(signal ** 2)))


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(samples, -0.999, 0.999) * 32767).astype("<i2").tobytes())
    return str(path)


def test_beeps_over_babble_are_found_once_each_at_their_start(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    signal = _babble(6 * RATE, np.random.default_rng(7))
    tone = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(int(0.1 * RATE)) / RATE)
    for start in BEEPS:
        first = int(start * RATE)
        signal[first:first + len(tone)] += tone
    source = _write_wav(tmp_path / "beeps-over-babble.wav", signal)

    result = handle_call("audio_onsets", {"source": source, "band_hz": [900, 1100], "min_strength": 0.7})

    assert result["ok"] is True
    times = [onset["time_sec"] for onset in result["data"]["onsets"]]
    # One onset per beep, each within 30 ms of the beep's start; neither the beep ends nor the speech fire.
    assert len(times) == len(BEEPS), times
    assert all(abs(hit - start) <= 0.03 for hit, start in zip(times, BEEPS)), times
    assert all(0.7 <= onset["strength"] <= 1.0 for onset in result["data"]["onsets"])


QUIET_STARTS = [0.5 + 0.5 * index for index in range(8)]


@pytest.mark.parametrize("kind, arguments", [
    ("beep", {}),  # 100 ms 1 kHz sines at -44 dBFS RMS over white noise at -69 dBFS, default min_strength
    ("click", {"band_hz": [1000, 4000], "min_strength": 0.36}),  # 15 ms 1 kHz squares at -40 dBFS over noise at -65
])
def test_quiet_starts_over_a_steady_bed_are_each_found(monkeypatch, tmp_path, kind, arguments):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    rng = np.random.default_rng(11)
    event_db = -44.0 if kind == "beep" else -40.0
    signal = rng.standard_normal(5 * RATE) * 10 ** ((event_db - 25.0) / 20)
    local = np.arange(int((0.1 if kind == "beep" else 0.015) * RATE)) / RATE
    if kind == "beep":
        event = np.sqrt(2) * 10 ** (event_db / 20) * np.sin(2 * np.pi * 1000 * local)
    else:
        event = 10 ** (event_db / 20) * np.sign(np.sin(2 * np.pi * 1000 * local + 1e-9))
    for start in QUIET_STARTS:
        first = int(start * RATE)
        signal[first:first + len(event)] += event
    source = _write_wav(tmp_path / f"quiet-{kind}s.wav", signal)

    result = handle_call("audio_onsets", {"source": source, **arguments})

    assert result["ok"] is True, result
    times = [onset["time_sec"] for onset in result["data"]["onsets"]]
    # Each start is found once, within 30 ms, and the steady noise between them never fires.
    assert len(times) == len(QUIET_STARTS), times
    assert all(abs(hit - start) <= 0.03 for hit, start in zip(times, QUIET_STARTS)), times


@pytest.mark.parametrize("band_hz", [None, [900, 1100], [1000, 4000]])
def test_clip_shorter_than_one_window_returns_normally(monkeypatch, tmp_path, band_hz):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    # 25 ms: a 1 kHz square burst after 5 ms of silence, shorter than the 2048- and 1024-sample windows.
    local = np.arange(int(0.025 * RATE)) / RATE
    signal = np.where(local >= 0.005, 0.5 * np.sign(np.sin(2 * np.pi * 1000 * local)), 0.0)
    source = _write_wav(tmp_path / "short.wav", signal)

    arguments = {"source": source} if band_hz is None else {"source": source, "band_hz": band_hz}
    result = handle_call("audio_onsets", arguments)

    assert result["ok"] is True, result
    assert all(0.0 <= onset["time_sec"] <= 0.025 for onset in result["data"]["onsets"])
