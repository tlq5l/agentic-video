"""Guards audio_onsets finding loud 1 kHz beeps over speech-like babble at band [900, 1100] and 0.7, through handle_call."""
import wave

import numpy as np

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
