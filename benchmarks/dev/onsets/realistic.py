"""Real-world-like beds, square clicks, sine beeps and decoys: the extended onset climb set.

Everything is generated from a seed; nothing is downloaded. All signals are
48 kHz float arrays. Bed generators take a target RMS in dBFS.

Beds:
- ``babble``: one speech-like voice. Syllables (120-300 ms, 3-8 per phrase,
  0.2-0.6 s pauses) of a gliding harmonic source shaped by three moving
  formants; some syllables open with a plosive-like noise burst (a decoy).
- ``crowd``: 10-16 babble voices at random gains over a pink roar, plus random
  low-level transients (claps, thuds).
- ``screen``: mains hum with harmonics, fan noise with a blade tone and sparse
  key clicks.
- ``codec``: a quiet noise floor whose spectral shape jumps every 20 ms (the
  frame-wise shape changes of a lossy codec at low bitrate).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

RATE = 48000


def _rms_to(signal: np.ndarray, level_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(signal * signal)))
    return signal * (10.0 ** (level_db / 20.0) / max(rms, 1e-12))


def _pink(count: int, rng: np.random.Generator, slope: float = 0.5) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(count))
    frequencies = np.fft.rfftfreq(count, 1.0 / RATE)
    frequencies[0] = frequencies[1]
    noise = np.fft.irfft(spectrum / frequencies ** slope, count)
    return noise / float(np.std(noise))


def _shaped_noise(count: int, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(count))
    frequencies = np.fft.rfftfreq(count, 1.0 / RATE)
    spectrum[(frequencies < low) | (frequencies > high)] = 0.0
    noise = np.fft.irfft(spectrum, count)
    return noise / max(1e-12, float(np.std(noise)))


def _syllable(rng: np.random.Generator, f0_base: float) -> np.ndarray:
    duration = rng.uniform(0.12, 0.30)
    count = int(duration * RATE)
    local = np.arange(count) / RATE
    f0 = f0_base * (1.0 + rng.uniform(-0.15, 0.15)) * (1.0 + rng.uniform(-0.12, 0.12) * local / duration)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    formants_start = np.array([rng.uniform(300, 800), rng.uniform(900, 2200), rng.uniform(2300, 3000)])
    formants_end = formants_start * rng.uniform(0.8, 1.25, size=3)
    widths = np.array([90.0, 120.0, 160.0])
    gains = np.array([1.0, 0.6, 0.3])
    progress = local / duration
    mean_f0 = float(np.mean(f0))
    harmonics = np.arange(1, int(4500 / mean_f0))
    formants = formants_start[None, :] * (1 - progress[:, None]) + formants_end[None, :] * progress[:, None]
    # Each harmonic's gain follows the moving formants (sampled every 5 ms) with a gentle spectral tilt.
    coarse = slice(None, None, 240)
    envelope = np.sum(gains * np.exp(-0.5 * ((harmonics[None, :, None] * mean_f0 - formants[coarse, None, :]) / widths) ** 2), axis=2)
    envelope = np.stack([np.interp(local, local[coarse], column) for column in envelope.T], axis=1) / harmonics ** 0.6
    offsets = rng.uniform(0, 2 * np.pi, size=len(harmonics))
    voiced = np.sum(envelope * np.sin(phase[:, None] * harmonics[None, :] + offsets), axis=1)
    attack = min(count, int(rng.uniform(0.02, 0.04) * RATE))
    release = min(count - attack, int(rng.uniform(0.04, 0.08) * RATE))
    shape = np.ones(count)
    shape[:attack] = 0.5 - 0.5 * np.cos(np.pi * np.arange(attack) / attack)
    shape[count - release:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(release) / release)
    return voiced * shape / max(1e-9, float(np.max(np.abs(voiced))))


def _plosive(rng: np.random.Generator) -> np.ndarray:
    count = int(rng.uniform(0.005, 0.015) * RATE)
    burst = _shaped_noise(max(count, 64), rng, rng.uniform(800, 1500), rng.uniform(4000, 8000))[:count]
    return burst * np.exp(-np.arange(count) / (0.3 * count)) * rng.uniform(0.5, 1.0)


def babble(count: int, rng: np.random.Generator, level_db: float, plosive_rate: float = 0.35) -> Tuple[np.ndarray, List[float]]:
    """One speech-like voice at `level_db` RMS; returns it and its plosive burst times (decoys)."""
    signal = np.zeros(count)
    plosives: List[float] = []
    f0_base = rng.uniform(100, 240)
    position = rng.uniform(0.0, 0.3)
    while position < count / RATE:
        for _ in range(int(rng.integers(3, 9))):
            first = int(position * RATE)
            if first >= count:
                break
            syllable = _syllable(rng, f0_base) * rng.uniform(0.5, 1.0)
            if rng.random() < plosive_rate:
                burst = _plosive(rng)
                last = min(count, first + len(burst))
                signal[first:last] += burst[:last - first]
                plosives.append(first / RATE)
                first += len(burst) + int(0.01 * RATE)
            last = min(count, first + len(syllable))
            if last > first:
                signal[first:last] += syllable[:last - first]
            position = last / RATE + rng.uniform(0.0, 0.06)
        position += rng.uniform(0.2, 0.6)
    # A room-noise floor 30-45 dB under the voice, so pauses are not digital silence.
    voice = _rms_to(signal, 0.0)
    floor = _pink(count, rng) * 10 ** (-rng.uniform(30.0, 45.0) / 20.0)
    return _rms_to(voice + floor, level_db), plosives


def crowd(count: int, rng: np.random.Generator, level_db: float) -> np.ndarray:
    """Many overlapping voices, a pink roar and random low-level transients at `level_db` RMS."""
    signal = 0.5 * _pink(count, rng)
    for _ in range(int(rng.integers(10, 17))):
        voice, _ = babble(count, rng, rng.uniform(-6.0, 0.0), plosive_rate=0.2)
        signal += voice
    position = rng.uniform(0.0, 0.5)
    while position < count / RATE:
        first = int(position * RATE)
        length = int(rng.uniform(0.01, 0.06) * RATE)
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.25 * length)) * rng.uniform(0.5, 2.0)
        last = min(count, first + length)
        signal[first:last] += burst[:last - first]
        position += rng.uniform(0.2, 1.2)
    return _rms_to(signal, level_db)


def screen(count: int, rng: np.random.Generator, level_db: float) -> np.ndarray:
    """Mains hum, fan noise with a blade tone and sparse key clicks at `level_db` RMS."""
    local = np.arange(count) / RATE
    mains = float(rng.choice([50.0, 60.0]))
    hum = sum(np.sin(2 * np.pi * harmonic * mains * local + rng.uniform(0, 6.3)) / harmonic for harmonic in (1, 2, 3, 5))
    fan = _pink(count, rng, slope=0.8)
    blade = rng.uniform(100.0, 300.0)
    fan += 0.4 * np.sin(2 * np.pi * blade * local) * (1.0 + 0.1 * np.sin(2 * np.pi * 0.3 * local))
    signal = 0.5 * hum + fan
    keys = np.zeros(count)
    position = rng.uniform(0.0, 1.0)
    while position < count / RATE:
        first = int(position * RATE)
        length = int(rng.uniform(0.003, 0.008) * RATE)
        burst = _shaped_noise(max(length, 64), rng, 1500, 9000)[:length] * np.exp(-np.arange(length) / (0.3 * length))
        last = min(count, first + length)
        keys[first:last] += burst[:last - first] * rng.uniform(1.0, 3.0)
        position += rng.uniform(0.15, 1.5)
    return _rms_to(signal + keys, level_db)


def codec(count: int, rng: np.random.Generator, level_db: float) -> np.ndarray:
    """A noise floor whose 1/3-octave band gains jump every 20 ms (codec-like) at `level_db` RMS."""
    frame = int(0.02 * RATE)
    noise = rng.standard_normal(count + frame)
    out = np.zeros(count + frame)
    window = np.hanning(2 * frame)
    edges = 125.0 * 2.0 ** (np.arange(0, 22) / 3.0)
    for first in range(0, count, frame):
        piece = noise[first:first + 2 * frame]
        if len(piece) < 2 * frame:
            break
        spectrum = np.fft.rfft(piece * window)
        frequencies = np.fft.rfftfreq(2 * frame, 1.0 / RATE)
        gains = np.zeros_like(frequencies)
        for low, high in zip(edges[:-1], edges[1:]):
            # Each band is kept or dropped and its gain jumps by up to 9 dB from frame to frame.
            if rng.random() < 0.75:
                gains[(frequencies >= low) & (frequencies < high)] = 10 ** (rng.uniform(-9, 0) / 20) / np.sqrt(low / 125.0)
        out[first:first + 2 * frame] += np.fft.irfft(spectrum * gains, 2 * frame)
    return _rms_to(out[:count], level_db)


def square_click(rng: np.random.Generator, frequency: float, duration: float, amplitude: float) -> np.ndarray:
    count = int(duration * RATE)
    local = np.arange(count) / RATE
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * local + 1e-9))


def sine_beep(rng: np.random.Generator, frequency: float, duration: float, amplitude: float) -> np.ndarray:
    count = int(duration * RATE)
    local = np.arange(count) / RATE
    tone = amplitude * np.sin(2 * np.pi * frequency * local)
    ramp = min(count // 2, int(rng.choice([0.0, 0.001, 0.003]) * RATE))
    if ramp:
        tone[:ramp] *= np.arange(ramp) / ramp
        tone[count - ramp:] *= np.arange(ramp)[::-1] / ramp
    return tone


# ---------------------------------------------------------------------------
# The extended climb set: 36 bed groups, each with one bed (kind and seed)
# shared by all of the group's clips, so a split by group keeps every bed on
# one side.

SEED = 20261101
GROUPS = 36
CLIMB_GROUPS = 24
DURATION_SEC = 6.0
BED_KINDS = ("codec", "babble", "crowd", "screen")
LEGACY_KINDS = ("click", "clap", "drum", "pluck")
CLICK_BAND = [1000.0, 4000.0]


def plan() -> List[dict]:
    """The 120 clips: per group a click clip, a beep clip, a legacy-kind clip and, every third group, a bed-only clip."""
    climb = {int(group) for group in np.random.default_rng(SEED).permutation(GROUPS)[:CLIMB_GROUPS]}
    entries = []
    for group in range(GROUPS):
        bed_seed = SEED + 1000 * group
        rng = np.random.default_rng(bed_seed)
        common = {"bed": BED_KINDS[group % len(BED_KINDS)], "bed_seed": bed_seed,
                  "split": "climb" if group in climb else "heldout"}
        click_level, beep_level, mixed_level, bed_level = (rng.uniform(-60, -20), rng.uniform(-32, -16),
                                                           rng.uniform(-50, -25), rng.uniform(-50, -16))
        beep_hz = float(rng.uniform(500.0, 3000.0))
        entries.append({**common, "id": f"g{group:02d}-click", "family": "click", "level": click_level, "seed": bed_seed + 1})
        entries.append({**common, "id": f"g{group:02d}-beep", "family": "beep", "level": beep_level, "seed": bed_seed + 2,
                        "beep_hz": beep_hz})
        entries.append({**common, "id": f"g{group:02d}-mixed", "family": "mixed", "level": mixed_level, "seed": bed_seed + 3,
                        "kind": LEGACY_KINDS[(group // len(BED_KINDS)) % len(LEGACY_KINDS)]})
        if group % 3 == 0:
            entries.append({**common, "id": f"g{group:02d}-bed", "family": "bed", "level": bed_level, "seed": bed_seed + 4,
                            "beep_hz": beep_hz})
    return entries


def settings(entry: dict) -> List[Tuple[str, Optional[List[float]], float]]:
    """(setting name, band_hz, min_strength) pairs a clip is scored under."""
    beep = None
    if "beep_hz" in entry:
        beep = ("beep", [entry["beep_hz"] - 100.0, entry["beep_hz"] + 100.0], 0.7)
    by_family = {
        "click": [("click", CLICK_BAND, 0.36), ("broadband", None, 0.12)],
        "beep": [beep],
        "mixed": [("mixed", CLICK_BAND, 0.36), ("broadband", None, 0.12)],
        "bed": [("click", CLICK_BAND, 0.36), beep, ("broadband", None, 0.12)],
    }
    return by_family[entry["family"]]


def _bed_signal(entry: dict, count: int) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(entry["bed_seed"])
    kind, level = entry["bed"], entry["level"]
    if kind == "babble":
        return babble(count, rng, level)
    return {"codec": codec, "crowd": crowd, "screen": screen}[kind](count, rng, level), []


def make_clip(entry: dict, legacy_event_wave) -> Tuple[np.ndarray, List[float], Dict[str, List[float]]]:
    """Waveform, planted onset times and decoy times (out-of-band beeps, plosive bursts) for one clip."""
    count = int(DURATION_SEC * RATE)
    signal, plosives = _bed_signal(entry, count)
    signal = signal.copy()
    rng = np.random.default_rng(entry["seed"])
    truth: List[float] = []
    decoys: Dict[str, List[float]] = {"plosive": plosives, "off_band_beep": []}
    position = rng.uniform(0.4, 0.9)
    family = entry["family"]
    while family != "bed" and position < DURATION_SEC - 0.6:
        first = int(round(position * RATE))
        if family == "click":
            wave = square_click(rng, rng.uniform(500.0, 3000.0), rng.uniform(0.005, 0.020), rng.uniform(0.25, 1.0))
            truth.append(first / RATE)
            gap = rng.uniform(0.15, 1.5)
        elif family == "beep":
            frequency = entry["beep_hz"]
            decoy = rng.random() < 0.3
            if decoy:
                frequency = float(np.clip(frequency + rng.choice([-1, 1]) * rng.uniform(400.0, 1500.0), 300.0, 6000.0))
                if abs(frequency - entry["beep_hz"]) < 400.0:
                    frequency = entry["beep_hz"] + 400.0 + rng.uniform(0.0, 1000.0)
            wave = sine_beep(rng, frequency, rng.uniform(0.05, 0.30), rng.uniform(0.15, 0.8))
            (decoys["off_band_beep"] if decoy else truth).append(first / RATE)
            gap = len(wave) / RATE + rng.uniform(0.15, 1.0)
        else:
            wave = legacy_event_wave(entry["kind"], rng)
            truth.append(first / RATE)
            gap = rng.uniform(0.1, 2.0)
        last = min(count, first + len(wave))
        signal[first:last] += wave[:last - first]
        position += gap
    return np.clip(signal, -0.95, 0.95).astype(np.float32), truth, decoys
