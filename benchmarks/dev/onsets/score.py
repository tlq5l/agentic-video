"""Onset timing climb set and scorer for ``audio_onsets``.

Generates 60 synthetic clips (6 s, 48 kHz mono, 16-bit WAV) with planted onset
times and scores the onset times ``audio_onsets`` reports: F1 at a 30 ms
tolerance with one-to-one matching, pooled over a split. Nothing is downloaded.

Sound kinds: clicks, claps (noise bursts), drum hits (decaying sweeps) and
plucks (decaying harmonic tones). Beds: silence, pink noise and a sustained
music-like chord. Clip ``i`` has kind ``i % 4`` and bed ``(i // 4) % 3``, so each
of the 12 pairs appears 5 times. Event spacings are uniform in 0.1-2 s.

A permutation seeded with 20261031 gives the climb split (first 40 of the
permutation) and the held-out split (last 20). Tune only on the climb split.
The held-out split is printed only when asked for it.

    python benchmarks/dev/onsets/score.py --split climb --breakdown
    python benchmarks/dev/onsets/score.py --split heldout

Each run scores two settings: the gallery setting (``band_hz`` [1000, 4000],
``min_strength`` 0.36; primary) and the tool default (broadband, 0.12; guard).
Prints ``METRIC onsets.<split>.<setting>_f1``, ``_precision``, ``_recall``,
``_tp``, ``_fp`` and ``_fn``. ``--min-gap-sec`` is passed to ``audio_onsets``
only when given.
"""
from __future__ import annotations

import argparse
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]

RATE = 48000
DURATION_SEC = 6.0
SEED = 20261031
TOLERANCE_SEC = 0.030
N_CLIPS = 60
CLIMB_COUNT = 40
KINDS = ("click", "clap", "drum", "pluck")
BEDS = ("silence", "pink", "music")


def clip_plan() -> List[Dict[str, Any]]:
    """The 60 clips with their kind, bed, split and seed."""
    climb = {int(index) for index in np.random.default_rng(SEED).permutation(N_CLIPS)[:CLIMB_COUNT]}
    return [
        {
            "id": f"c{index:02d}",
            "kind": KINDS[index % len(KINDS)],
            "bed": BEDS[(index // len(KINDS)) % len(BEDS)],
            "split": "climb" if index in climb else "heldout",
            "seed": SEED + index,
        }
        for index in range(N_CLIPS)
    ]


def _event_wave(kind: str, rng: np.random.Generator) -> np.ndarray:
    amplitude = rng.uniform(0.4, 0.8)
    if kind == "click":
        local = np.arange(int(0.018 * RATE)) / RATE
        wave = (0.65 * np.sin(2 * np.pi * 1700 * local) + 0.25 * np.sin(2 * np.pi * 7300 * local)) * np.exp(-110 * local)
    elif kind == "clap":
        local = np.arange(int(0.06 * RATE)) / RATE
        wave = rng.standard_normal(len(local)) * np.exp(-45 * local)
    elif kind == "drum":
        local = np.arange(int(0.2 * RATE)) / RATE
        sweep = 60 + 90 * np.exp(-25 * local)
        wave = np.sin(2 * np.pi * np.cumsum(sweep) / RATE) * np.exp(-22 * local)
        wave = wave + 0.3 * rng.standard_normal(len(local)) * np.exp(-150 * local)
    else:
        local = np.arange(int(0.4 * RATE)) / RATE
        f0 = rng.uniform(220.0, 880.0)
        wave = sum(np.sin(2 * np.pi * harmonic * f0 * local) / harmonic for harmonic in range(1, 6)) * np.exp(-8 * local)
    return amplitude * wave / max(1e-9, float(np.max(np.abs(wave))))


def _bed(bed: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if bed == "silence":
        return np.zeros(count)
    if bed == "pink":
        spectrum = np.fft.rfft(rng.standard_normal(count))
        frequencies = np.fft.rfftfreq(count, 1.0 / RATE)
        frequencies[0] = 1.0
        noise = np.fft.irfft(spectrum / np.sqrt(frequencies), count)
        return 0.02 * noise / float(np.std(noise))
    local = np.arange(count) / RATE
    roots = rng.choice([220.0, 262.0, 330.0, 392.0, 440.0, 523.0], size=3, replace=False)
    chord = sum(np.sin(2 * np.pi * harmonic * root * local) / harmonic for root in roots for harmonic in (1, 2, 3))
    tremolo = 1.0 - 0.3 * np.sin(2 * np.pi * 0.7 * local)
    return 0.012 * chord * tremolo


def make_clip(entry: Dict[str, Any]) -> Tuple[np.ndarray, List[float]]:
    """Waveform and planted onset times (seconds) for one planned clip."""
    rng = np.random.default_rng(entry["seed"])
    count = int(DURATION_SEC * RATE)
    signal = _bed(entry["bed"], count, rng)
    onsets: List[float] = []
    position = rng.uniform(0.4, 0.9)
    while position < DURATION_SEC - 0.6:
        first = int(round(position * RATE))
        wave = _event_wave(entry["kind"], rng)
        last = min(count, first + len(wave))
        signal[first:last] += wave[:last - first]
        onsets.append(first / RATE)
        position += rng.uniform(0.1, 2.0)
    return np.clip(signal, -0.95, 0.95).astype(np.float32), onsets


def write_wav(path: Path, samples: np.ndarray) -> Path:
    pcm = (np.clip(samples, -0.999, 0.999) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


def match(truth: Sequence[float], detected: Sequence[float], tolerance: float = TOLERANCE_SEC) -> Tuple[int, int, int]:
    """(true positives, false positives, false negatives) with one-to-one matching, closest pairs first."""
    pairs = sorted(
        (abs(hit - target), t_index, d_index)
        for t_index, target in enumerate(truth)
        for d_index, hit in enumerate(detected)
        if abs(hit - target) <= tolerance
    )
    used_truth: set = set()
    used_hits: set = set()
    for _, t_index, d_index in pairs:
        if t_index not in used_truth and d_index not in used_hits:
            used_truth.add(t_index)
            used_hits.add(d_index)
    true_positive = len(used_truth)
    return true_positive, len(detected) - true_positive, len(truth) - true_positive


def score(split: str, band_hz: Optional[List[float]], min_strength: float, min_gap_sec: Optional[float], work: Path) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO))
    from src.core.cache import MediaCache
    from src.media.audio import audio_onsets

    work.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {"min_strength": min_strength, "band_hz": band_hz, "cache": MediaCache(work / "cache")}
    if min_gap_sec is not None:
        kwargs["min_gap_sec"] = min_gap_sec
    totals = {"tp": 0, "fp": 0, "fn": 0}
    groups: Dict[str, Dict[str, int]] = {}
    for entry in clip_plan():
        if entry["split"] != split:
            continue
        samples, truth = make_clip(entry)
        path = write_wav(work / f"{entry['id']}.wav", samples)
        detected = [float(item["time_sec"]) for item in audio_onsets(str(path), **kwargs)["onsets"]]
        counts = match(truth, detected)
        for key, value in zip(("tp", "fp", "fn"), counts):
            totals[key] += value
            for group in (entry["kind"], entry["bed"]):
                groups.setdefault(group, {"tp": 0, "fp": 0, "fn": 0})[key] += value
    return {**totals, **_rates(totals), "groups": {name: _rates(counts) | counts for name, counts in groups.items()}}


def _rates(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["climb", "heldout"], required=True)
    parser.add_argument("--min-gap-sec", type=float, default=None, help="passed to audio_onsets only when given")
    parser.add_argument("--work", default=None, help="scratch folder for the generated WAVs (default: a new temp folder)")
    parser.add_argument("--breakdown", action="store_true", help="also print tp/fp/fn per sound kind and per bed")
    args = parser.parse_args(argv)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="onsets-score-"))
    work.mkdir(parents=True, exist_ok=True)
    # The reported gallery setting (primary) and the tool's default broadband setting (guard).
    configs = {
        "band": ([1000.0, 4000.0], 0.36),
        "broadband": (None, 0.12),
    }
    for label, (band, min_strength) in configs.items():
        result = score(args.split, band, min_strength, args.min_gap_sec, work / label)
        prefix = f"onsets.{args.split}.{label}"
        for key in ("f1", "precision", "recall"):
            print(f"METRIC {prefix}_{key}={result[key]:.4f}")
        for key in ("tp", "fp", "fn"):
            print(f"METRIC {prefix}_{key}={result[key]}")
        if args.breakdown:
            for name, counts in sorted(result["groups"].items()):
                print(f"BREAKDOWN {prefix}.{name} f1={counts['f1']:.4f} tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}")


if __name__ == "__main__":
    main()
