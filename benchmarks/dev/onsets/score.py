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

``--set real`` scores the 156 real-world-like clips of ``realistic.py`` instead
(``--set both``, the default, scores both sets): square clicks, sine beeps,
legacy kinds, quiet square bursts and bed-only clips over speech-like babble,
crowd, screen-recording and codec-floor beds, split by bed group (24 groups,
105 clips climb / 12 groups, 51 clips held out; groups differ in size).
Settings: click (band [1000, 4000], 0.36; at 30 and 50 ms), beep (band f +-100
Hz, 0.7; at 30 and 100 ms; off-band beeps and plosive bursts are decoys), mixed
(legacy kinds at the click setting), broadband (0.12), and quiet_click,
quiet_beep and quiet_broadband (the same three settings on quiet clips: bursts
at -50 to -30 dBFS, 20-30 dB over their bed). Every detection that is not a
planted event counts as a false positive, beds included.

``--set noloss --baseline-audio <file>`` runs a baseline detector (main's
``src/media/audio.py``) and this tree's on the same clips and lists every planted
start the baseline finds and this tree misses (see ``noloss.py``; ``--family``
picks the clip families, so slices can run in parallel).

Each run scores two settings: the gallery setting (``band_hz`` [1000, 4000],
``min_strength`` 0.36; primary) and the tool default (broadband, 0.12; guard).
Prints ``METRIC onsets.<split>.<setting>_f1``, ``_precision``, ``_recall``,
``_timing_median_ms``, ``_timing_p95_ms`` (absolute error of matched pairs),
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


def match_errors(truth: Sequence[float], detected: Sequence[float], tolerance: float = TOLERANCE_SEC) -> Tuple[int, int, int, List[float]]:
    """(true positives, false positives, false negatives, matched absolute errors in seconds); one-to-one, closest pairs first."""
    pairs = sorted(
        (abs(hit - target), t_index, d_index)
        for t_index, target in enumerate(truth)
        for d_index, hit in enumerate(detected)
        if abs(hit - target) <= tolerance
    )
    used_truth: set = set()
    used_hits: set = set()
    errors: List[float] = []
    for error, t_index, d_index in pairs:
        if t_index not in used_truth and d_index not in used_hits:
            used_truth.add(t_index)
            used_hits.add(d_index)
            errors.append(error)
    true_positive = len(used_truth)
    return true_positive, len(detected) - true_positive, len(truth) - true_positive, errors


def match(truth: Sequence[float], detected: Sequence[float], tolerance: float = TOLERANCE_SEC) -> Tuple[int, int, int]:
    """(true positives, false positives, false negatives) with one-to-one matching, closest pairs first."""
    return match_errors(truth, detected, tolerance)[:3]


def detect(path: Path, band_hz: Optional[List[float]], min_strength: float, cache: Any, min_gap_sec: Optional[float] = None) -> List[float]:
    import sys

    sys.path.insert(0, str(REPO))
    from src.media.audio import audio_onsets

    kwargs: Dict[str, Any] = {"min_strength": min_strength, "band_hz": band_hz, "cache": cache}
    if min_gap_sec is not None:
        kwargs["min_gap_sec"] = min_gap_sec
    return [float(item["time_sec"]) for item in audio_onsets(str(path), **kwargs)["onsets"]]


def score(split: str, band_hz: Optional[List[float]], min_strength: float, min_gap_sec: Optional[float], work: Path) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO))
    from src.core.cache import MediaCache

    work.mkdir(parents=True, exist_ok=True)
    cache = MediaCache(work / "cache")
    totals: Dict[str, Any] = {"tp": 0, "fp": 0, "fn": 0, "errors": []}
    groups: Dict[str, Dict[str, int]] = {}
    for entry in clip_plan():
        if entry["split"] != split:
            continue
        samples, truth = make_clip(entry)
        path = write_wav(work / f"{entry['id']}.wav", samples)
        detected = detect(path, band_hz, min_strength, cache, min_gap_sec)
        *counts, errors = match_errors(truth, detected)
        totals["errors"] += errors
        for key, value in zip(("tp", "fp", "fn"), counts):
            totals[key] += value
            for group in (entry["kind"], entry["bed"]):
                groups.setdefault(group, {"tp": 0, "fp": 0, "fn": 0})[key] += value
    return {**totals, **_rates(totals), **_timing(totals["errors"]),
            "groups": {name: _rates(counts) | counts for name, counts in groups.items()}}


def _timing(errors: Sequence[float]) -> Dict[str, float]:
    """Median and 95th percentile of matched absolute timing errors, in ms."""
    if not errors:
        return {"timing_median_ms": float("nan"), "timing_p95_ms": float("nan")}
    values = np.asarray(errors) * 1000.0
    return {"timing_median_ms": float(np.median(values)), "timing_p95_ms": float(np.percentile(values, 95))}


def score_real(split: str, work: Path) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Pooled counts per setting and tolerance on the extended (real-world-like) set."""
    import sys

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import realistic
    from src.core.cache import MediaCache

    work.mkdir(parents=True, exist_ok=True)
    cache = MediaCache(work / "cache")
    tolerances = {"click": (0.030, 0.050), "beep": (0.030, 0.100), "mixed": (0.030,), "broadband": (0.030,),
                  "quiet_click": (0.030, 0.050), "quiet_beep": (0.030, 0.100), "quiet_broadband": (0.030,)}
    pooled: Dict[str, Dict[float, Dict[str, Any]]] = {
        name: {tolerance: {"tp": 0, "fp": 0, "fn": 0, "errors": []} for tolerance in values}
        for name, values in tolerances.items()}
    for entry in realistic.plan():
        if entry["split"] != split:
            continue
        samples, truth, _ = realistic.make_clip(entry, _event_wave)
        path = write_wav(work / f"{entry['id']}.wav", samples)
        for name, band_hz, min_strength in realistic.settings(entry):
            detected = detect(path, band_hz, min_strength, cache)
            for tolerance, counts in pooled[name].items():
                tp, fp, fn, errors = match_errors(truth, detected, tolerance)
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
                counts["errors"] += errors
    return {name: {tolerance: {**counts, **_rates(counts), **_timing(counts["errors"])} for tolerance, counts in by_tolerance.items()}
            for name, by_tolerance in pooled.items()}


def _rates(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["climb", "heldout"], help="required except with --set noloss")
    parser.add_argument("--set", choices=["legacy", "real", "both", "noloss"], default="both",
                        help="legacy: the 60 original clips; real: the 156 real-world-like clips (realistic.py); "
                             "noloss: list planted starts --baseline-audio finds and this tree misses (noloss.py)")
    parser.add_argument("--baseline-audio", default=None,
                        help="--set noloss: the baseline's src/media/audio.py, e.g. from git show origin/main:src/media/audio.py")
    parser.add_argument("--family", action="append", choices=["legacy", "real", "short", "pair", "ranged"],
                        help="--set noloss: families to check (repeatable; default all)")
    parser.add_argument("--min-gap-sec", type=float, default=None, help="passed to audio_onsets only when given (legacy set)")
    parser.add_argument("--work", default=None, help="scratch folder for the generated WAVs (default: a new temp folder)")
    parser.add_argument("--breakdown", action="store_true", help="also print tp/fp/fn per sound kind and per bed (legacy set)")
    args = parser.parse_args(argv)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="onsets-score-"))
    work.mkdir(parents=True, exist_ok=True)
    if args.set == "noloss":
        if not args.baseline_audio:
            parser.error("--set noloss needs --baseline-audio")
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import noloss

        noloss.run(Path(args.baseline_audio), args.family or ["legacy", "real", "short", "pair", "ranged"], work / "noloss")
        return
    if args.split is None:
        parser.error("--split is required")
    timing_keys = ("timing_median_ms", "timing_p95_ms")
    if args.set in ("legacy", "both"):
        # The reported gallery setting (primary) and the tool's default broadband setting (guard).
        configs = {
            "band": ([1000.0, 4000.0], 0.36),
            "broadband": (None, 0.12),
        }
        for label, (band, min_strength) in configs.items():
            result = score(args.split, band, min_strength, args.min_gap_sec, work / label)
            prefix = f"onsets.{args.split}.{label}"
            for key in ("f1", "precision", "recall") + timing_keys:
                print(f"METRIC {prefix}_{key}={result[key]:.4f}")
            for key in ("tp", "fp", "fn"):
                print(f"METRIC {prefix}_{key}={result[key]}")
            if args.breakdown:
                for name, counts in sorted(result["groups"].items()):
                    print(f"BREAKDOWN {prefix}.{name} f1={counts['f1']:.4f} tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}")
    if args.set in ("real", "both"):
        for name, by_tolerance in score_real(args.split, work / "real").items():
            for tolerance, result in by_tolerance.items():
                prefix = f"onsets.real.{args.split}.{name}@{int(round(tolerance * 1000))}ms"
                print(f"METRIC {prefix} P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} "
                      f"timing_median_ms={result['timing_median_ms']:.1f} timing_p95_ms={result['timing_p95_ms']:.1f} "
                      f"tp={result['tp']} fp={result['fp']} fn={result['fn']}")


if __name__ == "__main__":
    main()
