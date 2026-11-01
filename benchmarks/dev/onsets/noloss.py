"""No-loss check for ``audio_onsets``: every planted start a baseline detector (main) finds, this tree must find too.

Run through ``score.py --set noloss``. The baseline is ``src/media/audio.py`` at
``--baseline-ref`` (default origin/main, read by ``export_ref`` with git show),
or the file given by ``--baseline-audio`` (use that on a box with no git).

Families (``--family``, default all):
- legacy: the 60 original clips (score.py) at the band and broadband settings;
- real: the 156 real-world-like clips (realistic.py) at their settings;
- short: 1-5 ms clicks (decaying sines and square bursts, peaks -30 to -6 dBFS)
  over silence, white and pink noise at -70 to -40 dBFS RMS;
- pair: two starts 20-60 ms apart (short clicks or decaying tones) over the same beds;
- ranged: single starts (loud clicks, quiet beeps 25 dB over steady noise, moderate
  beeps) analyzed through start_sec/end_sec windows 0.1-1 s long around them.
The generated families are scored at three settings: click (band [1000, 4000], 0.36),
beep (band f +-100 Hz around the event frequency, 0.7) and broadband (0.12).

A loss is a planted start the baseline matches (one-to-one, within the setting's
tolerance) and this tree does not. Two kinds are listed apart as allowed: losses on
babble beds (speech starts a level-free detector hits by chance), and chance hits,
where the baseline also fires within the tolerance on the same clip's bed alone, without
the event. Bed-only clips have no planted starts. Losses of a start with another planted
start closer than audio_onsets' default min_gap_sec still count as losses; they are marked
"within-gap" and counted apart (``within_gap``), because the default gap merges such
starts whenever both are timed within it.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import realistic  # noqa: E402
import score  # noqa: E402

RATE = 48000
SEED = 20261101
MIN_GAP_SEC = 0.035  # audio_onsets' default min_gap_sec
CLICK_BAND = [1000.0, 4000.0]
TOLERANCE = {"click": 0.05, "beep": 0.1, "broadband": 0.03, "band": 0.03, "mixed": 0.03,
             "quiet_click": 0.05, "quiet_beep": 0.1, "quiet_broadband": 0.03}
BED_KINDS = ("silence", "white", "pink")

# (family, clip id, bed, samples, bed-only samples, truth, setting, band_hz, min_strength, start_sec, end_sec)
Item = Tuple[str, str, str, np.ndarray, np.ndarray, List[float], str, Optional[List[float]], float, Optional[float], Optional[float]]


def _bed(kind: str, count: int, level_db: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "silence":
        return np.zeros(count)
    noise = rng.standard_normal(count)
    if kind == "pink":
        spectrum = np.fft.rfft(noise)
        frequencies = np.fft.rfftfreq(count, 1.0 / RATE)
        frequencies[0] = 1.0
        noise = np.fft.irfft(spectrum / np.sqrt(frequencies), count)
    return noise / float(np.std(noise)) * 10.0 ** (level_db / 20.0)


def _short_click(rng: np.random.Generator, frequency: float, peak_db: float, shape: str) -> np.ndarray:
    local = np.arange(int(rng.uniform(0.001, 0.005) * RATE)) / RATE
    peak = 10.0 ** (peak_db / 20.0)
    if shape == "square":
        return peak * np.sign(np.sin(2 * np.pi * frequency * local + 1e-9))
    return peak * np.sin(2 * np.pi * frequency * local) * np.exp(-local / (0.3 * local[-1] + 1e-9))


def _decaying_tone(rng: np.random.Generator, frequency: float, peak_db: float) -> np.ndarray:
    local = np.arange(int(rng.uniform(0.01, 0.05) * RATE)) / RATE
    return 10.0 ** (peak_db / 20.0) * np.sin(2 * np.pi * frequency * local) * np.exp(-local / 0.01)


def _settings(frequency: float) -> List[Tuple[str, Optional[List[float]], float]]:
    return [("click", CLICK_BAND, 0.36), ("beep", [frequency - 100.0, frequency + 100.0], 0.7), ("broadband", None, 0.12)]


def _generated(family: str) -> Iterator[Item]:
    count = int(4.0 * RATE)
    for index in range(24):
        rng = np.random.default_rng(SEED + 100 * index + {"short": 0, "pair": 1, "ranged": 2}[family])
        bed_kind = BED_KINDS[index % 3]
        signal = _bed(bed_kind, count if family != "ranged" else int(6.0 * RATE), rng.uniform(-70.0, -40.0), rng)
        bed_signal = signal.copy()
        frequency = float(rng.uniform(1100.0, 3400.0))
        truth: List[float] = []
        clip = f"{family}{index:02d}"
        if family == "short":
            shape = ("decay", "square")[(index // 3) % 2]
            position = rng.uniform(0.4, 0.7)
            while position < 3.6:
                wave = _short_click(rng, frequency, rng.uniform(-30.0, -6.0), shape)
                first = int(position * RATE)
                signal[first:first + len(wave)] += wave
                truth.append(first / RATE)
                position += rng.uniform(0.3, 0.8)
            clip += f"-{shape}"
        elif family == "pair":
            position = rng.uniform(0.4, 0.7)
            while position < 3.5:
                gap = rng.uniform(0.020, 0.060)
                for offset in (0.0, gap):
                    peak = rng.uniform(-20.0, -6.0)
                    wave = _short_click(rng, frequency, peak, "decay") if index % 2 else _decaying_tone(rng, frequency, peak)
                    first = int((position + offset) * RATE)
                    signal[first:first + len(wave)] += wave
                    truth.append(first / RATE)
                position += rng.uniform(0.4, 0.8)
        else:
            kind = ("loud_click", "quiet_beep", "beep")[(index // 3) % 3]
            if kind == "quiet_beep" and bed_kind == "silence":
                bed_kind = "white"
            if kind == "quiet_beep":
                bed_db = rng.uniform(-75.0, -55.0)
                signal = _bed(bed_kind, int(6.0 * RATE), bed_db, rng)
                bed_signal = signal.copy()
            for position in (1.0, 2.5, 4.0, 5.2):
                if kind == "loud_click":
                    wave = _short_click(rng, frequency, rng.uniform(-20.0, -6.0), "decay")
                else:
                    level = bed_db + 25.0 if kind == "quiet_beep" else rng.uniform(-30.0, -12.0)
                    local = np.arange(int(0.1 * RATE)) / RATE
                    wave = np.sqrt(2) * 10.0 ** (level / 20.0) * np.sin(2 * np.pi * frequency * local)
                first = int(position * RATE)
                signal[first:first + len(wave)] += wave
                truth.append(first / RATE)
            clip += f"-{kind}"
        samples = np.clip(signal, -0.95, 0.95).astype(np.float32)
        bed_only = np.clip(bed_signal, -0.95, 0.95).astype(np.float32)
        for setting, band, strength in _settings(frequency):
            if family != "ranged":
                yield family, clip, bed_kind, samples, bed_only, truth, setting, band, strength, None, None
                continue
            for start in truth:
                length = rng.uniform(0.1, 1.0)
                lead = rng.uniform(0.02, 0.8 * length)
                yield family, clip, bed_kind, samples, bed_only, [start], setting, band, strength, start - lead, min(5.99, start - lead + length)


def _legacy() -> Iterator[Item]:
    for entry in score.clip_plan():
        samples, truth = score.make_clip(entry)
        bed_only = np.clip(score._bed(entry["bed"], len(samples), np.random.default_rng(entry["seed"])), -0.95, 0.95).astype(np.float32)
        for setting, band, strength in (("band", CLICK_BAND, 0.36), ("broadband", None, 0.12)):
            yield "legacy", entry["id"], entry["bed"], samples, bed_only, truth, setting, band, strength, None, None


def _real() -> Iterator[Item]:
    for entry in realistic.plan():
        samples, truth, _ = realistic.make_clip(entry, score._event_wave)
        bed_only = np.clip(realistic._bed_signal(entry, len(samples))[0], -0.95, 0.95).astype(np.float32)
        for setting, band, strength in realistic.settings(entry):
            yield "real", entry["id"], entry["bed"], samples, bed_only, truth, setting, band, strength, None, None


def export_ref(ref: str, dest: Path) -> Path:
    """Write src/media/audio.py as it is at a git ref (e.g. origin/main) to dest."""
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:src/media/audio.py"], check=True, capture_output=True).stdout
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(text)
    return dest


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(baseline_audio: Path, families: List[str], work: Path) -> Dict[str, Any]:
    """Losses and counts per family and setting; prints one line per loss and a summary."""
    from src.core.cache import MediaCache
    from src.media import audio as current

    baseline = _load(baseline_audio, "baseline_audio")
    work.mkdir(parents=True, exist_ok=True)
    # Detector results are cached per file; key each cache folder by the detector's own bytes so a reused --work
    # never serves results from another baseline ref or another edit of this tree.
    caches = {name: MediaCache(work / f"cache-{name}-{score.detector_digest(path)}")
              for name, path in (("baseline", baseline_audio), ("current", Path(current.__file__)))}
    sources: Dict[str, Callable[[], Iterator[Item]]] = {
        "legacy": _legacy, "real": _real, "short": lambda: _generated("short"),
        "pair": lambda: _generated("pair"), "ranged": lambda: _generated("ranged")}
    summary: Dict[str, Dict[str, int]] = {}
    written: Dict[str, Path] = {}
    for family in families:
        for _, clip, bed, samples, bed_only, truth, setting, band, strength, start, end in sources[family]():
            path = written.get(f"{family}/{clip}")
            if path is None:
                path = written[f"{family}/{clip}"] = score.write_wav(work / f"{family}-{clip}.wav", samples)
            found = {}
            for name, module in (("baseline", baseline), ("current", current)):
                result = module.audio_onsets(str(path), start_sec=start, end_sec=end, min_strength=strength,
                                             band_hz=band, cache=caches[name])
                found[name] = [float(item["time_sec"]) for item in result["onsets"]]
            tolerance = TOLERANCE[setting]
            matched = {name: _matched(truth, times, tolerance) for name, times in found.items()}
            lost = sorted(matched["baseline"] - matched["current"])
            key = f"{family}/{setting}"
            row = summary.setdefault(key, {"events": 0, "baseline": 0, "current": 0, "lost": 0, "lost_babble": 0, "lost_chance": 0, "lost_within_gap": 0})
            row["events"] += len(truth)
            row["baseline"] += len(matched["baseline"])
            row["current"] += len(matched["current"])
            chance: set = set()
            if lost and bed != "babble":
                # A baseline hit is chance when the baseline also fires there on the bed alone, without the event.
                bed_path = written.get(f"{family}/{clip}/bed")
                if bed_path is None:
                    bed_path = written[f"{family}/{clip}/bed"] = score.write_wav(work / f"{family}-{clip}-bed.wav", bed_only)
                bed_times = [float(item["time_sec"]) for item in baseline.audio_onsets(
                    str(bed_path), start_sec=start, end_sec=end, min_strength=strength, band_hz=band, cache=caches["baseline"])["onsets"]]
                chance = {event for event in lost if any(abs(hit - truth[event]) <= tolerance for hit in bed_times)}
            for event in lost:
                reason = "allowed (babble bed) " if bed == "babble" else "allowed (chance: baseline fires there on the bed alone) " if event in chance else ""
                if not reason and any(0 < abs(other - truth[event]) < MIN_GAP_SEC for other in truth):
                    row["lost_within_gap"] += 1
                    reason = "within-gap (another planted start lies closer than the default min_gap_sec) "
                row["lost_babble" if bed == "babble" else "lost_chance" if event in chance else "lost"] += 1
                window = "" if start is None else f" range {start:.3f}-{end:.3f}"
                print(f"LOSS {reason}{key} {clip} bed={bed} t={truth[event]:.4f}{window} "
                      f"baseline={_nearest(found['baseline'], truth[event])} current={_nearest(found['current'], truth[event])}")
    for key, row in sorted(summary.items()):
        print(f"NOLOSS {key} events={row['events']} baseline_matched={row['baseline']} current_matched={row['current']} "
              f"lost={row['lost']} (within_gap={row['lost_within_gap']}) lost_chance={row['lost_chance']} lost_babble={row['lost_babble']}")
    print(f"NOLOSS total lost={sum(row['lost'] for row in summary.values())} "
          f"(within_gap={sum(row['lost_within_gap'] for row in summary.values())}) "
          f"lost_chance={sum(row['lost_chance'] for row in summary.values())} lost_babble={sum(row['lost_babble'] for row in summary.values())}")
    return summary


def _matched(truth: List[float], detected: List[float], tolerance: float) -> set:
    """Indexes of truth events in a maximum one-to-one matching within tolerance.

    Greedy closest-pair matching can take a hit that a second event also needs and
    count fewer matches than exist, which would hide a loss. Augmenting paths find the maximum.
    """
    options = [[d for d, hit in enumerate(detected) if abs(hit - target) <= tolerance] for target in truth]
    owner: Dict[int, int] = {}

    def assign(t: int, seen: set) -> bool:
        for d in options[t]:
            if d not in seen:
                seen.add(d)
                if d not in owner or assign(owner[d], seen):
                    owner[d] = t
                    return True
        return False

    return {t for t in range(len(truth)) if assign(t, set())}


def _nearest(times: List[float], target: float) -> str:
    return "none" if not times else f"{min(times, key=lambda value: abs(value - target)):.4f}"
