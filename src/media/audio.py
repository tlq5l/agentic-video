"""Non-speech audio measurements and timestamped spectrograms.

Audio is decoded by FFmpeg to content-addressed mono float32 PCM.  The
measurements use a 48 kHz clock and a 256-sample hop (5.333 ms), so reported
onset times are quantized to that resolution rather than inferred by ear.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.errors import (
    DependencyNotFoundError,
    ExtractionBoundsError,
    MediaCorruptError,
    MediaNotFoundError,
    SubprocessTimeoutError,
)
from src.core.types import format_timestamp


SAMPLE_RATE = 48000
HOP_SAMPLES = 256
N_FFT = 2048
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("sub", 0.0, 120.0),
    ("low_mid", 120.0, 500.0),
    ("mid", 500.0, 4000.0),
    ("high", 4000.0, 10000.0),
    ("air", 10000.0, SAMPLE_RATE / 2.0 + 1.0),
)
_PCM_NAME = "pcm_mono_48000_f32_v1.f32"
_PCM_META_NAME = "pcm_mono_48000_f32_v1_meta"
_DECODE_LOCKS: Dict[str, threading.Lock] = {}
_DECODE_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _DECODE_LOCKS_GUARD:
        return _DECODE_LOCKS.setdefault(key, threading.Lock())


def _cache_tag(prefix: str, options: Dict[str, Any]) -> str:
    payload = json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return "{}_v1_{}".format(prefix, digest)


def _decode_pcm(source: str, cache: Optional[MediaCache] = None) -> Tuple[np.ndarray, str, MediaCache]:
    """Return a read-only memory-mapped cache of FFmpeg-decoded mono PCM."""
    path = Path(source).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise MediaNotFoundError("Audio source file not found", {"path": str(path)})
    cache = cache or MediaCache()
    fingerprint = cache.compute_fingerprint(str(path))
    audio_dir = cache.get_audio_dir(fingerprint)
    pcm_path = audio_dir / _PCM_NAME
    meta = cache.load_json(fingerprint, _PCM_META_NAME)
    expected_bytes = None
    if isinstance(meta, dict) and meta.get("sample_rate") == SAMPLE_RATE:
        try:
            expected_bytes = int(meta.get("sample_count", -1)) * 4
        except (TypeError, ValueError):
            expected_bytes = None
    if not pcm_path.exists() or expected_bytes is None or pcm_path.stat().st_size != expected_bytes:
        lock = _lock_for(fingerprint)
        with lock:
            meta = cache.load_json(fingerprint, _PCM_META_NAME)
            expected_bytes = None
            if isinstance(meta, dict) and meta.get("sample_rate") == SAMPLE_RATE:
                try:
                    expected_bytes = int(meta.get("sample_count", -1)) * 4
                except (TypeError, ValueError):
                    expected_bytes = None
            if not pcm_path.exists() or expected_bytes is None or pcm_path.stat().st_size != expected_bytes:
                temporary = audio_dir / ".{}.{}.{}.tmp".format(
                    _PCM_NAME, os.getpid(), uuid.uuid4().hex[:8]
                )
                command = [
                    "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
                    "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-acodec", "pcm_f32le", "-f", "f32le", str(temporary),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=max(Config.SUBPROCESS_TIMEOUT_SEC * 4, 180.0),
                        check=False,
                    )
                except FileNotFoundError as exc:
                    raise DependencyNotFoundError(
                        "ffmpeg binary not found on system PATH", {"dependency": "ffmpeg"}
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                    raise SubprocessTimeoutError(
                        "Audio decoding timed out", {"source": str(path)}
                    ) from exc
                if completed.returncode != 0 or not temporary.exists():
                    stderr = completed.stderr.strip() if completed.stderr else "no decodable audio stream"
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                    raise MediaCorruptError(
                        "FFmpeg audio decode failed: {}; the file may have no audio track (probe_video reports has_audio)".format(stderr),
                        {"source": str(path), "stderr": stderr},
                    )
                if temporary.stat().st_size == 0 or temporary.stat().st_size % 4:
                    size = temporary.stat().st_size
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                    raise MediaCorruptError(
                        "FFmpeg returned empty or malformed PCM audio",
                        {"source": str(path), "byte_length": size},
                    )
                os.replace(str(temporary), str(pcm_path))
                sample_count = pcm_path.stat().st_size // 4
                cache.save_json(fingerprint, _PCM_META_NAME, {
                    "sample_rate": SAMPLE_RATE,
                    "sample_count": sample_count,
                    "source_path": str(path),
                    "decoder": "ffmpeg-pcm_f32le-v1",
                })

    sample_count = pcm_path.stat().st_size // 4
    if sample_count < 1:
        raise MediaCorruptError("Audio source contains no samples", {"source": str(path)})
    samples = np.memmap(str(pcm_path), dtype=np.float32, mode="r", shape=(sample_count,))
    return samples, fingerprint, cache


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExtractionBoundsError("{} must be numeric".format(name), {name: value}) from exc
    if not math.isfinite(number):
        raise ExtractionBoundsError("{} must be finite".format(name), {name: value})
    return number


def _range(samples: np.ndarray, start_sec: Optional[float], end_sec: Optional[float]) -> Tuple[float, float, int, int]:
    duration = len(samples) / float(SAMPLE_RATE)
    start = 0.0 if start_sec is None else _finite_number(start_sec, "start_sec")
    end = duration if end_sec is None else _finite_number(end_sec, "end_sec")
    if start < 0.0 or start >= duration or end <= start or end > duration + 0.001:
        raise ExtractionBoundsError(
            "Audio window must satisfy 0 <= start_sec < end_sec <= audio duration (audio_duration_sec in details); omit start_sec/end_sec to analyze the whole track",
            {"start_sec": start, "end_sec": end, "audio_duration_sec": duration},
        )
    end = min(end, duration)
    return start, end, max(0, int(round(start * SAMPLE_RATE))), min(len(samples), int(round(end * SAMPLE_RATE)))


def _db(power: float, floor_db: float = -120.0) -> float:
    if not math.isfinite(power) or power <= 10.0 ** (floor_db / 10.0):
        return floor_db
    return max(floor_db, 10.0 * math.log10(power))


@functools.lru_cache(maxsize=32)
def _hann(length: int) -> np.ndarray:
    """Read-only Hann window (float64); the same lengths recur for every onset."""
    window = np.hanning(length)
    window.setflags(write=False)
    return window


@functools.lru_cache(maxsize=32)
def _rfft_frequencies(size: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq(size, d=1.0 / SAMPLE_RATE)
    frequencies.setflags(write=False)
    return frequencies


@functools.lru_cache(maxsize=64)
def _hz_mask(size: int, low_hz: float, high_hz: float) -> np.ndarray:
    frequencies = _rfft_frequencies(size)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    mask.setflags(write=False)
    return mask


@functools.lru_cache(maxsize=32)
def _named_band_masks(size: int) -> Tuple[Tuple[str, np.ndarray], ...]:
    frequencies = _rfft_frequencies(size)
    return tuple((name, (frequencies >= lower) & (frequencies < upper)) for name, lower, upper in BANDS)


def _band_profile(segment: np.ndarray, fft_size: Optional[int] = None) -> Dict[str, float]:
    if len(segment) == 0:
        return {name: -120.0 for name, _, _ in BANDS}
    if fft_size is None:
        size = 1 << int(math.ceil(math.log2(max(32, len(segment)))))
    else:
        size = max(32, int(fft_size))
    window = _hann(len(segment))
    values = np.asarray(segment, dtype=np.float64)
    transformed = np.fft.rfft(values * window, n=size)
    power = (np.abs(transformed) ** 2) / max(1.0, float(np.sum(window) ** 2))
    if len(power) > 2:
        power[1:-1] *= 2.0
    profile: Dict[str, float] = {}
    for name, mask in _named_band_masks(size):
        profile[name] = round(_db(float(np.sum(power[mask]))), 3)
    return profile


def _normalize_band_hz(band_hz: Any) -> Optional[Dict[str, Any]]:
    """Validate a named band or an explicit [low, high] frequency range."""
    if band_hz is None:
        return None
    if isinstance(band_hz, str):
        matches = [(name, low, high) for name, low, high in BANDS if name == band_hz]
        if not matches:
            raise ExtractionBoundsError(
                "band_hz must be one of sub, low_mid, mid, high, air, or [low_hz, high_hz]",
                {"band_hz": band_hz},
            )
        name, low, high = matches[0]
        return {"name": name, "low_hz": float(low), "high_hz": float(high)}
    if not isinstance(band_hz, (list, tuple)) or len(band_hz) != 2:
        raise ExtractionBoundsError("band_hz must be a named band or [low_hz, high_hz]", {"band_hz": band_hz})
    low = _finite_number(band_hz[0], "band_hz[0]")
    high = _finite_number(band_hz[1], "band_hz[1]")
    if low < 0.0 or high <= low or high > SAMPLE_RATE / 2.0:
        raise ExtractionBoundsError(
            "band_hz must satisfy 0 <= low_hz < high_hz <= Nyquist",
            {"low_hz": low, "high_hz": high, "nyquist_hz": SAMPLE_RATE / 2.0},
        )
    return {"name": None, "low_hz": low, "high_hz": high}


def _band_spec(band: Optional[Dict[str, Any]]) -> Any:
    if band is None:
        return None
    if band["name"] is not None:
        return band["name"]
    return [band["low_hz"], band["high_hz"]]


def _band_energy_db(segment: np.ndarray, band: Dict[str, Any]) -> float:
    if len(segment) == 0:
        return -120.0
    values = np.asarray(segment, dtype=np.float64)
    size = 1 << int(math.ceil(math.log2(max(32, len(values)))))
    window = _hann(len(values))
    transformed = np.fft.rfft(values * window, n=size)
    power = np.abs(transformed) ** 2 / max(1.0, float(np.sum(window * window)))
    if len(power) > 2:
        power[1:-1] *= 2.0
    mask = _hz_mask(size, band["low_hz"], band["high_hz"])
    return round(_db(float(np.sum(power[mask]))), 3)


def _stft_flux(
    samples: np.ndarray,
    band: Optional[Dict[str, Any]] = None,
    fft_size: int = N_FFT,
) -> np.ndarray:
    """Normalized positive spectral flux, optionally limited to one frequency band."""
    if len(samples) < 2:
        return np.zeros(1, dtype=np.float32)
    fft_size = max(256, int(fft_size))
    half = fft_size // 2
    padded = np.pad(np.asarray(samples, dtype=np.float32), (half, fft_size - half), mode="constant")
    frame_count = int(math.ceil(len(samples) / float(HOP_SAMPLES)))
    flux = np.zeros(frame_count, dtype=np.float32)
    window = np.hanning(fft_size).astype(np.float32)
    if band is None:
        band_mask = None
    else:
        frequencies = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLE_RATE)
        band_mask = (frequencies >= band["low_hz"]) & (frequencies < band["high_hz"])
        if not np.any(band_mask):
            raise ExtractionBoundsError("band_hz is too narrow for the analysis FFT", {"band_hz": band})
    previous = None
    frames_per_chunk = int(10.0 * SAMPLE_RATE / HOP_SAMPLES)
    for first in range(0, frame_count, frames_per_chunk):
        count = min(frames_per_chunk, frame_count - first)
        view_start = first * HOP_SAMPLES
        view_end = (first + count - 1) * HOP_SAMPLES + fft_size
        view = np.lib.stride_tricks.sliding_window_view(padded[view_start:view_end], fft_size)[::HOP_SAMPLES]
        magnitudes = np.abs(np.fft.rfft(view * window, axis=1)).astype(np.float32)
        if band_mask is not None:
            magnitudes = magnitudes[:, band_mask]
        totals = np.sum(magnitudes, axis=1, keepdims=True)
        normalized = magnitudes / np.maximum(totals, 1e-12)
        if previous is not None:
            flux[first] = np.maximum(normalized[0] - previous, 0.0).sum()
        if count > 1:
            flux[first + 1:first + count] = np.maximum(normalized[1:] - normalized[:-1], 0.0).sum(axis=1)
        previous = normalized[-1]
    return flux


def _onset_spectra(
    samples: np.ndarray, spans: Sequence[Tuple[int, int]], band: Optional[Dict[str, Any]],
) -> List[Tuple[Dict[str, float], Optional[float]]]:
    """`_band_profile` (and `_band_energy_db` for a band) of many sample spans.

    Equal-length spans share one batched FFT; the per-span formulas are those
    of `_band_profile` and `_band_energy_db`.
    """
    results: List[Optional[Tuple[Dict[str, float], Optional[float]]]] = [None] * len(spans)
    by_length: Dict[int, List[int]] = {}
    for position, (first, last) in enumerate(spans):
        by_length.setdefault(last - first, []).append(position)
    for length, positions in by_length.items():
        size = 1 << int(math.ceil(math.log2(max(32, length))))
        window = _hann(length)
        profile_denominator = max(1.0, float(np.sum(window) ** 2))
        energy_denominator = max(1.0, float(np.sum(window * window)))
        masks = _named_band_masks(size)
        band_mask = None if band is None else _hz_mask(size, band["low_hz"], band["high_hz"])
        for chunk_start in range(0, len(positions), 512):
            chunk = positions[chunk_start:chunk_start + 512]
            values = np.stack([np.asarray(samples[spans[p][0]:spans[p][1]], dtype=np.float64) for p in chunk])
            squared = np.abs(np.fft.rfft(values * window, n=size, axis=1)) ** 2
            profile_power = squared / profile_denominator
            if profile_power.shape[1] > 2:
                profile_power[:, 1:-1] *= 2.0
            band_sums = {name: np.sum(profile_power[:, mask], axis=1) for name, mask in masks}
            if band_mask is not None:
                energy_power = squared / energy_denominator
                if energy_power.shape[1] > 2:
                    energy_power[:, 1:-1] *= 2.0
                energy_sums = np.sum(energy_power[:, band_mask], axis=1)
            for row, position in enumerate(chunk):
                profile = {name: round(_db(float(band_sums[name][row])), 3) for name, _ in masks}
                energy = None if band_mask is None else round(_db(float(energy_sums[row])), 3)
                results[position] = (profile, energy)
    return results  # type: ignore[return-value]


def _detect_onsets_in_samples(
    samples: np.ndarray,
    start: float,
    end: float,
    first_sample: int,
    last_sample: int,
    threshold: float,
    band: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context_first = max(0, first_sample - N_FFT)
    context_last = min(len(samples), last_sample + N_FFT)
    local = np.asarray(samples[context_first:context_last], dtype=np.float32)
    # Higher bands have enough bin density at 1024 samples and benefit from
    # a shorter window when pitched attacks sit only a few milliseconds apart.
    fft_size = 1024 if band is not None and band["low_hz"] >= 500.0 else N_FFT
    envelope = _stft_flux(local, band, fft_size=fft_size)
    peak_indices: List[int] = []
    for index in range(1, max(1, len(envelope) - 1)):
        if envelope[index] >= threshold and envelope[index] > envelope[index - 1] and envelope[index] >= envelope[index + 1]:
            peak_indices.append(index)
    if len(envelope) == 1 and envelope[0] >= threshold:
        peak_indices.append(0)

    # Debounce multiple spectral peaks from one attack, keeping the strongest.
    ranked = sorted(peak_indices, key=lambda idx: (-float(envelope[idx]), idx))
    selected: List[int] = []
    debounce_sec = 0.010 if band is not None else 0.030
    min_gap_frames = max(1, int(round(debounce_sec * SAMPLE_RATE / HOP_SAMPLES)))
    blocked = np.zeros(len(envelope), dtype=bool)
    for index in ranked:
        if not blocked[index]:
            selected.append(index)
            blocked[max(0, index - min_gap_frames + 1):index + min_gap_frames] = True
    selected.sort()

    onsets: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for index in selected:
        # A centered 2048-sample window first sees an attack as it enters its
        # right edge, half a window before the attack time. Correct that known
        # STFT lead; the remaining quantization is one 256-sample hop.
        # Band-limited flux peaks a little later for short pitched bursts than
        # broadband attacks. One hop corrects that band-specific window lag.
        bias = fft_size // 2 - (HOP_SAMPLES if band is not None else 0)
        time_sec = context_first / float(SAMPLE_RATE) + (index * HOP_SAMPLES + bias) / float(SAMPLE_RATE)
        if time_sec < start - 1e-6 or time_sec > end + 1e-6:
            continue
        sample_index = min(len(samples) - 1, max(0, int(round(time_sec * SAMPLE_RATE))))
        profile_end = min(len(samples), sample_index + int(0.085 * SAMPLE_RATE))
        activity = np.asarray(samples[sample_index:profile_end], dtype=np.float64)
        activity_rms = float(np.sqrt(np.mean(activity * activity))) if len(activity) else 0.0
        if activity_rms < 1e-4:
            # A normalized spectrum can show a large flux when codec noise
            # changes shape at a silence boundary. Do not call that a sound.
            continue
        onsets.append({
            "time_sec": round(time_sec, 6),
            "strength": round(float(envelope[index]), 6),
        })
        spans.append((sample_index, profile_end))
    spectra = _onset_spectra(samples, spans, band)
    for onset, (profile, band_energy) in zip(onsets, spectra):
        onset["band_db"] = profile
        if band_energy is not None:
            onset["band_energy_db"] = band_energy
    return onsets


def audio_onsets(
    source: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    min_strength: float = 0.12,
    cache: Optional[MediaCache] = None,
    band_hz: Any = None,
    compare_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect attacks in broadband or in a named/custom band; compare sources by in-band energy."""
    threshold = _finite_number(min_strength, "min_strength")
    if threshold < 0.0 or threshold > 1.0:
        raise ExtractionBoundsError("min_strength must satisfy 0 <= value <= 1 (default 0.12; raise it to keep only stronger onsets)", {"min_strength": threshold})
    band = _normalize_band_hz(band_hz)
    samples, fingerprint, cache = _decode_pcm(source, cache)
    start, end, first_sample, last_sample = _range(samples, start_sec, end_sec)
    compare_samples = None
    compare_fingerprint = None
    if compare_to is not None:
        compare_samples, compare_fingerprint, _ = _decode_pcm(compare_to, cache)
        _range(compare_samples, start, end)
    options = {
        "start_sec": round(start, 6), "end_sec": round(end, 6),
        "min_strength": round(threshold, 6), "band_hz": band,
        "compare_fingerprint": compare_fingerprint,
    }
    result_key = _cache_tag("audio_onsets_v7", options)
    cached = cache.load_json(fingerprint, result_key)
    if isinstance(cached, dict):
        return cached

    onsets = _detect_onsets_in_samples(samples, start, end, first_sample, last_sample, threshold, band)
    if compare_samples is not None:
        energy_band = band or {"name": "full", "low_hz": 0.0, "high_hz": SAMPLE_RATE / 2.0}
        kept: List[Dict[str, Any]] = []
        for onset in onsets:
            onset_time = float(onset["time_sec"])
            energy_start = max(0, int(round((onset_time - 0.010) * SAMPLE_RATE)))
            energy_end = min(len(samples), energy_start + int(round(0.030 * SAMPLE_RATE)))
            source_energy = _band_energy_db(samples[energy_start:energy_end], energy_band)
            baseline_energy = _band_energy_db(compare_samples[energy_start:min(len(compare_samples), energy_end)], energy_band)
            delta = round(source_energy - baseline_energy, 3)
            # Require a measurable energy increase even when the baseline
            # detector missed a weak event or its flux peak fell one hop away.
            if delta >= 2.5:
                onset["band_energy_db"] = source_energy
                onset["compare_band_energy_db"] = baseline_energy
                onset["compare_delta_db"] = delta
                kept.append(onset)
        # A single new pitched attack can produce neighboring flux maxima.
        # Keep the strongest energy-difference peak from each 30 ms cluster.
        kept_ranked = sorted(kept, key=lambda item: (-item["compare_delta_db"], item["time_sec"]))
        selected = []
        for onset in kept_ranked:
            if all(abs(onset["time_sec"] - other["time_sec"]) > 0.030 for other in selected):
                selected.append(onset)
        onsets = sorted(selected, key=lambda item: item["time_sec"])

    result = {
        "source_path": str(Path(source).expanduser().resolve()),
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "sample_rate": SAMPLE_RATE,
        "hop_samples": HOP_SAMPLES,
        "time_resolution_ms": round(HOP_SAMPLES * 1000.0 / SAMPLE_RATE, 3),
        "min_strength": round(threshold, 6),
        "band_hz": band,
        "compare_to": None if compare_to is None else str(Path(compare_to).expanduser().resolve()),
        "compare_min_delta_db": None if compare_to is None else 2.5,
        "onsets": onsets,
        "total": len(onsets),
    }
    cache.save_json(fingerprint, result_key, result)
    return result


def _measure_window(
    segment: np.ndarray, start_sec: float, end_sec: float, channel_peaks: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure one window of the mono mix. `channel_peaks` (per-sample loudest
    channel) sets peak_dbfs; without it the mono mix is used, which reads up to
    3 dB hot on stereo because FFmpeg's downmix is (L+R)/sqrt(2)."""
    if len(segment) == 0:
        return {
            "start_sec": round(start_sec, 6), "end_sec": round(end_sec, 6),
            "rms_dbfs": -120.0, "peak_dbfs": -120.0,
            "band_db": {name: -120.0 for name, _, _ in BANDS},
            "spectral_centroid_hz": 0.0, "silent": True,
        }
    values = np.asarray(segment, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values * values)))
    if channel_peaks is not None and len(channel_peaks):
        peak = float(np.max(channel_peaks))
    else:
        peak = float(np.max(np.abs(values)))
    size = 1 << int(math.ceil(math.log2(max(32, len(values)))))
    window = np.hanning(len(values))
    transformed = np.fft.rfft(values * window, n=size)
    power = np.abs(transformed) ** 2
    frequencies = np.fft.rfftfreq(size, d=1.0 / SAMPLE_RATE)
    if len(power) > 2:
        power[1:-1] *= 2.0
    total_power = float(np.sum(power))
    centroid = float(np.sum(frequencies * power) / total_power) if total_power > 0.0 else 0.0
    profile: Dict[str, float] = {}
    for name, lower, upper in BANDS:
        mask = (frequencies >= lower) & (frequencies < upper)
        window_power = max(1.0, float(np.sum(window * window)))
        profile[name] = round(_db(float(np.sum(power[mask]) / (max(1, len(values)) * window_power))), 3)
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-6)), 3)
    return {
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-6)), 3),
        "band_db": profile,
        "spectral_centroid_hz": round(centroid, 3),
        "silent": rms_dbfs < -60.0,
    }


def _silence_gaps(samples: np.ndarray, start_sec: float, end_sec: float, threshold_dbfs: float = -60.0) -> List[Dict[str, float]]:
    chunk = int(round(0.01 * SAMPLE_RATE))
    quiet: List[bool] = []
    for offset in range(0, len(samples), chunk):
        segment = np.asarray(samples[offset:offset + chunk], dtype=np.float64)
        rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0
        quiet.append(20.0 * math.log10(max(rms, 1e-12)) < threshold_dbfs)
    min_chunks = 3
    gaps: List[Dict[str, float]] = []
    index = 0
    while index < len(quiet):
        if not quiet[index]:
            index += 1
            continue
        first = index
        while index < len(quiet) and quiet[index]:
            index += 1
        if index - first >= min_chunks:
            gap_start = start_sec + first * chunk / float(SAMPLE_RATE)
            gap_end = min(end_sec, start_sec + index * chunk / float(SAMPLE_RATE))
            gaps.append({
                "start_sec": round(gap_start, 6),
                "end_sec": round(gap_end, 6),
                "duration_sec": round(gap_end - gap_start, 6),
                "threshold_dbfs": threshold_dbfs,
            })
    return gaps


# A boost in one band is fault-level only from this size: arrangement changes
# (a bass line entering, a layer dropping out) routinely move one band by
# 6-8 dB. Smaller boosts count only when several bands rise together.
_STEP_FAULT_MIN_DB = 8.0
_BROAD_STEP_MIN_BANDS = 3


def _outlier_sections(
    windows: List[Dict[str, Any]],
    threshold_db: float = 6.0,
    ignore_ranges: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Group band departures and distinguish flat steps from ramps and bursts."""
    if len(windows) < 2:
        return []
    sections: List[Dict[str, Any]] = []
    for band, _, _ in BANDS:
        candidates: Dict[str, List[int]] = {"boost": [], "cut": []}
        baselines: Dict[int, float] = {}
        deltas: Dict[int, float] = {}
        energies_all = [float(window["band_db"][band]) for window in windows]
        # Leave-one-out median of the non-silent, audible windows, computed
        # from one sorted copy: dropping the window at sorted rank r shifts the
        # middle element(s) by at most one place.
        usable = [
            index for index, window in enumerate(windows)
            if not window["silent"] and energies_all[index] > -90.0
        ]
        usable_values = np.asarray([energies_all[index] for index in usable], dtype=np.float64)
        order = np.argsort(usable_values, kind="stable")
        sorted_values = usable_values[order]
        rank_of = {usable[int(position)]: rank for rank, position in enumerate(order)}
        count = len(usable)

        def _median_without(rank: Optional[int]) -> Optional[float]:
            remaining = count - (0 if rank is None else 1)
            if remaining < 1:
                return None

            def at(position: int) -> float:
                if rank is not None and position >= rank:
                    position += 1
                return float(sorted_values[position])

            if remaining % 2:
                return at(remaining // 2)
            return float(np.mean([at(remaining // 2 - 1), at(remaining // 2)]))

        for index in range(len(windows)):
            energy = energies_all[index]
            if energy <= -90.0:
                continue
            baseline = _median_without(rank_of.get(index))
            if baseline is None:
                continue
            delta = energy - baseline
            baselines[index] = baseline
            deltas[index] = delta
            if delta >= threshold_db:
                candidates["boost"].append(index)
            elif delta <= -threshold_db:
                candidates["cut"].append(index)

        for direction, indices in candidates.items():
            if not indices:
                continue
            runs: List[List[int]] = []
            for index in indices:
                if runs and index == runs[-1][-1] + 1:
                    runs[-1].append(index)
                else:
                    runs.append([index])
            direction_sign = 1.0 if direction == "boost" else -1.0
            for run in runs:
                first, last = run[0], run[-1]
                energies = [float(windows[index]["band_db"][band]) for index in run]
                run_deltas = [float(deltas[index]) for index in run]
                before = float(windows[first - 1]["band_db"][band]) if first > 0 else None
                after = float(windows[last + 1]["band_db"][band]) if last + 1 < len(windows) else None
                entry = None if before is None else energies[0] - before
                exit_jump = None if after is None else energies[-1] - after
                internal_range = max(energies) - min(energies)
                trend = energies[-1] - energies[0]
                if len(energies) > 1:
                    transitions = [direction_sign * (right - left) for left, right in zip(energies, energies[1:])]
                    monotonic_fraction = sum(value >= -0.25 for value in transitions) / float(len(transitions))
                else:
                    monotonic_fraction = 0.0

                entry_edge = entry is None or direction_sign * entry >= 5.0
                exit_edge = exit_jump is None or direction_sign * exit_jump >= 5.0
                if len(run) >= 3 and internal_range <= 3.5 and entry_edge and exit_edge:
                    shape = "step"
                elif len(run) >= 3 and direction_sign * trend >= 3.0 and monotonic_fraction >= 0.75:
                    shape = "ramp"
                else:
                    shape = "transient"
                sections.append({
                    "start_sec": windows[first]["start_sec"],
                    "end_sec": windows[last]["end_sec"],
                    "band": band,
                    "direction": direction,
                    "energy_db": round(float(np.mean(energies)), 3),
                    "baseline_db": round(float(np.mean([baselines[index] for index in run])), 3),
                    "delta_db": round(float(np.mean(run_deltas)), 3),
                    "min_delta_db": round(min(run_deltas), 3),
                    "max_delta_db": round(max(run_deltas), 3),
                    "energy_start_db": round(energies[0], 3),
                    "energy_end_db": round(energies[-1], 3),
                    "entry_jump_db": None if entry is None else round(entry, 3),
                    "exit_jump_db": None if exit_jump is None else round(exit_jump, 3),
                    "internal_range_db": round(internal_range, 3),
                    "trend_db": round(trend, 3),
                    "monotonic_fraction": round(monotonic_fraction, 3),
                    "shape": shape,
                    "window_count": len(run),
                })

    # A high/air-only cut is a spectral tilt (often an intentional low-pass).
    # Treat a step-shaped cut as fault-level only when several separated bands
    # fall together over substantially the same interval. A boost is
    # fault-level when it is large, or when several bands rise together.
    for section in sections:
        together = []
        if section["shape"] == "step":
            for other in sections:
                if other is section or other["direction"] != section["direction"] or other["shape"] != "step":
                    continue
                overlap = max(0.0, min(section["end_sec"], other["end_sec"]) - max(section["start_sec"], other["start_sec"]))
                shorter = min(section["end_sec"] - section["start_sec"], other["end_sec"] - other["start_sec"])
                if shorter > 0 and overlap / shorter >= 0.5:
                    together.append(other["band"])
        broad = len(set(together + [section["band"]])) >= _BROAD_STEP_MIN_BANDS
        fault_level = section["shape"] == "step" and (
            broad or (section["direction"] == "boost" and section["delta_db"] >= _STEP_FAULT_MIN_DB)
        )
        section["fault_level"] = bool(fault_level)
        section["classification"] = "fault" if fault_level else "informational"
        ignored = next((
            item for item in (ignore_ranges or [])
            if _mostly_inside_ignore_range(section["start_sec"], section["end_sec"], item)
        ), None)
        section["ignored"] = ignored is not None
        section["ignore_reason"] = None if ignored is None else ignored["reason"]
        if ignored is not None:
            section["fault_level"] = False
            section["classification"] = "informational"
    return sorted(sections, key=lambda item: (item["start_sec"], item["band"], item["direction"]))


_NEW_CONTENT_EXCESS_DB = 6.0
_NEW_CONTENT_SHARE = 0.20
_RAISED_EXISTING_SHARE = 0.60


_SPECTRA_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _band_power_rows(
    samples: np.ndarray, size: int, hop: int, rows: int, bands: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Power spectra (one row per hop-spaced frame) restricted to each named band's bins."""
    masks = dict(_named_band_masks(size))
    window = _hann(size)
    out = {name: np.empty((rows, int(masks[name].sum()))) for name in bands}
    frames_per_chunk = 64
    for row in range(0, rows, frames_per_chunk):
        count = min(frames_per_chunk, rows - row)
        block = np.asarray(samples[row * hop:(row + count - 1) * hop + size], dtype=np.float64)
        frames = np.lib.stride_tricks.sliding_window_view(block, size)[::hop]
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        for name in bands:
            out[name][row:row + count] = power[:, masks[name]]
    return out


def _mark_content_changes(
    sections: List[Dict[str, Any]], samples: np.ndarray, ignore_ranges: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Downgrade band boosts that are new notes or sounds, not a gain change.

    A gain fault raises the band's usual content by about the same amount.
    When the section's extra energy sits on frequencies that are otherwise
    nearly absent (a bass line moving up into the band), or the band's usual
    content does not rise, the step is a change of arrangement: it stays in
    the outliers as informational.

    Frame spectra are computed once per frame size and shared by the sections
    that use it (bands are grouped so the kept matrices stay under a memory cap).
    """
    eligible: Dict[int, List[Dict[str, Any]]] = {}
    for section in sections:
        if not section.get("fault_level") or section.get("direction") != "boost":
            continue
        start, end = float(section["start_sec"]), float(section["end_sec"])
        size = 16384
        while size > 4096 and size > (end - start) * SAMPLE_RATE / 2.0:
            size //= 2
        if size > (end - start) * SAMPLE_RATE:
            continue
        eligible.setdefault(size, []).append(section)

    for size, size_sections in eligible.items():
        hop = size // 4
        first_samples = np.arange(0, len(samples) - size + 1, hop)
        rows = len(first_samples)
        seg_start = first_samples / float(SAMPLE_RATE)
        seg_end = (first_samples + size) / float(SAMPLE_RATE)
        bins_of = {name: int(mask.sum()) for name, mask in _named_band_masks(size)}
        wanted = list(dict.fromkeys(section["band"] for section in size_sections))
        groups: List[List[str]] = []
        for name in wanted:
            if groups and rows * 8 * sum(bins_of[n] for n in groups[-1] + [name]) <= _SPECTRA_CACHE_MAX_BYTES:
                groups[-1].append(name)
            else:
                groups.append([name])
        for group in groups:
            matrices = _band_power_rows(samples, size, hop, rows, group)
            for section in size_sections:
                if section["band"] not in matrices:
                    continue
                start, end = float(section["start_sec"]), float(section["end_sec"])
                matrix = matrices[section["band"]]
                inside_rows = (start <= seg_start) & (seg_end <= end)
                outside_rows = ~inside_rows & ((seg_end <= start) | (seg_start >= end))
                for item in ignore_ranges:
                    outside_rows &= ~((seg_start < item["end_sec"]) & (item["start_sec"] < seg_end))
                if not inside_rows.any() or int(outside_rows.sum()) < 4:
                    continue
                # Column blocks keep the row-selected copies small; columns are independent.
                in_power = np.empty(matrix.shape[1])
                out_power = np.empty(matrix.shape[1])
                for first in range(0, matrix.shape[1], 256):
                    block = matrix[:, first:first + 256]
                    in_power[first:first + 256] = np.mean(block[inside_rows], axis=0)
                    out_power[first:first + 256] = np.median(block[outside_rows], axis=0)
                if float(np.sum(in_power)) <= 0.0 or float(np.sum(out_power)) <= 0.0:
                    continue
                delta = 10.0 * math.log10(float(np.sum(in_power)) / float(np.sum(out_power)))
                rise = 10.0 * np.log10((in_power + 1e-20) / (out_power + 1e-20))
                new_share = float(np.sum(in_power[rise >= delta + _NEW_CONTENT_EXCESS_DB]) / np.sum(in_power))
                raised_share = float(np.sum(out_power[rise >= delta / 2.0]) / np.sum(out_power))
                section["new_content_share"] = round(new_share, 3)
                section["raised_existing_share"] = round(raised_share, 3)
                if new_share > _NEW_CONTENT_SHARE or raised_share < _RAISED_EXISTING_SHARE:
                    section["fault_level"] = False
                    section["classification"] = "informational"
                    section["content_change"] = True
    return sections


def _normalize_ignore_ranges(ignore_ranges: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized = []
    for index, item in enumerate(ignore_ranges or []):
        if not isinstance(item, dict) or "start_sec" not in item or "end_sec" not in item:
            raise ExtractionBoundsError("Each ignore range requires start_sec and end_sec", {"range_index": index})
        start = _finite_number(item["start_sec"], "ignore_ranges.start_sec")
        end = _finite_number(item["end_sec"], "ignore_ranges.end_sec")
        if start < 0.0 or end <= start:
            raise ExtractionBoundsError("Ignore ranges must satisfy 0 <= start_sec < end_sec", {"range_index": index})
        reason = str(item.get("reason", "intentional section"))
        if len(reason) > 200:
            reason = reason[:200]
        normalized.append({"start_sec": round(start, 6), "end_sec": round(end, 6), "reason": reason})
    return normalized


def _mostly_inside_ignore_range(start: float, end: float, ignored: Mapping[str, Any]) -> bool:
    """Whether more than half of a measured span lies inside an intent range."""
    duration = float(end) - float(start)
    if duration <= 0.0:
        return False
    overlap = max(0.0, min(float(end), float(ignored["end_sec"]))
                  - max(float(start), float(ignored["start_sec"])))
    return overlap / duration > 0.5


_BALANCE_RATE = 24000
_BALANCE_EDGES_HZ = (44.0, 88.0, 177.0, 354.0, 707.0, 1414.0, 2828.0, 5657.0, 11314.0)
_BALANCE_SHIFT_DB = 6.0
_BALANCE_MIN_WINDOWS = 2
_BALANCE_NEW_CONTENT_DB = 6.0
# Low end is mixed to the centre; a low band whose usual balance sits this far
# to one side for most of the piece was panned (or routed) to one channel.
_BALANCE_LOW_EDGE_HZ = 354.0
_BALANCE_OFF_CENTRE_DB = 6.0
_BALANCE_OFF_CENTRE_SHARE = 0.5


def _decode_stereo(source: str, rate: int = _BALANCE_RATE) -> Optional[np.ndarray]:
    """Two-channel PCM (24 kHz by default), or None for a mono or silent stream."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise MediaNotFoundError("Audio source file not found", {"path": str(path)})
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False,
        timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    if probe.returncode != 0 or probe.stdout.strip() in {"", "1"}:
        return None
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "2",
         "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True, check=False, timeout=max(Config.SUBPROCESS_TIMEOUT_SEC * 4, 180.0),
    )
    if completed.returncode != 0 or len(completed.stdout) < 8:
        return None
    return np.frombuffer(completed.stdout[:len(completed.stdout) // 8 * 8], dtype=np.float32).reshape(-1, 2)


def _channel_peaks(source: str, first_sample: int, last_sample: int) -> Optional[np.ndarray]:
    """Loudest channel at each SAMPLE_RATE sample in [first_sample, last_sample).

    Streamed, so memory is one float per sample however many channels there
    are. None for a mono stream or a failed decode (the caller then uses the
    mono mix, which is exact for mono).
    """
    path = Path(source).expanduser().resolve()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False,
        timeout=Config.SUBPROCESS_TIMEOUT_SEC,
    )
    try:
        channels = int(probe.stdout.strip().split(",")[0])
    except ValueError:
        return None
    count = last_sample - first_sample
    if probe.returncode != 0 or channels < 2 or count <= 0:
        return None
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{first_sample / SAMPLE_RATE:.6f}",
         "-t", f"{count / SAMPLE_RATE:.6f}", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    peaks = np.zeros(count, dtype=np.float32)
    frame_bytes = 4 * channels
    filled = 0
    pending = b""
    try:
        assert process.stdout is not None
        while filled < count:
            chunk = process.stdout.read(65536 * frame_bytes)
            if not chunk:
                break
            pending += chunk
            usable = len(pending) // frame_bytes * frame_bytes
            block = np.frombuffer(pending[:usable], dtype=np.float32).reshape(-1, channels)
            pending = pending[usable:]
            take = min(len(block), count - filled)
            peaks[filled:filled + take] = np.max(np.abs(block[:take]), axis=1)
            filled += take
    finally:
        process.kill()
        process.wait()
    # A short read means the decode failed partway; a few missing tail
    # samples are only rounding between the two decodes.
    if filled < count - SAMPLE_RATE // 100:
        return None
    return peaks


def stereo_balance_shifts(
    source: str, bpm: Optional[float] = None, ignore_ranges: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Stretches where one band's left/right balance leaves the piece's usual balance.

    Each bar (or 2 s window without a tempo) is measured per octave band as
    right-minus-left level. A band's usual balance is where most of its
    windows agree (within 3 dB; the opening wins a tie). Two or more consecutive windows at least 6 dB off it, on the same
    side, are a shift: something was re-panned. When the band is also 6 dB
    louder there, the shift is new content (an instrument entering on one
    side) and is kept as informational. Windows touching `ignore_ranges`
    (sections the author's notes describe) are skipped.

    A shift is only visible against the piece's own balance, so a band that is
    one-sided throughout is reported separately in `off_centre`: a band below
    354 Hz whose usual balance is at least 6 dB off centre in at least half
    of its windows.
    """
    stereo = _decode_stereo(source)
    window = 60.0 * 4.0 / float(bpm) if bpm else 2.0
    result: Dict[str, Any] = {"window_sec": round(window, 6), "shifts": [], "off_centre": [],
                              "stereo": stereo is not None}
    if stereo is None:
        return result
    size = int(round(window * _BALANCE_RATE))
    count = len(stereo) // size
    if count < 4:
        return result
    ignored = _normalize_ignore_ranges(ignore_ranges)
    frequencies = np.fft.rfftfreq(size, 1.0 / _BALANCE_RATE)
    taper = np.hanning(size)[:, None]
    bands = [(frequencies >= low) & (frequencies < high)
             for low, high in zip(_BALANCE_EDGES_HZ[:-1], _BALANCE_EDGES_HZ[1:])]
    balance = np.zeros((count, len(bands)))
    level = np.zeros((count, len(bands)))
    for index in range(count):
        segment = stereo[index * size:(index + 1) * size]
        power = np.abs(np.fft.rfft(segment * taper, axis=0)) ** 2
        for band, mask in enumerate(bands):
            left, right = float(np.sum(power[mask, 0])), float(np.sum(power[mask, 1]))
            balance[index, band] = 10.0 * math.log10((right + 1e-12) / (left + 1e-12))
            level[index, band] = 10.0 * math.log10(left + right + 1e-12)
    # Sections that only touch a window's edge (rounded times) leave it in.
    skip = np.asarray([any(float(item["start_sec"]) < (index + 1) * window - .05
                           and float(item["end_sec"]) > index * window + .05
                           for item in ignored) for index in range(count)], dtype=bool)
    shifts = []
    off_centre = []
    for band in range(len(bands)):
        usable = ~skip & (level[:, band] >= np.median(level[~skip, band]) - 20.0) if np.any(~skip) else ~skip
        if int(np.sum(usable)) < 4:
            continue
        # The usual balance is the largest group of windows within 3 dB of
        # each other; on a tie the earliest group (the opening) sets it.
        values = balance[usable, band]
        support = [int(np.sum(np.abs(values - value) <= 3.0)) for value in values]
        anchor = values[int(np.argmax(support))]
        usual = float(np.median(values[np.abs(values - anchor) <= 3.0]))
        share = float(np.mean(np.abs(values - usual) <= 3.0))
        if (_BALANCE_EDGES_HZ[band + 1] <= _BALANCE_LOW_EDGE_HZ and abs(usual) >= _BALANCE_OFF_CENTRE_DB
                and share >= _BALANCE_OFF_CENTRE_SHARE):
            windows = np.flatnonzero(usable)
            off_centre.append({
                "start_sec": round(float(windows[0]) * window, 3), "end_sec": round(float(windows[-1] + 1) * window, 3),
                "band": "{:g}-{:g} Hz".format(_BALANCE_EDGES_HZ[band], _BALANCE_EDGES_HZ[band + 1]),
                "balance_db": round(usual, 2), "side": "right" if usual > 0 else "left",
                "share_of_windows": round(share, 3),
            })
        offset = balance[:, band] - usual
        run: List[int] = []
        for index in list(range(count)) + [None]:
            same_side = (index is not None and usable[index] and abs(offset[index]) >= _BALANCE_SHIFT_DB
                         and (not run or np.sign(offset[index]) == np.sign(offset[run[0]])))
            if same_side:
                run.append(index)
                continue
            if len(run) >= _BALANCE_MIN_WINDOWS:
                inside = np.asarray(run)
                outside = usable.copy()
                outside[inside] = False
                level_change = (float(np.median(level[inside, band])) - float(np.median(level[outside, band]))
                                if np.any(outside) else 0.0)
                shifted = float(np.median(balance[inside, band]))
                shifts.append({
                    "start_sec": round(run[0] * window, 3), "end_sec": round((run[-1] + 1) * window, 3),
                    "band": "{:g}-{:g} Hz".format(_BALANCE_EDGES_HZ[band], _BALANCE_EDGES_HZ[band + 1]),
                    "balance_db": round(shifted, 2), "usual_balance_db": round(usual, 2),
                    "delta_db": round(shifted - usual, 2), "side": "right" if shifted > usual else "left",
                    "level_change_db": round(level_change, 2),
                    "new_content": level_change >= _BALANCE_NEW_CONTENT_DB,
                })
            run = [index] if index is not None and usable[index] and abs(offset[index]) >= _BALANCE_SHIFT_DB else []
    result["shifts"] = sorted(shifts, key=lambda item: (item["start_sec"], item["band"]))
    result["off_centre"] = off_centre
    return result


_CLIP_LEVEL = 0.95
_CLIP_WINDOW_SEC = 0.1
_CLIP_MIN_SAMPLES = 24
_CLIP_THROUGHOUT_SHARE = 0.2


def clipping_runs(source: str, ignore_ranges: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Stretches where the master sits at full scale: clipped or crushed peaks.

    Each 100 ms window counts samples (either channel) at or above 0.95 of
    full scale. A window with 24 or more is hot; neighbouring hot windows form
    one run. A lone overshoot (a few samples) is not clipping. When hot
    windows cover a fifth of the programme, the master is loud throughout
    (limited, not a local fault) and `throughout` is set. Runs touching
    A run is marked `in_ignored_range` only when more than half its duration lies inside an ignore range.
    """
    stereo = _decode_stereo(source, rate=SAMPLE_RATE)
    if stereo is None:
        samples, _, _ = _decode_pcm(source)
        stereo = np.asarray(samples, dtype=np.float32)[:, None]
    size = int(round(_CLIP_WINDOW_SEC * SAMPLE_RATE))
    count = len(stereo) // size
    result: Dict[str, Any] = {"level": _CLIP_LEVEL, "window_sec": _CLIP_WINDOW_SEC, "runs": [],
                              "throughout": False, "hot_share": 0.0,
                              "peak": round(float(np.max(np.abs(stereo))) if len(stereo) else 0.0, 4)}
    if not count:
        return result
    magnitude = np.max(np.abs(stereo[:count * size]), axis=1).reshape(count, size)
    hot_counts = np.sum(magnitude >= _CLIP_LEVEL, axis=1)
    hot = hot_counts >= _CLIP_MIN_SAMPLES
    result["hot_share"] = round(float(np.mean(hot)), 4)
    result["throughout"] = bool(result["hot_share"] >= _CLIP_THROUGHOUT_SHARE)
    ignored = _normalize_ignore_ranges(ignore_ranges)
    index = 0
    while index < count:
        if not hot[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and hot[end + 1]:
            end += 1
        start_sec, end_sec = index * _CLIP_WINDOW_SEC, (end + 1) * _CLIP_WINDOW_SEC
        result["runs"].append({
            "start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3),
            "samples_at_ceiling": int(np.sum(hot_counts[index:end + 1])),
            "peak": round(float(np.max(magnitude[index:end + 1])), 4),
            "in_ignored_range": any(_mostly_inside_ignore_range(start_sec, end_sec, item)
                                    for item in ignored),
        })
        index = end + 1
    return result


def loudness_timeline(
    source: str,
    window_sec: Optional[float] = None,
    bpm: Optional[float] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    cache: Optional[MediaCache] = None,
    ignore_ranges: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Measure RMS, peak, spectral bands, centroid, silence and band outliers."""
    if window_sec is not None and bpm is not None:
        raise ExtractionBoundsError("Specify either window_sec or bpm, not both; omit both for 0.25s windows", {"window_sec": window_sec, "bpm": bpm})
    if bpm is not None:
        bpm_value = _finite_number(bpm, "bpm")
        if bpm_value < 20.0 or bpm_value > 300.0:
            raise ExtractionBoundsError("bpm must satisfy 20 <= bpm <= 300", {"bpm": bpm_value})
        width = 60.0 / bpm_value
    else:
        width = 0.25 if window_sec is None else _finite_number(window_sec, "window_sec")
        if width < 0.01 or width > Config.MAX_WINDOW_DURATION_SEC:
            raise ExtractionBoundsError("window_sec must be in [0.01, 600] (default 0.25)", {"window_sec": width})
        bpm_value = None

    samples, fingerprint, cache = _decode_pcm(source, cache)
    start, end, first_sample, last_sample = _range(samples, start_sec, end_sec)
    normalized_ignore_ranges = _normalize_ignore_ranges(ignore_ranges)
    options = {"window_sec": round(width, 6), "bpm": None if bpm_value is None else round(bpm_value, 6), "start_sec": round(start, 6), "end_sec": round(end, 6), "ignore_ranges": normalized_ignore_ranges}
    # v5: peak_dbfs is the loudest sample on any channel, not on the mono mix.
    result_key = _cache_tag("loudness_timeline_v5", options)
    cached = cache.load_json(fingerprint, result_key)
    if isinstance(cached, dict):
        return cached

    window_count = int(math.ceil((end - start) / width))
    if window_count > 50000:
        raise ExtractionBoundsError(f"loudness_timeline would need {window_count} windows (limit 50,000); use a larger window_sec (at least {(end - start) / 50000:.3g}) or a narrower start_sec/end_sec", {"window_count": window_count})
    local = samples[first_sample:last_sample]
    channel_peaks = _channel_peaks(source, first_sample, last_sample)
    if channel_peaks is None:
        channel_peaks = np.abs(np.asarray(local, dtype=np.float32))
    windows: List[Dict[str, Any]] = []
    for index in range(window_count):
        window_start = start + index * width
        window_end = min(end, window_start + width)
        local_first = max(0, int(round(window_start * SAMPLE_RATE)) - first_sample)
        local_last = min(len(local), int(round(window_end * SAMPLE_RATE)) - first_sample)
        windows.append(_measure_window(local[local_first:local_last], window_start, window_end,
                                       channel_peaks[local_first:local_last]))
    local_array = np.asarray(local, dtype=np.float32)
    loudest = int(np.argmax(channel_peaks)) if len(channel_peaks) else 0
    peak_value = float(channel_peaks[loudest]) if len(channel_peaks) else 0.0
    result = {
        "source_path": str(Path(source).expanduser().resolve()),
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "window_sec": round(width, 6),
        "bpm": None if bpm_value is None else round(bpm_value, 6),
        "silence_threshold_dbfs": -60.0,
        "peak_dbfs": round(20.0 * math.log10(max(peak_value, 1e-6)), 3),
        "peak_sec": round((first_sample + loudest) / SAMPLE_RATE, 3),
        "windows": windows,
        "outliers": _mark_content_changes(
            _outlier_sections(windows, ignore_ranges=normalized_ignore_ranges),
            samples, normalized_ignore_ranges,
        ),
        "ignore_ranges": normalized_ignore_ranges,
        "silence_gaps": _silence_gaps(local_array, start, end),
        "total": len(windows),
    }
    cache.save_json(fingerprint, result_key, result)
    return result


def _color_map(values: np.ndarray) -> np.ndarray:
    # Dark violet -> blue -> cyan -> yellow -> pale, a compact perceptual ramp.
    stops = np.array([0, 64, 128, 192, 255], dtype=np.float32)
    colors = np.array([
        [8, 5, 24], [55, 28, 105], [27, 110, 160], [55, 190, 150], [250, 230, 120]
    ], dtype=np.float32)
    scaled = np.clip(values, 0.0, 255.0)
    output = np.empty(scaled.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.interp(scaled, stops, colors[:, channel]).astype(np.uint8)
    return output


def spectrogram_window(
    source: str,
    start_sec: float,
    end_sec: float,
    marks: Optional[Sequence[Dict[str, Any]]] = None,
    cache: Optional[MediaCache] = None,
) -> Dict[str, Any]:
    """Render a fixed-size log-frequency PNG with absolute timecode badges."""
    start_requested = _finite_number(start_sec, "start_sec")
    end_requested = _finite_number(end_sec, "end_sec")
    if end_requested <= start_requested:
        raise ExtractionBoundsError("spectrogram_window requires end_sec > start_sec (both are required, in seconds)", {"start_sec": start_requested, "end_sec": end_requested})
    if end_requested - start_requested > Config.MAX_WINDOW_DURATION_SEC:
        raise ExtractionBoundsError("spectrogram_window exceeds the 600 second limit; render windows of at most 600 s, and a short window (a few seconds) to see individual events", {"duration_sec": end_requested - start_requested})
    normalized_marks: List[Dict[str, Any]] = []
    for index, mark in enumerate(marks or []):
        if not isinstance(mark, dict) or ("time_sec" not in mark and "t" not in mark):
            raise ExtractionBoundsError('Each mark requires time_sec and a label, e.g. {"time_sec": 12.5, "label": "kick"}', {"mark_index": index})
        mark_time = _finite_number(mark.get("time_sec", mark.get("t")), "mark.time_sec")
        label = str(mark.get("label", ""))
        if len(label) > 100:
            label = label[:100]
        normalized_marks.append({"time_sec": round(mark_time, 6), "label": label})

    samples, fingerprint, cache = _decode_pcm(source, cache)
    start, end, first_sample, last_sample = _range(samples, start_requested, end_requested)
    active_marks = [item for item in normalized_marks if start <= item["time_sec"] <= end]
    options = {"start_sec": round(start, 6), "end_sec": round(end, 6), "marks": active_marks, "size": [1200, 600]}
    result_key = _cache_tag("spectrogram_v2", options)
    sheets_dir = cache.get_audio_dir(fingerprint)
    png_path = sheets_dir / (result_key + ".png")
    cached = cache.load_json(fingerprint, result_key)
    if isinstance(cached, dict) and png_path.exists():
        return cached

    signal = np.asarray(samples[first_sample:last_sample], dtype=np.float32)
    if len(signal) < 2:
        raise ExtractionBoundsError("spectrogram window contains too few audio samples", {"start_sec": start, "end_sec": end})
    half = N_FFT // 2
    padded = np.pad(signal, (half, N_FFT - half), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_SAMPLES]
    window = np.hanning(N_FFT).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).T
    scale = max(float(np.sum(window)) / 2.0, 1.0)
    db_values = 20.0 * np.log10(np.maximum(magnitude / scale, 1e-7))

    width, height = 1200, 600
    left, right, top, bottom = 76, 22, 58, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    nyquist = SAMPLE_RATE / 2.0
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    target_freqs = np.geomspace(20.0, min(20000.0, nyquist), plot_height)
    bin_indices = np.clip(np.searchsorted(frequencies, target_freqs), 1, len(frequencies) - 1)
    lower = bin_indices - 1
    frac = (np.log(target_freqs) - np.log(np.maximum(frequencies[lower], 1.0))) / (
        np.log(frequencies[bin_indices]) - np.log(np.maximum(frequencies[lower], 1.0)) + 1e-12
    )
    log_rows = db_values[lower, :] * (1.0 - frac[:, None]) + db_values[bin_indices, :] * frac[:, None]
    levels = np.clip((log_rows + 100.0) / 100.0 * 255.0, 0.0, 255.0).astype(np.uint8)
    image_data = Image.fromarray(_color_map(levels[::-1, :]))
    image_data = image_data.resize((plot_width, plot_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (17, 19, 26))
    canvas.paste(image_data, (left, top))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13)
        small_font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
        small_font = font

    # Burn the same index + absolute HH:MM:SS.mmm pattern used by frame evidence.
    badge_count = max(2, min(7, int(math.ceil(end - start)) + 1))
    badge_times = np.linspace(start, end, badge_count)
    for index, timestamp in enumerate(badge_times, 1):
        text = "#{:03d} [{}]".format(index, format_timestamp(float(timestamp)))
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x = left + int(round((float(timestamp) - start) / (end - start) * plot_width))
        bx = max(left, min(width - right - text_width - 12, x - text_width // 2 - 6))
        draw.rectangle((bx, 8, bx + text_width + 12, 8 + 22), fill=(15, 17, 23), outline=(0, 200, 255), width=1)
        draw.text((bx + 6, 12), text, fill=(255, 255, 255), font=font)

    # Log-frequency scale and 1-second-or-coarser absolute time ticks.
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=(155, 160, 177), width=1)
    frequency_ticks = []
    for frequency in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        if frequency < 20.0 or frequency > min(20000.0, nyquist):
            continue
        y = top + int(round((math.log(min(20000.0, nyquist)) - math.log(float(frequency))) / (math.log(min(20000.0, nyquist)) - math.log(20.0)) * plot_height))
        draw.line((left - 4, y, left + plot_width, y), fill=(80, 83, 98), width=1)
        label = "{}k".format(int(frequency / 1000)) if frequency >= 1000 else str(frequency)
        draw.text((4, max(top, y - 6)), label, fill=(220, 222, 230), font=small_font)
        frequency_ticks.append({"frequency_hz": int(frequency), "label": label, "y_px": y})
    tick_step = 1.0 if end - start <= 12.0 else max(1.0, math.ceil((end - start) / 12.0))
    tick = math.ceil(start / tick_step) * tick_step
    while tick <= end + 1e-6:
        x = left + int(round((tick - start) / (end - start) * plot_width))
        draw.line((x, top + plot_height, x, top + plot_height + 4), fill=(220, 222, 230), width=1)
        draw.text((max(left, min(width - right - 58, x - 22)), top + plot_height + 8), format_timestamp(tick)[-6:], fill=(220, 222, 230), font=small_font)
        tick += tick_step
    draw.text((left, height - 23), "Time (absolute video seconds)  |  Log frequency  |  dBFS", fill=(220, 222, 230), font=small_font)

    marker_results = []
    marker_palette = [(255, 93, 93), (255, 195, 70), (150, 225, 255), (255, 120, 220)]
    for index, mark in enumerate(active_marks):
        x = left + int(round((mark["time_sec"] - start) / (end - start) * plot_width))
        color = marker_palette[index % len(marker_palette)]
        for y in range(top, top + plot_height, 10):
            draw.line((x, y, x, min(y + 5, top + plot_height)), fill=color, width=2)
        label = mark["label"] or format_timestamp(mark["time_sec"])
        bbox = small_font.getbbox(label)
        label_width = bbox[2] - bbox[0] + 8
        lx = max(left, min(width - right - label_width, x + 4))
        draw.rectangle((lx, top + 5, lx + label_width, top + 21), fill=(15, 17, 23))
        draw.text((lx + 4, top + 7), label, fill=color, font=small_font)
        marker_results.append({"time_sec": mark["time_sec"], "label": mark["label"]})

    temporary = png_path.with_name(".{}.{}.tmp.png".format(png_path.stem, uuid.uuid4().hex[:8]))
    canvas.save(str(temporary), format="PNG")
    os.replace(str(temporary), str(png_path))
    result = {
        "source_path": str(Path(source).expanduser().resolve()),
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "file_path": str(png_path.resolve()),
        "width": width,
        "height": height,
        "sample_rate": SAMPLE_RATE,
        "hop_samples": HOP_SAMPLES,
        "timecode_badges": [format_timestamp(float(value)) for value in badge_times],
        "frequency_ticks": frequency_ticks,
        "marks": marker_results,
    }
    cache.save_json(fingerprint, result_key, result)
    return result


def _cue_group_name(label: str) -> str:
    """Use the leading sound kind, ignoring lyric text and cue serials."""
    head = re.split(r"\s*[|:—]\s*", str(label).strip(), maxsplit=1)[0]
    head = re.sub(r"[\s._#-]*\d+\s*$", "", head).strip().casefold()
    words = head.split()
    return " ".join(words[:2]) or "cue"


def _profile_power(profile: Mapping[str, float]) -> np.ndarray:
    return np.asarray([
        10.0 ** (max(-120.0, min(20.0, float(profile.get(name, -120.0)))) / 10.0)
        for name, _, _ in BANDS
    ], dtype=np.float64)


def _unit_vector(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(np.maximum(values, 0.0)))
    return np.maximum(values, 0.0) / total if total > 1e-15 else np.zeros_like(values, dtype=np.float64)


def _profile_for_onset(onset: Mapping[str, Any]) -> np.ndarray:
    profile = onset.get("band_db")
    return _unit_vector(_profile_power(profile)) if isinstance(profile, dict) else np.zeros(len(BANDS), dtype=np.float64)


def _cue_coincident(normalized: Sequence[Mapping[str, Any]], cue_index: int, coincidence_sec: float) -> bool:
    timestamp = float(normalized[cue_index]["time_sec"])
    return any(
        other_index != cue_index and abs(float(other["time_sec"]) - timestamp) <= coincidence_sec
        for other_index, other in enumerate(normalized)
    )


def _bar_control_profile(
    samples: np.ndarray,
    event_time: float,
    normalized_cues: Sequence[Mapping[str, Any]],
    bar_sec: Optional[float],
    exclusion_sec: float,
    control_cache: Optional[Dict[float, Optional[np.ndarray]]] = None,
) -> Optional[np.ndarray]:
    """Median spectrum at the same bar phase in uncued comparison bars."""
    if bar_sec is None or bar_sec <= 0.0:
        return None
    cache_key = round(event_time, 6)
    if control_cache is not None and cache_key in control_cache:
        return control_cache[cache_key]
    duration = len(samples) / float(SAMPLE_RATE)
    profiles = []
    for bar_index in range(-16, 17):
        if bar_index == 0:
            continue
        sample_time = event_time + bar_index * bar_sec
        if sample_time < 0.0 or sample_time + 0.06 > duration:
            continue
        if any(abs(float(cue["time_sec"]) - sample_time) <= exclusion_sec for cue in normalized_cues):
            continue
        first = max(0, int(round(sample_time * SAMPLE_RATE)))
        last = min(len(samples), first + int(round(0.085 * SAMPLE_RATE)))
        if last - first < int(0.04 * SAMPLE_RATE):
            continue
        profiles.append(_profile_power(_band_profile(samples[first:last])))
    if len(profiles) < 2:
        if control_cache is not None:
            control_cache[cache_key] = None
        return None
    result = np.median(np.asarray(profiles), axis=0)
    if control_cache is not None:
        control_cache[cache_key] = result
    return result


def _event_signature_vector(
    onset: Mapping[str, Any],
    samples: np.ndarray,
    normalized_cues: Sequence[Mapping[str, Any]],
    bar_sec: Optional[float],
    exclusion_sec: float,
    control_cache: Optional[Dict[float, Optional[np.ndarray]]] = None,
) -> np.ndarray:
    observed = _profile_power(onset.get("band_db", {}))
    control = _bar_control_profile(
        samples, float(onset["time_sec"]), normalized_cues, bar_sec, exclusion_sec, control_cache,
    )
    if control is not None:
        excess = np.maximum(observed - control, 0.0)
        # Keep a clear contribution over the bar-position bed; if subtraction
        # leaves too little signal, the full event profile is safer evidence.
        if float(np.sum(excess)) >= max(float(np.sum(observed)) * 0.04, 1e-12):
            return _unit_vector(excess)
    return _unit_vector(observed)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-15 else 0.0


def _signature_bands(profile: np.ndarray) -> List[str]:
    ranked = np.argsort(profile)[::-1]
    selected = []
    cumulative = 0.0
    for index in ranked:
        if profile[index] <= 0.08 or len(selected) >= 3:
            continue
        selected.append(BANDS[int(index)][0])
        cumulative += float(profile[index])
        if cumulative >= 0.78:
            break
    return selected


def _local_band_power(samples: np.ndarray, event_time: float) -> np.ndarray:
    """Measure a short attack window in the same units as onset spectra."""
    first = max(0, int(round((event_time - 0.005) * SAMPLE_RATE)))
    last = min(len(samples), first + int(round(0.090 * SAMPLE_RATE)))
    return _profile_power(_band_profile(samples[first:last]))


def _same_bar_phase(left: float, right: float, bar_sec: Optional[float]) -> bool:
    if bar_sec is None:
        return True
    distance = abs((left - right + bar_sec / 2.0) % bar_sec - bar_sec / 2.0)
    return distance <= 0.080


def _coincident_groups(cues: Sequence[Mapping[str, Any]], timestamp: float, coincidence_sec: float) -> set:
    return {str(cue["group"]) for cue in cues if abs(float(cue["time_sec"]) - timestamp) <= coincidence_sec}


def _differential_signature(
    group: str,
    indices: Sequence[int],
    normalized: Sequence[Mapping[str, Any]],
    samples: np.ndarray,
    bar_sec: Optional[float],
    coincidence_sec: float,
) -> Optional[Dict[str, Any]]:
    """Learn an always-overlaid layer from bars carrying its backing sound alone."""
    excess_examples = []
    baselines = []
    baseline_by_cue = {}
    example_times = []
    control_times = set()
    for index in indices:
        cue_time = float(normalized[index]["time_sec"])
        companions = _coincident_groups(normalized, cue_time, coincidence_sec) - {group}
        if not companions:
            continue
        controls = []
        for other in normalized:
            other_time = float(other["time_sec"])
            if abs(other_time - cue_time) <= coincidence_sec or not _same_bar_phase(cue_time, other_time, bar_sec):
                continue
            other_groups = _coincident_groups(normalized, other_time, coincidence_sec)
            if group in other_groups or not other_groups.intersection(companions):
                continue
            controls.append(other_time)
        controls = sorted(set(round(value, 6) for value in controls))
        if len(controls) < 2:
            continue
        # Exact backing configurations are preferred. Extra layers on a
        # control beat would inflate its spectrum and hide the target layer.
        exact = [time for time in controls
                 if _coincident_groups(normalized, time, coincidence_sec) == companions]
        if len(exact) >= 2:
            controls = exact
        baseline = np.median(np.asarray([_local_band_power(samples, time) for time in controls]), axis=0)
        observed = _local_band_power(samples, cue_time)
        baselines.append(baseline)
        baseline_by_cue[index] = baseline
        excess_examples.append(np.maximum(observed - baseline, 0.0))
        example_times.append(cue_time)
        control_times.update(controls)
    if len(excess_examples) < 3 or len(control_times) < 2:
        return None
    excess = np.median(np.asarray(excess_examples), axis=0)
    baseline = np.median(np.asarray(baselines), axis=0)
    if float(np.max(excess)) < 10.0 ** (-55.0 / 10.0):
        return None
    profile = _unit_vector(excess)
    bands = _signature_bands(profile)
    if not bands:
        return None
    return {
        "profile": profile,
        "bands": bands,
        "sample_count": len(excess_examples),
        "example_times_sec": example_times,
        "method": "differential_bar_excess",
        "baseline_power": baseline,
        "baseline_by_cue": baseline_by_cue,
        "excess_power": excess,
        "control_count": len(control_times),
    }


def _learn_cue_signatures(
    normalized: Sequence[Dict[str, Any]],
    broad_onsets: Sequence[Dict[str, Any]],
    samples: np.ndarray,
    *,
    search_window_sec: float,
    bar_sec: Optional[float],
    coincidence_sec: float,
    tolerance_sec: float,
    control_cache: Optional[Dict[float, Optional[np.ndarray]]] = None,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[int]] = {}
    for index, cue in enumerate(normalized):
        grouped.setdefault(cue["group"], []).append(index)
    models: Dict[str, Dict[str, Any]] = {}
    for name, indices in grouped.items():
        vectors = []
        example_times = []
        for index in indices:
            if _cue_coincident(normalized, index, coincidence_sec):
                continue
            cue_time = float(normalized[index]["time_sec"])
            nearby = [
                onset for onset in broad_onsets
                if abs(float(onset["time_sec"]) - cue_time) <= search_window_sec
            ]
            # A shifted/missing cue is not a trustworthy template example.
            # Learn signatures only when a singleton cue has an onset inside
            # the requested sync tolerance.
            nearby = [
                onset for onset in nearby
                if abs(float(onset["time_sec"]) - cue_time) <= tolerance_sec
            ]
            nearby.sort(key=lambda onset: abs(float(onset["time_sec"]) - cue_time))
            if not nearby:
                continue
            # A tie between separate attacks is not a safe training example.
            if len(nearby) > 1 and abs(
                abs(float(nearby[1]["time_sec"]) - cue_time)
                - abs(float(nearby[0]["time_sec"]) - cue_time)
            ) < 0.025:
                continue
            vector = _event_signature_vector(
                nearby[0], samples, normalized, bar_sec,
                max(coincidence_sec, tolerance_sec),
                control_cache,
            )
            if float(np.sum(vector)) <= 0.0:
                continue
            vectors.append(vector)
            example_times.append(cue_time)
        if vectors:
            profile = _unit_vector(np.median(np.asarray(vectors), axis=0))
            models[name] = {
                "profile": profile,
                "bands": _signature_bands(profile),
                "sample_count": len(vectors),
                "example_times_sec": example_times,
                "method": "singleton_signature",
            }
        else:
            differential = _differential_signature(
                name, indices, normalized, samples, bar_sec, coincidence_sec,
            )
            if differential is not None:
                models[name] = differential
    return models


def _cue_onset_candidates(
    cue: Mapping[str, Any],
    model: Optional[Mapping[str, Any]],
    broad_onsets: Sequence[Dict[str, Any]],
    onsets_by_band: Dict[str, List[Dict[str, Any]]],
    source: str,
    cache: MediaCache,
) -> List[Dict[str, Any]]:
    if cue.get("band") is not None:
        key = cue["band_key"]
        if key not in onsets_by_band:
            onsets_by_band[key] = audio_onsets(
                source, min_strength=0.10, cache=cache, band_hz=_band_spec(cue["band"])
            )["onsets"]
        pool = list(onsets_by_band[key])
    else:
        pool = list(broad_onsets)
        if model:
            for band_name in model.get("bands", []):
                if band_name not in onsets_by_band:
                    onsets_by_band[band_name] = audio_onsets(
                        source, min_strength=0.10, cache=cache, band_hz=band_name
                    )["onsets"]
                pool.extend(onsets_by_band[band_name])
    pool.sort(key=lambda item: float(item["time_sec"]))
    deduped: List[Dict[str, Any]] = []
    for onset in pool:
        if (
            deduped
            and abs(float(deduped[-1]["time_sec"]) - float(onset["time_sec"])) <= 0.060
            and min(float(onset.get("strength", 0.0)), float(deduped[-1].get("strength", 0.0)))
            <= 0.5 * max(float(onset.get("strength", 0.0)), float(deduped[-1].get("strength", 0.0)))
        ):
            if float(onset.get("strength", 0.0)) > float(deduped[-1].get("strength", 0.0)):
                deduped[-1] = onset
        else:
            deduped.append(onset)
    return deduped


def _spectral_attack(samples: np.ndarray, timestamp: float) -> np.ndarray:
    """Short attack spectrum; zero padding makes the first frame measurable."""
    width = int(0.060 * SAMPLE_RATE)
    first = int(round(timestamp * SAMPLE_RATE))
    chunk = np.zeros(width, dtype=np.float64)
    lo, hi = max(0, first), min(len(samples), first + width)
    if hi > lo:
        chunk[lo - first:hi - first] = samples[lo:hi]
    previous = np.zeros(width, dtype=np.float64)
    before = first - width
    lo, hi = max(0, before), min(len(samples), first)
    if hi > lo:
        previous[lo - before:hi - before] = samples[lo:hi]
    frequencies = np.fft.rfftfreq(width, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(40.0, 20000.0, 33)
    indices = np.searchsorted(edges, frequencies, side="right") - 1
    valid = (indices >= 0) & (indices < 32)
    window = np.hanning(width)
    power = np.abs(np.fft.rfft(chunk * window)) ** 2
    prior = np.abs(np.fft.rfft(previous * window)) ** 2
    return np.bincount(indices[valid], weights=np.maximum(power[valid] - 0.35 * prior[valid], 0.0), minlength=32)


def _overlap_templates(
    cues: Sequence[Mapping[str, Any]], samples: np.ndarray, coincidence_sec: float,
) -> Dict[str, Dict[str, Any]]:
    """Fit nonnegative sound spectra to cue incidence, including fully overlaid cues.

    Each distinct cue instant is one observation. A robust reweighting pass
    keeps a shifted or absent instance from defining its group's spectrum.
    Groups that always have exactly the same incidence remain unidentifiable.
    """
    groups = sorted({str(cue["group"]) for cue in cues})
    if not groups:
        return {}
    instants: List[float] = []
    rows: List[List[float]] = []
    for cue in sorted(cues, key=lambda item: float(item["time_sec"])):
        timestamp = float(cue["time_sec"])
        if instants and timestamp - instants[-1] <= coincidence_sec:
            rows[-1][groups.index(str(cue["group"]))] = 1.0
        else:
            instants.append(timestamp)
            row = [0.0] * len(groups)
            row[groups.index(str(cue["group"]))] = 1.0
            rows.append(row)
    design = np.asarray(rows)
    if len(instants) < 4:
        return {}
    # Some edits put a different overlay on *every* kick, so kick incidence
    # equals the sum of overlay incidences. The mixture can still identify
    # frequency components unique to each overlay. Only identical incidence
    # columns are wholly unidentifiable from these observations.
    indistinguishable = {
        group for index, group in enumerate(groups)
        if any(index != other and np.array_equal(design[:, index], design[:, other])
               for other in range(len(groups)))
    }
    observed = np.asarray([_spectral_attack(samples, time) for time in instants])
    weights = np.ones(len(instants))
    templates = np.zeros((len(groups), observed.shape[1]))
    for _ in range(5):
        weighted = design * np.sqrt(weights[:, None])
        templates = np.maximum(np.linalg.lstsq(weighted, observed * np.sqrt(weights[:, None]), rcond=None)[0], 0.0)
        residual = np.linalg.norm(observed - design @ templates, axis=1)
        scale = max(float(np.median(residual)), 1e-8)
        weights = np.clip(1.5 * scale / np.maximum(residual, 1e-8), 0.15, 1.0)
    result = {}
    for index, group in enumerate(groups):
        example_rows = design[:, index] > 0
        template = templates[index]
        if group in indistinguishable or example_rows.sum() < 3 or float(np.linalg.norm(template)) < 1e-7:
            continue
        # A group must contribute a measurable component after other cued
        # groups are explained. Otherwise its identity cannot be established.
        fitted = []
        for row_index in np.flatnonzero(example_rows):
            companions = design[row_index] @ templates - template
            scores = []
            for delta in (-0.030, -0.020, -0.010, 0.0, 0.010):
                at = instants[row_index] + delta
                if at < 0.0:
                    continue
                vector = np.maximum(_spectral_attack(samples, at) - companions, 0.0)
                scores.append(float(vector @ template) / max(float(template @ template), 1e-12) * _cosine(vector, template))
            fitted.append(max(scores, default=0.0))
        fitted = np.asarray(fitted)
        positive = fitted[fitted > 0.15]
        median_score = float(np.median(positive)) if len(positive) else 1.0
        result[group] = {
            "template": template, "median_score": median_score,
            "sample_count": int(example_rows.sum()), "method": "overlap_template",
            "example_times_sec": [instants[i] for i in np.flatnonzero(example_rows)],
        }
    return result


def _template_match(
    cue: Mapping[str, Any], cues: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]], samples: np.ndarray,
    tolerance_ms: float, search_window_ms: float, coincidence_sec: float,
) -> Dict[str, Any]:
    cue_time = float(cue["time_sec"])
    group = str(cue["group"])
    template = models[group]["template"]
    median = float(models[group]["median_score"])
    companions = _coincident_groups(cues, cue_time, coincidence_sec) - {group}
    companion_spectrum = sum((models[name]["template"] for name in companions if name in models), np.zeros_like(template))
    duration = len(samples) / SAMPLE_RATE
    start = max(0.0, cue_time - search_window_ms / 1000.0)
    end = min(duration, cue_time + search_window_ms / 1000.0)
    times = np.arange(start, end + 0.0001, 0.010)
    times = np.unique(np.r_[times, cue_time])
    scores = []
    norm = max(float(template @ template), 1e-12)
    for time in times:
        observed = _spectral_attack(samples, float(time))
        # Other cued sounds stay anchored at the cue, including when this
        # group's own sound is early or late.
        if abs(time - cue_time) < 0.065:
            observed = np.maximum(observed - companion_spectrum, 0.0)
        projection = float(observed @ template) / norm
        shape = _cosine(observed, template)
        scores.append(projection * shape)
    scores_array = np.asarray(scores)
    peak_indices = [i for i in range(1, len(times) - 1) if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]]
    peak_indices.extend((0, len(times) - 1, int(np.argmin(abs(times - cue_time)))))
    peak_indices = sorted(set(peak_indices), key=lambda i: scores[i], reverse=True)
    best = peak_indices[0]
    score = float(scores_array[best])
    threshold = max(0.40 * median, 0.15)
    rival = next((i for i in peak_indices[1:] if abs(times[i] - times[best]) >= 0.065 and scores[i] >= threshold), None)
    uncertain = rival is not None and scores[rival] >= 0.87 * score
    return {
        "time_sec": float(times[best]) if score >= threshold else None,
        "score": round(score, 4), "median_score": round(median, 4),
        "ambiguous_reason": "two comparable template peaks" if uncertain else None,
    }


def _spectral_track(samples: np.ndarray) -> np.ndarray:
    """A 10 ms grid of short-window band power for temporal sound templates."""
    hop = int(0.010 * SAMPLE_RATE)
    width = int(0.030 * SAMPLE_RATE)
    padded = np.pad(samples, (0, width))
    frames = np.lib.stride_tricks.sliding_window_view(padded, width)[::hop]
    window = np.hanning(width)
    frequencies = np.fft.rfftfreq(width, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(40.0, 20000.0, 41)
    slices = [(int(np.searchsorted(frequencies, edges[i])),
               int(np.searchsorted(frequencies, edges[i + 1]))) for i in range(40)]
    result = np.empty((len(frames), 40), dtype=np.float32)
    for start in range(0, len(frames), 256):
        power = np.abs(np.fft.rfft(frames[start:start + 256] * window, axis=1)) ** 2
        for band, (lo, hi) in enumerate(slices):
            result[start:start + len(power), band] = power[:, lo:max(lo + 1, hi)].sum(axis=1)
    return result


def _track_patch(track: np.ndarray, starts: np.ndarray, lag_steps: np.ndarray) -> np.ndarray:
    indices = np.rint(np.asarray(starts)[..., None] * 100.0).astype(int) + lag_steps
    valid = (indices >= 0) & (indices < len(track))
    return track[np.clip(indices, 0, len(track) - 1)] * valid[..., None]


def _soft_temporal_displacement_is_stronger(score: float, median_score: float, on_cue_score: float) -> bool:
    """Allow a below-floor shifted peak only when it clearly beats the cue peak."""
    return (score >= max(_TEMPORAL_SOFT_DISPLACED_SCORE_SHARE * median_score, 0.20)
            and score >= _TEMPORAL_SOFT_DISPLACED_WIN_RATIO * on_cue_score)


def _temporal_group_matches(
    cues: Sequence[Mapping[str, Any]], samples: np.ndarray, search_window_ms: float,
    tolerance_ms: float, bpm: Optional[float],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Contrast repeated cued sounds with uncued beats, then assign peaks once.

    The entire attack and swell envelope is aligned to the cue. The score is
    measured at the candidate *start*, so a whoosh's later energy maximum
    does not appear as an individual late cue.
    """
    if not cues:
        return {}, [], []
    track = _spectral_track(samples)
    lag_steps = np.arange(0, 33, 2, dtype=int)
    unique_times = sorted({float(cue["time_sec"]) for cue in cues})
    all_groups = sorted({str(cue["group"]) for cue in cues})
    offsets = np.arange(-search_window_ms / 1000.0, search_window_ms / 1000.0 + .0001, .010)
    group_results: Dict[int, Dict[str, Any]] = {}
    systematic: List[Dict[str, Any]] = []
    uncued: List[Dict[str, Any]] = []
    cue_times = np.asarray([float(cue["time_sec"]) for cue in cues])
    duration = len(samples) / float(SAMPLE_RATE)
    beat = 60.0 / bpm if bpm is not None else None

    def controls_for(group_times: np.ndarray) -> np.ndarray:
        """Comparison instants without this group: its own beat phase first.

        With a tempo, whole beats away from the group's cues carry the same
        backing (an off-beat tick is compared with other off-beats). Without
        one, or with too few, other cue times and then midpoints are used.
        """
        if beat is not None:
            grid = sorted({round(float(time) + step * beat, 4) for time in group_times
                           for step in range(-8, 9) if step})
            phase = np.asarray([time for time in grid
                                if .10 <= time <= duration - .40
                                and not np.any(np.abs(group_times - time) <= .030)])
            if len(phase) >= 4 and len(unique_times) > 1:
                # Only needed when the other cue times sit at another phase.
                others = np.asarray([time for time in unique_times
                                     if not np.any(np.abs(group_times - time) <= .005)])
                phases = lambda values: np.abs(((values / beat) + .5) % 1.0 - .5)
                if len(others) < 4 or abs(float(np.median(phases(others))) - float(np.median(phases(group_times)))) > .10:
                    return phase
        controls = np.asarray([time for time in unique_times
                               if not np.any(np.abs(group_times - time) <= .005)])
        if len(controls) < 4:
            spacing = np.median(np.diff(unique_times)) if len(unique_times) > 1 else 0.0
            controls = np.asarray(unique_times) + spacing / 2.0 if spacing > .15 else np.asarray([])
        return controls

    # Each group's raw attack pattern over its comparison instants. It lets
    # a coincident group's louder-than-usual instance (an accented kick) be
    # explained away before another group is scored at that cue.
    full_lags = np.arange(0, 35, dtype=int)
    raw_patterns: Dict[str, Dict[str, Any]] = {}
    for group in all_groups:
        members = np.asarray([cue_times[i] for i, cue in enumerate(cues)
                              if cue["group"] == group and cue.get("band") is None])
        if len(members) < 5:
            continue
        group_controls = controls_for(members)
        if beat is not None:
            # Off-beat instants carry the backing without this group's sound
            # (and without other beat-aligned cues), across all bands.
            between = np.asarray(sorted({round(float(time) + (step + .5) * beat, 4) for time in members
                                         for step in range(-4, 4)}))
            between = np.asarray([time for time in between if .10 <= time <= duration - .40
                                  and not np.any(np.abs(cue_times - time) <= .030)])
            if len(between) >= 4:
                group_controls = between
        if len(group_controls) < 4:
            continue
        raw_patterns[group] = {
            "pattern": np.maximum(np.median(_track_patch(track, members, full_lags), axis=0)
                                  - np.median(_track_patch(track, group_controls, full_lags), axis=0), 0.0),
        }
    companions_at = [
        {str(other["group"]) for other in cues
         if abs(float(other["time_sec"]) - float(cue["time_sec"])) <= .005} - {str(cue["group"])}
        for cue in cues
    ]

    for group in all_groups:
        indices = [i for i, cue in enumerate(cues) if cue["group"] == group and cue.get("band") is None]
        if len(indices) < _TEMPORAL_SMALL_GROUP_MIN_CUES:
            continue
        group_times = np.asarray([cue_times[i] for i in indices])
        controls = controls_for(group_times)
        if len(controls) < 4:
            continue
        full_baseline = np.median(_track_patch(track, controls, full_lags), axis=0)
        companion_gain: Dict[int, float] = {}
        control_share: Dict[str, float] = {}

        def explained_track(index: int) -> np.ndarray:
            """The track with co-occurring groups' above-usual energy removed at this cue.

            Only companions that the comparison instants normally contain
            are adjusted. Their excess gain is measured in their own
            strongest bands; their pattern times that gain is subtracted.
            This happens before the template is learned, so a click that
            always lands on an accented kick does not learn the accent.
            """
            first = int(round(cue_times[index] * 100.0))
            if first < 0 or first + len(full_lags) > len(track):
                return track
            removal = adjustments.get(index)
            if removal is None:
                return track
            adjusted = track.copy()
            adjusted[first:first + len(full_lags)] = np.maximum(adjusted[first:first + len(full_lags)] - removal, 0.0)
            return adjusted

        def companion_removal(index: int) -> Optional[np.ndarray]:
            """Above-usual energy of co-occurring groups at this cue, if any."""
            first = int(round(cue_times[index] * 100.0))
            if first < 0 or first + len(full_lags) > len(track):
                return None
            removal = None
            observed = track[first:first + len(full_lags)] - full_baseline
            for other in companions_at[index]:
                pattern = raw_patterns.get(other, {}).get("pattern")
                if pattern is None:
                    continue
                if other not in control_share:
                    other_times = cue_times[[i for i, cue in enumerate(cues) if cue["group"] == other]]
                    control_share[other] = float(np.mean([np.any(np.abs(other_times - time) <= .005)
                                                          for time in controls]))
                if control_share[other] < .5:
                    continue
                band_energy = np.sum(pattern, axis=0)
                strong = band_energy >= .25 * float(np.max(band_energy))
                energy = float(np.sum(pattern[:, strong]))
                if not np.any(strong) or energy <= 1e-9:
                    continue
                gain = float(np.clip(np.sum(observed[:, strong]) / energy, 0.0, 64.0))
                companion_gain[index] = max(companion_gain.get(index, 0.0), gain)
                if gain < .5:
                    continue
                removal = gain * pattern if removal is None else removal + gain * pattern
            return removal

        adjustments = {index: companion_removal(index) for index in indices}
        target = np.asarray([_track_patch(explained_track(index), np.asarray([cue_times[index]]), lag_steps)[0]
                             for index in indices])
        control = _track_patch(track, controls, lag_steps)
        baseline = np.median(control, axis=0)
        template = np.maximum(np.median(target, axis=0) - baseline, 0.0)
        variation = np.median(np.abs(control - baseline), axis=(0, 1))
        scale = np.maximum.reduce((variation, np.median(baseline, axis=0) * .12,
                                   np.full(track.shape[1], max(float(np.max(template)) * .002, 1e-8))))
        template /= scale[None, :]
        # Discard bands that do not consistently distinguish the sound from
        # the backing rhythm. This retains quiet but repeatable click bands.
        band_strength = np.linalg.norm(template, axis=0)
        excess_raw = np.maximum(np.median(target, axis=0) - baseline, 0.0)
        contrast = np.max(excess_raw, axis=0) / np.maximum(np.max(baseline, axis=0), 1e-7)
        contrast_floor = max(.8, .20 * min(float(np.max(contrast)), 20.0))
        eligible = contrast >= contrast_floor
        eligible_peak = float(np.max(band_strength[eligible])) if np.any(eligible) else 0.0
        keep = eligible & (band_strength >= max(eligible_peak * .025, 1e-7))
        if not np.any(keep):
            keep = band_strength >= max(float(np.max(band_strength)) * .12, 1e-7)
        centers = np.sqrt(np.geomspace(40.0, 20000.0, 41)[:-1]
                          * np.geomspace(40.0, 20000.0, 41)[1:])
        if any(word in group for word in ("click", "pop", "tick", "snap")):
            high = keep & (centers >= 600.0)
            if np.any(high):
                keep = high
        elif "whoosh" in group or "sweep" in group:
            high = keep & (centers >= 250.0)
            if np.any(high):
                keep = high
        template[:, ~keep] = 0.0
        norm = float(np.sum(template ** 2))
        if norm <= 1e-7:
            continue
        baseline_shift = np.asarray([
            np.median(_track_patch(track, controls + shift, lag_steps), axis=0)
            for shift in offsets
        ])

        def score_curve(at: float, source_track: np.ndarray = track) -> np.ndarray:
            observed = _track_patch(source_track, at + offsets, lag_steps)
            residual = (observed - baseline_shift) / scale[None, None, :]
            return np.maximum(np.sum(residual * template[None, :, :], axis=(1, 2)) / norm, 0.0)

        curves = {index: score_curve(cue_times[index], explained_track(index)) for index in indices}
        aligned = [float(np.max(curve[np.abs(offsets) <= .030])) for curve in curves.values()]
        median_score = float(np.median(aligned))
        if median_score <= 1e-8:
            continue
        mad_score = float(np.median(np.abs(np.asarray(aligned) - median_score)))
        credible_floor = max(0.20, max(.35 * median_score, min(.50 * median_score, median_score - 2.0 * mad_score)))
        remote_controls = [time for time in controls
                           if np.min(np.abs(group_times - time)) > search_window_ms / 1000.0 + .10]
        if len(remote_controls) < 4:
            remote_controls = list(controls)
        null_peaks = [float(np.max(score_curve(float(time)))) for time in remote_controls]
        shifted_floor = min(0.75 * median_score, max(credible_floor, 1.10 * float(np.percentile(null_peaks, 90))))
        envelope = np.sum(template ** 2, axis=1)
        attack_index = int(np.argmax(envelope >= .15 * float(np.max(envelope))))
        typical_lag_ms = int(lag_steps[attack_index] * 10)
        if typical_lag_ms > 80:
            systematic.append({"group": group, "lag_ms": typical_lag_ms,
                               "method": "group_template_attack", "classification": "informational"})
        pairs = []
        recovery_pairs = []
        slot_by_index = {}
        ordered = sorted(indices, key=lambda index: cue_times[index])
        beat_sec = 60.0 / bpm if bpm is not None else None
        for position, index in enumerate(ordered):
            previous = (cue_times[index] - cue_times[ordered[position - 1]]) / 2.0 if position else float("inf")
            following = (cue_times[ordered[position + 1]] - cue_times[index]) / 2.0 if position + 1 < len(ordered) else float("inf")
            primary_beat = .40 * beat_sec if beat_sec is not None else float("inf")
            recovery_beat = .65 * beat_sec if beat_sec is not None else float("inf")
            search_cap = max(tolerance_ms / 1000.0, search_window_ms / 1000.0 - .055)
            slot_by_index[index] = (
                min(previous, primary_beat, search_cap), min(following, primary_beat, search_cap),
                min(previous, recovery_beat, search_cap), min(following, recovery_beat, search_cap),
            )
        for index, curve in curves.items():
            cue_threshold = (.15 if cue_times[index] < .05 else .42) * median_score
            near = np.abs(offsets) <= tolerance_ms / 1000.0
            near_score = float(np.max(curve[near]))
            allowed = np.asarray([
                delta >= -slot_by_index[index][2] and delta <= slot_by_index[index][3]
                for delta in offsets
            ])
            credible_far = (~near) & allowed & (curve >= shifted_floor)
            far_score = float(np.max(curve[credible_far])) if np.any(credible_far) else 0.0
            prefer_near = near_score >= cue_threshold and not (cue_times[index] >= .05 and
                far_score >= 1.6 * near_score and far_score >= shifted_floor
            )
            for pos in range(len(offsets)):
                is_near = bool(near[pos])
                delta = float(offsets[pos])
                left_primary, right_primary, left_recovery, right_recovery = slot_by_index[index]
                if delta < -left_recovery or delta > right_recovery:
                    continue
                in_primary = -left_primary <= delta <= right_primary
                floor = cue_threshold if is_near else max(credible_floor, shifted_floor, .52 * median_score if not in_primary else 0.0)
                if curve[pos] < floor or (pos > 0 and curve[pos] < curve[pos - 1]) or (pos + 1 < len(curve) and curve[pos] < curve[pos + 1]):
                    continue
                event_time = cue_times[index] + offsets[pos]
                if event_time < 0 or event_time > len(samples) / SAMPLE_RATE:
                    continue
                # Neighbouring cues may see the same sound. Proximity settles
                # ties; each physical peak will be assigned at most once.
                priority = float(curve[pos]) / median_score - .35 * abs(float(offsets[pos])) / max(search_window_ms / 1000.0, .001)
                proposal = (priority, index, event_time, float(curve[pos]))
                recovery_pairs.append(proposal)
                if not (prefer_near and not is_near):
                    pairs.append(proposal)
        assigned: Dict[int, Tuple[float, float]] = {}
        soft_displacements = set()
        used_events: List[float] = []
        for _, index, event_time, score in sorted(pairs, key=lambda item: item[0], reverse=True):
            if index in assigned or any(abs(event_time - used) <= .075 for used in used_events):
                continue
            assigned[index] = (event_time, score)
            used_events.append(event_time)
        # An unassigned cue gets a second look at credible displaced peaks.
        # The first pass may have withheld them in favour of an on-cue peak
        # that was ultimately assigned to a neighbouring cue.
        for _, index, event_time, score in sorted(recovery_pairs, key=lambda item: item[0], reverse=True):
            if index in assigned or any(abs(event_time - used) <= .075 for used in used_events):
                continue
            assigned[index] = (event_time, score)
            used_events.append(event_time)

        def displaced_candidate(index: int, taken: Sequence[float]) -> Optional[Tuple[float, float, bool]]:
            """Best credible local peak inside the cue's slot but outside tolerance."""
            curve = curves[index]
            _, _, left, right = slot_by_index[index]
            near_score = float(np.max(curve[np.abs(offsets) <= .030]))
            best = None
            for pos in range(len(offsets)):
                delta = float(offsets[pos])
                if abs(delta) <= tolerance_ms / 1000.0 or delta < -left or delta > right:
                    continue
                score = float(curve[pos])
                soft_match = _soft_temporal_displacement_is_stronger(score, median_score, near_score)
                if score < max(credible_floor, 0.20) and not soft_match:
                    continue
                if (pos > 0 and score < curve[pos - 1]) or (pos + 1 < len(curve) and score < curve[pos + 1]):
                    continue
                event_time = cue_times[index] + delta
                if any(abs(event_time - used) <= .075 for used in taken):
                    continue
                if best is None or score > best[1]:
                    best = (event_time, score, score < max(credible_floor, 0.20))
            return best

        # A cue still without a match takes a credible displaced instance of
        # its group from its own slot: that is early/late, not missing.
        for index in sorted(indices, key=lambda item: -float(np.max(curves[item]))):
            if index in assigned:
                continue
            found = displaced_candidate(index, used_events)
            if found is not None:
                assigned[index] = (found[0], found[1])
                if found[2]:
                    soft_displacements.add(index)
                used_events.append(found[0])
        for index in indices:
            assigned_event = assigned.get(index)
            others_used = [used for used in used_events
                           if assigned_event is None or abs(used - assigned_event[0]) > 1e-9]
            alternative = displaced_candidate(index, others_used)
            best_score = float(np.max(curves[index]))
            event_time = None if assigned_event is None else assigned_event[0]
            event_score = best_score if assigned_event is None else assigned_event[1]
            if event_time is not None and abs(event_time - cue_times[index]) > tolerance_ms / 1000.0:
                if ((event_score < credible_floor or event_score < 0.20)
                        and index not in soft_displacements):
                    event_time = None
            group_results[index] = {
                "time_sec": event_time,
                "score": round(event_score, 4),
                "median_score": round(median_score, 4),
                "method": "temporal_group_template",
                "typical_attack_lag_ms": typical_lag_ms,
                "on_cue_score": round(float(np.max(curves[index][np.abs(offsets) <= .030])), 4),
                "shifted_null_floor": round(shifted_floor, 4),
                "credible_score_floor": round(credible_floor, 4),
                "slot_before_ms": round(slot_by_index[index][0] * 1000.0, 3),
                "slot_after_ms": round(slot_by_index[index][1] * 1000.0, 3),
                "displaced_time_sec": None if alternative is None else float(alternative[0]),
                "displaced_score": None if alternative is None else round(alternative[1], 4),
                "soft_displacement": index in soft_displacements,
                "companion_gain": round(companion_gain.get(index, 0.0), 3),
                "ambiguous_reason": None,
            }
        if len(indices) >= 5:
            uncued.extend(_uncued_instances(
                group, track, lag_steps, baseline, scale, template, norm, group_times, used_events,
                threshold=max(.85 * median_score, shifted_floor, credible_floor), median_score=median_score,
                clear_sec=search_window_ms / 1000.0 + .10, duration=duration,
            ))
    return group_results, systematic, uncued


def _uncued_instances(
    group: str, track: np.ndarray, lag_steps: np.ndarray, baseline: np.ndarray, scale: np.ndarray,
    template: np.ndarray, norm: float, group_times: np.ndarray, used_events: Sequence[float],
    threshold: float, median_score: float, clear_sec: float, duration: float,
) -> List[Dict[str, Any]]:
    """Strong instances of a group's sound far from every cue of that group.

    The whole file is scored against the group's template and must reach
    85% of the group's usual on-cue score, not merely clear the noise floor. Peaks within one search
    window of the group's own cues belong to those cues' timing checks.
    """
    times = np.arange(.10, max(.10, duration - .40), .010)
    if not len(times):
        return []
    scores = np.empty(len(times))
    for start in range(0, len(times), 2000):
        observed = _track_patch(track, times[start:start + 2000], lag_steps)
        residual = (observed - baseline[None, :, :]) / scale[None, None, :]
        scores[start:start + 2000] = np.maximum(np.sum(residual * template[None, :, :], axis=(1, 2)) / norm, 0.0)
    found: List[Dict[str, Any]] = []
    for pos in np.argsort(-scores):
        score = float(scores[pos])
        if score < threshold:
            break
        at = float(times[pos])
        if (pos > 0 and scores[pos - 1] > score) or (pos + 1 < len(scores) and scores[pos + 1] > score):
            continue
        if np.min(np.abs(group_times - at)) <= clear_sec or any(abs(at - used) <= .075 for used in used_events):
            continue
        if any(abs(at - item["time_sec"]) <= .150 for item in found):
            continue
        found.append({"time_sec": round(at, 3), "group": group, "score": round(score, 4),
                      "group_median_score": round(median_score, 4), "threshold": round(threshold, 4)})
    return sorted(found, key=lambda item: item["time_sec"])


# Third-octave bands for the transparent second-opinion check. They are
# narrow enough that a click, a whoosh and a kick each have a band of their own.
_SIMPLE_BANDS: Tuple[Tuple[float, float], ...] = tuple(
    (round(1000.0 * 2.0 ** (step / 3.0 - 1.0 / 6.0), 1), round(1000.0 * 2.0 ** (step / 3.0 + 1.0 / 6.0), 1))
    for step in range(-13, 13)
)
_SIMPLE_LEVEL_WINDOW = (-0.010, 0.090)
_SIMPLE_PRE_WINDOW = (-0.110, -0.020)
_SIMPLE_LEVEL_MARGIN_DB = 10.0
_SIMPLE_EXCESS_DB = 8.0
_SIMPLE_ATTACK_DB = 3.0
_SIMPLE_ABSENT_DB = 6.0
_SIMPLE_ABSENT_BELOW_TYPICAL_DB = 6.0
_SIMPLE_COMPARISONS = 8
_TEMPORAL_SMALL_GROUP_MIN_CUES = 3
_TEMPORAL_SOFT_DISPLACED_SCORE_SHARE = 0.35
_TEMPORAL_SOFT_DISPLACED_WIN_RATIO = 1.20
_SIMPLE_STRONG_DISPLACED_PEER_GAIN_DB = 3.0
_SIMPLE_STRONG_DISPLACED_CUE_GAIN_DB = 5.0
_SIMPLE_STRONG_DISPLACED_REFERENCE_GAIN_DB = 3.0
# Cues closer than this to the start of the file have no pre-roll: neither an
# attack rise nor the comparison moments (which start at 0.12 s) can be measured.
_START_PREROLL_SEC = 0.12
_USUAL_LEVEL_MARGIN_DB = 3.0
_SOFT_ATTACK_BUILD_BEATS = 4.0
_SOFT_ATTACK_BUILD_FIRST_PASS_BEATS = 2.0
_SOFT_ATTACK_BUILD_BINS = 4
_SOFT_ATTACK_BUILD_EXTENDED_BINS = 8
_SOFT_ATTACK_BUILD_MIN_RISE_DB = 6.0
_SOFT_ATTACK_BUILD_MIN_END_EXCESS_DB = 3.0
_SOFT_ATTACK_BUILD_MIN_PRE_MARGIN_DB = 3.0
_SOFT_ATTACK_BUILD_MIN_STEADY_STEPS = 2
_SOFT_ATTACK_BUILD_MIN_STEP_DB = 0.5
_SOFT_ATTACK_BUILD_MIN_COMPARISONS = 2


def _window_level_db(samples: np.ndarray, start: float, end: float, band: Mapping[str, Any]) -> float:
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(len(samples), int(round(end * SAMPLE_RATE)))
    if last - first < int(0.010 * SAMPLE_RATE):
        return -120.0
    return float(_band_energy_db(samples[first:last], dict(band)))


class _BandLevels:
    """dB level of every simple-check band over a short window, one FFT per moment."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples
        width = int(round((_SIMPLE_LEVEL_WINDOW[1] - _SIMPLE_LEVEL_WINDOW[0]) * SAMPLE_RATE))
        self.width = width
        self.size = 1 << int(math.ceil(math.log2(width)))
        self.window = np.hanning(width)
        self.norm = max(1.0, float(np.sum(self.window * self.window)))
        frequencies = np.fft.rfftfreq(self.size, d=1.0 / SAMPLE_RATE)
        self.masks = [(frequencies >= low) & (frequencies < high) for low, high in _SIMPLE_BANDS]
        self.cache: Dict[int, np.ndarray] = {}

    def at(self, start: float) -> np.ndarray:
        """Levels for the window beginning at `start` seconds."""
        first = int(round(start * SAMPLE_RATE))
        cached = self.cache.get(first)
        if cached is not None:
            return cached
        chunk = np.zeros(self.width, dtype=np.float64)
        lo, hi = max(0, first), min(len(self.samples), first + self.width)
        if hi > lo:
            chunk[lo - first:hi - first] = self.samples[lo:hi]
        power = np.abs(np.fft.rfft(chunk * self.window, n=self.size)) ** 2 / self.norm
        if len(power) > 2:
            power[1:-1] *= 2.0
        levels = np.asarray([_db(float(np.sum(power[mask]))) for mask in self.masks])
        self.cache[first] = levels
        return levels

    def cue(self, at: float) -> np.ndarray:
        return self.at(at + _SIMPLE_LEVEL_WINDOW[0])

    def before(self, at: float) -> np.ndarray:
        return self.at(at + _SIMPLE_PRE_WINDOW[0])


def _comparison_moments(
    at: float, group: str, cues: Sequence[Mapping[str, Any]], beat_sec: Optional[float],
    duration: float, coincidence_sec: float,
) -> List[float]:
    """The nearest moments without this group's cue, preferring the same companion cues.

    Candidates are the half-beat grid within 16 beats (with a tempo), plus
    other cue times and the midpoints between cue times within 8 s.
    """
    near = max(coincidence_sec, 0.030)

    def companions(moment: float) -> frozenset:
        return frozenset(str(cue["group"]) for cue in cues if abs(float(cue["time_sec"]) - moment) <= near)

    own_times = [float(cue["time_sec"]) for cue in cues if str(cue["group"]) == group]
    unique = sorted({float(cue["time_sec"]) for cue in cues})
    candidates = unique + [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    candidates = [moment for moment in candidates if abs(moment - at) <= 8.0]
    if beat_sec is not None:
        candidates += [at + step * beat_sec / 2.0 for step in range(-32, 33) if step]
    candidates = [moment for moment in sorted({round(value, 4) for value in candidates})
                  if 0.12 <= moment <= duration - 0.10
                  and all(abs(moment - own) > near for own in own_times)]
    target = companions(at) - {group}
    labelled = [(moment, companions(moment)) for moment in candidates]
    # Prefer moments with exactly the same other cues; then moments that at
    # least contain them; then any moment without this group's cue.
    for choose in (lambda other: other == target, lambda other: bool(target) and target <= other):
        chosen = [moment for moment, other in labelled if choose(other)]
        if len(chosen) >= 2:
            break
    else:
        chosen = candidates
    return sorted(chosen, key=lambda moment: abs(moment - at))[:_SIMPLE_COMPARISONS]


def _measure_soft_attack_build(
    samples: np.ndarray,
    at: float,
    band: Optional[Mapping[str, Any]],
    beat_sec: Optional[float],
    comparison_times: Sequence[float],
    next_cue_sec: Optional[float] = None,
    _span_sec: Optional[float] = None,
    _bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure a gradual build against the same beat span before and elsewhere.

    The cue band is used when it is reliable; otherwise the fallback is the
    full 0-to-Nyquist band. It checks two beats first and can extend to four
    beats, stopping at the next same-group cue or file end. Each span is split
    into four equal energy windows, or eight when extended. A build is
    measured only when its level relative to matching
    comparison moments rises by at least 6 dB, ends at least 3 dB above those
    moments, rises by at least 3 dB more than the similarly corrected span
    just before the cue, and increases by at least 0.5 dB in two steps.
    Fewer than two valid comparison moments makes the result
    inconclusive; this avoids calling an isolated level change a build.
    """
    duration = len(samples) / float(SAMPLE_RATE)
    available = max(0.0, duration - float(at))
    if next_cue_sec is not None:
        available = min(available, max(0.0, float(next_cue_sec) - float(at)))
    span = (_span_sec if _span_sec is not None else
            (_SOFT_ATTACK_BUILD_BEATS * beat_sec if beat_sec is not None else 1.0))
    span = min(float(span), available)
    first_span = (
        min(_SOFT_ATTACK_BUILD_FIRST_PASS_BEATS * beat_sec, available)
        if beat_sec is not None else span
    )
    bins = _bins or (
        _SOFT_ATTACK_BUILD_EXTENDED_BINS
        if beat_sec is not None and span > first_span + 0.001 else _SOFT_ATTACK_BUILD_BINS
    )
    result: Dict[str, Any] = {
        "conclusive": False,
        "measured": False,
        "band": "broadband" if band is None else "{:g}-{:g} Hz".format(
            float(band["low_hz"]), float(band["high_hz"])),
        "window_sec": round(span, 3),
        "comparison_count": 0,
        "relative_rise_db": None,
        "prior_relative_rise_db": None,
        "end_excess_db": None,
        "steady_steps": 0,
    }

    def first_pass_check() -> Optional[Dict[str, Any]]:
        if beat_sec is None or _span_sec is not None:
            return None
        first_pass = min(_SOFT_ATTACK_BUILD_FIRST_PASS_BEATS * beat_sec, available)
        if first_pass + 0.001 >= span:
            return None
        return _measure_soft_attack_build(
            samples, at, band, beat_sec, comparison_times, next_cue_sec, first_pass,
            _SOFT_ATTACK_BUILD_BINS,
        )

    if span < 0.4 or float(at) < span:
        return result
    measure_band = ({"name": None, "low_hz": 0.0, "high_hz": SAMPLE_RATE / 2.0 + 1.0}
                    if band is None else dict(band))
    step = span / bins
    if step < 0.010:
        return result
    controls = []
    for moment in comparison_times:
        time = float(moment)
        if time < span or time + span > duration:
            continue
        if all(abs(time - previous) > 0.05 for previous in controls):
            controls.append(time)
    result["comparison_count"] = len(controls)
    if len(controls) < _SOFT_ATTACK_BUILD_MIN_COMPARISONS:
        short_check = first_pass_check()
        if short_check is not None and (short_check["measured"] or short_check["conclusive"]):
            return short_check
        return result

    def profile(start: float) -> np.ndarray:
        return np.asarray([
            _window_level_db(samples, start + index * step, start + (index + 1) * step, measure_band)
            for index in range(bins)
        ], dtype=np.float64)

    cue_post = profile(float(at))
    cue_pre = profile(float(at) - span)
    control_post = np.median(np.asarray([profile(time) for time in controls]), axis=0)
    control_pre = np.median(np.asarray([profile(time - span) for time in controls]), axis=0)
    relative_post = cue_post - control_post
    relative_pre = cue_pre - control_pre
    relative_rise = float(relative_post[-1] - relative_post[0])
    prior_rise = float(relative_pre[-1] - relative_pre[0])
    end_excess = float(relative_post[-1])
    steady_steps = int(np.count_nonzero(np.diff(relative_post) >= _SOFT_ATTACK_BUILD_MIN_STEP_DB))
    result.update({
        "conclusive": True,
        "relative_rise_db": round(relative_rise, 2),
        "prior_relative_rise_db": round(prior_rise, 2),
        "end_excess_db": round(end_excess, 2),
        "steady_steps": steady_steps,
        "measured": (
            relative_rise >= _SOFT_ATTACK_BUILD_MIN_RISE_DB
            and end_excess >= _SOFT_ATTACK_BUILD_MIN_END_EXCESS_DB
            and relative_rise - prior_rise >= _SOFT_ATTACK_BUILD_MIN_PRE_MARGIN_DB
            and steady_steps >= _SOFT_ATTACK_BUILD_MIN_STEADY_STEPS
        ),
    })
    short_check = first_pass_check()
    if short_check is not None:
        if short_check["measured"]:
            return short_check
        if result["measured"]:
            return result
        if short_check["conclusive"]:
            return short_check
    return result


def _no_band_cue_outcome(build_check: Mapping[str, Any]) -> Tuple[str, Optional[str], str]:
    """Keep a no-band cue missing unless measured audio shows a soft attack."""
    if build_check.get("measured"):
        return (
            "ambiguous",
            "a gradual energy build follows the cue without a sharp attack; review the following beats",
            "gradual_build_after_cue",
        )
    return "missing", None, "no_band_beyond_companions"


def _power(db: Any) -> Any:
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)


def _simple_dominant_band_check(
    samples: np.ndarray,
    normalized: Sequence[Mapping[str, Any]],
    onsets_by_band: Dict[str, List[Dict[str, Any]]],
    source: str,
    cache: MediaCache,
    tolerance_sec: float,
    coincidence_sec: float,
    beat_sec: Optional[float],
) -> Dict[int, Dict[str, Any]]:
    """Transparent second opinion: an attack in the group's own band, beyond what else explains.

    Each cue is compared with nearby comparison moments: moments without this
    group's cue that carry the same other cues where possible. When a
    comparison moment lacks one of the cue's co-occurring groups, that group's
    typical contribution (its median excess over its own comparisons) is
    added, so energy from a coincident kick or click is explained away.

    A group's dominant band is the third-octave band where its cues most
    exceed both that expected level and the preceding 90 ms (median over the
    group). Bands already claimed by a more distinctive co-occurring group
    are skipped, so a louder-than-usual (accented) kick cannot become a
    click's band. A cue is `present` when, in that band:
    - an onset lies within tolerance;
    - the level is no more than 10 dB below the group's median level
      (louder-than-usual instances always pass);
    - the level is at least 8 dB above the expected level (6 dB when it is
      within 3 dB of the group's usual level); and
    - the level rises at least 3 dB over the preceding 90 ms (an attack, not
      a sustained pad or riser).
    """
    duration = len(samples) / float(SAMPLE_RATE)
    levels = _BandLevels(samples)
    near = max(coincidence_sec, 0.030)
    grouped: Dict[str, List[int]] = {}
    for index, cue in enumerate(normalized):
        grouped.setdefault(str(cue["group"]), []).append(index)

    def companions(moment: float) -> frozenset:
        return frozenset(str(cue["group"]) for cue in normalized if abs(float(cue["time_sec"]) - moment) <= near)

    comparisons = {
        index: _comparison_moments(float(cue["time_sec"]), str(cue["group"]), normalized,
                                   beat_sec, duration, coincidence_sec)
        for index, cue in enumerate(normalized)
    }
    target_of = {index: companions(float(cue["time_sec"])) - {str(cue["group"])}
                 for index, cue in enumerate(normalized)}
    moment_groups = {index: [companions(moment) for moment in comparisons[index]] for index in comparisons}

    def learn_contributions(known: Mapping[str, np.ndarray], matched_only: bool) -> Dict[str, np.ndarray]:
        """Each group's typical contribution per band, over its own comparisons.

        A comparison moment that lacks one of the cue's co-occurring groups
        gets that group's known contribution added first, so a click that
        always lands on a kick does not learn the kick as part of itself.
        """
        learned: Dict[str, np.ndarray] = {}
        for group_name, indices in grouped.items():
            excess = []
            for index in indices:
                if len(comparisons[index]) < 2:
                    continue
                missing = [target_of[index] - present for present in moment_groups[index]]
                if matched_only and any(missing):
                    continue
                own = _power(levels.cue(float(normalized[index]["time_sec"])))
                rows = []
                for moment, absent in zip(comparisons[index], missing):
                    row = _power(levels.cue(moment))
                    for other in absent:
                        row = row + known.get(other, 0.0)
                    rows.append(row)
                excess.append(np.maximum(own - np.median(np.asarray(rows), axis=0), 0.0))
            if excess:
                learned[group_name] = np.median(np.asarray(excess), axis=0)
        return learned

    # First from cues whose comparisons carry the same co-occurring groups
    # (or that have none); then every cue, explaining away the others.
    contribution = learn_contributions(learn_contributions({}, matched_only=True), matched_only=False)

    all_cue_times = np.asarray(sorted({float(cue["time_sec"]) for cue in normalized}))

    def expected_power(anchor_index: int, shift: float = 0.0) -> Tuple[Optional[np.ndarray], str]:
        """Expected band power at the cue from everything except this group."""
        pairs = list(zip(comparisons[anchor_index], moment_groups[anchor_index]))
        if abs(shift) > near:
            # A shifted comparison window must not land on another cued sound.
            pairs = [(moment, present) for moment, present in pairs
                     if not np.any((all_cue_times >= moment + shift - near)
                                   & (all_cue_times <= moment + shift + _SIMPLE_LEVEL_WINDOW[1]))]
        if len(pairs) < 2:
            return None, "preceding_90ms"
        rows = []
        for moment, present in pairs:
            row = _power(levels.cue(moment + shift))
            # Companions are anchored at the cue; at a displaced time they
            # are not expected, so only the cue instant is adjusted.
            if abs(shift) <= near:
                for other in target_of[anchor_index] - present:
                    row = row + contribution.get(other, 0.0)
            rows.append(row)
        return np.median(np.asarray(rows), axis=0), "comparison_moments_explained"

    band_scores: Dict[str, np.ndarray] = {}
    for group_name, indices in grouped.items():
        per_cue = []
        for index in indices:
            time = float(normalized[index]["time_sec"])
            own = levels.cue(time)
            expected, _ = expected_power(index)
            reference = levels.before(time) if expected is None else 10.0 * np.log10(np.maximum(expected, 1e-12))
            rise = own - levels.before(time)
            per_cue.append(np.minimum(own - np.maximum(reference, -90.0), np.maximum(rise, -90.0)))
        score = np.median(np.asarray(per_cue), axis=0)
        loud = np.median(np.asarray([levels.cue(float(normalized[i]["time_sec"])) for i in indices]), axis=0)
        score[loud <= -60.0] = -np.inf
        band_scores[group_name] = score

    # Most distinctive groups pick first; co-occurring groups do not share a band.
    cooccurs = {name: set().union(*(target_of[i] for i in indices)) for name, indices in grouped.items()}
    chosen_band: Dict[str, int] = {}
    for group_name in sorted(grouped, key=lambda name: -float(np.max(band_scores[name]))):
        score = band_scores[group_name].copy()
        for other in cooccurs[group_name]:
            if other in chosen_band:
                taken = chosen_band[other]
                score[max(0, taken - 1):taken + 2] = -np.inf
        if not np.any(np.isfinite(score)):
            score = band_scores[group_name]
        chosen_band[group_name] = int(np.argmax(score))

    results: Dict[int, Dict[str, Any]] = {}
    for group_name, indices in grouped.items():
        times = [float(normalized[i]["time_sec"]) for i in indices]
        explicit = next((normalized[i]["band"] for i in indices if normalized[i].get("band") is not None), None)
        if explicit is not None:
            band = {"name": explicit.get("name"), "low_hz": explicit["low_hz"], "high_hz": explicit["high_hz"]}
            band_index = None
        else:
            band_index = chosen_band[group_name]
            low, high = _SIMPLE_BANDS[band_index]
            band = {"name": None, "low_hz": low, "high_hz": high}
        band_name = band.get("name") or "{:g}-{:g} Hz".format(band["low_hz"], band["high_hz"])
        # Onsets use the octave around the band: the onset detector's FFT is
        # too coarse for low third-octaves.
        center = math.sqrt(band["low_hz"] * band["high_hz"])
        onset_low = min(band["low_hz"], center / math.sqrt(2.0))
        onset_high = min(SAMPLE_RATE / 2.0, max(band["high_hz"], center * math.sqrt(2.0), onset_low + 120.0))
        key = "simple:{:.1f}:{:.1f}".format(onset_low, onset_high)
        if key not in onsets_by_band:
            onsets_by_band[key] = audio_onsets(
                source, min_strength=0.08, cache=cache, band_hz=[onset_low, onset_high],
            )["onsets"]
        band_onsets = onsets_by_band[key]

        def band_level(at: float, window: str = "cue", band=band, band_index=band_index) -> float:
            if band_index is None:
                offset = _SIMPLE_LEVEL_WINDOW if window == "cue" else _SIMPLE_PRE_WINDOW
                return _window_level_db(samples, at + offset[0], at + offset[1], band)
            return float((levels.cue(at) if window == "cue" else levels.before(at))[band_index])

        audible = [level for level in (band_level(time) for time in times) if level > -60.0]
        typical = float(np.median(audible)) if len(audible) >= 3 else None

        def level_test(at: float, anchor_index: int, band_level=band_level, typical=typical,
                       band_index=band_index, band=band) -> Dict[str, Any]:
            anchor = float(normalized[anchor_index]["time_sec"])
            level = band_level(at)
            before = band_level(at, "before")
            reference_kind = "preceding_90ms"
            reference = before
            if band_index is not None:
                expected, kind = expected_power(anchor_index, at - anchor)
                if expected is not None:
                    reference, reference_kind = float(10.0 * np.log10(max(float(expected[band_index]), 1e-12))), kind
            elif len(comparisons[anchor_index]) >= 2:
                reference = float(np.median([band_level(moment + at - anchor) for moment in comparisons[anchor_index]]))
                reference_kind = "comparison_moments"
            passed = (level > -60.0
                      and (typical is None or level >= typical - _SIMPLE_LEVEL_MARGIN_DB)
                      and (level - reference >= (_SIMPLE_EXCESS_DB if reference_kind != "preceding_90ms" else 6.0)
                           or (typical is not None and level >= typical - 3.0 and level - reference >= 6.0))
                      and level - before >= _SIMPLE_ATTACK_DB)
            return {"passed": passed, "level_db": level, "reference_db": reference,
                    "reference_kind": reference_kind, "rise_db": level - before}

        for index, time in zip(indices, times):
            near_onsets = [onset for onset in band_onsets if abs(float(onset["time_sec"]) - time) <= tolerance_sec]
            best_onset = min(near_onsets, key=lambda onset: abs(float(onset["time_sec"]) - time), default=None)
            test = level_test(time, index)
            results[index] = {
                "dominant_band": band_name,
                "band": band,
                "typical_level_db": typical,
                "cue_level_db": test["level_db"],
                "reference_level_db": test["reference_db"],
                "reference_kind": test["reference_kind"],
                "rise_db": test["rise_db"],
                "level_ok": test["passed"],
                "on_time": best_onset is not None and test["passed"],
                "onset_time": None if best_onset is None else float(best_onset["time_sec"]),
                "offset_ms": None if best_onset is None else round((float(best_onset["time_sec"]) - time) * 1000.0, 3),
                "onsets": band_onsets,
                "level_test": level_test,
                "cue_index": index,
            }
    return results


def _small_group_peer_level_drop_db(
    cue_index: int,
    normalized: Sequence[Mapping[str, Any]],
    simple_checks: Mapping[int, Mapping[str, Any]],
) -> Optional[float]:
    """Return a clear peer-level outlier for a repeated cue group, if present."""
    cue = normalized[cue_index]
    group_indices = [index for index, other in enumerate(normalized)
                     if str(other["group"]) == str(cue["group"])]
    if len(group_indices) < 2:
        return None
    check = simple_checks[cue_index]
    band = check.get("dominant_band")
    peers = [
        float(simple_checks[index]["cue_level_db"])
        for index in group_indices if index != cue_index
        and simple_checks[index].get("dominant_band") == band
        and float(simple_checks[index].get("cue_level_db", -120.0)) > -60.0
    ]
    if not peers:
        return None
    peer_median = float(np.median(peers))
    drop = peer_median - float(check.get("cue_level_db", -120.0))
    if len(group_indices) == 2:
        return drop

    # With several examples, accept only a low level that is a clear outlier
    # against the spread of the other instances. A two-peer sample needs a
    # larger absolute gap because its spread estimate is fragile.
    peer_mad = float(np.median(np.abs(np.asarray(peers) - peer_median)))
    minimum_drop = 6.0 if len(peers) < 3 else 3.0
    required_drop = max(minimum_drop, 3.0 * 1.4826 * peer_mad)
    return drop if drop >= required_drop else None


def _small_group_match_is_unsupported(
    cue_index: int,
    normalized: Sequence[Mapping[str, Any]],
    simple_checks: Mapping[int, Mapping[str, Any]],
) -> bool:
    """A repeated-group match is suspect when its own band is far below peers."""
    drop = _small_group_peer_level_drop_db(cue_index, normalized, simple_checks)
    group_size = sum(str(other["group"]) == str(normalized[cue_index]["group"])
                     for other in normalized)
    if group_size == 2:
        return drop is not None and drop > _SIMPLE_LEVEL_MARGIN_DB
    return drop is not None


def _present_at_file_start(cue_time: float, simple: Mapping[str, Any]) -> bool:
    """At the very start there is no pre-roll to show an attack.

    A cue there whose group band is at the group's usual level is probably
    present, so it should be reviewed rather than reported missing.
    """
    if cue_time >= _START_PREROLL_SEC:
        return False
    typical = simple.get("typical_level_db")
    level = float(simple.get("cue_level_db", -120.0))
    return level > -60.0 and (typical is None or level >= float(typical) - _SIMPLE_ABSENT_BELOW_TYPICAL_DB)


_NO_SIGNATURE_REASON = "coincident cue has no unambiguous group signature"


def _excess_over_companions_db(
    levels: "_BandLevels", at: float, group: str, cues: Sequence[Mapping[str, Any]],
    beat_sec: Optional[float], duration: float, coincidence_sec: float,
) -> Optional[float]:
    """How far the loudest band at a cue rises above moments with the same other cues.

    A cue group with no learned signature (a one-off sound landing on a kick)
    cannot be told apart from its companions, but when no band at all stands
    out beyond what the companions make elsewhere, its sound is not there.
    """
    moments = _comparison_moments(at, group, cues, beat_sec, duration, coincidence_sec)
    if len(moments) < 2:
        return None
    reference = np.median([levels.cue(moment) for moment in moments], axis=0)
    return float(np.max(levels.cue(at) - reference))


def _band_at_usual_level(simple: Mapping[str, Any]) -> bool:
    """The group's own band sits at its usual level at the cue.

    Both matchers can miss a present sound when a co-occurring cue's energy
    is credited with it; a missing sound leaves its band well below usual.
    """
    typical = simple.get("typical_level_db")
    level = float(simple.get("cue_level_db", -120.0))
    return typical is not None and level > -60.0 and level >= float(typical) - _USUAL_LEVEL_MARGIN_DB


_SAME_TIME_SEC = 0.010
_UNCUED_MAX_SHARE = 0.25
_UNCUED_ATTACK_DB = 6.0
# The fallback band must be one of the group's strongest: within this much of
# the band where its cues rise most, not a companion's band it merely shares.
_UNCUED_OWN_BAND_MARGIN_DB = 6.0


def _uncued_attack_band(
    levels: "_BandLevels", at: float, cue_times: Sequence[float], claimed: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The candidate's own attack, in a band where the group's cues attack too.

    The group's simple-check band is chosen so co-occurring groups do not
    share one, so a sound that always lands with others (a ping on every scene
    change) can end up with a band it barely reaches. Here the band is the
    one where the candidate itself rises most, skipping bands (and their
    neighbours) claimed by other cues at that moment. The group's own cues must
    rise there by 6 dB too, within 6 dB of their strongest band (so it is the
    group's own sound, not a companion's it shares), and the candidate must
    reach within 3 dB of their usual level in it.
    """
    rise = levels.cue(at) - levels.before(at)
    blocked = np.zeros(len(_SIMPLE_BANDS), dtype=bool)
    for band in claimed:
        for index, (low, high) in enumerate(_SIMPLE_BANDS):
            if low < float(band["high_hz"]) and high > float(band["low_hz"]):
                blocked[max(0, index - 1):index + 2] = True
    rise = np.where(blocked, -np.inf, rise)
    usable = [time for time in cue_times if time >= _START_PREROLL_SEC]
    if not np.any(np.isfinite(rise)) or len(usable) < 2:
        return None
    band = int(np.argmax(rise))
    cue_rises = np.median(np.asarray([levels.cue(time) - levels.before(time) for time in usable]), axis=0)
    cue_rise = float(cue_rises[band])
    if cue_rise < float(np.max(cue_rises)) - _UNCUED_OWN_BAND_MARGIN_DB:
        return None
    typical = float(np.median([levels.cue(time)[band] for time in usable]))
    level = float(levels.cue(at)[band])
    if rise[band] < _UNCUED_ATTACK_DB or cue_rise < _UNCUED_ATTACK_DB or level < typical - _USUAL_LEVEL_MARGIN_DB:
        return None
    low, high = _SIMPLE_BANDS[band]
    return {"band": "{:g}-{:g} Hz".format(low, high), "level_db": round(level, 2), "typical_level_db": round(typical, 2),
            "rise_db": round(float(rise[band]), 2), "cue_rise_db": round(cue_rise, 2)}
_SHARED_GROUP_SHARE = 0.8


def _attribute_shared_displacement(cue_reports: List[Dict[str, Any]]) -> None:
    """Blame a shared move on the group that always rides along.

    When one group's cues (say whooshes) nearly always coincide with another
    group's cues (text pops), the broader group's learned signature carries
    the narrower sound too. If the narrower sound moves, both cues then
    "match" at the same displaced moment. The move belongs to the narrower
    group; the broader cue is marked ambiguous instead of a fault. If it
    really moved too, the re-check after the fix still reports it.
    """
    times_by_group: Dict[str, List[float]] = {}
    for report in cue_reports:
        times_by_group.setdefault(str(report.get("cue_group")), []).append(float(report["time_sec"]))

    def rides_along(narrow: str, broad: str) -> bool:
        narrow_times, broad_times = times_by_group.get(narrow, []), times_by_group.get(broad, [])
        if not narrow_times or len(broad_times) <= len(narrow_times):
            return False
        shared = sum(1 for time in narrow_times if any(abs(time - other) <= _SAME_TIME_SEC for other in broad_times))
        return shared >= _SHARED_GROUP_SHARE * len(narrow_times)

    moved = [report for report in cue_reports
             if report.get("status") in {"early", "late"} and report.get("matched_onset_time_sec") is not None]
    for broad in moved:
        for narrow in moved:
            if (narrow is broad or narrow.get("cue_group") == broad.get("cue_group")
                    or abs(float(narrow["time_sec"]) - float(broad["time_sec"])) > _SAME_TIME_SEC
                    or abs(float(narrow["matched_onset_time_sec"]) - float(broad["matched_onset_time_sec"]))
                    > _SAME_TIME_SEC
                    or not rides_along(str(narrow.get("cue_group")), str(broad.get("cue_group")))):
                continue
            broad["status"] = "ambiguous"
            broad["evidence"] = "shared_displacement"
            broad["ambiguous_reason"] = (
                "matched at the same moved spot ({:.3f} s) as '{}'; nearly every '{}' cue comes with a '{}' cue, "
                "so this match is probably that sound. Re-check after fixing it".format(
                    float(narrow["matched_onset_time_sec"]), narrow.get("label"), narrow.get("cue_group"),
                    broad.get("cue_group")))
            break


def _simple_displaced_attack(
    simple: Mapping[str, Any], cue_time: float, before_sec: float, after_sec: float, tolerance_sec: float,
) -> Optional[float]:
    """The loudest passing attack of the group's band inside the slot but outside tolerance."""
    best = None
    for onset in simple["onsets"]:
        onset_time = float(onset["time_sec"])
        delta = onset_time - cue_time
        if abs(delta) <= tolerance_sec or delta < -before_sec or delta > after_sec:
            continue
        test = simple["level_test"](onset_time, simple["cue_index"])
        if test["passed"] and (best is None or test["level_db"] > best[1]):
            best = (onset_time, test["level_db"])
    return None if best is None else best[0]


def _own_band_displaced(
    simple: Mapping[str, Any], cue_index: int, normalized: Sequence[Mapping[str, Any]],
    template_match: Optional[Mapping[str, Any]], search_window_sec: float, tolerance_sec: float,
) -> Optional[float]:
    """Where this cue's own sound really is, when its band is missing at the cue.

    Only when the group's band at the cue is more than 10 dB below the group's
    usual level (the simple check's own "not this sound" margin) and a passing
    attack of that band lies in the cue slot, away from the group's usual
    attack lag and nearer this cue than any other cue of the same group.
    """
    typical = simple.get("typical_level_db")
    if typical is None or float(simple["cue_level_db"]) >= float(typical) - _SIMPLE_LEVEL_MARGIN_DB:
        return None
    cue_time = float(normalized[cue_index]["time_sec"])
    before = after = search_window_sec
    if template_match is not None and template_match.get("slot_before_ms") is not None:
        before = float(template_match["slot_before_ms"]) / 1000.0
        after = float(template_match.get("slot_after_ms") or 0.0) / 1000.0
    displaced = _simple_displaced_attack(simple, cue_time, before, after, tolerance_sec)
    if displaced is None:
        return None
    # A group that always sounds a little after its cue (a pop with a slow
    # attack) is on time at its usual lag.
    usual_lag = float((template_match or {}).get("typical_attack_lag_ms") or 0.0) / 1000.0
    if abs(displaced - cue_time - usual_lag) <= tolerance_sec:
        return None
    group = normalized[cue_index]["group"]
    if any(other_index != cue_index and other["group"] == group
           and abs(float(other["time_sec"]) - displaced) < abs(cue_time - displaced)
           for other_index, other in enumerate(normalized)):
        return None
    return displaced


def _simple_supports_time(simple: Mapping[str, Any], at: float, tolerance_sec: float) -> bool:
    """True when the simple check also sees the displaced sound near `at`.

    Swells are detected a little after their start, so onsets up to 150 ms
    after the template's start time count.
    """
    for onset in simple["onsets"]:
        onset_time = float(onset["time_sec"])
        if at - tolerance_sec <= onset_time <= at + max(tolerance_sec, 0.150):
            if simple["level_test"](onset_time, simple["cue_index"])["passed"]:
                return True
    return False


def _simple_confirms_stronger_displaced_attack(
    simple: Mapping[str, Any], cue_time: float, displaced_time: float, tolerance_sec: float,
) -> bool:
    """Confirm a weak on-cue match when the same group's band attacks much later.

    This is reserved for a repeated group whose temporal signature prefers a
    displaced event. The on-cue band may contain a different stacked sound,
    so the later attack must rise clearly above both the cue's level and the
    group's usual level, while also clearing the measured comparison level.
    """
    typical = simple.get("typical_level_db")
    cue_level = float(simple.get("cue_level_db", -120.0))
    if (displaced_time <= cue_time + tolerance_sec or typical is None
            or cue_level >= float(typical)):
        return False
    for onset in simple.get("onsets", []):
        onset_time = float(onset["time_sec"])
        if abs(onset_time - displaced_time) > max(tolerance_sec, 0.030):
            continue
        test = simple["level_test"](onset_time, simple["cue_index"])
        if (test["rise_db"] >= _SIMPLE_ATTACK_DB
                and test["level_db"] >= float(typical) + _SIMPLE_STRONG_DISPLACED_PEER_GAIN_DB
                and test["level_db"] >= cue_level + _SIMPLE_STRONG_DISPLACED_CUE_GAIN_DB
                and test["level_db"] - test["reference_db"] >= _SIMPLE_STRONG_DISPLACED_REFERENCE_GAIN_DB):
            return True
    return False


def sync_check(
    source: str,
    cues: Optional[Sequence[Dict[str, Any]]] = None,
    tolerance_ms: float = 80.0,
    cache: Optional[MediaCache] = None,
    cut_detector: Any = None,
    search_window_ms: float = 500.0,
    bpm: Optional[float] = None,
    beats_per_bar: float = 4.0,
    coincidence_ms: float = 5.0,
) -> Dict[str, Any]:
    """Compare cuts and cues, learning cue-group signatures to separate overlaps.

    Existing arguments and result fields are retained. `search_window_ms` lets
    a cue find its own early/late sound beyond the in-tolerance match window.
    When `bpm` is known, candidate spectra are compared with uncued bars at the
    same beat phase so steady backing percussion does not mask a missing layer.
    """
    tolerance = _finite_number(tolerance_ms, "tolerance_ms")
    if tolerance < 0.0 or tolerance > 5000.0:
        raise ExtractionBoundsError("tolerance_ms must satisfy 0 <= value <= 5000 (default 80)", {"tolerance_ms": tolerance})
    search_window = _finite_number(search_window_ms, "search_window_ms")
    if search_window < tolerance or search_window > 5000.0:
        raise ExtractionBoundsError(
            "search_window_ms must satisfy tolerance_ms <= value <= 5000",
            {"search_window_ms": search_window, "tolerance_ms": tolerance},
        )
    coincidence = _finite_number(coincidence_ms, "coincidence_ms")
    if coincidence < 0.0 or coincidence > 100.0:
        raise ExtractionBoundsError("coincidence_ms must satisfy 0 <= value <= 100", {"coincidence_ms": coincidence})
    bar_sec: Optional[float] = None
    if bpm is not None:
        bpm_value = _finite_number(bpm, "bpm")
        beats_value = _finite_number(beats_per_bar, "beats_per_bar")
        if bpm_value < 20.0 or bpm_value > 300.0:
            raise ExtractionBoundsError("bpm must satisfy 20 <= value <= 300", {"bpm": bpm_value})
        if beats_value < 1.0 or beats_value > 12.0:
            raise ExtractionBoundsError("beats_per_bar must satisfy 1 <= value <= 12", {"beats_per_bar": beats_value})
        bar_sec = 60.0 * beats_value / bpm_value

    normalized: List[Dict[str, Any]] = []
    seen_cues: List[Tuple[float, str]] = []
    for index, cue in enumerate(cues or []):
        if not isinstance(cue, dict) or ("time_sec" not in cue and "t" not in cue):
            raise ExtractionBoundsError('Each cue requires time_sec (seconds), e.g. {"time_sec": 12.5, "label": "hit"}', {"cue_index": index})
        timestamp = _finite_number(cue.get("time_sec", cue.get("t")), "cue.time_sec")
        if timestamp < 0.0:
            raise ExtractionBoundsError("Cue timestamps must be non-negative", {"cue_index": index, "time_sec": timestamp})
        cue_label = " ".join(str(cue.get("label", "cue")).split())
        # The same label within 2 ms is one cue listed twice.
        if any(label == cue_label.casefold() and abs(time - timestamp) <= 0.002 for time, label in seen_cues):
            continue
        seen_cues.append((timestamp, cue_label.casefold()))
        band = _normalize_band_hz(cue.get("band_hz"))
        band_key = None if band is None else "{:.6f}:{:.6f}".format(band["low_hz"], band["high_hz"])
        normalized.append({
            "time_sec": round(timestamp, 6), "label": cue_label,
            "input_index": index, "band": band, "band_key": band_key,
            "group": _cue_group_name(cue_label),
        })

    broad_onsets = audio_onsets(source, min_strength=0.10, cache=cache)["onsets"]
    samples, _, cache = _decode_pcm(source, cache)
    all_band_levels = _BandLevels(samples)
    onsets_by_band: Dict[str, List[Dict[str, Any]]] = {}
    control_cache: Dict[float, Optional[np.ndarray]] = {}
    cue_signatures = _learn_cue_signatures(
        normalized, broad_onsets, samples,
        search_window_sec=search_window / 1000.0,
        bar_sec=bar_sec,
        coincidence_sec=coincidence / 1000.0,
        tolerance_sec=tolerance / 1000.0,
        control_cache=control_cache,
    )
    overlap_models = _overlap_templates(normalized, samples, coincidence / 1000.0)
    temporal_matches, systematic_lags, uncued_raw = _temporal_group_matches(
        normalized, samples, search_window, tolerance, bpm_value if bpm is not None else None,
    )
    simple_checks = _simple_dominant_band_check(
        samples, normalized, onsets_by_band, source, cache,
        tolerance / 1000.0, coincidence / 1000.0,
        None if bpm is None else 60.0 / bpm_value,
    )
    if cut_detector is None:
        from src.retrieval.engine import AgenticVideoEngine
        cut_engine = AgenticVideoEngine(source, cache=cache)
        cut_detector = cut_engine.detect_cuts
    cuts_raw = cut_detector()
    cuts = []
    for cut in cuts_raw:
        cut_time = float(cut.timestamp_sec if hasattr(cut, "timestamp_sec") else cut[0])
        cut_score = float(cut.score if hasattr(cut, "score") else cut[1])
        nearest = min(broad_onsets, key=lambda item: abs(item["time_sec"] - cut_time)) if broad_onsets else None
        offset = None if nearest is None else round((nearest["time_sec"] - cut_time) * 1000.0, 3)
        cuts.append({
            "time_sec": round(cut_time, 6),
            "score": round(cut_score, 6),
            "nearest_onset": None if nearest is None else {"time_sec": nearest["time_sec"], "strength": nearest["strength"]},
            "offset_ms": offset,
        })

    # Compare matching cue groups at a shared time. Similar templates cannot
    # establish which layer caused a single onset, so mark those cases unclear.
    tied_groups = set()
    for left_index, left in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            right = normalized[right_index]
            if abs(float(left["time_sec"]) - float(right["time_sec"])) > coincidence / 1000.0:
                continue
            if left["group"] == right["group"]:
                continue
            left_model = cue_signatures.get(left["group"])
            right_model = cue_signatures.get(right["group"])
            if (left_model and right_model
                    and left_model.get("method") == right_model.get("method")
                    and _cosine(left_model["profile"], right_model["profile"]) >= 0.94):
                tied_groups.add((left_index, right_index))

    selected_onsets: Dict[int, Optional[Dict[str, Any]]] = {}
    ambiguous_reasons: Dict[int, Optional[str]] = {}
    signature_scores: Dict[int, Optional[float]] = {}
    excess_ratios: Dict[int, Optional[float]] = {}
    candidates_by_cue: Dict[int, List[Dict[str, Any]]] = {}
    nearest_broad_by_cue: Dict[int, Optional[Dict[str, Any]]] = {}
    template_matches: Dict[int, Dict[str, Any]] = {}
    for cue_index, cue in enumerate(normalized):
        cue_time = float(cue["time_sec"])
        model = cue_signatures.get(cue["group"])
        coincident = _cue_coincident(normalized, cue_index, coincidence / 1000.0)
        pool = _cue_onset_candidates(cue, model, broad_onsets, onsets_by_band, source, cache)
        nearby = [
            onset for onset in pool
            if abs(float(onset["time_sec"]) - cue_time) <= search_window / 1000.0
        ]
        nearest_broad = min(
            (item for item in broad_onsets if abs(float(item["time_sec"]) - cue_time) <= search_window / 1000.0),
            key=lambda item: abs(float(item["time_sec"]) - cue_time), default=None,
        )
        nearest_broad_by_cue[cue_index] = nearest_broad
        candidates_by_cue[cue_index] = nearby
        # A learned mixture model can identify a sound even if it has never
        # appeared on a cue by itself. It also measures a cue at t=0, where
        # conventional onset detectors lack a preceding frame.
        if cue["band"] is None and cue["group"] in overlap_models and (coincident or cue_time == 0.0 or model is None):
            template_matches[cue_index] = _template_match(
                cue, normalized, overlap_models, samples, tolerance, search_window,
                coincidence / 1000.0,
            )
        supported = []
        borderline_scores = []
        cue_excess_ratios: Dict[float, float] = {}
        if model:
            for onset in nearby:
                if model.get("method") == "differential_bar_excess":
                    excess = np.maximum(
                        _local_band_power(samples, float(onset["time_sec"]))
                        - model["baseline_by_cue"].get(cue_index, model["baseline_power"]), 0.0,
                    )
                    observed = _unit_vector(excess)
                    band_indices = [i for i, band in enumerate(BANDS) if band[0] in model["bands"]]
                    expected_energy = float(np.sum(model["excess_power"][band_indices]))
                    excess_ratio = float(np.sum(excess[band_indices])) / max(expected_energy, 1e-15)
                    cue_excess_ratios[float(onset["time_sec"])] = excess_ratio
                else:
                    observed = _event_signature_vector(
                        onset, samples, normalized, bar_sec,
                        max(coincidence / 1000.0, tolerance / 1000.0),
                        control_cache,
                    )
                score = _cosine(model["profile"], observed)
                expected_band_mass = sum(model["profile"][next(i for i, band in enumerate(BANDS) if band[0] == name)] for name in model["bands"])
                actual_band_mass = sum(observed[next(i for i, band in enumerate(BANDS) if band[0] == name)] for name in model["bands"])
                band_support = actual_band_mass >= max(0.12, expected_band_mass * 0.30)
                borderline_scores.append(score)
                # A coincident composite can legitimately contain substantial
                # energy outside this group's signature, while still carrying
                # a clear contribution in its own band. Keep a stricter match
                # for isolated events and require explicit band mass for the
                # more permissive coincident case.
                score_floor = 0.40 if coincident else 0.50
                if model.get("method") == "differential_bar_excess":
                    accepted = score >= 0.55 and excess_ratio >= 0.35 and band_support
                else:
                    accepted = score >= score_floor and band_support
                if accepted:
                    supported.append((onset, score))
        elif cue["band"] is not None:
            # Explicit frequency annotations are already a sound-specific test.
            supported = [(onset, 1.0) for onset in nearby]
        elif not coincident:
            # With no learned signature, only use a salient event as evidence
            # for an out-of-tolerance cue. Do not reuse an onset at another
            # labeled cue's time: without two sound profiles, that overlap is
            # not evidence that both cue sounds are present.
            for onset in nearby:
                delta = abs(float(onset["time_sec"]) - cue_time)
                if delta > tolerance / 1000.0 and float(onset.get("strength", 0.0)) < 0.30:
                    continue
                claimed_by_other_cue = any(
                    other_index != cue_index
                    and other["group"] != cue["group"]
                    and abs(float(other["time_sec"]) - float(onset["time_sec"])) <= tolerance / 1000.0
                    for other_index, other in enumerate(normalized)
                )
                if not claimed_by_other_cue:
                    supported.append((onset, 1.0))

        if model is None and cue["band"] is None:
            in_tolerance = [pair for pair in supported
                            if abs(float(pair[0]["time_sec"]) - cue_time) <= tolerance / 1000.0]
            if in_tolerance:
                supported = in_tolerance
                supported.sort(key=lambda pair: abs(float(pair[0]["time_sec"]) - cue_time))
            else:
                supported.sort(key=lambda pair: (
                    -float(pair[0].get("strength", 0.0)),
                    abs(float(pair[0]["time_sec"]) - cue_time),
                ))
        else:
            supported.sort(key=lambda pair: (
                abs(float(pair[0]["time_sec"]) - cue_time) > tolerance / 1000.0,
                abs(float(pair[0]["time_sec"]) - cue_time), -pair[1],
            ))
        selected = supported[0][0] if supported else None
        score = supported[0][1] if supported else None
        reason = None
        if selected is None:
            if coincident and nearby and model is None and cue["band"] is None:
                reason = _NO_SIGNATURE_REASON
            elif model and model.get("method") != "differential_bar_excess" and borderline_scores and max(borderline_scores) >= 0.38:
                reason = "nearby spectrum only partly supports this cue signature"
            elif coincident and any(cue_index in pair for pair in tied_groups):
                reason = "coincident cue groups have indistinguishable signatures"
        selected_onsets[cue_index] = selected
        ambiguous_reasons[cue_index] = reason
        signature_scores[cue_index] = score if score is not None else (max(borderline_scores) if borderline_scores else None)
        excess_ratios[cue_index] = cue_excess_ratios.get(float(selected["time_sec"])) if selected else None

    # If coincident groups both appear to match the same attack but their
    # learned spectra are effectively alike, the evidence cannot assign layers.
    for left_index, right_index in tied_groups:
        left_onset = selected_onsets.get(left_index)
        right_onset = selected_onsets.get(right_index)
        if left_onset and right_onset and abs(float(left_onset["time_sec"]) - float(right_onset["time_sec"])) <= 0.025:
            ambiguous_reasons[left_index] = "coincident cue groups have indistinguishable signatures"
            ambiguous_reasons[right_index] = "coincident cue groups have indistinguishable signatures"

    def audible_in_any_band(at: float) -> bool:
        """Any attack at all near the cue, broadband or in any named band."""
        window = max(tolerance / 1000.0, 0.030)
        if any(abs(float(onset["time_sec"]) - at) <= window for onset in broad_onsets):
            return True
        for band_name, _, _ in BANDS:
            if band_name not in onsets_by_band:
                onsets_by_band[band_name] = audio_onsets(
                    source, min_strength=0.10, cache=cache, band_hz=band_name,
                )["onsets"]
            if any(abs(float(onset["time_sec"]) - at) <= window for onset in onsets_by_band[band_name]):
                return True
        return False

    cue_reports = []
    assigned_broad_indices = set()
    soft_attack_build_by_cue: Dict[int, Dict[str, Any]] = {}
    group_sizes: Dict[str, int] = {}
    for item in normalized:
        group_sizes[str(item["group"])] = group_sizes.get(str(item["group"]), 0) + 1
    for cue_index, cue in enumerate(normalized):
        cue_time = float(cue["time_sec"])
        selected = selected_onsets[cue_index]
        nearest_broad = nearest_broad_by_cue[cue_index]
        reason = ambiguous_reasons[cue_index]
        temporal_match = temporal_matches.get(cue_index)
        if (temporal_match is not None and group_sizes.get(str(cue["group"]), 0) < 5
                and not temporal_match.get("soft_displacement")):
            temporal_match = None
        template_match = temporal_match or template_matches.get(cue_index)
        if template_match is not None:
            matched = template_match["time_sec"]
            reason = template_match["ambiguous_reason"]
            selected = None if matched is None else {"time_sec": matched, "strength": template_match["score"]}
            signature_scores[cue_index] = template_match["score"]
        else:
            if signature_scores.get(cue_index) is None and selected is not None:
                signature_scores[cue_index] = float(selected.get("strength", 1.0))
        if selected is not None and cue_index not in temporal_matches:
            event_time = float(selected["time_sec"])
            displacement = abs(event_time - cue_time)
            # Older sparse-group matchers also must not turn a search-edge
            # percussion hit, or another cue's sound, into a timing fault.
            near_search_edge = displacement >= max(tolerance / 1000.0, search_window / 1000.0 - .055)
            beyond_beat = bpm is not None and displacement > .65 * 60.0 / bpm_value
            claimed_by_nearer_cue = any(
                other_index != cue_index and other["group"] == cue["group"]
                and abs(float(other["time_sec"]) - event_time) + .001 < displacement
                for other_index, other in enumerate(normalized)
            )
            if near_search_edge or beyond_beat or claimed_by_nearer_cue:
                selected = None
                reason = None
        score_val = signature_scores.get(cue_index)
        cred_floor = (template_match.get("credible_score_floor") if template_match else 0.20) or 0.20
        if selected is not None and abs(float(selected["time_sec"]) - cue_time) > tolerance / 1000.0:
            if (score_val is not None and (score_val < cred_floor or score_val < 0.20)
                    and not (template_match and template_match.get("soft_displacement"))):
                selected = None

        if reason:
            template_status = "ambiguous"
        elif selected is None:
            template_status = "missing"
        else:
            delta_ms = (float(selected["time_sec"]) - cue_time) * 1000.0
            template_status = "ok" if abs(delta_ms) <= tolerance else ("early" if delta_ms < 0.0 else "late")

        # Dual evidence: a cue fault needs the template matcher *and* the
        # simple band/level check to agree. The simple check finding the
        # sound on time always prevents a fault.
        simple = simple_checks[cue_index]
        simple_present = bool(simple["on_time"])
        shares_band = any(
            other_index != cue_index
            and abs(float(normalized[other_index]["time_sec"]) - cue_time) <= coincidence / 1000.0
            and simple_checks[other_index]["dominant_band"] == simple["dominant_band"]
            for other_index in range(len(normalized))
        )
        independent_present = simple_present and not shares_band
        simple_onset = simple["onset_time"]
        evidence = "template_and_simple_agree"
        matched_time: Optional[float] = None
        # Clearly absent: nothing beyond what the other sounds explain, and
        # well below the group's usual level in its own band.
        clearly_absent = (simple["typical_level_db"] is not None
                          and simple["cue_level_db"] - simple["reference_level_db"] < _SIMPLE_ABSENT_DB
                          and simple["cue_level_db"] < simple["typical_level_db"] - _SIMPLE_ABSENT_BELOW_TYPICAL_DB)
        explained_ok = (
            template_status == "ok" and template_match is not None
            and float(template_match.get("companion_gain") or 0.0) >= 2.0
            and not shares_band and not simple_present and clearly_absent
        )
        if explained_ok:
            # The template's on-cue match coincides with a companion (e.g. an
            # accented kick) far louder than usual, while this group's own
            # band shows nothing beyond what the other sounds explain.
            displaced = template_match.get("displaced_time_sec")
            evidence = "template_match_explained_by_louder_companion"
            if displaced is None:
                # The template cannot see past the louder companion; take a
                # clear attack of this group's band elsewhere in the cue slot.
                displaced = _simple_displaced_attack(
                    simple, cue_time, float(template_match.get("slot_before_ms") or 0.0) / 1000.0,
                    float(template_match.get("slot_after_ms") or 0.0) / 1000.0, tolerance / 1000.0,
                )
                if displaced is not None:
                    evidence = "template_explained_by_louder_companion_simple_check_displaced"
            if displaced is not None and _simple_supports_time(simple, float(displaced), tolerance / 1000.0):
                matched_time = float(displaced)
                status = "early" if matched_time < cue_time else "late"
                if template_match.get("displaced_time_sec") is not None:
                    signature_scores[cue_index] = template_match.get("displaced_score")
            else:
                status = "missing"
        elif template_status == "ok" and not simple_present and (displaced := _own_band_displaced(
                simple, cue_index, normalized, template_match, search_window / 1000.0,
                tolerance / 1000.0)) is not None:
            # The template's on-cue match is carried by the other sounds in a
            # stack: this group's own band is far below usual at the cue, and
            # a clear attack of that band sits elsewhere in the cue slot.
            matched_time = displaced
            status = "early" if matched_time < cue_time else "late"
            evidence = "own_band_far_below_usual_attack_displaced"
        elif template_status == "ok":
            status = "ok"
            matched_time = float(selected["time_sec"])
        elif template_status == "ambiguous":
            if independent_present:
                status, reason, evidence = "ok", None, "simple_check_on_time"
                matched_time = simple_onset
            elif (reason == _NO_SIGNATURE_REASON and cue_time >= _START_PREROLL_SEC
                  and (excess := _excess_over_companions_db(
                      all_band_levels, cue_time, str(cue["group"]), normalized,
                      60.0 / bpm_value if bpm is not None else None, len(samples) / SAMPLE_RATE,
                      coincidence / 1000.0)) is not None and excess < _SIMPLE_EXCESS_DB):
                # Its only instance is the one in question, so its "usual level"
                # says nothing. Check the following beats for a soft attack
                # before deciding that no band beyond its companions means
                # the sound is missing.
                comparison_times = _comparison_moments(
                    cue_time, str(cue["group"]), normalized,
                    60.0 / bpm_value if bpm is not None else None,
                    len(samples) / SAMPLE_RATE, coincidence / 1000.0,
                )
                build_band = cue["band"] or (
                    simple["band"] if group_sizes.get(str(cue["group"]), 0) >= 2 else None
                )
                next_same_group_cue = min(
                    (float(other["time_sec"]) for other in normalized
                     if str(other["group"]) == str(cue["group"])
                     and float(other["time_sec"]) > cue_time),
                    default=len(samples) / SAMPLE_RATE,
                )
                build_check = _measure_soft_attack_build(
                    samples, cue_time, build_band,
                    60.0 / bpm_value if bpm is not None else None,
                    comparison_times,
                    next_same_group_cue,
                )
                soft_attack_build_by_cue[cue_index] = build_check
                status, reason, evidence = _no_band_cue_outcome(build_check)
            else:
                status = "ambiguous"
        elif template_status == "missing":
            if independent_present:
                status, reason, evidence = "ok", None, "simple_check_on_time"
                matched_time = simple_onset
            elif simple_present:
                status = "ambiguous"
                reason = "template finds no match, but the cue's band has an on-time attack shared with a coincident cue"
                evidence = "disagree"
            else:
                status = "missing"
        else:
            displaced_time = float(selected["time_sec"])
            if independent_present:
                soft_temporal_displacement = (
                    template_match is not None
                    and template_match.get("method") == "temporal_group_template"
                    and template_match.get("soft_displacement")
                )
                displaced_is_stacked = any(
                    other_index != cue_index and other["group"] != cue["group"]
                    and abs(float(other["time_sec"]) - displaced_time) <= max(coincidence / 1000.0, 0.075)
                    for other_index, other in enumerate(normalized)
                )
                if (soft_temporal_displacement and not displaced_is_stacked
                        and _simple_confirms_stronger_displaced_attack(
                            simple, cue_time, displaced_time, tolerance / 1000.0)):
                    status = template_status
                    matched_time = displaced_time
                    evidence = "stronger_group_band_attack_at_displaced_peak"
                else:
                    status = "ambiguous"
                    reason = "simple band check finds the sound on time; template prefers a displaced peak"
                    evidence = "disagree"
            elif _simple_supports_time(simple, displaced_time, tolerance / 1000.0) or (
                    # A strong template peak needs only the band's energy
                    # there: a slow swell may not register as an onset.
                    template_match is not None and score_val is not None
                    and score_val >= float(template_match.get("median_score") or float("inf"))
                    and any(simple["level_test"](displaced_time + step, cue_index)["passed"]
                            for step in (0.0, 0.05, 0.10, 0.15))):
                status = template_status
                matched_time = displaced_time
            else:
                status = "ambiguous"
                reason = "template peak is displaced, but the simple band check sees no attack there"
                evidence = "disagree"

        group_size = group_sizes.get(str(cue["group"]), 0)
        peer_level_check_applies = (
            group_size == 2
            or (group_size >= 3 and template_match is not None
                and template_match.get("method") == "temporal_group_template"
                and not simple_present and cue_time >= _START_PREROLL_SEC
                and simple["cue_level_db"] > -60.0)
        )
        if (status == "ok" and template_status == "ok" and peer_level_check_applies
                and _small_group_match_is_unsupported(cue_index, normalized, simple_checks)):
            # A repeated temporal template can accept a coincident transient
            # even when this cue's own band is a clear peer-level outlier.
            status, matched_time = "missing", None
            evidence = "repeated_cue_far_below_peer_level"

        if status == "missing" and _present_at_file_start(cue_time, simple):
            status = "ambiguous"
            reason = ("cue at the start of the file: no pre-roll to measure its attack, "
                      "and its band is at the group's usual level")
            evidence = "start_of_file_unmeasurable"
        elif status == "missing" and evidence != "no_band_beyond_companions" and _band_at_usual_level(simple):
            status = "ambiguous"
            reason = "no match found, but the cue's band is at the group's usual level there"
            evidence = "band_at_usual_level"

        nearest_time = None if nearest_broad is None else float(nearest_broad["time_sec"])
        if matched_time is not None:
            delta_ms = (matched_time - cue_time) * 1000.0
            reported_offset = round(delta_ms, 3)
            broad_index = min(
                (index for index, onset in enumerate(broad_onsets)
                 if abs(float(onset["time_sec"]) - matched_time) <= 0.025),
                key=lambda index: abs(float(broad_onsets[index]["time_sec"]) - matched_time),
                default=None,
            )
            if broad_index is not None:
                assigned_broad_indices.add(broad_index)
        else:
            reported_offset = (
                None if status == "missing" or nearest_time is None
                else round((nearest_time - cue_time) * 1000.0, 3)
            )

        report = {
            "time_sec": cue["time_sec"],
            "label": cue["label"],
            "status": status,
            "matched_onset_time_sec": matched_time,
            "offset_ms": reported_offset,
            "nearest_onset_time_sec": nearest_time,
            "nearest_onset_offset_ms": None if nearest_time is None else round((nearest_time - cue_time) * 1000.0, 3),
            "cue_group": cue["group"],
            "dominant_band": simple["dominant_band"],
            "simple_check_on_time": simple["on_time"],
            "simple_check_onset_offset_ms": simple["offset_ms"],
            "simple_check_level_db": round(simple["cue_level_db"], 2),
            "simple_check_reference_db": round(simple["reference_level_db"], 2),
            "simple_check_reference": simple["reference_kind"],
            "group_typical_level_db": None if simple["typical_level_db"] is None else round(simple["typical_level_db"], 2),
            "template_status": template_status,
            "evidence": evidence,
            "audible_in_any_band": True if status == "ok" else audible_in_any_band(cue_time),
            "signature_bands": cue_signatures.get(cue["group"], {}).get("bands", []),
            "signature_sample_count": cue_signatures.get(cue["group"], {}).get("sample_count", 0),
            "signature_score": signature_scores[cue_index],
            "group_median_score": None if template_match is None else template_match["median_score"],
            "group_typical_attack_lag_ms": None if template_match is None else template_match.get("typical_attack_lag_ms"),
            "on_cue_score": None if template_match is None else template_match.get("on_cue_score"),
            "companion_gain": None if template_match is None else template_match.get("companion_gain"),
            "displaced_time_sec": None if template_match is None else template_match.get("displaced_time_sec"),
            "shifted_null_floor": None if template_match is None else template_match.get("shifted_null_floor"),
            "credible_score_floor": None if template_match is None else template_match.get("credible_score_floor"),
            "slot_before_ms": None if template_match is None else template_match.get("slot_before_ms"),
            "slot_after_ms": None if template_match is None else template_match.get("slot_after_ms"),
            "signature_excess_ratio": excess_ratios[cue_index],
            "search_window_ms": search_window,
            "match_method": "explicit_band" if cue["band"] is not None else (
                template_match.get("method", "overlap_template") if template_match is not None else
                (cue_signatures[cue["group"]]["method"] if cue["group"] in cue_signatures else "broad_onset")
            ),
            "ambiguous_reason": reason,
            "candidate_onsets": [
                {"time_sec": float(item["time_sec"]), "offset_ms": round((float(item["time_sec"]) - cue_time) * 1000.0, 3),
                 "strength": float(item.get("strength", 0.0))}
                for item in candidates_by_cue[cue_index]
            ],
        }
        if cue_index in soft_attack_build_by_cue:
            report["soft_attack_build"] = soft_attack_build_by_cue[cue_index]
        if cue["band"] is not None:
            report["band_hz"] = _band_spec(cue["band"])
        cue_reports.append(report)
    cue_reports.sort(key=lambda item: item["time_sec"])
    _attribute_shared_displacement(cue_reports)

    # An uncued instance also needs the simple check: an attack in the
    # group's own band at its usual level, beyond what the backing explains.
    # A peak at another cue whose sound shares this group's band (a kick
    # template firing on a downbeat kick) is that cue's sound. A group with
    # many such peaks is matching the backing, so none of them count. When
    # that band is one the group barely reaches (it was kept free for sounds
    # that always land with it), the candidate's own attack band is checked
    # instead (`_uncued_attack_band`).
    uncued_sounds = []
    attack_levels = _BandLevels(samples) if uncued_raw else None
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for item in uncued_raw:
        members = [index for index, cue in enumerate(normalized) if cue["group"] == item["group"]]
        anchor = min(members, key=lambda index: abs(float(normalized[index]["time_sec"]) - item["time_sec"]))
        simple = simple_checks[anchor]
        at = float(item["time_sec"])
        own_model = cue_signatures.get(item["group"])

        def explains(index: int, cue: Mapping[str, Any]) -> bool:
            """Another cue here whose sound may be what the template heard."""
            other_model = cue_signatures.get(cue["group"])
            return (simple_checks[index]["dominant_band"] == simple["dominant_band"]
                    or own_model is None or other_model is None
                    or _cosine(own_model["profile"], other_model["profile"]) >= 0.8)

        if any(cue["group"] != item["group"] and abs(float(cue["time_sec"]) - at) <= tolerance / 1000.0
               and explains(index, cue) for index, cue in enumerate(normalized)):
            continue
        # A cue's own early/late sound is reported by that cue.
        if any(report.get("status") in {"early", "late"} and report.get("matched_onset_time_sec") is not None
               and abs(float(report["matched_onset_time_sec"]) - at) <= tolerance / 1000.0
               for report in cue_reports):
            continue
        level = float(simple["level_test"](at, anchor)["level_db"])
        typical = simple["typical_level_db"]
        if (typical is not None and level >= float(typical) - _USUAL_LEVEL_MARGIN_DB
                and _simple_supports_time(simple, at, tolerance / 1000.0)):
            evidence = {"dominant_band": simple["dominant_band"], "simple_check_level_db": round(level, 2),
                        "group_typical_level_db": round(float(typical), 2), "confirmed_by": "group_band"}
        else:
            claimed = [simple_checks[index]["band"] for index, cue in enumerate(normalized)
                       if cue["group"] != item["group"] and abs(float(cue["time_sec"]) - at) <= tolerance / 1000.0]
            attack = _uncued_attack_band(attack_levels, at, [float(normalized[index]["time_sec"]) for index in members],
                                         claimed)
            if attack is None:
                continue
            evidence = {"dominant_band": attack["band"], "simple_check_level_db": attack["level_db"],
                        "group_typical_level_db": attack["typical_level_db"], "attack_rise_db": attack["rise_db"],
                        "group_cue_rise_db": attack["cue_rise_db"], "confirmed_by": "own_attack_band"}
        by_group.setdefault(item["group"], []).append(dict(item, **evidence, group_cue_count=len(members)))
    for group, items in by_group.items():
        if len(items) <= max(1, _UNCUED_MAX_SHARE * items[0]["group_cue_count"]):
            uncued_sounds.extend(items)
    # Groups that always sound together (a ping and a tick on every cut) can
    # both match one uncued sound: report it once, naming the others.
    merged: List[Dict[str, Any]] = []
    for item in sorted(uncued_sounds, key=lambda item: -float(item["score"]) / max(float(item["threshold"]), 1e-9)):
        same = next((kept for kept in merged if abs(float(kept["time_sec"]) - float(item["time_sec"])) <= .075), None)
        if same is None:
            merged.append(dict(item, also_matches=[]))
        else:
            same["also_matches"].append(item["group"])
    uncued_sounds = sorted(merged, key=lambda item: item["time_sec"])

    strong_threshold = 0.5
    unmatched = [dict(item) for index, item in enumerate(broad_onsets)
                 if index not in assigned_broad_indices and item["strength"] >= strong_threshold]
    return {
        "source_path": str(Path(source).expanduser().resolve()),
        "tolerance_ms": round(tolerance, 3),
        "cue_recovery_window_ms": search_window,
        "search_window_ms": search_window,
        "bpm": bpm,
        "beats_per_bar": beats_per_bar if bpm is not None else None,
        "cue_signatures": {
            name: {"bands": model["bands"], "sample_count": model["sample_count"],
                  "method": model["method"], "control_count": model.get("control_count", 0),
                  "profile": {BANDS[index][0]: round(float(value), 4) for index, value in enumerate(model["profile"])},
                  "example_times_sec": model["example_times_sec"]}
            for name, model in cue_signatures.items()
        },
        "systematic_lags": systematic_lags,
        "cuts": cuts,
        "cues": cue_reports,
        "unmatched_strong_onsets": unmatched,
        "uncued_sounds": uncued_sounds,
        "strong_onset_threshold": strong_threshold,
        "total_cuts": len(cuts),
        "total_cues": len(cue_reports),
    }
