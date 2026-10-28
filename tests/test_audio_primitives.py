"""Synthetic-audio coverage for non-speech measurements and sync checking."""

import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.core.cache import MediaCache
from src.core.config import Config
from src.executor.mcp_server import handle_call, tool_call_response
from src.media.audio import _outlier_sections, audio_onsets, loudness_timeline, spectrogram_window, sync_check
from src.transcript import asr


RATE = 48000


def _write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    clipped = np.clip(samples, -0.999, 0.999)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def _clicks(duration: float, times, rate: int = RATE) -> np.ndarray:
    signal = np.zeros(int(duration * rate), dtype=np.float32)
    pulse_len = int(0.018 * rate)
    local_time = np.arange(pulse_len, dtype=np.float32) / rate
    pulse = (0.65 * np.sin(2 * np.pi * 1700 * local_time) + 0.25 * np.sin(2 * np.pi * 7300 * local_time))
    pulse *= np.exp(-local_time * 110.0)
    for timestamp in times:
        first = int(round(timestamp * rate))
        signal[first:first + pulse_len] += pulse[:max(0, min(pulse_len, len(signal) - first))]
    return np.clip(signal, -0.95, 0.95)


def test_audio_onsets_locate_synthetic_attacks_within_ten_ms(tmp_path):
    expected = [0.73, 0.777, 0.824, 1.81, 2.47]
    audio_path = _write_wav(tmp_path / "attacks.wav", _clicks(3.0, expected))
    result = audio_onsets(str(audio_path), min_strength=0.08, cache=MediaCache(tmp_path / "cache"))

    assert result["total"] >= len(expected)
    actual = [min(result["onsets"], key=lambda onset: abs(onset["time_sec"] - target))["time_sec"] for target in expected]
    assert actual == pytest.approx(expected, abs=0.01)
    assert all(set(item["band_db"]) == {"sub", "low_mid", "mid", "high", "air"} for item in result["onsets"])
    assert result["time_resolution_ms"] < 10.0


def test_loudness_timeline_finds_silence_and_high_band_outlier(tmp_path):
    duration = 3.0
    count = int(duration * RATE)
    times = np.arange(count, dtype=np.float64) / RATE
    signal = 0.025 * np.sin(2 * np.pi * 440 * times)
    signal += 0.002 * np.sin(2 * np.pi * 7200 * times)
    boosted = (times >= 1.8) & (times < 2.4)
    signal[boosted] += 0.35 * np.sin(2 * np.pi * 7200 * times[boosted])
    silent = (times >= 0.9) & (times < 1.38)
    signal[silent] = 0.0
    audio_path = _write_wav(tmp_path / "mix.wav", signal.astype(np.float32))
    result = loudness_timeline(str(audio_path), window_sec=0.2, cache=MediaCache(tmp_path / "cache"))

    assert any(gap["start_sec"] == pytest.approx(0.9, abs=0.01) and gap["end_sec"] == pytest.approx(1.38, abs=0.01) for gap in result["silence_gaps"])
    high_outliers = [item for item in result["outliers"] if item["band"] in ("high", "air")]
    assert high_outliers
    assert any(item["start_sec"] < 2.4 and item["end_sec"] > 1.8 and item["delta_db"] >= 6.0 for item in high_outliers)
    assert any(window["silent"] for window in result["windows"])
    assert all("spectral_centroid_hz" in window and "peak_dbfs" in window for window in result["windows"])


def _outlier_windows(high_values):
    windows = []
    for index, high_db in enumerate(high_values):
        profile = {"sub": -40.0, "low_mid": -40.0, "mid": -40.0, "high": high_db, "air": -40.0}
        windows.append({
            "start_sec": index * 0.25,
            "end_sec": (index + 1) * 0.25,
            "band_db": profile,
            "silent": False,
        })
    return windows


def test_outlier_shape_distinguishes_step_ramp_and_transient():
    flat = [-40.0] * 12
    step = flat.copy()
    step[4:8] = [-28.0] * 4
    ramp = flat.copy()
    ramp[3:8] = [-38.0, -36.0, -34.0, -32.0, -30.0]
    transient = flat.copy()
    transient[6] = -20.0

    step_outliers = [item for item in _outlier_sections(_outlier_windows(step)) if item["band"] == "high"]
    ramp_outliers = [item for item in _outlier_sections(_outlier_windows(ramp)) if item["band"] == "high"]
    transient_outliers = [item for item in _outlier_sections(_outlier_windows(transient)) if item["band"] == "high"]

    assert len(step_outliers) == len(ramp_outliers) == len(transient_outliers) == 1
    assert step_outliers[0]["shape"] == "step" and step_outliers[0]["fault_level"] is True
    assert ramp_outliers[0]["shape"] == "ramp" and ramp_outliers[0]["fault_level"] is False
    assert transient_outliers[0]["shape"] == "transient" and transient_outliers[0]["fault_level"] is False
    assert {"direction", "energy_db", "baseline_db", "delta_db", "entry_jump_db", "exit_jump_db"} <= set(step_outliers[0])


def test_loudness_ignore_ranges_mark_intentional_band_outliers(tmp_path):
    duration = 3.0
    times = np.arange(int(duration * RATE), dtype=np.float64) / RATE
    signal = 0.025 * np.sin(2 * np.pi * 440 * times)
    signal += 0.002 * np.sin(2 * np.pi * 7200 * times)
    boosted = (times >= 1.0) & (times < 2.0)
    signal[boosted] += 0.35 * np.sin(2 * np.pi * 7200 * times[boosted])
    path = _write_wav(tmp_path / "intentional.wav", signal.astype(np.float32))
    result = loudness_timeline(
        str(path), window_sec=0.2,
        ignore_ranges=[{"start_sec": 1.0, "end_sec": 2.0, "reason": "intentional synth layer"}],
        cache=MediaCache(tmp_path / "cache"),
    )
    ignored = [item for item in result["outliers"] if item["band"] == "high" and item["start_sec"] < 2.0 and item["end_sec"] > 1.0]
    assert ignored
    assert all(item["ignored"] and item["ignore_reason"] == "intentional synth layer" for item in ignored)
    assert all(not item["fault_level"] and item["classification"] == "informational" for item in ignored)


def _ffmpeg_peak_dbfs(path: Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "astats=measure_perchannel=none", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    lines = [line for line in proc.stderr.splitlines() if "Peak level dB" in line]
    return float(lines[-1].rsplit(":", 1)[1])


def test_loudness_peak_matches_ffmpeg_on_centred_stereo(tmp_path):
    # Identical L and R: a mono downmix reads 3 dB hot (+1.6 dBFS here),
    # while every channel meter, ffmpeg included, reads about -1.4 dBFS.
    times = np.arange(int(2.0 * RATE), dtype=np.float64) / RATE
    signal = 0.1 * np.sin(2 * np.pi * 440 * times)
    burst = (times >= 1.0) & (times < 1.2)
    signal[burst] = 0.85 * np.sin(2 * np.pi * 440 * times[burst])
    pcm = (np.repeat(signal[:, None], 2, axis=1) * 32767.0).astype("<i2")
    path = tmp_path / "centred.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())

    result = loudness_timeline(str(path), window_sec=0.2, cache=MediaCache(tmp_path / "cache"))

    ffmpeg_peak = _ffmpeg_peak_dbfs(path)
    assert ffmpeg_peak == pytest.approx(-1.41, abs=0.05)
    assert result["peak_dbfs"] == pytest.approx(ffmpeg_peak, abs=0.1)
    assert 1.0 <= result["peak_sec"] < 1.2
    assert max(window["peak_dbfs"] for window in result["windows"]) == pytest.approx(ffmpeg_peak, abs=0.1)


def test_sync_check_reports_missing_and_shifted_cues(tmp_path):
    expected_sound_times = [0.72, 1.72, 1.9]
    audio_path = _write_wav(tmp_path / "sync.wav", _clicks(2.4, expected_sound_times))
    cues = [
        {"time_sec": 0.2, "label": "deleted event"},
        {"t": 1.6, "label": "shifted event"},
        {"time_sec": 1.9, "label": "aligned event"},
    ]
    result = sync_check(
        str(audio_path), cues=cues, tolerance_ms=40,
        cache=MediaCache(tmp_path / "cache"), cut_detector=lambda: [],
    )
    by_label = {item["label"]: item for item in result["cues"]}

    assert by_label["deleted event"]["status"] == "missing"
    assert by_label["shifted event"]["status"] == "late"
    assert by_label["shifted event"]["offset_ms"] == pytest.approx(120.0, abs=10.0)
    assert by_label["aligned event"]["status"] == "ok"


def test_spectrogram_window_writes_png_and_draws_marks(tmp_path):
    audio_path = _write_wav(tmp_path / "spectrum.wav", _clicks(3.0, [1.2, 2.0]))
    result = spectrogram_window(
        str(audio_path), 0.5, 2.5,
        marks=[{"time_sec": 1.2, "label": "cut"}, {"time_sec": 2.0, "label": "hit"}],
        cache=MediaCache(tmp_path / "cache"),
    )

    assert Path(result["file_path"]).is_file()
    with Image.open(result["file_path"]) as image:
        assert image.format == "PNG"
        assert image.size == (1200, 600)
        pixels = np.asarray(image.convert("RGB"))
    assert len(result["marks"]) == 2
    assert result["timecode_badges"]
    assert np.count_nonzero((pixels[:, :, 0] > 220) & (pixels[:, :, 1] < 150) & (pixels[:, :, 2] < 170)) > 100
    ticks = result["frequency_ticks"]
    assert [tick["label"] for tick in ticks] == ["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
    assert [tick["frequency_hz"] for tick in ticks] == [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    assert all(left["y_px"] > right["y_px"] for left, right in zip(ticks, ticks[1:]))
    assert all(tuple(pixels[tick["y_px"], 176]) == (80, 83, 98) for tick in ticks)


def test_band_limited_onsets_and_compare_find_quiet_tone_blips(tmp_path):
    duration = 2.0
    baseline = np.zeros(int(duration * RATE), dtype=np.float32)
    rng = np.random.default_rng(40)
    click_times = [0.5, 1.0, 1.5]
    tone_times = [0.507, 1.007, 1.507]
    tone_hz = [1760, 2090, 2640]
    for timestamp in click_times:
        first = int(timestamp * RATE)
        baseline[first:first + 96] += rng.normal(0.0, 0.8, 96).astype(np.float32)
    with_tones = baseline.copy()
    for timestamp, frequency in zip(tone_times, tone_hz):
        first = int(timestamp * RATE)
        count = int(0.025 * RATE)
        local_time = np.arange(count, dtype=np.float32) / RATE
        with_tones[first:first + count] += 0.12 * np.sin(2 * np.pi * frequency * local_time)
    baseline_path = _write_wav(tmp_path / "clicks.wav", baseline)
    source_path = _write_wav(tmp_path / "clicks-with-tones.wav", with_tones)
    cache = MediaCache(tmp_path / "cache")

    broadband = audio_onsets(str(source_path), min_strength=0.08, cache=cache)
    band_limited = audio_onsets(str(source_path), min_strength=0.08, band_hz=[1500, 3000], cache=cache)
    compared = audio_onsets(
        str(source_path), min_strength=0.08, band_hz=[1500, 3000],
        compare_to=str(baseline_path), cache=cache,
    )
    broadband_hits = [item["time_sec"] for item in broadband["onsets"]]
    pitched_hits = [item["time_sec"] for item in band_limited["onsets"]]
    added_hits = [item["time_sec"] for item in compared["onsets"]]

    assert len(broadband_hits) == len(click_times)
    assert all(any(abs(hit - target) <= 0.012 for hit in pitched_hits) for target in tone_times)
    assert len(added_hits) == len(tone_times)
    assert all(any(abs(hit - target) <= 0.012 for hit in added_hits) for target in tone_times)
    assert all(item["compare_delta_db"] >= 2.5 for item in compared["onsets"])

    sync = sync_check(
        str(source_path), cues=[{"time_sec": tone_times[0], "label": "pitched cue", "band_hz": [1500, 3000]}],
        tolerance_ms=20, cache=cache, cut_detector=lambda: [],
    )
    assert sync["cues"][0]["status"] == "ok"


def test_synthetic_testsrc_video_runs_end_to_end_through_sync_check(tmp_path):
    audio_path = _write_wav(tmp_path / "generated.wav", _clicks(2.0, [0.5, 1.5]))
    video_path = tmp_path / "testsrc-with-audio.mp4"
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24:duration=2",
        "-i", str(audio_path),
        "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='gte(t,1)'",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(video_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = sync_check(
        str(video_path),
        cues=[{"time_sec": 0.5, "label": "first hit"}, {"time_sec": 1.5, "label": "second hit"}],
        tolerance_ms=50,
        cache=MediaCache(tmp_path / "cache"),
    )

    assert result["total_cues"] == 2
    assert [cue["status"] for cue in result["cues"]] == ["ok", "ok"]
    assert result["total_cuts"] >= 1


def _audio_only_m4a(tmp_path: Path) -> str:
    path = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:a", "aac", str(path)],
        check=True,
    )
    return str(path)


def test_audio_only_m4a_works_with_audio_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = _audio_only_m4a(tmp_path)

    probe = handle_call("probe_video", {"source": source})
    assert probe["ok"] is True
    assert probe["data"]["has_video"] is False
    assert probe["data"]["has_audio"] is True
    assert probe["data"]["duration_sec"] == pytest.approx(3.0, abs=0.1)

    assert handle_call("loudness_timeline", {"source": source})["ok"] is True
    assert handle_call("search_transcript", {"source": source, "query": "tone", "mode": "lexical"})["ok"] is True
    transcript = handle_call("transcribe", {"source": source})
    assert transcript["ok"] is True

    # A background ASR attempt may still be running; let it finish so it cannot outlive this test.
    deadline = time.monotonic() + 120
    while asr._BACKGROUND_TRANSCRIPTION_SLOT._active is not None and time.monotonic() < deadline:
        time.sleep(0.2)


def test_visual_tools_refuse_audio_only_m4a_with_audio_only_message(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = _audio_only_m4a(tmp_path)

    for name, args in [("contact_sheet", {"source": source}),
                       ("sync_check", {"source": source, "cues": [{"time_sec": 1.0}]})]:
        response = json.loads(tool_call_response(1, name, args))
        assert response["result"]["isError"] is True, name
        assert "audio only" in response["result"]["content"][0]["text"], name
