"""AAC cue matching against layered, quiet, reverberant synthetic mixes."""

import math
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.cache import MediaCache
from src.media.audio import sync_check


RATE = 48000


def _add(track, at, sound, gain=1.0):
    first = int(round(at * RATE))
    if first < 0 or first >= len(track):
        return
    last = min(len(track), first + len(sound))
    track[first:last] += sound[:last - first] * gain


def _lowpass(segment, cutoff_hz):
    """Brick-wall-free FFT low-pass: a smooth roll-off like a muffle filter."""
    spectrum = np.fft.rfft(segment)
    frequencies = np.fft.rfftfreq(len(segment), 1.0 / RATE)
    spectrum *= 1.0 / np.sqrt(1.0 + (frequencies / cutoff_hz) ** 8)
    return np.fft.irfft(spectrum, n=len(segment))


def _soundtrack(tmp_path: Path, bpm: int, design: int, *, hardened=False, no_fault=False,
                accents=False, reveals=(), muffle=None, pop_faults=None, whoosh_faults=None,
                ticks=(), tick_faults=None):
    """Synthetic layered mix. `*_faults` map a cue index to a time shift in
    seconds, or None for a missing sound; they replace the default faults."""
    rng = np.random.default_rng(1000 + bpm * 10 + design)
    beat = 60.0 / bpm
    duration = beat * 36 + 0.5
    samples = np.zeros(int(round(duration * RATE)), dtype=np.float64)
    t = np.arange(len(samples)) / RATE
    # A sustained, moving pad and a pitched bassline underneath every cue.
    samples += 0.027 * (np.sin(2 * np.pi * 220 * t + 0.15 * np.sin(2 * np.pi * .23 * t))
                         + .6 * np.sin(2 * np.pi * 330 * t))

    kt = np.arange(int(.18 * RATE)) / RATE
    kick = .65 * np.sin(2 * np.pi * (58 * kt + 75 * (.035 / (2 * np.pi)) * (1 - np.exp(-kt / .035)))) * np.exp(-kt * 25)
    ht = np.arange(int(.055 * RATE)) / RATE
    noise = rng.normal(size=len(ht))
    hat = .045 * (noise - np.convolve(noise, np.ones(19) / 19, mode="same")) * np.exp(-ht * 85)
    bt = np.arange(int(.32 * RATE)) / RATE
    for index in range(36):
        at = index * beat
        # Accented bar downbeats are 12 dB louder than the other kicks.
        _add(samples, at, kick, (0.25 * 10 ** (12 / 20) if index % 4 == 0 else 0.25) if accents else 1.0)
        _add(samples, at, hat)
        _add(samples, at + beat / 2, hat, .65)
        bass = .065 * np.sin(2 * np.pi * [82, 110, 98, 73][index % 4] * bt) * np.exp(-bt * 8)
        _add(samples, at, bass)
        if hardened:
            # Alternating 16th hats and shakers occupy much of the same
            # high-frequency region as the quieter cue clicks.
            for subdivision in range(4):
                local = rng.normal(size=len(ht))
                shaker = .036 * (local - np.convolve(local, np.ones(11) / 11, mode="same")) * np.exp(-ht * 90)
                _add(samples, at + subdivision * beat / 4, shaker)
            if index % 5 == 1:
                rim = .055 * np.sin(2 * np.pi * (2850 if design == 0 else 3900) * ht) * np.exp(-ht * 75)
                _add(samples, at, rim)
            stab = .045 * np.sin(2 * np.pi * [165, 196, 147, 220][index % 4] * bt) * np.exp(-bt * 15)
            _add(samples, at + beat / 4, stab)

    ct = np.arange(int(.060 * RATE)) / RATE
    if design == 0:
        click = np.sin(2 * np.pi * 3900 * ct) * np.exp(-ct * 75)
    else:
        click = (np.sin(2 * np.pi * 2700 * ct) + .35 * np.sin(2 * np.pi * 6100 * ct)) * np.exp(-ct * 105)
    click *= .072 / max(np.max(np.abs(click)), 1e-9)  # 19 dB below kick peak.
    wt = np.arange(int(.32 * RATE)) / RATE
    white = rng.normal(size=len(wt))
    smooth = np.convolve(white, np.ones(25 if design == 0 else 9) / (25 if design == 0 else 9), mode="same")
    carrier = np.sin(2 * np.pi * (950 if design == 0 else 1550) * wt + 15 * wt ** 2)
    envelope = np.sin(np.pi * np.minimum(wt / (.19 if design == 0 else .23), 1) / 2) ** 2
    whoosh = .13 * envelope * (0.65 * smooth + 0.35 * carrier) * np.exp(-np.maximum(wt - .20, 0) * 18)

    click_indices = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    whoosh_indices = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
    pop_faults = {16: -.120, 24: None} if pop_faults is None else pop_faults
    whoosh_faults = {18: .250, 27: None} if whoosh_faults is None else whoosh_faults
    tick_faults = {} if tick_faults is None else tick_faults
    if no_fault:
        pop_faults, whoosh_faults, tick_faults = {}, {}, {}

    def kind(shift):
        return "missing" if shift is None else ("early" if shift < 0 else "late")

    faults = {f"text pop {i:02d}": kind(v) for i, v in pop_faults.items()}
    faults.update({f"cut whoosh {i:02d}": kind(v) for i, v in whoosh_faults.items()})
    faults.update({f"ui tick {i:02d}": kind(v) for i, v in tick_faults.items()})
    for index in click_indices:
        if pop_faults.get(index, 0.0) is not None:
            _add(samples, index * beat + pop_faults.get(index, 0.0), click)
    for index in ticks:
        # Off-beat UI ticks land on the 8th-note hats, between kicks.
        if tick_faults.get(index, 0.0) is not None:
            _add(samples, (index + .5) * beat + tick_faults.get(index, 0.0), click * .8)
    for index in whoosh_indices:
        if whoosh_faults.get(index, 0.0) is not None:
            sound = whoosh
            if hardened:
                # The same family of whooshes has modest duration and level
                # variation, as it would when edited to different cuts.
                tail = .17 + .015 * (index % 6)
                sound = whoosh * (0.86 + .05 * (index % 5)) * np.exp(-np.maximum(wt - tail, 0) * (2 + index % 3))
            _add(samples, index * beat + whoosh_faults.get(index, 0.0), sound)
    if hardened:
        # Irregular visual reveals and a rising noise bed make non-beat cues
        # and broad-spectrum attacks part of the comparison set.
        it = np.arange(int(.11 * RATE)) / RATE
        impact = .11 * (np.sin(2 * np.pi * 1800 * it) + .3 * np.sin(2 * np.pi * 440 * it)) * np.exp(-it * 35)
        for index, fraction in ((5, .23), (11, .13), (17, .31), (23, .19), (29, .27)):
            _add(samples, (index + fraction) * beat, impact)
        for index in (10, 21):
            rise_t = np.arange(int(.45 * RATE)) / RATE
            noise = rng.normal(size=len(rise_t))
            smooth_rise = np.convolve(noise, np.ones(7) / 7, mode="same")
            _add(samples, index * beat, .025 * smooth_rise * (rise_t / max(rise_t[-1], 1e-9)) ** 2)

    reveal_cues = []
    for index in reveals:
        # A loud reveal: a half-second noise riser swells into an impact that
        # lands with the kick (and any other cue on that beat).
        at = index * beat
        rise_t = np.arange(int(.50 * RATE)) / RATE
        noise = rng.normal(size=len(rise_t))
        bright = noise - np.convolve(noise, np.ones(5) / 5, mode="same")
        sweep = np.sin(2 * np.pi * (700 * rise_t + 2600 * rise_t ** 2))
        riser = (.10 * bright + .08 * sweep) * (rise_t / rise_t[-1]) ** 2.5
        _add(samples, at - len(rise_t) / RATE, riser)
        it = np.arange(int(.35 * RATE)) / RATE
        boom = (.45 * np.sin(2 * np.pi * (48 * it + 60 * (1 - np.exp(-it / .05)) * .05 / (2 * np.pi)))
                + .12 * rng.normal(size=len(it)) * np.exp(-it * 30)) * np.exp(-it * 9)
        _add(samples, at, boom)
        reveal_cues.append({"time_sec": at, "label": f"reveal riser impact {index:02d}"})
    if muffle is not None:
        # The author's deliberate "underwater" section: the whole mix is
        # low-passed, with short fades so the filter does not click.
        first, last = int(round(muffle[0] * RATE)), int(round(muffle[1] * RATE))
        section = samples[first:last].copy()
        filtered = _lowpass(section, 380.0)
        fade = np.clip(np.minimum(np.arange(len(section)), np.arange(len(section))[::-1]) / (.02 * RATE), 0, 1)
        samples[first:last] = fade * filtered + (1 - fade) * section

    if accents or reveals:
        samples *= 0.8
    # Short diffuse reflections, then AAC encode/decode via the video tool.
    dry = samples.copy()
    for delay, gain in ((.075, .17), (.135, .09)):
        shift = int(round(delay * RATE))
        samples[shift:] += gain * dry[:-shift]
    samples = np.clip(samples, -.95, .95)
    wav = tmp_path / "mix.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    video = tmp_path / "mix.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc=size=96x54:rate=12:duration={duration}", "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-q:v", "8",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(video),
    ], check=True, capture_output=True, text=True)
    cues = [{"time_sec": index * beat, "label": f"downbeat kick {index:02d}"} for index in range(36)]
    cues += [{"time_sec": index * beat, "label": f"text pop {index:02d}"} for index in click_indices]
    cues += [{"time_sec": index * beat, "label": f"cut whoosh {index:02d}"} for index in whoosh_indices]
    cues += [{"time_sec": (index + .5) * beat, "label": f"ui tick {index:02d}"} for index in ticks]
    if hardened:
        cues += [{"time_sec": (index + fraction) * beat, "label": f"reveal impact {index:02d}"}
                 for index, fraction in ((5, .23), (11, .13), (17, .31), (23, .19), (29, .27))]
        if no_fault:
            cues.append({"time_sec": 12 * beat, "label": "text pop 12"})
    cues += reveal_cues
    return video, cues, faults


@pytest.mark.parametrize("bpm", [104, 126])
@pytest.mark.parametrize("design", [0, 1])
def test_layered_aac_sync_precision_and_recall(tmp_path, bpm, design):
    video, cues, faults = _soundtrack(tmp_path, bpm, design)
    result = sync_check(str(video), cues=cues, bpm=bpm, tolerance_ms=80,
                        search_window_ms=500, cache=MediaCache(tmp_path / "cache"),
                        cut_detector=lambda: [])
    actual = {cue["label"]: cue["status"] for cue in result["cues"]}
    detected = {label: status for label, status in actual.items() if status in {"early", "late", "missing"}}
    true_positives = sum(detected.get(label) == kind for label, kind in faults.items())
    precision = true_positives / len(detected) if detected else 0.0
    recall = true_positives / len(faults)
    ambiguous = sum(status == "ambiguous" for status in actual.values()) / len(actual)
    assert precision == 1.0 and recall == 1.0 and ambiguous <= .05, (bpm, design, detected, faults, ambiguous)
    assert actual["downbeat kick 00"] == "ok"
    for label in ("text pop 16", "cut whoosh 18"):
        cue = next(item for item in result["cues"] if item["label"] == label)
        assert (cue["offset_ms"] < 0) == (faults[label] == "early")


@pytest.mark.parametrize("bpm", [104, 126])
@pytest.mark.parametrize("design", [0, 1])
@pytest.mark.parametrize("no_fault", [False, True])
def test_hardened_aac_cues(tmp_path, bpm, design, no_fault):
    video, cues, faults = _soundtrack(tmp_path, bpm, design, hardened=True, no_fault=no_fault)
    assert len(cues) >= 30 and len({cue["label"].split()[0] for cue in cues}) >= 4
    result = sync_check(str(video), cues=cues, bpm=bpm, tolerance_ms=80,
                        search_window_ms=500, cache=MediaCache(tmp_path / "cache"),
                        cut_detector=lambda: [])
    labels = [(cue["label"], cue["time_sec"]) for cue in result["cues"]]
    assert len(labels) == len(set(labels))
    detected = {cue["label"]: cue["status"] for cue in result["cues"]
                if cue["status"] in {"early", "late", "missing"}}
    true_positives = sum(detected.get(label) == kind for label, kind in faults.items())
    precision = true_positives / len(detected) if detected else 1.0
    recall = true_positives / len(faults) if faults else 1.0
    assert precision == 1.0 and recall == 1.0, (bpm, design, no_fault, detected, faults)


def _faults_found(result, faults):
    detected = {cue["label"]: cue["status"] for cue in result["cues"]
                if cue["status"] in {"early", "late", "missing"}}
    false_alarms = {label: status for label, status in detected.items() if faults.get(label) != status}
    found = {label for label, kind in faults.items() if detected.get(label) == kind}
    return detected, false_alarms, found


def _check(video, cues, bpm, tmp_path):
    return sync_check(str(video), cues=cues, bpm=bpm, tolerance_ms=80, search_window_ms=500,
                      cache=MediaCache(tmp_path / "cache"), cut_detector=lambda: [])


@pytest.mark.parametrize("bpm", [104, 126])
@pytest.mark.parametrize("no_fault", [True, False])
def test_accented_downbeats_are_not_missing(tmp_path, bpm, no_fault):
    """Bar downbeat kicks 12 dB louder than the rest are still the same, present kick."""
    video, cues, faults = _soundtrack(tmp_path, bpm, 0, hardened=True, no_fault=no_fault, accents=True)
    result = _check(video, cues, bpm, tmp_path)
    detected, false_alarms, found = _faults_found(result, faults)
    kicks = [cue for cue in result["cues"] if cue["cue_group"] == "downbeat kick"]
    assert all(cue["status"] == "ok" for cue in kicks), [c for c in kicks if c["status"] != "ok"]
    assert false_alarms == {}, (bpm, no_fault, detected)
    # The planted pop faults sit on accented beats: the louder kick must be
    # explained away, not mistaken for the pop.
    assert found == set(faults), (bpm, detected, faults)


@pytest.mark.parametrize("bpm", [104, 126])
@pytest.mark.parametrize("no_fault", [True, False])
def test_riser_into_impact_does_not_make_coincident_cues_early(tmp_path, bpm, no_fault):
    video, cues, faults = _soundtrack(tmp_path, bpm, 1, hardened=True, no_fault=no_fault, reveals=(12, 30))
    result = _check(video, cues, bpm, tmp_path)
    detected, false_alarms, found = _faults_found(result, faults)
    assert false_alarms == {}, (bpm, no_fault, detected)
    assert found == set(faults), (bpm, detected, faults)
    beat = 60.0 / bpm
    at_reveals = [cue for cue in result["cues"]
                  if any(abs(cue["time_sec"] - index * beat) < .005 for index in (12, 30))]
    assert len(at_reveals) >= 4 and all(cue["status"] == "ok" for cue in at_reveals), at_reveals


@pytest.mark.parametrize("bpm,phrase", [
    (104, "From {start:.2f} s to {end:.2f} s the whole mix is deliberately muffled, like being underwater."),
    (126, "The underwater bit between {start:.2f} and {end:.2f} seconds is low-passed on purpose."),
])
def test_intent_muffled_section_is_expected_not_a_fault(tmp_path, bpm, phrase):
    import json

    from src.loop import check_video

    beat = 60.0 / bpm
    section = (28.5 * beat, 33.5 * beat)
    video, cues, _ = _soundtrack(tmp_path, bpm, 0, hardened=True, no_fault=True, muffle=section)
    sentence = phrase.format(start=section[0], end=section[1])
    intent = tmp_path / "intent.md"
    intent.write_text("Punchy edit with hits on the beat.\n" + sentence + "\nEnd card stays clean.\n", encoding="utf-8")
    cues_path = tmp_path / "cues.json"
    cues_path.write_text(json.dumps({"tolerance_ms": 80, "cues": cues}), encoding="utf-8")
    report = check_video(str(video), cues_path=str(cues_path), bpm=bpm, intent_path=str(intent),
                         out_dir=str(tmp_path / "out"), judge="none")
    confirmed = [item for item in report["candidates"] if item["confirmed"]]
    assert confirmed == [], [(item["kind"], item["start_sec"], item.get("cue_label")) for item in confirmed]
    inside = [item for item in report["candidates"]
              if item["kind"].startswith("cue_") and section[0] <= item["start_sec"] <= section[1]]
    # The muffled cues are measured as altered, reported, and tied to the note.
    assert inside and all(item["rule_verdict"] == "informational" for item in inside)
    assert all(item["intent_note"] == sentence for item in inside)
    outliers = report["measurements"]["loudness_timeline"]["outliers"]
    assert not any(item["fault_level"] for item in outliers
                   if item["start_sec"] < section[1] and item["end_sec"] > section[0])


def _assert_exact(result, faults, context):
    detected, false_alarms, found = _faults_found(result, faults)
    assert false_alarms == {}, (context, detected)
    assert found == set(faults), (context, detected, faults)


@pytest.mark.parametrize("bpm,design", [(104, 0), (126, 1)])
@pytest.mark.parametrize("no_fault", [True, False])
def test_late_whoosh_under_kick_and_click(tmp_path, bpm, design, no_fault):
    """A whoosh 250 ms late on beats that also carry a kick and a text-pop click."""
    video, cues, faults = _soundtrack(tmp_path, bpm, design, hardened=True, no_fault=no_fault,
                                      accents=True, pop_faults={}, whoosh_faults={12: .250, 24: .250})
    _assert_exact(_check(video, cues, bpm, tmp_path), faults, (bpm, design, no_fault))


@pytest.mark.parametrize("bpm,design", [(104, 0), (126, 1)])
@pytest.mark.parametrize("accents", [False, True])
def test_early_click_is_early_not_missing(tmp_path, bpm, design, accents):
    video, cues, faults = _soundtrack(tmp_path, bpm, design, hardened=True, accents=accents,
                                      pop_faults={8: -.120, 20: -.120, 28: -.120}, whoosh_faults={})
    result = _check(video, cues, bpm, tmp_path)
    _assert_exact(result, faults, (bpm, design, accents))
    for label in faults:
        cue = next(item for item in result["cues"] if item["label"] == label)
        assert -180.0 <= cue["offset_ms"] <= -60.0, cue


@pytest.mark.parametrize("bpm,design", [(104, 1), (126, 0)])
@pytest.mark.parametrize("no_fault", [True, False])
def test_missing_offbeat_click_with_hats_running(tmp_path, bpm, design, no_fault):
    """UI ticks between kicks, on the 8th-note hats (and shakers); one is missing."""
    video, cues, faults = _soundtrack(tmp_path, bpm, design, hardened=True, no_fault=no_fault,
                                      pop_faults={}, whoosh_faults={},
                                      ticks=(1, 5, 9, 13, 17, 21, 25, 29, 33), tick_faults={25: None})
    _assert_exact(_check(video, cues, bpm, tmp_path), faults, (bpm, design, no_fault))
