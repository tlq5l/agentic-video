#!/usr/bin/env python3
"""Missing-sound signal study (climb, 2026-11-02): offline scores for one-off cues.

Each one-off cue occurrence gets four scores. Higher = more evidence the sound is PRESENT.
  S1  leave-one-out template gain: the group's onset-spectrum template is fitted by
      src/media/audio.py `_overlap_templates` WITHOUT the scored occurrence; coincident companions'
      templates are subtracted from the observed attack spectrum (as `_template_match` does); score =
      projection of the remainder on the template / template squared norm (1 = full sound, 0 = none).
  S2  bar-phase control: band rise (60 ms after minus 60 ms before) in the group's signature bands,
      minus the median rise at the same bar phase in the 4 nearest bars where the group is not cued.
  S3  occurrence outlier: robust z of the occurrence's S2 against the group's other occurrences
      (groups with 3 or more occurrences in the file).
  S4  matched onset: best cosine of the group template to a detected onset within +-40 ms, minus the
      best cosine of the companions' templates at that onset.

Labels: the removed cue in its variant is ABSENT (positive). Scored one-offs in the baseline files that
are not answer-key faults (0.5 s) and not removed are PRESENT (negatives). The four answer-key
cue_missing entries are scored on the faulted -pristine originals and reported separately.
Inputs: cues.json and the audio of each video.mp4 (decoded with src/media/audio.py `_decode_pcm`).
Generators and traces are not read; the manifest is read only for which cue was removed.

Run from the worktree root: .venv/bin/python scripts/climb/signal_study.py
Writes .climb-tmp/missing/signals.json and prints one table.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import missing_bench as M  # noqa: E402  (has_skip, read_pcm, WORK, MEDIA, BENCH, fault_times)
from src.media import audio as A  # noqa: E402

NAN = float("nan")
SR = M.SAMPLE_RATE
COINC = 0.005  # sync_check default coincidence_ms
WIN = 0.060  # rise window (s)
ONSET_WIN = 0.040  # S4 onset window (s)
CTRL_RANGE = 8  # bars searched each side for the control
CTRL_N = 4  # nearest uncued bars used
KEY_TOL = 0.5  # answer-key fault window (s), as in missing_bench.pick_case
REMOVAL_TOL = 0.02  # a cue at a removal time (s)
ROBUST_SCALE_FLOOR = 0.1  # dB, floor for the S3 MAD scale
KEY_CUES = [("newtest2", 4.364), ("newtest5", 25.333), ("newtest10", 6.667), ("newtest", 8.0)]
SPECTRUM_EDGES = np.geomspace(40.0, 20000.0, 33)  # bins of A._spectral_attack
SPECTRUM_CENTRES = np.sqrt(SPECTRUM_EDGES[:-1] * SPECTRUM_EDGES[1:])
SIGS = ["S1", "S2", "S3", "S4"]


def norm_label(text: str) -> str:
    return " ".join(str(text).split())


def cue_list(raw: list) -> list:
    out = []
    for cue in raw:
        label = norm_label(cue.get("label", "cue"))
        out.append({"time_sec": float(cue.get("time_sec", cue.get("t"))), "label": label,
                    "group": A._cue_group_name(label)})
    return out


def bar_sec_for(cues: list):
    """No bpm reaches the checker run, so the bar is 4 x the median spacing of the most frequent cue group."""
    if not cues:
        return None
    top = Counter(c["group"] for c in cues).most_common(1)[0][0]
    times = sorted(c["time_sec"] for c in cues if c["group"] == top)
    if len(times) < 2:
        return None
    return 4.0 * float(np.median(np.diff(times)))


def spectrum_band_names(spectrum: np.ndarray) -> list:
    """Map a 32-bin attack spectrum onto the five checker bands, then pick the signature bands."""
    power = np.zeros(len(A.BANDS))
    for i, centre in enumerate(SPECTRUM_CENTRES):
        for j, (_, lo, hi) in enumerate(A.BANDS):
            if lo <= centre < hi:
                power[j] += spectrum[i]
                break
    names = A._signature_bands(A._unit_vector(power))
    return names or [band[0] for band in A.BANDS]


def band_power(samples: np.ndarray, first: int, last: int, names: list) -> float:
    profile = A._band_profile(samples[first:last])
    return float(sum(10.0 ** (profile[name] / 10.0) for name in names))


def rise_db(samples: np.ndarray, t: float, names: list) -> float:
    width = int(round(WIN * SR))
    at = int(round(t * SR))
    if at - width < 0 or at + width > len(samples):
        return NAN
    after = band_power(samples, at, at + width, names)
    before = band_power(samples, at - width, at, names)
    return 10.0 * math.log10(after + 1e-12) - 10.0 * math.log10(before + 1e-12)


def control_db(samples: np.ndarray, t: float, names: list, group_times: list, bar, duration: float) -> float:
    if bar is None or bar <= 0.0:
        return NAN
    candidates = []
    for k in range(-CTRL_RANGE, CTRL_RANGE + 1):
        if k == 0:
            continue
        tk = t + k * bar
        if tk - WIN < 0.0 or tk + WIN > duration:
            continue
        if any(abs(tk - x) <= 0.1 for x in group_times):
            continue
        candidates.append((abs(k), k, tk))
    candidates.sort()
    values = [rise_db(samples, tk, names) for _, _, tk in candidates[:CTRL_N]]
    values = [v for v in values if math.isfinite(v)]
    return float(np.median(values)) if len(values) >= 2 else NAN


def align_to_wav(samples: np.ndarray, wav: np.ndarray, max_lag: int = 2400):
    """Lag (samples) between decoded video audio and the WAV it was muxed from; returns the shifted audio.

    Positive lag = the video audio is late (AAC priming). aligned[t] = samples[t + lag].
    """
    seg = min(len(wav), len(samples) - max_lag, 30 * SR)
    ref = wav[:seg]
    sig = samples[: seg + max_lag]
    n = 1 << int(math.ceil(math.log2(len(ref) + len(sig))))
    corr = np.fft.irfft(np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n)), n)
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    lag = int(lags[int(np.argmax(corr[lags % n]))])
    if lag > 0:
        aligned = np.r_[samples[lag:], np.zeros(lag)]
    elif lag < 0:
        aligned = np.r_[np.zeros(-lag), samples[:lag]]
    else:
        aligned = samples
    return aligned, lag


def mono_wav(path: Path) -> np.ndarray:
    pcm, _ = M.read_pcm(path)
    return pcm.astype(np.float64).mean(axis=1) / 32768.0


def analyse(video: Path, cues_path: Path, ref_wav: Path) -> dict:
    """Scores for every one-off cue occurrence in one file."""
    cues = cue_list(json.loads(cues_path.read_text(encoding="utf-8")))
    raw_samples, _, _ = A._decode_pcm(str(video))
    raw_samples = np.asarray(raw_samples, dtype=np.float64)
    samples, lag = align_to_wav(raw_samples, mono_wav(ref_wav))
    duration = len(samples) / SR
    onsets = A.audio_onsets(str(video), min_strength=0.10, min_gap_sec=0.0)["onsets"]
    onset_times = [float(o["time_sec"]) - lag / SR for o in onsets]
    bar = bar_sec_for(cues)
    group_times = defaultdict(list)
    for c in cues:
        group_times[c["group"]].append(c["time_sec"])
    recs = []
    for i, cue in enumerate(cues):
        if M.has_skip(cue["label"]):
            continue
        t, g = cue["time_sec"], cue["group"]
        loo = cues[:i] + cues[i + 1:]
        models = A._overlap_templates(loo, samples, COINC)
        comps = sorted(A._coincident_groups(cues, t, COINC) - {g})
        entry = models.get(g)
        tpl = entry["template"] if entry else None
        observed = A._spectral_attack(samples, t)
        if tpl is not None:
            companion = sum((models[n]["template"] for n in comps if n in models), np.zeros_like(tpl))
            remainder = np.maximum(observed - companion, 0.0)
            s1 = float(remainder @ tpl) / max(float(tpl @ tpl), 1e-12)
        else:
            s1 = NAN
        names = spectrum_band_names(tpl if tpl is not None else observed)
        s2 = rise_db(samples, t, names) - control_db(samples, t, names, group_times[g], bar, duration)
        near = [T for T in onset_times if abs(T - t) <= ONSET_WIN]
        s4 = NAN
        if tpl is not None and near:
            best = None
            for T in near:
                spectrum = A._spectral_attack(samples, T)
                cg = A._cosine(spectrum, tpl)
                if best is None or cg > best[0]:
                    cc = max((A._cosine(spectrum, models[n]["template"]) for n in comps if n in models), default=0.0)
                    best = (cg, cc)
            s4 = best[0] - best[1]
        recs.append({"time": t, "label": cue["label"], "group": g, "S1": s1, "S2": s2, "S3": NAN, "S4": s4,
                     "n_group": len(group_times[g]), "bands": names, "companions": comps})
    by_group = defaultdict(list)
    for r in recs:
        by_group[r["group"]].append(r)
    for rs in by_group.values():
        for r in rs:
            others = [o["S2"] for o in rs if o is not r and math.isfinite(o["S2"])]
            if r["n_group"] < 3 or len(others) < 2 or not math.isfinite(r["S2"]):
                continue
            med = float(np.median(others))
            scale = max(1.4826 * float(np.median(np.abs(np.asarray(others) - med))), ROBUST_SCALE_FLOOR)
            r["S3"] = (r["S2"] - med) / scale
    return {"recs": recs, "lag": lag, "duration": duration}


def find_one(recs: list, time: float, label: str, what: str) -> dict:
    hits = [r for r in recs if abs(r["time"] - time) <= REMOVAL_TOL and r["label"] == norm_label(label)]
    if len(hits) != 1:
        raise SystemExit(f"{what}: expected one one-off cue at {time} labelled {label!r}, found {len(hits)}")
    return hits[0]


def row(split: str, kind: str, case: str, rec: dict) -> dict:
    out = {"split": split, "kind": kind, "case": case, "time": rec["time"], "label": rec["label"],
           "group": rec["group"]}
    for sig in SIGS:
        out[sig] = rec[sig]
    return out


def arr(rows: list, sig: str) -> np.ndarray:
    return np.array([r[sig] for r in rows], dtype=float)


def auc_absent(absent: np.ndarray, present: np.ndarray) -> float:
    """P(absent scores below present); ties count half. NaN values are dropped."""
    a = absent[np.isfinite(absent)]
    p = present[np.isfinite(present)]
    if a.size == 0 or p.size == 0:
        return NAN
    d = a[:, None] - p[None, :]
    return float(np.mean((d < 0).astype(float) + 0.5 * (d == 0)))


def flag_single(x: np.ndarray, thr: float) -> np.ndarray:
    return np.isfinite(x) & (x < thr)


def flag_pair(x: np.ndarray, y: np.ndarray, tx: float, ty: float) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (x < tx) & (y < ty)


def main() -> int:
    bench = json.loads(M.BENCH.read_text(encoding="utf-8"))
    manifest = json.loads((M.WORK / "manifest.json").read_text(encoding="utf-8"))
    cases = list(bench["cases"])
    split_of = {c: bench["cases"][c]["split"] for c in cases}
    removed_times = {c: [r["time_sec"] for r in bench["cases"][c]["removals"]] for c in cases}
    lags = {}
    base = {}
    for case in cases:
        d = M.WORK / case / "base"
        base[case] = analyse(d / "video.mp4", d / "cues.json", d / "audio.wav")
        lags[f"{case}/base"] = base[case]["lag"]
        print(f"baseline {case}: {len(base[case]['recs'])} one-off occurrences, lag {base[case]['lag']} samples",
              file=sys.stderr)

    dev_pos, held_pos, pairs = [], [], []
    for e in manifest:
        if e["baseline"]:
            continue
        video = Path(e["video"])
        res = analyse(video, Path(e["cues"]), video.parent / "audio.wav")
        lags[f"{e['case']}/{e['variant']}"] = res["lag"]
        rec = find_one(res["recs"], float(e["time_sec"]), e["cue_label"], f"{e['case']} {e['variant']}")
        pos = row(e["split"], "absent", e["case"], rec)
        (dev_pos if e["split"] == "dev" else held_pos).append(pos)
        base_rec = find_one(base[e["case"]]["recs"], float(e["time_sec"]), e["cue_label"], f"{e['case']} base")
        pairs.append({"case": e["case"], "time": rec["time"], "label": rec["label"],
                      "base": {s: base_rec[s] for s in SIGS}, "variant": {s: rec[s] for s in SIGS}})
        print(f"variant {e['case']} {e['variant']} t={rec['time']} done", file=sys.stderr)

    dev_neg, held_neg = [], []
    for case in cases:
        faults = M.fault_times(case)
        for rec in base[case]["recs"]:
            if any(abs(rec["time"] - f) <= KEY_TOL for f in faults):
                continue
            if any(abs(rec["time"] - t) <= REMOVAL_TOL for t in removed_times[case]):
                continue
            target = dev_neg if split_of[case] == "dev" else held_neg
            target.append(row(split_of[case], "present", case, rec))

    key_rows = []
    for case, t in KEY_CUES:
        p = M.MEDIA / (case + "-pristine")
        res = analyse(p / "out.mp4", p / "cues.json", M.WORK / case / "base" / "audio.wav")
        lags[f"{case}/pristine"] = res["lag"]
        hits = [r for r in res["recs"] if abs(r["time"] - t) <= REMOVAL_TOL]
        if not hits:
            raise SystemExit(f"key entry {case} {t}: no one-off cue at that time in the pristine cues")
        key_rows.append(row(split_of[case], "key_cue_missing", case, min(hits, key=lambda r: abs(r["time"] - t))))

    dP = {s: arr(dev_pos, s) for s in SIGS}
    dN = {s: arr(dev_neg, s) for s in SIGS}
    hP = {s: arr(held_pos, s) for s in SIGS}
    hN = {s: arr(held_neg, s) for s in SIGS}
    kk = {s: arr(key_rows, s) for s in SIGS}

    # Single signals: threshold = the lowest dev negative (flag only below it), so dev FA is 0.
    singles = {}
    for s in SIGS:
        neg = dN[s][np.isfinite(dN[s])]
        thr = float(neg.min()) if neg.size else -math.inf
        singles[s] = {
            "thr": thr,
            "dev_auc": auc_absent(dP[s], dN[s]), "held_auc": auc_absent(hP[s], hN[s]),
            "dev_caught": int(flag_single(dP[s], thr).sum()), "dev_n": len(dev_pos),
            "dev_fa": int(flag_single(dN[s], thr).sum()),
            "held_caught": int(flag_single(hP[s], thr).sum()), "held_n": len(held_pos),
            "held_fa": int(flag_single(hN[s], thr).sum()), "held_neg_n": len(held_neg),
            "key_flagged": int(flag_single(kk[s], thr).sum()), "key_n": len(key_rows),
            "nan_n": int(np.sum(~np.isfinite(np.concatenate([dP[s], dN[s], hP[s], hN[s]])))),
            "items_n": len(dev_pos) + len(dev_neg) + len(held_pos) + len(held_neg),
        }

    # Pairs: both signals below their own thresholds (AND). Thresholds searched jointly on dev.
    pooled = {}
    for s in SIGS:
        v = np.concatenate([dP[s], dN[s]])
        v = v[np.isfinite(v)]
        pooled[s] = (float(v.mean()), float(v.std()) or 1.0)

    def zscore(x, s):
        mu, sd = pooled[s]
        return (x - mu) / sd

    def candidates(x, y):
        v = np.concatenate([x, y])
        v = v[np.isfinite(v)]
        return np.unique(v) + 1e-9

    pair_results = {}
    for a in SIGS:
        for b in SIGS:
            if a >= b:
                continue
            best = None
            for ta in candidates(dP[a], dN[a]):
                for tb in candidates(dP[b], dN[b]):
                    if flag_pair(dN[a], dN[b], ta, tb).any():
                        continue
                    caught = int(flag_pair(dP[a], dP[b], ta, tb).sum())
                    key = (caught, ta + tb)
                    if best is None or key > best[0]:
                        best = (key, ta, tb)
            if best is None:
                pair_results[f"{a}+{b}"] = None
                continue
            _, ta, tb = best
            mP = np.minimum(zscore(dP[a], a), zscore(dP[b], b))
            mN = np.minimum(zscore(dN[a], a), zscore(dN[b], b))
            pair_results[f"{a}+{b}"] = {
                "ta": float(ta), "tb": float(tb), "dev_caught": int(best[0][0]),
                "dev_auc": auc_absent(mP, mN),
                "held_auc": auc_absent(np.minimum(zscore(hP[a], a), zscore(hP[b], b)),
                                       np.minimum(zscore(hN[a], a), zscore(hN[b], b))),
                "dev_fa": int(flag_pair(dN[a], dN[b], ta, tb).sum()),
                "held_caught": int(flag_pair(hP[a], hP[b], ta, tb).sum()),
                "held_fa": int(flag_pair(hN[a], hN[b], ta, tb).sum()),
                "key_flagged": int(flag_pair(kk[a], kk[b], ta, tb).sum()),
                "nan_n": int(np.sum(~np.isfinite(np.concatenate([dP[a], dN[a], hP[a], hN[a]])) |
                                    ~np.isfinite(np.concatenate([dP[b], dN[b], hP[b], hN[b]])))),
                "held_neg_n": len(held_neg),
            }
    ranked = sorted((k for k, v in pair_results.items() if v), key=lambda k: (-pair_results[k]["dev_caught"],
                                                                            -pair_results[k]["dev_auc"]))
    top_pairs = ranked[:2]

    paired = {}
    for s in SIGS:
        bs = np.array([p["base"][s] for p in pairs], dtype=float)
        vs = np.array([p["variant"][s] for p in pairs], dtype=float)
        ok = np.isfinite(bs) & np.isfinite(vs)
        paired[s] = {"n": int(ok.sum()), "median_base": float(np.median(bs[ok])) if ok.any() else NAN,
                     "median_variant": float(np.median(vs[ok])) if ok.any() else NAN,
                     "variant_below_base": int(np.sum(vs[ok] < bs[ok]))}

    def fmt(x, nd=3):
        return "nan" if not math.isfinite(x) else f"{x:.{nd}f}"

    print("lags (samples, decoded video vs wav): " + ", ".join(f"{k}={v}" for k, v in sorted(lags.items())))
    print(f"dev positives {len(dev_pos)}, dev negatives {len(dev_neg)}, held positives {len(held_pos)}, "
          f"held negatives {len(held_neg)}, key entries {len(key_rows)}")
    print("signal  dev_AUC held_AUC NaN_n  thr        dev_caught(FA) held_caught held_FA  key_flagged")
    for s in SIGS:
        st = singles[s]
        print(f"{s:<7} {fmt(st['dev_auc'])}  {fmt(st['held_auc'])}  {st['nan_n']:>3}/{st['items_n']}  "
              f"{fmt(st['thr'], 4):<10} {st['dev_caught']}/{st['dev_n']} (FA {st['dev_fa']})  "
              f"{st['held_caught']}/{st['held_n']}  {st['held_fa']}/{st['held_neg_n']}  "
              f"{st['key_flagged']}/{st['key_n']}")
    for name in top_pairs:
        p = pair_results[name]
        print(f"{name:<7} {fmt(p['dev_auc'])}  {fmt(p['held_auc'])}  {p['nan_n']:>3}/{singles['S1']['items_n']}  "
              f"tA={fmt(p['ta'], 4)} tB={fmt(p['tb'], 4)} {p['dev_caught']}/{len(dev_pos)} (FA {p['dev_fa']})  "
              f"{p['held_caught']}/{len(held_pos)}  {p['held_fa']}/{p['held_neg_n']}  "
              f"{p['key_flagged']}/{len(key_rows)}")
    print("paired check (removed cue, baseline vs variant, median): "
          + "; ".join(f"{s} {fmt(paired[s]['median_base'], 2)} -> {fmt(paired[s]['median_variant'], 2)} "
                      f"(variant lower in {paired[s]['variant_below_base']}/{paired[s]['n']})" for s in SIGS))
    print("key entries (pristine, extra positives):")
    for kr in key_rows:
        print(f"  {kr['case']} t={kr['time']:.3f} {kr['label']!r} " + " ".join(f"{s}={fmt(kr[s])}" for s in SIGS))

    out = {"thresholds": "single: flag if score < min dev negative; pair: flag if both < their jointly searched "
                         "thresholds (zero dev FA, max dev catches, ties by larger threshold sum)",
           "singles": singles, "pairs": {k: pair_results[k] for k in pair_results}, "top_pairs": top_pairs,
           "paired": paired, "lags": lags, "keys": key_rows, "rows_dev_pos": dev_pos, "rows_held_pos": held_pos,
           "rows_dev_neg": dev_neg, "rows_held_neg": held_neg, "pairs_base_variant": pairs}
    (M.WORK / "signals.json").write_text(json.dumps(out, indent=1, default=lambda v: None if (isinstance(v, float) and not math.isfinite(v)) else v), encoding="utf-8")
    print("wrote", M.WORK / "signals.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
