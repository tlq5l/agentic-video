#!/usr/bin/env python3
"""Missing-sound bench for the render checker (climb step, 2026-11-02).

Question: when ONE one-off sound is removed from a case's soundtrack, does the checker flag
that sound at its cue, and does it stay quiet otherwise?

Subcommands (run with .venv/bin/python from the worktree root):
  trace CASE...    log every placement call of each case's make_audio.py
  pick             choose the removals and write scripts/climb/missing_bench.json
  build            render the baseline and one variant per removal, mux each onto out.mp4
  score [--label]  run the checker on every baseline and variant and print the totals

Temp output lives in .climb-tmp/missing/ (gitignored). The pristine case folders are only read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import inspect
import json
import linecache
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORK = REPO / ".climb-tmp" / "missing"
MEDIA = Path(os.path.expanduser("~/.local/share/media-loop-work"))
KEY = HERE / "answer_key.json"
BENCH = HERE / "missing_bench.json"
PY = REPO / ".venv" / "bin" / "python"
MAX_PARALLEL = 4
SAMPLE_RATE = 48000  # the ten generators place sounds at round(t * 48000)
MAX_RESIDUAL_LSB = 2  # a variant must equal its baseline outside the removed span to within this
FAULT_TOL = 0.5  # seconds: a removal may not sit this close to an answer-key fault
MIN_GAP_SEC = 1.0  # removals in one case are at least this far apart
# Checker kinds that count as "caught" (the cue family in src/loop.py). cue_ambiguous is a review
# item, not a verdict, so it does not count. cue_mismatch and cue_weak do not exist in the checker.
CAUGHT_KINDS = frozenset({"cue_missing", "cue_early", "cue_late"})
STRICT_KINDS = frozenset({"cue_missing"})
# Labels with any of these substrings are drum, beat or bed sounds, not one-offs.
# Extra words (beyond the brief's list) are patterns or drums too: when unsure, skip.
SKIP_WORDS = ("kick", "snare", "hat", "shaker", "clap groove", "bass", "pad", "bed", "groove",
              "ostinato", "arpeggio", "tom", "drum", "beat", "loop", "thump", "sub")
N_REMOVALS = {"dev": 4, "held_out": 2}
# case -> (split, placement function, name of its time parameter). Every placement time is seconds.
CASES = {
    "newtest": ("dev", "place", "start_sec"),
    "newtest2": ("dev", "add", "time_sec"),
    "newtest3": ("dev", "add", "time_sec"),
    "newtest4": ("dev", "add", "time_sec"),
    "newtest5": ("dev", "place", "t0"),
    "newtest6": ("dev", "place", "start"),
    "newtest7": ("held_out", "place", "start_sec"),
    "newtest8": ("held_out", "place", "start"),
    "newtest9": ("held_out", "place", "t"),
    "newtest10": ("held_out", "add", "t0"),
}
# Helper names that appear in placement lines but are not sound makers.
HELPERS = {"place", "add", "int", "round", "float", "min", "max", "len", "abs", "range", "sum",
           "sorted", "list", "tuple", "dict", "zip", "enumerate", "str", "db", "tvec", "sec", "idx",
           "tt", "at", "bt", "beat_time", "bar_time", "band", "norm", "pan_gains", "fade_edges",
           "smooth", "normalise", "normalize"}


def copy_src(case: str) -> Path:
    dst = WORK / "src" / case / "make_audio.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MEDIA / (case + "-pristine") / "make_audio.py", dst)
    return dst


def render(src: Path, out_wav: Path, fn: str, param: str, skip: set, trace: Path | None) -> None:
    """One render in its own process: the generators build a module-level mix, so they run once each."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), str(Path(__file__).resolve()), "_render", "--src", str(src), "--out", str(out_wav),
           "--fn", fn, "--param", param, "--skip", ",".join(str(i) for i in sorted(skip)) or "-"]
    if trace is not None:
        cmd += ["--trace", str(trace)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(src.parent))
    if proc.returncode != 0 or not out_wav.exists():
        raise SystemExit("render failed ({} skip={}):\n{}".format(src, sorted(skip), proc.stderr[-1500:]))


def run_render(argv: list[str]) -> None:
    """Hidden subcommand: load the copied generator, wrap its placement function, run its main()."""
    ap = argparse.ArgumentParser(prog="_render")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fn", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--skip", default="-")
    ap.add_argument("--trace", default=None)
    a = ap.parse_args(argv)
    skip = set() if a.skip == "-" else {int(x) for x in a.skip.split(",")}
    spec = importlib.util.spec_from_file_location("make_audio_copy", a.src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orig = getattr(mod, a.fn)
    sig = inspect.signature(orig)
    log: list[dict] = []

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        idx = len(log)
        caller = inspect.currentframe().f_back
        line = linecache.getline(caller.f_code.co_filename, caller.f_lineno).strip()
        # Which sound this is: the sound makers named on the line (a local holding a sound-maker
        # function resolves to that function's name), else the caller's sound-name strings
        # (a loop variable such as name = "stamp"). skip_names are the raw names, for drum/bed checks.
        def is_maker(v) -> bool:
            return callable(v) and getattr(v, "__module__", None) == mod.__name__ and hasattr(v, "__name__")

        call_names = [n for n in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\(", line) if n not in HELPERS]
        first_arg = re.match(r"\s*\w+\(\s*([A-Za-z_]\w*)", line)
        tags = []
        for n in call_names:
            v = caller.f_locals.get(n, caller.f_globals.get(n))
            tags.append(v.__name__ if is_maker(v) else n)
        if not tags:
            for v in caller.f_locals.values():
                if isinstance(v, str) and re.fullmatch(r"[a-z][a-z_]{2,30}", v):
                    tags.append(v)
                elif is_maker(v):
                    tags.append(v.__name__)
        skip_names = call_names + ([first_arg.group(1)] if first_arg else [])
        # The removed sound's span: the generator rounds start * SR like this (all ten do), and the
        # signal is the shortest array argument (the bus is always the longest one).
        t_sec = float(bound.arguments[a.param])
        arrays = [v for v in bound.arguments.values() if isinstance(v, np.ndarray)]
        sig_len = int(max(min(arrays, key=lambda v: v.size).shape)) if arrays else 0
        log.append({"call_index": idx, "time_sec": t_sec,
                    "span_start": int(round(t_sec * SAMPLE_RATE)), "span_len": sig_len,
                    "caller_line": line, "caller_fn": caller.f_code.co_name,
                    "tags": sorted(set(tags)), "skip_names": sorted(set(skip_names))})
        if idx in skip:
            return None
        return orig(*args, **kwargs)

    setattr(mod, a.fn, wrapper)
    sys.argv = [a.src, str(Path(a.out).resolve())]
    mod.main()
    if a.trace:
        Path(a.trace).write_text(json.dumps({"calls": log}, indent=1), encoding="utf-8")


def cmd_trace(cases: list[str]) -> None:
    for case in cases:
        _split, fn, param = CASES[case]
        src = copy_src(case)
        d = WORK / case
        render(src, d / "trace_render.wav", fn, param, set(), d / "trace.json")
        calls = json.loads((d / "trace.json").read_text(encoding="utf-8"))["calls"]
        last = calls[-1]["time_sec"] if calls else None
        print(f"{case}: {len(calls)} placement calls, last at {last}")


SOUND_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_ \-]{1,30}")


def tag_words(c: dict) -> set:
    """Words that name the sound a placement makes (see the trace tags)."""
    return words(" ".join(t for t in c["tags"] if SOUND_NAME.fullmatch(t)))


def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def overlaps(label: str, tw: set) -> bool:
    """A cue label names a sound when one of its words matches one of the sound's words (4-letter prefix)."""
    return any(a[:4] == b[:4] for a in words(label) for b in tw)


def has_skip(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in SKIP_WORDS)


def sound_type(label: str) -> str:
    """The sound's name without its lyric part or cue number, e.g. 'whoosh' for 'whoosh | FIRST LIGHT'."""
    return re.sub(r"[\d.]+", "", label.split("|")[0]).strip().lower()


def fault_times(case: str) -> list:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    times = []
    for group in ("dev", "held_out"):
        for name, entries in key.get(group, {}).items():
            if name in (case, case + "-pristine"):
                times += [float(e["time"]) for e in entries]
    return times


def pick_case(case: str) -> tuple:
    split, _fn, _param = CASES[case]
    calls = json.loads((WORK / case / "trace.json").read_text(encoding="utf-8"))["calls"]
    cues = json.loads((MEDIA / (case + "-pristine") / "cues.json").read_text(encoding="utf-8"))
    faults = fault_times(case)
    dur = max(float(c["time_sec"]) for c in cues)
    tws = {c["call_index"]: tag_words(c) for c in calls}
    cands = []
    for c in calls:
        t, tw = float(c["time_sec"]), tws[c["call_index"]]
        if not tw or has_skip(" ".join(c["skip_names"])):
            continue
        rel = [q for q in cues if abs(float(q["time_sec"]) - t) <= 0.02 and overlaps(str(q["label"]), tw)]
        if len(rel) != 1 or has_skip(str(rel[0]["label"])):
            continue
        label = str(rel[0]["label"])
        twins = [d for d in calls if d["call_index"] != c["call_index"]
                 and abs(float(d["time_sec"]) - t) <= 0.02 and overlaps(label, tws[d["call_index"]])]
        if twins or any(abs(f - t) <= FAULT_TOL for f in faults):
            continue
        cands.append({"call_index": c["call_index"], "time_sec": round(t, 4), "cue_label": label,
                      "caller_line": c["caller_line"], "sound": sound_type(label)})
    n = N_REMOVALS[split]
    chosen: list = []
    for i in range(n):
        target = (i + 0.5) * dur / n
        used = {x["sound"] for x in chosen}
        opts = [c for c in cands if c["sound"] not in used
                and all(abs(c["time_sec"] - x["time_sec"]) >= MIN_GAP_SEC for x in chosen)]
        if not opts:
            break
        chosen.append(min(opts, key=lambda c: abs(c["time_sec"] - target)))
    chosen.sort(key=lambda c: c["time_sec"])
    note = "" if len(chosen) == n else f"wanted {n}, found {len(chosen)} suitable one-off sounds"
    return chosen, note, len(cands)


def cmd_pick() -> None:
    cases_out = {}
    for case, (split, fn, param) in CASES.items():
        removals, note, n_cands = pick_case(case)
        cases_out[case] = {"split": split, "place_fn": fn, "time_param": param,
                           "removals": removals, "note": note}
        print(f"{case} ({split}): {len(removals)} removals from {n_cands} candidates {note}")
        for r in removals:
            print(f"   call {r['call_index']} t={r['time_sec']} {r['cue_label']!r} sound={r['sound']}")
    doc = {"_about": "Missing-sound bench removals, written by `missing_bench.py pick`. One removal per "
                     "one-off sound: a single cue within 0.02 s whose label names the sound, no drum/beat/bed "
                     "label, no answer-key fault within 0.5 s, and no other placement of that sound at that time.",
           "cases": cases_out}
    BENCH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def same_render(a: Path, b: Path) -> bool:
    x = np.frombuffer(a.read_bytes(), dtype=np.uint8)
    y = np.frombuffer(b.read_bytes(), dtype=np.uint8)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def mux_variant(case: str, split: str, variant: str, wav: Path, pristine: Path,
                manifest: list, rem: dict | None) -> None:
    vdir = wav.parent
    for name in ("cues.json", "INTENT.md"):
        shutil.copy2(pristine / name, vdir / name)
    video = vdir / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(pristine / "out.mp4"), "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-ac", "2", "-shortest", str(video)], check=True)
    manifest.append({"case": case, "split": split, "variant": variant, "baseline": rem is None,
                     "call_index": rem and rem["call_index"], "time_sec": rem and rem["time_sec"],
                     "cue_label": rem and rem["cue_label"], "caller_line": rem and rem["caller_line"],
                     "gain": rem and rem["gain"], "max_residual_lsb": rem and rem["max_residual_lsb"],
                     "span": rem and rem["span"],
                     "video": str(video), "cues": str(vdir / "cues.json"), "intent": str(vdir / "INTENT.md")})


def read_pcm(path: Path) -> tuple:
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        raw = w.readframes(params.nframes)
    if params.sampwidth != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {params.sampwidth} bytes per sample")
    return np.frombuffer(raw, dtype="<i2").reshape(params.nframes, params.nchannels), params


def write_pcm(path: Path, pcm: np.ndarray, params) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())


def pin_gain(base_wav: Path, var_wav: Path, trace_path: Path, idx: int) -> tuple:
    """Fit one gain g so g x variant matches the baseline outside the removed sound's span, rewrite
    the variant with it, and fail loudly when any sample outside the span is off by more than
    MAX_RESIDUAL_LSB. Why: the generators re-normalise the whole mix after a removal (peak
    normalisation), so the variant is otherwise a gain away from its baseline everywhere.
    The gain is applied to the written int16 audio (the mix is not reachable the same way in all ten
    generators), so the rounding costs at most about one LSB more."""
    base, params = read_pcm(base_wav)
    var, var_params = read_pcm(var_wav)
    if base.shape != var.shape or params.framerate != var_params.framerate:
        raise SystemExit(f"{var_wav}: shape or rate differs from the baseline")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rec = next(c for c in trace["calls"] if c["call_index"] == idx)
    n = base.shape[0]
    s0 = int(rec["span_start"])
    s1 = min(n, s0 + int(rec["span_len"]))
    keep = np.ones(n, dtype=bool)
    keep[s0:s1] = False
    b = base[keep].astype(np.float64).ravel()
    v = var[keep].astype(np.float64).ravel()
    g = float(np.dot(v, b) / np.dot(v, v))
    scaled = np.round(np.clip(var.astype(np.float64) / 32767.0 * g, -1.0, 1.0) * 32767.0).astype(np.int64)
    diff = int(np.max(np.abs(scaled[keep] - base[keep].astype(np.int64))))
    if diff > MAX_RESIDUAL_LSB:
        raise SystemExit(f"{var_wav}: max difference outside the removed span is {diff} LSB "
                         f"(> {MAX_RESIDUAL_LSB}) after gain {g:.6f}; span [{s0}, {s1})")
    write_pcm(var_wav, scaled, params)
    rec["gain"] = g
    rec["max_residual_lsb"] = diff
    rec["span"] = [s0, s1]
    trace_path.write_text(json.dumps(trace, indent=1), encoding="utf-8")
    return g, diff, (s0, s1)


def cmd_build() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    manifest: list = []
    for case, spec in bench["cases"].items():
        split, fn, param = spec["split"], spec["place_fn"], spec["time_param"]
        src = copy_src(case)
        d = WORK / case
        pristine = MEDIA / (case + "-pristine")
        base_wav = d / "base" / "audio.wav"
        render(src, base_wav, fn, param, set(), None)
        orig_wav = d / "orig_audio.wav"
        proc = subprocess.run([str(PY), str(src), str(orig_wav)], cwd=str(d), capture_output=True, text=True)
        if proc.returncode != 0 or not same_render(base_wav, orig_wav):
            raise SystemExit(f"{case}: wrapped no-skip render differs from the unwrapped generator {proc.stderr[-500:]}")
        mux_variant(case, split, "base", base_wav, pristine, manifest, None)
        for r in spec["removals"]:
            idx = r["call_index"]
            vwav = d / f"skip{idx}" / "audio.wav"
            vtrace = d / f"skip{idx}" / "trace.json"
            render(src, vwav, fn, param, {idx}, vtrace)
            if same_render(vwav, base_wav):
                print(f"{case} call {idx}: render unchanged (sound placed outside the audio); dropped")
                continue
            g, diff, span = pin_gain(base_wav, vwav, vtrace, idx)
            r = dict(r, gain=g, max_residual_lsb=diff, span=list(span))
            mux_variant(case, split, f"skip{idx}", vwav, pristine, manifest, r)
            print(f"{case} call {idx}: variant rendered, gain {g:.6f}, max residual outside span {diff} LSB")
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n = {s: sum(1 for m in manifest if m["split"] == s and not m["baseline"]) for s in ("dev", "held_out")}
    print(f"variants dev {n['dev']} held_out {n['held_out']}; baselines {sum(m['baseline'] for m in manifest)}")


def twin(c: dict, pool: list, tol: float) -> bool:
    return any(p["kind"] == c["kind"] and abs(float(p["start_sec"]) - float(c["start_sec"])) <= tol for p in pool)


def cmd_score(label: str) -> None:
    sys.path.insert(0, str(HERE))
    import score as S  # scripts/climb/score.py: run_check and TOL_SEC are reused, not copied
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
    root = WORK / "score_runs" / label

    def job(e: dict) -> dict:
        return S.run_check(Path(e["video"]), Path(e["cues"]), Path(e["intent"]), root / e["case"] / e["variant"])

    with cf.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        results = list(pool.map(job, manifest))
    for e, r in zip(manifest, results):
        if "error" in r:
            raise SystemExit(f"checker failed on {e['case']} {e['variant']}: {r['error']}")
        if not r.get("checked_ok", False):
            print(f"WARNING {e['case']} {e['variant']}: checked fields missing {r.get('checked_missing')}")
    base = {e["case"]: r["confirmed"] for e, r in zip(manifest, results) if e["baseline"]}
    rows = []
    for e, r in zip(manifest, results):
        if e["baseline"]:
            continue
        conf, pre = r["confirmed"], base[e["case"]]
        caught_items = [c for c in conf if c["kind"] in CAUGHT_KINDS
                        and abs(float(c["start_sec"]) - e["time_sec"]) <= S.TOL_SEC
                        and not twin(c, pre, S.TOL_SEC)]
        others = [c for c in conf if all(c is not x for x in caught_items) and not twin(c, pre, S.TOL_SEC)]
        rows.append({
            "case": e["case"], "split": e["split"], "time_sec": e["time_sec"], "cue_label": e["cue_label"],
            "caught": bool(caught_items), "strict": any(c["kind"] in STRICT_KINDS for c in caught_items),
            "kind_found": caught_items[0]["kind"] if caught_items else "-",
            "false_alarms": [f"{c['kind']}@{float(c['start_sec']):.2f}" for c in others],
        })
        row = rows[-1]
        fa = ",".join(row["false_alarms"]) or "none"
        print(f"{row['case']} {row['time_sec']:.3f} \"{row['cue_label']}\" caught={'yes' if row['caught'] else 'no'} "
              f"strict={'yes' if row['strict'] else 'no'} kind={row['kind_found']} FA={fa}")
    totals = {}
    for split in ("dev", "held_out"):
        rs = [x for x in rows if x["split"] == split]
        caught = sum(x["caught"] for x in rs)
        strict = sum(x["strict"] for x in rs)
        fa = sum(len(x["false_alarms"]) for x in rs)
        totals[split] = {"variants": len(rs), "caught": caught, "strict": strict, "false_alarms": fa,
                         "score": caught - 2 * fa}
        print(f"TOTAL {split}: variants {len(rs)} caught {caught} strict {strict} FA {fa} score {caught - 2 * fa}")
    base_counts = {c: len(v) for c, v in base.items()}
    out = {"label": label, "totals": totals, "rows": rows, "baseline_confirmed_per_case": base_counts,
           "kinds_caught": sorted(CAUGHT_KINDS), "kinds_strict": sorted(STRICT_KINDS)}
    (WORK / "score").mkdir(parents=True, exist_ok=True)
    (WORK / "score" / f"{label}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", WORK / "score" / f"{label}.json")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_render"]:
        run_render(argv[1:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("cases", nargs="+", choices=sorted(CASES))
    sub.add_parser("pick")
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--label", default="run")
    a = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "trace":
        cmd_trace(a.cases)
    elif a.cmd == "pick":
        cmd_pick()
    elif a.cmd == "build":
        cmd_build()
    else:
        cmd_score(a.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
