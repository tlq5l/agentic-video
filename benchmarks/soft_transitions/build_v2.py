"""Build the second, harder soft-transition benchmark set (deterministic, ffmpeg only).

Writes clips and manifest.json into $AGENTIC_VIDEO_BENCH_DIR/soft-transitions-v2.
Source films and screencasts are read from $SOFT_TRANSITIONS_SOURCES (flat folder,
names in FILES). Rebuilding needs them; the scorer does not.
Compared with v1: scenes come from real film shots and screencasts as well as
the two UI videos, scene lengths vary (1.2-4 s), transitions last 0.2-2.0 s,
some joins are between look-alike scenes, and the negatives include camera
moves, a short white flash, and screen activity. Every film segment and every
negative was checked by eye at 4 frames per second to be one shot.

Dev and holdout use different source videos, except the hand-labelled Sintel
dissolve in dev (Sintel shots are otherwise holdout only).
"""
import json
import os
import random
import shutil
from pathlib import Path

from build import BENCH_ROOT, LAVFI, MINE, REF, run

NORM = "fps={fps},scale=640:360,setsar=1,format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS"
OUT = BENCH_ROOT / "soft-transitions-v2"
SOURCES = Path(os.environ.get("SOFT_TRANSITIONS_SOURCES", "."))
FILES = {
    "bbb": SOURCES / "big_buck_bunny_720p.mp4",
    "ed": SOURCES / "elephants_dream_hd.mp4",
    "sintel": SOURCES / "sintel_512kb.mp4",
    "apollo": SOURCES / "time_of_apollo_1960_1975.webm",
    "mochi": SOURCES / "mochikit_intro_screencast.mp4",
    "wiki": SOURCES / "wikipedia_namespaces_screencast.webm",
    "piano": SOURCES / "piano.mp4",
    "ref": REF,
    "mine": MINE,
}
# key -> (file, start second, usable seconds). Each is the middle of one long shot.
SHOTS = {
    "bbb1": ("bbb", 49.89, 4.0), "bbb2": ("bbb", 82.8, 4.0), "bbb3": ("bbb", 93.6, 4.0), "bbb4": ("bbb", 106.3, 4.0),
    "bbb5": ("bbb", 235.2, 4.0), "bbb6": ("bbb", 245.1, 4.0), "bbb7": ("bbb", 286.5, 4.0),
    "ed0": ("ed", 57.1, 4.0), "ed1": ("ed", 73.1, 4.0), "ed2": ("ed", 82.1, 4.0), "ed3": ("ed", 90.9, 4.0),
    "ed4": ("ed", 153.3, 4.0), "ed5": ("ed", 184.1, 4.0), "ed6": ("ed", 226.6, 4.0), "ed7": ("ed", 251.9, 4.0),
    "mochi0": ("mochi", 64.7, 4.0), "mochi1": ("mochi", 309.1, 4.0), "mochi3": ("mochi", 336.6, 4.0),
    "M1": ("mine", 0.3, 3.0), "M2": ("mine", 10.0, 3.0), "M3": ("mine", 28.8, 3.0), "M4": ("mine", 17.5, 3.0),
    "sintel0": ("sintel", 34.9, 4.0), "sintel1": ("sintel", 75.8, 4.0), "sintel2": ("sintel", 92.9, 4.0),
    "sintel5": ("sintel", 344.0, 4.0),
    "apollo0": ("apollo", 136.6, 4.0), "apollo1": ("apollo", 181.0, 4.0), "apollo2": ("apollo", 190.2, 4.0),
    "apollo3": ("apollo", 215.1, 4.0), "apollo4": ("apollo", 224.5, 4.0), "apollo5": ("apollo", 242.6, 4.0),
    "wiki0": ("wiki", 45.4, 4.0), "wiki1": ("wiki", 73.5, 4.0), "wiki3": ("wiki", 113.5, 4.0), "wiki4": ("wiki", 123.5, 4.0),
    "piano0": ("piano", 20.0, 4.0),
    "R1": ("ref", 0.3, 2.8), "R2": ("ref", 11.5, 3.0), "R3": ("ref", 22.2, 3.0), "R4": ("ref", 29.5, 2.4),
    "R5": ("ref", 17.5, 3.0),
}
DEV_POOL = ["bbb1", "bbb2", "bbb3", "bbb4", "bbb5", "bbb6", "bbb7", "ed0", "ed1", "ed2", "ed3", "ed4", "ed5", "ed6",
            "ed7", "mochi0", "mochi1", "mochi3", "M1", "M2", "M3", "M4", "L1", "L3"]
HOLD_POOL = ["sintel0", "sintel1", "sintel2", "sintel5", "apollo0", "apollo1", "apollo2", "apollo3", "apollo4",
             "apollo5", "wiki0", "wiki1", "wiki3", "wiki4", "piano0", "R1", "R2", "R3", "R4", "R5", "L2", "L4"]
DEV_KINDS = ["cut", "fade", "dissolve", "slideleft", "slideup", "slidedown", "slideright", "wipeleft", "zoomin",
             "smoothup", "circleopen", "fadeblack"]
HOLD_KINDS = ["cut", "fade", "wiperight", "radial", "hblur", "fadewhite", "pixelize", "smoothleft", "squeezev",
              "distance", "vertopen", "circleclose", "diagtl"]
DURATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
# Hand-written joins for cases a random draw would rarely make.
SPECIAL_DEV = {
    # look-alike scenes: same meadow, same code editor
    "s1": (["bbb2", "bbb3", "bbb4", "bbb2"], [3.0, 3.0, 3.0, 3.0], [("dissolve", 0.8), ("fade", 0.5), ("slideleft", 0.4)]),
    "s2": (["mochi0", "mochi1", "mochi3", "mochi0"], [3.0, 3.5, 3.0, 3.0], [("fade", 0.6), ("cut", 0), ("dissolve", 1.0)]),
    # a short scene between two changes
    "s3": (["ed1", "bbb5", "ed4", "M2"], [3.0, 1.6, 3.0, 3.0], [("fade", 0.3), ("slideup", 0.3), ("cut", 0)]),
    # slow dissolves
    "s4": (["ed2", "ed3", "bbb7", "ed5"], [4.0, 4.0, 4.0, 4.0], [("dissolve", 2.0), ("fade", 1.5), ("fadeblack", 2.0)]),
    # dark, low-contrast scenes
    "s5": (["ed2", "ed7", "ed3", "ed4"], [3.0, 3.0, 3.0, 3.0], [("fade", 0.5), ("slideright", 0.5), ("dissolve", 0.5)]),
}
SPECIAL_HOLD = {
    "hs1": (["wiki0", "wiki1", "wiki3", "wiki4"], [3.0, 3.0, 3.0, 3.0], [("fade", 0.6), ("smoothleft", 0.4), ("cut", 0)]),
    "hs2": (["apollo3", "apollo5", "sintel1", "apollo4"], [4.0, 4.0, 4.0, 4.0], [("fade", 2.0), ("distance", 1.5), ("fadewhite", 1.2)]),
    "hs3": (["sintel0", "apollo1", "sintel5", "R2"], [3.0, 1.6, 3.0, 3.0], [("hblur", 0.3), ("wiperight", 0.3), ("cut", 0)]),
}
FLASH = "fade=t=out:st={a}:d=0.15:color=white,fade=t=in:st={b}:d=0.15:color=white"
# name -> (file, start, duration, extra filter, ignore zones)
NEG_DEV = {
    "n1": ("mine", 10.0, 6.0, "", []),               # taps, terminal lines, a screen sliding inside the phone
    "n2": ("mine", 21.9, 5.5, "", [[1.9, 3.3]]),     # text fading in, a page scroll (ignored), buttons changing
    "n3": ("ed", 38.0, 7.0, "", []),                 # two figures moving in one long shot
    "n4": ("ed", 199.0, 8.0, "", []),                # a light and a figure rising through a dark shot
    "n5": ("bbb", 350.0, 5.5, "", []),               # camera follows a falling tree trunk
    "n6": ("mochi", 180.0, 8.0, "", []),             # typing in a code screencast
    "n7": ("ed", 199.0, 6.0, FLASH.format(a=2.85, b=3.0), []),      # the same shot with a 0.3 s white flash
    "n8": ("mochi", 250.0, 6.0, FLASH.format(a=2.85, b=3.0), []),
}
NEG_HOLD = {
    "hn1": ("ref", 0.6, 2.6, "", []),                # typing into a form
    "hn2": ("apollo", 15.0, 8.0, "", []),            # slow clouds
    "hn3": ("apollo", 330.0, 8.0, "", []),           # hand-held camera drifting over the moon
    "hn4": ("piano", 30.0, 8.0, "", []),             # game footage with small changes
    "hn5": ("wiki", 10.0, 8.0, "", []),
    "hn6": ("wiki", 170.0, 6.0, FLASH.format(a=2.85, b=3.0), []),
    "hn7": ("apollo", 330.0, 6.0, FLASH.format(a=2.85, b=3.0), []),
}
# Real edits, labelled by eye from 4 fps sheets (seconds in clip time).
REAL_DEV = {
    # Sintel 358.5-365.9: a face dissolves slowly into a night sky with a moon
    "real_sintel_a": ("sintel", 358.5, 7.4, [[4.0, 6.5]], []),
    "real_mine": ("mine", 0.0, None, [[3.84, 4.9], [8.0, 8.0], [9.3, 9.3], [21.5, 21.5], [27.6, 27.6]],
                  [[5.0, 7.2], [16.2, 17.4], [23.8, 25.2]]),
}
REAL_HOLD = {
    # Sintel 398-408: a close-up dissolves into a mountain landscape
    "real_sintel_b": ("sintel", 398.0, 10.0, [[7.0, 8.75]], []),
    "real_ref": ("ref", 0.0, None, [[3.3, 4.4], [7.05, 7.35], [9.9, 10.7], [21.0, 21.03], [28.4, 29.3]],
                 [[4.4, 7.0], [16.2, 17.4]]),
}


def source_args(key, length):
    if key in SHOTS:
        which, start, _ = SHOTS[key]
        return ["-ss", f"{start}", "-t", f"{length}", "-i", str(FILES[which])]
    return ["-f", "lavfi", "-t", f"{length}", "-i", LAVFI[key]]


def build_joined(name, keys, lengths, transitions, fps=30):
    out = OUT / f"{name}.mp4"
    args, graph, boundaries = [], [], []
    for key, length in zip(keys, lengths):
        args += source_args(key, length)
    for i in range(len(keys)):
        graph.append(f"[{i}:v]{NORM.format(fps=fps)}[c{i}]")
    cur, total = "c0", lengths[0]
    for i, (kind, dur) in enumerate(transitions, 1):
        nxt = f"x{i}"
        if dur == 0:
            graph.append(f"[{cur}][c{i}]concat=n=2:v=1:a=0,settb=AVTB[{nxt}]")
            boundaries.append([round(total, 3), round(total, 3)])
            total += lengths[i]
        else:
            graph.append(f"[{cur}][c{i}]xfade=transition={kind}:duration={dur}:offset={total - dur:.3f}[{nxt}]")
            boundaries.append([round(total - dur, 3), round(total, 3)])
            total += lengths[i] - dur
        cur = nxt
    run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(graph), "-map", f"[{cur}]",
         "-c:v", "libx264", "-crf", "20", "-g", str(fps * 2), "-an", str(out)])
    return {"path": str(out), "boundaries": boundaries, "ignore": [],
            "kinds": ["hard" if d == 0 else "soft" for _, d in transitions],
            "detail": [f"{k}:{d}" for k, d in transitions], "sources": keys}


def random_joins(prefix, count, pool, kinds, seed):
    rng = random.Random(seed)
    joins = {}
    for n in range(1, count + 1):
        keys = rng.sample(pool, rng.choice([4, 5]))
        transitions = []
        for _ in keys[1:]:
            kind = rng.choice(kinds)
            transitions.append((kind, 0 if kind == "cut" else rng.choice(DURATIONS)))
        lengths = []
        for i, key in enumerate(keys):
            usable = SHOTS[key][2] if key in SHOTS else 4.0
            before = transitions[i - 1][1] if i > 0 else 0
            after = transitions[i][1] if i < len(transitions) else 0
            need = before + after + 0.8
            # shorten transitions that do not fit the shot
            while need > usable:
                if i > 0 and transitions[i - 1][1] >= after and transitions[i - 1][1] > 0.2:
                    transitions[i - 1] = (transitions[i - 1][0], round(transitions[i - 1][1] / 2, 2))
                else:
                    transitions[i] = (transitions[i][0], round(transitions[i][1] / 2, 2))
                before = transitions[i - 1][1] if i > 0 else 0
                after = transitions[i][1] if i < len(transitions) else 0
                need = before + after + 0.8
            lengths.append(round(rng.uniform(max(need, min(2.0, usable)), usable), 1))
        joins[f"{prefix}{n}"] = (keys, lengths, transitions)
    return joins


def build_negative(name, spec):
    which, start, dur, extra, ignore = spec
    out = OUT / f"{name}.mp4"
    vf = NORM.format(fps=30) + ("," + extra if extra else "")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(FILES[which]), "-vf", vf,
         "-c:v", "libx264", "-crf", "20", "-an", str(out)])
    return {"path": str(out), "boundaries": [], "ignore": ignore, "kinds": []}


def build_real(name, spec):
    which, start, dur, boundaries, ignore = spec
    out = OUT / f"{name}.mp4"
    if dur is None:
        shutil.copyfile(FILES[which], out)
    else:
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(FILES[which]),
             "-vf", "setpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "18", "-an", str(out)])
    kinds = ["hard" if e - s < 0.1 else "soft" for s, e in boundaries]
    return {"path": str(out), "boundaries": boundaries, "ignore": ignore, "kinds": kinds, "real": True}


def main():
    if (OUT / "manifest.json").exists():
        print(f"benchmark set already built at {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"dev": {}, "holdout": {}}
    dev_joins = {**random_joins("d", 12, DEV_POOL, DEV_KINDS, seed=20261003), **SPECIAL_DEV}
    hold_joins = {**random_joins("h", 8, HOLD_POOL, HOLD_KINDS, seed=31002026), **SPECIAL_HOLD}
    for name, (keys, lengths, trans) in dev_joins.items():
        manifest["dev"][name] = build_joined(name, keys, lengths, trans, fps=60 if name == "d12" else 30)
    for name, spec in NEG_DEV.items():
        manifest["dev"][name] = build_negative(name, spec)
    for name, spec in REAL_DEV.items():
        manifest["dev"][name] = build_real(name, spec)
    for name, (keys, lengths, trans) in hold_joins.items():
        manifest["holdout"][name] = build_joined(name, keys, lengths, trans, fps=60 if name == "h8" else 30)
    for name, spec in NEG_HOLD.items():
        manifest["holdout"][name] = build_negative(name, spec)
    for name, spec in REAL_HOLD.items():
        manifest["holdout"][name] = build_real(name, spec)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(manifest['dev'])} dev and {len(manifest['holdout'])} holdout clips in {OUT}")


if __name__ == "__main__":
    main()
