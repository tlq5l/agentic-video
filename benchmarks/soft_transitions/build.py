"""Build the soft-transition benchmark set (deterministic, ffmpeg only).

Writes clips and manifest.json into $AGENTIC_VIDEO_BENCH_DIR/soft-transitions-v1
(default ~/.cache/agentic-video-bench). The "ref" and "mine" sources are two
real screen recordings given by SOFT_TRANSITIONS_REF_VIDEO and
SOFT_TRANSITIONS_MINE_VIDEO; the set is already built, so this runs only once.
Each clip joins 3-second scenes with known transitions, so the ground truth
(boundary intervals in seconds) is exact. Negative clips have motion inside one
scene and no boundary. Two full real videos carry hand or timeline labels.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_ROOT = Path(os.environ.get("AGENTIC_VIDEO_BENCH_DIR", Path.home() / ".cache" / "agentic-video-bench"))
OUT = BENCH_ROOT / "soft-transitions-v1"
REF = Path(os.environ.get("SOFT_TRANSITIONS_REF_VIDEO", "ref.mp4"))
MINE = Path(os.environ.get("SOFT_TRANSITIONS_MINE_VIDEO", "mine.mp4"))
SEG = 3.0
NORM = "fps={fps},scale=640:360,format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS"

REAL = {
    "R1": ("ref", 0.6), "R2": ("ref", 11.5), "R3": ("ref", 22.2), "R4": ("ref", 29.0), "R5": ("ref", 17.5),
    "M1": ("mine", 0.3), "M2": ("mine", 10.0), "M3": ("mine", 22.0), "M4": ("mine", 28.8), "M5": ("mine", 17.5),
}
LAVFI = {
    "L1": "mandelbrot=size=640x360:rate=30",
    "L2": "testsrc2=size=640x360:rate=30",
    "L3": "gradients=size=640x360:rate=30:speed=0.02",
    "L4": "life=size=640x360:rate=30:mold=10:ratio=0.1:death_color=#101018:life_color=#30c060",
    "L5": "smptebars=size=640x360:rate=30",
}

# name -> (fps, sources, [(transition, duration)])  duration 0 = hard cut
DEV = {
    "d1": (30, ["M1", "R2", "M3", "R4"], [("cut", 0), ("fade", 0.5), ("slideup", 0.5)]),
    "d2": (30, ["R1", "M2", "R3", "M4"], [("slideleft", 0.4), ("cut", 0), ("dissolve", 0.8)]),
    "d3": (30, ["L1", "M5", "R5", "L2"], [("zoomin", 0.6), ("fade", 1.0), ("cut", 0)]),
    "d4": (30, ["R3", "R1", "M2", "M1"], [("wipeleft", 0.5), ("slideup", 0.3), ("fade", 0.3)]),
    "d5": (30, ["M4", "L3", "R2", "M5"], [("smoothup", 0.7), ("cut", 0), ("circleopen", 0.6)]),
    "d6": (30, ["R5", "M3", "L4", "R1"], [("fadeblack", 0.8), ("slidedown", 0.5), ("cut", 0)]),
    "d7": (30, ["M2", "R4", "M1", "R3"], [("dissolve", 0.4), ("zoomin", 0.4), ("slideright", 0.6)]),
    "d8": (60, ["L5", "R2", "M4", "L1"], [("cut", 0), ("cut", 0), ("fade", 0.8)]),
}
HOLD = {
    "h1": (30, ["R2", "M1", "R5", "M3"], [("wiperight", 0.5), ("cut", 0), ("slidedown", 0.8)]),
    "h2": (30, ["M5", "R1", "L2", "R4"], [("radial", 0.6), ("hblur", 0.5), ("fadewhite", 0.7)]),
    "h3": (30, ["L3", "M2", "R3", "M4"], [("pixelize", 0.6), ("cut", 0), ("smoothleft", 0.4)]),
    "h4": (60, ["R4", "M3", "M5", "R2"], [("fade", 0.25), ("squeezev", 0.5), ("distance", 0.8)]),
    "h5": (30, ["M1", "L4", "R1", "L1"], [("cut", 0), ("vertopen", 0.5), ("fade", 1.2)]),
}
# name -> (source, start, duration, ignore zones in clip time)
NEG_DEV = {
    "n1": ("mine", 10.0, 6.0, []),          # taps, terminal lines, a screen sliding up inside the phone
    "n2": ("ref", 10.9, 4.6, []),           # same kind of activity in the reference
    "n3": ("lavfi:" + LAVFI["L3"], 0, 8.0, []),
    "n4": ("mine", 21.9, 5.5, [[1.9, 3.3]]),  # text fading in, then a page scroll (ignored), buttons changing
}
NEG_HOLD = {
    "hn1": ("ref", 0.6, 2.6, []),           # typing into a form
    "hn2": ("mine", 0.3, 3.4, []),
    "hn3": ("lavfi:" + LAVFI["L1"], 0, 8.0, []),
}
REAL_FULL = {
    "ref_full": {
        "source": "ref",
        "boundaries": [[3.3, 4.4], [7.05, 7.35], [9.9, 10.7], [21.0, 21.03], [28.4, 29.3]],
        "ignore": [[4.4, 7.0], [16.2, 17.4]],
    },
    "mine_full": {
        "source": "mine",
        "boundaries": [[3.84, 4.9], [8.0, 8.0], [9.3, 9.3], [21.5, 21.5], [27.6, 27.6]],
        "ignore": [[5.0, 7.2], [16.2, 17.4], [23.8, 25.2]],
    },
}


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("ffmpeg failed: " + " ".join(cmd) + "\n" + proc.stderr[-800:])


def source_args(key, src, fps):
    if key in REAL:
        which, start = REAL[key]
        return ["-ss", f"{start}", "-t", f"{SEG}", "-i", str(src[which])]
    return ["-f", "lavfi", "-t", f"{SEG}", "-i", LAVFI[key]]


def build_joined(name, fps, keys, transitions, src):
    out = OUT / f"{name}.mp4"
    args, graph, boundaries = [], [], []
    for key in keys:
        args += source_args(key, src, fps)
    for i in range(len(keys)):
        graph.append(f"[{i}:v]{NORM.format(fps=fps)}[c{i}]")
    cur, length = "c0", SEG
    for i, (kind, dur) in enumerate(transitions, 1):
        nxt = f"x{i}"
        if dur == 0:
            graph.append(f"[{cur}][c{i}]concat=n=2:v=1:a=0,settb=AVTB[{nxt}]")
            boundaries.append([round(length, 3), round(length, 3)])
            length += SEG
        else:
            graph.append(f"[{cur}][c{i}]xfade=transition={kind}:duration={dur}:offset={length - dur:.3f}[{nxt}]")
            boundaries.append([round(length - dur, 3), round(length, 3)])
            length += SEG - dur
        cur = nxt
    run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(graph), "-map", f"[{cur}]",
         "-c:v", "libx264", "-crf", "20", "-g", str(fps * 2), "-an", str(out)])
    return {"path": str(out), "boundaries": boundaries, "ignore": [],
            "kinds": ["hard" if d == 0 else "soft" for _, d in transitions]}


def build_negative(name, spec, src):
    source, start, dur, ignore = spec
    out = OUT / f"{name}.mp4"
    if source.startswith("lavfi:"):
        args = ["-f", "lavfi", "-t", f"{dur}", "-i", source[6:]]
    else:
        args = ["-ss", f"{start}", "-t", f"{dur}", "-i", str(src[source])]
    run(["ffmpeg", "-v", "error", "-y", *args, "-vf", NORM.format(fps=30), "-c:v", "libx264", "-crf", "20", "-an", str(out)])
    return {"path": str(out), "boundaries": [], "ignore": ignore, "kinds": []}


def main():
    if (OUT / "manifest.json").exists():
        print(f"benchmark set already built at {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    src = {"ref": OUT / "src_ref.mp4", "mine": OUT / "src_mine.mp4"}
    shutil.copyfile(REF, src["ref"])
    shutil.copyfile(MINE, src["mine"])
    manifest = {"dev": {}, "holdout": {}, "steady": []}
    for name, (fps, keys, trans) in DEV.items():
        manifest["dev"][name] = build_joined(name, fps, keys, trans, src)
    for name, spec in NEG_DEV.items():
        manifest["dev"][name] = build_negative(name, spec, src)
    for name, (fps, keys, trans) in HOLD.items():
        manifest["holdout"][name] = build_joined(name, fps, keys, trans, src)
    for name, spec in NEG_HOLD.items():
        manifest["holdout"][name] = build_negative(name, spec, src)
    for name, spec in REAL_FULL.items():
        kinds = ["hard" if e - s < 0.1 else "soft" for s, e in spec["boundaries"]]
        manifest["holdout"][name] = {"path": str(src[spec["source"]]), "boundaries": spec["boundaries"],
                                     "ignore": spec["ignore"], "kinds": kinds}
    bg = {"x": 0, "y": 0.5, "width": 0.08, "height": 0.4}
    manifest["steady"] = [
        {"path": str(src["ref"]), "start": 0.3, "end": 3.3, "region": bg, "expect": True},
        {"path": str(src["mine"]), "start": 0.3, "end": 3.3, "region": bg, "expect": True},
        {"path": str(src["ref"]), "start": 11.0, "end": 14.0, "region": {"x": 0, "y": 0, "width": 0.006, "height": 1}, "expect": False},
        {"path": str(src["mine"]), "start": 10.0, "end": 13.0, "region": {"x": 0, "y": 0, "width": 0.02, "height": 1}, "expect": False},
    ]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(manifest['dev'])} dev and {len(manifest['holdout'])} holdout clips in {OUT}")


if __name__ == "__main__":
    main()
