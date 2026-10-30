"""Add fully inspected windows of a long real film to the soft-transition benchmark.

Writes clips and manifest.json into $AGENTIC_VIDEO_BENCH_DIR/soft-transitions-apollo.
Source: the Apollo documentary (28.7 min, 720x480, 29.97 fps). Twelve 60 s
windows, fixed before any detector output was looked at (start = 60 + 140 k
seconds). Every window was inspected by eye at 4 frames per second and every
scene change written down, so these clips measure recall as well as false
detections, and whether `kind` (cut or transition) is right.

Labels are film times in seconds. A change seen between the 4 fps frames at
t and t + 0.25 is the interval [t, t + 0.25]. "Unsure" stretches (a whip pan
that may hide a cut, a title fading in over the same shot, footage shot
through a window frame that comes and goes) are ignore zones. A change whose
kind could not be told at 4 fps (one blended frame) counts for detection but
not for kind accuracy.

Split: windows 0, 2, 4, ... are dev, windows 1, 3, 5, ... are holdout. The
v2 holdout set already uses Apollo shots at 15-23 s, 136-247 s and 330-338 s;
the only window that overlaps them (200-260 s) is in the holdout half.

Each clip is the window plus MARGIN seconds on each side, re-encoded at the
source size and frame rate. The margins are ignore zones.
"""
import json

from build import BENCH_ROOT, run
from build_v2 import FILES

OUT = BENCH_ROOT / "soft-transitions-apollo"
WINDOW_SEC = 60.0
MARGIN = 5.0
UNSURE_PAD = 0.3
STARTS = [60 + 140 * k for k in range(12)]

# (start, end, kind). kind: "cut", "transition", "either" (a change, kind not
# clear at 4 fps) or "unsure" (may or may not be a change: ignored).
LABELS = {
    60: [(76.75, 77.0, "cut"), (95.0, 100.25, "unsure"), (102.0, 102.75, "unsure"), (109.25, 109.5, "cut")],
    200: [(202.25, 202.5, "unsure"), (203.0, 203.25, "cut"), (204.5, 204.75, "cut"), (207.5, 207.75, "cut"),
          (211.5, 211.75, "cut"), (222.25, 222.5, "cut"), (230.5, 230.75, "cut"), (236.25, 236.5, "cut"),
          (237.0, 238.0, "unsure"), (240.0, 240.25, "cut"), (247.0, 247.75, "unsure"), (248.75, 249.0, "cut"),
          (250.25, 250.75, "unsure"), (251.25, 252.75, "unsure"), (255.0, 256.25, "unsure")],
    340: [(355.25, 356.25, "transition"), (363.5, 363.75, "cut"), (374.75, 375.0, "cut"), (383.25, 383.5, "cut"),
          (387.5, 387.75, "cut"), (391.5, 391.75, "cut"), (395.75, 396.0, "cut")],
    480: [(504.75, 505.0, "cut"), (512.75, 513.0, "cut"), (521.0, 521.25, "cut"), (523.0, 523.25, "cut"),
          (524.0, 524.25, "cut"), (525.0, 525.25, "cut"), (526.0, 526.25, "cut"), (526.75, 527.0, "cut"),
          (529.25, 530.5, "transition"), (532.0, 535.0, "unsure"), (535.25, 535.5, "either")],
    620: [(626.0, 626.25, "cut"), (655.0, 655.25, "cut"), (666.5, 666.75, "cut")],
    760: [(768.25, 768.5, "cut"), (773.5, 773.75, "cut"), (776.75, 777.0, "cut"), (783.75, 784.0, "cut"),
          (792.75, 793.0, "cut"), (801.75, 802.0, "cut"), (808.75, 809.0, "cut"), (816.75, 818.0, "transition")],
    900: [(903.25, 903.5, "cut"), (904.25, 905.0, "unsure"), (905.25, 905.5, "cut"), (909.5, 909.75, "cut"),
          (915.75, 916.0, "cut"), (917.25, 917.5, "unsure"), (919.25, 919.5, "cut"), (923.0, 923.25, "unsure"),
          (924.75, 925.0, "cut"), (929.0, 929.25, "cut"), (933.0, 933.25, "cut"), (933.75, 934.0, "unsure"),
          (934.75, 935.0, "unsure"), (937.5, 937.75, "cut"), (943.25, 943.5, "cut"),
          (953.75, 955.5, "transition")],
    1040: [(1041.25, 1042.75, "transition"), (1048.25, 1048.5, "either"), (1093.0, 1093.25, "cut")],
    1180: [(1180.75, 1181.0, "either"), (1190.75, 1192.25, "transition"), (1205.0, 1206.75, "transition"),
           (1207.25, 1207.5, "unsure"), (1211.0, 1212.5, "transition"), (1222.25, 1224.0, "unsure"),
           (1236.0, 1236.25, "cut")],
    1320: [(1325.75, 1326.25, "either"), (1338.0, 1338.25, "cut"), (1341.25, 1341.5, "cut"),
           (1342.75, 1343.0, "cut"), (1345.75, 1346.0, "cut"), (1347.0, 1347.25, "cut"), (1349.5, 1349.75, "cut"),
           (1352.5, 1352.75, "cut"), (1353.75, 1354.0, "cut"), (1355.5, 1355.75, "cut"), (1359.25, 1359.5, "cut"),
           (1361.0, 1361.25, "cut"), (1364.0, 1364.25, "cut"), (1367.5, 1367.75, "cut"), (1368.5, 1368.75, "cut"),
           (1371.0, 1371.25, "cut"), (1376.75, 1377.0, "cut"), (1379.25, 1379.5, "cut")],
    1460: [(1481.75, 1483.5, "transition")],
    1600: [(1603.0, 1604.75, "transition"), (1613.5, 1613.75, "cut"), (1619.5, 1619.75, "unsure"),
           (1624.75, 1625.25, "unsure"), (1632.5, 1633.0, "unsure"), (1634.5, 1635.25, "unsure"),
           (1657.5, 1658.5, "unsure")],
}


def entry(path, start):
    offset = start - MARGIN
    boundaries, kinds, labels = [], [], []
    ignore = [[0.0, MARGIN], [MARGIN + WINDOW_SEC, WINDOW_SEC + 2 * MARGIN]]
    for s, e, kind in sorted(LABELS[start]):
        if kind == "unsure":
            ignore.append([round(s - offset - UNSURE_PAD, 3), round(e - offset + UNSURE_PAD, 3)])
            continue
        boundaries.append([round(s - offset, 3), round(e - offset, 3)])
        kinds.append("soft" if kind == "transition" else "hard")
        labels.append(kind)
    return {"path": str(path), "boundaries": boundaries, "ignore": sorted(ignore), "kinds": kinds,
            "labels": labels, "film_offset_sec": offset, "real": True}


def main():
    if (OUT / "manifest.json").exists():
        print(f"benchmark set already built at {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"dev": {}, "holdout": {}}
    for index, start in enumerate(STARTS):
        clip = OUT / f"apollo_{start:04d}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start - MARGIN}", "-t", f"{WINDOW_SEC + 2 * MARGIN}",
             "-i", str(FILES["apollo"]), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(clip)])
        manifest["dev" if index % 2 == 0 else "holdout"][f"apollo_{start:04d}"] = entry(clip, start)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built apollo set in {OUT}")


if __name__ == "__main__":
    main()
