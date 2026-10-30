"""Add two real films to the soft-transition benchmark (precision on moving footage).

Writes manifest.json into $AGENTIC_VIDEO_BENCH_DIR/soft-transitions-film.
Dev: the full Big Buck Bunny file. Holdout: Sintel 0-382 s (stream copy).
Only animated changes are scored here: every hard cut is an ignore zone.

Labels were made by eye on what the detector reported at the time
(before / at / after frames), so they measure false transitions well and
recall only against those known changes. "Real" times are boundaries
(+-0.5 s). "Unclear" times are ignore zones (+-1 s). "Motion" times are not
listed: a detection there is a false positive. The hard cuts and the Sintel
"unclear" times were taken once from the detector at that commit and are
written out below, so a rebuild with a later detector gives the same ruler.
Clip paths are the source files named in build_v2.FILES; the scorer looks for
each clip in the set folder first, so a copy there takes precedence.
"""
import json

from build import BENCH_ROOT, run
from build_v2 import FILES

OUT = BENCH_ROOT / "soft-transitions-film"
SINTEL_CLIP_SEC = 382.0  # 380 s plus 2 s so the change at 378.8 s is not at the clip edge

BBB_REAL = [1.6, 20.2, 298.8, 300.7, 308.1, 337.5, 460.6, 462.3, 466.1, 488.6, 490.3]
BBB_UNCLEAR = [25.4, 29.4, 32.4, 212.7, 444.4]
# Motion inside one shot (for reference; not used by the scorer).
BBB_MOTION = [18.2, 49.2, 67.2, 78.3, 101.8, 109.9, 115.7, 143.1, 151.9, 173.5, 187.5, 201.3, 216.1, 218.1, 227.4,
              246.3, 247.7, 273.1, 283.1, 293.8, 320.8, 327.9, 331.9, 346.2, 353.1, 355.4, 359.9, 366.9, 371.1,
              385.6, 406.8, 452.9, 508.1, 526.0, 583.0, 589.4]
# detect_scenes(threshold=0.25) on the full file.
BBB_HARD = [
    11.875, 15.75, 23.042, 47.708, 56.083, 60.667, 63.792, 69.417, 71.292, 73.458, 75.542, 79.958, 89.75,
    91.417, 99.75, 103.917, 112.667, 117.958, 120.875, 123.458, 126.875, 146.375, 149.083, 156.0, 160.833,
    162.292, 163.292, 166.208, 168.0, 169.542, 176.125, 177.75, 180.5, 185.958, 189.333, 192.167, 194.667,
    195.708, 196.917, 199.833, 204.167, 205.917, 207.458, 208.875, 211.625, 213.792, 220.5, 223.792, 226.25,
    229.917, 232.458, 242.042, 243.5, 250.75, 252.0, 253.667, 255.333, 257.0, 258.667, 259.5, 261.167, 262.833,
    264.5, 266.167, 270.333, 275.958, 277.917, 279.708, 281.375, 284.708, 292.208, 295.542, 297.125, 305.208,
    312.5, 314.417, 317.208, 319.208, 322.167, 323.708, 334.333, 336.875, 339.75, 372.708, 375.375, 378.333,
    381.167, 390.792, 392.5, 394.5, 396.125, 398.958, 399.708, 404.958, 408.125, 410.417, 412.125, 416.125,
    417.333, 420.042, 420.708, 421.875, 424.333, 426.417, 427.5, 429.292, 430.792, 433.708, 435.542, 440.583,
    442.167, 443.208, 451.75, 458.292, 471.333, 472.875, 480.167, 481.667, 483.833, 486.292, 580.917, 596.417,
]

SINTEL_REAL = [4.8, 145.8, 151.2, 356.7, 366.6, 378.8]
SINTEL_REAL_SPANS = [[363.0, 365.0]]  # slow dissolve, from the 4 fps study
SINTEL_MOTION = [16.6, 24.1, 37.4, 51.3, 72.0, 169.2, 179.2, 199.9, 221.6, 245.2, 267.2, 270.4, 294.3, 299.4, 345.7]
# Transitions the detector reported that were not judged Real or Motion.
SINTEL_UNCLEAR = [
    7.05, 65.55, 109.25, 111.5, 115.75, 118.05, 119.95, 123.05, 126.25, 130.95, 140.95, 163.4, 224.05, 240.25,
    249.4, 251.65, 258.15, 290.75, 304.0, 307.95, 317.25, 325.95, 351.45,
]
# detect_scenes(threshold=0.25) on the 0-382 s clip.
SINTEL_HARD = [
    31.333, 42.417, 43.167, 44.0, 44.917, 45.833, 47.333, 48.708, 53.958, 54.958, 55.792, 57.083, 59.375,
    60.625, 62.25, 64.667, 68.208, 69.292, 74.042, 81.458, 85.833, 88.542, 101.208, 134.042, 138.125, 155.833,
    162.083, 176.417, 177.542, 180.083, 182.0, 184.375, 186.708, 189.792, 191.875, 194.625, 196.417, 197.625,
    203.125, 205.0, 211.75, 214.917, 219.458, 248.167, 250.875, 252.958, 254.667, 255.792, 259.208, 263.25,
    265.583, 271.625, 273.25, 275.625, 278.792, 283.458, 287.5, 289.333, 297.25, 302.75, 306.292, 308.917,
    311.125, 312.625, 314.958, 315.542, 319.0, 322.208, 323.708, 328.167, 330.375, 332.125, 332.708, 335.292,
    337.458, 339.333, 342.292, 349.75, 353.333, 380.542,
]

REAL_HALF = 0.5
UNCLEAR_HALF = 1.0
HARD_HALF = 0.2


def entry(path, real, spans, unclear, hard):
    boundaries = sorted([[round(t - REAL_HALF, 3), round(t + REAL_HALF, 3)] for t in real] + spans)
    ignore = [[round(t - UNCLEAR_HALF, 3), round(t + UNCLEAR_HALF, 3)] for t in unclear]
    ignore += [[round(t - HARD_HALF, 3), round(t + HARD_HALF, 3)] for t in hard]
    return {"path": str(path), "boundaries": boundaries, "ignore": sorted(ignore),
            "kinds": ["soft"] * len(boundaries), "real": True}


def main():
    if (OUT / "manifest.json").exists():
        print(f"benchmark set already built at {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / "sintel_0_382.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-t", f"{SINTEL_CLIP_SEC}", "-i", str(FILES["sintel"]), "-c", "copy", "-an",
         str(clip)])
    manifest = {
        "dev": {"film_bbb": entry(FILES["bbb"], BBB_REAL, [], BBB_UNCLEAR, BBB_HARD)},
        "holdout": {"film_sintel": entry(clip, SINTEL_REAL, SINTEL_REAL_SPANS, SINTEL_UNCLEAR, SINTEL_HARD)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built film set in {OUT}")


if __name__ == "__main__":
    main()
