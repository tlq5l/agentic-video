#!/usr/bin/env python3
"""Fetch REAL repetition dev clips from Countix TRAIN annotations for the count dev set.

Picks, for each action class below, one short and one long clip (counts 2-30) from
countix_train.csv, deterministically from --seed. Never uses a Countix val clip and never a clip
that bench_v1/fetch_real.py pins. Downloads only the annotated repetition span (480p) with yt-dlp;
a clip that fails to download is skipped. Media goes under ~/.cache/agentic-video-dev/count_dev/videos.
Writes real_reps.jsonl (id, class, count, split, source ids), which score_tools.py reads.

    python -m benchmarks.dev.count_dev.fetch_real_reps [--seed N] [--out DIR] [--no-download]

yt-dlp must be on PATH (or set YT_DLP). The annotations come from the archive at COUNTIX_URL,
cached in <out>/../real_rep/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from benchmarks.bench_v1.fetch_real import COUNTIX_ITEMS, COUNTIX_URL  # noqa: E402
from benchmarks.dev.count_dev.generate import data_root  # noqa: E402

SEED = 6021
CLASSES = [
    "hula hooping", "juggling soccer ball", "skipping rope", "bouncing ball (not juggling)",
    "jumping jacks", "push up", "squat", "battle rope training", "clapping", "situp",
    "pull ups", "playing ping pong", "swimming front crawl", "using a sledge hammer",
]
MIN_COUNT, MAX_COUNT = 2, 30
SHORT_MAX_SEC, MIN_SEC, MAX_SEC = 6.0, 1.5, 30.0
TEST_FRACTION = .3


def train_rows(cache_dir):
    archive = Path(cache_dir) / "countix.tar.gz"
    if not archive.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(COUNTIX_URL, headers={"User-Agent": "count-dev/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive.write_bytes(response.read())
    tar = tarfile.open(archive, mode="r:gz")
    member = next(name for name in tar.getnames() if name.endswith("countix_train.csv"))
    return list(csv.DictReader(io.StringIO(tar.extractfile(member).read().decode("utf-8-sig"))))


def source_span(row):
    """Absolute start/end seconds of the counted repetitions in the source video."""
    kin_start, kin_end = float(row["kinetics_start"]), float(row["kinetics_end"])
    rep_start, rep_end = float(row["repetition_start"]), float(row["repetition_end"])
    if rep_start < kin_start - .01 and rep_end <= (kin_end - kin_start) + .01:
        return kin_start + rep_start, kin_start + rep_end  # relative to the Kinetics clip
    return rep_start, rep_end


def candidates(rows, seed):
    """Per class, shuffled (short, long) candidate lists that satisfy the count and length limits."""
    pinned = {item[0] for item in COUNTIX_ITEMS}
    rng = random.Random(seed)
    out = {}
    for name in CLASSES:
        short, long_ = [], []
        for row in rows:
            if row["class"] != name or row["video_id"] in pinned:
                continue
            start, end = source_span(row)
            if not MIN_COUNT <= int(row["count"]) <= MAX_COUNT or not MIN_SEC <= end - start <= MAX_SEC:
                continue
            (short if end - start <= SHORT_MAX_SEC else long_).append(row)
        rng.shuffle(short)
        rng.shuffle(long_)
        out[name] = (short, long_)
    return out


def yt_dlp():
    override = os.environ.get("YT_DLP")
    found = Path(override).expanduser() if override else Path(shutil.which("yt-dlp") or "/missing/yt-dlp")
    if not found.is_file():
        raise RuntimeError("yt-dlp is required: put it on PATH or set YT_DLP")
    return str(found)


def download(row, destination):
    """Fetch the annotated span; return False on any failure so the caller can try another clip."""
    start, end = source_span(row)
    with tempfile.TemporaryDirectory(prefix="count-dev-ytdlp-") as stage:
        command = [yt_dlp(), "--js-runtimes", "node", "--no-warnings", "--no-playlist",
                   "--format", "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480]/b",
                   "--merge-output-format", "mp4", "--remux-video", "mp4",
                   "--output", str(Path(stage) / "clip.%(ext)s"),
                   "--download-sections", f"*{start:.3f}-{end:.3f}", "--force-keyframes-at-cuts",
                   f"https://www.youtube.com/watch?v={row['video_id']}"]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return False
        files = [p for p in Path(stage).glob("clip.*") if p.stat().st_size > 1024]
        if done.returncode or not files:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(files[0]), destination)
        return True


def slug(name):
    return re.sub(r"[^a-z]+", "_", name.split(" (")[0]).strip("_")


def build(seed, out, fetch=True):
    root = data_root(out)
    rows = train_rows(root.parent / "real_rep")
    picks = []
    for name, (short, long_) in candidates(rows, seed).items():
        for kind, pool in (("short", short), ("long", long_)):
            for row in pool:
                clip_id = f"real_{slug(name)}_{kind}_{row['video_id']}"
                path = root / "videos" / f"{clip_id}.mp4"
                if path.exists() or (fetch and download(row, path)):
                    picks.append((clip_id, name, row))
                    break
    rng = random.Random(seed)
    ids = sorted(clip_id for clip_id, _, _ in picks)
    rng.shuffle(ids)
    test = set(ids[:round(len(ids) * TEST_FRACTION)])
    return [{"id": clip_id, "video": f"{clip_id}.mp4", "class": name, "count": int(row["count"]),
             "video_id": row["video_id"], "source_start": round(source_span(row)[0], 3),
             "source_end": round(source_span(row)[1], 3), "split": "test" if clip_id in test else "train"}
            for clip_id, name, row in sorted(picks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", help="media directory (default ~/.cache/agentic-video-dev/count_dev)")
    parser.add_argument("--no-download", action="store_true", help="only use clips already downloaded")
    args = parser.parse_args()
    picks = build(args.seed, args.out, fetch=not args.no_download)
    (HERE / "real_reps.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks), encoding="utf-8")
    train = sum(p["split"] == "train" for p in picks)
    print(f"{len(picks)} real clips ({train} train, {len(picks) - train} test) under {data_root(args.out) / 'videos'}")


if __name__ == "__main__":
    main()
