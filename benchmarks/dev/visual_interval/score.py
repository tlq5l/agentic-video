"""Score the rank-1 interval of search_visual against a known inserted shot.

Each clip is a 30-90 s host with one distinctive 2-6 s shot spliced in at a known time. The shot comes
from a generated FFmpeg source (never downloaded media), and each source has one fixed text query.
The clips are split by a fixed seed into climb (24) and held-out (12). The clips and their visual
indexes are cached under --work, so a re-run only repeats the search.

Per clip, the same calls the MCP search_visual path makes: visual.build_index, then
visual.search_index(engine, query, top_k=5). The rank-1 interval is the interval that holds rank 1.

    python benchmarks/dev/visual_interval/score.py [--split climb|heldout] [--work DIR] [--limit N]

Metrics (METRIC lines):
  <split>_iou_mean          mean IoU of the rank-1 interval with the true shot (climb metric, higher)
  <split>_hit_rate          share of clips whose rank-1 interval covers at least half the shot (guard, higher)
  <split>_len_mean_sec      mean length of the rank-1 interval in seconds (diagnostic)
  <split>_rank1_in_shot     share of clips whose rank-1 hit time is inside the shot (diagnostic)
  <split>_top5_in_shot_mean mean number of top-5 hits inside the shot (diagnostic)

The held-out split is for the final comparison only: run it once for the baseline and once for the final commit.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.cache import MediaCache  # noqa: E402
from src.retrieval import visual  # noqa: E402
from src.retrieval.engine import AgenticVideoEngine  # noqa: E402

SEED = 20261031
N_CLIPS = 36
CLIMB_N = 24
SIZE = "320x180"
RATE = 10
TOP_K = 5
DEFAULT_WORK = Path("/tmp/visual-interval-scorer")

SOURCES = {
    "testsrc2": ("testsrc2=s={size}:r={rate}", "colourful test pattern with numbers"),
    "mandelbrot": ("mandelbrot=s={size}:r={rate}", "zooming colourful fractal"),
    "red_square": ("color=c=0x1f4fa0:s={size}:r={rate},drawbox=x='160+100*sin(2*t)':y=60:w=60:h=60:color=red:t=fill",
                   "a red square on a blue background"),
    "cellauto": ("cellauto=s={size}:r={rate}", "black and white cellular automaton pattern"),
    "life": ("life=s={size}:r={rate}", "black and white game of life grid"),
}
HOST = "gradients=s={size}:r={rate}:seed={seed}"


def make_specs():
    """Fixed-seed clip specs, shuffled, split into (climb, held-out)."""
    rng = random.Random(SEED)
    names = sorted(SOURCES)
    specs = []
    for index in range(N_CLIPS):
        source = names[index % len(names)]
        shot = round(rng.uniform(2.0, 6.0), 1)
        duration = rng.randint(30, 90)
        start = round(rng.uniform(4.0, duration - shot - 4.0), 1)
        specs.append({"id": f"clip{index:02d}", "source": source, "query": SOURCES[source][1],
                      "shot_sec": shot, "duration_sec": duration, "shot_start_sec": start,
                      "host_seed": rng.randint(1, 10**6)})
    rng.shuffle(specs)
    return specs[:CLIMB_N], specs[CLIMB_N:]


def make_clip(spec, path):
    """Host A, then the shot, then host B, joined losslessly with FFmpeg concat."""
    source_expr = SOURCES[spec["source"]][0].format(size=SIZE, rate=RATE)
    host_a = spec["shot_start_sec"]
    host_b = spec["duration_sec"] - host_a - spec["shot_sec"]
    host = HOST.format(size=SIZE, rate=RATE, seed=spec["host_seed"])
    host_tail = HOST.format(size=SIZE, rate=RATE, seed=spec["host_seed"] + 1)
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-t", f"{host_a:.3f}", "-f", "lavfi", "-i", host,
           "-t", f"{spec['shot_sec']:.3f}", "-f", "lavfi", "-i", source_expr,
           "-t", f"{host_b:.3f}", "-f", "lavfi", "-i", host_tail,
           "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
           "-map", "[v]", "-an", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
           "-threads", "2", str(path)]
    subprocess.run(cmd, check=True, timeout=600)
    return path


def ensure_clip(spec, work):
    path = work / "clips" / f"{spec['id']}.mp4"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        make_clip(spec, path)
    return path


def rank1_interval(result):
    return next(item for item in result["intervals"] if 1 in item["ranks"])


def score_clip(spec, result):
    shot_start = spec["shot_start_sec"]
    shot_end = shot_start + spec["shot_sec"]
    top = rank1_interval(result)
    start, end = top["start_sec"], top["end_sec"]
    overlap = max(0.0, min(end, shot_end) - max(start, shot_start))
    union = max(end, shot_end) - min(start, shot_start)
    rank1_time = next(hit["timestamp_sec"] for hit in result["hits"] if hit["rank"] == 1)
    top5_inside = sum(shot_start <= hit["timestamp_sec"] <= shot_end for hit in result["hits"])
    return {"id": spec["id"], "source": spec["source"], "shot_start_sec": shot_start,
            "shot_end_sec": shot_end, "interval_start_sec": start, "interval_end_sec": end,
            "iou": overlap / union, "hit": overlap >= 0.5 * spec["shot_sec"],
            "len_sec": end - start, "rank1_in_shot": shot_start <= rank1_time <= shot_end,
            "top5_in_shot": top5_inside}


def run_split(name, specs, work):
    rows = []
    for spec in specs:
        path = ensure_clip(spec, work)
        engine = AgenticVideoEngine(str(path), cache=MediaCache(work / "cache" / spec["id"]))
        status = visual.build_index(engine)["status"]
        if status != "complete":
            raise SystemExit(f"index for {spec['id']} is {status}, not complete")
        row = score_clip(spec, visual.search_index(engine, spec["query"], top_k=TOP_K))
        rows.append(row)
        print(f"DIAG {name} {row['id']} {row['source']:<11} iou={row['iou']:.3f} hit={int(row['hit'])} "
              f"len={row['len_sec']:.1f} rank1_in={int(row['rank1_in_shot'])} top5_in={row['top5_in_shot']}")
    work.mkdir(parents=True, exist_ok=True)
    (work / f"results-{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    count = len(rows)
    print(f"METRIC {name}_n={count}")
    print(f"METRIC {name}_iou_mean={sum(r['iou'] for r in rows) / count:.4f}")
    print(f"METRIC {name}_hit_rate={sum(r['hit'] for r in rows) / count:.4f}")
    print(f"METRIC {name}_len_mean_sec={sum(r['len_sec'] for r in rows) / count:.2f}")
    print(f"METRIC {name}_rank1_in_shot={sum(r['rank1_in_shot'] for r in rows) / count:.4f}")
    print(f"METRIC {name}_top5_in_shot_mean={sum(r['top5_in_shot'] for r in rows) / count:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["climb", "heldout"], default="climb")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N clips of the split (pilot)")
    args = parser.parse_args(argv)
    climb, heldout = make_specs()
    specs = climb if args.split == "climb" else heldout
    if args.limit is not None:
        specs = specs[:args.limit]
    run_split(args.split, specs, args.work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
