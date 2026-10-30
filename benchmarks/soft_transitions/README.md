# Soft-transition benchmark

Scores `detect_cuts` (hard cuts and soft transitions) on four clip sets with
exact ground truth. Each boundary is a time interval; a detection within 0.3 s
of an unmatched boundary counts as a hit, and any other detection is a false
positive unless it falls in an ignore zone.

| Set | Built by | Used for |
| --- | --- | --- |
| `soft-transitions-v1` | `build.py` | v1 guard (`v1_dev_f1`, `v1_holdout_f1`) |
| `soft-transitions-v2` | `build_v2.py` | primary dev (`dev_f1`, `v2_dev_f1`) and holdout |
| `soft-transitions-film` | `build_film.py` | film dev (Big Buck Bunny) and holdout (Sintel 0-382 s) |
| `soft-transitions-apollo` | `build_apollo.py` | 60 s windows of a documentary, cuts and transitions, with kind accuracy |

## Build the sets

Sets go to `$AGENTIC_VIDEO_BENCH_DIR` (default `~/.cache/agentic-video-bench`).
Building needs ffmpeg and the source videos:

- `SOFT_TRANSITIONS_REF_VIDEO` and `SOFT_TRANSITIONS_MINE_VIDEO`: the two screen
  recordings used by v1 and v2 (`ref`, `mine`).
- `SOFT_TRANSITIONS_SOURCES`: a folder with `big_buck_bunny_720p.mp4`,
  `elephants_dream_hd.mp4`, `sintel_512kb.mp4`, `time_of_apollo_1960_1975.webm`,
  `mochikit_intro_screencast.mp4`, `wikipedia_namespaces_screencast.webm`
  and `piano.mp4`.

```sh
cd benchmarks/soft_transitions
python build.py && python build_v2.py && python build_film.py && python build_apollo.py
```

Each builder exits early if its manifest already exists. Builds are deterministic.

## Run the scorer

Run from the repository root with the dependencies from `requirements.txt`
installed, and ffmpeg on the path:

```sh
python benchmarks/soft_transitions/run_v2.py            # climb metrics: dev splits and the v1 dev guard
python benchmarks/soft_transitions/run_v2.py --holdout  # held-out metrics
python benchmarks/soft_transitions/run.py               # v1 set alone, with steady-region checks
```

Output is `METRIC name=value` lines. The default mode does not read the
held-out clips, so use `--holdout` only for the final check of a change.
Each run uses a fresh engine cache folder, so no result carries over from an earlier run.
