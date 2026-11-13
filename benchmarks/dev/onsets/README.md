# Onset benchmarks for `audio_onsets`

Two synthetic sets, scored against planted onset times. Nothing is downloaded.

## What each set holds

- **Legacy set** (`score.py`, `--set legacy`): 60 clips of 6 s. Sounds: click, clap, drum, pluck. Beds: silence, pink noise, music-like chord. Split: 40 climb and 20 held out, by a seeded permutation of clips (not by bed group).
- **Realistic set** (`realistic.py`, `--set real`): 156 clips over 36 bed groups. Beds: babble (speech-like), crowd, screen (hum and fan), codec (frame-wise noise floor). Each group shares one bed, so a bed sits on one side of the split only. Split by group: 24 groups climb (105 clips), 12 groups held out (51 clips). Clip families: click, beep, mixed, quiet burst, and bed-only (every third group).

Tune on the climb split only. Use held-out to check, not to tune.

## How to run

- Scores: `python -m benchmarks.dev.onsets.score --split climb --set both`, then the same with `--split heldout`. Each prints `METRIC` lines (F1, precision, recall, timing, tp/fp/fn) per setting. The headline primary is `onsets.real.<split>.primary_f1@30ms`, the mean of the four unrounded realistic F1s at 30 ms.
- No-loss check: `python -m benchmarks.dev.onsets.score --set noloss`. It runs origin/main's `src/media/audio.py` (read with `git show`, set by `--baseline-ref`) and this tree on the same clips, and lists each planted start that main finds and this tree misses. The last line, `NOLOSS total lost=N`, passes at 0. On a machine without git, pass `--baseline-audio <file>`.
- Run these on the box with `box run`, not on the laptop. The box copy of the project has no `.git`, so the default `git show` baseline fails there (exit 128). On the box, put main's file in the project folder and pass it: `--baseline-audio benchmarks/dev/onsets/baseline_main_audio.py`. Make that copy with `git show origin/main:src/media/audio.py > benchmarks/dev/onsets/baseline_main_audio.py` on the laptop, and delete it before committing.

## Baseline: origin/main 6075109, 2026-11-01

Numbers from the box run:

- Legacy climb, band setting (1000-4000 Hz, 0.36): F1 0.7781. Broadband (0.12): F1 0.1960.
- Legacy held-out, band: F1 0.7902. Broadband: F1 0.3189.
- Realistic climb, click at 30 ms: F1 0.520; at 50 ms: 0.583. Quiet click at 30 ms: 0.565; at 50 ms: 0.595.
- Realistic climb, beep (both tolerances): F1 0.000. Quiet beep at 30 ms: 0.035; at 100 ms: 0.043.
- Realistic held-out, click at 30 ms: F1 0.523; at 50 ms: 0.556. Beep: 0.000. Quiet beep at 30 ms: 0.156; at 100 ms: 0.235.
- No-loss check, origin/main against itself: `NOLOSS total lost=0 (within_gap=0) lost_chance=0 lost_babble=0`.

## Union of origin/main and the strength pass, 2026-11-01

`audio_onsets` runs origin/main's detector, then a second detector (the strength terms of the v14 branch). It adds each second-pass onset that is more than 30 ms from every main onset and at least `min_gap_sec` (default 0.035 s) from every onset already kept, so the gap contract holds for the whole list. Main's onsets keep their times. The MCP reply caps the list at 500 and lists main-pass onsets first.

Numbers from the box run (after the review fixes; the union before them had no gap check):

- Legacy climb, band setting: F1 0.8696. Broadband (0.12): F1 0.1930.
- Legacy held-out, band: F1 0.8624. Broadband: F1 0.3121.
- Realistic climb, click at 30 ms: F1 0.597; beep at 30 ms: 0.966; quiet click at 30 ms: 0.659; quiet beep at 30 ms: 0.865. Primary (mean of those four unrounded F1s): 0.7718.
- Realistic held-out, click at 30 ms: F1 0.577; beep at 30 ms: 0.991; quiet click at 30 ms: 0.641; quiet beep at 30 ms: 0.680. Primary: 0.7224.
- No-loss check, each family against origin/main: `NOLOSS total lost=0 (within_gap=0) lost_chance=0 lost_babble=0` for legacy, real, short, pair and ranged.
- Gap contract: every adjacent pair of onsets is at least `min_gap_sec` apart (tested at 0, 0.035, 0.1 and 1.0 s on the review's repro clip and 4 random beep/click clips); origin/main's onsets are all kept.
