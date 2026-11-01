# Onset benchmarks for `audio_onsets`

Two synthetic sets, scored against planted onset times. Nothing is downloaded.

## What each set holds

- **Legacy set** (`score.py`, `--set legacy`): 60 clips of 6 s. Sounds: click, clap, drum, pluck. Beds: silence, pink noise, music-like chord. Split: 40 climb and 20 held out, by a seeded permutation of clips (not by bed group).
- **Realistic set** (`realistic.py`, `--set real`): 156 clips over 36 bed groups. Beds: babble (speech-like), crowd, screen (hum and fan), codec (frame-wise noise floor). Each group shares one bed, so a bed sits on one side of the split only. Split by group: 24 groups climb (105 clips), 12 groups held out (51 clips). Clip families: click, beep, mixed, quiet burst, and bed-only (every third group).

Tune on the climb split only. Use held-out to check, not to tune.

## How to run

- Scores: `python -m benchmarks.dev.onsets.score --split climb --set both`, then the same with `--split heldout`. Each prints `METRIC` lines (F1, precision, recall, timing, tp/fp/fn) per setting.
- No-loss check: `python -m benchmarks.dev.onsets.score --set noloss`. It runs origin/main's `src/media/audio.py` (read with `git show`, set by `--baseline-ref`) and this tree on the same clips, and lists each planted start that main finds and this tree misses. The last line, `NOLOSS total lost=N`, passes at 0. On a machine without git, pass `--baseline-audio <file>`.
- Run these on the box with `box run`, not on the laptop.

## Baseline: origin/main 6075109, 2026-11-01

Numbers from the box run:

- Legacy climb, band setting (1000-4000 Hz, 0.36): F1 0.7781. Broadband (0.12): F1 0.1960.
- Legacy held-out, band: F1 0.7902. Broadband: F1 0.3189.
- Realistic climb, click at 30 ms: F1 0.520; at 50 ms: 0.583. Quiet click at 30 ms: 0.565; at 50 ms: 0.595.
- Realistic climb, beep (both tolerances): F1 0.000. Quiet beep at 30 ms: 0.035; at 100 ms: 0.043.
- Realistic held-out, click at 30 ms: F1 0.523; at 50 ms: 0.556. Beep: 0.000. Quiet beep at 30 ms: 0.156; at 100 ms: 0.235.
- No-loss check, origin/main against itself: `NOLOSS total lost=0 (within_gap=0) lost_chance=0 lost_babble=0`.

## Union of origin/main and the strength pass, 2026-11-01

`audio_onsets` runs origin/main's detector, then a second detector (the strength terms of the v14 branch), and adds each second-pass onset that has no main onset within 30 ms. Main's onsets keep their times.

Numbers from the box run:

- Legacy climb, band setting: F1 0.8675. Broadband (0.12): F1 0.1889.
- Legacy held-out, band: F1 0.8624. Broadband: F1 0.3065.
- Realistic climb, click at 30 ms: F1 0.604; beep at 30 ms: 0.966; quiet click at 30 ms: 0.659; quiet beep at 30 ms: 0.865.
- Realistic held-out, click at 30 ms: F1 0.580; beep at 30 ms: 0.991; quiet click at 30 ms: 0.645; quiet beep at 30 ms: 0.680.
- No-loss check, each family against origin/main: `NOLOSS total lost=0 (within_gap=0) lost_chance=0 lost_babble=0` for legacy, real, short, pair and ranged.
