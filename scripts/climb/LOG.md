# Climb log: render checker (branch climb/checker-report-2026-11-02)

Scores: `dev` = newtest, newtest2-6 (keys in answer_key.json); `held_out` = newtest7-10;
`floor` = confirmed problems on the four Rakan clips (must be 0). Score per case = caught - 2 x false alarms.

## 2026-11-02 setup and first measurements

| # | change | dev | held_out | floor | checked_ok | verdict |
|---|--------|-----|----------|-------|------------|---------|
| 0 | main checker (9e9b0d5), first key (15 held-out, 19 dev), scorer v1 | 3 | 10 | 0 | n/a | baseline |
| 1 | Checked section added to report.md and report.json (src/loop.py) | 3 | 10 | 0 | all True | detection unchanged (24 reports compared, same candidates) |
| 2 | key: pad/bass +10 dB steps also accept `loudness_step` (mix kinds); scorer v2: a second confirmed item on a caught fault is a duplicate, not a false alarm | 9 | 12 | 0 | all True | scorer and key fix, not a checker change |

Known-good (floor, Rakan clips): 0 confirmed problems on all four clips (1-clean, 2-kill-rush, 3-ally-down, 4-replay).
Known-bad (newtest7-pristine): 3 of 3 caught, 0 false alarms, score 3.

Note on row 2: this is a scoring change, not a checker climb. Its numbers are the new baseline for the climb.
Row 2 is the baseline the checker climb is measured against (score.py --label v2).

## Key decisions (coordinator, 2026-11-02)

- Faulted side = `newtestN-pristine`. Plain `newtestN` are fixer outputs, run only to find shared faults.
- Shared faults (both sides confirm the same kind within 0.5 s) are listed per case and not scored. Some are
  checker false alarms that appear on both sides (for example newtest4: 5 shared items). Shared items are
  reported so they stay visible.
- Held-out has 15 keyed faults; memory says 20. 2 changes are out of scope (noise band, 12 px rectangle).
  The other 3 are not in the generator diffs.

## Climb experiments

None run yet. See STATE.md for the blocker list and the next experiment.
