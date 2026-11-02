# Checker climb: scorer

`score.py` runs the real render checker (`python -m src.loop check`) on planted-fault videos and on the
clean floor, then scores it against `answer_key.json`.

## Run

    .venv/bin/python scripts/climb/score.py --label NAME [--compare .climb-tmp/score/OTHER.json]

Takes about 2 minutes. Output: one line per case, a TOTAL line, and `.climb-tmp/score/NAME.json`.

## What is scored

- Faulted case: `~/.local/share/media-loop-work/<case>-pristine` (the untouched faulted original, with its
  `cues.json` and `INTENT.md`). The plain `<case>` dir (a fixer output, under `project/`) is run only to find
  faults shared by both sides.
- Caught: a key entry is caught when a confirmed problem has one of its kinds within 0.5 s of its time.
- False alarm: a confirmed problem on the faulted side that matches no key entry, is not shared with the
  plain side, and is not a duplicate of a caught fault.
- Shared, not scored: a confirmed problem that the plain side also confirms (same kind, within 0.5 s).
  Neither caught nor false alarm. Listed per case so they stay visible.
- Duplicate, not scored: a second confirmed problem on a fault that is already caught.
- Score per case = caught - 2 x false alarms. Dev = `dev` cases, held-out = `held_out` cases; each total
  sums its cases.
- Floor: confirmed problems on the four Rakan clips (`~/Code/rakan-edits/out`). Must be 0.
- Every report must carry a `checked` section; the scorer lists `checked_ok` per case.

## Decisions (coordinator, 2026-11-02)

- `-pristine` is the faulted side. The plain dirs are fixer outputs and are not scored, and not a clean floor.
- Dev keys: `newtest`, `newtest2`-`6`. Held-out keys: `newtest7`-`10`.
- The held-out key has 15 entries. The memory note (`agentic-video-vs-gemini.md`) says 20 faults. The
  generator diffs give 15 keyed faults plus 2 out-of-scope changes (below). The other 3 faults are not visible in
  the generator diffs, so they are not in the key.
- Out of scope (no checker kind can see them): newtest7 noise band 3000 vs 3500 Hz; newtest8 rectangle
  top 0 vs 12 px.
- Pad and bass +10 dB steps count as mix faults; the checker names them `loudness_step`. Added to the key on
  2026-11-02 before the first climb run (the baseline with the first key is in LOG.md).
- Clean floor: the four Rakan clips only. The white flashes in 2-kill-rush are intended and must not be
  confirmed.

## Sources

Every key entry has a `source` field: the generator diff between the faulted and plain files, with the
intended time from `cues.json` or from the generator's timing. Times are the faulted video's times.
