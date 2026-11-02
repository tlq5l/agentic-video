# Climb state: render checker

## Sound-file climb (2026-11-02): current best sounds-v2

- Optional cues.json "sound_file" (relative to the cues.json folder, or absolute). Without it the checker output is unchanged:
  score.py gives dev 12, held_out 12, floor 0 (same as trim; COMPARE +0 +0 +0).
- Thresholds (src/media/audio.py): gain >= 0.08 and correlation >= 0.10 mean present. Spec defaults 0.35 / 0.30 were tried and
  failed on dev (5/19); the mix is peak-normalised, so present sounds read a gain near 0.1 (LOG.md, Sound-file climb).
- Bench, final (review round 2 fix): dev caught 12/19, FA 2 (score 8; the nosounds baseline has FA 6); held_out caught 8/8, FA 1 (score 6; nosounds FA 2).
  nosounds unchanged: dev 4/19 FA 6, held_out 3/8 FA 2.
- Targets not reached: dev >= 17/19 with 0 FA, held_out >= 7/8 with 0 FA (held FA 1, nosounds 2).
- How to run: `missing_bench.py trace CASE...`, then `build`, then `score --label NAME` (mix only) or `score --sounds --label NAME`
  (cues carry sound_file). `--split dev` runs only the dev cases (use it for tuning).
- Left (not changed): 3 dev misses are inside author-note sections (newtest5 13.3 and 18.7, newtest6 17.1), which the checker
  reports as informational; 13.0 text pop click 06 reads present in both mixes (before the round 2 fix, noise at corr 0.05 won the start pick; now only starts with corr >= 0.10 count; status after the fix not re-diagnosed); 14.4 deep drop shares its time
  and kind with a baseline cue_missing (scorer twin rule); 10.0 reveal impact still matches its file after removal
  (gain 0.32, corr 0.48); 19.2 glass tick not diagnosed.


Goal: the checker (`python -m src.loop check <video>`) catches more planted faults with fewer false alarms, and
its report always says what it checked.

Check command (about 3 minutes, real end-to-end runs):

    .venv/bin/python -u scripts/climb/score.py --label NAME --compare .climb-tmp/score/v2.json

Keep a change only if: dev goes up, held_out does not drop, and floor stays 0. Record each try in LOG.md.

Current best: exp 1 (score.py --label exp1).
- dev 12, held_out 12, floor 0.
- Checked section present in every report (checked_ok True on all faulted and plain runs).

Current best: exp 1 (score.py --label exp1), unchanged by exp 3 and exp 4 (both are reports).
- dev 12, held_out 12, floor 0 (exp4-flash run: same).
- Checked section lists scene cuts, same-footage cuts (picture_changes.py) and bright flashes (kind bright_flash).
  checked_ok True on all faulted and plain runs.
- Cut-count report: `.venv/bin/python -u scripts/climb/cut_report.py --label NAME` (19 measured joins in
  cut_truth.json). Latest: scene only 8/19, plus same-footage 13/19, 0 extra.

Stop rule reached (see LOG.md "Flat rounds and the stop rule"): three flat rounds (exp 2 probe, exp 3, exp 4).
The blocker fix (three cue-missing rules, all scored offline) was rejected, and the rebuild was not built.

Remaining blockers:
1. Absent one-off or pop sounds not flagged (cue_missing). Present and absent cues share the same template and band
   fields, so a rule on those fields loses score (LOG.md). Needs a new signal. Entries: newtest2 pop 4.364,
   newtest5 clap 25.333, newtest10 ding 6.667 (accepted regression: now cue_early), newtest 8.0 (checker status ok).
2. Early pop not flagged. newtest2 pop at 22.909 (-120 ms).
3. Pan phrase shift. newtest3 at 17.143 reported at 18.0 (outside 0.5 s), so a false alarm.
4. Same-footage cuts: 6 of 19 joins still missed (1-clean 4.6, 6.833, 9.0, 11.3, 18.3; 3-ally-down 11.3). Weak
   cuts, isolation ratio under 5.
5. Bright flashes are listed, not scored. Faulted files show 2 bright runs since the PR 58 round 2 fix (7 before round 1, 3 after it): newtest10 11.9 s is not in the key; newtest9 19.667 s is keyed (LOG.md, PR 58 rounds 1-2).

Open side effect: an absent one-off cue near an unrelated onset now comes out early (newtest10 ding), accepted.

Rebuild done and failed (LOG.md "Rebuild verdict"): no mix-only signal flags removed one-off sounds on held-out.
Bench: `scripts/climb/missing_bench.py` (27 one-sound-removed variants; current checker 4/19 dev, 3/8 held-out).
Done (sound-file climb, above): an optional input with each cue's sound file, matched in the mix. Floor has no cues, so cue-path changes use the faulted and plain runs as their
false-alarm check.

## Standing grant and run rules (owner 2026-11-02, "Long climbs"; away mode until 2026-11-03 20:29 UTC)

- No questions mid-run. Decide alone, log the choice and why in LOG.md, keep going.
- Stop only when: the scored check fails (keep rule above), the rebuild's third review fails, no blocker fix is
  left, or the finish line is reached.
- Decisions already made: score = caught minus 2 x false alarms; held-out set newtest7-10 is never tuned on;
  one rebuild per climb; a planted clip with one removed one-off sound must be flagged and its clean twin must pass;
  landing goes through `away-ship merge` after a fresh Sol PASS on the exact commit.
- Load gate: start no worker while the 1-minute load is over 60 or free memory under 25%; at most 2 side agents.
- Log one spend line per run in LOG.md: date, model, tokens, what it bought.

