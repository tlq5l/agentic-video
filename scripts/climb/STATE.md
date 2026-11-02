# Climb state: render checker

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

Next: the one rebuild (owed under the long-climb rule): a leave-one-out band signal for blocker 1, on branch
climb/missing-sound-2026-11-02. Floor has no cues, so cue-path changes use the faulted and plain runs as their
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

