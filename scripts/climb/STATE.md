# Climb state: render checker

Goal: the checker (`python -m src.loop check <video>`) catches more planted faults with fewer false alarms, and
its report always says what it checked.

Check command (about 3 minutes, real end-to-end runs):

    .venv/bin/python -u scripts/climb/score.py --label NAME --compare .climb-tmp/score/v2.json

Keep a change only if: dev goes up, held_out does not drop, and floor stays 0. Record each try in LOG.md.

Current best: the code on this branch.
- dev 9, held_out 12, floor 0 (score.py --label v2).
- Checked section present in every report (checked_ok True on all 24 faulted and plain runs).

Biggest blockers (from the v2 run, per-case misses and false alarms):
1. Late sound reported as missing. newtest6 whistle at 6.429 (+214 ms) and newtest9 clank at 8.64 (+480 ms) are
   classified cue_missing, template_status "ambiguous", evidence "no_band_beyond_companions", simple check not on
   time. Counts as one miss and one false alarm each. Code: src/media/audio.py around lines 3400-3600.
2. Absent cued sound not flagged. newtest base click at 8.0, newtest2 pop at 4.364, newtest5 clap at 25.333.
3. Early pop not flagged. newtest2 pop at 22.909 (-120 ms) missed.
4. Pan phrase shift. newtest3 at 17.143 reported at 18.0 (outside 0.5 s), so counted as false alarm.
5. Same-footage cut detection (from the Rakan dry run, not in the score yet): 3-ally-down 0 cuts (plan 4),
   4-replay 0 (plan 1), 1-clean 3 (plan 4).
6. Bright white flash frames (2-kill-rush, 5 of them) are not detected by any check, so the Checked section
   reports none. Intended by the plan, so not a floor problem, but a reader cannot see them.

Next idea (not started): blocker 1. Let the simple check accept a band attack 80-500 ms after the cue
inside the cue slot when the template is ambiguous, and report it as cue_late. Measure with score.py.

Not done: same-footage cut detection, bright-flash listing, rebuilt-from-scratch attempt, ruff (not installed).
