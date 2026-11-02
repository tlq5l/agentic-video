# Climb state: render checker

Goal: the checker (`python -m src.loop check <video>`) catches more planted faults with fewer false alarms, and
its report always says what it checked.

Check command (about 3 minutes, real end-to-end runs):

    .venv/bin/python -u scripts/climb/score.py --label NAME --compare .climb-tmp/score/v2.json

Keep a change only if: dev goes up, held_out does not drop, and floor stays 0. Record each try in LOG.md.

Current best: exp 1 (score.py --label exp1).
- dev 12, held_out 12, floor 0.
- Checked section present in every report (checked_ok True on all faulted and plain runs).

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

Done in exp 1 (kept): blocker 1, one-off late sounds reported as late (see LOG.md).

Open side effect: an absent one-off cue near an unrelated onset now comes out early/late (newtest10 ding).

Next: bright-flash listing in the Checked section (informational only); same-footage cut detection; one
rebuilt-from-scratch attempt for the weakest kind. Floor has no cues, so new cue-path changes need the faulted
and plain runs as their false-alarm check.
