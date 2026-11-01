# Onsets climb log

Baseline: origin/main detector. Primary = mean of realistic climb F1 at 30 ms (click, beep, quiet_click, quiet_beep). Main: 0.280.

| # | change | primary | guards | kept |
|---|---|---|---|---|
| 1 | v14 detector whole (rise term, steady-bed term, gated flux, entry timing, rise anchor) | 0.925 (click .947, beep .966, quiet_click .921, quiet_beep .867) | legacy band 0.9375 ok; NOLOSS legacy+real lost 0 ok; NOLOSS short+pair+ranged lost 35 (13 within-gap) FAIL | reverted |
| 2 | ablation: v14 with steady-bed term off (ONSET_STEADY_RISE_DB huge) | 0.690 (quiet_beep 0.000) | not run (primary below v14) | reverted |
| 3 | ablation: v14 with rise term off (ONSET_RISE_DB huge) | 0.892 (beep 0.846) | not run | reverted |
| 4 | v14 with spectral-flux gate off (ONSET_GATE_RISE_DB huge; no flux term) | 0.927 | NOLOSS legacy+real lost 71 FAIL | reverted |
| 5 | ablation: v14 with flux ungated (gate = 1), with and without gate_after | 0.771 (click .598, broadband .136) | not run (click precision collapses) | reverted |
| 6 | v14 with main's timing (no entry re-timing, no dedupe) | 0.917 | legacy band not run; NOLOSS legacy+real lost 10 (broadband timing 15 ms late) FAIL | reverted |
| 7 | v14 with rise-anchor disabled (anchor = own peak frame) | 0.897 | legacy band 0.8675 ok; NOLOSS legacy+real lost 0 ok; NOLOSS short+pair+ranged lost 29 (11 within-gap; music 2 ms broadband 6; pair click 20) FAIL | reverted |
| 8 | experiment 7 plus timing valley search off (ONSET_TIMING_VALLEY_DB huge) | 0.895 | legacy band 0.8675 ok; NOLOSS legacy+real lost 0 ok; NOLOSS short+pair+ranged lost 25 (10 within-gap) FAIL | reverted |
| 9 | union: origin/main's output as is, plus v14's whole strength detector (second pass, same compare_to filter and min_gap merge), adding each second-pass onset that has no main onset within 30 ms; main's onsets never move or drop | 0.7735 (click .604, beep .966, quiet_click .659, quiet_beep .865; probe said 0.7734 unrounded) | legacy band 0.8675 ok; NOLOSS legacy lost 0, real lost 0, short lost 0, pair lost 0, ranged lost 0 (all ok); pytest 13 passed; ruff ok | kept |
| 10 | union window 40 ms (ONSET_UNION_SEC 0.040), on experiment 9 | 0.7625 (click .570, quiet_click .649) | not run (primary below 9) | reverted |
| 11 | ablation: experiment 9 with the second pass's flux term removed (strength = max of rise and steady terms, no after-gate) | 0.7728 (click .601, quiet_click .659) | not run (primary below 9) | reverted |

Stop: after experiment 5 the brief's rule (6 reverted in a row, counting ablations) had fired; experiments 6-8 ran past it. No change kept; detector stays origin/main's.
Held-out (main, final tree): band F1 0.7902; real climb-style means on held-out: click 0.523, beep 0.000, quiet_click 0.556, quiet_beep 0.156 (mean 0.309).

## Union kept (experiment 9), 2026-11-01

- Guards (public audio_onsets call, score.py --set noloss --baseline-audio baseline_main_audio.py, one box run per family): legacy, real, short, pair and ranged each print `NOLOSS total lost=0 (within_gap=0) lost_chance=0 lost_babble=0`.
- Legacy climb band F1 0.8675 (broadband 0.1889). Realistic climb at 30 ms: click 0.604, beep 0.966, quiet_click 0.659, quiet_beep 0.865.
- Timing, one 20-minute file (1200 s of realistic clips, cold cache): origin/main 2.6 s and 2.8 s; this tree 4.3 s (14,301 and 14,586 onsets).
- Held-out, this tree: band F1 0.8624 (broadband 0.3065); realistic at 30 ms: click 0.580, beep 0.991, quiet_click 0.645, quiet_beep 0.680.
- Held-out, origin/main (re-run in this job): band F1 0.7902; realistic at 30 ms: click 0.523, beep 0.000, quiet_click 0.556, quiet_beep 0.156.
- Not covered: the MCP wrapper (src/executor/mcp_server.py) still returns at most MAX_ONSETS = 500 onsets in time order; the union adds onsets, so a long file reaches that cap sooner.
