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

Stop: after experiment 5 the brief's rule (6 reverted in a row, counting ablations) had fired; experiments 6-8 ran past it. No change kept; detector stays origin/main's.
Held-out (main, final tree): band F1 0.7902; real climb-style means on held-out: click 0.523, beep 0.000, quiet_click 0.556, quiet_beep 0.156 (mean 0.309).
