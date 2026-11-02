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

### Exp 1 (kept): one-off cue whose sound lands 0.1-0.5 s away is reported as late or early
- Blocker: newtest6 whistle (+214 ms) and newtest9 clank (+480 ms) came out cue_missing ("ambiguous",
  no_band_beyond_companions). Simple band check cannot see them: at the cue, the companion (a kick) explains the
  level, and the whistle sits in another band.
- Change (src/media/audio.py, sync_check): a cue with no learned signature whose excess over companions is
  under 8 dB takes the nearest broadband onset 0.1-0.5 s from the cue (outside the 80 ms tolerance, strength >= 0.2,
  and not within 0.1 s of any cue). It is reported late or early. Nearest, not strongest: the strongest onset in
  the window was a louder sound on the other side of the cue.
- Scores: dev 9 -> 12 (newtest6 1/2 with 1 FA -> 2/2; newtest unchanged), held-out 12 -> 12 (newtest9 1 -> 4,
  newtest10 3 -> 0), floor 0 -> 0. Keep rule met (dev up, held-out not down, floor 0).
- Side effect (open, next blocker): newtest10's cash-register ding is removed in the faulted file. It used to be
  cue_missing. Now a nearby onset (-267 ms) makes it cue_early, so it counts as a wrong-kind item and a false
  alarm. Held-out is flat only because of the gain on newtest9.
- Floor limit: the Rakan clips have no cues.json, so this code path never runs on the floor. The floor is
  unchanged by construction, not measured. The faulted and plain runs are the only check of new false alarms.
- Checks: ruff (uvx) clean; related pytest 88 passed, 1 skipped.

### Exp 2 (probe only, no code change): can the newtest10 ding be separated from the whistle and clank?
- Probe: broadband onsets within 0.5 s of each one-off cue, with strength and band levels.
  whistle (newtest6): onset 6.437 s strength 0.399 (kick-like, sub heavy) and 6.661 s strength 0.223 (high band
  -35 dB, air -31 dB). clank (newtest9): onset 9.125 s strength 0.354 (low-mid heavy). ding removed (newtest10):
  onset 6.400 s strength 0.284 (mid -26 dB, high -43 dB).
- Result: strength does not separate the ding's onset from the real one-offs (0.284 vs 0.223 and 0.354). Simple
  spacing rules do not either. No gate was tested. Experiment 2 is open; it needs a decision on which feature to use.

## Decisions (2026-11-02, coordinator's open questions)

- Pyright at src/loop.py line 700: pyright 0 errors on this branch (`uvx pyright src/loop.py`). The line is
  unchanged from main. Nothing to fix.
- Shared items: on the -pristine side only, shared items stay "shared, not scored". The plain side is not scored.
- PR title: renamed to "Checker: report what it checked (climb)" (exp 0 gave no detection change). Renamed again
  to add "+ better catch rate" after exp 1, which is a kept detection change.

## Coordinator decisions (message 3, 2026-11-02)

- Newtest10 ding regression accepted (option a). It meets the keep rule (dev up, held-out flat, floor 0). Revisit it
  in the rebuild attempt.
- "Rakan-style case" for same-footage cuts: the four Rakan edits (no planted same-footage cut exists in newtest).
  Truth = the joins in each edit's latest Song-round notes (`scripts/climb/cut_truth.json`), not the PLAN.md counts
  (PLAN 4/5/4/1 was the Music round). Song round joins: 1-clean 10, 2-kill-rush 5, 3-ally-down 3, 4-replay 1 = 19.
  Joins were checked against the pixels: 3-ally-down and 4-replay by the frame-change peak at each join. For
  1-clean and 2-kill-rush the peaks were found with the same measure, but 1-clean 18.300 has no peak in the top 14,
  so it is a join on paper only (not confirmed on the pixels).
- The cut-count report (`scripts/climb/cut_report.py`) is the metric for these experiments, not the detection score.
  New cut and flash output goes only into the Checked section, not into candidates, so the scorer is the
  regression check.
- Pyright: `uvx pyright src/loop.py src/media/audio.py src/media/picture_changes.py` gives 0 errors and 0 warnings.
  No line my commits touched has an error, so nothing was changed.

## Exp 3 (Checked section, same-footage cuts): spike isolation

- Problem: the scene score at the real joins is 0.145-0.224 (3-ally-down 6.833 0.145, 11.300 0.169, 15.967 0.224;
  4-replay 6.867 0.153). Motion inside a segment reaches 0.18 (3-ally-down 12.07-12.70 s). A threshold on the scene
  score cannot separate them.
- Rule (src/media/picture_changes.py, `same_footage_cut_times`): a local peak of the mean gray change on a 54x96
  grid, at least 8, and at least 6 times the median change of the 14 frames around it (the 2 next to it are skipped).
  Peaks within 0.3 s keep the strongest. Known scene cuts (within 0.1 s) are not repeated.
- Threshold table (`.climb-tmp/cuts/thresholds.py`, same grid, all four clips, no scene cuts excluded):
  4.0: 16/19 found, 6 extra. 5.0: 11/19, 3 extra. 6.0: 11/19, 0 extra. 8.0: 8/19, 0 extra.
  Kept 6.0: the only threshold with no false cuts on any clip.
- Result with the scene-score cuts (cut-count report, `.climb-tmp/cuts/cuts-thr6`): scene only 8/19 found;
  scene plus same-footage 13/19 found, 0 extra. Misses: 1-clean 4.600, 6.833, 9.000, 11.300, 18.300; 3-ally-down 11.300.
  The misses score 3.4-4.9 on the isolation ratio, under the 6.0 line.
- Scorer (`exp3-same-footage`): dev 12, held-out 12, floor 0, all checked_ok True. Unchanged from exp 1, as intended.
- Not a detection gain: this is a report. The score rule is flat (dev +0, held-out +0).

## Exp 4 (Checked section, bright flashes): listing only

- Rule (`bright_flash_runs`): 1 to 3 frames whose mean luma is at least 60 above the median of the frames 3 to 9 away
  on each side. Listed as kind `bright_flash`; "intended" when within 0.1 s of any cut (scene or same-footage).
- 2-kill-rush: 5 of 5 flashes listed, all intended (each within 0.04 s of its cut). 1-clean, 3-ally-down and
  4-replay: none.
- Faulted files (not part of the cut report): bright runs at newtest 12.5 s, newtest10 11.9 and 22.2 s,
  newtest6 21.3 s, newtest8 27.4 s, newtest9 19.67 s (the planted endcard insert, already a candidate) and 23.07 s.
  Only 19.67 is in the answer key. The others are listed for the owner to read, not scored.
- Scorer (`exp4-flash`): dev 12, held-out 12, floor 0, all checked_ok True. Compare vs exp3: +0, +0, +0.
- Not a detection gain: a report only.

## Flat rounds and the stop rule (2026-11-02)

- Flat rounds: exp 2 (probe, no change), exp 3 (same-footage report), exp 4 (bright-flash report). Scores unchanged
  in each (dev 12, held-out 12, floor 0). That is 3 flat rounds, so the stop rule applies: fix the biggest blocker,
  then one rebuilt attempt with 3 tries.
- Biggest blocker (scorer misses, exp4): 6 misses in dev and held-out. Three are cue_missing entries (absent
  sounds) that the checker reports as cue_ambiguous, a review item that does not count as caught: newtest2 text pop
  4.364, newtest5 clap 25.333, and newtest10 ding 6.667 (now cue_early, the accepted regression). A fourth cue_missing
  entry, newtest 8.0, is covered below. One is a stereo kind (newtest3 phrase pan 17.143), one is early/late
  (newtest2 DOORS OPEN 22.909).
- newtest 8.0 text pop click 04 (cue_missing in the key): the checker's sync record for that cue is status ok with
  a matched onset at 8.000 s. This is not a classification miss, so the key entry or the faulted audio is the thing
  to check next. Not changed here.
- Why the cue_missing class is stuck: the sound's own band is "at the group's usual level" at the cue, and
  the template match is below its null floor for present sounds too (present kicks in newtest10 at 11.111, 20.0,
  24.444 have the same fields as the absent ones). Present and absent cues share the same fields.
- Blocker fix tries, scored offline on the logged fields (`cue_ambiguous` cues, truth from the answer key):
  A. template missing and ambiguous -> cue_missing: caught 2, false 8 (net about -14).
  B. as A and on-cue score under 0.01: caught 1, false 5 (net about -9).
  C. as A and the simple band level within 1 dB of its reference: caught 0, false 4 (net about -8).
  All three rejected. None of the existing per-cue fields separates an absent sound from a present one.
- Rebuild: not built. A from-scratch rebuild needs a new signal (for example, a leave-one-out comparison of the
  cue's own band after subtracting its companions' contribution). That is not designed or measured yet, so the
  rebuild attempt did not run. Stopping here, as the stop rule says when the blocker fix fails.
- Scored state at stop: dev 12, held-out 12, floor 0 (exp 1 kept; exp 3 and exp 4 are reports with no score change).

## Same-footage cut detection, status

- Gain: 8 to 13 of 19 Song-round joins now listed (plus the scene-score cuts), 0 false cuts on the four clips.
- Still missed: 6 joins (1-clean 4.6, 6.833, 9.0, 11.3, 18.3; 3-ally-down 11.3). These are weak cuts (ratio under 5).

## PR 58 review round 1 fixes (2026-11-02)

- Finding 1 (same-footage times on a variable-frame-rate clip): new in PR (picture_changes.py is new). Fixed: frames now carry their showinfo pts_time (passthrough, as scene.py reads them) and a cut is the time of the frame after the change. Repro (30 fps then 10 fps, cut at 1.000 s, no known cuts): before 1.72 s, after 1.0 s; with the 1.0 s scene cut known, before 1.72 s extra, after none.
- Finding 2 (bright flash on a sustained scene): new in PR. Fixed: a frame's lift is measured against the brighter of its two side medians, so only a brief spike above both sides counts. Repro: 40 to 220 to 40 at 30 fps, before 1.0-1.1 and 1.9-2.0 s, after none; a 3-frame blip is still listed at 1.0-1.1 s.
- Finding 3 (one-off fallback ignores search_window_ms): new in PR (origin/main has no one-off fallback and reports the newtest6 whistle as missing). Fixed: _one_off_displaced takes the search window, reach = min(0.5 s slot, window). Repro (newtest6-pristine, search_window_ms=80): before "late" at +232.333 ms, after "missing" (same as main).
- Finding 4 (text-timing failure not in errored_checks): new in PR. Fixed: the key "error" counts too. Repro: visual error gives errored_checks ['error'] and "Errored checks: error."; a *_error key still works.
- Finding 5 (score.py reports zeros and exits 0 with media missing): new in PR (score.py is new). Fixed: score.py checks every case input and the four cut_truth floor clips before running and exits 1 listing what is missing; it also exits 1 on a failed checker run or a report without Checked fields. Repro: HOME with no media, before exit 0 with zero totals, after exit 1 with 64 missing inputs listed and no scores written.
- Finding 6 (STATE.md count): new in PR. Re-counted after fixes 1 and 2: faulted files show 3 bright runs (was 7): newtest10 11.9 s and newtest6 21.333 s unkeyed, newtest9 19.667 s keyed. The six-unkeyed count was the pre-fix number; STATE.md now gives the measured one.
- Checks: ruff passes; pytest -k "loop or picture or audio or check" 89 passed, 1 skipped; score dev 12, held_out 12, floor 0 (compare to trim: +0, +0, +0); cut report 13/19, extra 0 (unchanged). The Rakan clips are constant 30 fps, so fix 1 does not move the cut report.
- Decision: same_footage_cut_times and bright_flash_runs keep their fps argument (guard and last-frame length), so loop.py's callers are unchanged.
- Decision: the one-off fallback keeps its 0.5 s cap, so a search window above 500 ms does not widen it; defaults are unchanged.
- Decision: the no-audio branch in check_video sets empty balance and clipping on purpose (no audio), not an error, so it is not counted.
- Spend: claude-haiku-5-5 (review fix worker, one run); tokens not reported by the harness. Bought: the six PR 58 round 1 fixes.

## PR 58 review round 2 fixes (2026-11-02)

- bright_flash_runs: a lifted run is measured over its whole bright stretch (frames at or above the side level
  plus half the lift); stretches over 3 frames are scenes. Synthetic 40/220/40 clips: 1-3 frames listed, 4, 9, 10,
  11 and 30 frames not listed.
- score.py: a checker run that fails with no output now carries "exit code N, no output", so it always counts as
  failed.
- Checks: ruff pass, pytest 89 passed 1 skipped, score dev 12 held_out 12 floor 0 (+0), cuts 13/19 extra 0.

## PR 58 review round 3 fixes (2026-11-02)

- score.py deletes the old report.json before each checker run, so a reused label never scores a stale report
  (checked: a failing interpreter now gives "exit code 1, no output").
- bright_flash_runs reports the whole bright stretch (first..last), so flashes at clip edges keep their frames
  (7 dark, 3 bright, 30 dark at 30 FPS: 0.233-0.333 s).
- STATE.md bright-run count updated to 2 (1 unkeyed).
- Checks: ruff pass, pytest 89 passed 1 skipped, score dev 12 held_out 12 floor 0 (+0).
- Review count: 3 failed Sol reviews (6, 2, 3 findings). Stopped here; merge waits for the owner.

## PR 58 review round 4 fixes (2026-11-02)

- Sol review 4 (owner OK quoted: "let claude code not claude desktop sonnet 5.5 to solve these blcokers and utilize imac pro and boxes") found 3 P2s on bda85ae; all fixed:
  1. score.py: items the plain render also confirms are removed before matching, so a shared item is never a catch.
  2. loop.py: `Peak: -inf dBFS` (silent audio) is read; true_peak_dbtp is None and the Checked line says "silent".
  3. picture_changes.py: the scan resumes after the expanded bright stretch, so one flash is one record.
- Checks (on a Cloudflare box): silent AAC clip gives integrated -70.0, peak None; synthetic 220/80/220 flash gives one run; score.py dev 12, held_out 12, floor 0 (same as before).
## Missing-sound bench (2026-11-02)

- Scored check for the next climb step: `scripts/climb/missing_bench.py` (subcommands `trace`, `build`, `score`; `pick` writes the removals and a hidden `_render` runs one render per process). Removals are in `scripts/climb/missing_bench.json`: 19 dev variants and 8 held-out variants over the 10 cases.
- Baseline (checker as committed on this branch, label `baseline`): dev caught 4, strict 3, FA 6, score -8; held-out caught 3, strict 2, FA 2, score -1. Per-variant results: `.climb-tmp/missing/score/baseline.json`.
- Caught kinds: `cue_missing`, `cue_early`, `cue_late` (the cue kinds `src/loop.py` emits). Strict: `cue_missing` only. `cue_ambiguous` is left out (review item, not a verdict). `cue_mismatch` and `cue_weak` do not exist in the checker, so they are not in the set.
- FA: any confirmed candidate in the variant with no same-kind twin within 0.5 s in the baseline. Literal, so a removal that makes the checker flip another real fault counts as FA. Seen: newtest2 (cue_missing at 4.36 and 22.91) and newtest5 (uncued_sound at 11.01). These are cross effects of the removal, reported as they are.
- Placement times are seconds in all ten generators (none takes samples). Time parameter: `start_sec` (newtest, newtest7), `time_sec` (newtest2-4), `t0` (newtest5, newtest10), `start` (newtest6, newtest8), `t` (newtest9).
- Sound identity: the sound makers named on the placement line (a local holding a sound-maker function resolves to that function's name); if none, the caller's sound-name strings (loop variables such as `name = "stamp"`). Logged per call in `.climb-tmp/missing/<case>/trace.json`.
- Pick rules: one cue within 0.02 s whose label shares a word (4-letter prefix) with the sound; no skip word in the label or in the call's sound names; no other call at that time for the same sound; no answer-key fault within 0.5 s; one removal per sound type per case; removals at least 1 s apart, spread over the case.
- Skip words: the brief's list, plus `ostinato`, `arpeggio`, `tom`, `drum`, `beat`, `loop`, `thump`, `sub` (patterns and drums; when unsure, skip). Matching is substring, as the brief words it.
- Shortfalls (wanted 4 dev / 2 held-out): newtest found 3, newtest2 2, newtest3 2. The rest met their count. No removal was dropped as a no-op.
- Sanity: the wrapped no-skip render is byte-identical to the unwrapped generator for all 10 cases, and every variant render differs from its baseline.
- Muxing: pristine `out.mp4` video with the variant wav (AAC 192 kbps, 48 kHz stereo); `cues.json` and `INTENT.md` copied beside each video. Pristine folders were only read.
- Checker runs go through `score.run_check` and `score.TOL_SEC` (imported, not copied). 37 runs, 4 at a time.
- Not checked: ruff (not installed in `.venv`); `py_compile` passes.

## Missing-sound signals (2026-11-02)

- Gain pin (bench): applied to the written int16 audio, not the float mix, because the ten generators do not expose the mix the same way (some keep it local, some write float32 buffers). Rounding costs at most about 1 LSB; the 2 LSB limit was not widened.
- Span: start = round(t x 48000), length = the shortest ndarray argument of the placement call (the bus is always the longest). Recorded per call in each trace.json, with gain and residual per variant.
- Pre-check before the rebuild: newtest 10.000 (reveal impact) had gain 0.8676 (about 1/1.153) and 4078 LSB raw residual outside the span, 1 LSB after the pin. The other 26 variants: gain 1.000, residual 0-1 LSB. Baseline bytes unchanged (no-skip render check passed). Trace times match missing_bench.json to the 4-decimal rounding.
- baseline2 score: dev caught 4/19, FA 6 (score -8); held caught 3/8, FA 2 (score -1). Same as baseline on all 27 rows. The pin moved no verdict.

### Missing-sound signal study (scripts/climb/signal_study.py, offline, no checker code changed)

- Labels: 27 removed cues (19 dev, 8 held) are ABSENT; 90 scored one-offs in the baseline files (58 dev, 32 held; not key faults, not removed) are PRESENT. Four answer-key cue_missing entries scored on the pristine files as extras. Output: .climb-tmp/missing/signals.json.
- Results (dev AUC / held AUC / NaN of 117 / dev caught at 0 dev FA / held caught / held FA / key flagged):
  - S1 leave-one-out template: 0.83 / 0.17 / 61 / 0 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
  - S2 bar-phase control: 0.61 / 0.69 / 44 / 1 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
  - S3 occurrence outlier: 0.87 / 0.82 / 90 / 0 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
  - S4 matched onset: 0.94 / 0.67 / 62 / 4 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
  - S1+S4 pair: 0.94 / 0.33 / 62 / 4 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
  - S2+S4 pair: 1.00 / 0.67 / 96 / 1 of 19 / 0 of 8 / 0 of 32 / 0 of 4.
- Key entries (pristine): newtest 8.0 text pop click 04 reads PRESENT (S1 1.33, S4 0.99), so that key entry is suspect. The other three read as absent or NaN, but no threshold flags them.
- Paired check (removed cue, baseline vs variant, median): S1 0.23 to 0.00, S2 0.98 to 0.06, S3 -0.68 to -0.98, S4 -0.41 to -0.74. S1 is not near 1 on baseline pairs.
- Cause (S1, S2 NaN): the one-off groups share their onset instants with drum or kick groups in every case (for example newtest "cut whoosh" lands on each "downbeat kick"). Their overlap template is not identifiable, and leave-one-out makes it worse. Most groups have 1-2 occurrences. Kept as specified; not a fixed result.
- S2 control: no bpm reaches the checker run, so the bar is 4 x the median spacing of the most frequent cue group. Labels carry serial numbers, so counting labels gave 1 each and was not used. In newtest the bar lands on the same whoosh instants, which leaves no uncued bar (NaN).
- Alignment: decoded video audio matches its WAV at lag 0 in all 41 files (cross-correlation), so no shift was applied. Pristine files were aligned to the base WAV of the same case.
- Thresholds: a single signal flags below the lowest dev negative (0 dev FA by construction). A pair flags when both are below their thresholds; thresholds are searched jointly on dev for zero dev FA, most dev catches, ties to the larger threshold sum. Pairs are ranked by dev catches, then dev AUC (AUC on the min of pooled z-scores).
- Conclusion: no signal or pair catches a held-out variant at zero dev FA (0 of 8 in every row). The checker's current 4/19 dev and 3/8 held-out baseline is not improved by any of these four scores as specified.
