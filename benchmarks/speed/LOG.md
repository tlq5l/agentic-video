# Speed pass log

Method: baseline and candidate code are unpacked side by side and run alternately
(A,B,A,B...) with `benchmarks/speed/run.py`; numbers are CPU s (user+sys, incl. ffmpeg
children) of the tool in the cold run. The machine is shared, so only same-pair
comparisons count. Keep rule: wins 3 of 4 pairs, no QUALITY_FAIL, >=10% on the target tool.

## 1. detect_cuts: gray frames only around candidate transitions (KEPT)
Idea: the scan scaled every frame to 32x18 gray, but only the +-0.6 s around each
candidate transition is read (`_sharp_step`). Scan without that branch; then read those
stretches with a short seek decode (frame times from `showinfo` are checked against the
scan; any mismatch falls back to the old all-frames scan). Stretches are fetched 4 at a time.
Result (4 pairs, quiet machine, detect_cuts CPU s, base -> new):
clip_415s 52.8/56.5/51.2/51.4 -> 46.6/45.2/45.8/46.0 (-11%);
x_post_127s 16.6/16.3/16.5/16.6 -> 13.3/13.1/13.2/13.1 (-20%). 4/4 pairs, no QUALITY_FAIL.
Wall of the scan unchanged (clip_415s about 5 s) once the stretches run in parallel
(sequentially they added 2-3 s).

## 2. detect_cuts: ffmpeg -hwaccel videotoolbox for the scan (UNDONE)
Idea: hardware H.264 decode. Output was byte-identical (md5 of soft.raw, native.raw, scene
txt on 60 s) and CPU fell about 20% (clip_415s scan 47 -> 37 CPU s). But the hardware
decoder is the bottleneck when 8 parts run at once: scan wall 5 s -> 7.5 s on a 7 min
video (1 part: 23 s), which would push an hour-long video past the 60 s Executor limit
(software 8 parts: about 43 s; hardware: about 64 s). Using hardware for only 2 of 8 parts
keeps the wall but saves under 5% CPU. Undone.

## 3. contact_sheet: one decode thread per frame ffmpeg, 8 at a time instead of 4 (KEPT)
Idea: a one-frame extraction gets nothing from frame threads (default = all cores), so
`threads=1` cuts CPU, and 8 side by side keeps the wall time down. Sheet md5 identical.
4 pairs, contact_sheet CPU s, prev -> new: x_post_127s 7.0/6.7/7.2/6.9 -> 4.8/5.3/5.2/5.1 (-25%, 4/4);
fixture_60s 1.77/1.81/2.05/1.94 -> 1.48/1.89/1.61/1.63 (3/4). In-process (generate_video_overview, 24 samples):
x_post wall 0.90 -> 0.77 s, CPU 6.85 -> 4.8; clip_415s wall 0.61 -> 0.51 s, CPU 4.2 -> 3.0. No QUALITY_FAIL.

## 4. detect_cuts: settle only the soft changes that are not echoes of a hard cut (KEPT)
Idea: every hard cut also leaves a soft-change candidate (plateau within 0.7 s), and
`detect_soft_transitions` drops those. In exp 1 each candidate still got its own gray-frame
decode (36 of them on clip_415s, 13 CPU s = a third of the tool). Now the scan stores the
frame times and scores around each candidate (a "window", cache key softtrans_v9), the engine
asks `scan_scene_changes(..., defer_echoes=True)` to leave the ones on a hard cut of the scanned
thresholds alone, and `detect_soft_transitions` settles (decodes) only those that survive its
own hard-cut filter, writing the result back to the cache. Other callers keep the old behaviour
(everything settled in the scan). If a window cannot be matched to the scan, it falls back to the
all-frames scan as before. Checked with a script (not committed): lazy == eager == fallback, with
and without hard cuts, on fixture_60s (3 transitions, mixed cuts) and x_post_127s.
4 pairs, clip_415s detect_cuts CPU s, prev -> new: 45.5/46.1/45.7/45.9 -> 32.2/32.3/32.3/32.2 (-30%, 4/4);
x_post_127s and fixture_60s unchanged (fixture 1.49 -> 1.30). No QUALITY_FAIL.

## 5. detect_cuts: one filter thread per scan part (UNDONE, under the 10% bar)
Idea: the 8 scan parts already fill the cores, so `-filter_threads 1 -filter_complex_threads 1` per part
should only remove waiting. 4 pairs, clip_415s detect_cuts CPU s, prev -> new: 32.0/32.3/30.1/32.4 ->
29.1/26.6/29.2/29.0 (median -9%, 4/4 wins, no QUALITY_FAIL). Below the 10% keep bar, so not committed.

## 6. detect_cuts: fewer decode threads per scan part (UNDONE, trades wall for CPU)
Idea: `-threads 1` per part instead of cores/parts (2). 3 pairs on clip_415s scan: CPU 24.7 vs 28.4 s (-13%)
but wall 2.75 vs 2.38 s (+15%). Wall time matters more under the 60 s Executor limit. Undone.

## 7. detect_cuts: small videos keep the in-scan gray frames (KEPT, fixes a regression from exp 1-4)
The final check against origin/main showed fixture_60s detect_cuts slower (0.66 -> 1.3 s): it has 10 soft
changes, and a short decode for each costs more than scaling every frame of a 640x360, 60 s video. Now the
scan scales every frame to gray when width x height x seconds < 60 million (fixture 14M, x_post_127s 263M,
clip_415s 860M; the cut-off is only checked at those three points) and settles the changes right there;
larger videos keep the lazy path. Same `_settle_entry` code on both paths.
4 pairs vs the commit before: fixture_60s detect_cuts 1.32/1.31/1.32/1.30 -> 0.75/0.73/0.73/0.73 s,
CPU total 3.35 -> 2.77 s (4/4); x_post_127s unchanged. No QUALITY_FAIL.

## Pass 2: background tools (benchmarks/speed/jobs.py)
`jobs.py` runs build_visual_index (+2 search_visual), count_occurrences (text, CLIP description, purple diamond) and
count_repetitions (30 s and 120 s windows) per video, cold cache, via handle_call, polling like an agent. `AV_REPO=<checkout>`
times other code; `--save-baseline` records outputs (index embeddings too); a run compares: counts/JSON exact, search hit
times exact and scores within 1e-3, index frame times exact and every embedding cosine >= 0.9999 (float reordering only;
measured min cosine was 1.00000000). Needs the sandbox off (Apple Vision). The machine was shared with load 15-290, so
walls are noisy; pairs are A,B alternated, A = the commit before.

### 8. build_visual_index: decode 3 chunks ahead (KEPT)
The next chunk's FFmpeg (1.25 s each on clip_415s) was the critical path; embedding (0.5 s a chunk) hid behind it.
Now 3 chunks decode at once (`PREFETCH_CHUNKS`). clip_415s build wall vs base, 4 pairs: 10.2/10.4/10.5/12.3 -> 7.2/6.6/6.8/7.0 s
(4/4, CPU +5%). Index identical (cosine 1.0). fixture_60s and x_post_127s unchanged (2.3 s, 4.3 s).
The resume test now picks the failing chunk by start time (chunks run in parallel) instead of call order.

### 9. build_visual_index: start decoding while the CLIP model loads (KEPT)
Model load (1.9 s) ran before the first FFmpeg. Quiet pairs vs exp 8: 7.3-7.8 -> 5.5 s (-25%); under heavy load about 5 of 7
pairs won, the rest tied. Mechanism is certain (overlap of two serial steps).

### 10. count_occurrences: CLIP tiles of 8 frames in one batch (KEPT)
Description-only counts (no colour/shape word, no text) embedded 5 crops per frame, batch 5. CLIP costs 8 ms/crop at batch 5
and 4.5 ms at batch 40, so 8 frames (40 crops) share one call (`FrameJudge.judge_many`, same results per frame: score
diff < 1e-6, boxes equal). clip_415s occ_clip, 4 pairs: A 111/112/126/107 -> B 77/71/93/114 s wall (3/4, under load), CPU -11..-15%.
Counts and span output identical.

### Tried and undone
- BATCH_SIZE 16 -> 64 in the index build: -4% wall, under the bar.
- TORCH_THREADS 4 -> 8: no gain on a loaded machine.
- OCR from a CGImage built from raw pixels instead of PNG data: output identical on 100 frames and 18% faster in one thread,
  but 4/4 pairs slower with the 2 OCR workers (62-93 s vs 60-62 s). Vision itself is the cost; 3+ workers are slower than 2.
- 2 judge workers for the batched CLIP path: one clean pair -10%, others lost in load noise. Retry on a quiet machine.
- CLIP fast image processor: needs torchvision (not installed); would add a dependency.

### Not tried
- count_repetitions JPEG round trip, double decode when fast_fps > fps, periodicity blur computed once (these tools take 2-4 s).
- Pre-shrinking frames before cutting CLIP crops and smaller decode for colour/shape: change pixels, so counts could move.
- Caching per-frame results so a rerun with only min_gap_sec/max_spans changed skips decoding (cold benchmark won't show it).

## Not tried / measured and dropped
- Process-wide fingerprint memo: dropped on the 0.4 MB benchmark files (0.01 s per hash); kept later for big files, see
  "Frame tools" below.
- Contact-sheet timeouts that leave ffmpeg running: only matters on timeout, no speed effect.

## count_occurrences rerun cache (perf/count-rerun)
Idea: only `SpanBuilder` reads `min_gap_sec` and `max_spans`, so the per-frame judgments (one hit or none per sampled frame, plus
frames analyzed) are saved as `<key>.judgments.json` under a key without them (ALGORITHM_VERSION bumped to occurrences-v5,
fingerprint, description, text, case_sensitive, start, end, requested fps). A call that differs only in those two arguments replays
the saved hits into a new `SpanBuilder` without frames, then decodes again only the representative frame of each listed span
(up to 24, at its sample time, starting one sample early so the fps filter picks the same source frame) to cut the contact-sheet tiles.
Cold calls only gain one small JSON write.
Same output: JSON identical to the old code (origin/main b514629) with the sheet file name and path normalised, and the contact-sheet
JPEG byte-identical (md5), for reruns with other min_gap_sec / max_spans (smaller and larger max_spans than the first call):
x_post_127s description "a man speaking into a microphone" (3 calls), clip_415s description 0-200 s (3 calls), clip_415s text "claude"
30-170 s (3 calls; non-zero start). x_post_127s text "claude" has 0 spans, so it only checks the empty path (identical).
Wall, handle_call polled like an agent, loaded machine (rerun vs the same args run fresh by the old code):
x_post_127s description 0.8 s vs 31.8 s and 0.8 s vs 38.5 s; clip_415s description 0-200 s 1.0 s vs 21.5 s (and 1.1 s vs 27.7 s);
clip_415s text 30-170 s 0.5 s vs 22.2 s (and 0.5 vs 22.8 s). CPU 0.6-1.5 s vs 30-75 s.
Cold call, 4 interleaved pairs old/new, clip_415s text 30-110 s, wall s: 14.5/14.4/14.1/13.6 -> 15.7/13.8/13.7/13.9 (no change, as expected).
Test added: a rerun with other min_gap_sec and max_spans does not judge frames again and equals a fresh run (JSON and sheet bytes).

## whisper-cli flags (perf/count-rerun, asr.py unchanged)
Input: clip_415s audio as the 16 kHz mono WAV asr.py makes (414.6 s, 101 segments, 1,594 words). Current command
`-t 8 -l auto -mc 0 -sns` (flash attention on by default; large-v3-turbo-q5_0, Metal). The current output is identical run to run
(checked). 4 interleaved pairs per variant (A = current, B = variant), wall s, loaded machine (load 14-20):
- -t 4: A 10.2/14.2/16.2/11.6 -> B 11.1/15.0/12.4/15.2 (1 of 4 faster). Output identical. Dropped.
- -t 6: 10.4/18.6/16.9/12.1 -> 10.3/19.8/14.9/11.3 (per pair -1/+6/-12/-7%, median wall -4%). Output identical. Dropped, under the 10% bar.
- -bo 3: 11.4/19.6/12.5/16.7 -> 14.3/15.0/12.3/11.5 (noise). Output identical. Dropped. -bo 1: 1 of 4 faster, identical. Dropped.
  (best-of only matters when temperature fallback samples, so a gain was not expected.)
- -bs 3 / 2 / 1: 9-20% faster wall in 3-4 of 4 pairs, but the transcript changes on every pair (-bs 2: 127 segments instead of 101; -bs 3 and -bs 1: same count,
  different words or boundaries). Dropped by the same-output rule.
Result: no variant is both faster by 10% and identical, so asr.py is not changed.

## Sound tools (audio_onsets x5 bands, loudness_timeline, sync_check)
Bench: `benchmarks/speed/audio.py` (result caches bypassed, output compared with a saved baseline).
clip_415s total CPU, 4 pairs old/new: 93.2/100.8/130.5/110.7 s -> 11.5/12.1/14.8/15.3 s (4/4 wins, ~8x).
Per tool, clip_415s (old -> new, CPU s): broad 3.1->0.8, low_mid 4.5->0.8, mid 15.8->1.1, high 18.4->1.2,
air 19.9->1.2, loudness 14.3->3.2, sync_check 33.5->4.4. fixture_60s 2.42->1.88, x_post_127s 17.5->5.3.
Synthetic 1 h (clip looped): new code ~266 s CPU before the memory fix; loudness peak memory 11 GB -> 3.8 GB
after. No old-code 1 h number (old debounce is quadratic). No QUALITY_FAIL on any video.
Kept: debounce mask, leave-one-out median from one sorted copy, lru_cache on windows/frequencies/band masks,
batched onset FFTs, band-planned content-change spectra (bands share a pass up to 256 MB; one band is not split).
Dropped (no gain): transposed-matrix median, vectorized _silence_gaps, median overwrite_input.
Not tried: cache save_json indent=2 encoder (~25% of an onset call; touches src/core/cache.py), global-sort
exact median, shared STFT in sync_check, batching _measure_window.

## Sound tools on 1 hour of audio (perf/long-audio)
Goal: loudness_timeline and sync_check each under the 60 s Executor limit on 1 h, same JSON as origin/main 0d4294e.
Input: clip_415s audio looped to 3600 s (m4a; mp4 for the handle_call video runs). Result caches bypassed. Mac loaded
(load 8-18), so walls are noisy; CPU is user+sys. "wall s / CPU s".
1 h, old -> new (one run each, same session):
- loudness_timeline default: 75.8 / 80.2 -> 10.2 / 29.5 (earlier runs under heavier load: 99-113 s wall). Peak 4.2 -> 4.5 GB.
- loudness_timeline 1 s windows: 35.3 / 39.7 -> 8.4 / 23.0. With ignore ranges + bpm: 84.2 / 88.5 (old).
- sync_check, 6 cues: 66.5 / 66.2 -> 20.9 / 66.2 (earlier old runs under heavier load: 131-157 s wall).
- sync_check, 13 cues (rich): 105.0 / 104.4 -> 46.1 / 103.1.
- sync_check, 13 cues + bpm=120 (direct call; handle_call has no bpm): 161.1 / 160.6 -> 94.8 / 152.4. NOT under 60 s:
  the time is _simple_supports_time (8,150 calls, ~627k small Python `at` calls and 56k level_test calls). Left alone.
- sync_check, no cues: 5.6 -> 1.1. audio_onsets broad: 5.4 (old); bands 6-9 s (old); not re-timed here.
- handle_call sync_check on a 1 h mp4 (6 cues): old 69.2 / 68.7 with cut scan cached; new 23.3 / 68.5 cached,
  50.1 / 442 with the first (uncached) cut scan, which is about 27 s of that. Thin margin on a loaded machine.
- handle_call loudness_timeline on the 1 h mp4: 87.4 / 92.0 -> 10.6 / 30.1.
Other videos, old -> new, wall / CPU s (loudness default | sync_check 6 cues):
- x_post_127s: 0.5 / 0.6 -> 0.4 / 0.6 | 3.9 / 14.1 -> 0.7 / 1.6.
- clip_415s: 2.2 / 2.7 -> 1.1 / 2.6 | 6.1 / 40.7 -> 1.1 / 3.2.
- fixture_60s: both under 1 s before and after (new: 0.1 s | 0.4 s).
Same output: byte-identical JSON on all 4 inputs (1 h m4a, clip_415s, x_post_127s, fixture_60s) for loudness_timeline
(default, ignore ranges + bpm, 1 s windows, range), sync_check (with and without bpm, cues 6/13/none) and audio_onsets
(broad + sub/low_mid/mid/high/air + min_strength/range); also handle_call sync_check and loudness_timeline on the 1 h mp4.
Kept:
- save_json writes json.dumps(compact) in one go: json.dump and any indent use the pure-Python encoder. All readers use
  json.load(s), so layout does not matter and old files still load.
- Exact leave-rows-out column median from one np.partition window per column block plus rank counts (replaces the sorted copy).
- Threaded numpy work (8 workers; 4 for the median): band power rows, STFT flux chunks (serial boundary fix-up), onset spectra.
- Vectorized onset peak picking, span grouping and per-onset RMS; bulk dB rounding with a boundary guard that falls back to scalar.
Dropped / not needed: shared FFT across the 2048-FFT bands and batching _measure_window (the above reached the goal without
them); cross-call span dedupe (each band needs its own band energy); global-sort median (the window median is cheaper).

## Fresh check, count rerun (2026-10-20, Opus)
- Trim 5f97e7b: the judgments file is staged in the result's temp dir; no separate save helper. Same behaviour.
- Same output vs b514629, no result cache: 30 calls (10 queries x {default, min_gap 1/max_spans 3, min_gap 0.1/max_spans 40}
  on fixture_60s, flower, sample_10s, x_post_127s, clip_415s; text and description; start 0.37 at fps 2.9): JSON equal,
  30 sheets byte-identical. Edge video with a span running to the last frame (fps 5, and fps 3.3 from 0.21 s): 6 calls equal,
  6 sheets byte-identical. Tile re-decode vs full pass: 163 sample indices on 10 windows (incl. first/last frame), 0 mismatches.
- Re-timed, 2 pairs, clip_415s text 30-90 s then min_gap 1.0/max_spans 3: rerun 8.84/8.82 s (old) -> 0.34/0.34 s (new);
  first call 8.9-9.3 s both.
- In-place same-size edit with mtime restored (merged with frame-tools' memo): the rerun judges again and returns the new spans.
## Checker scans (src/media/visual_timing.py, perf/checker-speed)
Target: the visual part of `check_video` (text_cue_timing, text_overflow, blank_flashes, frame_splices, frozen_pictures,
back to back). Bench: clip_415s, 20 text cues (and 20 forced "ok" cues for text_overflow), CPU = user+sys incl. ffmpeg.
Machine loaded (load 13-45); pairs alternated base, new. Base = b514629.
Final, 4 pairs, total CPU s / wall s, base -> new: 207.5/38.8, 209.9/37.1, 203.8/37.0, 207.5/35.6 -> 46.6/14.0, 45.7/13.9, 45.5/13.9, 44.8/13.7
(-78% CPU, -62% wall, 4/4). Per function CPU s (base -> new): text_cue_timing 31 -> 31 (it now pays the one shared decode),
text_overflow 29.5 -> 10.4, blank_flashes 35.6 -> 0.6, frame_splices 39.5 -> 2.9, frozen_pictures 34 -> 0.3.
Peak memory (RSS) 360 -> 820 MB on clip_415s (shared frames 180 MB + text_overflow stretch buffers).
Same output: every function's dict equal to base on fixture_60s, x_post_127s, clip_415s (change_shares too); the five
functions equal on newtest2..10 and -pristine (18 videos x bpm None/110: 18 overflows, 12 splices, 4 flashes, 8 frozen,
240 timed cues); full `check_video` report (judge none) byte-identical on those 18 videos.
Kept (in order):
- Scans of one file share one decode: `_grid_frames` keeps the 160x90 uint8 frames of the last file (up to 256 MiB, about
  17k frames; bigger files stream as before) and later scans read them. Pairs: 184/202/187/196 -> 70/67/65/65 CPU s.
  The cache holds the last file's frames until another file is scanned or the process ends.
- text_overflow: one ffmpeg per cue instead of four. `-ss` decodes from the keyframe before each time (GOP 6 s in clip_415s),
  so four seeks decode it four times. `_grey_frames` decodes the stretch once (`-threads 1 -copyts`, showinfo for timestamps)
  and each time takes the first frame at or after it, exactly as `-ss` does; a time within 1 microsecond of a frame (not
  equal), or one the stretch lacks, falls back to `_grey_frame`. The next cue's stretch is decoded while this one is
  measured (1-3 at a time, by frame size). Checked 500 times (random, frame-boundary, +-0.4 ms, past-end, negative) on 5
  videos incl. one with start_time 1.38 s and one 24 fps: pixel-identical to `_grey_frame`, 5 fell back. Pairs: 66.6/70.7/68.2/69.6 -> 48.1/46.7/47.2/47.5.
- Scan ffmpeg with `-filter_threads 1`: same bytes, decode CPU 29.7 -> 24.7 (pairs 44.4/46.3/44.6/44.4 -> 39.6/39.5/39.3/39.8).
Dropped (same output but wall too long, or under the bar):
- `-hwaccel videotoolbox` for the scan: CPU 16.6 -> 7.7 s, md5 same, but wall 2.6 -> 20.9 s (as in exp 2); an hour would take about 3 min.
- `-thread_type slice`/`low_delay`: CPU 16.2 -> 13.7 s, wall 2.6 -> 11 s (about 100 s per hour).
- `-threads 3/4/6` with filter threads 1: CPU -4..-9%, wall +15..+80%. `format=gray` before `scale`: different bytes, 2x CPU.
Not tried: vectorising frame_splices' adjacent differences (about 1.5 s of 46), a disk-backed frame store so files over
17k frames also share one decode, grouping cues that sit in one GOP into one decode.

## Fresh check, checker scans (2026-10-20, Opus)
- Fix 60adf99: the shared decode was keyed on (path, mtime, size). An in-place same-size edit with mtime restored served
  the old frames in the same process. Key now adds dev, inode and ctime, stored as one tuple.
- Same output vs b514629: text_cue_timing (bpm None/110), text_overflow (real and forced cues), blank_flashes, frame_splices,
  frozen_pictures, change_shares on fixture_60s, flower, sample_10s, x_post_127s, clip_415s, fixture_60s again: all dicts equal.
- Re-timed, 2 pairs, clip_415s all scans, CPU/wall s: 168.7/23.1, 169.4/23.2 -> 31.9/11.0, 32.2/11.1.
  Peak RSS 369 -> 938 MB (1080p, 415 s).
## Frame tools (perf/frame-tools, off b514629)
Bench: `benchmarks/speed/frames.py` (cold cache per tool through handle_call, clip_415s, JSON equal and images >= 40 dB
PSNR against a saved baseline). Mac loaded; interleaved old/new pairs, wall s.
Kept:
- One shared contact-sheet extraction pool (`build_contact_sheets`, 4 threads) for refine_temporal_beam children,
  inspect_moments and overview_chunks. refine_temporal_beam 2.23/2.37/3.49/2.52 -> 1.63/2.02/1.85/1.94 (4/4 pairs, ~-25%).
  overview_chunks and inspect_moments are flat (within noise), CPU +10 to 40%. Honest result: they were already
  decode-bound, only the per-item waiting of refine_temporal_beam went away.
- inspect_windows: windows extracted by 4 workers (WINDOW_WORKERS). 0.84/0.84/0.71/0.74 -> 0.49/0.46/0.58/0.50 (4/4, ~-36%).
- mcp_server text block as compact JSON (`separators=(",", ":")`): track_region_changes text 48767 -> 34393 B (-30%),
  build 1.17 -> 0.64 ms; json.loads equal. Under 1 ms absolute, kept because it is smaller and simpler.
- Process-wide, lock-protected fingerprint memo in cache.py (handle_call builds a new MediaCache per call, so the old
  per-instance memo never hit). Key: path + (dev, inode, size, mtime_ns, ctime_ns), re-checked after hashing so a file
  that changes mid-hash is not memoized. 2 GB file (87 copies of clip_415s), probe_video called 4 times in one process,
  4 pairs: old 0.85-0.95 s every call; new 0.85-0.88 s on the first call, 0.000 s on calls 2-4 (4/4). First call unchanged.
  Test `..._isolated_between_cache_instances` changed to `..._shared_...` (1 open instead of 2).
Dropped:
- track_region_changes/events scan segments in parallel (cap 4): changes 3.7/5.1/5.3/5.7 -> 4.9/5.5/4.9/4.5, no gain; the
  scan is decode-bound and timeout halving is untouched.
- `-skip_loop_filter` on gray analysis decodes: scores shift (max ~0.001) and on clip_415s cut times/kinds differ, so it
  fails the identical-cuts guard. (The jsonl description of run 6 says identical on 5 videos; that is wrong for clip_415s.)
- fit_response double dump: not changed (its size check still uses default separators, which only over-estimates; unmeasured).
Final state vs saved baseline: frames.py exit 0 (output equal), run.py no QUALITY_FAIL (SCORE 75.53 cpu s, WALL 21.58 s).
Final frames.py walls: overview_chunks 1.13, inspect_moments 2.34, inspect_windows 0.59, refine_temporal_beam 1.74,
track_region_changes 5.86, track_region_events 5.26 s.
