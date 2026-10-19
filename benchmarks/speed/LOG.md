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
- Process-wide fingerprint memo: a cold call hashes the source once or twice and the hash takes
  0.01 s per call on the benchmark files (MediaCache already memoizes per instance). Nothing to gain.
- Contact-sheet timeouts that leave ffmpeg running: only matters on timeout, no speed effect.
