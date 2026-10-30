# Review video

How a user reaches it: the `review_video` tool.

Code: `src/executor/mcp_server.py` (the `review_video` branch of `_dispatch`, `_loudness_summary`). It calls `AgenticVideoEngine.probe`, the `contact_sheet` and `detect_cuts` dispatch branches, and `AgenticVideoEngine.loudness_timeline`. The black scan is `black_ranges` in `src/media/visual_timing.py`, which uses the same 160x90 grey frame reader and near-black test as `blank_flashes`.

How to check it works: `pytest tests/test_review_video.py tests/test_schema_contract.py tests/test_agent_first_tools.py`.

Result fields:
- `black_ranges`: stretches of at least 0.2 s where the picture is near-black, in time order, each with `start_sec`, `end_sec`, `duration_sec`. At most 50 are listed (`black_ranges_omitted` says how many were left out).
- `black_total`: how many near-black stretches were found in all.
- `loudness.silence_gaps` (with `silence_gaps_total`) already lists silent stretches, so there is no separate silence field. It uses the `loudness_timeline` silence test (below -60 dBFS for at least 30 ms).

Near-black means the brightest 1% of a frame's 160x90 grey cells stay below 40 (the same `_BLANK_P99` that `blank_flashes` uses). A dim scene that keeps some bright cells is not listed. A fade to black is listed from the point where it gets below that level.

Gotchas:
- The whole call is meant to finish in under 45 s (Executor cuts calls at 60 s). On a 10-minute 1080p clip this branch misses that: the first call takes 53.2 s, against 30.7 s for the same steps without the black scan. Owner decision open: accept it, skip the black scan on long clips, or share one decode with the cuts scan.
- Timing, measured 2026-10-31 on the Cloudflare box with a 10-minute 1080p30 H.264 test clip, one cold run each: `review_video` 53.2 s (30.7 s for the same steps without the black scan, timed in a script, not at the base commit). The black scan alone takes 27.4 s and `detect_cuts` alone 21.3 s; a bare decode of the clip takes 23.8 s, so both scans are decode-bound. The black scan runs in a second worker thread beside the main thread, which saved about 5 s against running them one after the other. Repeat calls on the same file come from cache (`black_ranges_v1` in the media cache). The 45 s check in `tests/test_review_video.py` applies to the 60 s fixture only.
- Black-scan method, same clip, one run each: the repo's 160x90 frame reader 27.3 s; ffmpeg `blackdetect` on a 160x90 grey scale 33.6 s; ffmpeg `blackdetect` at full size 31.8 s. The repo reader is the fastest, so it is used.
- Accuracy, measured 2026-10-31 on 30 generated 640x360 clips: 14 with 1-3 planted black stretches of 0.2-2 s, 6 with a 1 s fade to black followed by a black hold, and 10 dark but not black (a dim test pattern, dim colour bars, a night-like noisy grey, a dim gradient). For the 36 planted and fade stretches, precision and recall at IoU 0.5 or more are both 1.0, and all 20 of those clips list exactly their stretches. The 10 dark clips list none; their brightest-1% levels run from 43 to 103 against the 40 threshold, and the closest clips (dim test pattern, night-like grey) sit at 43-44, so a clip only a little darker would be listed. The first run dimmed the mandelbrot clips at -0.6, which left their brightest-1% level at 3-14 (darker than video black), so they were near-black and got listed; they were redone at -0.3 (levels 91-103). Before this change the result had no black field, so recall was 0. The clips are synthetic; real dark scenes are not measured.
- No audio track: `loudness` is null and `note` says so. Any step failing fails the whole call with that step's error; nothing is returned half-filled.
- `loudness` is a summary (10 strongest outliers, 20 silence gaps). `average_rms_dbfs` is computed from all windows, so it does not depend on those limits.
