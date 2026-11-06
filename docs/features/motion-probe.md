# Motion probe (motion_ tools)

How a user reaches it: the eight `motion_*` MCP tools (`motion_open`, `motion_sheet`, `motion_stats`, `motion_boil_map`, `motion_zoom`, `motion_profile`, `motion_compare`, `motion_selftest`). Each takes a local GIF or video path or a video URL (`source`; `motion_compare` also takes `compare_to` for the second clip).

Code: `src/media/motion_probe.py` (decode, `build_index`, `find_flags`, the warm store and `dispatch`), routed from `_dispatch` in `src/executor/mcp_server.py`. The schemas are in `src/adapters/mcp_schemas.json`, with zod and declarations in `src/adapters/mcp_definitions.ts` and `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_motion_probe_e2e.py` (six end-to-end tests: five through `handle_call`, one that builds the wheel, installs it in a fresh venv and runs `motion_selftest` with default arguments there), then `motion_selftest`, which plants known faults in a clean sprite and reports whether each detector catches them and how many false flags the clean clip gets. The clean sprite ships inside the package at `src/media/clean-sprite-256.gif`; the other test GIFs are in `tests/fixtures/motion/` (about 1.6 MB in total).

What the numbers mean:
- `boil` is the flicker score on still pixels, using the README formula; lower is calmer. On the bundled fixtures the ordering is bad-256 > v0-256 > v1-256.
- `speed` counts only changes above 2 luma levels. There is no optical flow (no cv2 in the venv).
- Stutter flags need a change more than 3x the median of its neighbours. A single dropped frame is under that rule and is not flagged; swapped or dropped runs of 3 frames are.

Warm calls (measured on the fixture, 64 frames at 256 px): `motion_stats` cold about 71 ms, from disk about 1.4 ms, from memory about 0.1 ms; `motion_zoom` about 9-10 ms when warm (a repeat call reuses the saved PNG and writes nothing). `motion_open` starts a background build and returns `in_progress` with `retry_after_seconds`.

What depends on it: nothing else calls it. The cache lives under the media cache entry for the file fingerprint (`motion-v1`), so a changed file gets a new entry.

Gotchas:
- Change is luma-based, so motion that changes colour but not brightness can be missed.
- A URL source downloads first; the call answers `in_progress` with `retry_after_seconds` until the file is ready.
- The in-memory store never holds more than 384 MB. It evicts the oldest clip first; a clip bigger than the cap on its own is dropped from memory after the call and read from disk on the next one.
- A box whose top-left corner is outside the frame is an error for motion_zoom, motion_stats and motion_compare. A box that runs past the right or bottom edge is cut to the frame.
