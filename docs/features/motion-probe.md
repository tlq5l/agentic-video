# Motion probe (motion_ tools)

How a user reaches it: the eight `motion_*` MCP tools (`motion_open`, `motion_sheet`, `motion_stats`, `motion_boil_map`, `motion_zoom`, `motion_profile`, `motion_compare`, `motion_selftest`). Each takes a local GIF or video path (`source`; `motion_compare` takes `a` and `b`).

Code: `src/media/motion_probe.py` (decode, `build_index`, `find_flags`, the warm store and `dispatch`), routed from `_dispatch` in `src/executor/mcp_server.py`. The schemas are in `src/adapters/mcp_schemas.json`, with zod and declarations in `src/adapters/mcp_definitions.ts` and `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_motion_probe_e2e.py` (one end-to-end test through `handle_call`), then `motion_selftest`, which plants known faults in a clean sprite and reports whether each detector catches them and how many false flags the clean clip gets. The fixtures are in `tests/fixtures/motion/` (about 1.6 MB in total).

What the numbers mean:
- `boil` is the flicker score on still pixels, using the README formula; lower is calmer. On the bundled fixtures the ordering is bad-256 > v0-256 > v1-256.
- `speed` counts only changes above 2 luma levels. There is no optical flow (no cv2 in the venv).
- Stutter flags need a change more than 3x the median of its neighbours. A single dropped frame is under that rule and is not flagged; swapped or dropped runs of 3 frames are.

Warm calls (measured on the fixture, 64 frames at 256 px): `motion_stats` cold about 71 ms, from disk about 1.4 ms, from memory about 0.1 ms; `motion_zoom` about 9-10 ms when warm (it writes a PNG on each call). `motion_open` starts a background build and returns `in_progress` with `retry_after_seconds`.

What depends on it: nothing else calls it. The cache lives under the media cache entry for the file fingerprint (`motion-v1`), so a changed file gets a new entry.

Gotchas:
- Change is luma-based, so motion that changes colour but not brightness can be missed.
- `motion_compare` reads local paths only.
- The in-memory store is capped at 384 MB and evicts the oldest clip first.
