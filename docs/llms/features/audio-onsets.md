# Audio onsets

How a user reaches it: the `audio_onsets` tool (`band_hz`, `compare_to`, `min_strength`, `min_gap_sec`).

Code: `src/media/audio.py` (`audio_onsets`, `_detect_onsets_in_samples`, `_merge_close_onsets`), `src/retrieval/engine.py` (`AgenticVideoEngine.audio_onsets`), `src/executor/mcp_server.py` (dispatch), contract in `src/adapters/mcp_schemas.json`, `src/adapters/mcp_definitions.ts`, `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_audio_primitives.py -k onset` (the merge test drives `handle_call` on a clip made by ffmpeg). Detection accuracy: `python benchmarks/dev/onsets/score.py --split climb`.

Gotchas:
- `min_gap_sec` (default 0.035 s): an onset within the gap of its group's first onset is merged into it. The group keeps the first onset's time and the strongest strength. `merged_count` says how many were merged away. 0 keeps every raw onset.
- The default comes from the onset climb set. 0.04 s loses one recall hit on the broadband setting and 0.05 s loses recall on both settings.
- Plain clicks can double about 38-43 ms apart, which the default does not merge; `min_gap_sec` about 0.05 does.
- `sync_check` and its internal cue-analysis calls (`_cue_onset_candidates`, `_simple_dominant_band_check`) pass `min_gap_sec=0`, so their results do not change. The standalone checker (`check_video` in `src/loop.py`) calls `audio_onsets` with the default, so its onset list and strongest-onset summary can change.
- The result cache key is versioned (`audio_onsets_v9`) and holds the exact start, end, `min_strength` and `min_gap_sec`, not rounded copies; bump the version when the output changes.
