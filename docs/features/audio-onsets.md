# Audio onsets

How a user reaches it: the `audio_onsets` tool (`band_hz`, `compare_to`, `min_strength`, `min_gap_sec`).

Code: `src/media/audio.py` (`audio_onsets`, `_detect_onsets_in_samples`, `_merge_close_onsets`), `src/retrieval/engine.py` (`AgenticVideoEngine.audio_onsets`), `src/executor/mcp_server.py` (dispatch), contract in `src/adapters/mcp_schemas.json`, `src/adapters/mcp_definitions.ts`, `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_audio_primitives.py -k onset tests/test_onsets_beeps_over_speech_e2e.py` (both drive `handle_call` on generated clips). Detection accuracy: `python benchmarks/dev/onsets/score.py --split climb` (the 60 original clips and 120 real-world-like ones from `benchmarks/dev/onsets/realistic.py`: speech-like babble, crowd, screen-recording and codec-floor beds, square clicks, sine beeps and decoys).

What `strength` means (0..1, `min_strength` is a fixed cut on it): the larger of two terms, both measured in the analysis band (`band_hz`, or the whole spectrum).
- Spectral flux times a gate. The flux is how much the normalized spectrum changes from one frame to the next (it ignores level). The gate is 0 unless the band's level rises by more than 1 dB over one analysis window (fully open at 4 dB) to an audible level (fully open at -35 dBFS mean-square, shut below -50 dBFS). So a quiet bed's shape jitter, fades and the end of a tone do not fire, and sounds quieter than about -35 dBFS in the band score lower than before.
- A level-rise term: grows with the band's rise over one window (0 at 4 dB, 1 at 16 dB) times the level reached (0 at -30 dBFS, 1 at -15 dBFS). A loud tone over a busy bed, which barely changes the normalized spectrum of a narrow band, scores near 1.
The constants sit at the top of `src/media/audio.py` (`ONSET_*`), fitted on the climb splits only.

Gotchas:
- `min_gap_sec` (default 0.035 s): an onset within the gap of its group's first onset is merged into it. The group keeps the first onset's time and the strongest strength. `merged_count` says how many were merged away. 0 keeps every raw onset.
- The default comes from the original 60-clip climb set, measured with the v9 detector: 0.04 s lost one recall hit on the broadband setting and 0.05 s lost recall on both settings.
- Onsets are timed where the start entered the analysis window (the first frame 3 dB over the level just before), so a decaying click no longer shows a second onset about 40 ms later. Two real attacks closer than the window (a flam) still show as two; `min_gap_sec` about 0.05 merges them.
- `sync_check` and its internal cue-analysis calls (`_cue_onset_candidates`, `_simple_dominant_band_check`) pass `min_gap_sec=0`, so their results do not change. The standalone checker (`check_video` in `src/loop.py`) calls `audio_onsets` with the default, so its onset list and strongest-onset summary can change.
- The result cache key is versioned (`audio_onsets_v10`) and holds the exact start, end, `min_strength` and `min_gap_sec`, not rounded copies; bump the version when the output changes.
