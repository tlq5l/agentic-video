# Check audio/video sync

Problem: check that each sound lands within the tolerance of the time it should match.

Needs: ffmpeg and the `.venv` from the README Install section. The file needs an audio track (`sample_10s.mp4` has none).

Paste into your agent:

> Use agentic-video sync_check on `/path/to/video.mp4` with these cue times: 14.0 s "supernova chime", 24.0 s "low pulse", 34.0 s "cygnus tone". Report each cue as ok, early or late, with the offset in ms.

Tool call (from the repo folder). `cues` is a list of `time_sec` plus a `label`; `tolerance_ms` defaults to 80.

```sh
.venv/bin/python -m src.executor.mcp_server call sync_check '{"source": "fixtures/synthetic_bench_60s.mp4", "cues": [{"time_sec": 14.0, "label": "supernova chime"}, {"time_sec": 24.0, "label": "low pulse"}, {"time_sec": 26.0, "label": "mid pulse"}, {"time_sec": 28.0, "label": "high pulse"}, {"time_sec": 34.0, "label": "cygnus tone"}]}'
```

Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (structuredContent, trimmed):

```json
{"total_cues": 5, "total_cuts": 13, "tolerance_ms": 80.0,
 "cues": [
  {"time_sec": 14.0, "label": "supernova chime", "status": "ok", "offset_ms": 0.0, "dominant_band": "707.1-890.9 Hz"},
  {"time_sec": 24.0, "label": "low pulse", "status": "ok", "offset_ms": 0.0, "dominant_band": "176.8-222.7 Hz"},
  {"time_sec": 26.0, "label": "mid pulse", "status": "ok", "offset_ms": 0.0, "dominant_band": "445.4-561.2 Hz"},
  {"time_sec": 28.0, "label": "high pulse", "status": "ok", "offset_ms": 5.333, "dominant_band": "561.2-707.1 Hz"},
  {"time_sec": 34.0, "label": "cygnus tone", "status": "ok", "offset_ms": 5.333, "dominant_band": "707.1-890.9 Hz"}]}
```

All five cues are within 6 ms of an audio onset. Add `"verbose": true` to see each cue's candidate onsets.

Can miss: the sync and counting checks use heuristics tuned on the author's videos, so treat a result as evidence to check (README, Limits). Only the cue times you pass are checked; a sound you did not list appears only in `unmatched_strong_onsets`.
