# Cut list and average shot length

Problem: list every cut in a video and work out the average shot length.

Needs: ffmpeg and the `.venv` from the README Install section.

Paste into your agent:

> Use agentic-video to list the cuts in `/path/to/video.mp4` with their times. Then give the average shot length (video length divided by the number of shots). Say which entries are transitions.

Tool call (from the repo folder; `source` is required, `threshold` defaults to 0.25, `start_sec` and `end_sec` are optional):

```sh
.venv/bin/python -m src.executor.mcp_server call detect_cuts '{"source": "fixtures/synthetic_bench_60s.mp4"}'
```

Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (structuredContent, trimmed):

```json
{"cuts": [{"timestamp_sec": 7.45, "score": 0.060285, "kind": "transition"},
          {"timestamp_sec": 14.0, "score": 0.170098, "kind": "cut"},
          {"timestamp_sec": 16.0, "score": 0.170076, "kind": "cut"},
          {"timestamp_sec": 21.45, "score": 0.060308, "kind": "transition"},
          ... 9 more ...],
 "total": 13}
```

Average shot length: 60.0 s divided by (13 boundaries + 1 shot) is about 4.3 s. The three transitions count as boundaries here, so the average is approximate.

On reference_videos/sample_10s.mp4 the same call returns `{"cuts": [], "total": 0}`, which is one 10 s shot.

Can miss: a hard cut between two similar pictures that scores under the 0.25 threshold can be missed (README, Limits).
