# Count things crossing the frame one way

Problem: count how many separate times an object crosses the frame in one direction.

Needs: ffmpeg and the `.venv` from the README Install section. No CLIP model is needed when the description names a shape.

Paste into your agent:

> Use agentic-video count_occurrences on `/path/to/video.mp4` for the purple diamond, counting only left-to-right passes. List each span's time, then check each one with inspect_window at its representative_sec.

Tool call. `count_occurrences` runs as a background job inside the process that starts it. The one-line `call` form returns `status: in_progress` and never finishes, because each CLI run is a new process. Send the request to one server process, wait, and send it again in that same process:

```sh
{ echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"count_occurrences","arguments":{"source":"fixtures/synthetic_bench_60s.mp4","description":"purple diamond","direction":"left_to_right"}}}'
  sleep 25
  echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"count_occurrences","arguments":{"source":"fixtures/synthetic_bench_60s.mp4","description":"purple diamond","direction":"left_to_right"}}}'
} | .venv/bin/python -m src.executor.mcp_server
```

Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (response id 3, structuredContent, trimmed):

```json
{"status": "complete", "count": 0, "spans": [], "spans_returned": 0,
 "window": {"start_sec": 0.0, "end_sec": 60.0}, "fps_used": 5.0,
 "note": "direction=left_to_right: counted the 0 of 3 appearance(s) that travel left_to_right ... 0 moved another way and 3 stayed still, and are left out of count and spans."}
```

Without `direction` the same clip gives `count: 3`, at 7.0, 21.0 and 48.0 s, all `travel: stationary`. These are the purple badges, which do not move, so the one-way count is 0. The fixtures have no object that crosses the frame, so this run shows the filter, not a crossing count.

Can miss: appearances shorter than 1/fps can be missed (0.2 s at the default 5 fps). Spans too brief to show motion are counted with `travel: unknown`; check them by eye. Check each span with inspect_window. The counts are heuristics tuned on the author's videos (README, Limits).
