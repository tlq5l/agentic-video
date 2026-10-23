# Watch for

How a user reaches it: the `watch_for` tool, and `python -m src.media.watch VIDEO --when "QUESTION" [--when ...] [--start N --end N --interval N --cut N --consecutive N --cooldown N]`.

Code: `src/media/watch.py` (`watch_for`, `find_wakes`, `_ask`), `src/media/cloudflare.py` (credentials shared with label_sounds), `src/retrieval/engine.py` (`AgenticVideoEngine.watch_for`), `src/executor/mcp_server.py` (dispatch), contract in `src/adapters/mcp_schemas.json`, `src/adapters/mcp_definitions.ts`, `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_watch_for.py tests/test_cloud_token_command.py` (a local stand-in for Cloudflare answers, so no credentials are needed; the live test runs only with `CLOUDFLARE_ACCOUNT_ID` plus `CLOUDFLARE_API_TOKEN` or `AGENTIC_VIDEO_CF_TOKEN_COMMAND`). Live: call the tool on a clip where something appears and check that `wakes` has its time within one `interval_sec` of the first sample that shows it.

What depends on it: nothing else calls it; `AGENTS.md` points to it.

How it decides: one frame (640 px wide) every `interval_sec` goes to Clef with every condition as a yes/no question; a condition wakes at the sample where its chance of yes has been at least `cut` on `consecutive` samples in a row, and at most once per `cooldown_sec` per condition. `first_seen_sec` is the first sample of that run. `conditions[].max_chance` shows how close a quiet condition came.

Gotchas:
- The frames leave the machine (Cloudflare Workers AI, Clef). Never use it on personal recordings. The tool description says so.
- Only the questions you write can wake. A condition nobody asked about is never reported; a frame-difference watcher wakes on anything, which is why it false-wakes.
- An event must span `consecutive` samples: at the defaults an event under 10 s can be missed, and a wake comes about 5 s after the event starts. Lower `interval_sec` or `consecutive` for short events (more calls).
- Real footage has not been measured for accuracy: check each wake against the footage. The example stream is synthetic.
- One call is at most 720 samples. No new sample starts after 40 s from the call's start and a request already sent is cut off at 48 s, so a call answers within about 50 s even when Cloudflare is slow (Executor drops calls at 60 s). When the range is not finished by then, or a sample still fails after its one retry (once `consecutive` samples are done), the finished samples come back as `complete: false` with `next_start_sec` (`consecutive` - 1 samples back, so a run is not cut in half) and a `note` that says which of the two happened; a failure before that point is raised as the error. The cooldown does not carry over, so an event that straddles the cut can wake again in the second call. Results are not cached.
- Cost and compute: each sample is one request to Cloudflare Workers AI, billed there. Samples run 4 at a time: one FFmpeg run each (1 decode thread, one frame), and the temporary frames are deleted at once. Nothing is loaded locally and nothing keeps running after the call. The file is hashed once per call, not once per sample.
- Credentials: the account id comes from `CLOUDFLARE_ACCOUNT_ID` only. The token comes from `CLOUDFLARE_API_TOKEN` or the last line printed by `AGENTIC_VIDEO_CF_TOKEN_COMMAND`; that command runs at most once every 50 minutes per server process, shared with `label_sounds` (a wrangler-based command can be heavy, and two refreshes at once can sign you out of wrangler); the token is kept in memory only, refreshed after 50 minutes because wrangler sign-in tokens last about an hour, and dropped sooner on an HTTP 401 or 403. A token with spaces or other characters that cannot go in a header is refused without being shown. The token is never written into results or errors; a failed request shows at most 300 characters of Cloudflare's reply. A rate-limit or server error is retried once after 2 s.
- `AGENTIC_VIDEO_CF_API_BASE` replaces `https://api.cloudflare.com/client/v4`; the end-to-end test points it at a local stand-in.
