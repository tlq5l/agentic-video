# Label sounds

How a user reaches it: the `label_sounds` tool, and `python -m src.media.sound_labels VIDEO START END [--labels a,b] [--window-sec N]`.

Code: `src/media/sound_labels.py` (`label_sounds`, `_decode_range`, `_ask`), `src/media/cloudflare.py` (credentials shared with watch_for), `src/retrieval/engine.py` (`AgenticVideoEngine.label_sounds`), `src/executor/mcp_server.py` (dispatch), contract in `src/adapters/mcp_schemas.json`, `src/adapters/mcp_definitions.ts`, `src/executor/types.d.ts`.

How to check it works: `pytest tests/test_label_sounds.py tests/test_cloud_token_command.py` (the live test runs only with Cloudflare credentials set; the rest need none). Live: set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (or `AGENTIC_VIDEO_CF_TOKEN_COMMAND`, a command that prints the token) and call the tool on a range with a clear sound; the right label has the highest chance.

What depends on it: nothing else calls it; `audio_onsets` and `AGENTS.md` point to it for naming a sound.

Gotchas:
- The audio leaves the machine: each window is sent as a 16 kHz mono WAV to Cloudflare Workers AI (Clef-omni). Never use it on personal recordings. The tool description says so.
- It labels windows of a range you give it; it does not search a long video. Find candidate times with `audio_onsets` or `loudness_timeline` first. At most 60 windows per call (300 s at the default 5 s).
- Closed list: a sound not in `labels` is forced onto the nearest label or onto `none_chance`, so list every sound that may be present. Steady musical chords can get a false `church bells` label.
- Treat `top_chance` under 0.5 as weak and confirm with `spectrogram_window`.
- Credentials: the account id comes from `CLOUDFLARE_ACCOUNT_ID` only (none is built in). A token command (the last line it prints is the token) runs at most once every 50 minutes per server process, shared with `watch_for` (the token is kept in memory only, refreshed after 50 minutes because wrangler sign-in tokens last about an hour, and dropped sooner on an HTTP 401 or 403); it is a separate short-lived process.
- The token is never written into results or errors; a failed request shows at most 300 characters of Cloudflare's reply.
- Windows are sent 4 at a time. A rate-limit or server error is retried once after 2 s.
- Time limit: no window starts after about 40 s and a call answers within about 50 s. A longer range returns the windows finished in time, in order, with `complete: false` and `next_start_sec` to continue from.
- Results are not cached: repeating a call repeats the cloud cost.
