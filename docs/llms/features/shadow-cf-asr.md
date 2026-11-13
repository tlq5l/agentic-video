# Shadow Cloudflare transcription (trial)

This SENDS AUDIO TO CLOUDFLARE. It is a trial switch for the maintainer, on the maintainer's own machines. It is never on by default, and it is not in the public PyPI package's defaults.

How a user reaches it: nothing user-facing. It runs after a fresh local transcription (`transcribe`, `request_transcription`, the background ASR slot, or a synchronous `AgenticVideoEngine.transcribe` call). The user gets the local transcript, unchanged.

Code: `src/transcript/shadow_cf_asr.py` (the trial, the job and the log), the hook in `src/transcript/asr.py` (`_BackgroundTranscriptionSlot.start` for background runs, `transcribe_video` for synchronous ones), `src/media/cloudflare.py` (`credentials`, `run_model`). The module docstring quotes the Cloudflare docs for `@cf/openai/whisper-large-v3-turbo`.

How to turn it on: set `AGENTIC_VIDEO_SHADOW_CF_ASR` in the server's environment to this machine's hostname (`hostname` in a shell; a comma list of hostnames is allowed). The match is exact. Any other value, including `1`, does nothing and sends nothing. Credentials are also needed (`CLOUDFLARE_ACCOUNT_ID` plus `CLOUDFLARE_API_TOKEN` or `AGENTIC_VIDEO_CF_TOKEN_COMMAND`). If either check fails, nothing is sent.

What it does for each fresh local run:
- Skips cache hits, polls, subtitle sidecars and files with no audio. Only a fresh ASR run starts it.
- Starts a detached, low-priority child process after the local result is cached: after the background slot is released for a background run, or, for a synchronous call, from a daemon thread so the caller gets its transcript without waiting for credentials or the trial state. A synchronous call whose process exits before that thread starts its child skips the shadow run; the transcript is unaffected. It never runs inside `request_transcription` or inside the memory-capped worker, and any error in starting it is ignored.
- The child cuts the audio into 16 kHz mono MP3 pieces of 600 s with no overlap, and sends up to 4 pieces at a time. Each request has a 120 s timeout. Each returned segment gets `piece_index * 600` added to its start and end.
- The language is the local result's language. `vad_filter` is false. There is no cutoff; this runs in the background.
- If the local run found no speech, it still sends and logs `local_has_speech: false`. Cloudflare can make up text on silence; that is trial data.

Log: one JSON line per job in `~/.local/state/agentic-video/shadow-cf-asr.jsonl` (or under `$XDG_STATE_HOME`). Fields: `ts`, `agentic_video`, `source_fingerprint`, `source_path` (the owner's machine only), `duration_sec`, `language`, `local_model`, `local_engine`, `local_has_speech`, `local_text`, `local_segments`, `local_wall_sec`, `cf_model`, `cf_text`, `cf_segments`, `cf_wall_sec`, `pieces`, `piece_sec`, `piece_errors`, `status` (`ok`, `partial`, `failed`, `decode_failed`). The token and the raw audio are never written. The directory is mode 0700 and the file is 0600.

Auto-off: the state file `shadow-cf-asr-state.json` next to the log records `started_at` (the first admitted job) and the count of admitted uses. A use is admitted once per job, right before the pieces are sent; a failed Cloudflare attempt still counts. The trial ends when more than 14 days have passed since `started_at`, or when 200 uses are reached. The check runs before a child starts, again when the child admits the job right before sending, and before every piece and every retry (so an expiry during a job stops its remaining pieces; the job is still one use). The first refusal writes one `{"status": "trial_ended", ...}` line and the trial stays off; restarting the server does not reset it. An unreadable state file also ends the trial (it fails closed).

How to check it works: `pytest tests/test_shadow_cf_asr_e2e.py`. It drives `handle_call("transcribe")` on a 20 s clip with a local stand-in for Cloudflare (`AGENTIC_VIDEO_CF_API_BASE`). The one live test sends a 2 s clip to real Cloudflare through the same guarded path, and runs only when credentials, this host's switch and an open trial are all present; the test machine has none.

Gotchas:
- The audio leaves the machine. Do not switch this on for recordings you need to keep private.
- It runs at most once per fresh local run. An explicit `timeout_sec` run made directly from the engine (`AgenticVideoEngine.transcribe`) is a fresh run too, so it is sent as well.
- `AGENTIC_VIDEO_SHADOW_PIECE_SEC` shortens the piece length and `AGENTIC_VIDEO_SHADOW_PARALLEL` sets how many pieces run at once. The tests use them; leave both unset.
- Trial use is logged per job, not per piece, so 200 uses is 200 jobs.
