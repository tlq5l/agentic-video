# Cloud transcription

What it does: on hosts switched on for it, every fresh transcript is made by Cloudflare Workers AI
(@cf/openai/whisper-large-v3-turbo) instead of the local model. The audio is sent to Cloudflare. Nothing else is sent:
no frames, no language hint, no video.

Switch (per host, off by default):
- Environment: AGENTIC_VIDEO_CLOUD_ASR=<hostname>[,<hostname>]. Exact match on the machine's hostname.
- Or, when that variable is empty or unset: the settings file $XDG_CONFIG_HOME/agentic-video/cloud-asr.json
  (default ~/.config/agentic-video/cloud-asr.json):

  {"hosts": ["<hostname>"], "account_id": "<account id>", "token_command": "<command that prints a token>"}

The settings file never holds a token. The token comes from CLOUDFLARE_API_TOKEN, else AGENTIC_VIDEO_CF_TOKEN_COMMAND,
else "token_command". The account id comes from CLOUDFLARE_ACCOUNT_ID, else "account_id".

End date: 2026-12-01 00:00 +07 (2026-11-30 17:00 UTC), when Workers AI billing starts (owner decision 2026-11-16).
From then on no new cloud run is sent and the transcribe call says cloud transcription ended. A job that is running
across the end stops sending and fails. Cached transcripts and subtitle sidecars keep working.

No fallback: a cloud failure is an error that says local transcription is off on this machine. The local model
(whisper.cpp or faster-whisper) never runs on a switched-on host, not even after a cloud failure.

Other Workers AI features (watch_for, label_sounds, the shadow trial) stay off: WORKERS_AI_ON in
src/media/cloudflare.py is still False.

How it runs: the audio is cut into 10-minute MP3 pieces (16 kHz mono, 32 kb/s), four pieces at a time, each with
vad_filter off and no language (auto-detect). Piece offsets are added, the pieces are stitched, and the result is cached
under transcript_asr_cloud. A piece whose reply repeats one short phrase is sent once more with vad_filter on, and that reply is used if it has another line: quiet videos made whisper repeat "Thank you." and drop the real lines, and the filter stays off first because it drops sung lines. Code: src/transcript/cloud_asr.py, and the cloud branch in transcribe_video in src/transcript/asr.py.

Measured 2026-10-23 on a Vietnamese test clip, against local whisper.cpp q5_0: character error rate 8.95 against 15.77,
closer timestamps. 43 minutes of audio took 44 s, about 2 cents per 44 minutes.

Test: pytest tests/test_cloud_asr.py (no network; a local stand-in answers for Workers AI).
