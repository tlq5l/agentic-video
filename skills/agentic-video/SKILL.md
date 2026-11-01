---
name: agentic-video
description: Look at, listen to and check local video files or video URLs with the agentic-video MCP tools. Use when asked what happens in a video, when something is said or shown, how many times something appears, where cuts or silence are, or whether sound and picture line up.
---

# agentic-video

Pass each tool an absolute local path or a video URL as `source`. Start with `review_video` for a first look. Use `probe_video` before choosing any `start_sec`/`end_sec`, because it gives the duration.

## Jobs, in order of frequency

- **First look**: `review_video`. Example: `{"source": "/abs/path/clip.mp4"}`. One call gives metadata, a contact sheet, cuts, black stretches and loudness.
- **What is in it**: `contact_sheet` for one grid image. Example: `{"source": "/abs/path/clip.mp4", "samples": 12}`. For a long video use `overview_chunks`.
- **When is something said**: `transcribe` once, then `search_transcript`. Example: `{"source": "/abs/path/clip.mp4", "query": "launch date"}`. If `transcribe` returns `status: in_progress`, call it again after `retry_after_seconds`.
- **When does something appear on screen**: `build_visual_index` (repeat until complete), then `search_visual`. Example: `{"source": "/abs/path/clip.mp4", "query": "red car"}`. It cannot find words or numbers.
- **How many times something appears**: `count_occurrences`. Example: `{"source": "/abs/path/clip.mp4", "text": "RELEASE"}` for on-screen text, or pass `description` for an object.
- **Where the cuts are**: `detect_cuts`. Example: `{"source": "/abs/path/clip.mp4", "threshold": 0.25}`.
- **Check a moment closely**: `inspect_window` for dense frames of a short span, with `read_text: true` to read on-screen text. Example: `{"source": "/abs/path/clip.mp4", "start_sec": 12, "end_sec": 15, "fps": 4}`. `frame_at` gives one full-size frame, with `crop` and `scale` to read small text.
- **Sound**: `audio_onsets` for hits and claps, `loudness_timeline` for loudness over time, `spectrogram_window` for one sound. Example: `{"source": "/abs/path/clip.mp4", "start_sec": 0, "end_sec": 30}`.
- **Audio and picture sync**: `sync_check`. Example: `{"source": "/abs/path/clip.mp4"}`. It needs an audio track.

## Rules

- Verify a search hit with `inspect_window` or `frame_at` before you state it as fact.
- `label_sounds` and `watch_for` send audio or frames to Cloudflare. Never use them on personal recordings without asking the user first.
- Transcripts and `frame_at` images are cached: a repeat call returns the saved result.
