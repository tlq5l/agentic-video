# When to use agentic-video, and when a video-understanding model fits better

agentic-video is a Python library plus a stdio MCP server. It lets an agent that reads text and images, but not video, ask targeted questions about a video file on your machine and get back measurements, transcripts and contact sheets. The project calls itself **experimental**, and support is best-effort ([README](../README.md?ref=agentic-video-when-to-use-docs)).

This page helps you decide between agentic-video and a native video-understanding model. The repo gives no decision rule. The sections below quote what it says about its own limits and results, then state the split those facts suggest. The split is our reading, not a claim the repo makes.

## What an agent gets

- **25 MCP tools**, in five groups: look (`review_video`, `probe_video`, `contact_sheet`, `frame_at`), listen (`transcribe`, `search_transcript`), find (`detect_cuts`, `search_visual`), count and track (`count_occurrences`, `track_region_changes`), and audio and sync (`audio_onsets`, `loudness_timeline`, `sync_check`). Full schemas are in `src/adapters/mcp_schemas.json`; the README lists every tool ([README](../README.md?ref=agentic-video-when-to-use-docs)).
- **Local by default.** Everything runs locally with FFmpeg, Whisper and optional local models, except two optional cloud tools, `watch_for` and `label_sounds` ([README](../README.md?ref=agentic-video-when-to-use-docs)).
- **Images come back as files.** Image tools return a `file_path`; the agent has to open that file itself ([README](../README.md?ref=agentic-video-when-to-use-docs)).
- **Where each tool runs** (Apple silicon, Intel macOS, Linux, and which tools need Cloudflare credentials, Apple Vision or the CLIP model) is in [docs/capabilities.md](capabilities.md?ref=agentic-video-when-to-use-docs).

## When a native video-understanding model is the better fit

What the repo says about its results:

- "It does not beat Gemini's video understanding in general." ([README](../README.md?ref=agentic-video-when-to-use-docs), Results so far)
- In the author's planted-faults test (20 faults in promo videos), the checker found 14 of 20 with 2 false alarms. "Gemini in agentic video mode found 4 of 20." ([README](../README.md?ref=agentic-video-when-to-use-docs), Results so far)
- On the author's small bench-v1 benchmark, Claude using these tools beat Gemini's agentic video mode at counting, finding moments and long-video search. The README calls these "the author's own small tests, not independent benchmarks." ([README](../README.md?ref=agentic-video-when-to-use-docs), Results so far)
- On LVBench, the README reports "about 86%", against Gemini's published 88.6%, and says it "does not beat Gemini there." ([README](../README.md?ref=agentic-video-when-to-use-docs), Results so far)

What the repo says about its limits:

- **Sampling can miss things.** Contact sheets and visual search look at sampled frames, and "a missing hit does not prove absence." ([README](../README.md?ref=agentic-video-when-to-use-docs), Limits)
- **Cuts can be missed**, and counting and sync use heuristics "tuned on the author's videos." Treat their output "as evidence to check, not as ground truth." ([README](../README.md?ref=agentic-video-when-to-use-docs), Limits)
- **Transcripts can invent lines** over music or silence. ([README](../README.md?ref=agentic-video-when-to-use-docs), Limits)

From those facts, our reading:

- A native video model is the simpler first choice for a general description of a clip, when you do not need an answer that points at timestamps you can check, measurements, or a search for a spoken word.
- agentic-video fits better when the answer must point at a burned-in timestamp, when the video should stay on your machine, or when the question is a count, a sync check, a cut list, or "when is X said."

## What the self-run tests do and do not show

The repo has two kinds of self-run results: a smoke run on the author's machines, and a fresh install on Debian 12.

**What they show:**

- A smoke run on 2026-10-31 called every tool once through `handle_call`, on a 20-second clip with a tone and three scene changes, on a Linux machine and an Intel Mac (iMac Pro). On both: 17 tools returned a result, 6 started a background job, and 2 (`watch_for`, `label_sounds`) were not run ([docs/capabilities.md](capabilities.md?ref=agentic-video-when-to-use-docs)).
- A fresh install of 0.3.1 on Debian 12 (x86_64, ffmpeg 5.1) passed the probe, frame and speech-to-text checks ([README](../README.md?ref=agentic-video-when-to-use-docs), Install).
- The pytest suite is optional for users; the README says it takes "about 3 minutes" ([README](../README.md?ref=agentic-video-when-to-use-docs), Install).

**What they do not show:**

- **Apple silicon was not run for any tool** ([docs/capabilities.md](capabilities.md?ref=agentic-video-when-to-use-docs)).
- **A call returning a result is not a correct answer.** The smoke run checks that calls return; it does not check the results against the footage. The "started" cells were not checked: each one "still needs a repeat call or a poll of its job" ([docs/capabilities.md](capabilities.md?ref=agentic-video-when-to-use-docs)).
- **The cloud tools were not run**, because the test machines had no credentials ([docs/capabilities.md](capabilities.md?ref=agentic-video-when-to-use-docs)).
- **Accuracy on real footage is not established.** For `watch_for`: "Real footage has not been measured for accuracy" ([docs/features/watch.md](features/watch.md?ref=agentic-video-when-to-use-docs)).
- **Other distributions and CPUs are untested** ([README](../README.md?ref=agentic-video-when-to-use-docs), Install).
- **The raw runs are not in the repository** ([README](../README.md?ref=agentic-video-when-to-use-docs), Results so far).

## Keeping cloud tools off

`watch_for` sends sampled video frames to Cloudflare Workers AI. `label_sounds` sends audio windows there. Each one sends data only when you call it and the Cloudflare credentials are set. No other tool calls Cloudflare, except the maintainer's shadow trial, which is described below ([README](../README.md?ref=agentic-video-when-to-use-docs), Install).

To keep them off:

1. **Leave these unset** in the server's environment: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` and `AGENTIC_VIDEO_CF_TOKEN_COMMAND`. Without them, both tools fail with a message that names what is missing, and nothing is sent ([README](../README.md?ref=agentic-video-when-to-use-docs), Install).
2. **Leave `AGENTIC_VIDEO_SHADOW_CF_ASR` unset.** It is a trial for the maintainer. It is off unless it names the machine's hostname, and when it is on it sends audio to Cloudflare after every fresh local transcription. It stops by itself after 14 days or 200 uses ([docs/features/shadow-cf-asr.md](features/shadow-cf-asr.md?ref=agentic-video-when-to-use-docs)).
3. **Tell the agent not to call them.** The docs describe no setting that hides the two tools from the agent's tool list. Put the instruction in the agent's instructions file, such as AGENTS.md or CLAUDE.md. The README has a starting block for that file ([README](../README.md?ref=agentic-video-when-to-use-docs), Tell your agent to use it).
4. **Do not use them on recordings you need to keep private.** Each `watch_for` sample is one request that Cloudflare bills, and results are not cached, so a repeated `label_sounds` call repeats the cloud cost ([docs/features/watch.md](features/watch.md?ref=agentic-video-when-to-use-docs), [docs/features/label-sounds.md](features/label-sounds.md?ref=agentic-video-when-to-use-docs)).
