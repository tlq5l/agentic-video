# Feature map

One line per feature: name | what it does | words people use | details page.

- mcp-responses | how every MCP tool result is sent (size limit, inline images, errors) | Internal tool error, too big, tool_file | docs/features/mcp-responses.md
- review-video | review_video: metadata, contact sheet, cuts and loudness summary in one call (first look at a clip) | first look, review a video, probe then contact sheet then cuts, one call | docs/features/review-video.md
- contact-sheets | contact_sheet, inspect_window(s) and other sheet images | contact sheet, overview, inspect windows, slow | docs/features/contact-sheets.md
- frame-at | frame_at: one frame at native size as a PNG, optional crop and 1-4x Lanczos upscale, no timestamp drawn | exact frame, full size, zoom, crop, read small text, screenshot at time, raw ffmpeg | docs/features/frame-at.md
- loudness-timeline | loudness_timeline: RMS, peak, bands, silence, outliers | loudness, peak, dBFS, clipping | docs/features/loudness-timeline.md
- audio-onsets | audio_onsets: non-speech attacks with ms times and five-band profiles; merges hits closer than min_gap_sec | onset, hit, attack, double count, two onsets per sound, click, clap | docs/features/audio-onsets.md
- cut-detection | detect_cuts: hard cuts and animated scene changes, with times and scores | cuts, scene change, transition, one frame early, shot list | docs/features/cut-detection.md
- frame-cadence | check_frame_cadence: time ranges where the unique-frame rate is below the file's frame rate (doubled frames, judder) | judder, stutter, duplicate frames, 30 fps in a 60 fps file, dropped frames, cadence | docs/features/frame-cadence.md
- transcribe | transcribe and search_transcript: timestamped speech lines; flags a line repeated 3+ times in a row; search finds exact words first, then passages by meaning (on-device EmbeddingGemma 2, optional) | transcript, lyrics, what did they say, invented lines, repeated lines, hallucination, paraphrase, said something like, semantic search | docs/features/transcribe.md
- watch-for | watch_for: samples frames every few seconds and wakes when a yes/no question about the picture holds on 2 samples in a row (cloud model Clef) | watchdog, notify me when, tell me when a person appears, wait until, monitor a stream, alert on, motion detection | docs/features/watch.md
- label-sounds | label_sounds: names the sounds in audio windows with a cloud model (Clef-omni) | what sound is this, dog barking, siren, glass breaking, name the sound, acoustic ear | docs/features/label-sounds.md
- desktop-bundle | Claude Desktop one-click install: agentic-video-<version>.mcpb (uv server, macOS), built by scripts/build_mcpb.py and attached to each GitHub release | Claude Desktop, one click, double-click install, .mcpb, desktop extension, install in Desktop, bundle, ffmpeg not found in Desktop | docs/features/desktop-bundle.md
