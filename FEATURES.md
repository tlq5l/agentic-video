# Feature map

One line per feature: name | what it does | words people use | details page.

- mcp-responses | how every MCP tool result is sent (size limit, inline images, errors) | Internal tool error, too big, tool_file | docs/features/mcp-responses.md
- review-video | review_video: metadata, contact sheet, cuts and loudness summary in one call (first look at a clip) | first look, review a video, probe then contact sheet then cuts, one call | docs/features/review-video.md
- contact-sheets | contact_sheet, inspect_window(s) and other sheet images | contact sheet, overview, inspect windows, slow | docs/features/contact-sheets.md
- loudness-timeline | loudness_timeline: RMS, peak, bands, silence, outliers | loudness, peak, dBFS, clipping | docs/features/loudness-timeline.md
- cut-detection | detect_cuts: hard cuts and animated scene changes, with times and scores | cuts, scene change, transition, one frame early, shot list | docs/features/cut-detection.md
- frame-cadence | check_frame_cadence: time ranges where the unique-frame rate is below the file's frame rate (doubled frames, judder) | judder, stutter, duplicate frames, 30 fps in a 60 fps file, dropped frames, cadence | docs/features/frame-cadence.md
- transcribe | transcribe and search_transcript: timestamped speech lines; flags a line repeated 3+ times in a row | transcript, lyrics, what did they say, invented lines, repeated lines, hallucination | docs/features/transcribe.md
