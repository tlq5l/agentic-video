# Frame at

How a user reaches it: `frame_at` (source, time_sec, optional crop {x, y, width, height} in source pixels, optional scale 1-4).

Code: `src/media/frames.py` (`extract_frame_at`, `_decode_frame_shown_at`), `src/media/probe.py` (displayed size), `src/retrieval/engine.py` (`frame_at`), `src/executor/mcp_server.py` (`frame_at` branch of `_dispatch`).

How to check it works: `pytest tests/test_multi_window_inspection.py -k frame_at`.

Gotchas:
- The PNG has no timestamp overlay and is at native resolution; `inspect_window` burns timestamps and downscales its frames.
- `frame_time_sec` comes from the decoder's showinfo output for the same ffmpeg run that wrote the PNG.
- The frame is the one on screen at `time_sec`: the last frame starting at or before it. On variable-frame-rate video that can be much earlier than the request (a 0.2 s request after a gap can return the frame at 0.16 s), so read `frame_time_sec`.
- Rotated video: `source_width`/`source_height`, the crop and the PNG are all in the displayed (rotated) frame, the same coordinates ffmpeg decodes.
- The frame is converted to RGB before the crop, so a crop at an odd x or y has the same pixels as that region of the full frame.
- Results are cached per source fingerprint, time (to 1 ms), crop and scale; a sidecar `.json` next to each PNG keeps `frame_time_sec`.
