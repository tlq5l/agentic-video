# Frame at

How a user reaches it: `frame_at` (source, time_sec, optional crop {x, y, width, height} in source pixels, optional scale 1-4).

Code: `src/media/frames.py` (`extract_frame_at`, `_decode_one_frame`), `src/retrieval/engine.py` (`frame_at`), `src/executor/mcp_server.py` (`frame_at` branch of `_dispatch`).

How to check it works: `pytest tests/test_multi_window_inspection.py::test_frame_at_native_frame_crop_and_scale_end_to_end`.

Gotchas:
- The PNG has no timestamp overlay and is at native resolution; `inspect_window` burns timestamps and downscales its frames.
- `frame_time_sec` comes from the decoder's showinfo output for the same ffmpeg run that wrote the PNG.
- The frame is converted to RGB before the crop, so a crop at an odd x or y has the same pixels as that region of the full frame.
- Results are cached per source fingerprint, time (to 1 ms), crop and scale; a sidecar `.json` next to each PNG keeps `frame_time_sec`.
