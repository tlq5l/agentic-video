# Frame at

How a user reaches it: `frame_at` (source, time_sec, optional crop {x, y, width, height} in source pixels, optional scale 1-4).

Code: `src/media/frames.py` (`extract_frame_at`, `_video_end_ticks`, `_packets_through_request`, `_shown_frame_pts`, `_decode_frame_shown_at`), `src/media/timing.py` (`read_stream_clock`, `probe_packet_ticks`, `exact_request_seconds`, `parse_selected_frame_pts`), `src/media/probe.py` (displayed size), `src/retrieval/engine.py` (`frame_at`), `src/executor/mcp_server.py` (`frame_at` branch of `_dispatch`).

How to check it works: `pytest tests/test_multi_window_inspection.py -k frame_at`.

The rule: the frame on screen at `time_sec` is the last frame whose start time is at or before it. Selection is exact integer arithmetic on the container's frame times (pts, in the stream's time base); no tolerance is used.
- `time_sec` is read as the simplest fraction that rounds to the JSON number, so `n / fps` names frame `n` at 24, 25, 30, 60 and 2000 fps and at 30000/1001.
- A time equal to a frame's start returns that frame; a time between two starts returns the earlier one.
- The video ends where its last packet ends. A time up to that end is allowed (it returns the last frame); a time after it is refused with an error naming `time_sec` and the end. A time before the first frame is refused the same way.
- A time 1 tick before a frame's start returns the frame before it.

Gotchas:
- The PNG has no timestamp overlay and is at native resolution; `inspect_window` burns timestamps and downscales its frames.
- `frame_time_sec` is the chosen frame's start on the container clock (the same clock `inspect_window` counts from), read from the decoder's showinfo output for the same ffmpeg run that wrote the PNG. The run must report the chosen pts or the call fails.
- On variable-frame-rate video the frame on screen can be much earlier than the request (a 0.2 s request after a gap can return the frame at 0.16 s), so read `frame_time_sec`.
- Some containers store times in whole milliseconds (Matroska by default). Above 1000 fps two frames can share one stored start; the later frame is returned. At 29.97 fps the stored starts are rounded to the millisecond, so an `n / fps` request can return the frame before `n` when the rounding went up. The stored start is what the rule compares.
- Rotated video: `source_width`/`source_height`, the crop and the PNG are all in the displayed (rotated) frame, the same coordinates ffmpeg decodes.
- The frame is converted to RGB before the crop, so a crop at an odd x or y has the same pixels as that region of the full frame.
- Results are cached per source fingerprint, exact requested time, crop and scale; a sidecar `.json` next to each PNG keeps `frame_time_sec`.
