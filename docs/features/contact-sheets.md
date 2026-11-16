# Contact sheets

How a user reaches it: `contact_sheet`, `inspect_window`, `inspect_windows`, `inspect_moments`, `overview_chunks`.

Code: `src/media/contact_sheet.py` (`build_contact_sheet`, `compose_contact_sheet`), `src/retrieval/engine.py` (`inspect_window`).

How to check it works: `pytest tests/test_mcp_response_size.py tests/test_multi_window_inspection.py`, then a cold `contact_sheet` with 150 samples on a real video should finish well under 60 s.

Gotchas:
- `build_contact_sheet` seeks each frame with its own ffmpeg run (4 at a time). The MCP `contact_sheet` stops after 40 s with an error that says the frames so far are cached and a repeat call continues.
- `inspect_window` composes its sheet from the dense frames it already extracted; it must not seek each frame again (that took over a minute for 4 windows).
- `inspect_window` with `read_text: true` runs Apple Vision on each dense frame's saved image, the same pixels the frame list points to; each frame gets `text_lines` as in `frame_at`. The burned-in timestamp is part of that image, so its digits can appear as a text line too.

## count_occurrences direction

`count_occurrences` returns a contact sheet with one tile per span. Its optional `direction` (`any` by default, `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top`) narrows the count to appearances whose net motion across the frame is that way, plus spans too brief to show motion (see below); the spans and tiles are then only those appearances. `travel_counts` still lists every moving appearance in every direction, including the selected one, and the note says how many appearances moved another way or stayed still. Motion is estimated from each span's first and last centres (with three or more samples, the average of the first two and of the last two), so check the tiles with `inspect_window` before answering. Without a direction, a span with two samples reports travel from its first centre to its last, and a span with one sample reads `stationary`. With a direction, any span with under three samples has `travel` `unknown` and is counted under any direction (the note and `verify` say so), so check those tiles first. With a direction, each span it leaves out is listed in `excluded_spans` (start_sec, end_sec, travel), so stationary and other-direction appearances can be checked too.

Code: `src/retrieval/occurrence.py` (`count_video`, `DIRECTIONS`). Test: `pytest tests/test_occurrences.py -k direction`.
