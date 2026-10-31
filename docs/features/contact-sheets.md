# Contact sheets

How a user reaches it: `contact_sheet`, `inspect_window`, `inspect_windows`, `inspect_moments`, `overview_chunks`.

Code: `src/media/contact_sheet.py` (`build_contact_sheet`, `compose_contact_sheet`), `src/retrieval/engine.py` (`inspect_window`).

How to check it works: `pytest tests/test_mcp_response_size.py tests/test_multi_window_inspection.py`, then a cold `contact_sheet` with 150 samples on a real video should finish well under 60 s.

Gotchas:
- `build_contact_sheet` seeks each frame with its own ffmpeg run (4 at a time). The MCP `contact_sheet` stops after 40 s with an error that says the frames so far are cached and a repeat call continues.
- `inspect_window` composes its sheet from the dense frames it already extracted; it must not seek each frame again (that took over a minute for 4 windows).
- `inspect_window` with `read_text: true` runs Apple Vision on each dense frame's saved image, the same pixels the frame list points to; each frame gets `text_lines` as in `frame_at`. The burned-in timestamp is part of that image, so its digits can appear as a text line too.
