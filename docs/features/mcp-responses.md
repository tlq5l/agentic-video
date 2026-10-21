# MCP responses

How a user reaches it: any MCP `tools/call` to the stdio server, from any client (Claude Code, Claude Desktop, Cursor, Executor, ...).

Code: `src/executor/mcp_server.py` (`tool_call_response`, `fit_response`, `format_tool_error`).

How to check it works: `pytest tests/test_mcp_response_size.py`.

Gotchas:
- Some MCP clients (Executor, for one) close the connection on a message over 10 MiB and stop waiting after 60 s. Callers then see only a generic tool error; the real cause is in the client's own log.
- Responses are kept under `MAX_RESPONSE_BYTES` (8 MB). With `AGENTIC_VIDEO_INLINE_IMAGES=1`, images that don't fit are sent as a smaller inline JPEG with a `note`; `file_path` always has the full-size image.
- By default neither the text block nor `structuredContent` carries base64 image bytes (Claude Code rejects results over ~25k tokens); `file_path` has the image. `AGENTIC_VIDEO_INLINE_IMAGES=1` puts the images back in `structuredContent` only.
- A `transcribe` result over 20,000 characters lists the first segments, sets `segments_omitted` and writes the full transcript to `transcript_path`.
- While a long transcription runs, the `in_progress` result carries `elapsed_sec`, `estimated_total_sec` and `estimated_percent` (an estimate from audio length and model speed).
- Unexpected exceptions name the tool and the exception type.
