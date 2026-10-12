# MCP responses

How a user reaches it: any MCP `tools/call` to the stdio server, from any client (Claude Code, Claude Desktop, Cursor, Executor, ...).

Code: `src/executor/mcp_server.py` (`tool_call_response`, `fit_response`, `format_tool_error`).

How to check it works: `pytest tests/test_mcp_response_size.py`.

Gotchas:
- Some MCP clients (Executor, for one) close the connection on a message over 10 MiB and stop waiting after 60 s. Callers then see only a generic tool error; the real cause is in the client's own log.
- Responses are kept under `MAX_RESPONSE_BYTES` (8 MB). Images that don't fit are sent as a smaller inline JPEG with a `note`; `file_path` always has the full-size image.
- The text block carries the same JSON without base64 image bytes; the image is only in `structuredContent`.
- Unexpected exceptions name the tool and the exception type.
