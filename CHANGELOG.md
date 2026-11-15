# Changelog

## Unreleased (0.3.8)

- Missing ffmpeg: the one-line error names the tool that is missing (ffmpeg or ffprobe) and gives the install command for macOS, Debian or Ubuntu, and Windows. The `agentic-video-check` command exits with code 2 in that case. (#91)
- The on-screen text and visual search setup message ends with the download command, so pasting it runs as printed. Before, a stray period made Python fail with a SyntaxError. The "Nothing was downloaded" sentence now comes before the command. (#92)
- The CLIP setup error links the README section on GitHub instead of saying "see README", which the pip package does not include. (#87)
- On Linux and Windows, `count_occurrences` with `text` and `read_text` say that Apple Vision runs only on macOS, instead of giving a pip install command that cannot supply it. The macOS message is unchanged. (#90)
- The README's MCP and Python examples say they run from a git checkout, because a pip install has no `fixtures/` folder; pip users are pointed to their own video. (#89)
- Docs: the Claude Desktop bundle page quotes the current ffmpeg error text word for word.
- MCP over stdio: a notification (a message with no id) gets no reply, as JSON-RPC requires, and `ping` returns an empty result instead of "Method not found". All 33 tools were run once over stdio on 0.3.7 and main; each answered.
- MCP over stdio: `initialize` echoes the client's `protocolVersion` when the server supports it, and otherwise answers with the newest version it supports. The server supports only 2024-11-05 for now: 2025-03-26 requires accepting JSON-RPC batches, which this server does not do yet. Before, it always answered 2024-11-05.
- `count_repetitions`: `min_period_sec` and `max_period_sec` outside 0.133 to 180 s are moved into that range, and the note says so. Before, such values were used as given, with no note. The fps cap (15) and the 180 s window limit were already in place.
- CLI `call` keeps polling a background job (`build_visual_index`, `search_visual`) until it settles, for up to 900 s. Before, each one-line call answered "queued" and the job never finished. The MCP server path is unchanged. (#96)
- MCP over stdio: when the client closes stdin while a background job runs (visual index, motion index, counts, repetitions, transcript, URL download), the server waits for the job before it exits. A visual index stops after its current chunk, so saved chunks are kept. The server waits up to 660 s and exits 0 only when every job has finished; past that it exits 1 with a message. Before, it could abort with exit -6 ("terminate called without an active exception"), and a URL download still running at stdin close was dropped while the server exited 0.
- Tests: the stdio every-tool test expects the visual tools' reply for the CLIP state the server sees (extra missing, weights missing, or present). Before, it assumed the weights were absent, so a host that had them failed. It also lets the jobs its calls start settle before it closes stdin.
