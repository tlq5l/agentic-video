# Changelog

## Unreleased (0.3.8)

- Missing ffmpeg: the one-line error names the tool that is missing (ffmpeg or ffprobe) and gives the install command for macOS, Debian or Ubuntu, and Windows. The `agentic-video-check` command exits with code 2 in that case. (#91)
- The on-screen text and visual search setup message ends with the download command, so pasting it runs as printed. Before, a stray period made Python fail with a SyntaxError. The "Nothing was downloaded" sentence now comes before the command. (#92)
- The CLIP setup error links the README section on GitHub instead of saying "see README", which the pip package does not include. (#87)
- On Linux and Windows, `count_occurrences` with `text` and `read_text` say that Apple Vision runs only on macOS, instead of giving a pip install command that cannot supply it. The macOS message is unchanged. (#90)
- The README's MCP and Python examples say they run from a git checkout, because a pip install has no `fixtures/` folder; pip users are pointed to their own video. (#89)
- Docs: the Claude Desktop bundle page quotes the current ffmpeg error text word for word.
- MCP over stdio: a notification (a message with no id) gets no reply, as JSON-RPC requires, and `ping` returns an empty result instead of "Method not found". All 33 tools were run once over stdio on 0.3.7 and main; each answered.
