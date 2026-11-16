# Changelog

## Unreleased

- `count_occurrences` with `direction`: a detection fragment under three samples that continues the appearance before it (within 0.6 s, no jump) is joined to it before its travel is read, so a crossing split by detection dropouts counts once (count_dev seed 6102 occ26_q1: the 7.2-10.0 s crossing came back as three spans and was counted three times). Spans under three samples are still counted as too brief to show a direction.

- Cloud transcription: a piece whose reply repeats one short phrase (quiet videos made whisper answer "Thank you." and drop the real lines) is sent again with the voice filter on, and the filtered reply is used when it has other lines.

- A URL whose download thread the system refused to start no longer blocks the next call for that URL: the next call starts the download and answers ready, instead of answering in_progress forever.

- `count_occurrences` with `direction` lists the spans the filter leaves out (stationary, or moving another way) in a new `excluded_spans` field, each with start_sec, end_sec and travel; count, spans and every other field are unchanged.

- `count_occurrences` with `direction`: an appearance that crosses and comes straight back inside one span (its first and last positions match) now counts for a direction it moved at least 0.5 of the frame along. Before, it read stationary and was left out. count_dev back row occ27_q1: 0 to 4 (truth 4).

## 0.3.8 (2026-11-15)

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
- A URL source with yt-dlp missing names both installs: `brew install yt-dlp` on macOS and `uv pip install --python <the server's python> yt-dlp` on Linux. Before, the message gave only the brew line, which does not exist on Debian or Ubuntu.
- Tests: `test_closing_stdin_while_url_download_runs_waits_for_it` skips when the server cannot find yt-dlp (PATH, /opt/homebrew/bin or /usr/local/bin). The README install does not add yt-dlp, so a clean install failed that one test (1 failed, 866 passed, 39 skipped on the box). URL sources still need yt-dlp, and the test runs wherever it is installed.
- Tests: the visual-interval test (`test_search_visual_rank_one_window_stays_short_and_covers_shot`) uses a flat-colour background and asks for the top 1 hit, so its 15 s bound no longer depends on random gradient frames or on lower-ranked hits merged into the window. It failed about 1 run in 6 under load before.
- Tests: `test_search_visual_lower_hits_merge_into_rank_one_interval` runs search_visual with top_k 5 on the flat-colour clip and asserts that the hits ranked 2 and 3 merge into the rank-1 interval, which is longer than the 8 s cap.
- Visual search reads CLIP weights from `HUGGINGFACE_HUB_CACHE` when `HF_HUB_CACHE` is unset, as huggingface_hub does. `count_repetitions` keeps a valid min/max period pair valid after clamping: when both ends clamp to the same bound, the end on the bound moves one frame (1/15 s) inward, and the note says so.
