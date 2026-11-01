# Find where something is said

Problem: find the moment a phrase is spoken in a video, with its timecode.

Needs: ffmpeg and the `.venv` from the README Install section. Optional, not run here: whisper.cpp for better speech (README, "faster, more accurate speech").

Paste into your agent:

> Use agentic-video to find when "Project Cygnus" is said in `/path/to/video.mp4`. Quote the timecode and the full line. If it is not said, say so; do not guess from the frames.

Tool calls (from the repo folder). `transcribe` is cached after its first run. If it returns `status=in_progress`, repeat the same call after `retry_after_seconds`.

```sh
.venv/bin/python -m src.executor.mcp_server call transcribe '{"source": "fixtures/synthetic_bench_60s.mp4", "model_size": "tiny"}'
.venv/bin/python -m src.executor.mcp_server call search_transcript '{"source": "fixtures/synthetic_bench_60s.mp4", "query": "Project Cygnus"}'
```

`transcribe` returned 10 timestamped segments on its first call. The fixture's `.vtt` sidecar supplied them (`transcribe` reports "Subtitle sidecar used"), so this example is subtitle-backed search: it finds subtitle lines, and it does not show what the audio says. Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (structuredContent of `search_transcript`, trimmed):

```json
{"hits": [{"segment_id": 8, "start_sec": 34.0, "end_sec": 36.5,
           "text": "Project Cygnus activated. Visual verification token is displayed on screen.",
           "score": 1.0, "start_timecode": "00:00:34.000", "end_timecode": "00:00:36.500",
           "match": "exact"}],
 "total": 1,
 "note": "Semantic search is off (semantic search needs macOS ...); keyword matches only."}
```

A hit with `match: exact` or `match: all_terms` means the query's content words appear in the transcript. Negative answers: if there is neither an `exact` nor an `all_terms` hit, say the words were not found. An `all_terms` hit has every content word in one segment but not as one phrase. Common words (such as "a", "from", "the", "and"; the list is `_STOPWORDS` in `src/transcript/search.py`) are ignored by both matches, so they can be absent from the hit. `"Cygnus Project"` on this fixture returns segment 8 with `match: all_terms` (score 0.8) and no exact hit. `"Project from Cygnus"` also returns segment 8 as `all_terms` (score 1.0, as the content words "project cygnus" appear in order), although "from" is not in the line.

Can miss: Whisper can invent lines over music or silence (lines repeated 3 or more times in a row carry `suspect`). On Linux there are no meaning matches, so a reworded phrase is missed. Text shown on screen is not in the transcript (see the on-screen text recipe).
