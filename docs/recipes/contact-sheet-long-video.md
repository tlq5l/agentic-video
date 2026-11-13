# Timestamped contact sheet for a long video

Problem: see the layout of a whole video in one image, with a timestamp on each frame.

Needs: ffmpeg and the `.venv` from the README Install section.

Paste into your agent:

> Use agentic-video contact_sheet on `/path/to/video.mp4` with 24 frames. Open the JPEG at file_path and tell me what happens in each row, quoting the burned-in timestamps. Do not guess times from the filename.

Tool call (from the repo folder). `samples` defaults to 12 (max 150); `cols` defaults to 4. For a long video, pass `start_sec` and `end_sec` to make one sheet per range.

```sh
.venv/bin/python -m src.executor.mcp_server call contact_sheet '{"source": "fixtures/synthetic_bench_60s.mp4", "samples": 24, "cols": 6}'
```

Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (structuredContent, trimmed):

```json
{"file_path": "/home/agent/.cache/agentic-video/cc3956f3407c03c8c394/contact_sheets/sheet_24samples_6cols_a767f4.jpg",
 "sample_count": 24, "rows": 4, "cols": 6, "start_sec": 1.25, "end_sec": 58.75,
 "timestamps": [1.25, 3.75, 6.25, "... 20 more ...", 58.75]}
```

The longest fixture is 60 s, so this run did not test a long file. The `start_sec` and `end_sec` split is from the tool schema and is not run here. For a long file, `overview_chunks` returns consecutive sheets in one call (README); that was not run here either.

Can miss: 24 frames over 60 s is one frame every 2.5 s. Anything shorter than that between two frames is not on the sheet. A missing thing is not proof it is absent (README, Limits); open inspect_window on the range to check.
