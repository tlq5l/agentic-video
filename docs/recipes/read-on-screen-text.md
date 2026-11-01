# Read on-screen text at a moment

Problem: read the text that is on screen at one exact moment.

Needs: ffmpeg and the `.venv` from the README Install section. For `read_text: true` only (not run here): `uv pip install --python .venv/bin/python -r requirements-visual.txt` on macOS, which adds the Apple Vision bindings.

Paste into your agent:

> Use agentic-video frame_at on `/path/to/video.mp4` at 35 seconds. Open the PNG it returns and quote the on-screen text exactly, including any code.

Tool call (from the repo folder). `time_sec` is required. Optional `crop` (x, y, width, height in source pixels) and `scale` (1 to 4) enlarge small text; they are not run in this recipe.

```sh
.venv/bin/python -m src.executor.mcp_server call frame_at '{"source": "fixtures/synthetic_bench_60s.mp4", "time_sec": 35.0}'
```

Output from agentic-video 0.3.3 on synthetic_bench_60s.mp4 (structuredContent, trimmed; `file_path` is on the box):

```json
{"frame_time_sec": 35.0, "width": 640, "height": 360, "crop": null, "scale": 1.0,
 "file_path": "/home/agent/.cache/agentic-video/cc3956f3407c03c8c394/frame_at/frame_at_7fa625d43725449b.png"}
```

I opened that PNG. It shows "PROJECT CYGNUS CODE [CYGNUS-VERIFIED-77]" and "TOKEN: CYGNUS-VERIFIED-77", with "TIME: 00:35.000" burned in. The text was read by eye from the image, not by OCR.

`"read_text": true` adds Apple Vision OCR. On the Linux box it returns an error instead: `read_text needs Apple Vision through pyobjc (macOS only)`. That call was not run on a Mac.

Can miss: this is one frame at one time. Text that is on screen for less than the gap between your chosen times can be missed. Small text can be misread by eye, so crop and scale it (see the arguments above) before you quote it.
