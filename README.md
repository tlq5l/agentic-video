# agentic-video

agentic-video lets an AI agent that reads text and images, but not video, work with video files on your own machine. It is a Python library plus a stdio MCP server. The agent asks small, targeted questions ("show me 12 frames from 0:40 to 1:10", "when is 'launch' spoken?") and gets back measurements, transcripts and contact sheets: grids of frames, each with its timestamp burned into the pixels. Nothing is uploaded; everything runs locally with FFmpeg, Whisper and optional local models.

**Status:** experimental. The author uses it every day; support is best-effort. Issues and pull requests are welcome, but expect slow replies.

## What it gives an agent

The MCP server exposes 21 tools (full schemas in `src/adapters/mcp_schemas.json`):

- **Look:** `review_video` (first look in one call: metadata, contact sheet, cuts and a loudness summary), `probe_video` (duration, fps, size, codecs, audio), `contact_sheet`, `overview_chunks` (consecutive sheets across a long video), `inspect_window` / `inspect_windows` (dense frames for short ranges), `inspect_moments` (frames around given instants), `refine_temporal_beam` (zoom into candidate windows).
- **Listen:** `transcribe` (timestamped speech, cached), `search_transcript` (find when words are spoken).
- **Find:** `detect_cuts` (hard cuts and animated scene changes), `build_visual_index` + `search_visual` (local CLIP search for things that are visible, optional).
- **Count and track:** `count_occurrences` (separate appearances of an object or on-screen text), `count_repetitions` (cycles of a repeated action), `track_region_changes` / `track_region_events` (changes inside one fixed rectangle).
- **Audio and sync:** `audio_onsets`, `loudness_timeline`, `spectrogram_window`, `sync_check` (cuts and sound cues against measured audio).

There is also a command-line checker for rendered videos: `python -m src.loop check <video>` measures sync, loudness, onsets, cuts and some picture faults, then writes `report.md` plus evidence images.

`AGENTS.md` is a guide for agents on which tool to use when. `FEATURES.md` maps each feature to its code and tests.

## Install (macOS)

You need Homebrew, FFmpeg, Git and [uv](https://docs.astral.sh/uv/). Tested with Python 3.12 on Apple silicon.

```sh
brew install ffmpeg uv
git clone https://github.com/tlq5l/agentic-video.git
cd agentic-video
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest -q
```

To lint before every commit (`uvx ruff check .`, the core rules in `ruff.toml`), turn the hook on once per clone: `git config core.hooksPath .githooks`.

What is macOS-only:

- On-screen text matching in `count_occurrences` (`text=`) uses Apple Vision through pyobjc.
- The bench-v1 harness uses `sandbox-exec`; some tests and generators use `say`. Tests that need a missing tool skip themselves.

The core tools should also work on Linux with FFmpeg installed, but the author has not tested that.

### Optional: faster, more accurate speech (whisper.cpp turbo)

By default `transcribe` uses faster-whisper on the CPU (`tiny`, `base` or `small`; weights are downloaded on first use). On Apple silicon, whisper.cpp with the large-v3-turbo model is much faster and more accurate:

```sh
brew install whisper-cpp
mkdir -p ~/.cache/whisper.cpp
curl -L -o ~/.cache/whisper.cpp/ggml-large-v3-turbo-q5_0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
```

Set `AGENTIC_VIDEO_WHISPER_CPP_MODEL` if you keep the model somewhere else. When it is installed, `turbo` becomes the default model.

### Optional: visual search (CLIP) and on-screen text

```sh
uv pip install --python .venv/bin/python -r requirements-visual.txt
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])'
```

This installs PyTorch, Transformers and (on macOS) the Apple Vision bindings, and downloads about 605 MB of CLIP weights. The server only loads local files and never calls a paid API. Without these, `search_visual` and `build_visual_index` return an error that explains the setup.

## Run the MCP server

The server speaks MCP over stdio, one JSON message per line:

```sh
.venv/bin/python -m src.executor.mcp_server
```

To add it to an MCP client (Claude Code, Claude Desktop, Cursor and others), use the venv's Python and put the repository on `PYTHONPATH`, so the server starts from any folder:

```json
{
  "mcpServers": {
    "agentic-video": {
      "command": "/absolute/path/to/agentic-video/.venv/bin/python",
      "args": ["-m", "src.executor.mcp_server"],
      "env": { "PYTHONPATH": "/absolute/path/to/agentic-video" }
    }
  }
}
```

Give tools absolute video paths. Results are cached under `~/.cache/agentic-video` (change it with `AGENTIC_VIDEO_CACHE_DIR`).

### Five-line example

From the repository folder: a handshake, the video's metadata, and a 6-frame contact sheet (its JPEG path is in `file_path`):

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"probe_video","arguments":{"source":"fixtures/reference_videos/flower.mp4"}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"contact_sheet","arguments":{"source":"fixtures/reference_videos/flower.mp4","samples":6,"cols":3}}}' \
  | .venv/bin/python -m src.executor.mcp_server
```

The same from Python:

```python
from src.retrieval.engine import AgenticVideoEngine

engine = AgenticVideoEngine("fixtures/reference_videos/flower.mp4")
print(engine.metadata.duration_sec)                  # 5.005
print(engine.overview(samples=6, cols=3).file_path)  # contact sheet JPEG
```

## Results so far

These are the author's own small tests, not independent benchmarks. The raw runs are not part of this repository.

- **Planted faults:** in promo videos with 20 deliberately planted faults, the checker (`python -m src.loop check`) found 14 of 20, with 2 false alarms. Gemini in agentic video mode found 4 of 20.
- **bench-v1 (`benchmarks/bench_v1`):** on the author's small benchmark, Claude using these tools beat Gemini's agentic video mode at counting, finding moments and long-video search.
- **LVBench (public long-video benchmark):** it scored about 86%. Gemini's published score is 88.6%, so it does not beat Gemini there.

It does not beat Gemini's video understanding in general.

## Limits

- **Images come back as files.** Image tools return `file_path` plus a base64 copy inside `structuredContent`, not an MCP image content block. Many clients show the agent only the JSON text, so the agent has to open the JPEG at `file_path` itself (for example with its file-reading tool).
- **Sampling can miss things.** Contact sheets and visual search look at sampled frames; something brief between samples can be missed, and a missing hit does not prove absence. Visual search runs CLIP on the CPU and is slow on long videos.
- **Transcripts can invent lines.** Whisper (especially turbo) can invent lines over music or silence. A line repeated 3 or more times in a row is flagged `suspect`; a single invented line is not.
- **Cuts can be missed.** A hard cut between two similar pictures that scores under the 0.25 threshold can be missed.
- **Counting and sync use heuristics** tuned on the author's videos. Treat their output as evidence to check, not as ground truth.
- **The fix loop needs extra tools.** `python -m src.loop run` calls the Codex CLI as its fixer (default model ladder in `src/loop.py`). The optional `--judge jev` needs `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY` in the environment.
- **bench-v1 needs accounts.** Running it needs the Codex CLI, `GEMINI_API_KEY` in the environment, Node.js and macOS `sandbox-exec`; see `benchmarks/bench_v1/README.md`.
- **No stable API.** Tool names and outputs may change between commits.

## Repository layout

- `src/`: the library (`media/`, `transcript/`, `retrieval/`, `core/`), the MCP server (`executor/mcp_server.py`), the checker (`loop.py`) and tool schemas (`adapters/`).
- `tests/`: the pytest suite. `evals/adversarial_runner.py` holds the failure-injection checks that `tests/test_adversarial_failures.py` runs.
- `benchmarks/bench_v1/`: the blind three-contestant benchmark harness. Its data and answer key live outside the repo.
- `benchmarks/dev/count_dev/`: a small development set for the counting tools.
- `scripts/generate_synthetic_benchmark.py`: regenerates the synthetic fixture video.
- `fixtures/`: small test videos (see Media credits).
- `docs/`: architecture notes and per-feature pages.

## Media credits

- `fixtures/reference_videos/sample_10s.mp4`: a 10-second excerpt, with no audio, of *Big Buck Bunny* (Sunflower version). (c) 2008 Blender Foundation, www.bigbuckbunny.org / peach.blender.org; Sunflower version by Janus Bager Kristensen, 2013 (bbb3d.renderfarming.net). Licensed under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/).
- `fixtures/reference_videos/flower.mp4`: a sample video from MDN Web Docs, released under CC0 (public domain).
- `fixtures/synthetic_bench_60s.mp4` with `.srt`, `.vtt` and `fixtures/synthetic_ground_truth.json`: generated by this repository's `scripts/generate_synthetic_benchmark.py`. Same license as the code.

## License

MIT. See `LICENSE`.
