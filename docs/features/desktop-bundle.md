# Claude Desktop bundle

What it is: `agentic-video-<version>.mcpb`, a one-click install for Claude Desktop. Double-click the file and Claude Desktop installs the same stdio MCP server that `uvx agentic-video` runs. The tools do not change.

How a user gets it: download the file from the latest GitHub release and double-click it. The first start takes about a minute while Claude Desktop installs the Python packages with uv. ffmpeg still has to be installed (`brew install ffmpeg`).

How it is built: `python3 scripts/build_mcpb.py` (stdlib only). It stages `src/`, `pyproject.toml`, `requirements.txt`, `requirements-visual.txt`, `README.md`, `LICENSE` and `mcpb/manifest.json` in a fresh folder, validates the manifest and packs `dist/agentic-video-<version>.mcpb` with the pinned `@anthropic-ai/mcpb` CLI through `npx`. It then checks the archive: no tests, fixtures, evals, benchmarks, `.venv`, `.git` or `server/` entries, and under 5 MB. The version comes from `pyproject.toml`. The manifest is `mcpb/manifest.json` (server type `uv`, run `uv run --directory ${__dirname} agentic-video`).

How it is released: the `mcpb` job in `.github/workflows/publish.yml` runs on each published release, builds the bundle and uploads it to that release with `gh release upload --clobber`. The PyPI job is unchanged.

ffmpeg PATH rule: Claude Desktop starts servers with a minimal PATH, so Homebrew's ffmpeg is often not on it. `ensure_ffmpeg_on_path()` in `src/core/config.py` prepends the first of `/opt/homebrew/bin`, `/usr/local/bin` and `~/.local/bin` that holds both `ffmpeg` and `ffprobe`. When a tool still cannot find them, the error text says to run `brew install ffmpeg` and restart Claude Desktop (`format_tool_error` in `src/executor/mcp_server.py`).

Known limits:
- macOS only (the manifest lists the `darwin` platform).
- The optional visual extras (`requirements-visual.txt`: PyTorch, Transformers and about 605 MB of CLIP weights) are not installed by the bundle. `search_visual`, `build_visual_index` and on-screen text matching then return an error that explains the setup; see the README section "Optional: visual search (CLIP) and on-screen text".
- The first run needs network access for uv to download the packages.

How to check it works: `pytest tests/test_mcpb_bundle.py` (builds the bundle, runs the server from the unzipped bundle, checks the ffmpeg error text).
