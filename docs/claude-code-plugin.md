# Claude Code plugin

Installs the agentic-video MCP server and its skill into Claude Code in two commands. Install ffmpeg first (`brew install ffmpeg` on macOS, `sudo apt-get install -y ffmpeg` on Debian or Ubuntu) and install `uv` so `uvx` is on your PATH. The first run downloads the Python package.

## Install

```sh
claude plugin marketplace add tlq5l/agentic-video
claude plugin install agentic-video@agentic-video
```

## What it adds

- **MCP server `agentic-video`**: started with `uvx agentic-video`, the same command the README uses for `claude mcp add`. Declared inline in `.claude-plugin/plugin.json`.
- **Skill `agentic-video`**: in `skills/agentic-video/SKILL.md`. When you ask what happens in a video, when something is said or shown, how often something appears, where the cuts or silence are, or whether sound and picture line up, the skill picks the right tool, starting with `review_video`.

## Remove

```sh
claude plugin uninstall agentic-video@agentic-video
claude plugin marketplace remove agentic-video
```

## Reference

- Plugin manifest: https://code.claude.com/docs/en/plugins-reference
- Marketplaces: https://code.claude.com/docs/en/plugin-marketplaces
