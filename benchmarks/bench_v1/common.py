"""Shared paths and JSONL helpers for bench-v1.

Benchmark data lives outside the repository. Answer-key access is deliberately
kept out of contestant code paths.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path.home() / ".local/share/agentic-video-bench/bench-v1"
DEFAULT_SEALED_DIR = Path.home() / ".local/share/agentic-video-sealed"
DEFAULT_KEY_PATH = DEFAULT_SEALED_DIR / "bench-v1/answer_key.jsonl"
# The MCP checkout and Python the contestant's tools run from: this repository and
# the interpreter running the benchmark, unless AGENTIC_VIDEO_MCP_CWD / AGENTIC_VIDEO_PYTHON say otherwise.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS = ("moment", "search", "count")
CONTESTANTS = ("ours-codex", "gemini-agentic", "gemini-plain")


def data_root() -> Path:
    return Path(os.environ.get("AGENTIC_VIDEO_BENCH_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()


def scratch_dir(prefix: str) -> Path:
    """Make a per-run temp folder under a `.noindex` parent so Spotlight skips it."""
    parent = Path(tempfile.gettempdir()) / "agentic-video-bench.noindex"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def question_path() -> Path:
    return data_root() / "questions.jsonl"


def default_tool_cwd() -> Path:
    return Path(os.environ.get("AGENTIC_VIDEO_MCP_CWD", str(REPO_ROOT))).expanduser()


def default_tool_python() -> Path:
    return Path(os.environ.get("AGENTIC_VIDEO_PYTHON", sys.executable)).expanduser()


def key_path() -> Path:
    return Path(os.environ.get("AGENTIC_VIDEO_BENCH_KEY", str(DEFAULT_KEY_PATH))).expanduser()


def visual_model_environment(tool_python: Path) -> dict[str, str]:
    """Return the offline Hugging Face environment shared by ingest and Codex MCP."""
    configured_home = os.environ.get("AGENTIC_VIDEO_HF_HOME")
    if configured_home:
        hf_home = Path(configured_home).expanduser()
    else:
        python_path = Path(tool_python).expanduser().absolute()
        hf_home = python_path.parent.parent / "huggingface"
    return {"HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL item must be an object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]], *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if path == key_path() else 0o755)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temp_path.chmod(mode)
        os.replace(temp_path, path)
        if mode is not None:
            path.chmod(mode)
    finally:
        if temp_path.exists():
            trash = shutil.which("trash")
            if trash:
                subprocess.run([trash, str(temp_path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def merge_by_id(existing: list[dict[str, Any]], updates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {row["id"]: row for row in existing}
    merged.update({row["id"]: row for row in updates})
    return [merged[key] for key in sorted(merged)]
