#!/usr/bin/env python3
"""Build the Claude Desktop bundle dist/agentic-video-<version>.mcpb.

Stdlib only, so it runs on the system python3 (3.9+) and on CI. It stages the
runtime files in a fresh folder, validates the manifest and packs the bundle
with the pinned @anthropic-ai/mcpb CLI (run through npx), then checks the
archive. Any failure exits non-zero.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCPB_CLI = "@anthropic-ai/mcpb@2.1.2"
MANIFEST_SOURCE = ROOT / "mcpb" / "manifest.json"
# Copied to the bundle root, next to the manifest. uv reads pyproject.toml and the
# requirements files to install the packages on the user's Mac.
COPY_FILES = ["pyproject.toml", "requirements.txt", "requirements-visual.txt", "README.md", "LICENSE"]
FORBIDDEN_PARTS = {".venv", ".git", "tests", "fixtures", "evals", "benchmarks", "__pycache__", "dist"}
FORBIDDEN_PREFIXES = ("server/",)  # an extension's own server/lib or server/venv must never ship
MAX_BYTES = 5_000_000


class BuildError(Exception):
    pass


def pyproject_version(text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise BuildError("pyproject.toml has no version line")
    return match.group(1)


def stage(staging: Path, version: str) -> Path:
    """Copy the runtime files into staging and write the manifest with the package version."""
    shutil.copytree(ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in COPY_FILES:
        shutil.copy2(ROOT / name, staging / name)
    manifest = json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    manifest["version"] = version
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_mcpb(args: list) -> None:
    npx = shutil.which("npx")
    if npx is None:
        raise BuildError("npx not found on PATH; install Node.js 20 or later to build the bundle")
    result = subprocess.run([npx, "-y", MCPB_CLI, *args], check=False)
    if result.returncode != 0:
        raise BuildError(f"mcpb {args[0]} failed with exit code {result.returncode}")


def check_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    for name in names:
        parts = set(Path(name).parts)
        if parts & FORBIDDEN_PARTS or name.endswith(".pyc") or name.startswith(FORBIDDEN_PREFIXES):
            raise BuildError(f"bundle contains a file that must not ship: {name}")
    for required in ("manifest.json", "src/executor/mcp_server.py", "pyproject.toml"):
        if required not in names:
            raise BuildError(f"bundle is missing {required}")
    size = path.stat().st_size
    if size >= MAX_BYTES:
        raise BuildError(f"bundle is {size} bytes, over the 5 MB limit")


def build() -> Path:
    version = pyproject_version((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    output = dist / f"agentic-video-{version}.mcpb"
    staging = Path(tempfile.mkdtemp(prefix="agentic-video-mcpb-"))
    try:
        bundle_dir = staging / "bundle"
        bundle_dir.mkdir()
        manifest_path = stage(bundle_dir, version)
        run_mcpb(["validate", str(manifest_path)])
        run_mcpb(["pack", str(bundle_dir), str(output)])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not output.is_file():
        raise BuildError(f"pack did not write {output}")
    check_archive(output)
    return output


def main() -> int:
    try:
        output = build()
    except BuildError as exc:
        print(f"build_mcpb: {exc}", file=sys.stderr)
        return 1
    print(f"built {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
