#!/bin/sh
# Inner check for agents: ruff on the changed Python files, then the tests that import them.
# Usage: scripts/verify.sh [base-ref]   (default HEAD; untracked files count as changed)
set -eu
base="${1:-HEAD}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python=python3
[ -x .venv/bin/python ] && python=.venv/bin/python

changed=$( { git diff --name-only --diff-filter=d "$base" --; git ls-files --others --exclude-standard; } | grep '\.py$' | sort -u || true)
if [ -n "$changed" ]; then
  uvx ruff check $changed
fi
exec "$python" scripts/related_tests.py --base "$base"
