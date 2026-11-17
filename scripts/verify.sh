#!/bin/sh
# Inner check for agents: ruff on the changed Python files, then the tests that import them (model and slow tests left out).
# Usage: scripts/verify.sh [base-ref]           changed files vs base-ref (default HEAD, untracked files count) in a Git checkout
#        scripts/verify.sh --files <paths...>   the changed files are the ones named: the way to run it in a copy with no .git
# Both go through scripts/related_tests.py --lint.
set -eu
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python=python3
[ -x .venv/bin/python ] && python=.venv/bin/python

if [ "${1:-}" = "--files" ]; then
  shift
  if [ $# -eq 0 ]; then
    echo "verify.sh: --files needs the changed paths, e.g. scripts/verify.sh --files src/core/cache.py" >&2
    exit 2
  fi
  exec "$python" scripts/related_tests.py --lint --files "$@"
fi
exec "$python" scripts/related_tests.py --lint --base "${1:-HEAD}"
