"""scripts/verify.sh and scripts/related_tests.py on a miniature repo with no .git (how a box copy looks): the real shell
wrapper, selector, ruff and pytest run, and bad inputs fail with a message, never a traceback or a silent pass."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("scripts/verify.sh", "scripts/related_tests.py")
FILES = {
    "src/__init__.py": "",
    "src/util.py": "def double(x):\n    return 2 * x\n",
    "src/orphan.py": "VALUE = 1\n",
    "tests/test_util.py": "from src.util import double\n\n\ndef test_double():\n    assert double(2) == 4\n",
    "tests/test_model_import.py": "import pytest\n\nfrom src.util import double\n\npytestmark = pytest.mark.model\n\n\ndef test_model_double():\n    assert double(1) == 2\n",
    "tests/test_slow_import.py": "import pytest\n\nfrom src.util import double\n\npytestmark = pytest.mark.slow\n\n\ndef test_slow_double():\n    assert double(3) == 6\n",
    "tests/test_unrelated_sentinel.py": "def test_unrelated_sentinel():\n    assert False, 'an unrelated test ran'\n",
}


def _make_repo(tmp_path):
    """The miniature repo under tmp_path/copy: the real scripts copied in, a .venv/bin/python that runs this interpreter
    (verify.sh looks for it), and no Git metadata above it (GIT_CEILING_DIRECTORIES stops the search at tmp_path)."""
    root = tmp_path / "copy"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    for name in SCRIPTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, root / name)
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!/bin/sh\nexec {sys.executable!r} \"$@\"\n", encoding="utf-8")
    python.chmod(0o755)
    return root


def _verify(root, tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    return subprocess.run(["bash", "scripts/verify.sh", *args], cwd=root, env=env, capture_output=True, text=True, timeout=600)


def test_gitless_files_runs_only_the_affected_eligible_test(tmp_path):
    """--files src/util.py in a copy with no .git: ruff passes, and only tests/test_util.py runs. The unrelated failing
    sentinel, and the model and slow tests that import util, are left out."""
    root = _make_repo(tmp_path)
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "1 passed" in out, out
    assert "failed" not in out and "error" not in out.lower(), out
    assert "test_unrelated_sentinel" not in out and "test_model_double" not in out and "test_slow_double" not in out, out


def test_gitless_lint_failure_stops_before_pytest(tmp_path):
    """An unused import in the changed file: ruff reports F401 and fails, and pytest never starts."""
    root = _make_repo(tmp_path)
    (root / "src" / "util.py").write_text("import os\n\n\ndef double(x):\n    return 2 * x\n", encoding="utf-8")
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "F401" in out, out
    assert "passed" not in out and "failed" not in out, out


def test_gitless_selected_test_failure_is_a_failure(tmp_path):
    """The changed function breaks the selected test: the run exits non-zero and says so."""
    root = _make_repo(tmp_path)
    (root / "src" / "util.py").write_text("def double(x):\n    return 3 * x\n", encoding="utf-8")
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "1 failed" in out, out


def test_gitless_file_with_no_importing_test_runs_nothing_and_passes(tmp_path):
    """A changed module that no test imports: zero related tests, nothing runs, exit 0."""
    root = _make_repo(tmp_path)
    run = _verify(root, tmp_path, "--files", "src/orphan.py")
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "related tests: 0" in out and "nothing to run" in out, out


def test_gitless_without_files_says_to_name_them(tmp_path):
    """No arguments in a copy with no Git metadata: no traceback, and the message says to pass --files."""
    root = _make_repo(tmp_path)
    run = _verify(root, tmp_path)
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "--files" in out and "Traceback" not in out, out


def test_gitless_missing_explicit_file_is_a_spelling_error(tmp_path):
    """A --files path that does not exist, in a copy with no Git to check it against: a clear error, no traceback."""
    root = _make_repo(tmp_path)
    run = _verify(root, tmp_path, "--files", "src/utill.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "does not exist" in out and "Traceback" not in out, out
