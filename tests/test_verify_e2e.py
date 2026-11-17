"""scripts/verify.sh and scripts/related_tests.py on a miniature repo: the real shell wrapper, selector, ruff and pytest run,
in a copy with no .git (how a box copy looks) and in a copy that is a real Git checkout with one local commit. Bad inputs
fail with a message, never a traceback or a silent pass."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
# tests/test_plain.py is written by the tests that need it: it imports nothing, so only its own path selects it.
PLAIN_PASS = "def test_plain():\n    assert True\n"
PLAIN_FAIL = "def test_plain():\n    assert False\n"
GITIGNORE = ".venv/\n__pycache__/\n"
GIT_MODES = pytest.mark.parametrize("git", [False, True], ids=["gitless", "git"])


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    return env


def _git(root, *args):
    """A git command in the copy with a local-only identity, no global hooks and no signing."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    command = ["git", "-c", "user.name=r24 test", "-c", "user.email=r24@example.invalid",
               "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args]
    return subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)


def _make_repo(tmp_path, git=False):
    """The miniature repo under tmp_path: the real scripts copied in, and a .venv/bin/python that runs this interpreter
    (verify.sh looks for it). gitless: no Git metadata above it. git: a Git checkout with one local commit of everything
    but .venv and __pycache__, which .gitignore leaves out."""
    root = tmp_path / ("git-copy" if git else "copy")
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    for name in SCRIPTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes((REPO / name).read_bytes())
    (root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!/bin/sh\nexec {sys.executable!r} \"$@\"\n", encoding="utf-8")
    python.chmod(0o755)
    if git:
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "initial")
    return root


def _verify(root, tmp_path, *args):
    return subprocess.run(["bash", "scripts/verify.sh", *args], cwd=root, env=_env(tmp_path), capture_output=True, text=True, timeout=600)


def _related(root, tmp_path, *args):
    """scripts/related_tests.py run directly (the copy's own .venv interpreter), for its --list output."""
    command = [str(root / ".venv" / "bin" / "python"), "scripts/related_tests.py", *args]
    return subprocess.run(command, cwd=root, env=_env(tmp_path), capture_output=True, text=True, timeout=600)


def _chosen(out):
    """The test files the selector printed, one "  tests/..." line each."""
    return [line.strip() for line in out.splitlines() if line.startswith("  tests/")]


@GIT_MODES
def test_files_runs_only_the_affected_eligible_test(tmp_path, git):
    """--files src/util.py: ruff passes, and only tests/test_util.py runs. The unrelated failing sentinel, and the model
    and slow tests that import util, are left out."""
    root = _make_repo(tmp_path, git=git)
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "1 passed" in out, out
    assert "failed" not in out and "error" not in out.lower(), out
    assert "test_unrelated_sentinel" not in out and "test_model_double" not in out and "test_slow_double" not in out, out


def test_files_selects_the_same_exact_tests_with_or_without_git(tmp_path):
    """--files src/util.py names the same three test files in a copy with .git and in one without: the two tests that
    import util (the model and slow ones are chosen but not run) and tests/test_util.py. Nothing else."""
    chosen = {}
    for git in (False, True):
        root = _make_repo(tmp_path, git=git)
        run = _verify(root, tmp_path, "--files", "src/util.py")
        out = run.stdout + run.stderr
        assert run.returncode == 0, out
        chosen[git] = _chosen(out)
    expected = ["tests/test_model_import.py", "tests/test_slow_import.py", "tests/test_util.py"]
    assert chosen[False] == chosen[True] == expected, chosen


@GIT_MODES
def test_files_lint_failure_stops_before_pytest(tmp_path, git):
    """An unused import in the changed file: ruff reports F401 and fails, and pytest never starts."""
    root = _make_repo(tmp_path, git=git)
    (root / "src" / "util.py").write_text("import os\n\n\ndef double(x):\n    return 2 * x\n", encoding="utf-8")
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "F401" in out, out
    assert "passed" not in out and "failed" not in out, out


@GIT_MODES
def test_files_selected_test_failure_is_a_failure(tmp_path, git):
    """The changed function breaks the selected test: the run exits non-zero and says so."""
    root = _make_repo(tmp_path, git=git)
    (root / "src" / "util.py").write_text("def double(x):\n    return 3 * x\n", encoding="utf-8")
    run = _verify(root, tmp_path, "--files", "src/util.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "1 failed" in out, out


@GIT_MODES
def test_files_file_with_no_importing_test_runs_nothing_and_passes(tmp_path, git):
    """A changed module that no test imports: zero related tests, nothing runs, exit 0."""
    root = _make_repo(tmp_path, git=git)
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


@GIT_MODES
def test_files_missing_explicit_file_is_a_spelling_error(tmp_path, git):
    """A --files path that does not exist and that git has no record of: a clear error, no traceback."""
    root = _make_repo(tmp_path, git=git)
    run = _verify(root, tmp_path, "--files", "src/utill.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "does not exist" in out and "Traceback" not in out, out


@GIT_MODES
@pytest.mark.parametrize("failing", [False, True], ids=["passes", "fails"])
def test_import_free_changed_test_runs_by_its_own_name(tmp_path, git, failing):
    """tests/test_plain.py imports nothing, so no source file selects it. Named by its own path it runs, and a failing
    body fails the verify run. A selector that drops direct selection makes the passing case fail (nothing runs)."""
    root = _make_repo(tmp_path, git=git)
    (root / "tests" / "test_plain.py").write_text(PLAIN_FAIL if failing else PLAIN_PASS, encoding="utf-8")
    run = _verify(root, tmp_path, "--files", "tests/test_plain.py")
    out = run.stdout + run.stderr
    if failing:
        assert run.returncode != 0 and "1 failed" in out, out
    else:
        assert run.returncode == 0 and "1 passed" in out, out
    assert "Traceback" not in out, out


def test_git_default_diff_selects_what_explicit_files_select(tmp_path):
    """In a Git checkout with no --files: a modified tracked source file and an untracked import-free test are the
    changes, and the selector lists the same tests as --files naming those two."""
    root = _make_repo(tmp_path, git=True)
    (root / "src" / "util.py").write_text("def double(x):\n    return 2 * x  # touched\n", encoding="utf-8")
    (root / "tests" / "test_plain.py").write_text(PLAIN_PASS, encoding="utf-8")
    default = _related(root, tmp_path, "--list")
    explicit = _related(root, tmp_path, "--files", "src/util.py", "tests/test_plain.py", "--list")
    assert default.returncode == 0 and explicit.returncode == 0, default.stdout + default.stderr + explicit.stderr
    assert default.stdout == explicit.stdout, (default.stdout, explicit.stdout)
    assert "changed: 2 file(s)" in default.stdout and "  tests/test_plain.py" in default.stdout, default.stdout


@pytest.mark.parametrize("staged", [False, True], ids=["tracked", "staged"])
def test_git_deletions_are_accepted_through_history(tmp_path, staged):
    """src/orphan.py deleted from the folder (tracked deletion) or removed with git rm (staged deletion): git's history
    still has the file, so --files accepts it. Nothing imports orphan.py, so nothing runs and the exit is 0."""
    base = tmp_path / ("staged" if staged else "tracked")
    base.mkdir()
    root = _make_repo(base, git=True)
    if staged:
        _git(root, "rm", "-q", "src/orphan.py")
    else:
        (root / "src" / "orphan.py").unlink()
    run = _verify(root, base, "--files", "src/orphan.py")
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "changed: 1 file(s); related tests: 0" in out and "nothing to run" in out, out
    assert "Traceback" not in out, out


def test_gitless_missing_path_is_refused_without_a_traceback(tmp_path):
    """The same missing path in a copy with no .git: nothing records the deletion, so --files refuses it with a message."""
    root = _make_repo(tmp_path)
    (root / "src" / "orphan.py").unlink()
    run = _verify(root, tmp_path, "--files", "src/orphan.py")
    out = run.stdout + run.stderr
    assert run.returncode != 0, out
    assert "does not exist" in out and "Traceback" not in out, out
