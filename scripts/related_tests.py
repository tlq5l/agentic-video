"""Run only the tests that import the src/ files you changed, leaving out model and slow tests.

Usage:
  python scripts/related_tests.py                 # changes vs HEAD, plus untracked files
  python scripts/related_tests.py --base main     # changes vs another ref
  python scripts/related_tests.py --files src/retrieval/visual.py   # pick the files by hand
  python scripts/related_tests.py --list          # print the chosen tests, run nothing

A test is chosen when it imports a changed module directly, or imports a src module that
imports the changed one (one level). Importing a module also runs the __init__ files of its
parent packages, so a test importing src.core.cache is chosen by a change to src/core/__init__.py.
A rename counts as two changes: the old path and the new one. Changing tests/conftest.py,
pytest.ini or pyproject.toml selects every test file. Non-Python files under src/ select the
tests whose source names them.
--files takes paths relative to the current folder or absolute. Each must be a .py file, a file
under src/, or one of the config files above; a folder is refused, and so is any other path.
"""

import argparse
import ast
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SRC = ROOT / "src"
MARKER_EXPRESSION = "not model and not slow"
SELECT_ALL = {"tests/conftest.py", "conftest.py", "pytest.ini", "pyproject.toml"}


def git_lines(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def changed_files(base):
    tracked = git_lines("diff", "--no-renames", "--name-only", base, "--")
    return sorted(set(tracked + git_lines("ls-files", "--others", "--exclude-standard")))


def known_to_git(relative):
    """True for a path git tracks in the index or in HEAD, so a deleted file still counts as a change."""
    return bool(git_lines("ls-files", "--", relative) or git_lines("ls-tree", "--name-only", "HEAD", "--", relative))


def normalize_files(parser, names):
    """Repo-relative posix paths for --files; a path that cannot select a test is an error, not a silent pass."""
    normalized = set()
    for name in names:
        given = Path(name.replace("\\", "/"))
        resolved = (given if given.is_absolute() else Path.cwd() / given).resolve()
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            parser.error(f"--files: {name} is outside the repository")
        if resolved.is_dir():
            parser.error(f"--files: {name} is a folder; name the files in it")
        if not (relative.endswith(".py") or relative.startswith("src/") or relative in SELECT_ALL):
            parser.error(f"--files: {name} cannot select any test (use a .py file, a file under src/, or a config file)")
        if not resolved.exists() and not known_to_git(relative):
            parser.error(f"--files: {name} does not exist and git has no record of it (check the spelling)")
        normalized.add(relative)
    return sorted(normalized)


def module_of(relative_path):
    """"src/retrieval/visual.py" -> "src.retrieval.visual"; also works for a file that was deleted."""
    dotted = relative_path[: -len(".py")].replace("/", ".")
    return dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted


def dotted_prefixes(name):
    """"src.core.cache" -> {"src", "src.core", "src.core.cache"}."""
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)}


def src_imports(path):
    """Dotted names of the src modules this file imports, with their parent packages (absolute imports;
    `from pkg import mod` counts mod)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {
        prefix for name in found if name == "src" or name.startswith("src.") for prefix in dotted_prefixes(name)
    }


def select_tests(changed):
    if SELECT_ALL & set(changed):
        return sorted(TESTS.glob("test_*.py"))

    modules = {module_of(p.relative_to(ROOT).as_posix()): p for p in SRC.rglob("*.py")}
    changed_modules = {module_of(f) for f in changed if f.startswith("src/") and f.endswith(".py")}
    importers = {name: src_imports(path) for name, path in modules.items()}
    test_files = sorted(TESTS.glob("test_*.py"))
    chosen = {p for p in test_files if p.relative_to(ROOT).as_posix() in changed}

    for test in test_files:
        imports = src_imports(test)
        if imports & changed_modules:
            chosen.add(test)
            continue
        for name in imports & set(modules):
            if importers[name] & changed_modules:
                chosen.add(test)
                break

    named = [f for f in changed if f.startswith("src/") and not f.endswith(".py")]
    for test in test_files:
        text = test.read_text(encoding="utf-8")
        if any(Path(f).name in text for f in named):
            chosen.add(test)
    return sorted(chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD", help="git ref to diff against (default HEAD)")
    parser.add_argument("--files", nargs="+", help="use these changed files instead of git's (relative to the current folder, or absolute)")
    parser.add_argument("--list", action="store_true", help="print the chosen tests and do not run them")
    args = parser.parse_args(argv)

    changed = normalize_files(parser, args.files) if args.files else changed_files(args.base)
    chosen = [p.relative_to(ROOT).as_posix() for p in select_tests(changed)]
    print(f"changed: {len(changed)} file(s); related tests: {len(chosen)}")
    for name in chosen:
        print("  " + name)
    if args.list:
        return 0
    if not chosen:
        print("no test imports the changed files; nothing to run")
        return 0

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-m", MARKER_EXPRESSION, *chosen], cwd=ROOT
    )
    print(f"related_tests: {time.monotonic() - started:.1f} s, exit {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
