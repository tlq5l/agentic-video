"""Run only the tests that import the src/ files you changed, leaving out model and slow tests.

Usage:
  python scripts/related_tests.py                 # changes vs HEAD, plus untracked files
  python scripts/related_tests.py --base main     # changes vs another ref
  python scripts/related_tests.py --files src/retrieval/visual.py   # pick the files by hand
  python scripts/related_tests.py --list          # print the chosen tests, run nothing

A test is chosen when it imports a changed module directly, or imports a src module that
imports the changed one (one level). Changing tests/conftest.py, pytest.ini or pyproject.toml
selects every test file. Non-Python files under src/ select the tests whose source names them.
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
    return sorted(set(git_lines("diff", "--name-only", base, "--") + git_lines("ls-files", "--others", "--exclude-standard")))


def module_name(path):
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def src_imports(path):
    """Dotted names of the src modules this file imports (absolute imports; `from pkg import mod` counts mod)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name == "src" or name.startswith("src.")}


def select_tests(changed):
    if SELECT_ALL & set(changed):
        return sorted(TESTS.glob("test_*.py"))

    modules = {module_name(p): p for p in SRC.rglob("*.py")}
    module_of_file = {p.relative_to(ROOT).as_posix(): name for name, p in modules.items()}
    changed_modules = {module_of_file[f] for f in changed if f in module_of_file}
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
    parser.add_argument("--files", nargs="+", help="use these changed files instead of git's")
    parser.add_argument("--list", action="store_true", help="print the chosen tests and do not run them")
    args = parser.parse_args(argv)

    changed = [f.replace("\\", "/") for f in args.files] if args.files else changed_files(args.base)
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
