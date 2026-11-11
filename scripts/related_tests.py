"""Run only the tests that import the src/ files you changed, leaving out model and slow tests.

Usage:
  python scripts/related_tests.py                 # changes vs HEAD, plus untracked files
  python scripts/related_tests.py --base main     # changes vs another ref
  python scripts/related_tests.py --files src/retrieval/visual.py   # pick the files by hand
  python scripts/related_tests.py --list          # print the chosen tests, run nothing
  python scripts/related_tests.py --jobs 1        # one pytest process instead of the default 4

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
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SRC = ROOT / "src"
MARKER_EXPRESSION = "not model and not slow"
SELECT_ALL = {"tests/conftest.py", "conftest.py", "pytest.ini", "pyproject.toml"}
DEFAULT_JOBS = 4
SHARD_ENV = "RELATED_TESTS_SHARD"  # "<index>/<count>", read by the pytest plugin below
REPORT_ENV = "RELATED_TESTS_REPORT"  # a file path; the plugin writes {"collected": [...], "kept": [...]} there


def pytest_collection_modifyitems(config, items):
    """pytest plugin hook (loaded with `-p related_tests`): keep this worker's share of the collected tests.

    The share is a crc32 of each nodeid modulo the worker count. crc32 is the same in every process
    (the built-in hash() is not), so workers agree on the split even when they collect in different orders.
    The parent checks the report from each worker (see shard_problem)."""
    spec = os.environ.get(SHARD_ENV)
    if not spec:
        return
    index, count = (int(part) for part in spec.split("/"))
    collected = [item.nodeid for item in items]
    kept, dropped = [], []
    for item in items:
        (kept if zlib.crc32(item.nodeid.encode()) % count == index else dropped).append(item)
    report = os.environ.get(REPORT_ENV)
    if report:
        with open(report, "w", encoding="utf-8") as handle:
            json.dump({"collected": collected, "kept": [item.nodeid for item in kept]}, handle)
    config.hook.pytest_deselected(items=dropped)
    items[:] = kept


def worker_exit_code(codes):
    """One exit code for all workers. 0 (passed) and 5 (no tests) are fine; any other code fails the run.

    A worker killed by a signal has a negative code (-N for signal N); it is named on its own line.
    The first positive failure code is returned, else 1. All 5 gives 5; a mix of 0 and 5 gives 0."""
    for index, code in enumerate(codes):
        if code < 0:
            print(f"related_tests: worker {index + 1} died with signal {-code}")
    failed = [code for code in codes if code not in (0, 5)]
    if failed:
        positive = [code for code in failed if code > 0]
        return positive[0] if positive else 1
    return 5 if all(code == 5 for code in codes) else 0


def read_report(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def shard_problem(reports, codes):
    """Why the workers' shares are not one exact split of what they all collected, or None when they are.

    Each worker must have collected the same nodeids, and the kept lists must be pairwise disjoint
    with a union equal to the collected set."""
    missing = [index + 1 for index, report in enumerate(reports) if report is None]
    if missing:
        if any(code not in (0, 5) for code in codes):
            return None  # a worker that died is failed by its exit code (worker_exit_code)
        return f"related_tests: shard mismatch: no collection report from worker(s) {missing}"
    base = set(reports[0]["collected"])
    for index, report in enumerate(reports):
        seen = set(report["collected"])
        if seen != base:
            return (
                f"related_tests: shard mismatch: worker {index + 1} collected {len(seen)} tests, "
                f"worker 1 collected {len(base)}; the workers did not collect the same tests"
            )
    kept = [nodeid for report in reports for nodeid in report["kept"]]
    union = set(kept)
    duplicated = len(kept) - len(union)
    dropped = len(base - union)
    unknown = len(union - base)
    if duplicated or dropped or unknown:
        return (
            f"related_tests: shard mismatch: of {len(base)} collected tests, {duplicated} run by more than one "
            f"worker, {dropped} run by none, {unknown} not collected"
        )
    return None


def run_sharded(chosen, jobs):
    """Run the chosen test files in `jobs` pytest processes at once and return one exit code.

    Exit codes go through worker_exit_code. The workers' collection reports must split the tests exactly
    (shard_problem); a mismatch prints a line and turns a 0 or 5 result into 1."""
    with tempfile.TemporaryDirectory() as scratch:
        workers = []
        for index in range(jobs):
            report = os.path.join(scratch, f"report-{index}.json")
            env = dict(os.environ, **{SHARD_ENV: f"{index}/{jobs}", REPORT_ENV: report})
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).parent), env.get("PYTHONPATH")]))
            command = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "related_tests", "-m", MARKER_EXPRESSION, *chosen]
            log = tempfile.TemporaryFile(mode="w+")  # a file, not a pipe: a full pipe would stall a worker
            workers.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for index, (proc, log) in enumerate(workers):
            code = proc.wait()
            log.seek(0)
            if jobs > 1:
                print(f"--- worker {index + 1}/{jobs} ---")
            print(log.read(), end="")
            log.close()
            codes.append(code)
        returncode = worker_exit_code(codes)
        reports = [read_report(os.path.join(scratch, f"report-{index}.json")) for index in range(jobs)]
    problem = shard_problem(reports, codes)
    if problem:
        print(problem)
        if returncode in (0, 5):
            returncode = 1
    return returncode


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
        resolved = Path(name.replace("\\", "/")).resolve()
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
    affected = changed_modules | {name for name, path in modules.items() if src_imports(path) & changed_modules}
    test_files = sorted(TESTS.glob("test_*.py"))
    chosen = {p for p in test_files if p.relative_to(ROOT).as_posix() in changed}
    named = [Path(f).name for f in changed if f.startswith("src/") and not f.endswith(".py")]
    for test in test_files:
        imports = src_imports(test)
        if imports & affected or any(name in test.read_text(encoding="utf-8") for name in named):
            chosen.add(test)
    return sorted(chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD", help="git ref to diff against (default HEAD)")
    parser.add_argument("--files", nargs="+", help="use these changed files instead of git's (relative to the current folder, or absolute)")
    parser.add_argument("--list", action="store_true", help="print the chosen tests and do not run them")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help=f"pytest processes that share the tests (default {DEFAULT_JOBS}; 1 runs them in one)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

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
    returncode = run_sharded(chosen, args.jobs)
    print(f"related_tests: {time.monotonic() - started:.1f} s, exit {returncode}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
