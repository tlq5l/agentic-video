# Development checks

Timings and red-then-green evidence from round 24, run on Cloudflare boxes with `box run` against main at 88ef50f. Box seconds
count the command only, not dependency install or the copy to the box.

## Timing at 88ef50f

- One file: `bash scripts/verify.sh --files src/core/cache.py`: 155.8 box seconds, exit 0. The selector picks 53 test files
  (`scripts/related_tests.py --files src/core/cache.py --list`); its 4 workers reported 770 passed and 25 skipped.
- Full committed check (the `check` in `.away-ship.json`, which excludes model and slow tests): 582.9 box seconds, exit 0,
  946 passed, 36 skipped.

## Fetch budget cases without a cached model

`tests/test_fetch_test_models_e2e.py -k "retry_wait_budget or zero_wait"` (8 cases) with an empty `HF_HOME` and no model cached:

- Before (88ef50f): 8 skipped, 0 passed, so the zero-skip assertion fails (`8 skipped, 23 deselected in 0.33s`).
- After: 8 passed, 0 skipped, 24 deselected. The cases run against a local stand-in Hub that serves synthetic files.
- New case `test_synthetic_same_size_corruption_fails_the_sha256_check` (same size, one byte changed): 1 passed. The real sha256 check rejects the download.

## Verify controls with and without Git

- `tests/test_verify_e2e.py`: 20 passed on a box. Each runtime control runs in a copy with no `.git` and in a copy that is a Git
  checkout with one local commit.
- Planted mutant: the direct-selection line in `scripts/related_tests.py` replaced with `chosen = set()`. The four import-free
  cases in `tests/test_verify_e2e.py` then fail (`4 failed, 16 deselected`), so the mutant is caught.
