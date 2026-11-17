"""Put the tiny Whisper model that the transcribe tests run (tests/test_shadow_cf_asr_e2e.py) into this machine's Hugging
Face cache, so those tests pass with HF_HUB_OFFLINE=1 too. The repo's setup step (.away-ship.json) runs it. Safe to rerun:
a cached copy whose files all match their expected size and sha256 is used as it is, with no network call; a cached copy
with a file missing, short or changed (a download cut off half way) is fetched again. Prints how it got the model, its
size and its folder; exits 1 when the files still do not match after a fresh download.

Seed (WHISPER_TINY_SEED, a folder holding the four files): when the cache has no good copy and the seed matches, the seed
is copied into the cache with no network call. When the cache and the seed are both missing or bad, the Hub is used, and
a good result is copied into the seed folder so the next box starts from it.

When the Hub answers 429 (too many requests), the model-info call or a file download is tried again after 15, 30, 60 and
120 s (WHISPER_TINY_RETRY_WAITS, comma-separated seconds, replaces them; empty means the default). Each try is the whole
fetch, model-info then the four files; the Hub client makes its own quick retries inside a download first. setup exits 1
when the Hub still answers 429 after the last wait. The 429 is seen on the model-info call, which a cached copy cannot
hide. A wait too long to sleep stops setup with one line naming WHISPER_TINY_RETRY_WAITS. Other download errors fail at
once."""
import hashlib
import math
import os
import shutil
import sys
import time
from pathlib import Path

REPO = "Systran/faster-whisper-tiny"
# The revision the expected files below were read from (a box download, 2026-11-17).
REVISION = "d90ca5fe260221311c53c58e660288d3deb8d356"
# The files faster_whisper loads for "tiny": size and sha256 at that revision.
EXPECTED = {
    "config.json": (2249, "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8"),
    "model.bin": (75538270, "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"),
    "tokenizer.json": (2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
    "vocabulary.txt": (459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
}
DEFAULT_WAITS = "15,30,60,120"


RAW_WAITS = os.environ.get("WHISPER_TINY_RETRY_WAITS", "").strip() or DEFAULT_WAITS


def retry_waits():
    """Seconds to wait after each 429: WHISPER_TINY_RETRY_WAITS, or the default when it is unset or empty. Empty items
    (a stray comma) are skipped. A value that is not seconds 0 or more prints one warning line on stderr and the default."""
    raw = RAW_WAITS
    try:
        waits = [float(item) for item in raw.split(",") if item.strip()]
        if not waits or any(not math.isfinite(wait) or wait < 0 for wait in waits):
            raise ValueError(raw)
    except ValueError:
        print(f"WHISPER_TINY_RETRY_WAITS={raw!r} is not comma-separated seconds 0 or more; using {DEFAULT_WAITS}", file=sys.stderr, flush=True)
        waits = [float(wait) for wait in DEFAULT_WAITS.split(",")]
    return waits


WAITS = retry_waits()


def rate_limited(error):
    """True when the Hub answered 429 (too many requests) anywhere under `error`: huggingface_hub 2.4 raises it inside a
    LocalEntryNotFoundError when nothing is cached (round-17 breaker box log, r17/break-m2.log)."""
    for _ in range(10):
        if error is None:
            return False
        if getattr(getattr(error, "response", None), "status_code", None) == 429:
            return True
        error = error.__cause__ or error.__context__
    return False


def seed_folder():
    """The seed folder named by WHISPER_TINY_SEED, or None when it is unset."""
    raw = os.environ.get("WHISPER_TINY_SEED", "").strip()
    return Path(raw) if raw else None


def bad_files(path):
    """What is wrong with the expected files under `path`: missing, a size that differs, or a sha256 that differs."""
    bad = []
    for name, (size, digest) in EXPECTED.items():
        file = path / name
        if not file.is_file():
            bad.append(f"{name} missing")
        elif file.stat().st_size != size:
            bad.append(f"{name} {file.stat().st_size} bytes, expected {size}")
        elif hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            bad.append(f"{name} sha256 differs")
    return bad


def install_seed(seed):
    """Copy the seed's four files into the Hugging Face cache as the snapshot of REVISION, and point main at it."""
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / f"models--{REPO.replace('/', '--')}"
    snapshot = root / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        target = snapshot / name
        if target.is_symlink() or target.exists():
            target.unlink()  # a link into the blobs folder is replaced, never written through
        shutil.copyfile(seed / name, target)
    (root / "refs").mkdir(exist_ok=True)
    (root / "refs" / "main").write_text(REVISION)


def fill_seed(seed, path):
    """Copy the four verified files from `path` into the seed folder, each through a .part file that is then renamed."""
    seed.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        part = seed / (name + ".part")
        shutil.copyfile(path / name, part)
        os.replace(part, seed / name)


def cached_model():
    """(snapshot folder, what is wrong with it) for the copy in the Hugging Face cache. The folder is None when nothing is
    cached."""
    from faster_whisper.utils import download_model

    try:
        path = Path(download_model("tiny", local_files_only=True))
        return path, bad_files(path)
    except Exception as error:  # not cached
        return None, [f"not cached ({type(error).__name__})"]


class SetupError(Exception):
    """Setup cannot go on. The message is the one line it prints."""


def sleep_for(wait):
    """Sleep `wait` seconds. A wait too long for time.sleep (1e308) raises SetupError naming the setting."""
    try:
        time.sleep(wait)
    except OverflowError:
        raise SetupError(f"WHISPER_TINY_RETRY_WAITS={RAW_WAITS!r}: a wait of {wait:g} s is too long to sleep; use shorter waits") from None


def fetch():
    """Download the 4 files afresh once the Hub answers. One loop of tries covers the model-info call and the download, and
    they share the WAITS: a 429 on either waits the next of WAITS and runs the whole fetch again. Returns the snapshot
    folder, or None when the Hub still answers 429 after the last wait. Other errors are raised at once."""
    from huggingface_hub import model_info, snapshot_download

    for wait in [*WAITS, None]:
        try:
            model_info(REPO, revision="main")
            return Path(snapshot_download(REPO, allow_patterns=list(EXPECTED), force_download=True))
        except Exception as error:
            if not rate_limited(error):
                raise
            if wait is None:
                return None
            print(f"WHISPER_TINY hub answered 429 (too many requests), next try in {wait:g} s", flush=True)
            sleep_for(wait)


def main():
    os.environ["HF_HUB_OFFLINE"] = "0"  # before huggingface_hub is imported: setup is the one step allowed to fetch
    seed = seed_folder()
    path, bad = cached_model()
    how = "cached"
    if bad and seed is not None and not bad_files(seed):
        install_seed(seed)
        path, bad = cached_model()
        how = "seed"
    if bad:
        print("WHISPER_TINY fetching: " + "; ".join(bad), flush=True)
        try:
            path = fetch()
        except SetupError as error:
            print(error, file=sys.stderr, flush=True)
            return 1
        if path is None:
            print(f"WHISPER_TINY hub still answered 429 after {len(WAITS) + 1} tries", flush=True)
            return 1
        how, bad = "downloaded", bad_files(path)
    if bad:
        print("WHISPER_TINY still wrong after a fresh download: " + "; ".join(bad), flush=True)
        return 1
    if seed is not None and bad_files(seed):
        fill_seed(seed, path)
        print(f"WHISPER_TINY seed filled at {seed}", flush=True)
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f"WHISPER_TINY {how} {size} bytes {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
