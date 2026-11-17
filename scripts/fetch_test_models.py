"""Put the tiny Whisper model that the transcribe tests run (tests/test_shadow_cf_asr_e2e.py) into this machine's Hugging
Face cache, so those tests pass with HF_HUB_OFFLINE=1 too. The repo's setup step (.away-ship.json) runs it. Safe to rerun:
a cached copy whose files all match their expected size and sha256 is used as it is, with no network call; a cached copy
with a file missing, short or changed (a download cut off half way) is fetched again. Prints how it got the model, its
size and its folder; exits 1 when the files still do not match after a fresh download. When the Hub answers 429 (too
many requests), the download is tried again after 15, 30, 60 and 120 s (WHISPER_TINY_RETRY_WAITS, comma-separated
seconds, replaces them), and setup exits 1 when the Hub still answers 429 after the last wait. Other download errors
fail at once. Not covered: a 429 while a broken copy is cached, where the Hub hands back the cached copy without an
error and setup exits 1 at once (round-18 box probe); it needs a cut-off download and a 429 together."""
import hashlib
import os
import sys
import time
from pathlib import Path

REPO = "Systran/faster-whisper-tiny"
# The files faster_whisper loads for "tiny": size and sha256 at revision d90ca5fe260221311c53c58e660288d3deb8d356 (read
# from a box download, 2026-11-17).
EXPECTED = {
    "config.json": (2249, "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8"),
    "model.bin": (75538270, "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"),
    "tokenizer.json": (2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
    "vocabulary.txt": (459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
}

WAITS = [float(wait) for wait in os.environ.get("WHISPER_TINY_RETRY_WAITS", "15,30,60,120").split(",")]


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


def fetch():
    """snapshot_download of the 4 files with force_download, tried again after each wait in WAITS while the Hub answers
    429; returns the snapshot folder, or None when the Hub still answers 429 after the last wait."""
    from huggingface_hub import snapshot_download

    for wait in [*WAITS, None]:
        try:
            return Path(snapshot_download(REPO, allow_patterns=list(EXPECTED), force_download=True))
        except Exception as error:
            if not rate_limited(error):
                raise
            if wait is None:
                return None
            print(f"WHISPER_TINY hub answered 429 (too many requests), next try in {wait:g} s", flush=True)
            time.sleep(wait)


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


def main():
    os.environ["HF_HUB_OFFLINE"] = "0"  # before huggingface_hub is imported: setup is the one step allowed to fetch
    from faster_whisper.utils import download_model

    try:
        path = Path(download_model("tiny", local_files_only=True))
        bad = bad_files(path)
    except Exception as error:  # not cached, or cached with a file missing
        bad = [f"not cached ({type(error).__name__})"]
    how = "cached"
    if bad:
        print("WHISPER_TINY fetching: " + "; ".join(bad), flush=True)
        path = fetch()
        if path is None:
            print(f"WHISPER_TINY hub still answered 429 after {len(WAITS) + 1} tries", flush=True)
            return 1
        how, bad = "downloaded", bad_files(path)
    if bad:
        print("WHISPER_TINY still wrong after a fresh download: " + "; ".join(bad), flush=True)
        return 1
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f"WHISPER_TINY {how} {size} bytes {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())