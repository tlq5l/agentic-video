"""Put the tiny Whisper model that the transcribe tests run (tests/test_shadow_cf_asr_e2e.py) into this machine's Hugging
Face cache, so those tests pass with HF_HUB_OFFLINE=1 too. The repo's setup step (.away-ship.json) runs it. Safe to rerun:
a cached copy whose files all match their expected size and sha256 is used as it is, with no network call; a cached copy
with a file missing, short or changed (a download cut off half way) is fetched again. Prints how it got the model, its
size and its folder; exits 1 when the files still do not match after a fresh download."""
import hashlib
import os
import sys
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
    from huggingface_hub import snapshot_download

    try:
        path = Path(download_model("tiny", local_files_only=True))
        bad = bad_files(path)
    except Exception as error:  # not cached, or cached with a file missing
        bad = [f"not cached ({type(error).__name__})"]
    how = "cached"
    if bad:
        print("WHISPER_TINY fetching: " + "; ".join(bad), flush=True)
        path = Path(snapshot_download(REPO, allow_patterns=list(EXPECTED), force_download=True))
        how, bad = "downloaded", bad_files(path)
    if bad:
        print("WHISPER_TINY still wrong after a fresh download: " + "; ".join(bad), flush=True)
        return 1
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f"WHISPER_TINY {how} {size} bytes {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
