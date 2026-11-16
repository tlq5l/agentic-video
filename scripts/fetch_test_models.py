"""Put the tiny Whisper model that the transcribe tests run (tests/test_shadow_cf_asr_e2e.py) into this machine's Hugging
Face cache, so those tests pass with HF_HUB_OFFLINE=1 too. The repo's setup step (.away-ship.json) runs it. Safe to rerun:
a model already cached is used as it is, with no network call. Prints how it got the model, its size and its folder."""
import os
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "0"  # before huggingface_hub is imported: setup is the one step allowed to fetch
    from faster_whisper.utils import download_model

    try:
        path, how = Path(download_model("tiny", local_files_only=True)), "cached"
    except Exception:
        path, how = Path(download_model("tiny")), "downloaded"
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f"WHISPER_TINY {how} {size} bytes {path}")


if __name__ == "__main__":
    main()
