"""scripts/fetch_test_models.py, the merge recipe's setup step, run as setup runs it on a copy of this machine's cached tiny
Whisper model, with the Hugging Face endpoint pointed at a closed port so any download attempt fails (no network)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_test_models.py"


def _copy_cache(tmp_path):
    """A copy of the cached Systran/faster-whisper-tiny folder under a new HF_HOME, its snapshot files copied as plain files
    (the hub keeps their blobs outside the model folder); returns (HF_HOME, the copy's model.bin)."""
    from faster_whisper.utils import download_model

    try:
        snapshot = Path(download_model("tiny", local_files_only=True))
    except Exception:
        pytest.skip("tiny Whisper model not cached here; the recipe's setup step fetches it")
    home = tmp_path / "hf"
    model_dir = home / "hub" / snapshot.parents[1].name
    shutil.copytree(snapshot.parents[1], model_dir)
    return home, model_dir / "snapshots" / snapshot.name / "model.bin"


def _setup(home):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
    env.update(HF_HOME=str(home), HF_ENDPOINT="http://127.0.0.1:9")
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300)


def test_a_complete_cached_model_passes_setup_with_no_network(tmp_path):
    """Cache intact, no network: setup uses the cached copy (no download tried, or the closed port would fail it)."""
    home, _ = _copy_cache(tmp_path)
    run = _setup(home)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.startswith("WHISPER_TINY cached 78203619 bytes")


def test_a_half_downloaded_model_does_not_count_as_cached(tmp_path):
    """model.bin cut to half its size (a download cut off half way): setup does not report it cached; it tries a fresh
    download, which the closed port fails, so setup fails (round-17 probe: before this check it printed
    "WHISPER_TINY cached 40434484 bytes" and exited 0)."""
    home, model = _copy_cache(tmp_path)
    os.chmod(model, 0o644)
    with open(model, "r+b") as handle:
        handle.truncate(75538270 // 2)
    run = _setup(home)
    assert "WHISPER_TINY cached" not in run.stdout
    assert "WHISPER_TINY fetching: model.bin 37769135 bytes, expected 75538270" in run.stdout
    assert run.returncode != 0
