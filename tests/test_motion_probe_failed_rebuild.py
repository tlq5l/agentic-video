"""A failed rebuild of a motion disk entry must leave the old entry readable or absent, never meta.json over missing arrays."""
import os
from pathlib import Path

import pytest

from src.core.config import Config
from src.media import motion_probe

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "motion" / "v1-256.gif"


def test_failed_rebuild_keeps_the_old_entry_readable_or_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    motion_probe.clear_memory_store()
    clip, _origin = motion_probe._clip(str(FIXTURE))
    entry = motion_probe._entry(clip.fingerprint)
    motion_probe._write_disk(entry, clip)  # the complete entry the rebuild replaces
    real_replace = os.replace

    def fail_on_meta(src, dst, *args, **kwargs):
        if Path(dst).name == "meta.json":
            raise OSError("injected: meta.json placement failed")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(motion_probe.os, "replace", fail_on_meta)
    with pytest.raises(OSError, match="injected"):
        motion_probe._write_disk(entry, clip)
    monkeypatch.setattr(motion_probe.os, "replace", real_replace)
    try:
        reread = motion_probe._read_disk(entry, clip.fingerprint)
    except Exception as exc:
        raise AssertionError(f"entry still has meta.json but its arrays are gone: {type(exc).__name__}: {exc}") from exc
    assert reread is not None or not (entry / "meta.json").exists()
