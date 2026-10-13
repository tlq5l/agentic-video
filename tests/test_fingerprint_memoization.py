"""Focused tests for per-cache fingerprint memoization."""
import builtins
import hashlib

from src.core.cache import MediaCache


def _count_binary_opens(monkeypatch):
    real_open = builtins.open
    calls = []

    def spy(file, mode="r", *args, **kwargs):
        if mode == "rb":
            calls.append(file)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    return calls


def test_repeated_fingerprint_hashes_bytes_once_per_cache(tmp_path, monkeypatch):
    source = tmp_path / "video.bin"
    source.write_bytes(b"content" * 1000)
    cache = MediaCache(tmp_path / "cache")
    opens = _count_binary_opens(monkeypatch)

    first = cache.compute_fingerprint(str(source))
    second = cache.compute_fingerprint(str(source))

    assert first == second
    assert len(opens) == 1


def test_stat_change_invalidates_fingerprint_memo(tmp_path, monkeypatch):
    source = tmp_path / "video.bin"
    source.write_bytes(b"A" * 4096)
    cache = MediaCache(tmp_path / "cache")
    opens = _count_binary_opens(monkeypatch)

    first = cache.compute_fingerprint(str(source))
    source.write_bytes(b"B" * 4096)
    second = cache.compute_fingerprint(str(source))

    assert first != second
    assert len(opens) == 2


def test_fingerprint_memo_isolated_between_cache_instances(tmp_path, monkeypatch):
    source = tmp_path / "video.bin"
    source.write_bytes(b"content")
    first_cache = MediaCache(tmp_path / "cache-a")
    second_cache = MediaCache(tmp_path / "cache-b")
    opens = _count_binary_opens(monkeypatch)

    first = first_cache.compute_fingerprint(str(source))
    second = second_cache.compute_fingerprint(str(source))

    assert first == second
    assert len(opens) == 2


def test_fingerprint_exact_digest_and_missing_path_behavior(tmp_path):
    payload = b"exact bytes\x00" * 200
    source = tmp_path / "video.bin"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()[:20]

    cache = MediaCache(tmp_path / "cache")
    assert cache.compute_fingerprint(str(source)) == expected
    # Preserve the historical class-level static invocation as well.
    assert MediaCache.compute_fingerprint(str(source)) == expected

    missing = str(tmp_path / "missing.bin")
    expected_missing = hashlib.sha256(missing.encode("utf-8")).hexdigest()[:20]
    assert cache.compute_fingerprint(missing) == expected_missing
    assert cache.compute_fingerprint(missing) == expected_missing

    source.write_bytes(payload)
    assert cache.compute_fingerprint(missing) == expected_missing