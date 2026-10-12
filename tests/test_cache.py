"""Tests for media cache and fail-safe bounds enforcement."""
import pytest
from pathlib import Path
from src.core.cache import MediaCache
from src.core.errors import ExtractionBoundsError


def test_compute_fingerprint(tmp_path):
    # Test fingerprint generation
    test_file = tmp_path / "dummy_video.mp4"
    test_file.write_bytes(b"0" * 150000)
    
    cache = MediaCache(base_dir=tmp_path / "cache")
    fp1 = cache.compute_fingerprint(str(test_file))
    assert len(fp1) == 20
    
    # Identical file should have identical fingerprint
    fp2 = cache.compute_fingerprint(str(test_file))
    assert fp1 == fp2
    
    # Modifying file should change fingerprint
    test_file.write_bytes(b"1" * 150000)
    fp3 = cache.compute_fingerprint(str(test_file))
    assert fp1 != fp3


def test_compute_fingerprint_detects_middle_byte_change(tmp_path):
    test_file = tmp_path / "same_size.mp4"
    payload = bytearray(b"A" * 200_000)
    test_file.write_bytes(payload)
    cache = MediaCache(base_dir=tmp_path / "cache")
    first = cache.compute_fingerprint(str(test_file))

    payload[len(payload) // 2] = ord("B")
    test_file.write_bytes(payload)
    second = cache.compute_fingerprint(str(test_file))

    assert first != second


def test_atomic_json_cache(tmp_path):
    cache = MediaCache(base_dir=tmp_path / "cache")
    fp = "test_fp_12345"
    
    data = {"name": "video", "fps": 25.0, "duration": 10.5}
    saved_path = cache.save_json(fp, "meta", data)
    assert saved_path.exists()
    
    loaded = cache.load_json(fp, "meta")
    assert loaded == data
    
    # Non-existent entry returns None
    assert cache.load_json(fp, "non_existent") is None


def test_validate_bounds_enforcement():
    # Valid bounds
    MediaCache.validate_extraction_bounds(fps=0.2, duration_sec=60.0)
    MediaCache.validate_extraction_bounds(fps=4.0, duration_sec=10.0)
    
    # Invalid FPS
    with pytest.raises(ExtractionBoundsError):
        MediaCache.validate_extraction_bounds(fps=35.0, duration_sec=5.0)
        
    with pytest.raises(ExtractionBoundsError):
        MediaCache.validate_extraction_bounds(fps=-1.0, duration_sec=5.0)
        
    # Invalid duration
    with pytest.raises(ExtractionBoundsError):
        MediaCache.validate_extraction_bounds(fps=1.0, duration_sec=650.0)
        
    # Excessive total frames
    with pytest.raises(ExtractionBoundsError):
        MediaCache.validate_extraction_bounds(fps=10.0, duration_sec=20.0, max_frames=50)
