"""Pytest integration for the 12 Adversarial Failure Injection Modes."""

import pytest
from evals.adversarial_runner import AdversarialFailureHarness, SANDBOX_DIR

pytestmark = pytest.mark.slow  # the module-scoped harness builds its fixtures in about 24 s

@pytest.fixture(scope="module")
def harness():
    h = AdversarialFailureHarness(SANDBOX_DIR)
    h.prepare_fixtures()
    return h

def test_adversarial_01_missing_ffmpeg(harness):
    res = harness.test_01_missing_ffmpeg_binary()
    assert res.status == "PASS", f"Test 1 failed: {res.details}"

def test_adversarial_02_corrupted_container(harness):
    res = harness.test_02_corrupted_video_container()
    assert res.status == "PASS", f"Test 2 failed: {res.details}"

def test_adversarial_03_vfr_seeking(harness):
    res = harness.test_03_vfr_seeking_accuracy()
    assert res.status == "PASS", f"Test 3 failed: {res.details}"

def test_adversarial_04_silent_video(harness):
    res = harness.test_04_silent_video()
    assert res.status == "PASS", f"Test 4 failed: {res.details}"

def test_adversarial_05_no_subtitles_asr(harness):
    res = harness.test_05_video_no_subtitles_asr_fallback()
    assert res.status == "PASS", f"Test 5 failed: {res.details}"

def test_adversarial_06_asr_timeout(harness):
    res = harness.test_06_asr_timeout_handling()
    assert res.status == "PASS", f"Test 6 failed: {res.details}"

def test_adversarial_07_mcp_crash_recovery(harness):
    res = harness.test_07_mcp_crash_recovery()
    assert res.status == "PASS", f"Test 7 failed: {res.details}"

def test_adversarial_08_daemon_restart(harness):
    res = harness.test_08_daemon_restart()
    assert res.status == "PASS", f"Test 8 failed: {res.details}"

def test_adversarial_09_cached_artifact_deleted(harness):
    res = harness.test_09_cached_artifact_deleted()
    assert res.status == "PASS", f"Test 9 failed: {res.details}"

def test_adversarial_10_very_long_video_seek(harness):
    res = harness.test_10_very_long_video_seek()
    assert res.status == "PASS", f"Test 10 failed: {res.details}"

def test_adversarial_11_unicode_and_spaces(harness):
    res = harness.test_11_spaces_and_unicode_path()
    assert res.status == "PASS", f"Test 11 failed: {res.details}"

def test_adversarial_12_concurrent_queries(harness):
    res = harness.test_12_concurrent_queries_read_only_invariant()
    assert res.status == "PASS", f"Test 12 failed: {res.details}"
