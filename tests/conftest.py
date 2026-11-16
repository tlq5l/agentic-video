import pytest

from src.transcript import asr


@pytest.fixture(autouse=True)
def _tiny_default(monkeypatch, request):
    # Pin the no-model default to tiny so tests never start real whisper.cpp jobs on machines that have turbo.
    if not request.node.name.startswith("test_default_model_is_turbo"):
        monkeypatch.setattr(asr, "default_model_size", lambda: "tiny")


@pytest.fixture(autouse=True)
def _no_semantic_worker_left_behind():
    # A search_transcript call in any test may start the EmbeddingGemma worker (a large model process); never let it outlive the test.
    from src.transcript import semantic

    semantic._WORKER.reopen()
    yield
    semantic._WORKER.close()  # stays closed until the next test, so a still-running index build cannot start a worker
    semantic._MEMO.clear()


@pytest.fixture(autouse=True)
def _no_cloud_asr_settings(monkeypatch, tmp_path):
    # A cloud-asr.json on the owner's Mac, or a cloud switch in the environment, must never put the suite in cloud mode.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("AGENTIC_VIDEO_CLOUD_ASR", raising=False)
