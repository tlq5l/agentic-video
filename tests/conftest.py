import pytest

from src.transcript import asr


@pytest.fixture(autouse=True)
def _tiny_default(monkeypatch, request):
    # Pin the no-model default to tiny so tests never start real whisper.cpp jobs on machines that have turbo.
    if not request.node.name.startswith("test_default_model_is_turbo"):
        monkeypatch.setattr(asr, "default_model_size", lambda: "tiny")
