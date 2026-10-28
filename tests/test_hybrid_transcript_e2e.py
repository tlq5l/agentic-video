"""End to end: search_transcript through the MCP entry point, hybrid by default, on the fixture video."""
import shutil
import time
from pathlib import Path

import pytest

from src.core.config import Config
from src.executor.mcp_server import handle_call
from src.transcript import semantic

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PARAPHRASE = "a very bright explosion of a star"  # the speech says "Supernova optical flash detected"


@pytest.fixture
def video(tmp_path, monkeypatch):
    for suffix in (".mp4", ".srt"):
        shutil.copy(FIXTURES / f"synthetic_bench_60s{suffix}", tmp_path / f"clip{suffix}")
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(semantic, "_PERMANENT", None)
    semantic._WORKER.close()  # an earlier test must not leave a worker or in-memory vectors behind
    semantic._WORKER.reopen()
    semantic._MEMO.clear()
    yield tmp_path / "clip.mp4"
    semantic._WORKER.close()  # leave no worker process behind
    semantic._MEMO.clear()


def _call(video, query, **args):
    response = handle_call("search_transcript", {"source": str(video), "query": query, **args})
    assert response["ok"], response
    return response["data"]


def _search(video, query):
    for _ in range(12):  # the first call on a new video may answer while the index is still building
        result = _call(video, query)
        if "still building" not in result.get("note", ""):
            return result
        time.sleep(5)
    return result


def test_hybrid_without_the_model_falls_back_to_keywords_and_says_why(video, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_VIDEO_EG2_DIR", str(tmp_path / "no-such-model"))
    result = _call(video, "Supernova")
    assert result["hits"][0]["match"] == "exact" and "Supernova" in result["hits"][0]["text"]
    assert "Semantic search is off" in result["note"]
    assert semantic._WORKER.proc is None  # no worker was started
    assert not list((tmp_path / "cache").rglob("transcript_eg2*"))  # nothing cached


def test_hybrid_with_a_missing_interpreter_still_answers_with_keywords_every_time(video, tmp_path, monkeypatch):
    model = tmp_path / "model"
    (model / "EmbeddingGemma2Text.mlpackage").mkdir(parents=True)  # present, so only the interpreter is wrong
    monkeypatch.setenv("AGENTIC_VIDEO_EG2_DIR", str(model))
    monkeypatch.setenv("AGENTIC_VIDEO_EG2_PYTHON", str(tmp_path / "gone" / "python"))
    for _ in range(2):
        result = _call(video, "Supernova")
        assert result["hits"][0]["match"] == "exact" and "Supernova" in result["hits"][0]["text"]
        assert "could not be started" in result["note"]
    assert semantic._WORKER.proc is None


def test_an_unknown_mode_is_rejected_with_a_clear_error(video):
    with pytest.raises(ValueError, match=r"search_transcript: argument 'mode' is not one of the allowed values \(got 'semantic'\); expected one of: hybrid, lexical"):
        handle_call("search_transcript", {"source": str(video), "query": "Supernova", "mode": "semantic"})


@pytest.mark.skipif(not (semantic.model_dir() / "EmbeddingGemma2Text.mlpackage").exists(),
                    reason="EmbeddingGemma 2 Core ML model not set up (AGENTIC_VIDEO_EG2_DIR)")
def test_hybrid_finds_a_paraphrase_and_pins_exact_hits_first(video):
    lexical = _call(video, PARAPHRASE, mode="lexical")
    assert not any("Supernova" in hit["text"] for hit in lexical["hits"])

    result = _search(video, PARAPHRASE)
    assert any(hit["match"] == "semantic" and "Supernova" in hit["text"] for hit in result["hits"]), result
    assert "No keyword match" in result["note"]

    exact = _search(video, "Supernova")
    assert exact["hits"][0]["match"] == "exact" and "Supernova" in exact["hits"][0]["text"]
    assert list((Path(Config.BASE_CACHE_DIR)).rglob("transcript_eg2.npz"))  # vectors cached for the next call


def test_exact_phrase_is_listed_first_even_after_five_partial_matches(video, tmp_path, monkeypatch):
    """Five earlier lines hold only 'dog'; the one line with the whole phrase 'the dog' must still come first."""
    monkeypatch.setenv("AGENTIC_VIDEO_EG2_DIR", str(tmp_path / "no-such-model"))  # keyword path only: no model load
    lines = ["a dog ran", "one dog barked", "dog food", "my dog sleeps", "dog days", "the dog ate the bone"]
    srt = "".join(f"{i}\n00:00:{2 * i:02d},000 --> 00:00:{2 * i + 1:02d},000\n{text}\n\n" for i, text in enumerate(lines))
    video.with_suffix(".srt").write_text(srt, encoding="utf-8")
    hits = _call(video, "the dog")["hits"]
    assert hits[0]["match"] == "exact" and hits[0]["text"] == "the dog ate the bone"
