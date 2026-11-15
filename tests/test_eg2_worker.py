"""A request without an id must not kill the eg2 worker."""
import json

from src.transcript import eg2_worker


class _FakeVectors:
    def tobytes(self):
        return b"\x00\x01"


class _FakeEmbedder:
    def embed(self, texts, prompt):
        return _FakeVectors()


class _FakeReplies:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def flush(self):
        pass


def test_serve_survives_request_without_id(tmp_path, monkeypatch):
    payload = (
        json.dumps({"texts": ["x"], "prompt": "Document"}) + "\n"
        + json.dumps({"id": 2, "texts": ["x"], "prompt": "Document"}) + "\n"
    ).encode("utf-8")
    chunks = iter([payload, b""])
    replies = _FakeReplies()

    monkeypatch.setattr(eg2_worker.os, "dup", lambda fd: 99)
    monkeypatch.setattr(eg2_worker.os, "dup2", lambda src, dst: None)
    monkeypatch.setattr(eg2_worker.os, "fdopen", lambda fd, mode: replies)
    monkeypatch.setattr(eg2_worker.os, "read", lambda fd, size: next(chunks))
    monkeypatch.setattr(eg2_worker.select, "select", lambda r, w, x, t: ([0], [], []))

    eg2_worker._serve(tmp_path, [_FakeEmbedder()])

    decoded = [json.loads(line) for line in replies.lines]
    assert len(decoded) == 2
    assert decoded[0]["id"] is None and "KeyError" in decoded[0]["error"]
    assert decoded[1]["id"] == 2 and "vectors" in decoded[1]


def test_serve_survives_malformed_and_blank_lines(tmp_path, monkeypatch):
    class _Stub:
        def embed(self, *args, **kwargs):
            return _FakeVectors()

    good = json.dumps({"id": 7, "texts": ["x"], "prompt": "Document"}).encode("utf-8") + b"\n"
    payload = b"not json\n\n" + good
    chunks = iter([payload, b""])
    replies = _FakeReplies()

    monkeypatch.setattr(eg2_worker.os, "dup", lambda fd: 99)
    monkeypatch.setattr(eg2_worker.os, "dup2", lambda src, dst: None)
    monkeypatch.setattr(eg2_worker.os, "fdopen", lambda fd, mode: replies)
    monkeypatch.setattr(eg2_worker.os, "read", lambda fd, size: next(chunks))
    monkeypatch.setattr(eg2_worker.select, "select", lambda r, w, x, t: ([0], [], []))

    eg2_worker._serve(tmp_path, [_Stub()])

    decoded = [json.loads(line) for line in replies.lines]
    assert len(decoded) == 3
    assert "error" in decoded[0] and decoded[0]["id"] is None
    assert "error" in decoded[1] and decoded[1]["id"] is None
    assert decoded[2]["id"] == 7 and "vectors" in decoded[2]
