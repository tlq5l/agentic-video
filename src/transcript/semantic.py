"""Semantic half of hybrid transcript search: EmbeddingGemma 2 over 20 s chunks of the transcript.

The model runs in a child process (`eg2_worker.py`, Core ML on the Neural Engine) that starts on
the first search, loads its model once per function (the 256-token packed one; a query over 256 tokens
also loads the 512-token one, see docs/features/transcribe.md), serves every call in this process, and
exits after at most 10 idle minutes or when this process ends. Chunk vectors are cached next to the
transcript cache. Any failure of this half (missing interpreter, dead worker, bad reply) is raised as
SemanticUnavailable, so the keyword hits and a note always come back.

Setup (optional; without it search stays keyword-only and says so):
  AGENTIC_VIDEO_EG2_DIR     folder with EmbeddingGemma2Text.mlpackage, embeddings.bf16, tokenizer.json,
                            config.json (default ~/.cache/agentic-video/models/embeddinggemma2-coreml)
  AGENTIC_VIDEO_EG2_PYTHON  interpreter with coremltools, tokenizers and numpy (default: this one)
"""
import atexit
import base64
import bisect
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.cache import MediaCache

MODEL_ID = "embeddinggemma-2-coreml-768"
CHUNK_SEC = 20.0
STRIDE_SEC = 10.0
GAP_SEC = 20.0
MIN_SCORE = 0.70           # cosine below this is not a match by meaning; see docs/features/transcribe.md
EMBED_BATCH = 128
INDEX_WAIT_SEC = 30.0       # a search waits this long for a missing index, then answers keyword-only
QUERY_WAIT_SEC = 30.0       # first query pays the model load (about 20 s)
CLOSE_WAIT_SEC = 5.0        # close() gives the worker this long to exit on its own after its input closes
TERMINATE_WAIT_SEC = 40.0   # after SIGTERM: a model load (about 25 s on a busy Mac) must finish before the exit is normal
JOB_TIMEOUT_SEC = 1800.0    # first-ever load on a machine compiles for the Neural Engine
_WORKER_PATH = Path(__file__).with_name("eg2_worker.py")


class SemanticUnavailable(Exception):
    """Semantic search cannot run. `permanent` is True when retrying in this process is pointless."""

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


def model_dir() -> Path:
    default = Path.home() / ".cache" / "agentic-video" / "models" / "embeddinggemma2-coreml"
    return Path(os.environ.get("AGENTIC_VIDEO_EG2_DIR") or default).expanduser()


class _Worker:
    """The child process: spawned on first use, one request at a time, one retry if it dies."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()  # guards proc and closed, never held while waiting on the child
        self.proc: Optional[subprocess.Popen] = None
        self.closed = False
        self.buffer = b""
        self.next_id = 0
        atexit.register(self.close)

    def close(self) -> None:
        """Shut down for good: stop the child and refuse to start another (an index build still running in a
        background thread must not load a second model after the server is gone). `reopen()` undoes this."""
        with self.state_lock:
            self.closed = True
        self._stop_child()

    def reopen(self) -> None:
        """Allow a new child after `close()` (tests share one process-wide worker)."""
        with self.state_lock:
            self.closed = False

    def _stop_child(self) -> None:
        """Stop the child so it exits normally. The worker never gets SIGKILL: Core ML leaves a model folder
        in TMPDIR for a worker that is killed. A worker still alive after the waits is left to exit on its own."""
        with self.state_lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()  # the worker sees end of input and exits by itself
        except OSError:
            pass
        # Wait 5 s for that, then SIGTERM (the worker handles it and exits normally, but only once a model load or
        # batch in progress finishes: up to about 30 s), and wait again. No further signal.
        for stop, wait in ((None, CLOSE_WAIT_SEC), (proc.terminate, TERMINATE_WAIT_SEC)):
            if stop is not None:
                stop()
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        proc.stdout.close()

    def _start(self) -> subprocess.Popen:
        python = os.environ.get("AGENTIC_VIDEO_EG2_PYTHON") or sys.executable
        with self.state_lock:
            if self.closed:
                raise SemanticUnavailable("the embedding worker was shut down")
            self.proc = proc = subprocess.Popen(
                [python, str(_WORKER_PATH), str(model_dir())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buffer = b""
        # Blocks in wait(), no polling: reaps the child the moment it exits (idle exit) so no zombie lingers.
        threading.Thread(target=proc.wait, daemon=True).start()
        return proc

    def _read_reply(self, proc: subprocess.Popen, want_id: int, deadline: float) -> dict:
        fd = proc.stdout.fileno()
        while True:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                reply = json.loads(line)
                if reply.get("id") == want_id:
                    return reply  # replies to abandoned earlier requests are skipped
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([fd], [], [], wait)[0]:
                raise TimeoutError
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise EOFError
            self.buffer += chunk

    def embed(self, texts: List[str], prompt: str, timeout: float) -> np.ndarray:
        """Vectors for the texts. Any failure (spawn, pipe, reply, decoding) is raised as SemanticUnavailable."""
        if not self.lock.acquire(timeout=timeout):
            raise SemanticUnavailable("the embedding worker is busy with another job")
        try:
            return self._embed_locked(texts, prompt, timeout)
        except SemanticUnavailable:
            raise
        except OSError as exc:  # the interpreter is missing or not executable: retrying cannot help
            self._stop_child()
            raise SemanticUnavailable(
                f"the embedding worker could not be started ({type(exc).__name__}: {exc}; "
                "check AGENTIC_VIDEO_EG2_PYTHON and restart the server)", permanent=True) from None
        except Exception as exc:  # a malformed reply, a decoding error: drop this worker, keyword search still answers
            self._stop_child()
            raise SemanticUnavailable(f"the embedding worker failed ({type(exc).__name__}: {exc})") from None
        finally:
            self.lock.release()

    def _embed_locked(self, texts: List[str], prompt: str, timeout: float) -> np.ndarray:
        deadline = time.monotonic() + timeout
        for attempt in (1, 2):
            proc = self.proc
            if proc is None or proc.poll() is not None:
                proc = self._start()
            self.next_id += 1
            request = {"id": self.next_id, "prompt": prompt, "texts": texts}
            try:
                proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                reply = self._read_reply(proc, self.next_id, deadline)
            except TimeoutError:
                raise SemanticUnavailable(f"the embedding worker did not answer within {timeout:.0f} s") from None
            except (BrokenPipeError, EOFError):
                self._stop_child()
                if attempt == 2:
                    raise SemanticUnavailable("the embedding worker stopped unexpectedly") from None
                continue
            if "error" in reply:
                raise SemanticUnavailable(
                    f"the embedding worker failed ({reply['error']})",
                    permanent=reply["error"].startswith(("ModuleNotFoundError", "FileNotFoundError", "ImportError")))
            return np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float32).reshape(len(texts), -1)
        raise SemanticUnavailable("the embedding worker stopped unexpectedly")


_WORKER = _Worker()
_PERMANENT: Optional[str] = None
_MEMO: "OrderedDict[str, np.ndarray]" = OrderedDict()
_JOBS: Dict[str, "_Job"] = {}
_JOBS_LOCK = threading.Lock()


class _Job:
    """One background index build; searches for the same video join it instead of starting another."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[SemanticUnavailable] = None
        self.vectors: Optional[np.ndarray] = None
        self.started = time.monotonic()


def make_chunks(segments: List[dict]) -> List[dict]:
    """Overlapping 20 s windows every 10 s; a chunk's text is the segments that start inside its window."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: float(s["start_sec"]))
    starts = [float(s["start_sec"]) for s in segs]
    end_of_video = max(float(s["end_sec"]) for s in segs)
    chunks, t = [], 0.0
    while t < end_of_video:
        inside = segs[bisect.bisect_left(starts, t):bisect.bisect_left(starts, t + CHUNK_SEC)]
        if inside:
            chunks.append({
                "start": t, "end": t + CHUNK_SEC, "segment_id": inside[0]["id"],
                "text": " ".join(s["text"].strip() for s in inside),
                "suspect": next((s["suspect"] for s in inside if s.get("suspect")), None),
            })
        t += STRIDE_SEC
    return chunks


def _index_key(chunks: List[dict]) -> str:
    digest = hashlib.sha1(f"{MODEL_ID}|{CHUNK_SEC}|{STRIDE_SEC}|".encode())
    for chunk in chunks:
        digest.update(chunk["text"].encode("utf-8") + b"\0")
    return digest.hexdigest()


def _cache_file(fingerprint: str) -> Optional[Path]:
    try:
        return MediaCache().get_entry_dir(fingerprint) / "transcript_eg2.npz"
    except Exception:  # CacheWriteError: the cache is an optimisation, never a failure
        return None


def _remember(key: str, vectors: np.ndarray) -> np.ndarray:
    _MEMO[key] = vectors
    while len(_MEMO) > 4:
        _MEMO.popitem(last=False)
    return vectors


def _load_cached(fingerprint: str, key: str) -> Optional[np.ndarray]:
    if key in _MEMO:
        return _MEMO[key]
    path = _cache_file(fingerprint)
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) == key:
                return _remember(key, data["vectors"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def _save_cached(fingerprint: str, key: str, vectors: np.ndarray) -> None:
    _remember(key, vectors)
    path = _cache_file(fingerprint)
    if path is None:
        return
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, vectors=vectors, key=np.array(key))
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)


def _build(fingerprint: str, key: str, chunks: List[dict], job: _Job) -> None:
    try:
        parts = [_WORKER.embed([c["text"] for c in chunks[i:i + EMBED_BATCH]], "Document", JOB_TIMEOUT_SEC)
                 for i in range(0, len(chunks), EMBED_BATCH)]
        job.vectors = np.concatenate(parts)
        _save_cached(fingerprint, key, job.vectors)
    except SemanticUnavailable as exc:
        job.error = exc
    except Exception as exc:  # never leave the waiting search without an answer
        job.error = SemanticUnavailable(f"building the semantic index failed ({type(exc).__name__}: {exc})")
    finally:
        job.done.set()
        with _JOBS_LOCK:
            _JOBS.pop(key, None)


def _chunk_vectors(fingerprint: str, chunks: List[dict]) -> np.ndarray:
    key = _index_key(chunks)
    vectors = _load_cached(fingerprint, key)
    if vectors is not None:
        return vectors
    with _JOBS_LOCK:
        job = _JOBS.get(key)
        if job is None:
            job = _JOBS[key] = _Job()
            threading.Thread(target=_build, args=(fingerprint, key, chunks, job), daemon=True).start()
    if not job.done.wait(INDEX_WAIT_SEC):
        raise SemanticUnavailable(
            f"the semantic index for this video is still building ({time.monotonic() - job.started:.0f} s so far); "
            "repeat the call in about a minute")
    if job.error:
        raise job.error
    if job.vectors is None:
        raise SemanticUnavailable("building the semantic index ended without vectors")
    return job.vectors


def search(fingerprint: str, segments: List[dict], query: str, top_k: int,
           time_range: Optional[Tuple[float, float]] = None) -> List[dict]:
    """Closest chunks to the query: dicts with start, end, text, segment_id, suspect, score (cosine).

    Raises SemanticUnavailable when semantic search cannot answer right now.
    """
    global _PERMANENT
    if _PERMANENT:
        raise SemanticUnavailable(_PERMANENT, permanent=True)
    if sys.platform != "darwin":
        _PERMANENT = "semantic search needs macOS: the EmbeddingGemma 2 model runs as a Core ML model"
        raise SemanticUnavailable(_PERMANENT, permanent=True)
    if not (model_dir() / "EmbeddingGemma2Text.mlpackage").exists():
        _PERMANENT = f"the EmbeddingGemma 2 Core ML model was not found in {model_dir()} (set AGENTIC_VIDEO_EG2_DIR)"
        raise SemanticUnavailable(_PERMANENT, permanent=True)
    try:
        chunks = make_chunks(segments)
        if not chunks:
            return []
        vectors = _chunk_vectors(fingerprint, chunks)
        query_vector = _WORKER.embed([query], "SearchQuery", QUERY_WAIT_SEC)[0]
        if vectors.shape != (len(chunks), query_vector.shape[0]):
            raise SemanticUnavailable("the cached semantic index does not match this transcript")
    except SemanticUnavailable as exc:
        if exc.permanent:
            _PERMANENT = str(exc)
        raise
    except Exception as exc:  # whatever else goes wrong in this half must not cost the keyword answer
        raise SemanticUnavailable(f"semantic search failed ({type(exc).__name__}: {exc})") from None
    scores = vectors @ query_vector
    t_start, t_end = time_range if time_range else (-1.0, float("inf"))
    picked: List[int] = []
    for i in np.argsort(-scores)[:200]:
        chunk = chunks[i]
        if scores[i] < MIN_SCORE:
            break
        if chunk["end"] < t_start or chunk["start"] > t_end:
            continue
        if all(abs(chunk["start"] - chunks[j]["start"]) >= GAP_SEC for j in picked):
            picked.append(int(i))
            if len(picked) == top_k:
                break
    return [dict(chunks[i], score=float(scores[i])) for i in picked]
