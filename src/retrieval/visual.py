"""Local CLIP retrieval: bounded chunks, transactional resume, nonblocking jobs.

No model downloads occur at tool runtime. Install requirements-visual.txt and
pre-fetch the model explicitly. One build and one search worker run per process.
"""
import heapq
import errno
import fcntl
import math
import re
import shutil
import subprocess
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.core.types import FrameInfo, format_timestamp
from src.media.contact_sheet import compose_contact_sheet
from src.media.frames import extract_single_frame

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
INDEX_NAME = f"visual-clip-b32-{MODEL_REVISION[:7]}-v3.sqlite3"
CHUNK_SEC = 60.0
BATCH_SIZE = 16
FFMPEG_THREADS = 6
TORCH_THREADS = 4
SCENE_THRESHOLD = 0.3
ORPHAN_TEMP_AGE_SEC = 300


class ClipEncoder:
    """CPU-only inference prevents unbounded unified-memory accelerator allocation."""
    def __init__(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Visual search needs requirements-visual.txt (torch and transformers).") from exc
        self.torch = torch
        torch.set_num_threads(TORCH_THREADS)
        try:
            self.processor = CLIPProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
            self.model = CLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).cpu().eval()
        except OSError as exc:
            raise RuntimeError(f"Cache {MODEL_ID} locally first; see README visual search setup. No runtime download attempted.") from exc

    def _normalize(self, features):
        values = features.float().cpu().numpy()
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def pil_images(self, images):
        """Embed in-memory RGB PIL images without a disk round trip."""
        inputs = self.processor(images=list(images), return_tensors="pt")
        with self.torch.inference_mode():
            return self._normalize(self.model.get_image_features(**inputs))

    def images(self, paths):
        from PIL import Image
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            return self.pil_images(images)
        finally:
            for image in images:
                image.close()

    def text(self, query):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with self.torch.inference_mode():
            return self._normalize(self.model.get_text_features(**inputs))[0]


_encoder = None
_encoder_lock = threading.Lock()

def get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = ClipEncoder()
    return _encoder


def _connect(engine, readonly=False):
    meta = engine.metadata
    path = engine._cache.get_entry_dir(meta.fingerprint) / INDEX_NAME
    if readonly:
        if not path.exists():
            return None
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
    conn = sqlite3.connect(path, timeout=1)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, until_sec REAL, count INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS frames (timestamp REAL PRIMARY KEY, path TEXT, width INTEGER, height INTEGER, embedding BLOB, thumbnail BLOB)")
    conn.execute("INSERT OR IGNORE INTO progress VALUES (1, 0, 0)")
    conn.commit()
    return conn


def _progress(conn, duration):
    until, count = conn.execute("SELECT until_sec, count FROM progress WHERE id=1").fetchone()
    return {"indexed_until_sec": until, "duration_sec": duration, "frames_indexed": count,
            "progress": min(1.0, until / duration), "model": MODEL_ID, "model_revision": MODEL_REVISION}


def _source_stamp(engine):
    stat = Path(engine.source_path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _check_source(engine, stamp):
    if _source_stamp(engine) != stamp:
        raise RuntimeError("Source changed during visual indexing/search; retry with the current source")


def _remove_orphaned_temp_dirs(entry, include_chunks):
    """Only the build-lock holder creates visual-chunk-* dirs, so under the lock every
    one is orphaned. Searches run unlocked, so only stale visual-search-* dirs go."""
    cutoff = time.time() - ORPHAN_TEMP_AGE_SEC
    for path in entry.iterdir():
        is_chunk = path.name.startswith("visual-chunk-")
        if not (is_chunk and include_chunks) and not path.name.startswith("visual-search-"):
            continue
        try:
            # Another worker may remove its own directory concurrently; that is not an error.
            if not path.is_symlink() and path.is_dir() and (is_chunk or path.stat().st_mtime < cutoff):
                shutil.rmtree(path)
        except FileNotFoundError:
            pass


def build_index(engine, report=lambda data: None):
    """Synchronous worker API, also usable by an ingest hook. MCP runs it off-thread."""
    stamp = _source_stamp(engine)
    duration = engine.metadata.duration_sec
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Visual indexing requires a finite positive video duration")
    entry = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    with (entry / f"{INDEX_NAME}.lock").open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            conn = None
            try:
                conn = _connect(engine, readonly=True)
                progress = _progress(conn, duration) if conn is not None else {}
            except sqlite3.OperationalError:
                progress = {}
            finally:
                if conn is not None:
                    conn.close()
            return {"status": "busy", "retry_after_seconds": 2,
                    "message": "Another process is building this visual index; retry shortly.", **progress}
        try:
            _remove_orphaned_temp_dirs(entry, include_chunks=True)
            return _build_locked(engine, stamp, duration, report)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_PTS = re.compile(r"pts_time:\s*([0-9.eE+-]+).*? s:(\d+)x(\d+)")


def sample_chunk(source_path, start_sec, end_sec, output_dir):
    """One FFmpeg pass: a frame at least every second plus every scene cut, at 224px.

    One decode replaces separate 1 fps sampling, scene detection and per-cut seeks."""
    select = f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,1)+gt(scene\\,{SCENE_THRESHOLD})"
    threads = str(FFMPEG_THREADS)
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-y", "-threads", threads, "-filter_threads", threads,
           "-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}", "-i", str(source_path),
           "-an", "-vf", f"select='{select}',scale='min(224,iw)':-2,showinfo",
           "-fps_mode", "vfr", "-q:v", "3", "-threads", threads, str(Path(output_dir) / "f_%05d.jpg")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg chunk sampling failed: {proc.stderr.strip()[-500:]}")
    infos = [m for line in proc.stderr.splitlines() if "Parsed_showinfo" in line
             for m in [_PTS.search(line)] if m]
    files = sorted(Path(output_dir).glob("f_*.jpg"))
    frames = []
    for index, (path, m) in enumerate(zip(files, infos)):
        t = round(min(end_sec, start_sec + float(m.group(1))), 3)
        frames.append(FrameInfo(index=index + 1, timestamp_sec=t, timecode=format_timestamp(t),
                                file_path=str(path), width=int(m.group(2)), height=int(m.group(3))))
    return frames


def _build_locked(engine, stamp, duration, report):
    conn = _connect(engine)
    cache_dir = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    pending = {}  # chunk start -> (temp dir, future): the next chunk decodes while this one is embedded

    def prefetch(pool, start):
        if start < duration and start not in pending:
            temp = tempfile.TemporaryDirectory(prefix="visual-chunk-", dir=cache_dir)
            end = min(duration, start + CHUNK_SEC)
            pending[start] = (temp, pool.submit(sample_chunk, engine.source_path, start_sec=start, end_sec=end,
                                                output_dir=temp.name))
    pool = ThreadPoolExecutor(1)
    try:
        report(_progress(conn, duration))
        while True:
            start = _progress(conn, duration)["indexed_until_sec"]
            if start >= duration:
                return {"status": "complete", **_progress(conn, duration)}
            end = min(duration, start + CHUNK_SEC)
            encoder = get_encoder()
            rows = []
            prefetch(pool, start)
            temp, future = pending.pop(start)
            with temp as temp_dir:
                frames = future.result()
                prefetch(pool, end)
                if not frames:
                    frames.append(extract_single_frame(engine.source_path, start,
                                                       output_path=Path(temp_dir) / "tail.jpg", width=224,
                                                       overlay_timestamp=False, cache=engine._cache, threads=FFMPEG_THREADS))
                frames.sort(key=lambda f: f.timestamp_sec)
                for offset in range(0, len(frames), BATCH_SIZE):
                    batch = frames[offset:offset+BATCH_SIZE]
                    vectors = encoder.images([f.file_path for f in batch])
                    if len(vectors) != len(batch) or not np.isfinite(vectors).all():
                        raise ValueError("Invalid visual embeddings")
                    rows.extend((frame.timestamp_sec, "", frame.width, frame.height,
                                 np.asarray(vector, dtype=np.float32).tobytes(), Path(frame.file_path).read_bytes())
                                for frame, vector in zip(batch, vectors))
            _check_source(engine, stamp)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany("INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?)", rows)
                conn.execute("UPDATE progress SET until_sec=?, count=(SELECT COUNT(*) FROM frames) WHERE id=1", (end,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            report(_progress(conn, duration))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for temp, _ in pending.values():
            temp.cleanup()
        conn.close()


def _validate_search(query, top_k, start, end):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("query must contain 1 to 2000 characters; describe the visible content in a short phrase, e.g. 'person in a red jacket'")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError(f"top_k must be an integer between 1 and 20 (got {top_k!r}); the search_visual tool clamps larger values to 20. "
                         "To count distinct appearances use count_occurrences")
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError(f"Invalid visual search time range (start_sec={start!r}, end_sec={end!r}); need 0 <= start_sec < end_sec, "
                         "or omit end_sec to search to the end of the video")


def _restore_thumbnail(conn, frame):
    """Materialize indexed pixels only for the lifetime of a search sheet."""
    from PIL import Image
    path = Path(frame.file_path)
    data = conn.execute("SELECT thumbnail FROM frames WHERE timestamp=?", (frame.timestamp_sec,)).fetchone()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    with Image.open(path) as image:
        image.load()


def search_index(engine, query, top_k=5, start_sec=0.0, end_sec=None):
    _validate_search(query, top_k, start_sec, end_sec)
    stamp = _source_stamp(engine)
    duration = engine.metadata.duration_sec
    end = duration if end_sec is None else end_sec
    if start_sec >= duration or end > duration or end <= start_sec:
        raise ValueError(f"Visual search range must lie within the video (duration {duration:g}s, got start_sec={start_sec:g}, "
                         f"end_sec={end:g}); use 0 <= start_sec < end_sec <= {duration:g}, or omit end_sec")
    # Every search, including early returns, clears stale temp dirs from killed searches.
    entry = engine._cache.get_entry_dir(engine.metadata.fingerprint)
    _remove_orphaned_temp_dirs(entry, include_chunks=False)
    try:
        conn = _connect(engine, readonly=True)
        if conn is None:
            return {"status": "index_required", "index_status": "missing",
                    "indexed_until_sec": 0.0, "duration_sec": duration, "frames_indexed": 0,
                    "progress": 0.0, "model": MODEL_ID, "model_revision": MODEL_REVISION,
                    "retry_after_seconds": 2,
                    "message": "Call build_visual_index, then repeat this search."}
    except sqlite3.OperationalError:
        return {"status": "busy", "retry_after_seconds": 2,
                "message": "Visual index is being initialized; retry shortly."}
    try:
        try:
            progress = _progress(conn, duration)
        except sqlite3.OperationalError:
            return {"status": "busy", "retry_after_seconds": 2,
                    "message": "Visual index is being initialized; retry shortly."}
        partial = progress["indexed_until_sec"] < duration
        state = "partial" if partial else "complete"
        pending = {"index_status": state, "retry_after_seconds": 2,
                   "message": "Later moments are not yet indexed; repeat this search after indexing advances."} if partial else {"index_status": state}
        if not progress["frames_indexed"]:
            return {"status": "index_required", **progress, **pending,
                    "message": "Call build_visual_index, then repeat this search."}
        # Hits and intervals never reach past indexed footage.
        covered_end = min(end, progress["indexed_until_sec"])
        vector = get_encoder().text(query)
        best = []
        cursor = conn.execute("SELECT timestamp,path,width,height,embedding FROM frames WHERE timestamp>=? AND timestamp<? ORDER BY timestamp", (start_sec, covered_end))
        while True:
            rows = cursor.fetchmany(256)
            if not rows:
                break
            matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
            scores = matrix @ vector
            for row, score in zip(rows, scores):
                item = (float(score), -row[0], row[:4])
                if len(best) < top_k:
                    heapq.heappush(best, item)
                elif item[:2] > best[0][:2]:
                    heapq.heapreplace(best, item)
        ranked = sorted(best, reverse=True)
        hits = [{"rank": i+1, "timestamp_sec": row[0], "timecode": format_timestamp(row[0]),
                 "score": score, "frame_badge": f"#{i+1}",
                 "timestamp_kind": "sample_target", "decoded_timestamp": None} for i, (score, _, row) in enumerate(ranked)]
        intervals = []
        for hit in sorted(hits, key=lambda h: h["timestamp_sec"]):
            t = hit["timestamp_sec"]
            if intervals and t - intervals[-1]["last_sample_sec"] <= 1.05:
                intervals[-1]["end_sec"] = min(covered_end, t + 1.0)
                intervals[-1]["last_sample_sec"] = t
                intervals[-1]["score"] = max(intervals[-1]["score"], hit["score"])
                intervals[-1]["ranks"].append(hit["rank"])
            else:
                intervals.append({"start_sec": t, "end_sec": min(covered_end, t+1.0), "last_sample_sec": t,
                                  "score": hit["score"], "ranks": [hit["rank"]]})
        sheet = None
        if ranked:
            import hashlib
            tag = hashlib.sha256(repr((query, top_k, start_sec, end, progress["indexed_until_sec"])).encode()).hexdigest()[:20]
            path = engine._cache.get_sheets_dir(engine.metadata.fingerprint) / f"visual_{tag}.jpg"
            with tempfile.TemporaryDirectory(prefix="visual-search-", dir=entry) as temp_dir:
                frames = [FrameInfo(i+1, row[0], format_timestamp(row[0]),
                                    str(Path(temp_dir) / f"rank_{i+1}.jpg"), row[2], row[3])
                          for i, (_, _, row) in enumerate(ranked)]
                for frame in frames:
                    _restore_thumbnail(conn, frame)
                sheet = compose_contact_sheet(frames, path, title="VISUAL SEARCH (rank order)").to_dict()
        _check_source(engine, stamp)
        return {"status": state, "hits": hits, "intervals": intervals, "contact_sheet": sheet, **pending,
                "index_complete": progress["indexed_until_sec"] >= duration, **progress,
                "note": "Cosine similarity is not confidence. Sample-based intervals are approximate; verify badges and inspect_window. Empty results do not establish absence."}
    finally:
        conn.close()


_jobs = OrderedDict()
_jobs_lock = threading.Lock()
_worker = None
_search_worker = None


def submit(engine, operation, **args):
    """Start/poll work without hashing, decoding or loading ML on the MCP thread.

    No unbounded queue: at most one active job, up to 32 small completed results.
    Restarting the server loses jobs but not committed index chunks.
    """
    global _worker, _search_worker
    if operation == "search":
        query, top_k = args["query"], args["top_k"]
        start, end = args["start_sec"], args["end_sec"]
        _validate_search(query, top_k, start, end)
    source = Path(engine.source_path).resolve()
    stat = source.stat()
    key = (str(engine._cache.base_dir.resolve()), str(source), stat.st_size, stat.st_mtime_ns,
           stat.st_ctime_ns, operation, tuple(sorted(args.items())))
    with _jobs_lock:
        if key in _jobs:
            result = dict(_jobs[key])
            # Consumed results can be retried after dependency installation or
            # after a partial index advances. Active jobs remain deduplicated.
            if result["status"] not in ("running", "queued"):
                del _jobs[key]
            return result
        if operation == "build" and _worker is not None and _worker.is_alive():
            return {"status": "busy", "retry_after_seconds": 2,
                    "message": "Another visual build is active. Repeat this call later."}
        if operation == "search" and _search_worker is not None and _search_worker.is_alive():
            return {"status": "busy", "retry_after_seconds": 2,
                    "message": "Another visual search is active. Repeat this call later."}
        while len(_jobs) >= 32:
            _jobs.popitem(last=False)
        _jobs[key] = {"status": "queued"}

        def report(data):
            with _jobs_lock:
                _jobs[key] = {"status": "running", **data}

        def work():
            report({})
            try:
                result = build_index(engine, report) if operation == "build" else search_index(engine, **args)
            except Exception as exc:
                result = {"status": "error", "message": str(exc), "retryable": True}
            with _jobs_lock:
                _jobs[key] = result

        worker = threading.Thread(target=work, name=f"visual-{operation}", daemon=True)
        if operation == "build":
            _worker = worker
        else:
            _search_worker = worker
        worker.start()
        return {"status": "queued", "message": "Repeat the same tool call to poll progress."}
