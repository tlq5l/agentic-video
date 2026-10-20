"""Content-addressed deterministic cache for video artifacts."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Any, Dict
from src.core.config import Config
from src.core.errors import ExtractionBoundsError, CacheWriteError

# Process-wide fingerprint memo: handle_call builds a fresh MediaCache per
# call, so a per-instance memo never hit. Entries are valid only while the
# file's (dev, inode, size, mtime, ctime) stat tuple is unchanged.
_FINGERPRINT_MEMO: Dict[str, tuple[tuple[int, int, int, int, int], str]] = {}


def _stat_key(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class MediaCache:
    """Manages cached metadata, scenes, contact sheets, and extracted frames by media fingerprint."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Config.BASE_CACHE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # The public method is assigned on the instance so the historical
        # class-level static API remains available as
        # ``MediaCache.compute_fingerprint(path)``.
        self.compute_fingerprint = self._compute_fingerprint_memoized
    
    @staticmethod
    def compute_fingerprint(file_path: str) -> str:
        """Compute a stable full-content fingerprint for a media source.

        Cache identity must change when any byte changes. The previous
        head/tail sampling optimization admitted same-size middle-byte
        collisions and could serve stale frames or transcripts. Hashing is
        chunked to keep memory bounded; the cost is paid once per source and
        buys an honest content-addressed cache key.
        """
        p = Path(file_path)
        if not p.exists():
            return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:20]
        
        hasher = hashlib.sha256()
        chunk_size = 1024 * 1024
        try:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hasher.update(chunk)
        except (OSError, IOError):
            # Preserve deterministic failure identity when a source cannot be
            # opened; probe/extraction will still surface the real error.
            try:
                stat = p.stat()
                hasher.update(f"unreadable:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
            except OSError:
                hasher.update(b"unreadable")
                
        return hasher.hexdigest()[:20]
    def _compute_fingerprint_memoized(self, file_path: str) -> str:
        """Return a memoized fingerprint when the source stat is unchanged."""
        p = Path(file_path)
        try:
            stat = p.stat()
        except OSError:
            # Do not memoize missing/unreadable paths: creation or replacement
            # must immediately retain the historical deterministic behavior.
            return MediaCache.compute_fingerprint(file_path)

        stat_key = _stat_key(stat)
        memo = _FINGERPRINT_MEMO.get(str(p))
        if memo is not None and memo[0] == stat_key:
            return memo[1]

        fingerprint = MediaCache.compute_fingerprint(file_path)
        # Re-stat after hashing: if the file changed while it was read, the
        # hash may not match either stat, so do not memoize it.
        try:
            after = p.stat()
        except OSError:
            return fingerprint
        if _stat_key(after) == stat_key:
            _FINGERPRINT_MEMO[str(p)] = (stat_key, fingerprint)
        return fingerprint

    def get_entry_dir(self, fingerprint: str) -> Path:
        """Get isolated root directory for this media fingerprint."""
        entry_dir = self.base_dir / fingerprint
        try:
            entry_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheWriteError(
                f"Cache directory is not writable: {entry_dir}",
                {"base_dir": str(self.base_dir), "reason": str(exc)}
            ) from exc
        return entry_dir

    def get_frames_dir(self, fingerprint: str, tag: str = "frames") -> Path:
        """Get subdirectory for extracted frames."""
        frames_dir = self.get_entry_dir(fingerprint) / tag
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheWriteError(
                f"Cache directory is not writable: {frames_dir}",
                {"base_dir": str(self.base_dir), "reason": str(exc)}
            ) from exc
        return frames_dir

    def get_sheets_dir(self, fingerprint: str) -> Path:
        """Get subdirectory for generated contact sheets."""
        sheets_dir = self.get_entry_dir(fingerprint) / "contact_sheets"
        try:
            sheets_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheWriteError(
                f"Cache directory is not writable: {sheets_dir}",
                {"base_dir": str(self.base_dir), "reason": str(exc)}
            ) from exc
        return sheets_dir

    def get_audio_dir(self, fingerprint: str) -> Path:
        """Get subdirectory for audio extractions."""
        audio_dir = self.get_entry_dir(fingerprint) / "audio"
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheWriteError(
                f"Cache directory is not writable: {audio_dir}",
                {"base_dir": str(self.base_dir), "reason": str(exc)}
            ) from exc
        return audio_dir

    def load_json(self, fingerprint: str, name: str) -> Optional[Any]:
        """Atomically load JSON document from media cache entry with retry on concurrent replacement.

        Never raises: unreadable/corrupt entries are treated as cache misses (returns None)
        so callers fall back to re-extraction instead of crashing.
        """
        path = self.base_dir / fingerprint / f"{name}.json"
        if not path.exists():
            return None
        for attempt in range(4):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, IOError, ValueError):
                # ValueError covers json.JSONDecodeError and UnicodeDecodeError (binary garbage)
                if attempt < 3:
                    time.sleep(0.01 * (attempt + 1))
                else:
                    return None
        return None

    def save_json(self, fingerprint: str, name: str, data: Any) -> Optional[Path]:
        """Atomically write JSON document to media cache entry via unique temp file.

        Guarantees thread-safe and process-safe concurrent writes without file lock collisions on Windows.
        Best-effort: if the cache is unwritable, returns None instead of failing the operation
        (the cache is an optimization and must never break a successful probe/extraction).
        """
        try:
            entry_dir = self.get_entry_dir(fingerprint)
            target_path = entry_dir / f"{name}.json"
            unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_path = entry_dir / f"{name}.{unique_id}.tmp"

            # json.dumps takes the C encoder; json.dump (and any indent) the pure-Python one,
            # which costs seconds on a multi-MB document. Readers parse, so layout is free.
            payload = json.dumps(data, separators=(",", ":"))
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            for attempt in range(5):
                try:
                    os.replace(tmp_path, target_path)
                    return target_path
                except (OSError, IOError):
                    if attempt < 4:
                        time.sleep(0.015 * (attempt + 1))

            # Final fallback: move instead of replace
            try:
                import shutil
                shutil.move(str(tmp_path), str(target_path))
                return target_path
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                return None
        except (OSError, CacheWriteError):
            return None

    @staticmethod
    def validate_extraction_bounds(
        fps: float,
        duration_sec: float,
        max_frames: Optional[int] = None
    ) -> None:
        """Enforce safety invariants on media operations."""
        if fps <= 0.0 or fps > Config.MAX_EXTRACTION_FPS:
            raise ExtractionBoundsError(
                f"Requested extraction FPS ({fps}) out of bounds (must be 0.01 <= fps <= {Config.MAX_EXTRACTION_FPS:g})",
                {"requested_fps": fps, "max_fps": Config.MAX_EXTRACTION_FPS}
            )
        if duration_sec < 0.0 or duration_sec > Config.MAX_WINDOW_DURATION_SEC:
            raise ExtractionBoundsError(
                f"Requested duration ({duration_sec:.2f}s) exceeds maximum window ({Config.MAX_WINDOW_DURATION_SEC:g}s); "
                "use a shorter start_sec/end_sec range, or contact_sheet / overview_chunks to survey a long video",
                {"requested_duration": duration_sec, "max_duration": Config.MAX_WINDOW_DURATION_SEC}
            )
        
        limit = max_frames or Config.MAX_FRAMES_PER_CALL
        est_frames = int(fps * duration_sec) + 1
        if est_frames > limit:
            raise ExtractionBoundsError(
                f"Requested extraction would yield ~{est_frames} frames, exceeding maximum ({limit}); "
                f"lower fps to at most {max((limit - 1) / duration_sec, 0.01) if duration_sec > 0 else fps:.3g} for this {duration_sec:.1f}s window "
                "or shorten it, or use contact_sheet to survey a long span",
                {"estimated_frames": est_frames, "max_allowed": limit}
            )
