"""Fail-closed structured exceptions for agentic-video."""

from __future__ import annotations

class AgenticVideoError(Exception):
    """Base exception for all agentic-video operations."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }

class MediaNotFoundError(AgenticVideoError):
    """Raised when the specified video file or URL does not exist or cannot be accessed."""
    pass

class MediaCorruptError(AgenticVideoError):
    """Raised when media fails ffprobe validation or has no decodable streams."""
    pass

class ExtractionBoundsError(AgenticVideoError):
    """Raised when requested extraction exceeds safety bounds (e.g. too many frames, invalid FPS)."""
    pass

class SubprocessTimeoutError(AgenticVideoError):
    """Raised when an external tool (ffmpeg, ffprobe) times out."""
    pass

class ASRModelError(AgenticVideoError):
    """Raised when speech recognition fails or no audio is present."""
    pass

class DependencyNotFoundError(AgenticVideoError):
    """Raised when an external system dependency (e.g. ffmpeg, ffprobe) is missing or not executable."""
    FFMPEG_FIX = (". Install FFmpeg with `brew install ffmpeg` on macOS (Windows: `winget install Gyan.FFmpeg`), "
                  "then restart Claude Desktop (or the MCP client) so it starts the server again.")

    def __init__(self, message: str, details: dict | None = None):
        # The fix is part of the message, so every caller (tools that raise it and workers that turn it into a string) shows it.
        if (details or {}).get("dependency") in ("ffmpeg", "ffprobe"):
            message += self.FFMPEG_FIX
        super().__init__(message, details)

def as_dependency_error(exc: BaseException) -> BaseException:
    """Turn a plain 'ffmpeg not found' OSError from a subprocess call into DependencyNotFoundError.

    Some call sites let Popen's FileNotFoundError escape unchanged. Translating it here means the
    install fix comes from DependencyNotFoundError, once, for every caller. Other errors pass through.
    """
    if isinstance(exc, FileNotFoundError) and exc.filename in ("ffmpeg", "ffprobe"):
        name = exc.filename
        return DependencyNotFoundError(f"{name} binary not found on system PATH: {exc}", {"dependency": name})
    return exc

class CacheWriteError(AgenticVideoError):
    """Raised when the media cache directory cannot be created or written (e.g. ACL/permission failure)."""
    pass

