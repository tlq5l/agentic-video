import shutil
import glob
import sys
from dataclasses import dataclass
import os
from pathlib import Path

def ensure_ffmpeg_on_path():
    """Make ffmpeg and ffprobe discoverable when the host starts the server with a minimal PATH.

    macOS (Claude Desktop does not inherit the shell PATH): prepend the first of Homebrew's
    and ~/.local/bin that holds both binaries. Windows: search the WinGet, Scoop and
    Chocolatey install folders.
    """
    if shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None:
        return
    if sys.platform == "darwin":
        for folder in (Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path.home() / ".local/bin"):
            if all((folder / name).is_file() and os.access(folder / name, os.X_OK) for name in ("ffmpeg", "ffprobe")):
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
                return
        return
    # Check WinGet Gyan.FFmpeg packages
    winget_pattern = str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin")
    matches = glob.glob(winget_pattern, recursive=True)
    for m in matches:
        if (Path(m) / "ffmpeg.exe").exists():
            os.environ["PATH"] = m + os.pathsep + os.environ.get("PATH", "")
            return
    # Check common alternatives
    for c in [Path.home() / "scoop/shims", Path("C:/ProgramData/chocolatey/bin"), Path("C:/ffmpeg/bin")]:
        if (c / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            return

ensure_ffmpeg_on_path()

@dataclass(frozen=True)
class Config:
    # Execution bounds
    MAX_EXTRACTION_FPS: float = 30.0
    DEFAULT_SPARSE_FPS: float = 0.2
    DEFAULT_DENSE_FPS: float = 4.0
    MAX_FRAMES_PER_CALL: int = 150
    MAX_WINDOW_DURATION_SEC: float = 600.0
    SUBPROCESS_TIMEOUT_SEC: float = 45.0
    
    # Image formatting
    DEFAULT_FRAME_WIDTH: int = 768
    THUMBNAIL_WIDTH: int = 256
    JPEG_QUALITY: int = 85
    
    # Vision tokens estimation
    TOKENS_PER_IMAGE: int = 258
    
    # ASR settings
    DEFAULT_WHISPER_MODEL: str = "tiny"
    FALLBACK_WHISPER_MODEL: str = "base"
    WHISPER_COMPUTE_TYPE: str = "int8"  # fast on Windows CPU
    
    # Storage & Cache
    BASE_CACHE_DIR: Path = Path(os.getenv("AGENTIC_VIDEO_CACHE_DIR", Path.home() / ".cache" / "agentic-video"))
    TEMP_DIR: Path = Path(os.getenv("AGENTIC_VIDEO_TEMP_DIR", Path.home() / ".cache" / "agentic-video" / "tmp"))

    @classmethod
    def ensure_directories(cls):
        cls.BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
