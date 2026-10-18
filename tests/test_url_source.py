"""A video URL as `source`: normalized, downloaded once in the background, cached, and reported."""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from src.core.config import Config
from src.executor import mcp_server
from src.executor.mcp_server import handle_call
from src.media import fetch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "src/adapters/mcp_schemas.json").read_text(encoding="utf-8"))["tools"]
X_URL = "https://x.com/DilumSanjaya/status/2111864428010582082"


class FakeEngine:
    sources = []

    def __init__(self, source):
        FakeEngine.sources.append(source)

    def probe(self):
        class Meta:
            def to_dict(self):
                return {"duration_sec": 63.0}
        return Meta()


class FakeYtDlp:
    """Stands in for subprocess.run: writes the file yt-dlp would, or fails like yt-dlp does."""

    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.release.set()
        self.fail_stderr = None

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        self.release.wait(5)
        if self.fail_stderr:
            return subprocess.CompletedProcess(cmd, 1, "", self.fail_stderr)
        out = Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4"))
        out.write_bytes(b"fake video")
        return subprocess.CompletedProcess(cmd, 0, f"{out}\n", "")


@pytest.fixture
def ytdlp(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetch, "_find_binary", lambda name: f"/fake/{name}")
    monkeypatch.setattr(fetch, "WAIT_SEC", 2.0)
    monkeypatch.setattr(fetch, "_JOBS", {})
    fake = FakeYtDlp()
    monkeypatch.setattr(fetch.subprocess, "run", fake)
    FakeEngine.sources = []
    monkeypatch.setattr(mcp_server, "AgenticVideoEngine", FakeEngine)
    return fake


@pytest.mark.parametrize("variant,canonical", [
    ("https://twitter.com/someone/status/2111864428010582082?s=20", "https://x.com/i/status/2111864428010582082"),
    ("https://mobile.x.com/DilumSanjaya/status/2111864428010582082/video/1", "https://x.com/i/status/2111864428010582082"),
    ("https://youtu.be/5TNXByrtHc4?si=abc", "https://www.youtube.com/watch?v=5TNXByrtHc4"),
    ("https://m.youtube.com/watch?v=5TNXByrtHc4&t=42&feature=share", "https://www.youtube.com/watch?v=5TNXByrtHc4"),
    ("https://www.youtube.com/shorts/5TNXByrtHc4", "https://www.youtube.com/watch?v=5TNXByrtHc4"),
    ("HTTPS://Example.COM/clip.mp4?utm_source=x&b=2&a=1#t=3", "https://example.com/clip.mp4?a=1&b=2"),
])
def test_url_spellings_of_one_video_normalize_to_one_key(variant, canonical):
    assert fetch.normalize_url(variant) == canonical


def test_local_path_passes_through_untouched(ytdlp):
    result = handle_call("probe_video", {"source": "clip.mp4"})
    assert result == {"ok": True, "data": {"duration_sec": 63.0}}
    assert FakeEngine.sources == ["clip.mp4"] and ytdlp.calls == []


def test_url_downloads_once_and_reports_source_url_and_local_path(ytdlp, tmp_path):
    first = handle_call("probe_video", {"source": X_URL})
    local = str(fetch.cached_path(X_URL))
    assert first == {"ok": True, "data": {"duration_sec": 63.0, "source_url": X_URL, "local_path": local}}
    assert Path(local).read_bytes() == b"fake video" and Path(local).parent == tmp_path / "downloads"
    assert list((tmp_path / "downloads").iterdir()) == [Path(local)]  # no temp file left behind
    cmd = ytdlp.calls[0]
    assert "--no-playlist" in cmd and cmd[-1] == X_URL

    again = handle_call("probe_video", {"source": "https://twitter.com/x/status/2111864428010582082"})
    assert again["data"]["local_path"] == local
    assert len(ytdlp.calls) == 1
    assert FakeEngine.sources == [local, local]


def test_slow_download_returns_in_progress_then_ready(ytdlp, monkeypatch):
    monkeypatch.setattr(fetch, "WAIT_SEC", 0.05)
    ytdlp.release.clear()
    pending = handle_call("probe_video", {"source": X_URL})
    assert pending == {"ok": True, "data": fetch.pending_result(X_URL)}
    assert pending["data"]["status"] == "in_progress" and pending["data"]["retry_after_seconds"] > 0
    assert X_URL in pending["data"]["note"]
    assert handle_call("probe_video", {"source": X_URL})["data"]["status"] == "in_progress"
    assert len(ytdlp.calls) == 1  # the second call joined the running download

    ytdlp.release.set()
    monkeypatch.setattr(fetch, "WAIT_SEC", 2.0)
    ready = handle_call("probe_video", {"source": X_URL})
    assert ready["data"]["duration_sec"] == 63.0 and ready["data"]["local_path"]
    assert len(ytdlp.calls) == 1


def test_failure_names_the_url_and_last_yt_dlp_error_then_retries(ytdlp):
    ytdlp.fail_stderr = ("WARNING: [twitter] something minor\n"
                         "ERROR: [twitter] 2111864428010582082: No video could be found in this tweet\n")
    result = handle_call("probe_video", {"source": X_URL})
    assert result["ok"] is False and result["error"]["code"] == "download_failed"
    assert result["error"]["message"] == (
        f"Could not download {X_URL}: ERROR: [twitter] 2111864428010582082: No video could be found in this tweet")
    assert not fetch.cached_path(X_URL).exists() and FakeEngine.sources == []

    ytdlp.fail_stderr = None
    assert handle_call("probe_video", {"source": X_URL})["ok"] is True
    assert len(ytdlp.calls) == 2


def test_missing_yt_dlp_says_how_to_fix(ytdlp, monkeypatch):
    monkeypatch.setattr(fetch, "_find_binary", lambda name: None)
    result = handle_call("probe_video", {"source": X_URL})
    assert "yt-dlp is not installed" in result["error"]["message"] and "local path" in result["error"]["message"]


# ---- output contract --------------------------------------------------------

def _accepts(schema, value):
    """Top-level JSON Schema check for the keywords these output schemas use."""
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False and not set(value) <= set(props):
        return False
    if not set(schema.get("required", [])) <= set(value):
        return False
    for key, item in value.items():
        spec = props.get(key, {})
        if "enum" in spec and item not in spec["enum"]:
            return False
        if spec.get("type") == "string" and not isinstance(item, str):
            return False
    if "anyOf" in schema and not any(_accepts(branch, value) for branch in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(_accepts(branch, value) for branch in schema["oneOf"]) != 1:
        return False
    if "not" in schema and _accepts(schema["not"], value):
        return False
    return True


@pytest.mark.parametrize("tool", sorted(CONTRACT))
def test_every_output_schema_accepts_the_download_pending_result(tool):
    output = CONTRACT[tool]["outputSchema"]
    assert _accepts(output, fetch.pending_result(X_URL))
    if "properties" in output:
        assert {"source_url", "local_path"} <= set(output["properties"])


def test_url_fields_keep_ready_and_asr_progress_results_valid():
    url = {"source_url": X_URL, "local_path": "/cache/x.mp4"}
    probe = {"source_path": "/cache/x.mp4", "duration_sec": 63.0, "width": 1280, "height": 720, "fps": 30.0,
             "format_name": "mp4", "size_bytes": 1, "has_audio": True, "audio_codec": "aac",
             "video_codec": "h264", "fingerprint": "abc"}
    assert _accepts(CONTRACT["probe_video"]["outputSchema"], {**probe, **url})
    asr_progress = {"status": "in_progress", "requested_model_size": "tiny", "requested_model_cached": False,
                    "transcription_status": "started", "retry_after_seconds": 30, "transcription_message": "m"}
    assert _accepts(CONTRACT["transcribe"]["outputSchema"], {**asr_progress, **url})


def test_source_descriptions_mention_urls():
    for tool, spec in CONTRACT.items():
        assert "video URL" in spec["inputSchema"]["properties"]["source"]["description"], tool
    probe = CONTRACT["probe_video"]["description"]
    assert "x.com" in probe and "YouTube" in probe
