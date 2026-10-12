import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from benchmarks.bench_v1 import ingest


def _question(path: Path, question_id: str = "search-1") -> dict:
    return {"id": question_id, "track": "search", "video_path": str(path)}


def test_ingest_writes_timing_and_preserves_original_cache_observation(monkeypatch, tmp_path):
    video = tmp_path / "long-video.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    # A venv python is a symlink; resolving it would drop the venv's packages.
    real_python = tmp_path / "base-python"
    real_python.write_text("")
    tool_python = tmp_path / "venv" / "bin" / "python"
    tool_python.parent.mkdir(parents=True)
    tool_python.symlink_to(real_python)
    manifest_path = tmp_path / "runs" / "ingest.json"
    monkeypatch.delenv("AGENTIC_VIDEO_HF_HOME", raising=False)
    payloads = [
        {"status": "ok", "cache_hit_before": False, "duration_sec": 7207.0, "segment_count": 812, "model_size": "tiny",
         "transcript_time_s": 4.25, "visual_index_time_s": 7.5, "visual_index_status": "complete"},
        {"status": "ok", "cache_hit_before": True, "duration_sec": 7207.0, "segment_count": 812, "model_size": "tiny",
         "transcript_time_s": 0.25, "visual_index_time_s": 0.5, "visual_index_status": "complete"},
    ]
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payloads.pop(0)), stderr="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    first = ingest.ingest_questions(
        [_question(video), _question(video, "search-2")],
        data_dir=tmp_path, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest_path,
    )

    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[0] == str(tool_python)
    assert command[1] == "-c"
    assert command[3] == str(video.resolve())
    assert command[4] == str(float(ingest.DEFAULT_ASR_TIMEOUT_SEC))
    assert command[5] == str(float(ingest.DEFAULT_VISUAL_TIMEOUT_SEC))
    assert "transcribe_video(str(source))" in command[2]
    assert "src.retrieval.visual import build_index" in command[2]
    assert "engine = AgenticVideoEngine(str(source))" in command[2]
    assert "visual_result = build_index(engine)" in command[2]
    assert "timeout" not in kwargs
    assert kwargs["cwd"] == str(tool_cwd.resolve())
    assert kwargs["env"]["HF_HOME"] == str(tool_python.parent.parent / "huggingface")
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    record = first[str(video.resolve())]
    assert record["status"] == "ok"
    assert record["duration_sec"] == 7207.0
    assert record["segment_count"] == 812
    assert record["cache_hit_before"] is False
    assert record["ingest_time_s"] == 4.25
    assert record["last_ingest_time_s"] == 4.25
    assert record["visual_index_time_s"] == 7.5
    assert record["visual_index_status"] == "ok"
    assert record["last_visual_index_status"] == "complete"
    assert record["last_visual_index_time_s"] == 7.5

    first_time = record["ingest_time_s"]
    second = ingest.ingest_questions(
        [_question(video)], data_dir=tmp_path, tool_cwd=tool_cwd,
        tool_python=tool_python, manifest_path=manifest_path,
    )
    reread = second[str(video.resolve())]
    assert reread["cache_hit_before"] is False
    assert reread["ingest_time_s"] == first_time
    assert reread["last_cache_hit_before"] is True
    assert reread["last_ingest_time_s"] == 0.25
    assert reread["visual_index_time_s"] == 7.5
    assert reread["last_visual_index_time_s"] == 0.5

    manifest = json.loads(manifest_path.read_text())
    assert manifest["tool_cwd"] == str(tool_cwd.resolve())
    assert manifest["python"] == str(tool_python)
    assert manifest["model_size"] == "tiny"
    assert manifest["totals"]["video_count"] == 1
    assert manifest["totals"]["failed"] == 0
    assert manifest["totals"]["ingest_time_s"] == 4.25
    assert manifest["totals"]["visual_index_time_s"] == 7.5
    assert manifest["totals"]["preparation_time_s"] == 11.75
    assert manifest["totals"]["last_run_ingest_time_s"] == 0.25
    assert manifest["totals"]["last_run_visual_index_time_s"] == 0.5


def test_visual_timing_tracks_latest_failure_until_first_success_then_preserves_it(monkeypatch, tmp_path):
    video = tmp_path / "visual-retry.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    tool_python = tmp_path / "python"
    tool_python.write_text("")
    manifest_path = tmp_path / "runs" / "ingest.json"
    payloads = [
        {"status": "ok", "cache_hit_before": False, "transcript_time_s": 1.0,
         "visual_index_time_s": 1.0, "visual_index_status": "error", "visual_index_error": "first failure"},
        {"status": "ok", "cache_hit_before": True, "transcript_time_s": 0.2,
         "visual_index_time_s": 20.0, "visual_index_status": "complete"},
        {"status": "ok", "cache_hit_before": True, "transcript_time_s": 0.2,
         "visual_index_time_s": 0.5, "visual_index_status": "complete"},
        {"status": "ok", "cache_hit_before": True, "transcript_time_s": 0.2,
         "visual_index_time_s": 0.3, "visual_index_status": "error", "visual_index_error": "later failure"},
    ]

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payloads.pop(0)), stderr="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    question = [_question(video)]
    first = ingest.ingest_questions(
        question, data_dir=tmp_path, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest_path,
    )[str(video.resolve())]
    assert first["visual_index_time_s"] == 1.0
    assert first["visual_index_status"] == "error"

    second = ingest.ingest_questions(
        question, data_dir=tmp_path, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest_path,
    )[str(video.resolve())]
    assert second["visual_index_time_s"] == 20.0
    assert second["visual_index_status"] == "ok"

    third = ingest.ingest_questions(
        question, data_dir=tmp_path, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest_path,
    )[str(video.resolve())]
    assert third["visual_index_time_s"] == 20.0
    assert third["visual_index_status"] == "ok"
    assert third["last_visual_index_time_s"] == 0.5
    manifest = json.loads(manifest_path.read_text())
    assert manifest["totals"]["visual_index_time_s"] == 20.0
    assert manifest["totals"]["visual_index_failed"] == 0

    fourth = ingest.ingest_questions(
        question, data_dir=tmp_path, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest_path,
    )[str(video.resolve())]
    assert fourth["visual_index_time_s"] == 20.0
    assert fourth["visual_index_status"] == "ok"
    assert fourth["last_visual_index_status"] == "error"
    assert json.loads(manifest_path.read_text())["totals"]["visual_index_failed"] == 1


@pytest.mark.parametrize("visual_status", ["error", "busy_timeout"])
def test_visual_index_failure_is_recorded_without_failing_transcript_ingest(monkeypatch, tmp_path, visual_status):
    video = tmp_path / "visual-failure.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    tool_python = tmp_path / "venv" / "bin" / "python"
    tool_python.parent.mkdir(parents=True)
    tool_python.write_text("")
    monkeypatch.delenv("AGENTIC_VIDEO_HF_HOME", raising=False)
    command_result = subprocess.CompletedProcess(
        [str(tool_python), "-c"], 0,
        stdout=json.dumps({
            "status": "ok", "cache_hit_before": False,
            "transcript_time_s": 1.25, "visual_index_time_s": 0.75,
            "visual_index_status": visual_status, "visual_index_error": "RuntimeError: CLIP weights missing",
        }),
        stderr="",
    )
    monkeypatch.setattr(ingest.subprocess, "run", lambda command, **kwargs: command_result)

    result = ingest.ingest_questions(
        [_question(video)], data_dir=tmp_path, tool_cwd=tool_cwd,
        tool_python=tool_python,
    )[str(video.resolve())]

    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["ingest_time_s"] == 1.25
    assert result["visual_index_time_s"] == 0.75
    assert result["visual_index_status"] == visual_status
    assert result["visual_index_error"] == "RuntimeError: CLIP weights missing"
    manifest = json.loads((tmp_path / "runs" / ingest.MANIFEST_NAME).read_text())
    assert manifest["totals"]["failed"] == 0
    assert manifest["totals"]["visual_index_failed"] == 1
    assert manifest["videos"][0]["visual_index_status"] == visual_status


def test_ingest_persists_subprocess_error_in_manifest(monkeypatch, tmp_path):
    video = tmp_path / "bad-video.mp4"
    video.write_bytes(b"video fixture")
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    command_result = subprocess.CompletedProcess(
        ["python", "-c"], 2,
        stdout=json.dumps({"status": "error", "cache_hit_before": False, "error": "RuntimeError: ASR unavailable"}),
        stderr="",
    )
    monkeypatch.setattr(ingest.subprocess, "run", lambda command, **kwargs: command_result)

    results = ingest.ingest_questions(
        [_question(video)], data_dir=tmp_path, tool_cwd=tool_cwd,
        tool_python=tmp_path / "python",
    )
    record = results[str(video.resolve())]
    assert record["status"] == "error"
    assert record["error"] == "RuntimeError: ASR unavailable"
    assert record["ingest_time_s"] >= 0
    manifest = json.loads((tmp_path / "runs" / ingest.MANIFEST_NAME).read_text())
    assert manifest["totals"]["failed"] == 1


def test_sandboxed_verification_uses_video_symlink_and_fails_on_cache_miss(monkeypatch, tmp_path):
    video = tmp_path / "long-video.mp4"
    video.write_bytes(b"video fixture")
    data_dir = tmp_path / "data"
    (data_dir / "videos").mkdir(parents=True)
    tool_cwd = tmp_path / "mcp"
    tool_cwd.mkdir()
    tool_python = tmp_path / "python"
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    temp_roots = []

    def make_temp_root(prefix, dir=None):
        path = tmp_path / f"verify-work-{len(temp_roots)}"
        path.mkdir()
        temp_roots.append(path)
        return str(path)

    monkeypatch.setattr(ingest.tempfile, "mkdtemp", make_temp_root)
    monkeypatch.setattr(
        ingest.shutil, "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    calls = []
    ok_result = subprocess.CompletedProcess(
        ["sandbox-exec"], 0,
        stdout=json.dumps({"status": "ok", "model_size": "tiny", "match_count": 0}),
        stderr="",
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return ok_result

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    result = ingest.verify_video_cache_hit(
        video, data_dir=data_dir, tool_cwd=tool_cwd, tool_python=tool_python,
        sealed_dir=sealed_dir, timeout_sec=12,
    )
    assert result["status"] == "ok"
    assert result["verify_time_s"] >= 0
    command, kwargs = calls[0]
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "(deny file-read*" in command[2]
    assert str(sealed_dir.resolve()) in command[2]
    assert command[3] == str(tool_python.absolute())
    assert "search_transcript(str(source)" in command[5]
    assert command[-1].endswith("/case/video.mp4")
    assert Path(command[-1]).is_symlink()
    assert kwargs["timeout"] == 12

    failed = subprocess.CompletedProcess(
        ["sandbox-exec"], 2,
        stdout=json.dumps({"status": "error", "error": "transcript cache entry is missing inside sandbox"}),
        stderr="",
    )
    monkeypatch.setattr(ingest.subprocess, "run", lambda command, **kwargs: failed)
    with pytest.raises(RuntimeError, match="cache entry is missing inside sandbox"):
        ingest.verify_video_cache_hit(
            video, data_dir=data_dir, tool_cwd=tool_cwd, tool_python=tool_python,
            sealed_dir=sealed_dir, timeout_sec=12,
        )


def _fake_mcp_checkout(root: Path, cache_hit: bool) -> Path:
    """Minimal stand-in for the MCP modules the ingest script imports."""
    calls = root / "calls.jsonl"
    files = {
        "src/__init__.py": "",
        "src/core/__init__.py": "",
        "src/retrieval/__init__.py": "",
        "src/transcript/__init__.py": "",
        "src/retrieval/engine.py": textwrap.dedent("""
            class AgenticVideoEngine:
                def __init__(self, source):
                    self.source = source
        """),
        "src/retrieval/visual.py": textwrap.dedent(f"""
            import json
            from pathlib import Path
            _STATUS_FILE = Path({str(root / 'visual-statuses.json')!r})
            _CALLS_FILE = Path({str(root / 'visual-calls.txt')!r})
            def build_index(engine):
                with _CALLS_FILE.open('a') as handle:
                    handle.write('call\\n')
                statuses = json.loads(_STATUS_FILE.read_text()) if _STATUS_FILE.exists() else []
                if statuses:
                    result = statuses.pop(0)
                    _STATUS_FILE.write_text(json.dumps(statuses))
                    return result
                return {{'status': 'complete'}}
        """),
        "src/core/config.py": "class Config:\n    DEFAULT_WHISPER_MODEL = 'tiny'\n",
        "src/core/cache.py": textwrap.dedent(f"""
            class MediaCache:
                def compute_fingerprint(self, path):
                    return 'fp'
                def load_json(self, fingerprint, name):
                    if {cache_hit!r}:
                        return {{'_media_fingerprint': 'fp', '_subtitle_fingerprint': None, '_model_size': 'tiny'}}
                    return None
        """),
        "src/transcript/asr.py": textwrap.dedent(f"""
            import json
            class _T:
                duration_sec = 7207.0
                segment_count = 3
            def _find_sidecar(source, override):
                return None
            def _subtitle_fingerprint(sidecar):
                return None
            def transcribe_video(source, **kwargs):
                with open({str(calls)!r}, 'a') as fh:
                    fh.write(json.dumps(kwargs) + '\\n')
                return _T()
        """),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return calls


@pytest.mark.parametrize("cache_hit", [False, True])
def test_ingest_script_uses_long_asr_timeout_only_on_cache_miss(tmp_path, cache_hit):
    # MCP's default 45s ASR timeout fails two-hour videos, but an explicit
    # timeout also bypasses the transcript cache, so pass it only on a miss.
    calls = _fake_mcp_checkout(tmp_path, cache_hit)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    process = subprocess.run(
        [sys.executable, "-c", ingest.INGEST_SCRIPT, str(video), "3600.0"],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["cache_hit_before"] is cache_hit
    kwargs = [json.loads(line) for line in calls.read_text().splitlines()]
    assert kwargs == ([{}] if cache_hit else [{"timeout_sec": 3600.0}])


def test_ingest_script_retries_busy_visual_index_until_it_finishes(tmp_path):
    _fake_mcp_checkout(tmp_path, cache_hit=False)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    (tmp_path / "visual-statuses.json").write_text(json.dumps([
        {"status": "busy", "retry_after_seconds": 0.01},
        {"status": "complete"},
    ]))

    process = subprocess.run(
        [sys.executable, "-c", ingest.INGEST_SCRIPT, str(video), "3600.0", "1.0"],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    assert payload["visual_index_status"] == "complete"
    assert payload["visual_index_time_s"] >= 0.01
    assert len((tmp_path / "visual-calls.txt").read_text().splitlines()) == 2


def test_ingest_script_reports_bounded_busy_timeout_as_visual_failure(tmp_path):
    _fake_mcp_checkout(tmp_path, cache_hit=False)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    (tmp_path / "visual-statuses.json").write_text(json.dumps([
        *[{"status": "busy", "retry_after_seconds": 0.01} for _ in range(100)],
    ]))

    process = subprocess.run(
        [sys.executable, "-c", ingest.INGEST_SCRIPT, str(video), "3600.0", "0.03"],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    assert payload["visual_index_status"] == "busy_timeout"
    assert "timeout" in payload["visual_index_error"].lower()
    assert payload["visual_index_time_s"] >= 0.02
