#!/usr/bin/env python3
"""Pre-ingest bench-v1 videos into the agentic-video transcript and visual caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .common import DEFAULT_DATA_ROOT, DEFAULT_SEALED_DIR, data_root, default_tool_cwd, default_tool_python, question_path, read_jsonl, visual_model_environment, scratch_dir
except ImportError:  # direct script execution
    from common import DEFAULT_DATA_ROOT, DEFAULT_SEALED_DIR, data_root, default_tool_cwd, default_tool_python, question_path, read_jsonl, visual_model_environment, scratch_dir


DEFAULT_VERIFY_TIMEOUT_SEC = 45
DEFAULT_ASR_TIMEOUT_SEC = 3600.0
DEFAULT_VISUAL_TIMEOUT_SEC = 1800.0
MANIFEST_NAME = "ingest-ours-codex.json"

# Keep the cache identity check aligned with transcribe_video's default model,
# sidecar selection, cache key, and binding checks. This runs in the MCP
# checkout, so it uses that checkout's live Config and MediaCache code.
_CACHE_CHECK = r'''
import re
from pathlib import Path
from src.core.cache import MediaCache
from src.core.config import Config
from src.transcript.asr import _find_sidecar, _subtitle_fingerprint

def transcript_cache(source):
    source = Path(source).resolve()
    cache = MediaCache()
    fingerprint = cache.compute_fingerprint(str(source))
    sidecar = _find_sidecar(source, None)
    subtitle_fingerprint = _subtitle_fingerprint(sidecar)
    model_size = Config.DEFAULT_WHISPER_MODEL
    cache_name = (
        f"transcript_subtitle_{subtitle_fingerprint}"
        if sidecar is not None and subtitle_fingerprint
        else f"transcript_asr_{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_size)}"
    )
    value = cache.load_json(fingerprint, cache_name)
    valid = (
        isinstance(value, dict)
        and value.get("_media_fingerprint") == fingerprint
        and value.get("_subtitle_fingerprint") == subtitle_fingerprint
        and (subtitle_fingerprint is not None or value.get("_model_size") == model_size)
    )
    return bool(valid), model_size
'''

INGEST_SCRIPT = _CACHE_CHECK + r'''
import json
import math
import sys
import time
from pathlib import Path
from src.transcript.asr import transcribe_video

source = Path(sys.argv[1]).resolve()
asr_timeout_sec = float(sys.argv[2])
visual_timeout_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
cache_hit_before = False
model_size = None
transcript_started = time.monotonic()
try:
    cache_hit_before, model_size = transcript_cache(source)
    if cache_hit_before:
        # No timeout argument: an explicit timeout forces a fresh ASR run.
        transcript = transcribe_video(str(source))
    else:
        # The MCP default subprocess timeout (45s) is too short for long videos.
        transcript = transcribe_video(str(source), timeout_sec=asr_timeout_sec)
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "cache_hit_before": bool(cache_hit_before),
        "model_size": model_size,
        "transcript_time_s": round(time.monotonic() - transcript_started, 3),
        "visual_index_status": "not_attempted",
        "error": f"{type(exc).__name__}: {exc}",
    }))
    raise SystemExit(2)

transcript_time_s = round(time.monotonic() - transcript_started, 3)
visual_started = time.monotonic()
visual_index_status = "error"
visual_index_error = None
try:
    from src.retrieval.engine import AgenticVideoEngine
    from src.retrieval.visual import build_index

    engine = AgenticVideoEngine(str(source))
    visual_deadline = visual_started + max(0.0, visual_timeout_sec)
    while True:
        visual_result = build_index(engine)
        if not isinstance(visual_result, dict):
            visual_index_status = "ok"
            break
        result_status = str(visual_result.get("status") or "ok")
        if result_status != "busy":
            visual_index_status = result_status
            if result_status not in ("ok", "complete", "done"):
                visual_index_error = str(visual_result.get("message") or f"visual index build returned {result_status}")
            break

        remaining = visual_deadline - time.monotonic()
        if remaining <= 0:
            visual_index_status = "busy_timeout"
            visual_index_error = "Visual index remained busy until the configured timeout"
            break
        try:
            retry_after = float(visual_result.get("retry_after_seconds", 2.0))
        except (TypeError, ValueError):
            retry_after = 2.0
        if not math.isfinite(retry_after) or retry_after <= 0:
            retry_after = 2.0
        time.sleep(min(retry_after, remaining))
except Exception as exc:
    visual_index_status = "error"
    visual_index_error = f"{type(exc).__name__}: {exc}"
visual_index_time_s = round(time.monotonic() - visual_started, 3)

print(json.dumps({
    "status": "ok",
    "cache_hit_before": cache_hit_before,
    "duration_sec": transcript.duration_sec,
    "segment_count": transcript.segment_count,
    "model_size": model_size,
    "transcript_time_s": transcript_time_s,
    "visual_index_time_s": visual_index_time_s,
    "visual_index_status": visual_index_status,
    "visual_index_error": visual_index_error,
}))
'''

VERIFY_SCRIPT = _CACHE_CHECK + r'''
import json
import sys
from pathlib import Path
from src.transcript.search import search_transcript

source = Path(sys.argv[1])
try:
    cache_hit, model_size = transcript_cache(source)
    if not cache_hit:
        raise RuntimeError("default transcript cache entry is missing or unreadable inside sandbox")
    hits = search_transcript(str(source), query="__bench_v1_cache_verification__")
    print(json.dumps({
        "status": "ok",
        "model_size": model_size,
        "match_count": len(hits),
    }))
except Exception as exc:
    print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(2)
'''


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _unique_video_paths(questions: list[dict[str, Any]]) -> list[Path]:
    paths: dict[str, Path] = {}
    for question in questions:
        raw_path = question.get("video_path")
        if not raw_path:
            raise ValueError(f"question {question.get('id', '<unknown>')} has no video_path")
        path = Path(raw_path).expanduser().resolve()
        paths.setdefault(str(path), path)
    return list(paths.values())


def _last_json_object(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"videos": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("videos", []), list):
        raise ValueError(f"invalid ingest manifest: {path}")
    return value


def _worker_error(process: subprocess.CompletedProcess[str], payload: dict[str, Any] | None) -> str:
    if payload and payload.get("error"):
        return str(payload["error"])
    detail = (process.stderr or process.stdout or "").strip()
    if process.returncode:
        prefix = f"ingest subprocess exited with code {process.returncode}"
        return f"{prefix}: {detail[:1000]}" if detail else prefix
    return "ingest subprocess returned no status JSON"


def _payload_seconds(payload: dict[str, Any] | None, name: str) -> float | None:
    value = (payload or {}).get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return round(seconds, 3)


def ingest_questions(
    questions: list[dict[str, Any]],
    *,
    data_dir: Path = DEFAULT_DATA_ROOT,
    tool_cwd: Path,
    tool_python: Path,
    manifest_path: Path | None = None,
    asr_timeout_sec: float = DEFAULT_ASR_TIMEOUT_SEC,
    visual_timeout_sec: float = DEFAULT_VISUAL_TIMEOUT_SEC,
) -> dict[str, dict[str, Any]]:
    """Ingest each distinct selected video and preserve each video's first timing."""
    data_dir = Path(data_dir).expanduser()
    tool_cwd = Path(tool_cwd).expanduser().resolve()
    tool_python = Path(tool_python).expanduser().absolute()  # keep venv symlink
    if not math.isfinite(visual_timeout_sec) or visual_timeout_sec <= 0:
        raise ValueError("visual_timeout_sec must be a finite positive number")
    manifest_path = Path(manifest_path or data_dir / "runs" / MANIFEST_NAME).expanduser()
    manifest = _read_manifest(manifest_path)
    records = {
        str(item.get("path")): dict(item)
        for item in manifest.get("videos", [])
        if isinstance(item, dict) and item.get("path")
    }
    latest_elapsed: dict[str, float] = {}
    latest_visual_elapsed: dict[str, float] = {}
    latest_model_size: str | None = None

    for video_path in _unique_video_paths(questions):
        key = str(video_path)
        started = time.monotonic()
        payload: dict[str, Any] | None = None
        process: subprocess.CompletedProcess[str] | None = None
        error: str | None = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"video path does not exist: {video_path}")
            process = subprocess.run(
                [
                    str(tool_python), "-c", INGEST_SCRIPT, str(video_path),
                    str(asr_timeout_sec), str(visual_timeout_sec),
                ],
                cwd=str(tool_cwd),
                env={**os.environ, **visual_model_environment(tool_python)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = _last_json_object(process.stdout)
            if process.returncode != 0 or not payload or payload.get("status") != "ok":
                error = _worker_error(process, payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - started, 3)
        transcript_elapsed = _payload_seconds(payload, "transcript_time_s")
        if transcript_elapsed is None:
            transcript_elapsed = elapsed
        visual_elapsed = _payload_seconds(payload, "visual_index_time_s")
        latest_elapsed[key] = transcript_elapsed
        if visual_elapsed is not None:
            latest_visual_elapsed[key] = visual_elapsed
        if payload and isinstance(payload.get("model_size"), str):
            latest_model_size = payload["model_size"]

        previous = records.get(key, {})
        cache_hit_before = bool(payload.get("cache_hit_before")) if payload else False
        current_status = "ok" if error is None else "error"
        current = dict(previous)
        current.update({
            "path": key,
            "duration_sec": (payload or {}).get("duration_sec", previous.get("duration_sec")),
            "segment_count": (payload or {}).get("segment_count", previous.get("segment_count")),
            "status": current_status,
            "error": error,
            "last_ingest_time_s": transcript_elapsed,
            "last_cache_hit_before": cache_hit_before,
            "last_status": current_status,
            "last_error": error,
            "last_ingested_at": _timestamp(),
            "last_total_ingest_time_s": elapsed,
        })
        prior_visual_status = current.get("visual_index_status")
        if prior_visual_status in {"ok", "complete", "done"}:
            # Normalize successful records written by older versions.
            current["visual_index_status"] = "ok"
            current["visual_index_error"] = None
        visual_status = (payload or {}).get("visual_index_status")
        if visual_status and visual_status != "not_attempted":
            current.update({
                "last_visual_index_time_s": visual_elapsed,
                "last_visual_index_status": visual_status,
                "last_visual_index_error": (payload or {}).get("visual_index_error"),
            })
            previous_visual_status = current.get("visual_index_status")
            previous_visual_succeeded = previous_visual_status in {"ok", "complete", "done"}
            current_visual_succeeded = visual_status in {"ok", "complete", "done"}
            if current_visual_succeeded:
                if not previous_visual_succeeded or current.get("visual_index_time_s") is None:
                    current["visual_index_time_s"] = visual_elapsed
                current["visual_index_status"] = "ok"
                current["visual_index_error"] = None
            elif not previous_visual_succeeded:
                # Before the first successful build, keep the most recent attempt.
                current["visual_index_time_s"] = visual_elapsed
                current["visual_index_status"] = visual_status
                current["visual_index_error"] = (payload or {}).get("visual_index_error")
        first_ingest_succeeded = previous.get("first_ingest_succeeded")
        if first_ingest_succeeded is None:
            first_ingest_succeeded = previous.get("status") == "ok"
        if not first_ingest_succeeded:
            current["ingest_time_s"] = transcript_elapsed
            current["cache_hit_before"] = cache_hit_before
            current["first_ingested_at"] = _timestamp()
            current["first_ingest_succeeded"] = current_status == "ok"
        else:
            current["first_ingest_succeeded"] = True
        records[key] = current
        print(
            f"ingest {video_path.name}: {current_status}, transcript={transcript_elapsed:.3f}s, "
            f"visual={visual_status or 'not attempted'} "
            f"({visual_elapsed if visual_elapsed is not None else 0:.3f}s), "
            f"cache_hit_before={cache_hit_before}", flush=True,
        )

    ordered = [records[key] for key in sorted(records)]
    selected = set(latest_elapsed)
    total_recorded = sum(float(item.get("ingest_time_s") or 0) for item in ordered)
    visual_total_recorded = sum(float(item.get("visual_index_time_s") or 0) for item in ordered)
    total_this_run = sum(latest_elapsed.values())
    visual_total_this_run = sum(latest_visual_elapsed.values())
    statuses = [records[key].get("status") for key in selected]
    visual_statuses = [records[key].get("last_visual_index_status") for key in selected]
    manifest.update({
        "updated_at": _timestamp(),
        "tool_cwd": str(tool_cwd),
        "python": str(tool_python),
        "model_size": latest_model_size or manifest.get("model_size") or "MCP Config.DEFAULT_WHISPER_MODEL",
        "videos": ordered,
        "totals": {
            "video_count": len(ordered),
            "selected_video_count": len(selected),
            "succeeded": sum(status == "ok" for status in statuses),
            "failed": sum(status != "ok" for status in statuses),
            "ingest_time_s": round(total_recorded, 3),
            "visual_index_time_s": round(visual_total_recorded, 3),
            "preparation_time_s": round(total_recorded + visual_total_recorded, 3),
            "last_run_ingest_time_s": round(total_this_run, 3),
            "last_run_visual_index_time_s": round(visual_total_this_run, 3),
            "last_run_preparation_time_s": round(total_this_run + visual_total_this_run, 3),
            "visual_index_failed": sum(
                status not in (None, "not_attempted", "ok", "complete", "done")
                for status in visual_statuses
            ),
        },
    })
    _write_manifest(manifest_path, manifest)
    return {key: records[key] for key in selected}


def verify_video_cache_hit(
    video_path: Path,
    *,
    data_dir: Path,
    tool_cwd: Path,
    tool_python: Path,
    sealed_dir: Path,
    timeout_sec: int = DEFAULT_VERIFY_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Run cached search in sandbox-exec, refusing to search on a cache miss."""
    # Import lazily so normal ingest does not need runner sandbox code.
    try:
        from .run_contestant import REPO_ROOT, _sandbox_profile
    except ImportError:  # direct script execution
        from run_contestant import REPO_ROOT, _sandbox_profile

    video_path = Path(video_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser()
    tool_cwd = Path(tool_cwd).expanduser().resolve()
    tool_python = Path(tool_python).expanduser().absolute()  # keep venv symlink
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        raise RuntimeError("sandbox-exec is required for transcript-cache verification")
    if not video_path.is_file():
        raise FileNotFoundError(f"video path does not exist: {video_path}")

    temporary = scratch_dir("bench-v1-ingest-verify-")
    workspace = temporary / "case"
    workspace.mkdir()
    alias = workspace / "video.mp4"
    alias.symlink_to(video_path)
    private_code_dir = REPO_ROOT if REPO_ROOT.resolve() != tool_cwd.resolve() else None
    profile = _sandbox_profile(Path(sealed_dir), data_dir, video_path, private_code_dir)
    command = [
        sandbox_exec, "-p", profile, str(tool_python), "-c", VERIFY_SCRIPT, str(alias),
    ]
    started = time.monotonic()
    try:
        try:
            process = subprocess.run(
                command, cwd=str(tool_cwd), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_sec, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"sandboxed transcript search exceeded {timeout_sec}s; cache may be unreadable or search may be retranscribing"
            ) from exc
        elapsed = round(time.monotonic() - started, 3)
        payload = _last_json_object(process.stdout)
        if process.returncode != 0 or not payload or payload.get("status") != "ok":
            detail = _worker_error(process, payload)
            raise RuntimeError(f"sandboxed transcript-cache verification failed: {detail}")
        return {
            "status": "ok",
            "verify_time_s": elapsed,
            "model_size": payload.get("model_size"),
            "match_count": payload.get("match_count", 0),
        }
    finally:
        trash = shutil.which("trash")
        if trash:
            subprocess.run([trash, str(temporary)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def verify_questions(
    questions: list[dict[str, Any]],
    *,
    data_dir: Path,
    tool_cwd: Path,
    tool_python: Path,
    sealed_dir: Path,
    manifest_path: Path | None = None,
    timeout_sec: int = DEFAULT_VERIFY_TIMEOUT_SEC,
) -> dict[str, dict[str, Any]]:
    data_dir = Path(data_dir).expanduser()
    manifest_path = Path(manifest_path or data_dir / "runs" / MANIFEST_NAME).expanduser()
    manifest = _read_manifest(manifest_path)
    records = {
        str(item.get("path")): dict(item)
        for item in manifest.get("videos", [])
        if isinstance(item, dict) and item.get("path")
    }
    results: dict[str, dict[str, Any]] = {}
    for video_path in _unique_video_paths(questions):
        key = str(video_path)
        record = records.get(key)
        if not record or record.get("status") != "ok":
            results[key] = {"status": "error", "error": "video has no successful ingest manifest record"}
            print(f"verify {video_path.name}: missing successful ingest record", flush=True)
            continue
        started_at = _timestamp()
        verify_started = time.monotonic()
        try:
            result = verify_video_cache_hit(
                video_path, data_dir=data_dir, tool_cwd=tool_cwd,
                tool_python=tool_python, sealed_dir=sealed_dir,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            result = {
                "status": "error",
                "verify_time_s": round(time.monotonic() - verify_started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["verified_at"] = started_at
        record["verification"] = result
        records[key] = record
        results[key] = result
        elapsed = result.get("verify_time_s")
        elapsed_text = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "n/a"
        print(f"verify {video_path.name}: {result['status']}, {elapsed_text}", flush=True)

    manifest["updated_at"] = _timestamp()
    manifest["videos"] = [records[key] for key in sorted(records)]
    _write_manifest(manifest_path, manifest)
    return results


def _cli_questions(question_file: Path, selectors: list[str] | None) -> list[dict[str, Any]]:
    if selectors:
        try:
            from .run_contestant import _read_questions
        except ImportError:  # direct script execution
            from run_contestant import _read_questions
        return _read_questions(question_file, selectors)
    return read_jsonl(question_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", nargs="*", help="Question ids, comma-separated ids, or track selectors")
    parser.add_argument("--question-file", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tool-cwd", type=Path, default=None)
    parser.add_argument("--tool-python", type=Path, default=None)
    parser.add_argument("--verify", action="store_true", help="Verify cache-hit transcript search inside sandbox-exec")
    parser.add_argument("--verify-timeout-sec", type=int, default=DEFAULT_VERIFY_TIMEOUT_SEC)
    parser.add_argument("--asr-timeout-sec", type=float, default=DEFAULT_ASR_TIMEOUT_SEC,
                        help="ASR timeout for cache-miss ingest (MCP default is 45s)")
    parser.add_argument("--visual-timeout-sec", type=float, default=DEFAULT_VISUAL_TIMEOUT_SEC,
                        help="Maximum time to wait while another process holds the visual index lock; a build this process starts runs to completion")
    args = parser.parse_args()

    root = (args.data_root or data_root()).expanduser()
    questions_file = (args.question_file or question_path()).expanduser()
    questions = _cli_questions(questions_file, args.questions)
    tool_cwd = (args.tool_cwd or default_tool_cwd()).expanduser()
    tool_python = (args.tool_python or default_tool_python()).expanduser()
    manifest = args.manifest or root / "runs" / MANIFEST_NAME
    if args.verify:
        results = verify_questions(
            questions, data_dir=root, tool_cwd=tool_cwd,
            tool_python=tool_python,
            sealed_dir=DEFAULT_SEALED_DIR,
            manifest_path=manifest, timeout_sec=args.verify_timeout_sec,
        )
        return 1 if any(item.get("status") != "ok" for item in results.values()) else 0

    results = ingest_questions(
        questions, data_dir=root, tool_cwd=tool_cwd, tool_python=tool_python,
        manifest_path=manifest, asr_timeout_sec=args.asr_timeout_sec,
        visual_timeout_sec=args.visual_timeout_sec,
    )
    failed = [record for record in results.values() if record.get("status") != "ok"]
    successful = len(results) - len(failed)
    print(
        f"Ingested {successful}/{len(results)} distinct videos; "
        f"first transcript total {sum(float(item.get('ingest_time_s') or 0) for item in results.values()):.3f}s; "
        f"first visual-index total {sum(float(item.get('visual_index_time_s') or 0) for item in results.values()):.3f}s; "
        f"this-run transcript {sum(float(item.get('last_ingest_time_s') or 0) for item in results.values()):.3f}s; "
        f"this-run visual {sum(float(item.get('last_visual_index_time_s') or 0) for item in results.values()):.3f}s; "
        f"manifest: {manifest}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
