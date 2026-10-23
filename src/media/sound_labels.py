"""Name the sounds in audio windows with Cloudflare Workers AI (Clef-omni).

Each window of the requested range is cut out with FFmpeg, sent as a 16 kHz
mono WAV with one multiple-choice question over the label list, and the
per-label chances come back. The media leaves the machine, so personal
recordings must not be passed in. No model is loaded locally and nothing stays
running: one ffmpeg process per call, a few short HTTPS requests.

Command line: ``python -m src.media.sound_labels VIDEO START END [--labels "dog barking,siren"]``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import subprocess
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import Config
from src.core.errors import (
    AgenticVideoError,
    DependencyNotFoundError,
    ExtractionBoundsError,
    MediaCorruptError,
    MediaNotFoundError,
    SubprocessTimeoutError,
)
from src.media.cloudflare import CloudCallError, api_base, credentials, forget_command_token, hide_token

MODEL = "clef-omni"
SAMPLE_RATE = 16000
DEFAULT_WINDOW_SEC = 5.0
MIN_WINDOW_SEC = 1.0
MAX_WINDOW_SEC = 30.0
MAX_WINDOWS = 60
MAX_LABELS = 50
PARALLEL_CALLS = 4
REQUEST_TIMEOUT_SEC = 60.0
NONE_LABEL = "none of these (silence, speech, music or another sound)"
# The default closed list of sound names offered to the model when `labels` is not given.
DEFAULT_LABELS: Tuple[str, ...] = (
    "dog barking", "rooster crowing", "ocean waves", "crackling fire", "clock ticking",
    "glass breaking", "siren", "church bells", "chainsaw", "car horn", "thunderstorm",
    "train", "helicopter", "baby crying", "sneezing", "clapping", "footsteps",
    "keyboard typing", "vacuum cleaner", "knocking on a wooden door",
)


def _credentials() -> Tuple[str, str]:
    return credentials("label_sounds", "audio")


def _decode_range(source: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Mono 16 kHz samples of [start_sec, end_sec) as float32, decoded by one bounded ffmpeg run."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise MediaNotFoundError("Audio source file not found", {"path": str(path)})
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}",
        "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    try:
        done = subprocess.run(command, capture_output=True, timeout=Config.SUBPROCESS_TIMEOUT_SEC, check=False)
    except FileNotFoundError as exc:
        raise DependencyNotFoundError("ffmpeg binary not found on system PATH", {"dependency": "ffmpeg"}) from exc
    except subprocess.TimeoutExpired as exc:
        raise SubprocessTimeoutError("Audio decoding timed out", {"source": str(path)}) from exc
    if done.returncode != 0:
        stderr = done.stderr.decode("utf-8", "replace").strip() or "no decodable audio stream"
        raise MediaCorruptError(
            f"FFmpeg audio decode failed: {stderr}; the file may have no audio track (probe_video reports has_audio)",
            {"source": str(path)},
        )
    return np.frombuffer(done.stdout[: len(done.stdout) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def _wav_base64(samples: np.ndarray) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ask(account: str, token: str, audio_b64: str, options: Dict[str, str]) -> Tuple[Dict[str, float], int]:
    """One Clef-omni choice question. Returns ({option id: chance}, input tokens)."""
    body = {
        "model": MODEL,
        "state": "A short sound clip. Name the sound that is heard.",
        "audio": [{"content_type": "audio/wav", "base64": audio_b64}],
        "questions": {"sound": {"type": "choice", "instructions": "Which sound is heard in this clip?", "criteria": options}},
    }
    request = urllib.request.Request(
        f"{api_base()}/accounts/{account}/ai/run/@cf/cloudflare/{MODEL}", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as reply:
                payload = json.loads(reply.read())
            break
        except urllib.error.HTTPError as exc:
            if attempt == 1 and exc.code in (429, 500, 502, 503, 504):
                time.sleep(2.0)
                continue
            detail = exc.read()[:300].decode("utf-8", "replace")
            if exc.code in (401, 403):
                forget_command_token()
            hint = " The token or account id is wrong or lacks Workers AI access." if exc.code in (401, 403) else ""
            raise CloudCallError(f"Cloudflare Workers AI answered HTTP {exc.code}.{hint}", {"body": detail}) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 1:
                time.sleep(2.0)
                continue
            raise CloudCallError(f"Cloudflare Workers AI could not be reached: {hide_token(str(exc), token)}") from exc
        except ValueError as exc:  # a bad header or reply; its text can quote the request, so only the type is shown
            raise CloudCallError(f"Cloudflare Workers AI request failed ({type(exc).__name__})") from None
    result = payload.get("result") or {}
    answer = (result.get("answers") or {}).get("sound") or {}
    chances = answer.get("probabilities")
    if not isinstance(chances, dict) or not chances:
        raise CloudCallError("Cloudflare Workers AI returned no label chances", {"reply": json.dumps(payload)[:300]})
    return {key: float(value) for key, value in chances.items()}, int((result.get("usage") or {}).get("input_tokens") or 0)


def _check_labels(labels: Optional[Sequence[str]]) -> List[str]:
    if labels is None:
        return list(DEFAULT_LABELS)
    cleaned = [str(label).strip() for label in labels]
    if not cleaned or any(not label for label in cleaned) or len(set(cleaned)) != len(cleaned):
        raise ValueError("labels must be a non-empty list of distinct, non-empty sound names, e.g. [\"dog barking\", \"siren\"]")
    if len(cleaned) > MAX_LABELS:
        raise ValueError(f"labels has {len(cleaned)} entries; at most {MAX_LABELS} are allowed. Ask about the most likely sounds first.")
    return cleaned


def label_sounds(
    source: str,
    start_sec: float,
    end_sec: float,
    labels: Optional[Sequence[str]] = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> Dict[str, Any]:
    """Per-window chances for each label over [start_sec, end_sec] of the source's first audio track."""
    labels = _check_labels(labels)
    if not (math.isfinite(start_sec) and math.isfinite(end_sec) and math.isfinite(window_sec)):
        raise ValueError("start_sec, end_sec and window_sec must be finite numbers")
    if start_sec < 0 or end_sec - start_sec < MIN_WINDOW_SEC:
        raise ValueError(f"end_sec must be at least {MIN_WINDOW_SEC:g} s after start_sec (got start_sec={start_sec:g}, end_sec={end_sec:g})")
    if not MIN_WINDOW_SEC <= window_sec <= MAX_WINDOW_SEC:
        raise ValueError(f"window_sec must be from {MIN_WINDOW_SEC:g} to {MAX_WINDOW_SEC:g} (got {window_sec:g})")
    count = math.ceil((end_sec - start_sec) / window_sec - 1e-9)
    if count > MAX_WINDOWS:
        raise ExtractionBoundsError(
            f"{count} windows requested; at most {MAX_WINDOWS} per call. Raise window_sec or narrow start_sec/end_sec "
            "(find the candidate range first with audio_onsets or loudness_timeline).",
            {"windows": count, "max_windows": MAX_WINDOWS},
        )
    account, token = _credentials()
    samples = _decode_range(source, start_sec, end_sec)
    size = int(round(window_sec * SAMPLE_RATE))
    jobs: List[Tuple[float, float, np.ndarray]] = []
    for index in range(count):
        chunk = samples[index * size:(index + 1) * size]
        if len(chunk) >= MIN_WINDOW_SEC * SAMPLE_RATE:
            begin = start_sec + index * window_sec
            jobs.append((begin, begin + len(chunk) / SAMPLE_RATE, chunk))
    if not jobs:
        raise MediaCorruptError("The audio track has no sound inside the requested range", {"source": source})
    ids = {f"o{i}": label for i, label in enumerate(labels)}
    options = {**ids, "none": NONE_LABEL}

    def run(job: Tuple[float, float, np.ndarray]) -> Tuple[Dict[str, float], int]:
        return _ask(account, token, _wav_base64(job[2]), options)

    with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
        answers = list(pool.map(run, jobs))
    windows = []
    tokens = 0
    for (begin, end, _), (chances, used) in zip(jobs, answers):
        tokens += used
        named = {ids[key]: round(chances.get(key, 0.0), 4) for key in ids}
        none_chance = round(chances.get("none", 0.0), 4)
        best = max(named, key=named.get)
        windows.append({
            "start_sec": round(begin, 3), "end_sec": round(end, 3),
            "top_label": best if named[best] >= none_chance else None,
            "top_chance": named[best], "none_chance": none_chance, "chances": named,
        })
    return {
        "source_path": str(Path(source).expanduser().resolve()), "start_sec": start_sec, "end_sec": end_sec,
        "window_sec": window_sec, "model": MODEL, "labels": labels, "windows": windows, "input_tokens": tokens,
        "note": "Audio was sent to Cloudflare Workers AI. Chances are the model's own and sum to 1 with none_chance; "
                "top_label is null when 'none of these' is more likely than every label. Sounds outside `labels` are forced "
                "onto the nearest label or none, so give every sound that may be present.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Name the sounds in a video's audio windows (sends audio to Cloudflare Workers AI).")
    parser.add_argument("source")
    parser.add_argument("start_sec", type=float)
    parser.add_argument("end_sec", type=float)
    parser.add_argument("--labels", help="comma-separated sound names (default: 20 common sounds)")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    args = parser.parse_args(argv)
    labels = None if args.labels is None else args.labels.split(",")
    try:
        print(json.dumps(label_sounds(args.source, args.start_sec, args.end_sec, labels, args.window_sec), indent=2))
    except (AgenticVideoError, ValueError) as exc:
        print(f"{type(exc).__name__}: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
