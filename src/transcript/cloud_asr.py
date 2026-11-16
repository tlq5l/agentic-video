"""Cloud transcription: whisper-large-v3-turbo on Cloudflare Workers AI in place of the local model, on the hosts
that are switched on (owner decision 2026-11-16; docs/features/cloud-asr.md).

Switched on only where this machine's hostname is named by AGENTIC_VIDEO_CLOUD_ASR (comma list), or, when that env
is empty, by "hosts" in the settings file. Public users never get it by default.

In cloud mode every fresh ASR run goes to Workers AI and no local model runs, not even after a cloud failure.
From ENDS_AT on, no new run is sent (Workers AI billing starts 2026-12-01). Cached transcripts still work.
"""

from __future__ import annotations

import base64
import functools
import json
import os
import socket
import string
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import ASRModelError
from src.media import cloudflare
from src.transcript import shadow_cf_asr

MODEL = "@cf/openai/whisper-large-v3-turbo"
FEATURE = "Cloud transcription"  # names the feature in credential errors
SENDS = "audio"
SWITCH_ENV = "AGENTIC_VIDEO_CLOUD_ASR"
MODEL_KEY = "cloud"  # the model name used for the cache, the progress estimate and the lock record
ENDS_AT = datetime(2026, 11, 30, 17, 0, tzinfo=timezone.utc)  # 2026-12-01 00:00 +07
ENDS_AT_LABEL = "2026-12-01 00:00 +07"
PIECE_SEC = 600
PARALLEL_PIECES = 4
PIECE_TIMEOUT_SEC = 120.0
REPEAT_MIN_SEGMENTS = 3  # a reply looks hallucinated only with at least this many segments
REPEAT_MAX_WORDS = 4  # ...and its most common normalized text has at most this many words
REPEAT_MIN_SHARE = 0.6  # ...and covers at least this share of the segments
LOCAL_OFF = "Local transcription is off on this machine, so no local model ran."
NOTICE_SUFFIX = " Transcribed by Cloudflare Workers AI (whisper-large-v3-turbo); the audio was sent to Cloudflare."


def settings_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    return Path(base) / "agentic-video" / "cloud-asr.json"


def _settings() -> Dict[str, Any]:
    """The settings file. Missing or invalid: {}. Only string and list-of-string values are used."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value for key, value in data.items()
        if isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))
    }


def _setting_text(key: str) -> Optional[str]:
    value = _settings().get(key)
    return value if isinstance(value, str) else None


def _hosts() -> set:
    raw = os.environ.get(SWITCH_ENV, "").strip()
    if raw:
        names = raw.split(",")
    else:
        value = _settings().get("hosts", [])
        names = [value] if isinstance(value, str) else value
    return {name.strip() for name in names if name.strip()}


def selected() -> bool:
    """True on a host named for cloud transcription. Cheap: no network, no token command."""
    return socket.gethostname() in _hosts()


def ended(now: Optional[float] = None) -> bool:
    return (time.time() if now is None else now) >= ENDS_AT.timestamp()


def refusal_message() -> str:
    return (
        f"Cloud transcription ended at {ENDS_AT_LABEL}, when Workers AI billing starts (owner decision 2026-11-16). "
        "Local transcription is off on this machine (no fallback), so nothing was sent and no local model ran. "
        "Cached transcripts and subtitle sidecars still work."
    )


def _credentials() -> Tuple[str, str]:
    # require_switch=False: this feature has its own switch (selected()); WORKERS_AI_ON stays False for the others.
    return cloudflare.credentials(
        FEATURE, SENDS, require_switch=False,
        account_fallback=_setting_text("account_id"), command_fallback=_setting_text("token_command"),
    )


def _normal(text_value: str) -> str:
    """Lowercase, with punctuation and spaces at the ends removed, for comparing segment texts."""
    return text_value.lower().strip(string.punctuation + string.whitespace)


def _repeated_phrase(segments: List[Dict[str, Any]]) -> Optional[str]:
    """The short phrase a reply keeps repeating ("Thank you." on a quiet piece), or None if it does not look hallucinated."""
    if len(segments) < REPEAT_MIN_SEGMENTS:
        return None
    phrase, count = Counter(_normal(seg["text"]) for seg in segments).most_common(1)[0]
    if len(phrase.split()) <= REPEAT_MAX_WORDS and count / len(segments) >= REPEAT_MIN_SHARE:
        return phrase
    return None


def _send_piece(index: int, path: str, *, account: str, token: str) -> Dict[str, Any]:
    """One piece with the voice filter off. If its reply looks hallucinated, the same piece is sent once more with the
    filter on, and that reply is used only if it has a line other than the repeated phrase. A failed retry keeps the first reply."""
    first = _request_piece(index, path, account=account, token=token, vad_filter=False)
    if "error" in first:
        return first
    phrase = _repeated_phrase(first["segments"])
    if phrase is None:
        return first
    filtered = _request_piece(index, path, account=account, token=token, vad_filter=True)
    if "error" not in filtered and any(_normal(seg["text"]) != phrase for seg in filtered["segments"]):
        return filtered
    return first


def _request_piece(index: int, path: str, *, account: str, token: str, vad_filter: bool) -> Dict[str, Any]:
    """One piece through Workers AI: its segments with offsets added, or its error (token removed)."""
    try:
        with open(path, "rb") as handle:
            body = {"audio": base64.b64encode(handle.read()).decode("ascii"), "vad_filter": vad_filter}
        raw = cloudflare.run_model(MODEL, account, token, json.dumps(body).encode("utf-8"), PIECE_TIMEOUT_SEC,
                                   allowed=lambda: not ended())
        reply = json.loads(raw)
        if not isinstance(reply, dict) or reply.get("success") is not True or not isinstance(reply.get("result"), dict):
            raise ValueError("the reply did not report success")
        result = reply["result"]
        offset = float(index * PIECE_SEC)
        segments = []
        for seg in result.get("segments") or []:
            text_value = str(seg["text"]).strip()
            if not text_value:
                continue
            start = round(float(seg["start"]) + offset, 3)
            end = round(max(float(seg["end"]) + offset, start), 3)
            prob = seg.get("no_speech_prob")
            confidence = (
                round(max(0.0, min(1.0, 1.0 - float(prob))), 3)
                if isinstance(prob, (int, float)) and not isinstance(prob, bool) else None
            )
            segments.append({"start_sec": start, "end_sec": end, "text": text_value, "confidence": confidence})
        info = result.get("transcription_info")
        language = info.get("language") if isinstance(info, dict) else None
        return {"segments": segments, "language": language if isinstance(language, str) and language else None}
    except Exception as exc:  # recorded for this piece; the transcript fails as a whole
        return {"error": cloudflare.hide_token(f"{type(exc).__name__}: {exc}", token)}


def transcribe(source_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Send the audio to Workers AI in pieces. Returns (segments with id, start_sec, end_sec, text, confidence; language).
    Any piece error raises ASRModelError: no partial transcript, no local model."""
    if ended():
        raise ASRModelError(refusal_message())
    try:
        account, token = _credentials()
    except cloudflare.CloudCredentialsError as exc:
        raise ASRModelError(f"{exc} {LOCAL_OFF}") from exc
    with tempfile.TemporaryDirectory(prefix="agentic-video-cloud-asr-") as temp_dir:
        try:
            pieces = shadow_cf_asr._cut_pieces(source_path, temp_dir, PIECE_SEC)
        except OSError as exc:
            raise ASRModelError(f"FFmpeg could not run: {type(exc).__name__}. {LOCAL_OFF}") from exc
        if pieces is None:
            raise ASRModelError(f"FFmpeg could not decode the audio stream. {LOCAL_OFF}")
        send = functools.partial(_send_piece, account=account, token=token)
        with ThreadPoolExecutor(max_workers=PARALLEL_PIECES) as pool:
            replies = list(pool.map(send, range(len(pieces)), pieces))
    for index, reply in enumerate(replies):
        if "error" in reply:
            if ended():
                raise ASRModelError(refusal_message())
            raise ASRModelError(
                f"Cloud transcription failed on piece {index + 1} of {len(replies)}: {reply['error']}. {LOCAL_OFF}",
                {"piece_index": index, "piece_count": len(replies)},
            )
    segments = [seg for reply in replies for seg in reply["segments"]]
    segments.sort(key=lambda seg: seg["start_sec"])
    for number, seg in enumerate(segments, 1):
        seg["id"] = number
    languages = Counter(reply["language"] for reply in replies if reply.get("language"))
    language = languages.most_common(1)[0][0] if languages else "en"
    return segments, language
