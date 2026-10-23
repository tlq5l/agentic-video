"""Cloudflare Workers AI calls and credentials shared by watch_for and label_sounds.

One token cache per server process, so the token command (it runs wrangler, which is slow, and two
wrangler refreshes at once can log the user out) runs once per refresh, not once per tool.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from src.core.errors import AgenticVideoError

API_BASE = "https://api.cloudflare.com/client/v4"
COMMAND_TOKEN_MAX_AGE_SEC = 50 * 60  # wrangler sign-in tokens expire after about an hour
COMMAND_TIMEOUT_SEC = 30
# Executor gives up on a tool call after 60 s. No new request starts after DEFAULT_DEADLINE_SEC; a request
# already sent is cut off REQUEST_GRACE_SEC later, so a call answers inside the limit even when Cloudflare is slow.
DEFAULT_DEADLINE_SEC = 40.0
REQUEST_GRACE_SEC = 8.0
MIN_REQUEST_SEC = 1.0  # a request is not started with less time left than this
ERROR_EXCERPT_CHARS = 300

_command_token: Optional[str] = None  # last line printed by AGENTIC_VIDEO_CF_TOKEN_COMMAND, kept in memory only
_command_token_at = 0.0  # time.monotonic() when it was printed
_lock = threading.Lock()  # one token command at a time, so two tools never refresh together
_TOKEN_SHAPE = re.compile(r"[\x21-\x7e]+")  # visible ASCII only: anything else cannot go in a header


class CloudCredentialsError(AgenticVideoError):
    """Raised when the Cloudflare token or account id is not configured."""


class CloudCallError(AgenticVideoError):
    """Raised when a Workers AI request fails."""


def credentials(feature: str, sends: str) -> Tuple[str, str]:
    """(account id, token). The token comes from CLOUDFLARE_API_TOKEN or the last line the
    AGENTIC_VIDEO_CF_TOKEN_COMMAND prints (plain `wrangler auth token` prints a banner first).

    The command's token is kept for later calls by either tool for up to 50 minutes; a 401 or 403 from
    Cloudflare drops it sooner. A token that is not one run of visible ASCII is refused without being shown.
    """
    global _command_token, _command_token_at
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    command = os.environ.get("AGENTIC_VIDEO_CF_TOKEN_COMMAND", "").strip()
    if not token and command:
        with _lock:
            if not (_command_token and time.monotonic() - _command_token_at < COMMAND_TOKEN_MAX_AGE_SEC):
                _command_token, _command_token_at = _run_token_command(command), time.monotonic()
            token = _command_token
    missing = [name for name, value in (("CLOUDFLARE_ACCOUNT_ID", account), ("CLOUDFLARE_API_TOKEN", token)) if not value]
    if missing:
        raise CloudCredentialsError(
            f"{feature} sends {sends} to Cloudflare Workers AI and needs " + " and ".join(missing)
            + " in the server's environment (a token with Workers AI access; AGENTIC_VIDEO_CF_TOKEN_COMMAND may "
            "name a command that prints the token instead of CLOUDFLARE_API_TOKEN). Nothing was sent."
        )
    if not _TOKEN_SHAPE.fullmatch(token):
        forget_command_token()
        raise CloudCredentialsError(
            "The Cloudflare token has spaces or characters that cannot go in an HTTP header (the value is not shown). "
            "Nothing was sent."
        )
    return account, token


def api_base() -> str:
    """Cloudflare API root; AGENTIC_VIDEO_CF_API_BASE points tests at a local stand-in."""
    return os.environ.get("AGENTIC_VIDEO_CF_API_BASE", API_BASE).rstrip("/")


def _run_token_command(command: str) -> str:
    try:
        done = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SEC, check=False)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise CloudCredentialsError(f"AGENTIC_VIDEO_CF_TOKEN_COMMAND could not run: {type(exc).__name__}") from exc
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or not lines:
        raise CloudCredentialsError(f"AGENTIC_VIDEO_CF_TOKEN_COMMAND exited {done.returncode} without printing a token")
    return lines[-1]


def forget_command_token() -> None:
    """Drop the kept token (Cloudflare refused it); the next call runs the command again."""
    global _command_token
    _command_token = None


def hide_token(text: str, token: str) -> str:
    """`text` with the token replaced, for messages built from an exception that may quote a request header."""
    return text.replace(token, "[token]") if token else text


def reply_excerpt(raw: bytes | str, token: str) -> str:
    """The start of a Cloudflare reply for an error message. The token is replaced before the cut, so a
    token that runs past the cut is not left in part."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    return hide_token(text, token)[:ERROR_EXCERPT_CHARS]


def run_model(model: str, account: str, token: str, body: bytes, timeout_sec: float, cutoff: Optional[float] = None) -> bytes:
    """POST one Workers AI run and return the raw reply.

    One retry after a 429, 5xx or network error, never past `cutoff` (a time.monotonic() value; None means no
    limit). A 401 or 403 drops the kept command token. Every error text has the token removed.
    """
    request = urllib.request.Request(
        f"{api_base()}/accounts/{account}/ai/run/@cf/cloudflare/{model}", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    failure = "no reply"
    for attempt in range(2):
        timeout = timeout_sec if cutoff is None else min(timeout_sec, cutoff - time.monotonic())
        if timeout < MIN_REQUEST_SEC:
            failure = "out of time before the call's limit" if attempt == 0 else f"{failure} (no time left to retry)"
            break
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return reply.read()
        except urllib.error.HTTPError as exc:
            authorization = exc.code in (401, 403)
            hint = " (the token or account id is wrong, or the token lacks Workers AI access)" if authorization else ""
            failure = f"HTTP {exc.code}{hint}: {reply_excerpt(exc.read(), token)}"
            if authorization:
                forget_command_token()
                break
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failure = f"{type(exc).__name__}: {hide_token(str(exc), token)[:ERROR_EXCERPT_CHARS]}"
        if attempt == 0:
            time.sleep(2.0 if cutoff is None else max(0.0, min(2.0, cutoff - time.monotonic())))
    raise CloudCallError(f"Cloudflare Workers AI request failed: {failure}")
