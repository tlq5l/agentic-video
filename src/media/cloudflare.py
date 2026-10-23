"""Cloudflare Workers AI credentials shared by watch_for and label_sounds.

One token cache per server process, so the token command (heavy: the wrangler path peaks near 250 MB, and two
wrangler refreshes at once can log the user out) runs once per refresh, not once per tool.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from typing import Optional, Tuple

from src.core.errors import AgenticVideoError

API_BASE = "https://api.cloudflare.com/client/v4"
COMMAND_TOKEN_MAX_AGE_SEC = 50 * 60  # wrangler sign-in tokens expire after about an hour
COMMAND_TIMEOUT_SEC = 30

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
