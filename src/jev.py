"""Small urllib client for TypeSafe System One (Jev), with OpenRouter fallback."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional


PROVIDERS = (
    ("typesafe", "TYPESAFE_API_KEY", "https://api.typesafe.ai/v1/systemone", "jev-latest"),
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/systemone", "~typesafe/jev-latest"),
)


def load_api_keys() -> Dict[str, str]:
    """Read the provider keys from the environment at call time."""
    return {name: os.environ[name] for _, name, _, _ in PROVIDERS if os.environ.get(name)}


def scrub(text: Any, keys: Optional[Mapping[str, str]] = None) -> str:
    """Remove configured tokens and common key-shaped strings from untrusted errors."""
    result = str(text)
    for value in (keys or load_api_keys()).values():
        if value:
            result = result.replace(value, "[redacted]")
    result = re.sub(r"(?i)(bearer\s+)[^\s,;\"']+", r"\1[redacted]", result)
    result = re.sub(r"\bsk-[A-Za-z0-9_-]{6}[A-Za-z0-9_-]+", "[redacted]", result)
    return result


def _read_response(response: Any) -> Dict[str, Any]:
    raw = response.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Jev response must be a JSON object")
    return result


def _scrub_tree(value: Any, keys: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return scrub(value, keys)
    if isinstance(value, list):
        return [_scrub_tree(item, keys) for item in value]
    if isinstance(value, tuple):
        return [_scrub_tree(item, keys) for item in value]
    if isinstance(value, dict):
        return {scrub(key, keys): _scrub_tree(item, keys) for key, item in value.items()}
    return value


def decide(
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    *,
    timeout: float = 15.0,
    urlopen: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Ask Jev, trying TypeSafe first and OpenRouter if available after an error."""
    opener = urlopen or urllib.request.urlopen
    keys = load_api_keys()
    configured = [(name, key_name, url, model, keys.get(key_name, ""))
                  for name, key_name, url, model in PROVIDERS if keys.get(key_name, "")]
    if not configured:
        raise RuntimeError("No Jev API key is configured; set TYPESAFE_API_KEY or OPENROUTER_API_KEY")

    errors = []
    for name, key_name, url, model, key in configured:
        payload = json.dumps(
            {"state": _scrub_tree(state, keys), "model": model, "questions": _scrub_tree(questions, keys)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                body = _read_response(response)
            if status < 200 or status >= 300:
                raise RuntimeError("HTTP {}: {}".format(status, json.dumps(body, ensure_ascii=False)[:500]))
            body["provider"] = name
            body["model"] = model
            return body
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = "request failed"
            message = "{} HTTP {} {}".format(name, exc.code, detail)
            errors.append(scrub(message, keys))
        except Exception as exc:  # network, timeout, invalid JSON, and provider errors
            errors.append(scrub("{}: {}".format(name, exc), keys))

    raise RuntimeError("Jev request failed; fallback exhausted: " + "; ".join(errors))
