"""End to end: every Workers AI tool refuses a bad or short call with the switched-off message before any other error.

Owner 2026-11-14: Workers AI is off. The refusal comes first, before argument checks, bounds checks, probe or extraction,
so a call that is also wrong still gets "switched off ... Nothing was sent". Each case runs the stdio MCP server with no
Cloudflare credentials and checks the reply text.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAD_ARGUMENT_WORDS = ("must be at least", "below the minimum", "above the maximum", "windows requested", "distinct",
                      "non-empty", "must be from", "start_sec must", "no audio track", "fewer than consecutive")


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory):
    """1 s clip with a sine track, so the audio tool has a track to read."""
    path = tmp_path_factory.mktemp("switched_off") / "one_second.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=10",
                    "-f", "lavfi", "-i", "sine=f=440:d=1:r=16000", "-shortest", str(path)], check=True)
    return path


def _call(name, arguments):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("CLOUDFLARE_") and key != "AGENTIC_VIDEO_CF_TOKEN_COMMAND"}
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    done = subprocess.run([sys.executable, "-m", "src.executor.mcp_server"], input=json.dumps(request) + "\n",
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=180, check=False)
    replies = {reply["id"]: reply for reply in map(json.loads, done.stdout.splitlines())}
    assert done.returncode == 0
    return replies[1]["result"]


CASES = [
    pytest.param("label_sounds", {"start_sec": 0, "end_sec": 1}, id="label-1s-default-window"),
    pytest.param("label_sounds", {"start_sec": 0, "end_sec": 0.5}, id="label-range-under-1s"),
    pytest.param("label_sounds", {"start_sec": 0, "end_sec": 1, "labels": []}, id="label-empty-labels"),
    pytest.param("label_sounds", {"start_sec": 0, "end_sec": 1, "window_sec": 0.2}, id="label-window-below-min"),
    pytest.param("label_sounds", {"start_sec": 0, "end_sec": 400}, id="label-too-many-windows"),
    pytest.param("watch_for", {"conditions": ["Is a dog visible?"], "interval_sec": 0.2}, id="watch-interval-below-min"),
    pytest.param("watch_for", {"conditions": []}, id="watch-no-conditions"),
    pytest.param("watch_for", {"conditions": ["a"], "start_sec": 0, "end_sec": 0.5}, id="watch-range-too-short"),
]


@pytest.mark.parametrize("tool, arguments", CASES)
def test_switched_off_refusal_comes_before_any_other_error(short_clip, tool, arguments):
    result = _call(tool, {"source": str(short_clip), **arguments})
    text = result["content"][0]["text"]
    assert result["isError"], result
    assert "switched off" in text and "Nothing was sent" in text, text
    assert not any(word in text for word in BAD_ARGUMENT_WORDS), text


@pytest.mark.parametrize("tool", ["label_sounds", "watch_for"])
def test_switched_off_refusal_comes_before_a_non_object_arguments_error(tool):
    """0.3.6 break-it: arguments sent as a list got the 'must be an object' error instead of the switched-off refusal."""
    result = _call(tool, [])
    text = result["content"][0]["text"]
    assert result["isError"], result
    assert "switched off" in text and "Nothing was sent" in text, text
