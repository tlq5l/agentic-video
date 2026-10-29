"""End-to-end checks of the Claude Desktop bundle (mcpb/manifest.json and scripts/build_mcpb.py).

No mocks: the build script really runs, the .mcpb is unzipped, and the server is
started with the command the manifest says and driven over stdio.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_mcpb.py"
SAMPLE_VIDEO = ROOT / "fixtures" / "reference_videos" / "sample_10s.mp4"
TOOL_CONTRACT = ROOT / "src" / "adapters" / "mcp_schemas.json"
# Folders the server also searches for ffmpeg on macOS (src/core/config.py).
MAC_FFMPEG_FOLDERS = [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path.home() / ".local" / "bin"]
EXPECTED_SERVER = {
    "type": "uv",
    "entry_point": "src/executor/mcp_server.py",
    "mcp_config": {
        "command": "uv",
        "args": ["run", "--directory", "${__dirname}", "agentic-video"],
    },
}


def _pyproject_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("pyproject.toml has no version line")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Build the bundle with scripts/build_mcpb.py, unzip it, and return (archive, unzipped folder)."""
    if shutil.which("npx") is None:
        pytest.skip("npx is missing; the bundle is validated and packed with npx")
    assert BUILD.is_file(), "scripts/build_mcpb.py does not exist; the Claude Desktop bundle cannot be built"
    result = subprocess.run(
        [sys.executable, str(BUILD)], cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, f"build_mcpb.py failed:\n{result.stdout}\n{result.stderr}"
    built = [line.split(" ", 1)[1].split(" (")[0] for line in result.stdout.splitlines() if line.startswith("built ")]
    assert len(built) == 1, f"build_mcpb.py did not print the output path:\n{result.stdout}"
    archive = Path(built[0])
    assert archive.is_file(), f"build_mcpb.py reported {archive}, which does not exist"
    extracted = tmp_path_factory.mktemp("unzipped") / "agentic-video"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    return archive, extracted


def _manifest_server_command(extracted: Path) -> list:
    """The command the host runs, read from the bundle's own manifest with ${__dirname} filled in."""
    manifest = json.loads((extracted / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["server"]["mcp_config"]
    return [config["command"], *(arg.replace("${__dirname}", str(extracted)) for arg in config["args"])]


def _talk_to_server(command: list, env: dict) -> dict:
    """Send initialize, initialized, tools/list and one probe_video call over stdio; return replies by id."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "probe_video", "arguments": {"source": str(SAMPLE_VIDEO)}}},
    ]
    stdin = "".join(json.dumps(message) + "\n" for message in messages)
    proc = subprocess.run(command, input=stdin, capture_output=True, text=True, env=env, timeout=900, check=False)
    replies = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            message = json.loads(line)
            replies[message.get("id")] = message
    assert {1, 2, 3} <= set(replies), f"server did not answer every request (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
    return replies


def _open_session(command: list, env: dict, stderr_log: Path):
    """Start the server with stdin kept open. A reader thread queues each reply as it arrives."""
    with stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                text=True, env=env)
    replies: queue.Queue = queue.Queue()

    def read_replies():
        for out_line in proc.stdout:
            if out_line.strip():
                replies.put(json.loads(out_line))

    threading.Thread(target=read_replies, daemon=True).start()
    return proc, replies


def _send(proc, message: dict) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _reply_to(replies: queue.Queue, request_id: int, timeout: float = 120) -> dict:
    """Return the reply with this id. Raises queue.Empty if the server stays silent for `timeout` seconds."""
    while True:
        reply = replies.get(timeout=timeout)
        if reply.get("id") == request_id:
            return reply


def test_manifest_fields_and_archive_contents(bundle):
    archive, _ = bundle
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["manifest_version"] == "0.4"
    assert manifest["name"] == "agentic-video"
    assert manifest["display_name"] == "agentic-video"
    assert manifest["version"] == _pyproject_version()
    assert manifest["description"]
    assert manifest["author"] == {"name": "tlq5l"}
    assert manifest["license"] == "MIT"
    assert manifest["homepage"] == "https://github.com/tlq5l/agentic-video"
    assert manifest["repository"]["url"] == "https://github.com/tlq5l/agentic-video"
    assert manifest["server"] == EXPECTED_SERVER
    assert manifest["compatibility"] == {"platforms": ["darwin"], "runtimes": {"python": ">=3.10"}}
    assert "user_config" not in manifest
    for forbidden in ("tests/", "fixtures/", ".venv/", "server/", ".git/"):
        assert not any(name.startswith(forbidden) for name in names), f"bundle contains {forbidden}"
    assert "src/executor/mcp_server.py" in names


def test_server_starts_from_the_manifest_and_lists_contract_tools(bundle):
    if shutil.which("uv") is None:
        pytest.skip("uv is missing; the manifest runs the server with uv")
    _, extracted = bundle
    replies = _talk_to_server(_manifest_server_command(extracted), dict(os.environ))

    listed = {tool["name"] for tool in replies[2]["result"]["tools"]}
    contract = set(json.loads(TOOL_CONTRACT.read_text(encoding="utf-8"))["tools"])
    assert listed == contract

    probe = replies[3]["result"]
    assert not probe.get("isError"), probe
    assert json.loads(probe["content"][0]["text"])["duration_sec"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.parametrize("missing, path_extra", [
    ("ffprobe", None),
    pytest.param("ffmpeg", "ffprobe-only",
                 marks=pytest.mark.skipif(shutil.which("ffprobe") is None, reason="no real ffprobe on this host to link to")),
])
def test_missing_ffmpeg_error_tells_the_user_the_fix(bundle, tmp_path, missing, path_extra):
    """Missing ffprobe (no ffmpeg or ffprobe on PATH), and missing ffmpeg alone (ffprobe linked, ffmpeg absent).

    Each case runs one server session: probe_video, then count_repetitions polled until it is no longer in_progress.
    The final error from the background job must carry the install fix once.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is missing; the manifest runs the server with uv")
    if any((folder / "ffprobe").exists() for folder in MAC_FFMPEG_FOLDERS):
        pytest.skip("ffprobe is installed in a folder the server searches on macOS, so it cannot go missing here")
    _, extracted = bundle
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    cache = os.environ.get("UV_CACHE_DIR") or str(Path.home() / ".cache" / "uv")
    # PATH holds only uv's folder: no ffmpeg there, and /usr/bin (which holds ffmpeg on some hosts) is left out.
    path_folders = [Path(uv).parent]
    if path_extra == "ffprobe-only":
        if shutil.which("ffmpeg", path=str(path_folders[0])) is not None:
            pytest.skip("ffmpeg is in uv's folder, so it cannot go missing here")
        only_ffprobe = tmp_path / "only-ffprobe"
        only_ffprobe.mkdir()
        (only_ffprobe / "ffprobe").symlink_to(shutil.which("ffprobe"))
        path_folders.append(only_ffprobe)
    env = {
        "PATH": os.pathsep.join(str(folder) for folder in path_folders),
        "HOME": str(empty_home),
        "UV_CACHE_DIR": cache,
        "UV_PYTHON": sys.executable,
    }
    proc, replies = _open_session(_manifest_server_command(extracted), env, tmp_path / "server-stderr.log")
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "test", "version": "0"}}})
        _reply_to(replies, 1)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "probe_video", "arguments": {"source": str(SAMPLE_VIDEO)}}})
        probe = _reply_to(replies, 2)["result"]
        if missing == "ffprobe":
            text = probe["content"][0]["text"]
            assert probe.get("isError") is True
            assert "ffprobe binary not found" in text
            assert "brew install ffmpeg" in text
            assert "restart Claude Desktop" in text
        else:
            assert not probe.get("isError"), probe  # ffprobe is linked here, so probe_video works

        count_params = {"name": "count_repetitions", "arguments": {"source": str(SAMPLE_VIDEO)}}
        deadline = time.monotonic() + 120
        request_id = 3
        while True:
            _send(proc, {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": count_params})
            reply_text = _reply_to(replies, request_id)["result"]["content"][0]["text"]
            if json.loads(reply_text)["status"] != "in_progress":
                break
            assert time.monotonic() < deadline, f"count_repetitions still in_progress after 120 s: {reply_text}"
            time.sleep(1)  # the job runs in a background thread; re-send the same call until it finishes
            request_id += 1
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert json.loads(reply_text)["status"] == "error", reply_text
    assert f"{missing} binary not found" in reply_text, reply_text
    assert reply_text.count("brew install ffmpeg") == 1, reply_text
    assert reply_text.count("restart Claude Desktop") == 1, reply_text
