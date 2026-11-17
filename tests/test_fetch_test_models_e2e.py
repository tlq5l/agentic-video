"""scripts/fetch_test_models.py, the merge recipe's setup step, run as setup runs it on a copy of this machine's cached tiny
Whisper model, with the Hugging Face endpoint pointed at a closed port so any download attempt fails (no network), or at a
local stand-in for the Hub that serves this machine's cached files (and can answer 429, too many requests)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_test_models.py"


def _copy_cache(tmp_path):
    """A copy of the cached Systran/faster-whisper-tiny folder under a new HF_HOME, its snapshot files copied as plain files
    (the hub keeps their blobs outside the model folder); returns (HF_HOME, the copy's model.bin)."""
    from faster_whisper.utils import download_model

    try:
        snapshot = Path(download_model("tiny", local_files_only=True))
    except Exception:
        pytest.skip("tiny Whisper model not cached here; the recipe's setup step fetches it")
    home = tmp_path / "hf"
    model_dir = home / "hub" / snapshot.parents[1].name
    shutil.copytree(snapshot.parents[1], model_dir)
    return home, model_dir / "snapshots" / snapshot.name / "model.bin"


def _setup(home, endpoint="http://127.0.0.1:9", **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "WHISPER_TINY_SEED", "WHISPER_TINY_RETRY_WAITS")}
    env.update(HF_HOME=str(home), HF_ENDPOINT=endpoint, **extra)
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300)


class _Hub:
    """A local stand-in for the Hugging Face Hub serving this machine's cached tiny Whisper files (the four the script
    fetches) at the cached commit: the model-info call, the file list, then a HEAD and a GET per file, over plain HTTP (the
    tests set HF_HUB_DISABLE_XET=1, so the hub client asks for no Xet token). Its first `too_many` model-info calls get 429
    with Retry-After 0 (-1: every call). `requests` lists (method, path, status) as they arrive."""

    def __init__(self, too_many=0):
        from faster_whisper.utils import download_model

        try:
            snapshot = Path(download_model("tiny", local_files_only=True))
        except Exception:
            pytest.skip("tiny Whisper model not cached here; the recipe's setup step fetches it")
        self.commit = snapshot.name
        self.files = {f.name: f.read_bytes() for f in snapshot.iterdir() if f.is_file()}
        # the ETag the real Hub gave each file: the name of the blob its snapshot link points to
        self.etags = {f.name: Path(os.readlink(f)).name if f.is_symlink() else hashlib.sha256(self.files[f.name]).hexdigest()
                      for f in snapshot.iterdir() if f.is_file()}
        self.too_many, self.requests, self.times = too_many, [], []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, body):
                path = self.path.split("?")[0]
                file = re.fullmatch(r"/Systran/faster-whisper-tiny/resolve/[^/]+/(.+)", path)
                if re.fullmatch(r"/api/models/Systran/faster-whisper-tiny/revision/[^/]+", path):
                    if hub.too_many != 0:
                        hub.too_many -= 1
                        return self._send(path, 429, b'{"error": "too many requests"}', {"Retry-After": "0"}, body)
                    info = {"id": "Systran/faster-whisper-tiny", "sha": hub.commit,
                            "siblings": [{"rfilename": name} for name in sorted(hub.files)]}
                    return self._send(path, 200, json.dumps(info).encode(), {"Content-Type": "application/json"}, body)
                if re.fullmatch(r"/api/models/Systran/faster-whisper-tiny/tree/[^/]+", path):
                    tree = [{"type": "file", "path": name, "size": len(data), "oid": hashlib.sha1(data).hexdigest()}
                            for name, data in sorted(hub.files.items())]
                    return self._send(path, 200, json.dumps(tree).encode(), {"Content-Type": "application/json"}, body)
                if file and file.group(1) in hub.files:
                    data = hub.files[file.group(1)]
                    headers = {"X-Repo-Commit": hub.commit, "ETag": '"' + hub.etags[file.group(1)] + '"'}
                    return self._send(path, 200, data, headers, body)
                return self._send(path, 404, b"{}", {"X-Error-Code": "EntryNotFound"}, body)

            def _send(self, path, status, data, headers, body):
                hub.requests.append((self.command, path, status))
                hub.times.append(time.monotonic())
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if body:
                    self.wfile.write(data)

            def do_HEAD(self):
                self._answer(False)

            def do_GET(self):
                self._answer(True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hub():
    made = []

    def start(too_many=0):
        made.append(_Hub(too_many))
        return made[-1]

    yield start
    for one in made:
        one.close()


def test_a_complete_cached_model_passes_setup_with_no_network(tmp_path):
    """Cache intact, no network: setup uses the cached copy (no download tried, or the closed port would fail it)."""
    home, _ = _copy_cache(tmp_path)
    run = _setup(home)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.startswith("WHISPER_TINY cached 78203619 bytes")


def test_a_half_downloaded_model_does_not_count_as_cached(tmp_path):
    """model.bin cut to half its size (a download cut off half way): setup does not report it cached; it tries a fresh
    download, which the closed port fails, so setup fails (round-17 probe: before this check it printed
    "WHISPER_TINY cached 40434484 bytes" and exited 0)."""
    home, model = _copy_cache(tmp_path)
    os.chmod(model, 0o644)
    with open(model, "r+b") as handle:
        handle.truncate(75538270 // 2)
    run = _setup(home)
    assert "WHISPER_TINY cached" not in run.stdout
    assert "WHISPER_TINY fetching: model.bin 37769135 bytes, expected 75538270" in run.stdout
    assert run.returncode != 0


def test_a_half_downloaded_model_is_fetched_again(tmp_path, hub):
    """model.bin cut to half as a cut-off download leaves it (the snapshot link into the model's blobs folder, the blob half
    written), the Hub reachable (a local stand-in): setup downloads the files again over the half one and passes. Round-17
    breaker: with force_download off the hub keeps the half file (37769135 bytes) and setup exits 1, a change the two tests
    above missed because the closed port fails the download either way."""
    served = hub()
    home, model = _copy_cache(tmp_path)
    blob = model.parents[2] / "blobs" / served.etags["model.bin"]
    model.unlink()
    model.symlink_to(Path("..", "..", "blobs", blob.name))
    os.chmod(blob, 0o644)
    with open(blob, "r+b") as handle:
        handle.truncate(75538270 // 2)
    run = _setup(home, served.url, HF_HUB_DISABLE_XET="1")
    assert run.returncode == 0, run.stdout + run.stderr + repr(served.requests)
    assert "WHISPER_TINY fetching: model.bin 37769135 bytes, expected 75538270" in run.stdout
    assert "WHISPER_TINY downloaded 78203619 bytes" in run.stdout
    assert model.stat().st_size == 75538270
    assert ("GET", f"/Systran/faster-whisper-tiny/resolve/{served.commit}/model.bin", 200) in served.requests


def _cut_model_off(tmp_path, served):
    """The cache copy with model.bin cut to half, the way a cut-off download leaves it (a link into the blobs folder, the
    blob half written); returns the HF_HOME to set up with."""
    home, model = _copy_cache(tmp_path)
    blob = model.parents[2] / "blobs" / served.etags["model.bin"]
    model.unlink()
    model.symlink_to(Path("..", "..", "blobs", blob.name))
    os.chmod(blob, 0o644)
    with open(blob, "r+b") as handle:
        handle.truncate(75538270 // 2)
    return home


def _revision_statuses(served):
    return [status for method, path, status in served.requests if "/revision/" in path]


def test_setup_tries_again_when_the_hub_answers_429(tmp_path, hub):
    """Nothing cached, the Hub answers its first two info calls with 429 (too many requests): setup waits 0.4 s each time,
    a real wait so a removed sleep shows, then tries again and downloads the model. Round-17 breaker box log
    (r17/break-m2.log): setup exited 1 on the first 429."""
    served = hub(too_many=2)
    run = _setup(tmp_path / "hf", served.url, HF_HUB_DISABLE_XET="1", WHISPER_TINY_RETRY_WAITS="0.4,0.4,0.4,0.4")
    assert run.returncode == 0, run.stdout + run.stderr + repr(served.requests)
    assert run.stdout.count("WHISPER_TINY hub answered 429 (too many requests), next try in 0.4 s") == 2
    assert "WHISPER_TINY downloaded 78203619 bytes" in run.stdout
    assert _revision_statuses(served)[:3] == [429, 429, 200]
    waited = [stamp for (method, path, status), stamp in zip(served.requests, served.times) if status == 429]
    assert waited[1] - waited[0] >= 0.4, waited


def test_setup_stops_after_the_last_wait_when_the_hub_keeps_answering_429(tmp_path, hub):
    """Every info call answered 429: setup tries 5 times (4 waits), then exits 1 with no download."""
    served = hub(too_many=-1)
    run = _setup(tmp_path / "hf", served.url, HF_HUB_DISABLE_XET="1", WHISPER_TINY_RETRY_WAITS="0,0,0,0")
    assert run.returncode == 1, run.stdout + run.stderr + repr(served.requests)
    assert "WHISPER_TINY hub still answered 429 after 5 tries" in run.stdout
    assert _revision_statuses(served) == [429] * 5
    assert not [path for method, path, status in served.requests if "/resolve/" in path], served.requests


def test_a_429_while_a_broken_copy_is_cached_is_waited_out(tmp_path, hub):
    """model.bin cut to half in the cache and the Hub answering 429 on every info call. The cached copy used to come back
    with no error, so setup exited 1 at once with no 429 line (round-18 box probe). Now it waits the 4 waits, then says 429."""
    served = hub(too_many=-1)
    home = _cut_model_off(tmp_path, served)
    run = _setup(home, served.url, HF_HUB_DISABLE_XET="1", WHISPER_TINY_RETRY_WAITS="0,0,0,0")
    assert run.returncode == 1, run.stdout + run.stderr + repr(served.requests)
    assert "WHISPER_TINY hub still answered 429 after 5 tries" in run.stdout
    assert _revision_statuses(served) == [429] * 5


def test_a_good_seed_passes_setup_with_no_call_to_the_hub(tmp_path, hub):
    """No cache, the Hub answering 429 on every call, and a seed folder holding the four good files: setup copies the seed
    into the cache and passes, with no request to the Hub. Short waits, so a setup that ignores the seed fails in seconds, not after
    the default 225 s (round-19 breaker: the seed-install-no-refs mutant was caught only after the default waits)."""
    served = hub(too_many=-1)
    seed = tmp_path / "seed"
    seed.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (seed / name).write_bytes(served.files[name])
    run = _setup(tmp_path / "hf", served.url, HF_HUB_DISABLE_XET="1", WHISPER_TINY_SEED=str(seed), WHISPER_TINY_RETRY_WAITS="0,0,0,0")
    assert run.returncode == 0, run.stdout + run.stderr + repr(served.requests)
    assert "WHISPER_TINY seed 78203619 bytes" in run.stdout
    assert served.requests == [], served.requests


def test_a_bad_seed_is_not_used_and_a_fresh_download_refills_it(tmp_path, hub):
    """No cache, a seed whose model.bin is cut short, the Hub reachable: setup skips the seed, downloads the model, and
    copies the four good files over the seed."""
    served = hub()
    seed = tmp_path / "seed"
    seed.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (seed / name).write_bytes(served.files[name])
    (seed / "model.bin").write_bytes(served.files["model.bin"][:1000])
    run = _setup(tmp_path / "hf", served.url, HF_HUB_DISABLE_XET="1", WHISPER_TINY_SEED=str(seed))
    assert run.returncode == 0, run.stdout + run.stderr + repr(served.requests)
    assert "WHISPER_TINY downloaded 78203619 bytes" in run.stdout
    assert "WHISPER_TINY seed filled at " in run.stdout
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        assert (seed / name).read_bytes() == served.files[name], name


def test_an_empty_retry_wait_setting_is_the_default(tmp_path):
    """WHISPER_TINY_RETRY_WAITS="" used to crash at import (float of an empty string). Empty now means the default waits:
    with nothing cached and the Hub unreachable, setup gets as far as the download and fails there."""
    run = _setup(tmp_path / "hf", WHISPER_TINY_RETRY_WAITS="")
    assert run.returncode != 0
    assert "WHISPER_TINY fetching: not cached" in run.stdout, run.stdout + run.stderr


def _waits(**extra):
    """The WAITS list the script computes when it is imported, read in a child process, and the warning it printed."""
    env = {k: v for k, v in os.environ.items() if k not in ("WHISPER_TINY_SEED", "WHISPER_TINY_RETRY_WAITS")}
    env.update(extra)
    code = "import importlib.util as u, sys; s = u.spec_from_file_location('fetch', sys.argv[1]); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.WAITS)"
    run = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return run.stdout.strip(), run.stderr


@pytest.mark.parametrize("waits, expected", [("15,", "[15.0]"), (",15", "[15.0]"), ("15,,30", "[15.0, 30.0]")])
def test_retry_waits_with_empty_items_start_and_use_the_given_waits(tmp_path, waits, expected):
    """WHISPER_TINY_RETRY_WAITS with an empty item (a stray comma) crashed the app at startup: float of an empty string
    (round-19 breaker). The empty items are skipped: setup starts on the cached copy, with no warning, and uses the given waits."""
    home, _ = _copy_cache(tmp_path)
    run = _setup(home, WHISPER_TINY_RETRY_WAITS=waits)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.startswith("WHISPER_TINY cached 78203619 bytes")
    assert "RETRY_WAITS" not in run.stderr, run.stderr
    seen, stderr = _waits(WHISPER_TINY_RETRY_WAITS=waits)
    assert seen == expected, seen
    assert "RETRY_WAITS" not in stderr, stderr


@pytest.mark.parametrize("waits", ["abc", "-5", ","])
def test_bad_retry_waits_print_one_warning_and_use_the_default_waits(tmp_path, waits):
    """A value that is not seconds 0 or more ("abc", "-5", or only commas) crashed the app at startup or failed later at the
    first sleep (round-19 breaker). Now setup starts on the cached copy, prints one warning line on stderr and waits the
    default 15, 30, 60, 120 s."""
    home, _ = _copy_cache(tmp_path)
    run = _setup(home, WHISPER_TINY_RETRY_WAITS=waits)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.startswith("WHISPER_TINY cached 78203619 bytes")
    assert len(run.stderr.splitlines()) == 1 and "RETRY_WAITS" in run.stderr, run.stderr
    seen, stderr = _waits(WHISPER_TINY_RETRY_WAITS=waits)
    assert seen == "[15.0, 30.0, 60.0, 120.0]", seen
    assert len(stderr.splitlines()) == 1, stderr


def test_setup_fails_at_once_on_errors_other_than_429(tmp_path):
    """Nothing cached and the Hub unreachable (closed port): setup fails with no retry; only a 429 is waited out."""
    run = _setup(tmp_path / "hf", WHISPER_TINY_RETRY_WAITS="0,0,0,0")
    assert run.returncode != 0
    assert "WHISPER_TINY fetching: not cached" in run.stdout
    assert "429" not in run.stdout, run.stdout
