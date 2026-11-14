"""Pip-user matrix, end to end: build the wheel, install it with no extras (and with [visual] in the slow test), and call
every MCP tool through the installed CLI with good input, a missing optional package, and malformed arguments.

Every reply a pip user can see must be true for a pip install: no repo-only file or path, no .venv path, no "bug" wording,
no traceback, no hang. The visual extra's weights message must name the running Python. No Cloudflare call is made: the
Workers AI tools answer with the switched-off refusal.
"""

import json
import os
import re
import sys
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_tools = pytest.mark.skipif(shutil.which("uv") is None or shutil.which("ffmpeg") is None,
                                 reason="the pip matrix builds the wheel with uv and makes its clip with ffmpeg")

# Names a pip user cannot use. A parenthetical "source checkout: ..." note (the #81 wording) is allowed. Only error replies are
# checked for these names: a success reply carries paths of the caller's own files, which may contain any folder name.
REPO_ONLY = re.compile(r"requirements(?:-visual)?\.txt|scripts/|fixtures/|\.venv|build_mcpb|pytest|tests/|AGENTS\.md|mcpb"
                       r"|(?<![\w/.-])src/|\b[\w-]+\.py\b|uv (?:run|pip)|llms\.txt|pyproject")
SOURCE_CHECKOUT_NOTE = re.compile(r"\((?:in a )?source checkout:[^)]*\)")
PIP_HINT = "python -m pip install 'agentic-video[visual]'"
# (tool, case) cells whose reply must give the pip command for the visual extra.
PIP_HINT_CELLS = {("search_visual", "good"), ("build_visual_index", "good"),
                  ("count_occurrences", "good: description needs CLIP"), ("count_occurrences", "good: text (OCR)")}
# (tool, case) cells that need the CLIP weights; their reply must name the Python that runs the download command.
# count_occurrences with a CLIP description answers in_progress and reports the weights gap on a later poll, so it is not one call's reply.
WEIGHTS_CELLS = {("search_visual", "good"), ("build_visual_index", "good")}
VISUAL_TOOLS = ("search_visual", "build_visual_index", "count_occurrences")
OCR_TEXT_CELL = ("count_occurrences", "good: text (OCR)")  # off macOS its reply must say the tool runs only on macOS
SWITCHED_OFF_TOOLS = ("label_sounds", "watch_for")

REQUIRED = {
    "probe_video": ["source"], "transcribe": ["source"], "search_transcript": ["source", "query"],
    "build_visual_index": ["source"], "search_visual": ["source", "query"], "count_occurrences": ["source"],
    "count_repetitions": ["source"], "contact_sheet": ["source"], "review_video": ["source"], "overview_chunks": ["source"],
    "refine_temporal_beam": ["source", "windows"], "track_region_changes": ["source", "start_sec", "end_sec", "region"],
    "track_region_events": ["source", "start_sec", "end_sec", "region"], "inspect_moments": ["source", "moments"],
    "inspect_windows": ["source", "windows"], "inspect_window": ["source", "start_sec", "end_sec"],
    "frame_at": ["source", "time_sec"], "detect_cuts": ["source"], "check_frame_cadence": ["source"],
    "audio_onsets": ["source"], "loudness_timeline": ["source"], "spectrogram_window": ["source", "start_sec", "end_sec"],
    "sync_check": ["source"], "label_sounds": ["source", "start_sec", "end_sec", "labels"], "watch_for": ["source", "conditions"],
    "motion_open": ["source"], "motion_sheet": ["source"], "motion_stats": ["source", "box"], "motion_boil_map": ["source"],
    "motion_zoom": ["source", "box"], "motion_profile": ["source"], "motion_compare": ["source", "compare_to", "box"],
    "motion_selftest": [],
}


def _good_args(clip):
    region = {"x": 0, "y": 0, "width": 0.5, "height": 0.5}
    return {
        "probe_video": {"source": clip},
        "transcribe": {"source": clip, "model_size": "tiny"},
        "search_transcript": {"source": clip, "query": "hello"},
        "build_visual_index": {"source": clip},
        "search_visual": {"source": clip, "query": "red square"},
        "count_occurrences": {"source": clip, "description": "yellow circle", "end_sec": 6},
        "count_repetitions": {"source": clip, "start_sec": 0, "end_sec": 6},
        "contact_sheet": {"source": clip, "samples": 4},
        "review_video": {"source": clip, "samples": 4},
        "overview_chunks": {"source": clip, "chunk_sec": 3},
        "refine_temporal_beam": {"source": clip, "windows": [{"start_sec": 0, "end_sec": 3}]},
        "track_region_changes": {"source": clip, "start_sec": 0, "end_sec": 5, "sample_fps": 2, "region": region},
        "track_region_events": {"source": clip, "start_sec": 0, "end_sec": 5, "sample_fps": 2, "region": region},
        "inspect_moments": {"source": clip, "moments": [1.0]},
        "inspect_windows": {"source": clip, "windows": [{"start_sec": 0, "end_sec": 2}]},
        "inspect_window": {"source": clip, "start_sec": 0, "end_sec": 2},
        "frame_at": {"source": clip, "time_sec": 1.0},
        "detect_cuts": {"source": clip},
        "check_frame_cadence": {"source": clip},
        "audio_onsets": {"source": clip},
        "loudness_timeline": {"source": clip},
        "spectrogram_window": {"source": clip, "start_sec": 0, "end_sec": 2},
        "sync_check": {"source": clip},
        "label_sounds": {"source": clip, "start_sec": 0, "end_sec": 2, "labels": ["beep"]},
        "watch_for": {"source": clip, "conditions": ["Is a red square visible?"]},
        "motion_open": {"source": clip},
        "motion_sheet": {"source": clip},
        "motion_stats": {"source": clip, "box": [0, 0, 50, 50]},
        "motion_boil_map": {"source": clip},
        "motion_zoom": {"source": clip, "box": [0, 0, 50, 50]},
        "motion_profile": {"source": clip},
        "motion_compare": {"source": clip, "compare_to": clip, "box": [0, 0, 50, 50]},
        "motion_selftest": {},
    }


def _cells(clip, env):
    """(tool, case, arguments JSON text) for one environment."""
    good = _good_args(clip)
    cells = []
    tools = VISUAL_TOOLS if env == "visual" else tuple(REQUIRED)
    for tool in tools:
        cells.append((tool, "good", json.dumps(good[tool])))
        if tool == "count_occurrences":
            cells.append((tool, "good: text (OCR)", json.dumps({"source": clip, "text": "hello", "end_sec": 6})))
            cells.append((tool, "good: description needs CLIP",
                          json.dumps({"source": clip, "description": "a person in a red jacket", "end_sec": 6})))
        if tool == "transcribe":
            cells.append((tool, "good: turbo model", json.dumps({"source": clip, "model_size": "turbo"})))
        if tool == "motion_selftest":
            continue
        required = REQUIRED[tool]
        wrong = {key: [] for key in required if key != "source"}
        if "source" in required:
            wrong["source"] = 12345
        cells.append((tool, "malformed: missing keys", "{}"))
        cells.append((tool, "malformed: wrong types", json.dumps(wrong)))
        cells.append((tool, "malformed: not an object", "[]"))
    return cells


def _call(python, cwd, home, tool, arguments):
    """One tools/call through the installed package's CLI, from a folder that is not the repo."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "HF_HOME": str(home / "hf"), "LANG": "C.UTF-8"}
    try:
        proc = subprocess.run([str(python), "-m", "src.executor.mcp_server", "call", tool, arguments], cwd=str(cwd),
                              env=env, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return {"hang": True, "text": "", "stderr": ""}
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    text = "".join(item.get("text", "") for item in reply.get("content", []))
    return {"hang": False, "text": text, "stderr": proc.stderr, "isError": bool(reply.get("isError"))}


def _matrix_failures(python, env, clip, tmp_path):
    run_dir = tmp_path / f"run-{env}"
    run_dir.mkdir()
    cells = _cells(clip, env)

    def one(indexed):
        index, (tool, case, arguments) = indexed
        home = tmp_path / f"home-{env}-{index}"
        home.mkdir()
        return _call(python, run_dir, home, tool, arguments)

    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(one, enumerate(cells)))

    failures = []
    for (tool, case, _), got in zip(cells, replies):
        text = got["text"]
        problems = []
        if got["hang"]:
            problems.append("no reply within 240 s (loop or hang)")
        if "Traceback" in text or "Traceback" in got["stderr"]:
            problems.append("traceback")
        if not text:
            problems.append("no reply text")
        shown = text.replace(str(python), "<the running python>")  # the test's own venv path is not a user-visible name
        if got["isError"]:
            match = REPO_ONLY.search(SOURCE_CHECKOUT_NOTE.sub("", shown))
            if match:
                problems.append(f"names a repo-only file or path: {match.group(0)!r}")
            if ".venv" in shown:
                problems.append("names a .venv path")
            if re.search(r"\bbug\b", shown, re.IGNORECASE):
                problems.append("says this is a bug")
        if (tool, case) == OCR_TEXT_CELL and sys.platform != "darwin":
            if "runs only on macOS" not in text or "pip install" in text:
                problems.append("text search off macOS does not say it runs only on macOS, or names a pip command")
        elif env == "base" and (tool, case) in PIP_HINT_CELLS and PIP_HINT not in text:
            problems.append(f"setup reply lacks the pip command {PIP_HINT!r}")
        if env == "visual" and (tool, case) in WEIGHTS_CELLS and str(python) not in text:
            problems.append("weights command does not name the running Python")
        if tool in SWITCHED_OFF_TOOLS and "switched off" not in text:
            problems.append("switched-off tool did not refuse as switched off")
        if problems:
            failures.append(f"{env}: {tool} [{case}]: {'; '.join(problems)} :: {text[:400]}")
    return failures


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clip")
    path = folder / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True, timeout=120)
    return str(path)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(["uv", "build", "--wheel", "-o", str(out), str(ROOT)], check=True, capture_output=True, text=True, timeout=300)
    wheels = list(out.glob("agentic_video-*.whl"))
    assert len(wheels) == 1, f"uv build made {wheels}"
    return wheels[0]


def _venv(tmp_path_factory, name, spec):
    venv = tmp_path_factory.mktemp(name) / "venv"
    subprocess.run(["uv", "venv", "-q", "--python", sys.executable, str(venv)], check=True, timeout=120)
    python = venv / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), *spec], check=True, capture_output=True,
                   text=True, timeout=900)
    return python


@pytest.fixture(scope="module")
def base_python(tmp_path_factory, wheel):
    return _venv(tmp_path_factory, "base", [str(wheel)])


@needs_tools
def test_pip_install_without_extras_gives_pip_users_true_replies(base_python, clip, tmp_path):
    failures = _matrix_failures(base_python, "base", clip, tmp_path)
    assert not failures, "\n".join(failures)


@needs_tools
@pytest.mark.slow
def test_pip_install_with_visual_extra_names_the_running_python_for_weights(tmp_path_factory, wheel, clip, tmp_path):
    visual_python = _venv(tmp_path_factory, "visual", ["--torch-backend=cpu", f"{wheel}[visual]"])
    failures = _matrix_failures(visual_python, "visual", clip, tmp_path)
    assert not failures, "\n".join(failures)
