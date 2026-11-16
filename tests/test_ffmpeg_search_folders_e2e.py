"""The ffmpeg search folders in src/core/config.py match the README and the function docstring, read from the real files."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "core" / "config.py"
README = ROOT / "README.md"


def _function():
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "ensure_ffmpeg_on_path"]
    assert found, "src/core/config.py has no ensure_ffmpeg_on_path"
    return found[0]


def _body(fn):
    return fn.body[1:] if ast.get_docstring(fn) else fn.body


def _key(text):
    text = text.lower().replace("\\", "/").rstrip(".,;")
    if "gyan.ffmpeg" in text:
        return "gyan.ffmpeg"
    text = text.replace("%userprofile%/", "~/")
    if not text.startswith(("/", "~/", "c:/")):
        text = "~/" + text
    return text


def code_folders():
    strings = [n.value for stmt in _body(_function()) for n in ast.walk(stmt)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    folders = {_key(s) for s in strings if "/" in s}
    assert folders, "no search folders found in ensure_ffmpeg_on_path"
    return folders


def readme_paragraph():
    paragraphs = [p for p in README.read_text(encoding="utf-8").split("\n\n") if "AGENTIC_VIDEO_FFMPEG_SEARCH" in p]
    assert paragraphs, "README.md has no paragraph naming AGENTIC_VIDEO_FFMPEG_SEARCH and the ffmpeg search folders"
    return paragraphs[0]


def _diff(named, code):
    return {"only in the docs": sorted(named - code), "only in config.py": sorted(code - named)}


def test_readme_names_exactly_the_code_folders():
    tokens = re.findall(r"`([^`]+)`", readme_paragraph())
    named = {_key(t) for t in tokens if "/" in t.replace("\\", "/") or "gyan.ffmpeg" in t.lower()}
    code = code_folders()
    assert named == code, _diff(named, code)


def test_docstring_names_exactly_the_code_folders():
    doc = (ast.get_docstring(_function()) or "").lower().replace("\\", "/")
    named = {_key(t) for t in re.findall(r"gyan\.ffmpeg|(?:~|c:)?/[\w./*-]+", doc)}
    code = code_folders()
    assert named == code, _diff(named, code)


def test_search_can_be_turned_off():
    code = "\n".join(ast.unparse(stmt) for stmt in _body(_function()))
    assert "AGENTIC_VIDEO_FFMPEG_SEARCH" in code and "AGENTIC_VIDEO_FFMPEG_SEARCH=0" in readme_paragraph()


def test_readme_says_windows_checks_ffmpeg_exe_only():
    code = "\n".join(ast.unparse(stmt) for stmt in _body(_function()))
    paragraph = readme_paragraph()
    if "ffmpeg.exe" in code and "ffprobe.exe" not in code:
        assert "ffmpeg.exe" in paragraph and "looks for both" not in paragraph, paragraph
