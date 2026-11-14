"""End-to-end: CLIP weights that sit in the standard Hugging Face cache are found, whatever cache location the
environment names (XDG_CACHE_HOME, or a literal HF_HOME=~/hf). Drives search_visual through the MCP entry point.

Torch and transformers are stubbed as importable, so the weights check is the only thing that can raise the setup error."""

import importlib.machinery
import subprocess
import sys
import types
from pathlib import Path

from src.core.config import Config
from src.executor import mcp_server
from src.retrieval import visual

WEIGHTS_ERROR = "needs the CLIP model weights"


def _stub_packages(monkeypatch):
    def stub(name):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return module

    class NoCachedWeights:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise OSError(f"{model_id} is not a cached local model")

    torch = stub("torch")
    torch.set_num_threads = lambda threads: None
    transformers = stub("transformers")
    transformers.CLIPModel = NoCachedWeights
    transformers.CLIPProcessor = NoCachedWeights
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _plant_weights(hub: Path):
    snap = hub / "models--openai--clip-vit-base-patch32" / "snapshots" / visual.MODEL_REVISION
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "preprocessor_config.json").write_text("{}")
    (snap / "pytorch_model.bin").write_bytes(b"planted, never loaded")


def _assert_search_not_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "colors.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5", "-t", "2",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    try:
        reply = mcp_server.handle_call("search_visual", {"source": str(source), "query": "red"})
    except RuntimeError as exc:
        raise AssertionError(f"weights are cached, yet the setup error was raised: {exc}") from exc
    assert WEIGHTS_ERROR not in str(reply), reply


def _clear_hf_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)


def test_cached_weights_under_xdg_cache_home_are_found(tmp_path, monkeypatch):
    _stub_packages(monkeypatch)
    _clear_hf_env(monkeypatch, tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    _plant_weights(tmp_path / "xdg" / "huggingface" / "hub")
    _assert_search_not_blocked(tmp_path, monkeypatch)


def test_cached_weights_under_literal_tilde_hf_home_are_found(tmp_path, monkeypatch):
    _stub_packages(monkeypatch)
    _clear_hf_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_HOME", "~/hf")
    _plant_weights(tmp_path / "home" / "hf" / "hub")
    _assert_search_not_blocked(tmp_path, monkeypatch)
