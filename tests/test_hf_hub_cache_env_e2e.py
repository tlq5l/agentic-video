"""Regression for gpt6 finding 1: hf_hub_cache() ignored HUGGINGFACE_HUB_CACHE. Drives search_visual through the MCP handle_call entry point."""
import importlib.machinery
import subprocess
import sys
import types

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


def test_huggingface_hub_cache_env_var_is_honored(tmp_path, monkeypatch):
    _stub_packages(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    hub = tmp_path / "custom-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    snap = hub / "models--openai--clip-vit-base-patch32" / "snapshots" / visual.MODEL_REVISION
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "preprocessor_config.json").write_text("{}")
    (snap / "pytorch_model.bin").write_bytes(b"planted, never loaded")
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "colors.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5", "-t", "2",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    try:
        reply = mcp_server.handle_call("search_visual", {"source": str(source), "query": "red"})
    except RuntimeError as exc:
        raise AssertionError(f"HUGGINGFACE_HUB_CACHE holds the weights, yet the setup error was raised: {exc}") from exc
    assert WEIGHTS_ERROR not in str(reply), reply
