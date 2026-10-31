"""End-to-end check through handle_call on a lossless 29.97 fps clip with clicks: sync_check's
per-cue candidate list is opt-in."""

import subprocess
import wave

import numpy as np

import src.executor.mcp_server as mcp_server
from src.core.config import Config


def _write_counter_clip(path, seconds=3.0):
    """FFV1 (lossless) 29.97 fps picture with a 1 kHz click at 1 s and 2 s."""
    rate = 16000
    samples = np.zeros(int(rate * seconds), dtype=np.float32)
    for second in (1, 2):
        start = second * rate
        samples[start:start + 160] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(160) / rate)
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=29.97:d={seconds}",
         "-i", str(wav),
         "-vf", "format=yuv444p,geq=lum='16+N':cb=128:cr=128",
         "-c:v", "ffv1", "-pix_fmt", "yuv444p", "-c:a", "pcm_s16le", "-shortest", str(path)],
        check=True,
    )
    return str(path)


def _without_candidate_lists(cues):
    """The cue reports minus the per-cue onset fields, which differ between the two modes."""
    return [{key: value for key, value in cue.items() if not key.startswith("candidate_onsets")} for cue in cues]


def test_sync_check_candidate_onsets_are_opt_in_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    clip = _write_counter_clip(tmp_path / "clicks.mkv")
    cues = [{"time_sec": 1.0, "label": "click"}, {"time_sec": 2.0, "label": "click"}]

    default = mcp_server.handle_call("sync_check", {"source": clip, "cues": cues})
    assert default["ok"] is True
    assert len(default["data"]["cues"]) == 2
    for cue in default["data"]["cues"]:
        assert "candidate_onsets" not in cue
        assert isinstance(cue["candidate_onsets_count"], int)

    verbose = mcp_server.handle_call("sync_check", {"source": clip, "cues": cues, "verbose": True})
    assert verbose["ok"] is True
    for cue in verbose["data"]["cues"]:
        assert isinstance(cue["candidate_onsets"], list)
        assert "candidate_onsets_count" not in cue
    for default_cue, verbose_cue in zip(default["data"]["cues"], verbose["data"]["cues"]):
        assert default_cue["candidate_onsets_count"] == len(verbose_cue["candidate_onsets"])
    assert _without_candidate_lists(default["data"]["cues"]) == _without_candidate_lists(verbose["data"]["cues"])
