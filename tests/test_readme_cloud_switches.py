"""End to end: the README sentence that says what leaves the machine must name the cloud transcription switch."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
SWITCH = "AGENTIC_VIDEO_CLOUD_ASR"


def test_readme_leaves_your_machine_sentence_names_cloud_asr_switch():
    text = README.read_text(encoding="utf-8")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    leaving = [s for s in sentences if "leaves your machine" in s]
    assert len(leaving) == 1, leaving
    assert SWITCH in leaving[0], leaving[0]
