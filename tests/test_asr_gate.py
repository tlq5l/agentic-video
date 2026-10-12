"""Narrow Gate B regression checks for subtitle ASR and no-audio handling."""

from pathlib import Path
import shutil

import pytest

from src.core.config import Config
from src.transcript.asr import parse_srt_or_vtt, transcribe_video
from src.transcript.search import search_transcript


FIXTURE_VIDEO = Path(__file__).parents[1] / "fixtures" / "reference_videos" / "sample_10s.mp4"

SRT_TEXT = """1
00:00:02,135 --> 00:00:03,420
Alpha bravo rendezvous.

2
00:00:04,005 --> 00:00:05,250
Unicode café 東京 spaces.

3
00:00:06,750 --> 00:00:07,100
Late speech marker.

4
00:00:08,500 --> 00:00:09,900
Final long late phrase.
"""

VTT_TEXT = """WEBVTT

00:00:02.135 --> 00:00:03.420
Alpha bravo rendezvous.

00:00:04.005 --> 00:00:05.250
Unicode café 東京 spaces.

00:00:06.750 --> 00:00:07.100
Late speech marker.

00:00:08.500 --> 00:00:09.900
Final long late phrase.
"""


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "BASE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TEMP_DIR", tmp_path / "tmp")


def _segment_tuples(result):
    return [(s.id, s.start_sec, s.end_sec, s.text) for s in result.segments]


def test_gate_b_subtitle_alignment_unicode_spaces_and_srt_vtt(isolated_cache, tmp_path):
    if not FIXTURE_VIDEO.exists():
        pytest.skip("local sample fixture is unavailable")

    vtt_video = tmp_path / "gate_vtt.mp4"
    srt_video = tmp_path / "gate_srt.mp4"
    shutil.copy2(FIXTURE_VIDEO, vtt_video)
    shutil.copy2(FIXTURE_VIDEO, srt_video)
    vtt_path = vtt_video.with_suffix(".vtt")
    srt_path = srt_video.with_suffix(".srt")
    vtt_path.write_text(VTT_TEXT, encoding="utf-8")
    srt_path.write_text(SRT_TEXT, encoding="utf-8")

    expected = [
        (1, 2.135, 3.420, "Alpha bravo rendezvous."),
        (2, 4.005, 5.250, "Unicode café 東京 spaces."),
        (3, 6.750, 7.100, "Late speech marker."),
        (4, 8.500, 9.900, "Final long late phrase."),
    ]

    vtt_result = transcribe_video(str(vtt_video))
    srt_result = transcribe_video(str(srt_video), subtitle_override=str(srt_path))

    assert _segment_tuples(vtt_result) == expected
    assert _segment_tuples(srt_result) == expected
    assert vtt_result.has_speech is True
    assert srt_result.has_speech is True

    assert [hit.segment_id for hit in search_transcript(str(vtt_video), "  Unicode   café\t東京  ")] == [2]
    assert [hit.segment_id for hit in search_transcript(str(vtt_video), "CAFÉ")] == [2]
    assert search_transcript(str(vtt_video), "emerald elephant") == []
    assert search_transcript(str(vtt_video), "final", time_range=(0.0, 8.0)) == []
    assert [hit.segment_id for hit in search_transcript(str(vtt_video), "final", time_range=(8.0, 10.0))] == [4]


def test_gate_b_no_audio_is_explicitly_empty(isolated_cache, tmp_path):
    if not FIXTURE_VIDEO.exists():
        pytest.skip("local sample fixture is unavailable")

    no_audio_video = tmp_path / "no_audio_gate.mp4"
    shutil.copy2(FIXTURE_VIDEO, no_audio_video)
    result = transcribe_video(str(no_audio_video))

    assert result.language == "none"
    assert result.segment_count == 0
    assert result.segments == []
    assert result.has_speech is False


def test_gate_b_parser_accepts_unicode_srt_directly(tmp_path):
    path = tmp_path / "unicode.srt"
    path.write_text(SRT_TEXT, encoding="utf-8")
    segments = parse_srt_or_vtt(path)

    assert len(segments) == 4
    assert segments[1].text == "Unicode café 東京 spaces."
    assert segments[1].start_sec == pytest.approx(4.005)
    assert segments[3].end_sec == pytest.approx(9.900)


def test_gate_b_parser_accepts_short_webvtt_and_skips_invalid_cues(tmp_path):
    path = tmp_path / "short.vtt"
    path.write_text(
        "WEBVTT\n\n00:02.5 --> 00:03.25\nShort form cue\n\n"
        "00:04.000 --> 00:03.000\nReversed cue\n\n"
        "00:05.000 --> 00:06.000\nValid after invalid\n",
        encoding="utf-8",
    )
    segments = parse_srt_or_vtt(path)
    assert [(s.start_sec, s.end_sec, s.text) for s in segments] == [
        (2.5, 3.25, "Short form cue"),
        (5.0, 6.0, "Valid after invalid"),
    ]
