"""Embedded cover art should keep an audio file on the audio tool path."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def album_track(tmp_path):
    cover = tmp_path / "cover.jpg"
    Image.new("RGB", (64, 64), "red").save(cover)
    track = tmp_path / "track.mp3"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1",
         "-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "copy",
         "-disposition:v", "attached_pic", str(track)],
        capture_output=True, check=True, timeout=30,
    )
    return track


def call(tool, source, cache, **arguments):
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "src.executor.mcp_server", "call", tool,
         json.dumps({"source": str(source), **arguments})],
        cwd=repo, env={**os.environ, "AGENTIC_VIDEO_CACHE_DIR": str(cache)},
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_probe_album_track_is_audio_only(album_track, tmp_path, legacy_cache):
    cache = tmp_path / "cache"
    if legacy_cache:
        fingerprint = hashlib.sha256(album_track.read_bytes()).hexdigest()[:20]
        entry = cache / fingerprint
        entry.mkdir(parents=True)
        (entry / "metadata_display_size.json").write_text(json.dumps({
            "source_path": str(album_track), "duration_sec": 1.0, "width": 64, "height": 64,
            "fps": 90000.0, "video_codec": "mjpeg", "audio_codec": "mp3", "has_audio": True,
            "has_video": True, "size_bytes": album_track.stat().st_size, "format_name": "mp3",
            "fingerprint": fingerprint,
        }), encoding="utf-8")

    proc = call("probe_video", album_track, cache)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)["structuredContent"]
    assert data["has_video"] is False
    assert data["has_audio"] is True and data["audio_codec"] == "mp3"
    assert data["width"] == data["height"] == data["fps"] == 0
    assert data["video_codec"] is None
    assert data["duration_sec"] == pytest.approx(1, abs=0.1)


def test_album_track_audio_tools_work_and_visual_tools_explain_the_error(album_track, tmp_path):
    cache = tmp_path / "cache"
    audio = call("loudness_timeline", album_track, cache)
    assert audio.returncode == 0, audio.stdout + audio.stderr
    assert json.loads(audio.stdout)["structuredContent"]["windows"]

    visual = call("frame_at", album_track, cache, time_sec=0)
    assert visual.returncode == 1, visual.stdout + visual.stderr
    assert json.loads(visual.stdout)["isError"] is True
    assert "audio only" in visual.stdout
    assert "pass a video file as source" in visual.stdout


def mp4_atoms(data):
    while data:
        size = int.from_bytes(data[:4], "big")
        assert 8 <= size <= len(data)
        yield data[:size]
        data = data[size:]


def test_frame_at_reads_video_when_cover_stream_comes_first(tmp_path):
    cover = tmp_path / "cover.jpg"
    Image.new("RGB", (64, 64), "red").save(cover)
    movie = tmp_path / "movie.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=320x180:r=10:d=2",
         "-i", str(cover), "-map", "1:v", "-map", "0:v", "-c:v:0", "copy", "-disposition:v:0", "attached_pic",
         "-c:v:1", "libx264", str(movie)],
        capture_output=True, check=True, timeout=30,
    )
    # FFmpeg's MP4 muxer writes the cover in udta after the tracks. Keep the
    # atom sizes and offsets the same but put udta before the track entries.
    atoms = []
    for atom in mp4_atoms(movie.read_bytes()):
        if atom[4:8] == b"moov":
            children = sorted(mp4_atoms(atom[8:]), key=lambda item: {b"mvhd": 0, b"udta": 1}.get(item[4:8], 2))
            atom = atom[:8] + b"".join(children)
        atoms.append(atom)
    movie.write_bytes(b"".join(atoms))
    streams = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(movie)], text=True, timeout=30,
    )
    assert json.loads(streams)["streams"][0]["disposition"]["attached_pic"] == 1

    proc = call("frame_at", movie, tmp_path / "cache", time_sec=0.8)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)["structuredContent"]
    assert (data["source_width"], data["source_height"]) == (320, 180)
    assert data["frame_time_sec"] == pytest.approx(0.8)
    with Image.open(data["file_path"]) as frame:
        assert frame.size == (320, 180)
        assert frame.getpixel((160, 90))[2] > 200
