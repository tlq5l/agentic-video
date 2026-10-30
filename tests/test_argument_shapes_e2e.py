"""Argument shapes agents guess, through handle_call on a real lossless clip with a tone and a scene change."""
import shutil
import subprocess

import pytest

from src.executor.mcp_server import handle_call

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

REGION_LIST = [0.1, 0.1, 0.5, 0.5]
REGION_OBJECT = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}


def _clip(path):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=red:s=160x120:r=10:d=3",
         "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(path)],
        check=True)
    return str(path)


def test_guessed_argument_shapes_match_the_documented_ones_and_wrong_modes_name_the_fix(tmp_path):
    source = _clip(tmp_path / "shapes.mkv")

    def sample_times(moments):
        result = handle_call("inspect_moments", {"source": source, "moments": moments, "samples_per_moment": 3})
        return [item["sample_timestamps"] for item in result["data"]["moments"]]

    documented = sample_times([{"timestamp_sec": 2.5}])
    assert documented[0]
    assert sample_times([2.5]) == documented
    assert sample_times([{"time": 2.5}]) == documented

    def events(region):
        data = handle_call("track_region_events", {"source": source, "start_sec": 0, "end_sec": 6, "sample_fps": 2,
                                                   "region": region})["data"]
        return {key: data.get(key) for key in ("region", "event_count", "transitions", "sample_count")}

    assert events(REGION_LIST) == events(REGION_OBJECT)

    with pytest.raises(ValueError) as refused:
        handle_call("search_transcript", {"source": source, "query": "hello", "mode": "keyword"})
    message = str(refused.value)
    assert "argument 'mode'" in message
    assert "hybrid" in message and "lexical" in message
    assert '"mode": "hybrid"' in message
