"""Argument shapes agents guess, through handle_call on a real lossless clip with a tone and a scene change."""
import json
import shutil
import subprocess

import jsonschema
import pytest

from src.executor.mcp_server import TOOLS_MANIFEST, handle_call

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

REGION_LIST = [0.1, 0.1, 0.5, 0.5]
REGION_OBJECT = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
MIXED_REGION = {"x": 0.1, "y": 0.1, "w": 0.5, "height": 0.5}
MIXED_REGION_OTHER = {"x": 0.1, "y": 0.1, "width": 0.5, "h": 0.5}


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

    # The published input schemas take every shortcut the server reads, and refuse the shapes it does not.
    published = {tool["name"]: tool["inputSchema"] for tool in TOOLS_MANIFEST}
    # Each region tool takes the list and both object spellings, and the mixed spellings (w with height, width with h).
    assert events(MIXED_REGION) == events(REGION_OBJECT)
    shortcuts = [
        ("inspect_moments", {"source": source, "moments": [1.0]}),
        ("inspect_moments", {"source": source, "moments": [{"time": 2.5}, {"t": 3}, {"time_sec": "00:02.500"}]}),
        ("track_region_events", {"source": source, "start_sec": 0, "end_sec": 6, "region": REGION_LIST}),
        ("track_region_events", {"source": source, "start_sec": 0, "end_sec": 6, "region": MIXED_REGION}),
        ("track_region_events", {"source": source, "start_sec": 0, "end_sec": 6, "region": MIXED_REGION_OTHER}),
        ("track_region_changes", {"source": source, "start_sec": 0, "end_sec": 6, "region": MIXED_REGION_OTHER}),
        ("count_repetitions", {"source": source, "region": REGION_LIST}),
        ("count_repetitions", {"source": source, "region": MIXED_REGION}),
        ("count_repetitions", {"source": source, "region": MIXED_REGION_OTHER}),
    ]
    for name, args in shortcuts:
        jsonschema.validate(args, published[name])
    for name, args in [
        ("inspect_moments", {"source": source, "moments": ["2.5"]}),
        ("inspect_moments", {"source": source, "moments": [{"time": 2, "t": 3}]}),
        ("track_region_events", {"source": source, "start_sec": 0, "end_sec": 6, "region": [0.1, 0.1, 0.5]}),
    ]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(args, published[name])

    with pytest.raises(ValueError) as refused:
        handle_call("search_transcript", {"source": source, "query": "hello", "mode": "keyword"})
    message = str(refused.value)
    assert "argument 'mode'" in message
    assert "hybrid" in message and "lexical" in message
    assert '"mode": "hybrid"' in message


def test_wrong_nested_fields_lengths_and_missing_arguments_name_the_fix_and_the_example_runs(tmp_path):
    source = _clip(tmp_path / "shapes.mkv")

    with pytest.raises(ValueError) as unknown_field:
        handle_call("inspect_moments", {"source": source, "moments": [{"time": 2, "timestamp": 4}]})
    message = str(unknown_field.value)
    assert "argument 'moments[0].timestamp' is not an accepted field" in message
    assert "timestamp_sec, label" in message
    assert 'Example: "moments": [{"timestamp_sec": 12.5}].' in message

    with pytest.raises(ValueError) as wrong_length:
        handle_call("audio_onsets", {"source": source, "band_hz": [300, 600, 900]})
    message = str(wrong_length.value)
    assert "argument 'band_hz' has 3 item(s)" in message
    assert "exactly 2 item(s)" in message

    with pytest.raises(ValueError) as missing_source:
        handle_call("inspect_moments", {"moments": [{"timestamp_sec": 2.5}]})
    message = str(missing_source.value)
    assert "argument 'source' is missing or empty" in message
    assert 'Example: "source": "/path/to/video.mp4".' in message

    with pytest.raises(ValueError) as bad_ranges:
        handle_call("loudness_timeline", {"source": source, "ignore_ranges": "bad"})
    example = str(bad_ranges.value).split('Example: "ignore_ranges": ')[1].rstrip(".")
    result = handle_call("loudness_timeline", {"source": source, "ignore_ranges": json.loads(example)})
    assert result["ok"] is True
