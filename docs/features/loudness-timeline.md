# Loudness timeline

How a user reaches it: the `loudness_timeline` tool, and `python -m src.loop check`.

Code: `src/media/audio.py` (`loudness_timeline`, `_measure_window`, `_channel_peaks`).

How to check it works: `pytest tests/test_audio_primitives.py -k peak` compares `peak_dbfs` with ffmpeg's astats peak.

Gotchas:
- `peak_dbfs` (top level with `peak_sec`, and per window) is the loudest sample on any channel, read from a separate streamed decode of the range.
- `rms_dbfs` and `band_db` still use the mono mix, which ffmpeg makes as (L+R)/1.414. On centred stereo they read up to 3 dB above a per-channel meter. Thresholds in the check loop are tuned to that.
- The result cache key is versioned (`loudness_timeline_v5`); bump it when the output changes.
