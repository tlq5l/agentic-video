# Frame cadence

How a user reaches it: the `check_frame_cadence` tool, and `python -m src.loop cadence <video>` (prints the same JSON).

Code: `src/media/visual_timing.py` (`frame_cadence`, `_near_identical`, `_rolling_sums`); the tool branch is in `src/executor/mcp_server.py` (`_dispatch`), the command in `src/loop.py` (`_cmd_cadence`).

How to check it works: `python -m src.loop cadence <video>` on a clip with a 30 fps stretch inside a 60 fps file should print one range there with `effective_fps` near 30, and `"ranges": []` on a clean clip; `pytest tests/test_schema_contract.py`.

Gotchas:
- A frame is a repeat when it is near-identical to the one before (the same test `frozen_pictures` uses, on a 160x90 grey picture). A range needs repeats in at least a quarter of the frames of a half-second window and lasts at least 0.5 s.
- A run of repeats longer than 0.5 s is a hold (still shot, title card, stall) and is not a range; `held_frames` counts them. `frozen_pictures` reports holds inside moving shots.
- It measures, it does not judge: a 24 fps film inside a 60 fps file is one range over the whole file, and a very slow, smooth move can read as repeats.
- Whole file only, no range arguments. Measured 2026-10-22: 0.8 s for the 12.6 s 60 fps earth-brawler ad (757 frames).
- Found first on earth-brawler `ad-tall.mp4`: 4.63-7.90 s and 8.52-9.50 s at about 29 fps, already present in its screen capture.
