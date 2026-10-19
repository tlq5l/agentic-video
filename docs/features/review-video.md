# Review video

How a user reaches it: the `review_video` tool.

Code: `src/executor/mcp_server.py` (`_review_video`, `_loudness_summary`). It calls `AgenticVideoEngine.probe`, the `contact_sheet` and `detect_cuts` dispatch branches, and `AgenticVideoEngine.loudness_timeline`; it has no analysis code of its own.

How to check it works: `pytest tests/test_review_video.py tests/test_schema_contract.py tests/test_agent_first_tools.py`.

Gotchas:
- The whole call must finish in under 45 s (Executor cuts calls at 60 s). Cold timings measured on 2026-10-20: 0.8 s for the 60 s fixture, 11 s for a 166 s clip, 4.5 s for a 10-minute clip.
- No audio track: `loudness` is null and `note` says so. Any step failing fails the whole call with that step's error; nothing is returned half-filled.
- `loudness` is a summary (10 strongest outliers, 20 silence gaps). `average_rms_dbfs` is computed from all windows, so it does not depend on those limits.
