# Argument shapes

What it does: the server reads three wrong shapes whose meaning is certain, and rewrites each to the documented shape before the schema check:

- `inspect_moments` `moments`: a bare number (`5.0`) or an object with exactly one of `timestamp_sec`, `time_sec`, `time`, `t`.
- `region` of `track_region_changes`, `track_region_events` and `count_repetitions`: a list `[x, y, width, height]` (`count_repetitions` reads the same four numbers as `x, y, w, h`), in the same units as the object form (fractions of the frame).

Everything else stays an error. `search_transcript` `mode: "keyword"` is refused, not mapped to `lexical`. A moment with two time keys is refused, because the meaning is unclear.

Every shape or value error names the parameter, what it accepts, and one call fragment to copy, for example `Example: "moments": [{"timestamp_sec": 12.5}].` The example comes from the schema default, the first enum value, or a known example in `_CALL_EXAMPLES`.

Code: `src/executor/mcp_server.py`, `_accepted_shapes`, `_moment_shape`, `_argument_error` and `_example_value`, called from `_checked_arguments`.

How to check it works: `pytest tests/test_argument_shapes_e2e.py` (needs ffmpeg). It checks that number and `{"time": t}` moments give the same sample times as `{"timestamp_sec": t}`, that a region list gives the same events as the object form, and that `mode: "keyword"` names `mode`, both accepted values and an example.

Gotchas: the schema descriptions for these arguments carry one example each, and the zod copy in `src/adapters/mcp_definitions.ts` must match them (`tests/test_agent_first_tools.py` checks it).
