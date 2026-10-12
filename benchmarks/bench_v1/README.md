# bench-v1

`bench-v1` is a blind, three-contestant comparison of video moment localization, long-video search, and visual/action counting. The split is 10 deterministic synthetic and 10 real questions per track (60 total), with one negative control in each split of each track. Questions and media live outside this repository; the truth key is stored in a separate sealed directory.

## Data locations

- Videos: `~/.local/share/agentic-video-bench/bench-v1/videos/`
- Questions without truth: `~/.local/share/agentic-video-bench/bench-v1/questions.jsonl`
- Results: `~/.local/share/agentic-video-bench/bench-v1/runs/<contestant>/`
- Truth key: `~/.local/share/agentic-video-sealed/bench-v1/answer_key.jsonl` (mode `0600`)

`fetch_real.py` reads upstream annotations into memory and writes truth only to the sealed key. It does not save annotation dumps. The Codex contestant workspace is a fresh temporary directory with `question.json` and a `video.mp4` symlink. Both Codex and Gemini adapter processes use `sandbox-exec` to deny reads and writes under the sealed directory and either key's parent directory, deny the question bank and prior runs, and deny sibling video files while permitting the selected video. Text-only Gemini repair cannot read any benchmark video. Codex also uses its read-only sandbox and tells the agent the key is off limits. The contestant never receives dataset metadata or answer-key fields.

## Build the data

Requirements: Python 3.10+, FFmpeg, macOS `say` for deterministic speech generation, Node.js 22+, yt-dlp with its JavaScript challenge solver, and PyArrow for the pinned annotation shard. Run these from the repository root. A separate tools venv keeps the download tools out of the main `.venv`:

```sh
V=.venv
B=~/.local/share/agentic-video-bench/bench-v1/.tools-venv
$V/bin/python -m venv "$B"
"$B/bin/python" -m pip install yt-dlp yt-dlp-ejs pyarrow
```

Then run:

```sh
$V/bin/python benchmarks/bench_v1/generate_synthetic.py
$B/bin/python benchmarks/bench_v1/fetch_real.py
```

The synthetic generator uses seed `8448` and writes exact labels directly into the sealed key. Each moment and count track uses three generated videos, with object appearances, text changes, hard cuts, motion-blurred crossings, and look-alike distractors. The three synthetic search videos add visual and text events, spoken phrases, ascending and descending audio cues, scene cuts, and negative controls. Before writing, the generator rejects duplicate question text and checks that each target's declared occurrence count matches its answer format. The real fetcher pins QVHighlights validation moments, Countix validation repetitions, and Extreme-When long-video search annotations. Search uses three distinct long sources of at least 30 minutes, including one at least two hours long, plus a separate negative-control video. It refuses to write the real question/key rows until all selected downloads finish and the long-source durations are verified.

## Real data sources and rights

Each real question's answer-key `source` object records its annotation ID, source URL, clip URL, and license note. Source video rights remain distinct from annotation rights.

| Track | Dataset | License record |
|---|---|---|
| Moment | [QVHighlights validation annotations](https://github.com/jayleicn/moment_detr/tree/main/data) | Annotation files are CC BY-NC-SA 4.0. The dataset does not assert a license for each original YouTube video. |
| Count | [Countix validation annotations](https://s3.amazonaws.com/kinetics/700_2020/annotations/countix.tar.gz), listed by the [Kinetics dataset project](https://github.com/cvdfoundation/kinetics-dataset) | The Countix archive README does not state a separate annotation license. Original videos keep their uploader-specific rights. |
| Long search | [Extreme-When benchmark](https://huggingface.co/datasets/min1321/extreme-when-bench), selected from the LVBench source corpus | The benchmark annotations are CC BY 4.0. [LVBench](https://github.com/zai-org/LVBench#license) restricts use to academic research and prohibits commercial use; it does not own source-video copyright. Source-video terms remain applicable and the local videos are excluded from git. |

The original URLs are recorded for each clip in the sealed key. Check the applicable video-owner and source-corpus terms before sharing or publishing any run artifacts.

## Contestants

- `ours-codex`: `codex exec --json`, default `gpt-6-luna` / medium effort, with the tools from `src/executor/mcp_server.py` exposed via a stdio MCP server. The tools run from this repository with the Python that runs the benchmark; override them with `AGENTIC_VIDEO_MCP_CWD` and `AGENTIC_VIDEO_PYTHON` if needed.
- `gemini-agentic`: Gemini 3.8 Flash through the Interactions API with video `processing: "agentic"`.
- `gemini-plain`: same model and API without that field, which uses the API's default static video handling.

Both Gemini modes upload the video with the Files API, wait for it to become active, call the Interactions API, then delete the uploaded file. The API key is read at run time from the `GEMINI_API_KEY` environment variable, which only the Gemini adapter process receives; it is never written to the result file and is scrubbed from adapter errors. Gemini JSON is validated once; if needed, one text-only repair call receives the initial raw answer and no deleted video reference. Both initial and repaired raw text are retained in the result. Numeric strings and JSON code fences are normalized for validation; answer content is not changed. API, timeout, missing-answer, and invalid-JSON failures are saved as errors and scored wrong. Codex JSON failures get one fresh, format-only Codex retry. Results are written one file per question and existing IDs are skipped on resume.

The Codex invocation retains `--ignore-user-config` and supplies an inline MCP configuration with `mcp_servers.agentic_video.default_tools_approval_mode = "approve"`, so video tool calls work with `approval_policy = "never"`. The stdio server receives `HF_HOME` from `AGENTIC_VIDEO_HF_HOME`, or defaults to `<tool-python venv>/huggingface`, plus `HF_HUB_OFFLINE=1` for local-only CLIP weights. Each run records successful and failed video-tool calls; Gemini agentic success requires a matching `processing_result` step. An otherwise valid run with no successful video-tool calls, or an explicit statement that the video could not be inspected, is classified as a harness failure. Existing errors such as timeouts keep their original failure type. Every failure, including a harness failure, counts as wrong in the headline accuracy metrics; harness failures are also counted separately. The first Codex run calibrates the fixed token overhead using a trivial no-video question. The baseline is saved once under `runs/codex-overhead.json`; each run reports raw total tokens and adjusted total tokens after subtracting that baseline for each `codex exec` invocation.

The Codex MCP allow-list includes `build_visual_index`, `search_visual`, `count_repetitions`, and `count_occurrences` alongside `probe_video`, `detect_cuts`, `inspect_window`, `contact_sheet`, `search_transcript`, `transcribe`, `audio_onsets`, `loudness_timeline`, `spectrogram_window`, and `sync_check`. No shell or general-purpose file-inspection tools are requested in the prompt.

## Transcript ingest, visual indexing, and verification

Transcript creation and visual indexing run before an `ours-codex` question loop, outside `sandbox-exec`. For each distinct selected video, the runner calls the configured MCP checkout's `transcribe_video(real_path)` with its default model, then calls `src.retrieval.visual.build_index(engine)` synchronously. This populates the same caches used by `search_transcript` and `search_visual` through the contestant's `video.mp4` symlink. Visual build errors and their durations are recorded but do not fail transcript ingest; the contestant can still call `build_visual_index`. The Codex MCP tool timeout is set to 600 seconds in both contestant and overhead configurations.

You can ingest a selected set ahead of time and inspect cache-hit search from the contestant sandbox:

```sh
V=.venv
$V/bin/python benchmarks/bench_v1/ingest.py --questions real_search_01
$V/bin/python benchmarks/bench_v1/ingest.py --questions real_search_01 --verify
```

Selectors accept question IDs, comma-separated IDs, and tracks such as `track:search`. Omit `--questions` to ingest or verify every distinct video in `questions.jsonl`. Override the MCP checkout and Python with `AGENTIC_VIDEO_MCP_CWD` and `AGENTIC_VIDEO_PYTHON`, or pass `--tool-cwd` and `--tool-python` directly. The manifest defaults to `runs/ingest-ours-codex.json` under the benchmark data root; `--manifest` can select another location.

The manifest records duration when returned by ASR, segment count, transcript status/error, cache state before ingest, transcript and visual-index durations/statuses, MCP path, Python, model, and totals. Later runs preserve the first `ingest_time_s` and `visual_index_time_s` while updating latest-attempt fields. Verification first checks the transcript entry inside the sandbox, then makes a time-bounded search call on the symlink; it records elapsed seconds in the manifest and fails if the cache is unavailable or the call times out. The runner automatically prepares every distinct video in its selected questions before the contestant loop. If transcript ingest fails, it writes a `failure_type: "ingest"` result for every selected question on that video and continues with questions whose videos ingested successfully.

Each result records `answer_time_s`, `preparation_time_s`, and `end_to_end_time_s`. Gemini answer time comes from the adapter's timer around the Interactions API request and response; it excludes video upload, file processing, local parsing, and temporary-file deletion. `cleanup_time_s` reports adapter-measured deletion time separately. `wall_time_s` is a compatibility alias for answer time and is null when the answer/preparation split is unknown. Codex preparation is the first transcript-ingest and visual-index durations, charged once per distinct video even if a later run hits the cache. Gemini upload plus processing is counted once per answer run. The score report shows mean answer time with its timed-answer count, total preparation time, cleanup total, and end-to-end total. It also reports mean preparation per distinct video for Codex and per run for Gemini. For legacy Gemini results without `answer_time_s`, the scorer subtracts upload and processing plus measured cleanup from `wall_time_s`. When cleanup was not measured, it leaves the single Files API delete in the derived answer duration and includes that run in the mean. The table prefixes that mean with `≤` to mark an upper bound and shows the number of affected runs; it does not estimate cleanup. If a Gemini adapter failure does not return both upload/processing and answer timing, preparation is unknown and the result is excluded from the timed-answer mean; the outer runner duration remains available as end-to-end time.

Each new Codex question result records `tool_timeout_sec`, `ingest_time_s`, `visual_index_time_s`, `visual_index_status`, and `ingest_cache_hit_before`. The scorer counts transcript ingest plus visual indexing once per distinct video in Codex preparation time. Failed tool-call event text is retained in the call's `outcome_detail` when Codex supplies it.

## Run

Smoke one synthetic and one real question per track (six runs per contestant, 18 total):

```sh
V=.venv
$V/bin/python benchmarks/bench_v1/run_contestant.py --contestant ours-codex --questions synth_moment_01 real_moment_01 synth_search_01 real_search_01 synth_count_01 real_count_01
$V/bin/python benchmarks/bench_v1/run_contestant.py --contestant gemini-agentic --questions synth_moment_01 real_moment_01 synth_search_01 real_search_01 synth_count_01 real_count_01
$V/bin/python benchmarks/bench_v1/run_contestant.py --contestant gemini-plain --questions synth_moment_01 real_moment_01 synth_search_01 real_search_01 synth_count_01 real_count_01
$V/bin/python benchmarks/bench_v1/score.py
```

The run command accepts one or more question IDs, comma-separated ID lists, or track selectors such as `--questions track:search`. Omit `--questions` to run the full 60-question set. Use `--rerun` to replace prior results. The scorer reads only result JSON and the sealed key; it prints aggregate metrics and writes no predictions or truths.

## Scoring

- Synthetic moment: numeric answers are measured against the truth point; interval answers use sub-second boundary tolerances. Real QVHighlights moment: all annotated relevant windows are accepted, scored by best interval IoU (primary IoU ≥ 0.5); also reports IoU ≥ 0.7 and predicted-midpoint distance ≤ 2 seconds.
- Search: a point answer is measured by distance to the truth interval; an interval answer is measured by its midpoint's distance to the truth interval. Reports hit rate within 2 and 5 seconds, plus interval IoU.
- Count: exact match and within one.
- `not_found` is correct only for a negative-control truth. Every failed run counts as wrong, including harness and ingest failures; the table reports harness failures separately.
- Synthetic moment tolerance rates include synthetic negative controls because `not_found` receives a defined hit value. Search hit rates and count exact/within-one rates also include their negative controls for the same reason. Real-moment IoU ≥ 0.5, IoU ≥ 0.7, and midpoint ≤ 2 seconds are positional rates over real questions with interval truths; negative controls and `not_found` truths have no positional value and are reported in the separate negative-control rate. Mean interval IoU likewise uses interval truths only. Every failed or missing result in an eligible rate denominator counts as a miss; failed or missing interval answers and point answers contribute zero mean IoU. Cue-type spelling with spaces, hyphens, or underscores is treated the same for negative controls.
- By default the scorer includes all three bench contestants, plus any other result directories; `--contestant` selects a specific set. For each included contestant, it uses every question in the supplied question set as the denominator. Missing result files are counted as failures and wrong answers, and the table reports their count as `Missing`. The default question set is the full benchmark; pass `--questions <selected-questions.jsonl>` to score a partial run against its selected question set. An explicit `--contestant` is included even when it has no result files.
- Per track and contestant the table includes sample count, failures, missing results, harness failures, accuracy, secondary metric, mean total tokens, mean tokens after fixed Codex overhead, mean answer time, preparation totals, cleanup total, end-to-end total, and Gemini estimated cost. Token means and Gemini cost show measured runs over total runs. When any run lacks the relevant measurement, the cell says `incomplete`; the displayed token mean covers measured runs only, and Gemini cost is the recorded subtotal. For Gemini, adjusted tokens equal total tokens.
- Timing boundaries are consistent: answer time excludes preparation for all contestants. Codex preparation is transcript ingest plus visual indexing once per distinct video; Gemini preparation is upload plus processing once per run. Gemini cleanup is reported separately and remains in end-to-end duration. Codex end-to-end total adds once-only transcript ingest and visual indexing to measured answer times; Gemini end-to-end total uses the outer run duration.

Gemini usage and cost include agentic navigation thinking and on-demand video/audio/transcript tokens when returned by the API. Gemini 3.8 Flash's standard paid tier is currently $0.75 per million input tokens and $3.75 per million output tokens through 2026-12-31; thinking tokens use the output rate. The scorer uses the usage fields from each actual result rather than estimating per-video cost. Current pricing: [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). Static video defaults to approximately 100 input tokens/second at low resolution or 300/second at high resolution; agentic usage is content and query dependent, with Google describing up to 88% fewer input tokens on long videos. See [video token accounting](https://ai.google.dev/gemini-api/docs/tokens) and [agentic video mode](https://ai.google.dev/gemini-api/docs/video-understanding).

### Smoke-based full-run estimate (2026-09-27)

The six-question smoke samples one synthetic and one real item per track. These estimates scale each track's two measured runs by 10 to approximate all 20 questions in that track; they are planning estimates, not a full benchmark run. Search timing and Gemini cost vary most because real search sources are long videos.

| Contestant | Estimated time for 60 | Estimated cost or usage |
|---|---:|---:|
| `ours-codex` | 73 min | 26.0M total tokens; 24.8M after fixed-overhead adjustment |
| `gemini-agentic` | 109 min | $26.41 |
| `gemini-plain` | 62 min | $5.83 |

Track estimates (20 questions each): Codex 13/16/44 min for count/moment/search; Gemini agentic 26/13/70 min and $5.20/$1.01/$20.20; Gemini plain 9/9/44 min and $0.18/$0.23/$5.41. Estimates use two smoke observations per track and should be treated as rough, especially for long-video search.

## Tests

The runner's parsing, MCP-call accounting, Codex tool allow-list and environment config, harness and ingest failure handling, visual-index timing, Gemini timing split, text-only repair dispatch/content preservation, duplicate-question guard, and the scorer's once-per-video preparation, missing-result denominator, tolerance, QVHighlights interval-IoU, negative-control, and count rules have unit coverage in `tests/test_bench_v1_ingest.py`, `tests/test_bench_v1_runner.py`, and `tests/test_bench_v1_score.py`. Run them with `python -m pytest -q tests/test_bench_v1_ingest.py tests/test_bench_v1_runner.py tests/test_bench_v1_score.py`.
