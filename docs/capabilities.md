# Where each tool runs

One row per tool (25 in `src/adapters/mcp_schemas.json`). The last three columns say what a tool needs. "yes" means the code needs it; "no" means it does not.

Cells in the two run columns come from one smoke run per machine on 2026-10-31. The script called every tool once through `handle_call` on a 20-second clip with a tone and three scene changes. The run had no Apple silicon Mac.

- **ok**: the call returned a result.
- **started**: the call returned `in_progress` or `queued`. The tool runs a background job, and the smoke run did not wait for it.
- **not run**: nothing was run for this cell in this check.

| Tool | Apple silicon macOS | Intel macOS (iMac Pro) | Linux (Cloudflare box) | Cloudflare Workers AI credentials | Apple Vision (pyobjc) | torch>=2.6 and CLIP model |
|---|---|---|---|---|---|---|
| `probe_video` | not run | ok | ok | no | no | no |
| `transcribe` | not run | started | started | no | no | no |
| `search_transcript` | not run | started | started | no | no | no |
| `build_visual_index` | not run | started | started | no | no | yes |
| `search_visual` | not run | started | started | no | no | yes |
| `count_repetitions` | not run | started, ok on a repeat call | started, ok on a repeat call | no | no | required for `method=embedding`; optional for `auto`, which uses it only on windows up to 30 s where motion and periodicity disagree, and runs without it otherwise |
| `count_occurrences` | not run | started | started | no | only for `text=` | only for `description=` that names no shape (`yellow circle` needs none; `red mug` does) |
| `contact_sheet` | not run | ok | ok | no | no | no |
| `review_video` | not run | ok | ok | no | no | no |
| `overview_chunks` | not run | ok | ok | no | no | no |
| `refine_temporal_beam` | not run | ok | ok | no | no | no |
| `track_region_changes` | not run | ok | ok | no | no | no |
| `track_region_events` | not run | ok | ok | no | no | no |
| `inspect_moments` | not run | ok | ok | no | no | no |
| `inspect_windows` | not run | ok | ok | no | no | no |
| `inspect_window` | not run | ok | ok | no | only for `read_text: true` | no |
| `frame_at` | not run | ok | ok | no | only for `read_text: true` | no |
| `detect_cuts` | not run | ok | ok | no | no | no |
| `check_frame_cadence` | not run | ok | ok | no | no | no |
| `audio_onsets` | not run | ok | ok | no | no | no |
| `loudness_timeline` | not run | ok | ok | no | no | no |
| `spectrogram_window` | not run | ok | ok | no | no | no |
| `sync_check` | not run | ok | ok | no | no | no |
| `watch_for` | not run | not run | not run | yes | no | no |
| `label_sounds` | not run | not run | not run | yes | no | no |

Notes:

- Apple silicon was not run for any tool. The MacBook runs no heavy work, so this column is empty until someone runs the smoke script on an Apple silicon Mac.
- `transcribe` and `search_transcript` use local speech recognition. `turbo` needs whisper.cpp (`whisper-cli` and its model). `search_transcript` hybrid matches by meaning need EmbeddingGemma 2 on Apple CoreML; that was not run here, and without it the search stays keyword-only.
- The "started" cells are not failures. Their results are not checked here: each one still needs a repeat call or a poll of its job.
- `watch_for` and `label_sounds` send frames or audio to Cloudflare Workers AI. Without `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` they stop before sending anything.
- The smoke run also covered argument shapes agents guess (see `docs/features/contact-sheets.md` and the README). Those calls are not in this table.
