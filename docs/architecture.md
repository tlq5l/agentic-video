# Architecture & Design Decisions: Agentic Video Understanding

## 1. Context & Motivation

Historically, AI models process video via **static uniform ingestion**: extracting frames at a fixed frequency (e.g. 1.0 FPS) and concatenating hundreds of images into model context. 

For long-form video, this brute-force approach fails across four dimensions:
1. **Context Window Saturation**: A 10-minute video at 1.0 FPS produces 600 full images ($600 \times 258 \approx 154,800$ tokens), exhausting prompt limits and causing severe attention dilution.
2. **Sub-Second Blindness**: Events that occur between integer seconds (e.g. at $t = 39.5\text{s}$) or brief visual transitions (< 0.5s duration) suffer from temporal aliasing and are completely missed by 1.0 FPS sampling.
3. **Temporal Hallucination**: Without pixel-level timecodes, vision-language models frequently fabricate timestamps or drift out of sync.
4. **Cloud & Modality Lock-in**: Relying on proprietary server-side video ingestion locks agents into cloud vendor pipelines where video processing logic is an uninspectable black box.

`agentic-video` solves these challenges by providing a **local-first, modular capability surface** that reproduces the core loop of Google's agentic video understanding: **Think → Act → Observe → Refine → Ground**.

---

## 2. Core Subsystems

### 2.1 Media Systems & Seeking Engine (`src/media/`)
- **Seeking Strategy**: Two-stage seeking (`-ss [t - margin]` before `-i` + `-ss [margin]` after `-i`) jumps via demuxer keyframe to $(t - 5\text{s})$ in $O(1)$ time, then decodes only 5 seconds forward. This keeps seeks fast while guaranteeing exact frame alignment without decoding from the beginning of large files.
- **Visual Timestamp Overlays**: Rather than using FFmpeg's `drawtext` filter (which requires system-dependent TTF font paths and fails across operating systems), visual badges (`#idx [HH:MM:SS.mmm]`) are composited using Pillow. This provides deterministic OCR text for vision models across Windows, macOS, and Linux Docker containers.
- **Contact Sheet Stitching**: High-contrast composite grids (e.g. 3x3 or 4x4 thumbnails with burned timestamps) compress 9–16 frames into a single image, so a coarse look at the timeline costs one image instead of many.
- **Scene Cut Detection**: High-speed stream filtering (`select='gt(scene,0.3)',metadata=print:key=lavfi.scene_score`) scans keyframes and packet headers at 180–300+ FPS without saving uncompressed frames to disk.

### 2.2 Transcript & ASR Engine (`src/transcript/`)
- **Priority Hierarchy**:
  1. Sidecar Subtitles (`.srt` / `.vtt`): Loaded in <2ms with zero compute overhead.
  2. Embedded Subtitle Streams: Demuxed using `ffmpeg -map 0:s:0`.
  3. Local ASR (`faster-whisper`): Uses `tiny` (39M) or `base` (74M) models running with `int8` quantization on CPU. Operates faster than real-time with <200MB memory footprint.
- **Transcript Search**: Keyword and n-gram matching with contextual padding returns exact segment start and end timestamps.

### 2.3 Retrieval & Orchestration (`src/retrieval/`)
- `AgenticVideoEngine` coordinates the workflow:
  - `probe()`: extracts container parameters and checks audio presence.
  - `transcribe()` and `search_transcript()`: finds verbal mentions.
  - `overview()`: builds uniform coarse contact sheets.
  - `inspect_window()`: zooms into candidate temporal windows at 1–10 FPS.
  - `build_evidence()`: packages verified timestamp spans, frame paths, and observable provenance (`evidence_basis`, `selection_reason`, `provenance`, `retrieval_method`) into immutable `EvidenceItem` objects.

### 2.4 Executor & MCP Capability Surface (`src/executor/`)
- Connects to **Executor** (`@executor-js/plugin-mcp`) via standard JSON-RPC 2.0 stdio transport.
- In Executor's code-mode (`execute`), agents write concise TypeScript scripts that run multi-step loops within the QuickJS WASM sandbox, collapsing 5+ conversational roundtrips into a single execution turn.
- Image artifacts are wrapped in Executor `ToolFile` payloads, allowing the host to project them directly as native MCP `{ type: "image" }` content blocks.
