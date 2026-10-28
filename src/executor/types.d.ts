/**
 * TypeScript Declarations for Executor's `tools.agentic_video.*` namespace.
 * 
 * In Executor code-mode (`execute`), these declarations define the global `tools`
 * proxy interface available to sandboxed agent scripts.
 *
 * Address formats in Executor:
 * - Full address: `tools.agentic_video.org.default.<tool>(args)`
 * - Shortened address (single connection / artifact binding): `tools.agentic_video.<tool>(args)`
 */

// ============================================================================
// Core Executor Primitives
// ============================================================================

/**
 * Standard error structure returned on expected tool/domain failures.
 */
export interface ToolError {
  /** Machine-readable error code (e.g. 'invalid_source', 'codec_unsupported', 'ffmpeg_error') */
  code: string;
  /** Human-readable explanation of the error */
  message: string;
  /** Optional HTTP status code if HTTP-backed */
  status?: number;
  /** Detailed error context or validation diagnostics */
  details?: unknown;
  /** Whether the caller can safely retry the operation */
  retryable?: boolean;
}

/**
 * HTTP transport metadata attached to responses when applicable.
 */
export interface ToolHttpMeta {
  status: number;
  headers: Record<string, string>;
}

/**
 * Executor standard tool outcome envelope.
 */
export type ToolResult<T> =
  | { readonly ok: true; readonly data: T; readonly http?: ToolHttpMeta }
  | { readonly ok: false; readonly error: ToolError };

/**
 * Executor binary file carrier.
 * When passed to `emit(toolFile)`, Executor's MCP host converts this into native
 * MCP content blocks (`image/*` becomes `{ type: "image", data, mimeType }`, etc.).
 */
export interface ToolFile {
  readonly _tag: "ToolFile";
  readonly name?: string;
  readonly mimeType: string;
  readonly encoding: "base64";
  readonly data: string;
  readonly byteLength: number;
}

/** Added to every successful result when `source` was a URL. */
export interface UrlSourceFields {
  /** The URL passed as `source`. */
  source_url?: string;
  /** Cached local copy of the URL; reuse it with other tools or ffmpeg. */
  local_path?: string;
}

/** Returned by any tool while a URL `source` is still downloading; call again with the same source. */
export interface SourceDownloadPending {
  status: "in_progress";
  retry_after_seconds: number;
  source_url: string;
  note: string;
}

export type WithUrlSource<T> = (T & UrlSourceFields) | SourceDownloadPending;

// ============================================================================
// Tool 1: probe_video
// ============================================================================

export interface ProbeVideoInput {
  /**
   * Local video file path, or a video URL (X/Twitter post, YouTube link, other yt-dlp sites, direct .mp4 link);
   * every other tool takes the same value. A URL is downloaded once and cached.
   */
  source: string;
}

export interface ProbeVideoOutput {
  /** Resolved canonical source path */
  source_path: string;
  /** Duration of video stream in seconds */
  duration_sec: number;
  /** Video frame dimensions */
  width: number;
  height: number;
  /** Frames per second (e.g. 29.97, 30.0, 60.0) */
  fps: number;
  /** Container format name */
  format_name: string;
  /** File size in bytes */
  size_bytes: number;
  /** Indicates whether an audio track is present in container */
  has_audio: boolean;
  audio_codec?: string;
  video_codec?: string;
  fingerprint: string;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
}

// ============================================================================
// Tool 2: transcribe
// ============================================================================

export type WhisperModelSize = "tiny" | "base" | "small" | "turbo";

export interface TranscribeInput {
  /** Video or audio source path */
  source: string;
  /** Whisper model size (default: tiny) */
  model_size?: WhisperModelSize;
}

export interface TranscriptSegment {
  /** Sequential segment identifier */
  id: number;
  /** Start timestamp in seconds */
  start_sec: number;
  /** End timestamp in seconds */
  end_sec: number;
  /** Transcribed speech text */
  text: string;
  start_timecode: string;
  end_timecode: string;
  /** Optional confidence; null when the source provides no confidence. */
  confidence: number | null;
  /** Only on a flagged line: why it may be invented (the same line 3 or more times in a row). A line without it is not verified. */
  suspect?: string;
}

export interface TranscribeTranscriptOutput {
  source_path: string;
  /** Detected or specified language */
  language: string;
  /** Total transcribed duration in seconds */
  duration_sec: number;
  segment_count: number;
  /** Ordered list of speech segments */
  segments: TranscriptSegment[];
  has_speech: boolean;
  notice: string;
  /** Aggregate confidence or null when unavailable. */
  confidence: number | null;
  /** Honest explanation of ASR/source uncertainty. */
  uncertainty: string;
  /** Present when a cached model is returned while the requested model is unavailable. */
  requested_model_size?: WhisperModelSize;
  cached_model_size?: WhisperModelSize;
  requested_model_cached?: false;
  transcription_status?: "started" | "in_progress" | "busy" | "start_failed" | "failed";
  retry_after_seconds?: number;
  transcription_message?: string;
  /** Only when the requested model is not installed: names the installed model used instead. */
  model_substitution?: string;
  /** Only when the transcript was too long to list in full: path of a JSON file with every segment. */
  transcript_path?: string;
  /** Only with transcript_path: how many segments are left out of `segments`. */
  segments_omitted?: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
}

export interface TranscriptionStatusOutput {
  status: "in_progress" | "busy" | "start_failed" | "failed";
  requested_model_size: WhisperModelSize;
  requested_model_cached: false;
  transcription_status: "started" | "in_progress" | "busy" | "start_failed" | "failed";
  retry_after_seconds: number;
  transcription_message: string;
  /** Only when the requested model is not installed: names the installed model used instead. */
  model_substitution?: string;
  /** While status is in_progress: seconds since the background transcription started. */
  elapsed_sec?: number;
  /** While status is in_progress: rough estimate of the whole run. */
  estimated_total_sec?: number;
  /** While status is in_progress: elapsed / estimated total, capped at 99. An estimate. */
  estimated_percent?: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
}

export type TranscribeOutput = TranscribeTranscriptOutput | TranscriptionStatusOutput;

// ============================================================================
// Tool 3: search_transcript
// ============================================================================

export interface SearchTranscriptInput {
  /** Video source path */
  source: string;
  /** Words or a phrase to find in the speech. */
  query: string;
  /** Maximum matches to return (default 5, maximum 50; larger values are clamped and the result says so). */
  top_k?: number;
  /** "hybrid" (default): keyword matches first, then matches by meaning. "lexical": keyword matches only. */
  mode?: "hybrid" | "lexical";
}

export interface TranscriptMatch {
  /** ID of corresponding transcript segment */
  segment_id: number;
  /** Exact start timestamp in seconds */
  start_sec: number;
  /** Exact end timestamp in seconds */
  end_sec: number;
  /** Matched transcript text */
  text: string;
  /** Relevance similarity score (0.0 to 1.0) */
  score: number;
  start_timecode: string;
  end_timecode: string;
  context: string;
  /** Only when the matched line is flagged; see TranscriptSegment.suspect. */
  suspect?: string;
  /** exact: the query as typed is in the line; all_terms: every content word is; semantic: close in meaning only, read its text before relying on it. */
  match?: "exact" | "all_terms" | "semantic";
}

export interface SearchTranscriptHitsOutput {
  hits: TranscriptMatch[];
  total: number;
  requested_model_size?: WhisperModelSize;
  cached_model_size?: WhisperModelSize;
  requested_model_cached?: false;
  transcription_status?: "started" | "in_progress" | "busy" | "start_failed" | "failed";
  retry_after_seconds?: number;
  transcription_message?: string;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The top_k you requested, when it was lowered to its cap (maximum 50). */
  top_k_clamped_from?: number;
}

export type SearchTranscriptOutput = SearchTranscriptHitsOutput | TranscriptionStatusOutput;

// ============================================================================
// Local visual index and semantic visual search
// ============================================================================

export type VisualJobStatus = "queued" | "running" | "complete" | "partial" | "error" | "busy" | "index_required";

export interface BuildVisualIndexInput {
  /** Video source path. */
  source: string;
}

export interface VisualIndexProgress {
  status: VisualJobStatus;
  /** End of the durably indexed prefix, in seconds. */
  indexed_until_sec?: number;
  duration_sec?: number;
  frames_indexed?: number;
  /** Fraction of video indexed, from 0 to 1. */
  progress?: number;
  model?: string;
  model_revision?: string;
  message?: string;
  retryable?: boolean;
  retry_after_seconds?: number;
}

export interface BuildVisualIndexOutput extends VisualIndexProgress {}

export interface SearchVisualInput {
  /** Video source path. */
  source: string;
  /** Natural-language description of visible content. */
  query: string;
  /** Maximum frame matches (default 5, maximum 20; larger values are clamped and the result says so). This ranks frames and cannot count: use count_occurrences for counts. */
  top_k?: number;
  /** Inclusive start of the range to search, in seconds. */
  start_sec?: number;
  /** Exclusive end of the range to search, in seconds; defaults to the end of the video. */
  end_sec?: number | null;
}

export interface VisualSearchHit {
  rank: number;
  timestamp_sec: number;
  timecode: string;
  /** Cosine similarity; this is a ranking score, not a probability. */
  score: number;
  frame_badge: string;
  timestamp_kind: "sample_target";
  decoded_timestamp: number | null;
}

export interface VisualSearchInterval {
  start_sec: number;
  end_sec: number;
  last_sample_sec?: number;
  score: number;
  ranks: number[];
}

export interface VisualContactSheet {
  file_path: string;
  tool_file?: ToolFile;
}

export interface SearchVisualOutput extends VisualIndexProgress {
  /** Missing, partially indexed, or complete; partial searches cover committed frames only. */
  index_status?: "missing" | "partial" | "complete";
  hits?: VisualSearchHit[];
  intervals?: VisualSearchInterval[];
  contact_sheet?: VisualContactSheet | null;
  index_complete?: boolean;
  note?: string;
  /** The top_k you requested, when it was lowered to its cap (maximum 20). */
  top_k_clamped_from?: number;
}

export interface CountRepetitionsInput {
  source: string;
  /** Start of the window in seconds (default 0). The window may be at most 180 s long. */
  start_sec?: number;
  /** End of the window in seconds; defaults to the end of the video. */
  end_sec?: number | null;
  /** Fractions of the frame; uses w/h (width/height with the same meaning is also accepted). Omit for the whole frame. */
  region?: { x: number; y: number; w: number; h: number } | { x: number; y: number; width: number; height: number };
  /** Sampling rate (default 10, maximum 15; larger values are clamped and the result says so). */
  fps?: number;
  /** auto (default) picks between motion, embedding and a periodicity check; motion or embedding forces that amplitude counter alone. */
  method?: "auto" | "motion" | "embedding";
  /** Shortest expected cycle in seconds (default 0.4). Bounds the amplitude counter only; the periodicity check ignores it and searches from 0.25 s. */
  min_period_sec?: number;
  /** Longest expected cycle in seconds (default 3); must exceed min_period_sec. Bounds the amplitude counter only; the periodicity check ignores it. */
  max_period_sec?: number;
  /** Short phrase naming the action being counted. */
  action_hint?: string;
}

export interface RepetitionCycle {
  index: number;
  start_sec: number;
  end_sec: number;
  peak_sec: number;
  frame_badge?: string;
}

export interface CountRepetitionsOutput {
  status: "in_progress" | "complete" | "error";
  retry_after_seconds?: number;
  message?: string;
  count?: number;
  cycles?: RepetitionCycle[];
  period_sec?: number | null;
  median_period_sec?: number | null;
  regularity?: number;
  confidence?: number;
  signal_quality?: string;
  method_used?: "motion" | "embedding" | "periodic";
  motion_count?: number;
  embedding_count?: number | null;
  periodic_count?: number | null;
  similarity_anchor_sec?: number;
  self_similarity_score?: number;
  contact_sheet?: VisualContactSheet | null;
  note?: string;
  /** The fps you requested, when it was lowered to its cap (maximum 15). */
  fps_clamped_from?: number;
}

export interface CountOccurrencesInput {
  source: string;
  /** Visible thing to count, e.g. "yellow outlined circle". Colour and shape words are checked. */
  description?: string;
  /** Exact on-screen phrase, matched as whole words by offline OCR (RELEASE does not match RELEASED). */
  text?: string;
  case_sensitive?: boolean;
  start_sec?: number;
  end_sec?: number | null;
  /** Sampling rate (default 5, maximum 15; larger values are clamped and the result says so); lowered automatically on long windows. */
  fps?: number;
  /** Absence of at least this long starts a new span (default 0.3). A larger value JOINS separate appearances whose gap is shorter than it; the result then sets gaps_merged and count_without_merges. */
  min_gap_sec?: number;
  /** Maximum spans listed (default 20, maximum 100; larger values are clamped and the result says so); count always includes omitted spans. */
  max_spans?: number;
}

export interface OccurrenceSpan {
  index: number;
  start_sec: number;
  end_sec: number;
  duration_sec?: number;
  score?: number;
  /** Timestamp to pass to inspect_window or contact_sheet to verify this span. */
  representative_sec: number;
  frame_badge?: string;
  /** Normalized box of the match at representative_sec. */
  region?: { x: number; y: number; w: number; h: number };
  travel?: "stationary" | "left_to_right" | "right_to_left" | "top_to_bottom" | "bottom_to_top";
  /** True for a single detection, a span shorter than 0.5 s, or a score below 75% of the median span score. Still counted; its tile label ends in *. */
  weak?: boolean;
}

/** One image with a labeled tile per listed span, in span order (the first 24). */
export interface OccurrenceContactSheet extends VisualContactSheet {
  /** Tile labels such as "#2* 29.9s" (span number, * when weak, time of the tile's frame). */
  labels?: string[];
}

export interface CountOccurrencesOutput {
  status: "in_progress" | "complete" | "error";
  retry_after_seconds?: number;
  message?: string;
  /** Number of separate appearances; null when `capped` is true. */
  count?: number | null;
  /** True when 100 or more spans matched: the description is too broad and count is null. Do not report a number. */
  capped?: boolean;
  /** How many spans matched; only set when capped. */
  spans_found?: number;
  /** Appearance pairs joined because min_gap_sec is above the default 0.3; only set when it joined some. */
  gaps_merged?: number;
  /** The count if the joined pairs were kept apart. */
  count_without_merges?: number;
  spans?: OccurrenceSpan[];
  spans_returned?: number;
  spans_omitted?: number;
  window?: { start_sec: number; end_sec: number };
  fps_used?: number;
  frames_sampled?: number;
  frames_analyzed?: number;
  checks?: { text: string | null; description: string | null; color: string | null; shape: string | null };
  /** Zoomed tile of each span's best detection, match boxed. Look at every tile and drop spans that are not the asked-for thing. */
  contact_sheet?: OccurrenceContactSheet | null;
  note?: string;
  /** How to check the spans against the sheet before answering. */
  verify?: string;
  /** The fps you requested, when it was lowered to its cap (maximum 15). */
  fps_clamped_from?: number;
  /** The max_spans you requested, when it was lowered to its cap (maximum 100). */
  max_spans_clamped_from?: number;
}

// ============================================================================
// Tool 4: detect_cuts
// ============================================================================

export interface DetectCutsInput {
  /** Video source path */
  source: string;
  /** Cut strength cutoff 0-1 (default 0.25); raise it to keep only stronger cuts. */
  threshold?: number;
  /** Optional inclusive range start in absolute video seconds */
  start_sec?: number;
  /** Optional exclusive range end in absolute video seconds */
  end_sec?: number;
}

export interface SceneCut {
  /** Time of the scene change in seconds. For a cut this is the first frame of the new shot; for a transition it is the middle. */
  timestamp_sec: number;
  /** Detector strength 0-1. Not comparable between kinds. A cut between two similar pictures can score below the threshold. */
  score: number;
  /** "cut" for a hard cut, "transition" for an animated change (fade, dissolve, slide, wipe, zoom). Film footage with heavy motion can still give false transitions. */
  kind: "cut" | "transition";
}

export interface DetectCutsOutput {
  cuts: SceneCut[];
  total: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** How many `cuts` entries were left out of the list; `note` says how to see them. */
  cuts_omitted?: number;
}

// ============================================================================
// Tool: check_frame_cadence
// ============================================================================

export interface CheckFrameCadenceInput {
  /** Video source path */
  source: string;
}

export interface CadenceRange {
  /** Time of the frame the first repeat copies, in seconds. */
  start_sec: number;
  /** Time just after the last repeated frame in seconds. */
  end_sec: number;
  /** Frames from start_sec to end_sec. */
  frames: number;
  /** Frames in the range that repeat the frame before them. */
  duplicate_frames: number;
  /** Unique frames per second inside the range: (frames - duplicate_frames) / (end_sec - start_sec). */
  effective_fps: number;
}

export interface CheckFrameCadenceOutput {
  /** Frame rate in frames per second (the stream's r_frame_rate; a variable-rate file is decoded at it, gaps filled with copies). */
  frame_rate: number;
  /** Frames decoded. */
  frames: number;
  /** Frames inside holds (runs of repeats longer than 0.5 s), left out of the ranges. */
  held_frames: number;
  ranges: CadenceRange[];
  /** Number of ranges found. */
  total: number;
  note?: string;
  /** How many `ranges` entries were left out of the list; `note` says how to see them. */
  ranges_omitted?: number;
}

// ============================================================================
// Tool 5: inspect_window
// ============================================================================

export interface InspectWindowInput {
  /** Video source path */
  source: string;
  /** Window start timestamp in seconds */
  start_sec: number;
  /** Window end timestamp in seconds; must be within the video. */
  end_sec: number;
  /** Sampling rate (default 4, maximum 30; larger values are clamped). Frames = fps x duration and must stay within about 150 (at most 600 s per call). */
  fps?: number;
}

export interface DenseWindowFrame {
  index: number;
  timestamp_sec: number;
  timecode: string;
  file_path: string;
  width: number;
  height: number;
  requested_timestamp: number | null;
  decoded_timestamp: number | null;
  timestamp_delta: number | null;
}

export interface InspectWindowOutput {
  source_path: string;
  start_sec: number;
  end_sec: number;
  fps: number;
  frames: DenseWindowFrame[];
  contact_sheet_path?: string;
  estimated_vision_tokens: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The fps you requested, when it was lowered to its cap (maximum 30). */
  fps_clamped_from?: number;
}

// ============================================================================
// Tool 6: contact_sheet
// ============================================================================

export interface RefineTemporalBeamInput {
  source: string;
  windows: Array<{ start_sec: number; end_sec: number }>;
  /** Child ranges per window (default 4, maximum 8; larger values are clamped). */
  fanout?: number;
  /** Thumbnails per child (default 3, maximum 12; larger values are clamped). */
  samples_per_child?: number;
  cols?: number;
  context_sec?: number;
  /** Budget: windows x fanout x samples_per_child must not exceed it (default 96, maximum 300; larger values are clamped). */
  max_total_samples?: number;
  align_to_cuts?: boolean;
  compact_parent_sheet?: boolean;
}
export interface RefineTemporalBeamChild {
  child_index: number; start_sec: number; end_sec: number;
  sample_timestamps: number[]; contact_sheet_path: string; sample_count: number; tool_file?: ToolFile;
}
export interface RefineTemporalBeamParent {
  parent_index: number; input_start_sec: number; input_end_sec: number;
  start_sec: number; end_sec: number; children: RefineTemporalBeamChild[];
  compact_contact_sheet_path?: string; compact_sample_timestamps?: number[];
  compact_sample_count?: number; compact_tool_file?: ToolFile;
}
export interface RefineTemporalBeamOutput {
  parents: RefineTemporalBeamParent[]; total_samples: number; estimated_vision_tokens: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The fanout you requested, when it was lowered to its cap (maximum 8). */
  fanout_clamped_from?: number;
  /** The samples_per_child you requested, when it was lowered to its cap (maximum 12). */
  samples_per_child_clamped_from?: number;
  /** The max_total_samples you requested, when it was lowered to its cap (maximum 300). */
  max_total_samples_clamped_from?: number;
}
export interface TrackRegionChangesInput {
  source: string;
  start_sec: number;
  end_sec: number;
  /** Fractions of the frame; uses width/height, not w/h (w/h with the same meaning is also accepted). */
  region: { x: number; y: number; width: number; height: number } | { x: number; y: number; w: number; h: number };
  /** Sampling rate (default 0.25, maximum 10; larger values are clamped and the result says so). */
  sample_fps?: number;
  /** Decoded frame width in pixels (default 640, maximum 1920; larger values are clamped). */
  frame_width?: number;
  min_change_score?: number;
  robust_sigma?: number;
  min_separation_sec?: number;
  /** Most candidates returned (default 32, maximum 128; larger values are clamped). */
  max_candidates?: number;
  /** Frame budget: (end_sec - start_sec) x sample_fps must not exceed it (default 600, maximum 2000; larger values are clamped). */
  max_total_frames?: number;
}

export interface TimestampProvenance {
  requested_timestamp: number | null;
  decoded_timestamp: number | null;
  timestamp_delta: number | null;
}

export interface TrackRegionChangeSample {
  from_timestamp_sec: number;
  timestamp_sec: number;
  from_timestamp_provenance: TimestampProvenance;
  timestamp_provenance: TimestampProvenance;
  change_score: number;
  mean_abs_diff: number;
  changed_fraction: number;
  rank?: number;
}

export interface TrackRegionChangesOutput {
  region: { x: number; y: number; width: number; height: number };
  start_sec: number;
  end_sec: number;
  sample_fps: number;
  sample_count: number;
  temporal_resolution_sec: number;
  scan_segments: Array<{ start_sec: number; end_sec: number; sample_count: number }>;
  threshold: number;
  baseline_median: number;
  baseline_mad: number;
  steady_change: { detected: boolean; active_fraction: number; activity_floor: number; note: string };
  samples: TrackRegionChangeSample[];
  candidates: TrackRegionChangeSample[];
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The sample_fps you requested, when it was lowered to its cap (maximum 10). */
  sample_fps_clamped_from?: number;
  /** The frame_width you requested, when it was lowered to its cap (maximum 1920). */
  frame_width_clamped_from?: number;
  /** The max_candidates you requested, when it was lowered to its cap (maximum 128). */
  max_candidates_clamped_from?: number;
  /** The max_total_frames you requested, when it was lowered to its cap (maximum 2000). */
  max_total_frames_clamped_from?: number;
}

export interface TrackRegionEventsInput {
  source: string;
  start_sec: number;
  end_sec: number;
  /** Fractions of the frame; uses width/height, not w/h (w/h with the same meaning is also accepted). */
  region: { x: number; y: number; width: number; height: number } | { x: number; y: number; w: number; h: number };
  /** Sampling rate (default 0.25, maximum 10; larger values are clamped and the result says so). */
  sample_fps?: number;
  /** Decoded frame width in pixels (default 640, maximum 1920; larger values are clamped). */
  frame_width?: number;
  change_threshold?: number;
  ambiguity_margin?: number;
  min_separation_sec?: number;
  /** Most events returned (default 32, maximum 128; larger values are clamped). */
  max_events?: number;
  /** Frame budget: (end_sec - start_sec) x sample_fps must not exceed it (default 600, maximum 2000; larger values are clamped). */
  max_total_frames?: number;
}

export interface TrackRegionEventTransition {
  from_timestamp_sec: number;
  timestamp_sec: number;
  from_timestamp_provenance: TimestampProvenance;
  timestamp_provenance: TimestampProvenance;
  change_score: number;
  mean_abs_diff: number;
  changed_fraction: number;
  evidence_moments: {
    before: { timestamp_sec: number; timestamp_provenance: TimestampProvenance };
    after: { timestamp_sec: number; timestamp_provenance: TimestampProvenance };
  };
  rank: number;
}

export interface TrackRegionEventsOutput {
  region: { x: number; y: number; width: number; height: number };
  start_sec: number;
  end_sec: number;
  sample_fps: number;
  sample_count: number;
  temporal_resolution_sec: number;
  scan_segments: Array<{ start_sec: number; end_sec: number; sample_count: number }>;
  transitions: TrackRegionEventTransition[];
  event_count: number | null;
  uncertain: boolean;
  incomplete: boolean;
  uncertainty_reason: string | null;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The sample_fps you requested, when it was lowered to its cap (maximum 10). */
  sample_fps_clamped_from?: number;
  /** The frame_width you requested, when it was lowered to its cap (maximum 1920). */
  frame_width_clamped_from?: number;
  /** The max_events you requested, when it was lowered to its cap (maximum 128). */
  max_events_clamped_from?: number;
  /** The max_total_frames you requested, when it was lowered to its cap (maximum 2000). */
  max_total_frames_clamped_from?: number;
}

export interface InspectMomentsInput {
  source: string;
  moments: Array<{ timestamp_sec: number; label?: string }>;
  /** Seconds of context before each moment (default 4). */
  before_sec?: number;
  /** Seconds of context after each moment (default 4). */
  after_sec?: number;
  /** Frames per moment, an odd number 3-25 (default 9; larger values are clamped to 25). */
  samples_per_moment?: number;
  cols?: number;
  /** Most moments accepted (default and maximum 16; larger values are clamped). */
  max_moments?: number;
  /** Budget: moments x samples_per_moment must not exceed it (default 144, maximum 400; larger values are clamped). */
  max_total_samples?: number;
}

export interface InspectMomentItem {
  index: number;
  timestamp_sec: number;
  label?: string | null;
  start_sec: number;
  end_sec: number;
  anchor_sample_index: number;
  sample_timestamps: number[];
  relative_offsets_sec: number[];
  contact_sheet_path: string;
  sample_count: number;
  tool_file?: ToolFile;
}

export interface InspectMomentsOutput {
  moments: InspectMomentItem[];
  total: number;
  total_samples: number;
  estimated_vision_tokens: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The samples_per_moment you requested, when it was lowered to its cap (maximum 25). */
  samples_per_moment_clamped_from?: number;
  /** The max_moments you requested, when it was lowered to its cap (maximum 16). */
  max_moments_clamped_from?: number;
  /** The max_total_samples you requested, when it was lowered to its cap (maximum 400). */
  max_total_samples_clamped_from?: number;
}

export interface InspectWindowsInput {
  source: string;
  /** Absolute half-open source windows to inspect. */
  windows: Array<{ start_sec: number; end_sec: number }>;
  /** Sampling rate (default 4, maximum 30; larger values are clamped). */
  fps?: number;
  context_sec?: number;
  merge_overlaps?: boolean;
  /** Most windows accepted (default and maximum 16; larger values are clamped). */
  max_windows?: number;
  /** Frame budget for all windows together (default 150, maximum 500; larger values are clamped). */
  max_total_frames?: number;
}

export interface InspectWindowsItem {
  index: number;
  /** Zero-based indices of input windows represented by this merged item. */
  input_indices: number[];
  start_sec: number;
  end_sec: number;
  fps: number;
  frames: DenseWindowFrame[];
  contact_sheet_path?: string | null;
  estimated_vision_tokens: number;
  tool_file?: ToolFile;
}

export interface InspectWindowsOutput {
  windows: InspectWindowsItem[];
  total: number;
  total_frames: number;
  estimated_vision_tokens: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The fps you requested, when it was lowered to its cap (maximum 30). */
  fps_clamped_from?: number;
  /** The max_windows you requested, when it was lowered to its cap (maximum 16). */
  max_windows_clamped_from?: number;
  /** The max_total_frames you requested, when it was lowered to its cap (maximum 500). */
  max_total_frames_clamped_from?: number;
}

export interface ContactSheetInput {
  /** Video source path */
  source: string;
  /** Requested thumbnail count (default 12); values above 150 are accepted and clamped to 150, and the result says so. */
  samples?: number;
  /** Number of grid columns (default: 4) */
  cols?: number;
  /** Optional inclusive range start in absolute video seconds */
  start_sec?: number;
  /** Optional exclusive range end in absolute video seconds */
  end_sec?: number;
}

export interface ContactSheetOutput {
  file_path: string;
  sample_count: number;
  rows: number;
  cols: number;
  start_sec: number;
  end_sec: number;
  timestamps: number[];
  frame_width: number;
  frame_height: number;
  samples_clamped_from?: number;
  note?: string;
  /** Absolute file path to composite image */
  /** Executor ToolFile object ready for emit(ToolFile) */
  tool_file?: ToolFile;
}

// ============================================================================
// ============================================================================
// Unified Agentic Video Namespace Interface in Executor
// ============================================================================

export interface ReviewVideoInput {
  source: string;
  /** Contact sheet thumbnails (default 24); values above 150 are clamped to 150 and the result says so. */
  samples?: number;
}

export interface ReviewVideoLoudness {
  /** Energy-averaged RMS level over the whole video, in dBFS. */
  average_rms_dbfs: number | null;
  peak_dbfs: number;
  peak_sec: number;
  window_sec: number;
  silence_threshold_dbfs: number;
  /** At most 20 silent stretches, in time order. */
  silence_gaps: Array<{ start_sec: number; end_sec: number; duration_sec: number; threshold_dbfs: number }>;
  silence_gaps_total: number;
  outliers_total: number;
  fault_outliers_total: number;
  /** At most 10: fault-level first, then the largest delta_db. Call loudness_timeline for the full list. */
  outliers: Array<{
    start_sec: number;
    end_sec: number;
    band: string;
    direction: string;
    shape: string;
    delta_db: number;
    fault_level: boolean;
  }>;
}

export interface ReviewVideoOutput {
  /** Same fields as probe_video. */
  metadata: ProbeVideoOutput;
  /** Same fields as contact_sheet. */
  contact_sheet: ContactSheetOutput;
  cuts: SceneCut[];
  cuts_total: number;
  cuts_omitted?: number;
  /** Null when the video has no audio track. */
  loudness: ReviewVideoLoudness | null;
  note?: string;
}

export interface OverviewChunksInput {
  source: string;
  /** Target duration of each contiguous coarse chunk, in seconds (default 60, maximum 600). */
  chunk_sec?: number;
  /** Uniform samples placed inside each chunk (default 8; values above 150 are clamped to 150 and the result says so). */
  samples_per_chunk?: number;
  cols?: number;
  /** Optional inclusive absolute-video range start. */
  start_sec?: number;
  /** Optional exclusive absolute-video range end. */
  end_sec?: number;
  /** Most sheets in one call (default and maximum 32; larger values are clamped). Raise chunk_sec or narrow the range for a longer video. */
  max_chunks?: number;
}

export interface OverviewChunk extends ContactSheetOutput {
  index: number;
  range_start_sec: number;
  range_end_sec: number;
}

export interface OverviewChunksOutput {
  chunks: OverviewChunk[];
  total: number;
  start_sec: number;
  end_sec: number;
  chunk_sec: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The samples_per_chunk you requested, when it was lowered to its cap (maximum 150). */
  samples_per_chunk_clamped_from?: number;
  /** The max_chunks you requested, when it was lowered to its cap (maximum 32). */
  max_chunks_clamped_from?: number;
}

// ============================================================================
// Non-speech audio and synchronization tools
// ============================================================================

export type AudioBandName = "sub" | "low_mid" | "mid" | "high" | "air";
export type AudioBandSelection = AudioBandName | [low_hz: number, high_hz: number];

export interface AudioFrequencyBand {
  name: AudioBandName | null;
  low_hz: number;
  high_hz: number;
}

export interface AudioBandProfile {
  sub: number;
  low_mid: number;
  mid: number;
  high: number;
  air: number;
}

export interface AudioOnsetsInput {
  source: string;
  start_sec?: number;
  end_sec?: number;
  /** Minimum normalized spectral-flux strength, from 0 to 1 (default 0.12); raise it to keep only stronger onsets. */
  min_strength?: number;
  /** Restrict onset detection to a named band or [low_hz, high_hz]. */
  band_hz?: AudioBandSelection;
  /** Return only source onsets that are new or materially louder in-band than this source (a path or video URL). */
  compare_to?: string;
}

export interface AudioOnset {
  time_sec: number;
  strength: number;
  band_db: AudioBandProfile;
  band_energy_db?: number;
  compare_band_energy_db?: number;
  compare_delta_db?: number;
}

export interface AudioOnsetsOutput {
  source_path: string;
  start_sec: number;
  end_sec: number;
  sample_rate: number;
  hop_samples: number;
  time_resolution_ms: number;
  min_strength: number;
  band_hz: AudioFrequencyBand | null;
  compare_to: string | null;
  compare_min_delta_db: number | null;
  onsets: AudioOnset[];
  total: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** How many `onsets` entries were left out of the list; `note` says how to see them. */
  onsets_omitted?: number;
}

export interface LoudnessTimelineInput {
  source: string;
  /** Fixed analysis window, 0.01-600 seconds (default 0.25; larger values are clamped to 600). Give this or bpm, not both. */
  window_sec?: number;
  /** Analyze one beat per window; overrides the default fixed window. */
  bpm?: number;
  start_sec?: number;
  end_sec?: number;
  ignore_ranges?: Array<{ start_sec: number; end_sec: number; reason: string }>;
}

export interface LoudnessWindow {
  start_sec: number;
  end_sec: number;
  rms_dbfs: number;
  peak_dbfs: number;
  band_db: AudioBandProfile;
  spectral_centroid_hz: number;
  silent: boolean;
}

export interface AudioBandOutlier {
  start_sec: number;
  end_sec: number;
  band: AudioBandName;
  energy_db: number;
  baseline_db: number;
  delta_db: number;
  direction: "boost" | "cut";
  shape: "step" | "ramp" | "transient";
  classification: "fault" | "informational";
  fault_level: boolean;
  ignored: boolean;
  ignore_reason: string | null;
  min_delta_db: number;
  max_delta_db: number;
  energy_start_db: number;
  energy_end_db: number;
  entry_jump_db: number | null;
  exit_jump_db: number | null;
  internal_range_db: number;
  trend_db: number;
  monotonic_fraction: number;
  window_count: number;
}

export interface SilenceGap {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  threshold_dbfs: number;
}

export interface LoudnessTimelineOutput {
  source_path: string;
  start_sec: number;
  end_sec: number;
  window_sec: number;
  bpm: number | null;
  silence_threshold_dbfs: number;
  /** Loudest sample on any channel in the range (sample peak, as ffmpeg astats/ebur128 report it). */
  peak_dbfs: number;
  /** Time of that loudest sample. */
  peak_sec: number;
  windows: LoudnessWindow[];
  outliers: AudioBandOutlier[];
  ignore_ranges: Array<{ start_sec: number; end_sec: number; reason: string }>;
  silence_gaps: SilenceGap[];
  total: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The window_sec you requested, when it was lowered to its cap (maximum 600). */
  window_sec_clamped_from?: number;
  /** How many `windows` entries were left out of the list; `note` says how to see them. */
  windows_omitted?: number;
}

export interface SpectrogramMark {
  time_sec: number;
  label: string;
}

export interface SpectrogramFrequencyTick {
  frequency_hz: number;
  label: string;
  y_px: number;
}

export interface SpectrogramWindowInput {
  source: string;
  start_sec: number;
  end_sec: number;
  marks?: SpectrogramMark[];
}

export interface SpectrogramWindowOutput {
  source_path: string;
  start_sec: number;
  end_sec: number;
  file_path: string;
  width: number;
  height: number;
  sample_rate: number;
  hop_samples: number;
  timecode_badges: string[];
  frequency_ticks: SpectrogramFrequencyTick[];
  marks: SpectrogramMark[];
  tool_file?: ToolFile;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
}

export interface WatchForInput {
  source: string;
  conditions: string[];
  start_sec?: number;
  end_sec?: number;
  interval_sec?: number;
  cut?: number;
  consecutive?: number;
  cooldown_sec?: number;
}

export interface WatchCondition {
  index: number;
  condition: string;
  wakes: number;
  dropped_by_cooldown: number;
  /** Highest chance of yes over all samples, and the sample time it happened. */
  max_chance: number;
  max_chance_sec: number;
}

export interface WatchWake {
  /** Sample that completed the run of yes samples. */
  time_sec: number;
  condition_index: number;
  condition: string;
  /** First sample of the unbroken run of yes samples. */
  first_seen_sec: number;
  chance: number;
}

export interface WatchForOutput {
  source_path: string;
  start_sec: number;
  /** Time of the last sample looked at. */
  scanned_to_sec: number;
  interval_sec: number;
  cut: number;
  consecutive: number;
  cooldown_sec: number;
  model: string;
  conditions: WatchCondition[];
  wakes: WatchWake[];
  samples: number;
  /** False when the call stopped early to stay under the 60 s tool limit; continue from next_start_sec. */
  complete: boolean;
  next_start_sec: number | null;
  input_tokens: number;
  note?: string;
}

export interface LabelSoundsInput {
  source: string;
  start_sec: number;
  end_sec: number;
  labels?: string[];
  window_sec?: number;
}

export interface SoundLabelWindow {
  start_sec: number;
  end_sec: number;
  /** The likeliest label, or null when none_chance beats every label. */
  top_label: string | null;
  top_chance: number;
  none_chance: number;
  /** Chance of each label for this window; with none_chance they sum to 1. */
  chances: Record<string, number>;
}

export interface LabelSoundsOutput {
  source_path: string;
  start_sec: number;
  end_sec: number;
  window_sec: number;
  model: string;
  labels: string[];
  windows: SoundLabelWindow[];
  input_tokens: number;
  complete: boolean;
  next_start_sec: number | null;
  note?: string;
}

export type SyncCueInput =
  | { time_sec: number; t?: number; label?: string; band_hz?: AudioBandSelection }
  | { t: number; time_sec?: number; label?: string; band_hz?: AudioBandSelection };

export interface SyncCheckInput {
  source: string;
  cues?: SyncCueInput[];
  /** Match window in milliseconds (default 80, maximum 5000; larger values are clamped). */
  tolerance_ms?: number;
  /** Tempo of the music, 20-300: compares cues with uncued moments at the same beat phase. */
  bpm?: number;
}

export interface SyncCutMatch {
  time_sec: number;
  score: number;
  nearest_onset: { time_sec: number; strength: number } | null;
  offset_ms: number | null;
}

export interface SyncCueResult {
  time_sec: number;
  label: string;
  status: "ok" | "early" | "late" | "missing";
  matched_onset_time_sec: number | null;
  offset_ms: number | null;
  band_hz?: AudioBandSelection;
}

export interface SyncCheckOutput {
  source_path: string;
  tolerance_ms: number;
  cue_recovery_window_ms: number;
  cuts: SyncCutMatch[];
  cues: SyncCueResult[];
  unmatched_strong_onsets: AudioOnset[];
  strong_onset_threshold: number;
  total_cuts: number;
  total_cues: number;
  /** Plain-English remark about the call: a capped argument, omitted items, or an ignored unknown argument. */
  note?: string;
  /** The tolerance_ms you requested, when it was lowered to its cap (maximum 5000). */
  tolerance_ms_clamped_from?: number;
  /** How many `cuts` entries were left out of the list; `note` says how to see them. */
  cuts_omitted?: number;
  /** How many `unmatched_strong_onsets` entries were left out of the list; `note` says how to see them. */
  unmatched_strong_onsets_omitted?: number;
}

export interface AgenticVideoIntegration {
  /**
   * Read duration, resolution, fps, codecs, and audio presence. Call first: duration bounds every start_sec/end_sec
   * and has_audio says whether transcribe and the audio tools can work.
   */
  probe_video(input: ProbeVideoInput): Promise<ToolResult<WithUrlSource<ProbeVideoOutput>>>;

  /**
   * Full speech transcript as timestamped segments. To find a phrase use search_transcript. Repeat the same call
   * while status is in_progress.
   */
  transcribe(input: TranscribeInput): Promise<ToolResult<WithUrlSource<TranscribeOutput>>>;

  /**
   * Find when a word or phrase is spoken (keyword matches first, then matches by meaning). For on-screen text use count_occurrences; for visible
   * things use search_visual.
   */
  search_transcript(input: SearchTranscriptInput): Promise<ToolResult<WithUrlSource<SearchTranscriptOutput>>>;

  /** Start or poll bounded background construction of the local visual index. */
  build_visual_index(input: BuildVisualIndexInput): Promise<ToolResult<WithUrlSource<BuildVisualIndexOutput>>>;

  /**
   * Find when something visible happens; returns ranked frames (top_k up to 20). It ranks frames and cannot count:
   * use count_occurrences for separate appearances and count_repetitions for repeated actions.
   */
  search_visual(input: SearchVisualInput): Promise<ToolResult<WithUrlSource<SearchVisualOutput>>>;

  /** Count cycles of a repeated visual action inside a window of at most 180 s. */
  count_repetitions(input: CountRepetitionsInput): Promise<ToolResult<WithUrlSource<CountRepetitionsOutput>>>;

  /** Count separate appearances of a described symbol or exact on-screen text; no visual index needed. */
  count_occurrences(input: CountOccurrencesInput): Promise<ToolResult<WithUrlSource<CountOccurrencesOutput>>>;

  /**
   * List hard scene cuts and transitions (at most 500 listed; total is the full count).
   */
  detect_cuts(input: DetectCutsInput): Promise<ToolResult<WithUrlSource<DetectCutsOutput>>>;

  /** Find stretches where the unique-frame rate is below the container frame rate (duplicate-frame judder); at most 500 listed, total is the full count. */
  check_frame_cadence(input: CheckFrameCadenceInput): Promise<ToolResult<WithUrlSource<CheckFrameCadenceOutput>>>;

  /** Split candidate windows into contiguous child ranges with one contact sheet each (budget: windows x fanout x samples_per_child). */
  refine_temporal_beam(input: RefineTemporalBeamInput): Promise<ToolResult<WithUrlSource<RefineTemporalBeamOutput>>>;

  /** Rank the moments a fixed frame rectangle changes most; candidates only, not a count. */
  track_region_changes(input: TrackRegionChangesInput): Promise<ToolResult<WithUrlSource<TrackRegionChangesOutput>>>;

  /** Count state transitions inside a fixed frame rectangle, with before/after evidence moments. */
  track_region_events(input: TrackRegionEventsInput): Promise<ToolResult<WithUrlSource<TrackRegionEventsOutput>>>;

  /** Contact sheets of frames before, at and after specific instants (at most 16 moments per call). */
  inspect_moments(input: InspectMomentsInput): Promise<ToolResult<WithUrlSource<InspectMomentsOutput>>>;

  /** Dense frames for several time ranges under one frame budget. */
  inspect_windows(input: InspectWindowsInput): Promise<ToolResult<WithUrlSource<InspectWindowsOutput>>>;

  /** Dense consecutive frames of one short window (about 150 frames per call). */
  inspect_window(input: InspectWindowInput): Promise<ToolResult<WithUrlSource<InspectWindowOutput>>>;

  /**
   * One grid image of evenly spaced frames with timestamps: the cheapest way to survey a video (samples up to 150).
   */
  contact_sheet(input: ContactSheetInput): Promise<ToolResult<WithUrlSource<ContactSheetOutput>>>;

  /** First look in one call: metadata, contact sheet, cuts and a loudness summary (runs probe_video, contact_sheet, detect_cuts, loudness_timeline). */
  review_video(input: ReviewVideoInput): Promise<ToolResult<WithUrlSource<ReviewVideoOutput>>>;

  /** Survey a long video as consecutive contact sheets (at most 32 per call). */
  overview_chunks(input: OverviewChunksInput): Promise<ToolResult<WithUrlSource<OverviewChunksOutput>>>;

  /** List non-speech sound attacks with band profiles (at most 500 listed; total is the full count). */
  audio_onsets(input: AudioOnsetsInput): Promise<ToolResult<WithUrlSource<AudioOnsetsOutput>>>;

  /** Measure loudness in fixed or beat-length windows and report silence and band outliers (at most 500 windows listed). */
  loudness_timeline(input: LoudnessTimelineInput): Promise<ToolResult<WithUrlSource<LoudnessTimelineOutput>>>;

  /** Render a timecoded log-frequency spectrogram as a PNG ToolFile (window of at most 600 s). */
  spectrogram_window(input: SpectrogramWindowInput): Promise<ToolResult<WithUrlSource<SpectrogramWindowOutput>>>;

  /** Compare visual cut times, sound onsets and optional intended cues (each list shows at most 500 items). */
  sync_check(input: SyncCheckInput): Promise<ToolResult<WithUrlSource<SyncCheckOutput>>>;

  /** Watch a range for yes/no conditions with a cloud model (sends frames to Cloudflare Workers AI; at most 720 samples; answers within about 50 s). */
  watch_for(input: WatchForInput): Promise<ToolResult<WatchForOutput>>;

  /** Name the sounds in audio windows with a cloud model (sends the audio to Cloudflare Workers AI; at most 60 windows). */
  label_sounds(input: LabelSoundsInput): Promise<ToolResult<LabelSoundsOutput>>;

}

// ============================================================================
// Global Sandbox Augmented Interface
// ============================================================================

declare global {
  /**
   * Emits user-visible output in the Executor MCP session.
   * Passing a ToolFile with image/* MIME renders an inline image directly in the client.
   */
  function emit(value: unknown | ToolFile): void;

  /**
   * Sandboxed Executor tools root proxy.
   */
  const tools: {
    agentic_video: {
      org: {
        default: AgenticVideoIntegration;
        [connectionName: string]: AgenticVideoIntegration;
      };
      [role: string]: any;
    };
    [integration: string]: any;
  };
}
