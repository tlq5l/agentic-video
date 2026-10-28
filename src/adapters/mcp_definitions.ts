/**
 * MCP registration for the live agentic-video contract.
 *
 * The language-neutral source of truth is mcp_schemas.json.  This adapter
 * mirrors the current tools exposed by src.executor.mcp_server and deliberately
 * does not register the retired sampling/clip surface.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

/** Every tool's `source`: a local path, or a URL that the server downloads once and caches. */
const SourceSchema = z.string().trim().min(1).describe(
  "Local video file path, or a video URL (X/Twitter post, YouTube link, other sites yt-dlp supports, direct .mp4 link). " +
  "A URL is downloaded once and cached; while it downloads the result is status=in_progress with retry_after_seconds, so call again with the same source.",
);

export const ProbeVideoInputSchema = {
  source: SourceSchema,
};

export const TranscribeInputSchema = {
  source: SourceSchema,
  model_size: z.enum(["tiny", "base", "small", "turbo"]).optional(),
};

export const SearchTranscriptInputSchema = {
  source: SourceSchema,
  query: z.string().trim().min(1).describe("Words or a phrase to find in the speech, e.g. 'launch date'. Punctuation and case are ignored."),
  top_k: z.number().int().min(1).max(50).describe("Maximum hits (default 5, maximum 50; larger values are clamped to 50 and the result says so).").optional(),
  mode: z.enum(["hybrid", "lexical"]).describe("hybrid (default): keyword matches first, then matches by meaning. lexical: keyword matches only.").optional(),
};

export const BuildVisualIndexInputSchema = {
  source: SourceSchema,
};

export const SearchVisualInputSchema = {
  source: SourceSchema,
  query: z.string().trim().min(1).max(2000).describe("Natural-language description of visible content."),
  top_k: z.number().int().min(1).max(20).describe("Maximum frame hits (default 5, maximum 20; larger values are clamped to 20 and the result says so). This ranks frames and cannot count: use count_occurrences for counts.").optional(),
  start_sec: z.number().min(0).describe("Inclusive start of the range to search, in seconds.").optional(),
  end_sec: z.number().min(0).nullable().describe("Exclusive end of the range to search, in seconds; defaults to the end of the video.").optional(),
};

export const CountRepetitionsInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the window in seconds (default 0). The window may be at most 180 s long.").optional(),
  end_sec: z.number().min(0).nullable().describe("End of the window in seconds; defaults to the end of the video. Give both bounds for a video longer than 180 s.").optional(),
  region: z.object({ x: z.number().min(0).max(1), y: z.number().min(0).max(1), w: z.number().min(0).max(1), h: z.number().min(0).max(1) }).describe("Rectangle that contains the repeating motion, as fractions of the frame: x, y (top-left) and w, h (size), with x+w <= 1 and y+h <= 1. Uses w/h, not width/height. Omit for the whole frame.").optional(),
  fps: z.number().positive().max(15).describe("Sampling rate, default 10. Larger values than 15 are clamped to 15 and the result says so.").optional(),
  method: z.enum(["auto", "motion", "embedding"]).describe("auto (default) picks between motion, embedding and a periodicity check; motion or embedding forces that amplitude counter alone.").optional(),
  min_period_sec: z.number().positive().describe("Shortest expected cycle in seconds (default 0.4). Bounds the amplitude counter only; the periodicity check ignores it and searches from 0.25 s.").optional(),
  max_period_sec: z.number().positive().describe("Longest expected cycle in seconds (default 3); must exceed min_period_sec. Bounds the amplitude counter only; the periodicity check ignores it.").optional(),
  action_hint: z.string().max(200).describe("Short phrase naming the action being counted, e.g. 'jumping jacks'.").optional(),
};

export const CountOccurrencesInputSchema = {
  source: SourceSchema,
  description: z.string().min(1).max(500).describe("Visible thing to count, e.g. 'yellow outlined circle', 'green diamond', 'red star'. Name its colour and shape; text on screen goes in `text`.").optional(),
  text: z.string().min(1).max(200).describe("Exact on-screen phrase, matched as whole words (RELEASE matches 'RELEASE' and 'RELEASE NOTES', not 'RELEASED' or 'PRE-RELEASE').").optional(),
  case_sensitive: z.boolean().describe("Require the same letter case for `text`.").optional(),
  start_sec: z.number().min(0).optional(),
  end_sec: z.number().min(0).nullable().describe("Defaults to the end of the video. At most 2400 frames are analyzed per call, so fps is lowered automatically on long windows.").optional(),
  fps: z.number().positive().max(15).describe("Sampling rate, default 5. Appearances shorter than 1/fps can be missed; raise it for brief flashes. Larger values than 15 are clamped to 15 and the result says so.").optional(),
  min_gap_sec: z.number().positive().describe("Absence of at least this long starts a new span; shorter dropouts stay inside one span. Raising it above the default 0.3 JOINS separate appearances whose gap is shorter than it (0.8 merges appearances 0.7 s apart); the result reports gaps_merged and count_without_merges when that happens.").optional(),
  max_spans: z.number().int().min(1).max(100).describe("Maximum spans listed (default 20, maximum 100; larger values are clamped to 100 and the result says so). `count` always includes every span.").optional(),
};

export const ContactSheetInputSchema = {
  source: SourceSchema,
  samples: z.number().int().min(1).describe("Requested number of thumbnails; values above 150 are accepted and clamped to 150.").optional(),
  cols: z.number().int().min(1).describe("Grid columns (default 4).").optional(),
  start_sec: z.number().min(0).describe("Optional inclusive start of the overview range in absolute video seconds.").optional(),
  end_sec: z.number().min(0).describe("Optional exclusive end of the overview range in absolute video seconds.").optional(),
};

export const ReviewVideoInputSchema = {
  source: SourceSchema,
  samples: z.number().int().min(1).describe("Contact sheet thumbnails (default 24; values above 150 are clamped to 150 and the result says so).").optional(),
};

export const OverviewChunksInputSchema = {
  source: SourceSchema,
  chunk_sec: z.number().positive().max(600).describe("Seconds covered by each sheet (default 60, maximum 600).").optional(),
  samples_per_chunk: z.number().int().min(1).max(150).describe("Thumbnails per sheet (default 8; values above 150 are clamped to 150 and the result says so).").optional(),
  cols: z.number().int().min(1).optional(),
  start_sec: z.number().min(0).describe("Optional inclusive start in absolute video seconds.").optional(),
  end_sec: z.number().min(0).describe("Optional exclusive end in absolute video seconds; defaults to the end of the video.").optional(),
  max_chunks: z.number().int().min(1).max(32).describe("Most sheets in one call (default and maximum 32; larger values are clamped). The call fails if the range needs more chunks: raise chunk_sec or narrow the range.").optional(),
};

export const RefineTemporalBeamInputSchema = {
  source: SourceSchema,
  windows: z.array(z.object({ start_sec: z.number().min(0), end_sec: z.number().min(0) })).min(1).describe("Absolute time ranges to zoom into, e.g. [{\"start_sec\": 10, \"end_sec\": 40}], each within the video."),
  fanout: z.number().int().min(2).max(8).describe("Child ranges per window (default 4, maximum 8; larger values are clamped and the result says so).").optional(),
  samples_per_child: z.number().int().min(1).max(12).describe("Thumbnails per child range (default 3, maximum 12; larger values are clamped and the result says so).").optional(),
  cols: z.number().int().min(1).optional(),
  context_sec: z.number().min(0).describe("Seconds of padding added to each side of every window (default 0).").optional(),
  max_total_samples: z.number().int().min(1).max(300).describe("Frame budget: windows x fanout x samples_per_child must not exceed it (default 96, maximum 300; larger values are clamped and the result says so).").optional(),
  align_to_cuts: z.boolean().optional(),
  compact_parent_sheet: z.boolean().optional(),
};
export const TrackRegionChangesInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the range in seconds."),
  end_sec: z.number().min(0).describe("End of the range in seconds; must be within the video."),
  region: z.object({
    x: z.number().min(0).max(1),
    y: z.number().min(0).max(1),
    width: z.number().positive().max(1),
    height: z.number().positive().max(1),
  }).describe("Rectangle to watch, as fractions of the frame: x, y (top-left) and width, height, with x+width <= 1 and y+height <= 1. Uses width/height, not w/h."),
  sample_fps: z.number().positive().max(10).describe("Sampling rate (default 0.25, maximum 10; larger values are clamped and the result says so). Changes shorter than 1/sample_fps can be missed.").optional(),
  frame_width: z.number().int().min(64).max(1920).describe("Decoded frame width in pixels (default 640, maximum 1920; larger values are clamped).").optional(),
  min_change_score: z.number().min(0).max(1).optional(),
  robust_sigma: z.number().min(0).optional(),
  min_separation_sec: z.number().min(0).optional(),
  max_candidates: z.number().int().min(1).max(128).describe("Most candidates returned (default 32, maximum 128; larger values are clamped). Raise min_change_score to keep only stronger ones.").optional(),
  max_total_frames: z.number().int().min(2).max(2000).describe("Frame budget for the whole range (default 600, maximum 2000; larger values are clamped). The call fails when (end_sec - start_sec) x sample_fps exceeds it.").optional(),
};

export const TrackRegionEventsInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the range in seconds."),
  end_sec: z.number().min(0).describe("End of the range in seconds; must be within the video."),
  region: z.object({
    x: z.number().min(0).max(1),
    y: z.number().min(0).max(1),
    width: z.number().positive().max(1),
    height: z.number().positive().max(1),
  }).describe("Rectangle to watch, as fractions of the frame: x, y (top-left) and width, height, with x+width <= 1 and y+height <= 1. Uses width/height, not w/h."),
  sample_fps: z.number().positive().max(10).describe("Sampling rate (default 0.25, maximum 10; larger values are clamped and the result says so). Transitions shorter than 1/sample_fps can be missed.").optional(),
  frame_width: z.number().int().min(64).max(1920).describe("Decoded frame width in pixels (default 640, maximum 1920; larger values are clamped).").optional(),
  change_threshold: z.number().min(0).max(1).optional(),
  ambiguity_margin: z.number().min(0).max(1).optional(),
  min_separation_sec: z.number().min(0).optional(),
  max_events: z.number().int().min(1).max(128).describe("Most events returned (default 32, maximum 128; larger values are clamped). Raise change_threshold to keep only stronger ones.").optional(),
  max_total_frames: z.number().int().min(2).max(2000).describe("Frame budget for the whole range (default 600, maximum 2000; larger values are clamped). The call fails when (end_sec - start_sec) x sample_fps exceeds it.").optional(),
};

export const InspectMomentsInputSchema = {
  source: SourceSchema,
  moments: z.array(z.object({
    timestamp_sec: z.number().min(0),
    label: z.string().trim().min(1).max(80).optional(),
  })).min(1).max(16).describe("Instants to inspect, e.g. [{\"timestamp_sec\": 12.5, \"label\": \"door opens\"}], at most 16, each within the video."),
  before_sec: z.number().min(0).describe("Seconds of context before each moment (default 4).").optional(),
  after_sec: z.number().min(0).describe("Seconds of context after each moment (default 4).").optional(),
  samples_per_moment: z.number().int().min(3).max(25).describe("Frames per moment, an odd number 3-25 (default 9; larger values are clamped to 25).").optional(),
  cols: z.number().int().min(1).optional(),
  max_moments: z.number().int().min(1).max(16).describe("Most moments accepted (default and maximum 16; larger values are clamped).").optional(),
  max_total_samples: z.number().int().min(1).max(400).describe("Frame budget: moments x samples_per_moment must not exceed it (default 144, maximum 400; larger values are clamped).").optional(),
};

export const InspectWindowsInputSchema = {
  source: SourceSchema,
  windows: z.array(z.object({
    start_sec: z.number().min(0),
    end_sec: z.number().min(0),
  })).min(1).max(16).describe("Time ranges to inspect, e.g. [{\"start_sec\": 10, \"end_sec\": 14}], at most 16, each within the video."),
  fps: z.number().min(0.01).max(30).describe("Sampling rate (default 4, maximum 30; larger values are clamped).").optional(),
  context_sec: z.number().min(0).describe("Seconds of padding added to each side of every window (default 0).").optional(),
  merge_overlaps: z.boolean().describe("Merge overlapping windows into one (default true).").optional(),
  max_windows: z.number().int().min(1).max(16).describe("Most windows accepted (default and maximum 16; larger values are clamped).").optional(),
  max_total_frames: z.number().int().min(1).max(500).describe("Frame budget for all windows together (default 150, maximum 500; larger values are clamped). Lower fps or shorten the windows to fit.").optional(),
};

export const InspectWindowInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the window in seconds."),
  end_sec: z.number().min(0).describe("End of the window in seconds; must be within the video."),
  fps: z.number().min(0.01).max(30).describe("Sampling rate (default 4, maximum 30; larger values are clamped). Frames = fps x duration and must stay within about 150.").optional(),
};

export const FrameAtInputSchema = {
  source: SourceSchema,
  time_sec: z.number().min(0).describe("Moment in seconds, from 0 to the media duration. Returns the frame shown at that moment."),
  crop: z.object({
    x: z.number().int().min(0),
    y: z.number().int().min(0),
    width: z.number().int().min(1),
    height: z.number().int().min(1),
  }).describe("Optional rectangle in source pixels: x and y (top-left corner, >= 0) and width and height (>= 1). It must lie inside the frame as displayed (rotation applied): x + width <= source width and y + height <= source height. Omit to get the whole frame.").optional(),
  scale: z.number().min(1).max(4).describe("Enlargement of the (cropped) frame, 1 to 4. The default 1 leaves the pixels as they are. Use 2 to 4 to read small text.").optional(),
};

export const DetectCutsInputSchema = {
  source: SourceSchema,
  threshold: z.number().min(0).max(1).describe("Cut strength cutoff 0-1 (default 0.25). Raise it to keep only stronger cuts.").optional(),
  start_sec: z.number().min(0).describe("Optional inclusive start of the cut-search range in absolute video seconds.").optional(),
  end_sec: z.number().min(0).describe("Optional exclusive end of the cut-search range in absolute video seconds.").optional(),
};

export const CheckFrameCadenceInputSchema = {
  source: SourceSchema,
};

const AudioBandSelectionSchema = z.union([
  z.enum(["sub", "low_mid", "mid", "high", "air"]),
  z.tuple([z.number().min(0), z.number().positive()]).refine((band) => band[1] > band[0] && band[1] <= 24000),
]);

export const AudioOnsetsInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Optional inclusive start in seconds; omit to analyze from the beginning.").optional(),
  end_sec: z.number().min(0).describe("Optional exclusive end in seconds; omit to analyze to the end.").optional(),
  min_strength: z.number().min(0).max(1).describe("Onset strength cutoff 0-1 (default 0.12). Raise it to keep only stronger onsets.").optional(),
  band_hz: AudioBandSelectionSchema.describe("Restrict to a named band (sub, low_mid, mid, high, air) or a [low_hz, high_hz] range.").optional(),
  compare_to: z.string().trim().min(1).describe("Path or video URL of another media file whose in-band energy is compared at each onset.").optional(),
};

export const LoudnessTimelineInputSchema = {
  source: SourceSchema,
  window_sec: z.number().min(0.01).max(600).describe("Fixed window length in seconds, 0.01-600 (default 0.25). Larger values than 600 are clamped to 600. Give this or bpm, not both.").optional(),
  bpm: z.number().min(20).max(300).describe("Use one-beat windows at this tempo, 20-300. Give this or window_sec, not both.").optional(),
  start_sec: z.number().min(0).describe("Optional inclusive start in seconds; omit to analyze from the beginning.").optional(),
  end_sec: z.number().min(0).describe("Optional exclusive end in seconds; omit to analyze to the end.").optional(),
  ignore_ranges: z.array(z.object({
    start_sec: z.number().min(0),
    end_sec: z.number().min(0),
    reason: z.string().trim().min(1).max(200),
  })).describe("Ranges with a reason (e.g. an intentional silence) that are excluded from outlier detection.").optional(),
};

export const SpectrogramWindowInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the window in seconds."),
  end_sec: z.number().min(0).describe("End of the window in seconds; the window may be at most 600 s long."),
  marks: z.array(z.object({
    time_sec: z.number().min(0).optional(),
    t: z.number().min(0).optional(),
    label: z.string(),
  }).refine((mark) => mark.time_sec !== undefined || mark.t !== undefined)).describe("Optional event markers drawn on the image, e.g. [{\"time_sec\": 12.5, \"label\": \"kick\"}].").optional(),
};

export const WatchForInputSchema = {
  source: SourceSchema,
  conditions: z.array(z.string().trim().min(1)).min(1).max(8).describe("1 to 8 distinct yes/no questions about one frame, e.g. [\"Is a person visible?\", \"Is a red rectangle visible?\"]. Name what to look for; vague questions give vague chances."),
  start_sec: z.number().min(0).describe("Start of the range in seconds (default 0).").optional(),
  end_sec: z.number().min(0).describe("End of the range in seconds (default: the end of the video).").optional(),
  interval_sec: z.number().min(1).max(600).describe("Seconds between samples (default 5, from 1 to 600). The range may hold at most 720 samples.").optional(),
  cut: z.number().positive().max(1).describe("Chance of yes (above 0, up to 1) at which a sample counts as yes (default 0.5).").optional(),
  consecutive: z.number().int().min(1).max(5).describe("Samples in a row that must count as yes before a wake (1 to 5, default 2).").optional(),
  cooldown_sec: z.number().min(0).max(3600).describe("Seconds after a wake during which the same condition wakes no more (0 to 3600, default 30).").optional(),
};

export const LabelSoundsInputSchema = {
  source: SourceSchema,
  start_sec: z.number().min(0).describe("Start of the range in seconds."),
  end_sec: z.number().min(0).describe("End of the range in seconds; at least 1 s after start_sec, and the range may hold at most 60 windows."),
  labels: z.array(z.string().trim().min(1)).max(50).describe("Sound names to choose from, e.g. [\"dog barking\", \"siren\"] (at most 50, distinct). Default: 20 common sounds.").optional(),
  window_sec: z.number().min(1).max(30).describe("Length of each labelled window in seconds (default 5, from 1 to 30). The last window may be shorter; a remainder under 1 s is not labelled.").optional(),
};

export const SyncCheckInputSchema = {
  source: SourceSchema,
  cues: z.array(z.object({
    time_sec: z.number().min(0).optional(),
    t: z.number().min(0).optional(),
    label: z.string().optional(),
    band_hz: AudioBandSelectionSchema.optional(),
  }).refine((cue) => cue.time_sec !== undefined || cue.t !== undefined)).describe("Intended sound cues to check, e.g. [{\"time_sec\": 12.5, \"label\": \"hit\"}]. Omit to compare cuts with onsets only.").optional(),
  tolerance_ms: z.number().min(0).max(5000).describe("Match window in milliseconds (default 80, maximum 5000; larger values are clamped).").optional(),
  bpm: z.number().min(20).max(300).describe("Tempo of the music, 20-300. Compares each cue with uncued moments at the same beat phase, so steady backing percussion does not hide a missing sound.").optional(),
};

export interface AgenticVideoHandlers {
  probe_video: (input: any) => Promise<any>;
  transcribe: (input: any) => Promise<any>;
  search_transcript: (input: any) => Promise<any>;
  build_visual_index: (input: any) => Promise<any>;
  search_visual: (input: any) => Promise<any>;
  count_repetitions: (input: any) => Promise<any>;
  count_occurrences: (input: any) => Promise<any>;
  contact_sheet: (input: any) => Promise<any>;
  review_video: (input: any) => Promise<any>;
  overview_chunks: (input: any) => Promise<any>;
  refine_temporal_beam: (input: any) => Promise<any>;
  track_region_changes: (input: any) => Promise<any>;
  track_region_events: (input: any) => Promise<any>;
  inspect_moments: (input: any) => Promise<any>;
  inspect_windows: (input: any) => Promise<any>;
  inspect_window: (input: any) => Promise<any>;
  frame_at: (input: any) => Promise<any>;
  detect_cuts: (input: any) => Promise<any>;
  check_frame_cadence: (input: any) => Promise<any>;
  audio_onsets: (input: any) => Promise<any>;
  loudness_timeline: (input: any) => Promise<any>;
  spectrogram_window: (input: any) => Promise<any>;
  sync_check: (input: any) => Promise<any>;
  watch_for: (input: any) => Promise<any>;
  label_sounds: (input: any) => Promise<any>;
}

export function registerAgenticVideoTools(server: McpServer, handlers: AgenticVideoHandlers): void {
  const definitions: Array<[string, Record<string, unknown>, (input: any) => Promise<any>]> = [
    ["probe_video", { description: "Read a video's duration, frame rate, resolution, codecs, and whether it has an audio track. Call this first: duration_sec bounds every start_sec/end_sec you pass to other tools, and has_audio tells you whether transcribe and the audio tools (audio_onsets, loudness_timeline, spectrogram_window, sync_check) can work. `source` is a local file path or a video URL and is the same in every tool: an x.com / Twitter post with a video, a YouTube link, another site yt-dlp supports, or a direct .mp4 link. A URL is downloaded once (up to 1080p) and cached; the result gives source_url and local_path, and while the download runs it is status=in_progress, so call again.", inputSchema: ProbeVideoInputSchema }, handlers.probe_video],
    ["transcribe", { description: "Get the full speech transcript as timestamped segments (cached after the first run). To find where a specific word or phrase is spoken, use search_transcript instead of reading it all. If the requested model is not cached, returns another cached transcript with cached_model_size and requested_model_cached=false while one background job runs; with no transcript cached, returns status=in_progress and retry_after_seconds, so repeat the same call after that delay. model_size: tiny, base or small (more accurate, slower on the CPU), or turbo (Whisper large-v3-turbo on the GPU via whisper.cpp: the most accurate and, where installed, the fastest, about 1 minute per hour of audio). Omit model_size to get turbo where it is installed, else tiny. Long audio is decoded in 60-second chunks. Lines repeated 3 or more times in a row carry `suspect` and `notice` names them: speech models often invent such lines over music or silence, though a real chant can repeat too. A line without `suspect` is not verified.", inputSchema: TranscribeInputSchema }, handlers.transcribe],
    ["search_transcript", { description: "Find when a word or phrase is SPOKEN. Default mode=hybrid lists exact keyword matches first (match=exact: the phrase; match=all_terms: every content word in one segment), then passages that match by meaning (match=semantic: a 20 s window, found on this Mac with EmbeddingGemma 2; nothing is sent to the cloud), so a reworded question still finds the moment. mode=lexical gives keyword matches only. When no hit is exact or all_terms, the words were not found in the speech: semantic hits are the closest passages, not proof, so read their text before answering. If semantic search is not set up or not ready, hits are keyword-only and `note` says why. Returns up to top_k timestamped hits (default 5, maximum 50; larger values are clamped and the result says so; `note` says when more keyword matches exist). For text shown on screen use count_occurrences, and for visible things use search_visual. If ASR is needed, starts or joins background transcription and returns status=in_progress with retry_after_seconds instead of waiting; repeat the same call after that delay. A hit on a flagged line carries `suspect` (see transcribe).", inputSchema: SearchTranscriptInputSchema }, handlers.search_transcript],
    ["build_visual_index", { description: "Start or poll construction of the local visual frame index that search_visual needs. Call it once, then repeat the same call (or search_visual, which reports status=index_required or partial) until the status is complete; frames are processed in bounded chunks. count_occurrences, contact_sheet and the inspect tools do not need the index.", inputSchema: BuildVisualIndexInputSchema }, handlers.build_visual_index],
    ["search_visual", { description: "Find WHEN something visible happens: an object, person, action or scene (semantic match between the query and sampled frames). It matches meaning, not letters, so it cannot find on-screen words or numbers (names, scores, jersey or bib numbers): find those with count_occurrences `text`, which reads text in frames, one window of up to 2400 s per call. Returns up to top_k ranked frame hits (default 5, maximum 20; larger values are clamped and the result says so), merged intervals (the one holding rank 1 is at least 15 seconds long, or the whole searched range when that is shorter; a shorter one is widened evenly around its middle and kept inside the searched range; intervals that overlap it are merged into it), and a contact sheet to verify them. It ranks frames, so it cannot count: to count separate appearances of a symbol or exact on-screen text use count_occurrences, for a repeated action use count_repetitions, for spoken words use search_transcript. Build the local visual index first (build_visual_index), then repeat the same query to poll until complete. Use start_sec/end_sec to search one range at a time.", inputSchema: SearchVisualInputSchema }, handlers.search_visual],
    ["count_repetitions", { description: "Count completed cycles of a repeated visual action (jumps, claps, bounces, swings) inside one window of at most 180 seconds. Give `region` (fractions of the frame: x, y, w, h) to isolate the moving thing. Repeat the same call while status=in_progress; verify peak badges and ambiguous edges with inspect_window. For separate appearances of a specific symbol or exact on-screen text use count_occurrences; to locate one moment use search_visual. For a longer clip call once per window, cutting at pauses between cycles so none straddles an edge. The count is a rough first pass and can be far off on real footage. Before answering, count cycles by eye with inspect_window or contact_sheet over the window; if your count differs, trust your own.", inputSchema: CountRepetitionsInputSchema }, handlers.count_repetitions],
    ["count_occurrences", { description: "Count how many separate times a specific symbol/object or exact on-screen text appears. Use for questions like 'how many separate times does the yellow outlined circle appear (ignore other colours)', 'how many times does the exact word RELEASE appear (not RELEASED)', 'how many times does the green diamond cross the screen'. Give `description` (colour and shape words are enforced, so look-alikes in another colour or shape do not count) and/or `text` (whole-word match by offline OCR). Returns `count` plus one span per continuous appearance (start_sec, end_sec, representative_sec, region, travel, weak); gaps of min_gap_sec or more separate spans. Verify a span with inspect_window at its representative_sec. Not for periodic reps or cycles (use count_repetitions) and not for locating one moment (use search_visual, which ranks frames, is capped at top_k 20 and cannot count). No visual index needed. Repeat the same call while status=in_progress. Lists at most max_spans spans (default 20, maximum 100; larger values are clamped and the result says so); `count` includes any omitted. If 100 or more spans match, the description is too broad: `count` is null, `capped` is true and `spans_found` says how many matched, so narrow the description or window and rerun instead of reporting a number. A min_gap_sec above 0.3 can join separate appearances; then `gaps_merged` and `count_without_merges` are set. The count is a rough first pass: it can merge separate appearances that are split by a scene cut, split one appearance into many spans, and count appearances rather than crossings. The result also carries `contact_sheet`, one labeled image with a tile per listed span (the first 24), in span order: a zoomed frame of that span's best detection with the match boxed, labeled like '#2 29.9s'. Before answering, look at each tile and drop any span whose tile is not the asked-for thing; a span with `weak: true` (one detection, a very short span, or a low score; its label ends in *) is likelier false but is still included in `count`. The answer is the number of tiles that match, plus any spans without a tile, and it must respect `capped`, `gaps_merged` and `count_without_merges` above; if your own count by eye differs, trust yours.", inputSchema: CountOccurrencesInputSchema }, handlers.count_occurrences],
    ["contact_sheet", { description: "Get ONE grid image of evenly spaced frames, with burned timestamp badges, across the whole video or a start_sec/end_sec range: the cheapest way to see what a video contains before searching. Default 12 samples, at most 150 (larger values are clamped and the result says so). Use it to survey; use search_visual to find a specific thing, overview_chunks to survey a long video as ordered sheets, and inspect_window or inspect_windows for dense consecutive frames of a short span.", inputSchema: ContactSheetInputSchema }, handlers.contact_sheet],
    ["review_video", { description: "First look at a video in ONE call: container metadata (duration, size, fps, codecs, has_audio), a contact sheet image with burned timestamp badges (default 24 samples; `samples` up to 150, larger values are clamped and the result says so), every cut and transition from detect_cuts, and a loudness summary (average and peak level, silence gaps, the strongest band-energy outliers with their times). It runs probe_video, contact_sheet, detect_cuts and loudness_timeline for you, so start here instead of calling them one by one. Finishes in a few seconds on clips up to about 10 minutes. `loudness` is null when the video has no audio track. Use the times it returns to zoom in with inspect_window or spectrogram_window; for a different threshold, window or time range call detect_cuts or loudness_timeline directly. Any step failing fails the whole call with that step's error message.", inputSchema: ReviewVideoInputSchema }, handlers.review_video],
    ["overview_chunks", { description: "Survey a long video as consecutive contact sheets, one per chunk_sec (default 60 s), so every stretch is covered in order. At most max_chunks sheets per call (default and maximum 32): for a longer video raise chunk_sec or page through it with start_sec/end_sec; a call over budget fails and says what chunk_sec would fit. samples_per_chunk defaults to 8 (maximum 150, larger values are clamped). Use contact_sheet instead when one sheet is enough.", inputSchema: OverviewChunksInputSchema }, handlers.overview_chunks],
    ["refine_temporal_beam", { description: "Zoom into candidate time windows: splits each window into `fanout` contiguous child ranges and returns one timestamped contact sheet per child, without ranking relevance. Use it after contact_sheet or search_visual has narrowed the answer to a few windows and you want to localize inside them; use inspect_windows for dense consecutive frames instead. Budget: windows x fanout x samples_per_child must be at most max_total_samples (default 96, maximum 300); a call over budget fails and says which of the three to lower.", inputSchema: RefineTemporalBeamInputSchema }, handlers.refine_temporal_beam],
    ["track_region_changes", { description: "Watch one fixed rectangle of the frame (region: x, y, width, height as fractions 0-1) from start_sec to end_sec (both required) and rank the moments its pixels change most, as candidates to verify with inspect_moments or inspect_window. Use for a scoreboard, counter, indicator light or door. Slow continuous motion ranks no candidates: read steady_change.detected for that. Returns ranked candidates, not a count: for a bounded count of state transitions use track_region_events. Budget: about (end_sec - start_sec) x sample_fps frames, at most max_total_frames (default 600, maximum 2000); a call over budget fails and says the sample_fps that fits. This tool's region uses width/height (count_repetitions uses w/h).", inputSchema: TrackRegionChangesInputSchema }, handlers.track_region_changes],
    ["track_region_events", { description: "Count discrete state changes (transitions) inside one fixed rectangle of the frame (region: x, y, width, height as fractions 0-1) between start_sec and end_sec (both required). Returns event_count plus before/after evidence moments to verify. Use for 'how many times did the light turn on' or 'how often does the counter change'. For ranked change candidates without a count use track_region_changes; for a periodic action use count_repetitions; for a symbol appearing anywhere on screen use count_occurrences. Budget: about (end_sec - start_sec) x sample_fps frames, at most max_total_frames (default 600, maximum 2000); a call over budget fails and says the sample_fps that fits. This tool's region uses width/height (count_repetitions uses w/h).", inputSchema: TrackRegionEventsInputSchema }, handlers.track_region_events],
    ["inspect_moments", { description: "Look closely around specific instants: for each moment (timestamp_sec, optional label) returns a contact sheet of frames before, at and after it (default 4 s before and 4 s after, samples_per_moment 9). Use it to verify a hit from search_visual, search_transcript, detect_cuts, audio_onsets or track_region_changes. At most 16 moments per call and max_total_samples frames in all (default 144, maximum 400); a call over budget fails and says what to lower. For one continuous span of frames use inspect_window.", inputSchema: InspectMomentsInputSchema }, handlers.inspect_moments],
    ["inspect_windows", { description: "Dense consecutive frames (default 4 fps) for several time ranges in one call, with optional context_sec padding and merging of overlapping windows. At most 16 windows and max_total_frames frames in all (default 150, maximum 500); a call over budget fails and says to lower fps or shorten the windows. For one range use inspect_window, for a coarse survey use contact_sheet, for frames around single instants use inspect_moments.", inputSchema: InspectWindowsInputSchema }, handlers.inspect_windows],
    ["inspect_window", { description: "Dense consecutive frames of ONE short window [start_sec, end_sec] (both required) with timestamp overlays, as a contact sheet plus frame list. Default 4 fps, maximum 30 (larger values are clamped). At most about 150 frames and 600 seconds per call: frames = fps x duration, so a 10 s window allows at most 15 fps; a call over budget fails and says the fps that fits. Use it to verify a moment or a count. For a wide overview use contact_sheet; for several ranges use inspect_windows.", inputSchema: InspectWindowInputSchema }, handlers.inspect_window],
    ["frame_at", { description: "Use this when you need one exact frame at full size, or a crop to read small text; for many frames use inspect_window or contact_sheet. Returns one PNG of the frame shown at time_sec, at the source's native resolution, with no timestamp drawn on it. An optional crop (x, y, width, height in source pixels) cuts a rectangle out of that frame, and scale (1 to 4, default 1) enlarges the crop with Lanczos so small text is easier to read. frame_time_sec is the start time of the frame actually decoded: the last frame starting at or before time_sec, so on variable-frame-rate video it can be well before time_sec. A repeat call with the same source, time_sec, crop and scale returns the saved file.", inputSchema: FrameAtInputSchema }, handlers.frame_at],
    ["detect_cuts", { description: "List scene changes (timestamp_sec, score, kind) in the whole video or a start_sec/end_sec range. Hard cuts and animated changes (fades, dissolves, slides, wipes, zooms) are both returned; `kind` tells them apart: \"cut\" or \"transition\". A transition's timestamp is its middle. Scores of the two kinds are not comparable. threshold 0-1 (default 0.25) applies to hard cuts only: raise it to keep only stronger cuts. A hard cut between two similar pictures is found by a second check and listed as a cut with a score below the default threshold; any threshold above 0.25 leaves these out. A change that is over within about 4 frames (a very fast wipe) is listed as a cut, and a few similar-picture cuts still come back as a transition. Film footage with heavy motion can still give false transitions; check a doubtful one with inspect_moments. At most 500 cuts are listed; `total` is the full count and the result says how to see the rest. Use it to segment a video into shots or find edit points; use audio_onsets for sounds and sync_check to compare both.", inputSchema: DetectCutsInputSchema }, handlers.detect_cuts],
    ["check_frame_cadence", { description: "Find stretches where the picture updates less often than the file's frame rate, e.g. a 30 fps capture padded to 60 fps so every frame is doubled (judder). It decodes every frame at 160x90 grey and calls a frame a repeat when it is near-identical to the one before; a range is a stretch of at least 0.5 s where repeats fill at least a quarter of every half second while other frames still change. Each range gives start_sec, end_sec, frames, duplicate_frames and effective_fps (unique frames per second; compare with frame_rate). Runs of repeats longer than 0.5 s are holds (still shots, title cards, stalls) and are not ranges; held_frames counts them (use detect_cuts and the frozen-picture check in `python -m src.loop check` for those). It does not judge whether a range is intended: a 24 fps film in a 60 fps file is one range covering the whole file, and a slow, smooth pan can look like repeats. Look at a range with inspect_window to see what moves there. At most 500 ranges are listed; `total` is the full count. Reads the whole file: about 1 s per 700 frames of 1080p; there are no range arguments.", inputSchema: CheckFrameCadenceInputSchema }, handlers.check_frame_cadence],
    ["audio_onsets", { description: "List non-speech sound attacks (hits, claps, beats) with millisecond timestamps and five-band energy profiles, optionally restricted to a frequency band (band_hz: sub, low_mid, mid, high, air, or [low_hz, high_hz]) and compared with another source (compare_to). min_strength 0-1 (default 0.12): raise it for only strong onsets. At most 500 onsets are listed; `total` is the full count and the result says how to see the rest. Needs an audio track (probe_video has_audio). For level over time use loudness_timeline; for speech use transcribe. To name what a sound is, use label_sounds (cloud).", inputSchema: AudioOnsetsInputSchema }, handlers.audio_onsets],
    ["loudness_timeline", { description: "Measure loudness over time: RMS, peak, band energies, spectral centroid, silence gaps, and band-energy outliers classified as steps, ramps or transients, in fixed windows (window_sec 0.01-600, default 0.25) or beat-length windows (bpm 20-300; give one or the other). At most 500 windows are listed in `windows` (`total` is the full count; `outliers` and `silence_gaps` always cover the whole range); use a larger window_sec or a start_sec/end_sec range for the rest. Ranges needing over 50,000 windows are rejected with the window_sec that fits. Needs an audio track. For discrete hits use audio_onsets. `peak_dbfs` (top level, with `peak_sec`, and per window) is the loudest sample on any channel, as ffmpeg reports sample peak; `rms_dbfs` and `band_db` are measured on the mono mix (L+R)/1.414, so on centred stereo they read up to 3 dB above a per-channel meter.", inputSchema: LoudnessTimelineInputSchema }, handlers.loudness_timeline],
    ["spectrogram_window", { description: "Render a log-frequency dB spectrogram PNG for [start_sec, end_sec] (both required, at most 600 s) with absolute timecode badges and optional event marks (time_sec + label). Use a short window (a few seconds) to look at one sound; find candidate times with audio_onsets or loudness_timeline first. Needs an audio track.", inputSchema: SpectrogramWindowInputSchema }, handlers.spectrogram_window],
    ["sync_check", { description: "Check audio/video sync: compares visual cuts with measured audio onsets, and optionally with intended sound cues (time_sec, label, optional band_hz), within tolerance_ms (0-5000, default 80). Returns per-cut and per-cue matches and unmatched strong onsets (each list shows at most 500 items; total_cuts and total_cues are the full counts). Needs an audio track. To look at the two signals separately use detect_cuts and audio_onsets.", inputSchema: SyncCheckInputSchema }, handlers.sync_check],
    ["watch_for", { description: "Watch a stretch of a video for yes/no conditions with a cloud model (Clef on Cloudflare Workers AI) and return the moments each one started to hold. One frame every interval_sec (default 5 s, 1 to 600) from [start_sec, end_sec] is asked every condition (1 to 8 yes/no questions about ONE frame, e.g. \"Is a person visible?\"); a condition wakes at the sample where its chance of yes has been at least cut (default 0.5) on `consecutive` samples in a row (default 2), at most once per cooldown_sec (default 30) per condition. THIS SENDS VIDEO FRAMES TO CLOUDFLARE: never use it on personal recordings. The server needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or AGENTIC_VIDEO_CF_TOKEN_COMMAND, a command that prints the token) and the call fails naming what is missing, before anything is sent. Only the questions you write can wake: a condition nobody asked about is never reported, so list every event you care about. An event must last at least consecutive x interval_sec to be caught (two samples 5 s apart need about 10 s); lower interval_sec or consecutive for shorter ones, which costs more calls. Accuracy on real footage is not yet measured. At most 720 samples per call; the call answers within about 50 s even when Cloudflare is slow: a longer range, or a sample that keeps failing after its retry, returns the finished samples with complete=false and next_start_sec to continue from (the cooldown does not carry over). This is for finding when something starts to be true; to inspect what happened use inspect_window, for scene changes use detect_cuts, and for sounds use audio_onsets. Results are not cached.", inputSchema: WatchForInputSchema }, handlers.watch_for],
    ["label_sounds", { description: "Name the sounds in a stretch of a video's audio: cuts [start_sec, end_sec] (both required) into windows of window_sec (default 5 s, at most 60 windows per call) and returns, per window, the chance of each label plus none_chance, from a cloud model (Clef-omni on Cloudflare Workers AI). THIS SENDS THE AUDIO TO CLOUDFLARE: never use it on personal recordings. The server needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or AGENTIC_VIDEO_CF_TOKEN_COMMAND, a command that prints the token) and the call fails with a message naming what is missing. `labels` (default: 20 common sounds such as dog barking, siren, glass breaking, church bells) is the closed list to choose from, so a sound outside the list is forced onto the nearest label or none: list every sound that may be present. Steady musical chords can be named church bells, so treat top_chance under 0.5 as weak and check a hit with spectrogram_window. top_label is null when none_chance beats every label. It does not search: find candidate times with audio_onsets or loudness_timeline first, then label those ranges. For speech use transcribe or search_transcript. Needs an audio track. No window starts after about 40 s and the call answers within about 50 s: a longer range returns the windows finished in time with complete=false and next_start_sec to continue from.", inputSchema: LabelSoundsInputSchema }, handlers.label_sounds],
  ];

  for (const [name, definition, handler] of definitions) {
    server.registerTool(name, definition as any, async (args) => {
      const data = await handler(args);
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
        structuredContent: data,
      };
    });
  }
}
