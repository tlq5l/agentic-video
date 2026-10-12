/**
 * Canonical six-tool adapter. The JSON contract in mcp_schemas.json is the
 * language-neutral source of truth; this Zod surface mirrors only the live
 * Python MCP server and intentionally does not advertise legacy tools.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export const ProbeVideoInputSchema = {
  source: z.string().trim().min(1),
};

export const TranscribeInputSchema = {
  source: z.string().trim().min(1),
  model_size: z.enum(["tiny", "base", "small"]).optional(),
};

export const SearchTranscriptInputSchema = {
  source: z.string().trim().min(1),
  query: z.string().trim().min(1),
  top_k: z.number().int().min(1).max(50).optional(),
};

export const ContactSheetInputSchema = {
  source: z.string().trim().min(1),
  samples: z.number().int().min(1).max(150).optional(),
  cols: z.number().int().min(1).optional(),
};

export const InspectWindowInputSchema = {
  source: z.string().trim().min(1),
  start_sec: z.number().min(0),
  end_sec: z.number().min(0),
  fps: z.number().min(0.01).max(30).optional(),
};

export const DetectCutsInputSchema = {
  source: z.string().trim().min(1),
  threshold: z.number().min(0).max(1).optional(),
};

export interface AgenticVideoHandlers {
  probe_video: (input: any) => Promise<any>;
  transcribe: (input: any) => Promise<any>;
  search_transcript: (input: any) => Promise<any>;
  contact_sheet: (input: any) => Promise<any>;
  inspect_window: (input: any) => Promise<any>;
  detect_cuts: (input: any) => Promise<any>;
}

export function registerAgenticVideoTools(server: McpServer, handlers: AgenticVideoHandlers): void {
  const tools: Array<[string, Record<string, unknown>, (input: any) => Promise<any>]> = [
    ["probe_video", { description: "Extract container metadata, duration, fps, resolution, codecs, and audio presence.", inputSchema: ProbeVideoInputSchema }, handlers.probe_video],
    ["transcribe", { description: "Transcribe speech audio into timestamped segments using faster-whisper or sidecar subtitles.", inputSchema: TranscribeInputSchema }, handlers.transcribe],
    ["search_transcript", { description: "Search timestamped transcript for query terms, returning matching segments and timecodes.", inputSchema: SearchTranscriptInputSchema }, handlers.search_transcript],
    ["contact_sheet", { description: "Generate a visual overview contact sheet with burned timestamp badges.", inputSchema: ContactSheetInputSchema }, handlers.contact_sheet],
    ["inspect_window", { description: "Extract dense frames in [start_sec, end_sec] with timestamp overlay.", inputSchema: InspectWindowInputSchema }, handlers.inspect_window],
    ["detect_cuts", { description: "Detect visual scene cuts and transitions using FFmpeg packet analysis.", inputSchema: DetectCutsInputSchema }, handlers.detect_cuts],
  ];
  for (const [name, definition, handler] of tools) {
    server.registerTool(name, definition as any, async (args) => {
      const data = await handler(args);
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
        structuredContent: data,
      };
    });
  }
}

/* The previous seven-tool implementation is retained below as a recoverable
 * historical reference only; it is not part of the compiled/public surface.
 */
/*
 * MCP Server Tool Registration and Definitions for Agentic Video.
 *
 * Implements the standard Model Context Protocol (MCP) tool contract using
 * `@modelcontextprotocol/sdk` and `zod`.
 *
 * Can be run standalone via stdio (for Claude, Cursor, Grok, etc.) or
 * connected directly as an Executor integration via `@executor-js/plugin-mcp`.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

/**
 * Zod Schemas for Tool Inputs
 */
export const ProbeVideoInputSchema = {
  source: z.string().trim().min(1).describe("Local filesystem path or accessible URI to video file."),
};

export const TranscribeInputSchema = {
  source: z.string().trim().min(1).describe("Video or audio file path."),
  options: z
    .object({
      model_size: z
        .enum(["tiny", "base", "small", "medium", "large-v3"])
        .default("base")
        .describe("Whisper model size for transcription accuracy vs speed tradeoff."),
      language: z
        .string()
        .optional()
        .describe("ISO 639-1 language code (e.g. 'en', 'es') or 'auto' for auto-detection."),
      vad_filter: z
        .boolean()
        .default(true)
        .describe("Apply Voice Activity Detection filter to strip silence and noise."),
      word_timestamps: z
        .boolean()
        .default(false)
        .describe("Include start and end timestamps for each individual word."),
      time_range: z
        .object({
          start: z.number().min(0).optional().describe("Start time offset in seconds."),
          end: z.number().min(0).optional().describe("End time offset in seconds."),
        })
        .optional()
        .describe("Optional temporal window restriction."),
    })
    .optional(),
};

export const SearchTranscriptInputSchema = {
  source: z.string().trim().min(1).describe("Video source path."),
  query: z.string().trim().min(1).describe("Natural language query or keywords to search in transcript."),
  top_k: z.number().int().min(1).max(50).default(5).describe("Maximum number of matched segments to return."),
  time_range: z
    .object({
      start: z.number().min(0).optional(),
      end: z.number().min(0).optional(),
    })
    .optional()
    .describe("Optional bounding temporal window."),
  search_mode: z
    .enum(["hybrid", "semantic", "lexical"])
    .default("hybrid")
    .describe("Retrieval algorithm: 'hybrid' (vector + BM25), 'semantic' (embeddings), or 'lexical' (BM25/fuzzy)."),
};

export const SampleFramesInputSchema = {
  source: z.string().trim().min(1).describe("Video source path."),
  start: z.number().min(0).default(0).describe("Start timestamp in seconds."),
  end: z.number().min(0).optional().describe("End timestamp in seconds (defaults to video duration)."),
  fps: z.number().positive().max(30).optional().describe("Sample rate in frames per second (e.g. 0.5 = 1 frame every 2s)."),
  count: z.number().int().positive().max(500).optional().describe("Exact count of evenly spaced frames across window (overrides fps)."),
  width: z.number().int().positive().default(720).describe("Thumbnail width in pixels preserving aspect ratio."),
  quality: z.number().int().min(1).max(100).default(85).describe("JPEG compression quality (1-100)."),
  output_format: z
    .enum(["path", "base64", "tool_file"])
    .default("path")
    .describe("Payload format: disk file paths ('path'), base64 strings ('base64'), or Executor ToolFile ('tool_file')."),
};

export const InspectWindowInputSchema = {
  source: z.string().trim().min(1).describe("Video source path."),
  start: z.number().min(0).describe("Window start timestamp in seconds."),
  end: z.number().min(0).describe("Window end timestamp in seconds (max recommended: 30s)."),
  fps: z.number().positive().max(10).default(2.0).describe("Dense sampling frame rate (default: 2 fps)."),
  width: z.number().int().positive().default(720).describe("Frame image width in pixels."),
  include_motion: z.boolean().default(false).describe("Compute optical flow / motion delta score per frame."),
  output_format: z.enum(["path", "base64", "tool_file"]).default("path"),
};

export const ContactSheetInputSchema = {
  source: z.string().trim().min(1).describe("Video source path."),
  start: z.number().min(0).default(0).describe("Start timestamp in seconds."),
  end: z.number().min(0).optional().describe("End timestamp in seconds."),
  samples: z.number().int().min(4).max(64).default(16).describe("Number of thumbnail tiles in contact sheet grid."),
  cols: z.number().int().min(2).max(8).default(4).describe("Number of grid columns."),
  tile_width: z.number().int().positive().default(320).describe("Pixel width of each thumbnail tile."),
  show_timestamps: z.boolean().default(true).describe("Burn timestamp badges onto thumbnail tiles."),
  output_format: z.enum(["path", "base64", "tool_file"]).default("path"),
};

export const GetClipInputSchema = {
  source: z.string().trim().min(1).describe("Video source path."),
  start: z.number().min(0).describe("Clip start timestamp in seconds."),
  end: z.number().min(0).describe("Clip end timestamp in seconds."),
  re_encode: z
    .boolean()
    .default(false)
    .describe("If false, fast stream copy on keyframes; if true, re-encodes for frame-exact trimming."),
  output_format: z.enum(["mp4", "webm"]).default("mp4").describe("Container format for extracted subclip."),
};

/**
 * Tool Handler Interface
 */
export interface AgenticVideoHandlers {
  probe_video: (input: z.infer<z.ZodObject<typeof ProbeVideoInputSchema>>) => Promise<any>;
  transcribe: (input: z.infer<z.ZodObject<typeof TranscribeInputSchema>>) => Promise<any>;
  search_transcript: (input: z.infer<z.ZodObject<typeof SearchTranscriptInputSchema>>) => Promise<any>;
  sample_frames: (input: z.infer<z.ZodObject<typeof SampleFramesInputSchema>>) => Promise<any>;
  inspect_window: (input: z.infer<z.ZodObject<typeof InspectWindowInputSchema>>) => Promise<any>;
  contact_sheet: (input: z.infer<z.ZodObject<typeof ContactSheetInputSchema>>) => Promise<any>;
  get_clip: (input: z.infer<z.ZodObject<typeof GetClipInputSchema>>) => Promise<any>;
}

/**
 * Register all agentic_video tools on an MCP Server instance.
 */
export function registerAgenticVideoTools(server: McpServer, handlers: AgenticVideoHandlers): void {
  server.registerTool(
    "probe_video",
    {
      description:
        "Extract container and stream metadata including duration, resolution, frame rate, codecs, and audio channel presence.",
      inputSchema: ProbeVideoInputSchema,
    },
    async (args) => {
      const data = await handlers.probe_video(args);
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "transcribe",
    {
      description:
        "Transcribe spoken audio from the video into time-aligned speech segments with start/end timestamps.",
      inputSchema: TranscribeInputSchema,
    },
    async (args) => {
      const data = await handlers.transcribe(args);
      return {
        content: [
          {
            type: "text",
            text: `Transcribed ${data.segments.length} segments (${data.duration.toFixed(1)}s, lang: ${data.language}).`,
          },
        ],
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "search_transcript",
    {
      description:
        "Search video speech transcript using lexical/semantic retrieval to pinpoint timestamps of interest.",
      inputSchema: SearchTranscriptInputSchema,
    },
    async (args) => {
      const data = await handlers.search_transcript(args);
      return {
        content: [
          {
            type: "text",
            text: `Found ${data.matches.length} matching segment(s) for query: "${data.query}".`,
          },
        ],
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "sample_frames",
    {
      description:
        "Sample video frames at uniform intervals or target count, returning file paths with exact timestamps and dimensions.",
      inputSchema: SampleFramesInputSchema,
    },
    async (args) => {
      const data = await handlers.sample_frames(args);
      const content: any[] = [
        {
          type: "text",
          text: `Sampled ${data.frames.length} frames from ${args.source}.`,
        },
      ];

      // If base64 is present or tool_file format, attach image content block
      for (const frame of data.frames) {
        if (frame.base64_data) {
          content.push({
            type: "image",
            data: frame.base64_data,
            mimeType: "image/jpeg",
          });
        }
      }

      return {
        content,
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "inspect_window",
    {
      description:
        "Densely sample frames within a narrow temporal window (e.g. 1-15s) for fine-grained action or visual inspection.",
      inputSchema: InspectWindowInputSchema,
    },
    async (args) => {
      const data = await handlers.inspect_window(args);
      return {
        content: [
          {
            type: "text",
            text: `Inspected window [${data.window_start.toFixed(2)}s - ${data.window_end.toFixed(2)}s]: ${data.frames.length} frames extracted.`,
          },
        ],
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "contact_sheet",
    {
      description:
        "Render a composite grid image (contact sheet) with burned-in timestamps for rapid visual timeline scanning.",
      inputSchema: ContactSheetInputSchema,
    },
    async (args) => {
      const data = await handlers.contact_sheet(args);
      const content: any[] = [
        {
          type: "text",
          text: `Rendered ${data.tile_count}-tile contact sheet (${data.cols}x${data.rows}) saved to: ${data.grid_image_path}`,
        },
      ];
      if (data.base64_data) {
        content.push({
          type: "image",
          data: data.base64_data,
          mimeType: "image/jpeg",
        });
      }
      return {
        content,
        structuredContent: data,
      };
    }
  );

  server.registerTool(
    "get_clip",
    {
      description: "Extract a temporal video subclip between start and end timestamps.",
      inputSchema: GetClipInputSchema,
    },
    async (args) => {
      const data = await handlers.get_clip(args);
      return {
        content: [
          {
            type: "text",
            text: `Extracted clip (${data.duration.toFixed(2)}s) saved to: ${data.clip_path}`,
          },
        ],
        structuredContent: data,
      };
    }
  );
}
*/
