// Gemini Interactions API adapter for bench-v1.
// Reads GEMINI_API_KEY from the environment at run time and never writes or prints it.
import fs from 'node:fs';
import path from 'node:path';
import { createReadStream } from 'node:fs';
import { performance } from 'node:perf_hooks';

const API = 'https://generativelanguage.googleapis.com';
let API_KEY = '';

function readKey() {
  const key = (process.env.GEMINI_API_KEY || '').trim();
  if (key) return key;
  throw new Error('GEMINI_API_KEY is not set in the environment');
}

function scrub(value) {
  const text = String(value ?? '');
  return API_KEY ? text.split(API_KEY).join('[redacted]') : text;
}

async function apiCall(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.text();
  if (!response.ok) {
    const detail = scrub(body).slice(0, 500);
    throw new Error(`Gemini API HTTP ${response.status}: ${detail}`);
  }
  return { response, body };
}

async function uploadVideo(filePath) {
  const bytes = fs.statSync(filePath).size;
  const start = await apiCall(`${API}/upload/v1beta/files`, {
    method: 'POST',
    headers: {
      'x-goog-api-key': API_KEY,
      'X-Goog-Upload-Protocol': 'resumable',
      'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': String(bytes),
      'X-Goog-Upload-Header-Content-Type': 'video/mp4',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file: { display_name: path.basename(filePath) } }),
  });
  const uploadUrl = start.response.headers.get('x-goog-upload-url');
  if (!uploadUrl) throw new Error('Gemini upload did not return a resumable upload URL');
  const finalized = await apiCall(uploadUrl, {
    method: 'POST',
    headers: {
      'Content-Length': String(bytes),
      'X-Goog-Upload-Offset': '0',
      'X-Goog-Upload-Command': 'upload, finalize',
    },
    body: createReadStream(filePath),
    duplex: 'half',
  });
  const uploaded = JSON.parse(finalized.body).file;
  if (!uploaded?.name || !uploaded?.uri) throw new Error('Gemini upload returned incomplete file metadata');
  const deadline = Date.now() + 300_000;
  let current = uploaded;
  while (Date.now() < deadline) {
    const state = await apiCall(`${API}/v1beta/${current.name}`, {
      headers: { 'x-goog-api-key': API_KEY },
    });
    current = JSON.parse(state.body);
    if (current.state === 'ACTIVE') return current;
    if (current.state === 'FAILED') throw new Error('Gemini video file processing failed');
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  throw new Error('Timed out waiting for Gemini video upload processing');
}

async function removeVideo(name) {
  await apiCall(`${API}/v1beta/${name}`, {
    method: 'DELETE',
    headers: { 'x-goog-api-key': API_KEY },
  });
}

function extractText(interaction) {
  if (typeof interaction.output_text === 'string') return interaction.output_text;
  const steps = interaction.steps || [];
  return steps
    .filter(step => step.type === 'model_output')
    .flatMap(step => step.content || [])
    .filter(part => part.type === 'text' && typeof part.text === 'string')
    .map(part => part.text)
    .join('\n');
}

function usageOf(interaction) {
  return interaction.usage || interaction.usage_metadata || interaction.usageMetadata || null;
}

function videoToolCallCounts(steps) {
  let succeeded = 0;
  let failed = 0;
  let resultsReceived = 0;
  const resultsByCallId = new Map();
  for (const step of steps || []) {
    if (String(step?.type || '').toLowerCase() === 'processing_result' && step?.call_id) {
      resultsByCallId.set(String(step.call_id), step);
    }
  }
  for (const step of steps || []) {
    const type = String(step?.type || '').toLowerCase();
    if (type !== 'processing_call' && type !== 'tool_call') continue;
    const result = type === 'processing_call' ? resultsByCallId.get(String(step?.id || '')) : null;
    if (result) resultsReceived += 1;
    const status = String(step?.status || step?.state || result?.status || result?.state || '').toLowerCase();
    const hasError = step?.error || result?.error || ['failed', 'error', 'cancelled', 'canceled', 'rejected'].includes(status);
    if (hasError || (type === 'processing_call' && !result)) failed += 1;
    else succeeded += 1;
  }
  return { succeeded, failed, resultsReceived };
}

async function interact({ model, video, prompt, mode, previousInteractionId }) {
  const input = [];
  if (video) {
    const part = { type: 'video', uri: video.uri, mime_type: video.mimeType || 'video/mp4' };
    if (mode === 'agentic') part.processing = 'agentic';
    input.push(part);
  }
  input.push({ type: 'text', text: prompt });
  const body = { model, input };
  if (previousInteractionId) body.previous_interaction_id = previousInteractionId;
  const response = await apiCall(`${API}/v1beta/interactions`, {
    method: 'POST',
    headers: { 'x-goog-api-key': API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const interaction = JSON.parse(response.body);
  return {
    id: interaction.id,
    text: extractText(interaction),
    usage: usageOf(interaction),
    steps: interaction.steps || [],
  };
}

function addUsage(target, usage) {
  if (!usage || typeof usage !== 'object') return;
  for (const [key, value] of Object.entries(usage)) {
    if (typeof value === 'number' && Number.isFinite(value)) target[key] = (target[key] || 0) + value;
  }
}

function numberFrom(object, names) {
  for (const name of names) {
    const value = object?.[name];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return 0;
}

async function main() {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  const wallStart = performance.now();
  const model = request.model || 'gemini-3.8-flash';
  const mode = request.mode || 'agentic';
  const usage = {};
  const requestUsage = [];
  const stepTypes = [];
  let file = null;
  let firstInteractionId = null;
  let answerText = '';
  let parseRetries = 0;
  let videoToolCalls = { succeeded: 0, failed: 0, resultsReceived: 0 };
  let uploadProcessingS = null;
  let answerTimeMs = 0;
  let answerWasMeasured = false;
  let cleanupTimeMs = 0;
  let resultPayload;
  async function timedInteract(options) {
    const started = performance.now();
    try {
      return await interact(options);
    } finally {
      answerTimeMs += performance.now() - started;
      answerWasMeasured = true;
    }
  }
  try {
    API_KEY = readKey();
    if (request.operation === 'repair') {
      const repaired = await timedInteract({
        model,
        prompt: request.prompt,
        mode,
        previousInteractionId: request.previous_interaction_id || undefined,
      });
      answerText = repaired.text;
      addUsage(usage, repaired.usage);
      requestUsage.push(repaired.usage);
      stepTypes.push(...repaired.steps.map(step => step.type));
    } else {
      const uploadStarted = performance.now();
      try {
        file = await uploadVideo(request.video_path);
      } finally {
        uploadProcessingS = Number(((performance.now() - uploadStarted) / 1000).toFixed(3));
      }
      const initial = await timedInteract({ model, video: file, prompt: request.prompt, mode });
      answerText = initial.text;
      firstInteractionId = initial.id;
      videoToolCalls = videoToolCallCounts(initial.steps);
      addUsage(usage, initial.usage);
      requestUsage.push(initial.usage);
      stepTypes.push(...initial.steps.map(step => step.type));

      if (request.retry_prompt) {
        const retry = await timedInteract({
          model,
          prompt: request.retry_prompt(answerText),
          mode,
          previousInteractionId: firstInteractionId,
        });
        parseRetries = 1;
        answerText = retry.text;
        addUsage(usage, retry.usage);
        requestUsage.push(retry.usage);
        stepTypes.push(...retry.steps.map(step => step.type));
      }
    }
    const billableInput = numberFrom(usage, ['total_input_tokens', 'totalInputTokens', 'input_tokens', 'inputTokens', 'prompt_tokens'])
      + numberFrom(usage, ['total_tool_use_tokens', 'tool_use_prompt_token_count', 'totalToolUseTokens']);
    const billableOutput = numberFrom(usage, ['total_output_tokens', 'totalOutputTokens', 'output_tokens', 'outputTokens', 'completion_tokens'])
      + numberFrom(usage, ['total_thought_tokens', 'thoughts_token_count', 'totalThoughtTokens']);
    // A request without numeric usage makes the totals unknown, not zero.
    // Only token-count fields count as measured; cache or other metadata numbers do not.
    const tokenFields = ['total_tokens', 'totalTokens', 'total_input_tokens', 'totalInputTokens', 'input_tokens',
      'inputTokens', 'prompt_tokens', 'total_output_tokens', 'totalOutputTokens', 'output_tokens', 'outputTokens',
      'completion_tokens', 'total_tool_use_tokens', 'tool_use_prompt_token_count', 'totalToolUseTokens',
      'total_thought_tokens', 'thoughts_token_count', 'totalThoughtTokens'];
    const usageMissingRequests = requestUsage.filter(item => !item || typeof item !== 'object'
      || !tokenFields.some(name => typeof item[name] === 'number' && Number.isFinite(item[name]))).length;
    const normalizedUsage = {
      ...usage,
      total_tokens: usageMissingRequests ? null : (usage.total_tokens ?? usage.totalTokens ?? (billableInput + billableOutput)),
      billable_tokens: usageMissingRequests ? null : billableInput + billableOutput,
      usage_missing_requests: usageMissingRequests,
      requests: requestUsage,
    };
    resultPayload = {
      status: 'ok', model, mode, answer_text: answerText, usage: normalizedUsage,
      interaction_id: firstInteractionId, step_types: stepTypes, parse_retries: parseRetries,
      video_tool_calls_succeeded: videoToolCalls.succeeded,
      video_tool_calls_failed: videoToolCalls.failed,
      video_tool_results_received: videoToolCalls.resultsReceived,
      upload_processing_s: uploadProcessingS,
    };
  } catch (error) {
    resultPayload = {
      status: 'error', model, mode, error: scrub(error?.message || error),
      video_tool_calls_succeeded: videoToolCalls.succeeded,
      video_tool_calls_failed: videoToolCalls.failed,
      video_tool_results_received: videoToolCalls.resultsReceived,
      upload_processing_s: uploadProcessingS,
    };
  } finally {
    if (file?.name) {
      const cleanupStarted = performance.now();
      try {
        await removeVideo(file.name);
      } catch (error) {
        process.stderr.write(`Gemini temporary-file cleanup failed: ${scrub(error?.message || error)}\n`);
      } finally {
        cleanupTimeMs = performance.now() - cleanupStarted;
      }
    }
  }
  resultPayload.answer_ms = answerWasMeasured ? Number(answerTimeMs.toFixed(3)) : null;
  resultPayload.cleanup_ms = Number(cleanupTimeMs.toFixed(3));
  resultPayload.wall_time_s = Number(((performance.now() - wallStart) / 1000).toFixed(3));
  process.stdout.write(JSON.stringify(resultPayload));
}

main();
