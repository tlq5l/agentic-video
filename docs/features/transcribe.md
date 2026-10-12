# Transcribe

How a user reaches it: the `transcribe` and `search_transcript` tools.

Code: `src/transcript/asr.py` (`transcribe_video`, `request_transcription`, `find_cached_transcript`, `_cached_result`, and the worker script inside `transcribe_video` that runs whisper.cpp for turbo or faster-whisper for tiny, base and small), `src/transcript/repeats.py` (`flag_repeated_segments`), `src/transcript/search.py` (`search_transcript`), `src/core/types.py` (`TranscriptSegment`, `TranscriptResult`, `SearchResult`).

How to check it works: `pytest tests/test_transcript_repeats.py tests/test_transcript_suspect.py tests/test_transcript_flag_wiring.py tests/test_asr_gate.py`. Live: run `transcribe` on a music video whose intro has no singing; the invented intro lines come back with `suspect`, and `notice` includes "Flagged N lines as suspect".

What depends on it: `search_transcript` (same transcript; passes `suspect` through to hits), `src/adapters/model_adapter.py` (rebuilds `SearchResult(**hit)`), and the background transcription job started from `src/executor/mcp_server.py`.

Gotchas:
- Turbo can invent lines over music or silence. Seen on a music video: "I don't know what to do." 14 times in a row over a 29 s intro, and "Thank you." over a quiet outro. Model confidence does not tell them apart (0.76 and 0.99, against 0.85 for real lyrics).
- A line that comes 3 or more times in a row (after lowercasing and dropping punctuation) gets `suspect` on every copy, and `notice` names the first run. A real chant gets flagged too; a lone invented line does not. A line without `suspect` is not verified.
- Flags are added each time a transcript is returned and are never stored: the cache keeps the unflagged transcript and its base notice, so transcripts cached before flags existed get them without re-running ASR. Subtitle sidecars are never flagged.
- whisper.cpp's voice check (Silero VAD, `--vad`) is off on purpose: on a music video it dropped 79-92% of the audio, including every sung line. The faster-whisper path (tiny, base, small) uses its own voice filter (`vad_filter=True`); it is untested on music and may drop sung lines too.
- A `.srt` or `.vtt` sidecar next to the video replaces ASR completely.
