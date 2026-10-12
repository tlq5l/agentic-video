"""Fast timestamped transcript search."""
import re
import unicodedata
from typing import List, Optional, Union, Dict, Any
from src.core.types import SearchResult
from src.transcript import asr


_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}


def _content_tokens(text: str) -> list[str]:
    """Normalize searchable words while preserving phrase order.

    Search is deliberately lexical: it supports punctuation, Unicode case,
    and possessives, but never turns a partial OR hit into positive evidence.
    Semantic paraphrase remains a separate, explicitly unmeasured capability.
    """
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    raw_tokens = re.findall(r"[\w]+(?:['’][\w]+)?", normalized, flags=re.UNICODE)
    tokens = []
    for token in raw_tokens:
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if token and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _contains_contiguous(tokens: list[str], phrase: list[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    return any(tokens[i : i + len(phrase)] == phrase for i in range(len(tokens) - len(phrase) + 1))

def search_transcript(
    source_path: str,
    query: str,
    top_k: int = 5,
    time_range: Optional[tuple[float, float]] = None
) -> Union[List[SearchResult], Dict[str, Any]]:
    """Search transcript for phrase/all-term lexical matches.

    Multi-word queries require either the normalized phrase or every content
    term to occur in the same segment.  This avoids the previous OR behavior,
    which could return a convincing but unrelated segment for a negative query.
    """
    transcript = asr.request_transcription(source_path, asr.default_model_size())
    if "segments" not in transcript:
        # Preserve the transcription status payload so MCP callers can retry
        # instead of waiting inside the search call for ASR to finish.
        return transcript
    segments = transcript["segments"]
    status_fields = {
        key: transcript[key]
        for key in (
            "requested_model_size", "cached_model_size", "requested_model_cached",
            "transcription_status", "retry_after_seconds", "transcription_message",
        )
        if key in transcript
    }
    if not segments:
        return {"hits": [], "total": 0, **status_fields} if status_fields else []

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return {"hits": [], "total": 0, **status_fields} if status_fields else []
    
    matches = []
    t_start, t_end = time_range if time_range else (-1.0, 999999.0)
    
    for seg in segments:
        start_sec = float(seg["start_sec"])
        end_sec = float(seg["end_sec"])
        if end_sec < t_start or start_sec > t_end:
            continue

        text = seg["text"]
        segment_tokens = _content_tokens(text)
        if not segment_tokens:
            continue
        phrase_match = _contains_contiguous(segment_tokens, query_tokens)
        all_terms_match = all(term in segment_tokens for term in query_tokens)
        if phrase_match or all_terms_match:
            # Exact normalized phrases rank above order-independent all-term
            # matches; both are grounded in one transcript segment.
            score = 1.0 if phrase_match else 0.8
            matches.append(
                SearchResult(
                    segment_id=seg["id"],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    score=round(score, 3),
                    start_timecode=seg["start_timecode"],
                    end_timecode=seg["end_timecode"],
                    context=text,
                    suspect=seg.get("suspect"),
                )
            )
            
    matches.sort(key=lambda m: m.score, reverse=True)
    matches = matches[:top_k]
    if status_fields:
        return {
            "hits": [hit.to_dict() for hit in matches],
            "total": len(matches),
            **status_fields,
        }
    return matches
