"""Fast timestamped transcript search: keyword matches, optionally fused with semantic matches."""
import re
import unicodedata
from typing import List, Optional, Union, Dict, Any
from src.core.types import SearchResult, format_timestamp
from src.media.probe import probe_video
from src.transcript import asr, semantic

RRF_K = 60
NEAR_SEC = 15.0
GAP_SEC = 20.0


_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}


def _content_tokens(text: str) -> list[str]:
    """Normalize searchable words while preserving phrase order.

    The keyword search supports punctuation, Unicode case, and possessives, but
    never turns a partial OR hit into positive evidence. Matches by meaning come
    only from the separate semantic half (`semantic.py`) and are labelled as such.
    """
    raw_tokens = _words(text)
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

def _words(text: str) -> list[str]:
    """Every word of the text, stop words and possessives kept."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return re.findall(r"[\w]+(?:['’][\w]+)?", normalized)


def _is_exact(query: str, text: str) -> bool:
    """The whole query as a word run in the text, or a single query word that is not a stop word."""
    query_words, text_words = _words(query), _words(text)
    if not query_words:
        return False
    if len(query_words) == 1:
        return query_words[0] not in _STOPWORDS and query_words[0] in text_words
    return _contains_contiguous(text_words, query_words)


def _lexical_matches(segments: list, query_tokens: list[str], t_start: float, t_end: float) -> List[SearchResult]:
    """Every segment holding the query phrase or all its content words, exact phrases first."""
    matches = []
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
    return matches


def _semantic_result(chunk: dict, last_end: float) -> SearchResult:
    end = min(chunk["end"], last_end)
    return SearchResult(
        segment_id=chunk["segment_id"], start_sec=chunk["start"], end_sec=end, text=chunk["text"],
        score=round(max(0.0, min(1.0, chunk["score"])), 3),
        start_timecode=format_timestamp(chunk["start"]), end_timecode=format_timestamp(end),
        context=chunk["text"], suspect=chunk["suspect"], match="semantic",
    )


def _fuse(lexical: List[SearchResult], chunks: List[dict], query: str, last_end: float, top_k: int) -> List[SearchResult]:
    """Exact keyword hits first, then keyword and semantic lists merged by reciprocal-rank fusion.

    Each candidate scores the sum of 1/(60 + rank) over the lists with a hit within 15 s of it
    (the best rank in that list); places closer than 20 s to one already chosen are dropped.
    A semantic pick below MIN_SCORE takes its place in the fusion but is not returned.
    """
    keyword = [(h.start_sec, h.end_sec, h) for h in lexical[:top_k]]
    related = [(c["start"], c["end"], _semantic_result(c, last_end), c["score"]) for c in chunks]
    out = [hit for _, _, hit in keyword if _is_exact(query, hit.text)]
    for hit in out:
        hit.match = "exact"

    def centre(item):
        return (item[0] + item[1]) / 2

    chosen = [centre((h.start_sec, h.end_sec)) for h in out]
    scored = []
    for source, items in ((0, keyword), (1, related)):
        for item in items:
            score = 0.0
            for other in (keyword, related):
                rank = next((r for r, o in enumerate(other) if abs(centre(o) - centre(item)) <= NEAR_SEC), None)
                if rank is not None:
                    score += 1.0 / (RRF_K + rank + 1)
            scored.append((-score, source, item[0], item))
    scored.sort(key=lambda row: row[:3])
    selected = len(out)
    for _, source, _, item in scored:
        if selected >= top_k:
            break
        if all(abs(centre(item) - c) >= GAP_SEC for c in chosen):
            hit = item[2]
            if source == 0:
                hit.match = "all_terms"
            if source == 0 or item[3] >= semantic.MIN_SCORE:
                out.append(hit)
            chosen.append(centre(item))
            selected += 1
    return out


def search_transcript(
    source_path: str,
    query: str,
    top_k: int = 5,
    time_range: Optional[tuple[float, float]] = None,
    hybrid: bool = False,
) -> Union[List[SearchResult], Dict[str, Any]]:
    """Search the transcript.

    Lexical (default): multi-word queries need either the normalized phrase or every content
    term in the same segment. This avoids the previous OR behavior, which could return a
    convincing but unrelated segment for a negative query. Returns a list of hits.

    hybrid=True: exact keyword hits first (match="exact"), then keyword hits and EmbeddingGemma 2
    matches over 20 s chunks fused by rank (match="all_terms" / "semantic"). Returns a dict with
    `hits`, `total` and a `note`; when semantic search cannot run, the hits are the keyword hits
    and the note says why.
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
    if not segments or not query.strip():
        return {"hits": [], "total": 0, **status_fields} if status_fields or hybrid else []

    t_start, t_end = time_range if time_range else (-1.0, 999999.0)
    query_tokens = _content_tokens(query)
    matches = _lexical_matches(segments, query_tokens, t_start, t_end) if query_tokens else []
    # Whole-phrase hits first (stable: score order kept within each group). This runs before any top_k cut,
    # so a later exact phrase is not dropped behind earlier partial matches of the same words.
    matches.sort(key=lambda m: not _is_exact(query, m.text))
    if not hybrid:
        matches = matches[:top_k]
        if status_fields:
            return {"hits": [hit.to_dict() for hit in matches], "total": len(matches), **status_fields}
        return matches

    notes = []
    try:
        chunks = semantic.search(
            probe_video(source_path, allow_audio_only=True).fingerprint, segments, query, top_k, (t_start, t_end))
        hits = _fuse(matches, chunks, query, max(float(s["end_sec"]) for s in segments), top_k)
        if not any(hit.match in ("exact", "all_terms") for hit in hits):
            notes.append("No keyword match: the words were not found in the speech." + (
                " match=semantic hits are the closest passages by meaning, not proof the words were said; "
                "read their text first." if hits else " No passage is close enough by meaning either."))
    except semantic.SemanticUnavailable as exc:
        hits = matches[:top_k]
        for hit in hits:
            hit.match = "exact" if _is_exact(query, hit.text) else "all_terms"
        notes.append(f"Semantic search is off ({exc}); keyword matches only.")
    if len(matches) > top_k:
        notes.append(f"More keyword matches exist beyond top_k={top_k}; raise top_k (maximum 50) or make the query more specific.")
    data: Dict[str, Any] = {"hits": [hit.to_dict() for hit in hits], "total": len(hits), **status_fields}
    if notes:
        data["note"] = " ".join(notes)
    return data
