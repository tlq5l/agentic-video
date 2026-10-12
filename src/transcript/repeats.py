"""Flag runs of identical transcript lines, which speech models often invent over music or silence."""

import re
from itertools import groupby

MIN_RUN = 3


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def _runs(segments: list) -> list:
    """Return maximal runs of 3+ consecutive segments with the same non-empty normalized text."""
    by_text = groupby(segments, key=lambda seg: _normalize(seg.text))
    runs = (list(group) for text, group in by_text if text)
    return [run for run in runs if len(run) >= MIN_RUN]


def flag_repeated_segments(segments: list) -> str:
    """Mark suspect segments in place. Return a one-line notice, or "" when nothing is flagged."""
    runs = _runs(segments)
    if not runs:
        return ""
    for run in runs:
        for seg in run:
            seg.suspect = f"same line {len(run)} times in a row; may be invented over music or silence"

    total = sum(len(run) for run in runs)
    where = "run" if len(runs) == 1 else f"first of {len(runs)} runs"
    first = runs[0]
    text = first[0].text.strip()
    if len(text) > 60:
        text = text[:57] + "..."
    return (
        f"Flagged {total} lines as suspect: one line repeated {MIN_RUN}+ times in a row "
        f'({where}: {first[0].start_timecode}-{first[-1].end_timecode} "{text}" x{len(first)}). '
        "Such repeats can be a real chant, but speech models often invent them over music "
        "or silence; check the audio before quoting or dropping them."
    )
