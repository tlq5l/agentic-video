from src.core.types import TranscriptSegment
from src.transcript.repeats import flag_repeated_segments

SUSPECT_14 = "same line 14 times in a row; may be invented over music or silence"


def _segs(*texts, length=2.0):
    return [
        TranscriptSegment(id=i, start_sec=i * length, end_sec=(i + 1) * length, text=text)
        for i, text in enumerate(texts)
    ]


def _unflagged(seg):
    return getattr(seg, "suspect", None) is None


def test_long_invented_run_is_flagged_with_exact_notice():
    segs = [TranscriptSegment(id=0, start_sec=0.0, end_sec=2.0, text="Intro line")]
    starts = [2.0 + 2.0 * i for i in range(14)]  # 2.0 .. 28.0
    ends = starts[1:] + [29.0]  # 13 copies are 2 s long; the last one is 1 s (28.0-29.0)
    for i, (start, end) in enumerate(zip(starts, ends)):
        segs.append(TranscriptSegment(id=1 + i, start_sec=start, end_sec=end, text="I don't know what to do."))
    segs.append(TranscriptSegment(id=15, start_sec=29.0, end_sec=32.6, text="Now you know my name"))

    notice = flag_repeated_segments(segs)

    assert notice == (
        'Flagged 14 lines as suspect: one line repeated 3+ times in a row (run: '
        '00:00:02.000-00:00:29.000 "I don\'t know what to do." x14). Such repeats can be a real chant, '
        "but speech models often invent them over music or silence; check the audio before quoting or dropping them."
    )
    for seg in segs[1:15]:
        assert seg.suspect == SUSPECT_14
    assert _unflagged(segs[0])
    assert _unflagged(segs[-1])


def test_case_and_punctuation_are_ignored():
    segs = _segs("Thank you.", "thank you", "THANK YOU!")

    notice = flag_repeated_segments(segs)

    assert notice.startswith("Flagged 3 lines as suspect")
    assert "x3)" in notice
    assert all(seg.suspect for seg in segs)


def test_two_identical_lines_are_not_flagged():
    segs = _segs("Yeah.", "Yeah.", "Something else")

    assert flag_repeated_segments(segs) == ""
    assert all(_unflagged(seg) for seg in segs)


def test_repeated_chorus_block_is_not_flagged():
    segs = _segs("A line", "B line", "A line", "B line", "A line", "B line")

    assert flag_repeated_segments(segs) == ""
    assert all(_unflagged(seg) for seg in segs)


def test_punctuation_only_or_empty_lines_are_not_flagged():
    segs = _segs("...", "♪", "...")
    empty = _segs("", "  ", "")

    assert flag_repeated_segments(segs) == ""
    assert flag_repeated_segments(empty) == ""
    assert all(_unflagged(seg) for seg in segs + empty)


def test_accents_matter():
    segs = _segs("Cảm ơn.", "cảm ơn", "Cam on")
    assert flag_repeated_segments(segs) == ""
    assert all(_unflagged(seg) for seg in segs)

    repeated = _segs("Cảm ơn các bạn", "Cảm ơn các bạn", "Cảm ơn các bạn")
    notice = flag_repeated_segments(repeated)
    assert notice.startswith("Flagged 3 lines as suspect")
    assert all(seg.suspect for seg in repeated)


def test_two_runs_report_first_run_and_flag_both():
    segs = _segs("X line", "X line", "X line", "Other", "Y line", "Y line", "Y line", "Y line")

    notice = flag_repeated_segments(segs)

    assert notice.startswith("Flagged 7 lines as suspect")
    assert "(first of 2 runs:" in notice
    assert "x3)" in notice
    assert all(seg.suspect for seg in segs[0:3])
    assert _unflagged(segs[3])
    assert all(seg.suspect for seg in segs[4:8])


def test_long_first_run_line_is_cut_in_notice():
    line = "This is a very long line that keeps going well past the sixty character limit"
    segs = _segs(line, line, line)

    notice = flag_repeated_segments(segs)

    assert f'"{line[:57]}..." x3)' in notice


def test_empty_list_returns_empty_string():
    assert flag_repeated_segments([]) == ""
