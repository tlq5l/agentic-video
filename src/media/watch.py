"""Watch a video for yes/no conditions with Cloudflare Workers AI (Clef).

One frame every ``interval_sec`` is cut out with FFmpeg and sent with every
condition as a yes/no question; Clef returns a chance of "yes" for each. A
condition wakes the caller when its chance is at or above ``cut`` on
``consecutive`` samples in a row, and at most once per ``cooldown_sec``. The
frames leave the machine, so personal recordings must not be passed in. No
model is loaded locally and nothing stays running: a few short FFmpeg runs and
HTTPS requests, 4 at a time, and the temporary frames are deleted.

Command line: ``python -m src.media.watch VIDEO --when "Is a person visible?" [--when ...]``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.cache import MediaCache
from src.core.config import Config
from src.core.errors import AgenticVideoError, ExtractionBoundsError
from src.media.cloudflare import CloudCallError, api_base, credentials, forget_command_token, hide_token
from src.media.frames import extract_single_frame
from src.media.probe import probe_video

MODEL = "clef"
DEFAULT_INTERVAL_SEC = 5.0
MIN_INTERVAL_SEC = 1.0
MAX_INTERVAL_SEC = 600.0
DEFAULT_CUT = 0.5
DEFAULT_CONSECUTIVE = 2
MAX_CONSECUTIVE = 5
DEFAULT_COOLDOWN_SEC = 30.0
MAX_COOLDOWN_SEC = 3600.0
MAX_CONDITIONS = 8
MAX_SAMPLES = 720  # one hour at the default interval
FRAME_WIDTH = 640
PARALLEL_CALLS = 4
REQUEST_TIMEOUT_SEC = 20.0
# Executor gives up on a tool call after 60 s. After DEFAULT_DEADLINE_SEC no new sample starts; requests already
# sent are cut off at deadline + REQUEST_GRACE_SEC, so a call answers within about 50 s even when Cloudflare is slow.
DEFAULT_DEADLINE_SEC = 40.0
REQUEST_GRACE_SEC = 8.0
MIN_REQUEST_SEC = 1.0  # a request is not started with less time left than this

def _credentials() -> Tuple[str, str]:
    return credentials("watch_for", "video frames")


def _ask(
    account: str, token: str, jpeg: bytes, questions: Sequence[str], cutoff: Optional[float] = None,
) -> Tuple[List[float], int]:
    """One Clef call: the chance of yes for each question about one frame, and the input tokens used.

    `cutoff` is a time.monotonic() value: no request waits past it, and the retry is skipped when too little time is left.
    """
    base = api_base()
    body = json.dumps({
        "model": MODEL,
        "state": "",
        "images": ["data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")],
        "questions": {f"q{i}": {"type": "noul", "instructions": text} for i, text in enumerate(questions)},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/accounts/{account}/ai/run/@cf/cloudflare/{MODEL}", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    failure = "no reply"
    for attempt in range(2):
        timeout = REQUEST_TIMEOUT_SEC if cutoff is None else min(REQUEST_TIMEOUT_SEC, cutoff - time.monotonic())
        if timeout < MIN_REQUEST_SEC:
            failure = "out of time before the call's limit" if attempt == 0 else f"{failure} (no time left to retry)"
            break
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                payload = json.loads(reply.read())
            answers = (payload.get("result") or {}).get("answers") or {}
            chances = [float(answers[f"q{i}"]["noul"]) for i in range(len(questions))]
            usage = (payload.get("result") or {}).get("usage") or {}
            return chances, int(usage.get("input_tokens") or 0)
        except urllib.error.HTTPError as exc:
            failure = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            if exc.code in (401, 403):
                forget_command_token()
                break
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failure = f"{type(exc).__name__}: {hide_token(str(exc), token)[:300]}"
        except (KeyError, TypeError, ValueError) as exc:
            failure = f"unexpected reply ({type(exc).__name__}): {hide_token(str(exc), token)[:200]}"
            break
        if attempt == 0:
            time.sleep(2.0 if cutoff is None else max(0.0, min(2.0, cutoff - time.monotonic())))
    raise CloudCallError(f"Cloudflare Workers AI request failed: {failure}")


def find_wakes(
    times: Sequence[float], chances: Sequence[float], cut: float, consecutive: int, cooldown_sec: float,
) -> Tuple[List[Dict[str, float]], int]:
    """Wakes for one condition. A wake is the sample that makes `consecutive` samples in a row reach `cut`;
    a wake within `cooldown_sec` of the previous wake is dropped. Returns (wakes, how many the cooldown dropped).
    `first_seen_sec` is the first sample of the unbroken run that produced the wake."""
    wakes: List[Dict[str, float]] = []
    dropped = 0
    run = 0
    last: Optional[float] = None
    for i, (t, chance) in enumerate(zip(times, chances)):
        run = run + 1 if chance >= cut else 0
        if run < consecutive:
            continue
        if last is not None and t - last < cooldown_sec:
            dropped += 1
            continue
        wakes.append({"time_sec": t, "first_seen_sec": times[i - run + 1], "chance": chance})
        last = t
    return wakes, dropped


def _check_args(
    conditions: Sequence[str], interval_sec: float, cut: float, consecutive: int, cooldown_sec: float,
) -> List[str]:
    cleaned = [c.strip() for c in conditions if isinstance(c, str) and c.strip()]
    if not cleaned or len(cleaned) != len(conditions):
        raise ExtractionBoundsError("conditions must be a list of non-empty yes/no questions about one frame")
    if len(cleaned) > MAX_CONDITIONS:
        raise ExtractionBoundsError(f"At most {MAX_CONDITIONS} conditions per call (got {len(cleaned)}); split them over two calls.")
    if len(set(cleaned)) != len(cleaned):
        raise ExtractionBoundsError("conditions must be distinct")
    if not MIN_INTERVAL_SEC <= interval_sec <= MAX_INTERVAL_SEC:
        raise ExtractionBoundsError(f"interval_sec must be from {MIN_INTERVAL_SEC:g} to {MAX_INTERVAL_SEC:g} (got {interval_sec:g})")
    if not 0.0 < cut <= 1.0:
        raise ExtractionBoundsError(f"cut must be above 0 and at most 1 (got {cut:g})")
    if not 1 <= consecutive <= MAX_CONSECUTIVE:
        raise ExtractionBoundsError(f"consecutive must be from 1 to {MAX_CONSECUTIVE} (got {consecutive})")
    if not 0.0 <= cooldown_sec <= MAX_COOLDOWN_SEC:
        raise ExtractionBoundsError(f"cooldown_sec must be from 0 to {MAX_COOLDOWN_SEC:g} (got {cooldown_sec:g})")
    return cleaned


def watch_for(
    source: str,
    conditions: Sequence[str],
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    cut: float = DEFAULT_CUT,
    consecutive: int = DEFAULT_CONSECUTIVE,
    cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
    cache: Optional[MediaCache] = None,
    deadline_sec: Optional[float] = DEFAULT_DEADLINE_SEC,
) -> Dict[str, Any]:
    """Sample the range every interval_sec and wake on conditions that hold on `consecutive` samples in a row."""
    started = time.monotonic()  # the 60 s tool limit also counts the file hash and the token command below
    questions = _check_args(conditions, interval_sec, cut, consecutive, cooldown_sec)
    cache = cache or MediaCache()
    path = Path(source).expanduser().resolve()
    duration = probe_video(str(path), cache=cache).duration_sec
    end = duration if end_sec is None else min(float(end_sec), duration)
    start = float(start_sec)
    if start < 0 or start >= end:
        raise ExtractionBoundsError(f"start_sec must be from 0 up to before end_sec (got {start:g}, end {end:g}, duration {duration:g} s)")
    last_frame_sec = max(0.0, duration - 0.1)
    times = [round(start + k * interval_sec, 3) for k in range(int((end - start) / interval_sec) + 1)]
    times = [t for t in times if t <= end and t <= last_frame_sec]
    if len(times) < consecutive:
        raise ExtractionBoundsError(
            f"The range holds {len(times)} sample(s) at interval_sec={interval_sec:g}, fewer than consecutive={consecutive}, "
            "so no condition could wake. Lower interval_sec or widen start_sec/end_sec."
        )
    if len(times) > MAX_SAMPLES:
        raise ExtractionBoundsError(
            f"{len(times)} samples is over the {MAX_SAMPLES} per call. Raise interval_sec to {(end - start) / MAX_SAMPLES:.1f} or more, "
            "or call again on a shorter start_sec/end_sec range."
        )
    account, token = _credentials()

    series: List[List[float]] = [[] for _ in questions]
    done_times: List[float] = []
    input_tokens = 0
    stop_reason = ""
    # No new sample starts after `deadline_sec`; a request already sent is cut off REQUEST_GRACE_SEC later.
    cutoff = None if deadline_sec is None else started + deadline_sec + REQUEST_GRACE_SEC

    Config.ensure_directories()
    with tempfile.TemporaryDirectory(prefix="watch_", dir=str(Config.TEMP_DIR)) as tmp:
        def sample(item: Tuple[int, float]) -> Tuple[List[float], int]:
            index, t = item
            frame = Path(tmp) / f"{index:05d}.jpg"
            extract_single_frame(path, t, output_path=frame, width=FRAME_WIDTH, burn_timestamps=False, cache=cache, threads=1)
            jpeg = frame.read_bytes()
            frame.unlink()
            return _ask(account, token, jpeg, questions, cutoff)

        # A moving window of PARALLEL_CALLS samples, taken in time order. Results are read in order so the finished
        # samples always form an unbroken run from the start; the deadline is checked before each sample starts.
        pool = ThreadPoolExecutor(PARALLEL_CALLS)
        window: "deque[Tuple[float, Future]]" = deque()
        next_index = 0
        try:
            while True:
                while not stop_reason and next_index < len(times) and len(window) < PARALLEL_CALLS:
                    late = deadline_sec is not None and time.monotonic() - started > deadline_sec
                    if late and next_index >= consecutive:  # always start enough samples for one wake to be possible
                        stop_reason = "deadline"
                        break
                    window.append((times[next_index], pool.submit(sample, (next_index, times[next_index]))))
                    next_index += 1
                if not window:
                    break
                t, future = window.popleft()
                try:
                    chances, tokens = future.result()
                except Exception as exc:
                    if len(done_times) < consecutive:
                        raise  # nothing usable yet: report the failure itself
                    stop_reason = f"error: {type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
                    for _, pending in window:
                        pending.cancel()
                    window.clear()
                    break
                done_times.append(t)
                input_tokens += tokens
                for k, chance in enumerate(chances):
                    series[k].append(chance)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # requests in flight end by the cutoff

    wakes: List[Dict[str, Any]] = []
    per_condition: List[Dict[str, Any]] = []
    for k, text in enumerate(questions):
        found, dropped = find_wakes(done_times, series[k], cut, consecutive, cooldown_sec)
        for wake in found:
            wakes.append({"time_sec": wake["time_sec"], "condition_index": k, "condition": text,
                          "first_seen_sec": wake["first_seen_sec"], "chance": round(wake["chance"], 4)})
        top = max(range(len(done_times)), key=lambda i: series[k][i])
        per_condition.append({"index": k, "condition": text, "wakes": len(found), "dropped_by_cooldown": dropped,
                              "max_chance": round(series[k][top], 4), "max_chance_sec": done_times[top]})
    wakes.sort(key=lambda w: (w["time_sec"], w["condition_index"]))

    complete = len(done_times) == len(times)
    # Resume where an unbroken run of `consecutive` samples that ended at the last sample would have begun,
    # so a run is not cut in half; with consecutive=1 nothing needs repeating.
    resume_sec = done_times[-(consecutive - 1)] if consecutive > 1 else round(done_times[-1] + interval_sec, 3)
    result: Dict[str, Any] = {
        "source_path": str(path),
        "start_sec": start,
        "scanned_to_sec": done_times[-1],
        "interval_sec": interval_sec,
        "cut": cut,
        "consecutive": consecutive,
        "cooldown_sec": cooldown_sec,
        "model": MODEL,
        "conditions": per_condition,
        "wakes": wakes,
        "samples": len(done_times),
        "complete": complete,
        "next_start_sec": None if complete else resume_sec,
        "input_tokens": input_tokens,
    }
    if not complete:
        why = ("to stay under the 60 s tool limit" if stop_reason == "deadline"
               else f"because a sample failed after its retry ({stop_reason[len('error: '):]}); the finished samples are kept")
        result["note"] = (
            f"Stopped after {len(done_times)} of {len(times)} samples {why}. Call again with "
            f"start_sec={resume_sec:g} (it starts {consecutive - 1} sample(s) back so a run of samples is not cut in half); "
            "the cooldown does not carry over between calls."
        )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a video for yes/no conditions (sends frames to Cloudflare Workers AI).")
    parser.add_argument("video")
    parser.add_argument("--when", action="append", required=True, metavar="QUESTION", help="a yes/no question about one frame; repeat for several")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--cut", type=float, default=DEFAULT_CUT)
    parser.add_argument("--consecutive", type=int, default=DEFAULT_CONSECUTIVE)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SEC)
    args = parser.parse_args(argv)
    try:
        result = watch_for(args.video, args.when, args.start, args.end, args.interval, args.cut, args.consecutive,
                           args.cooldown, deadline_sec=None)
    except AgenticVideoError as exc:
        print(f"{type(exc).__name__}: {exc.message}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
