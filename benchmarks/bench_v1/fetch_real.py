#!/usr/bin/env python3
"""Fetch the small, pinned public-data split used by bench-v1.

Annotations are fetched into memory and answer values are written only to the
sealed answer key. This script never writes upstream annotation dumps to disk.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .common import data_root, key_path, merge_by_id, question_path, read_jsonl, write_jsonl_atomic
except ImportError:  # direct script execution
    from common import data_root, key_path, merge_by_id, question_path, read_jsonl, write_jsonl_atomic


QVH_URL = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
QVH_DATA_URL = "https://github.com/jayleicn/moment_detr/tree/main/data"
QVH_LICENSE = "CC BY-NC-SA 4.0 (annotations); source video remains under its YouTube uploader's license"
COUNTIX_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/countix.tar.gz"
COUNTIX_REPO = "https://github.com/cvdfoundation/kinetics-dataset"
COUNTIX_LICENSE = "No annotation license is stated in the Countix archive README; each YouTube video has uploader-specific rights"
EXTREME_DATASET = "min1321/extreme-when-bench"
EXTREME_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
EXTREME_CARD_URL = "https://huggingface.co/datasets/min1321/extreme-when-bench"
EXTREME_LICENSE = "CC BY 4.0 (benchmark annotations); source video remains under its YouTube uploader's license"
LVBENCH_LICENSE_URL = "https://github.com/zai-org/LVBench#license"
VIDEOMME_LICENSE_URL = "https://github.com/MME-Benchmarks/Video-MME#dataset"

QVH_QIDS = [5979, 8737, 781, 3503, 4625, 8056, 2321, 342, 6083]
QVH_NEGATIVE_QID = 2638
EXTREME_QIDS = [
    "f8DKD78BrQA_q005", "f8DKD78BrQA_q006", "f8DKD78BrQA_q011",
    "uusf1qG_uZ4_q041", "uusf1qG_uZ4_q044", "uusf1qG_uZ4_q178",
    "xECIRjlxM3U_q008", "xECIRjlxM3U_q028", "xECIRjlxM3U_q053",
]
EXTREME_NEGATIVE_VIDEO_ID = "068rdc75mHM"

COUNTIX_ITEMS = [
    ("ADtYZmfgeSM", "battle rope training", "rope-wave cycles"),
    ("eovgxTEV-sE", "jumping jacks", "jumping-jack repetitions"),
    ("QFcCRwncPmM", "bench pressing", "bench-press repetitions"),
    ("AeTkP7iuXMA", "bouncing ball (not juggling)", "ball bounces"),
    ("Bpg5dKmPXXU", "hula hooping", "complete hula-hoop cycles"),
    ("EmVbT6WRwHI", "juggling soccer ball", "soccer-ball juggles"),
    ("AFF5uioXScE", "push up", "push-ups"),
    ("E3Hfdy0v-cU", "squat", "squats"),
    ("DFzj_b5t98o", "skipping rope", "rope-skipping jumps"),
]


def _fetch_bytes(url: str, *, timeout: int = 45, retries: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "bench-v1-research/1.0"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < retries:
                # The Hugging Face datasets server can briefly return 429 or
                # 5xx while serving paginated rows. Give it a bounded backoff.
                delay = min(2 ** attempt, 12)
                if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Retry-After"):
                    try:
                        delay = min(float(exc.headers["Retry-After"]), 20)
                    except ValueError:
                        pass
                time.sleep(delay)
    raise RuntimeError(f"could not fetch public annotation source: {type(last).__name__}")


def _qvh_rows() -> dict[int, dict[str, Any]]:
    text = _fetch_bytes(QVH_URL).decode("utf-8")
    return {int(row["qid"]): row for row in (json.loads(line) for line in text.splitlines() if line.strip())}


def _countix_rows() -> dict[tuple[str, str], dict[str, Any]]:
    archive = tarfile.open(fileobj=io.BytesIO(_fetch_bytes(COUNTIX_URL)), mode="r:gz")
    member = next((name for name in archive.getnames() if name.endswith("countix_val.csv")), None)
    if member is None:
        raise RuntimeError("Countix validation annotations are missing countix_val.csv")
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError("could not read Countix validation annotations")
    text = stream.read().decode("utf-8-sig")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        selected.setdefault((row["video_id"], row["class"]), row)
    return selected


def _extreme_rows(qids: set[str]) -> dict[str, dict[str, Any]]:
    # Read the small, converted Parquet shard in memory. The paginated rows API
    # needs dozens of calls and is prone to rate limiting; this avoids saving an
    # upstream annotation dump while returning only the pinned questions.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the pinned Extreme-When annotations; install it in the local bench tools venv") from exc
    params = urllib.parse.urlencode({"dataset": EXTREME_DATASET, "config": "default", "split": "test"})
    listing = json.loads(_fetch_bytes(f"{EXTREME_PARQUET_URL}?{params}").decode("utf-8"))
    columns = ["qid", "question", "video_id", "video_duration_s", "correct_interval", "source_corpus", "youtube_url"]
    found: dict[str, dict[str, Any]] = {}
    for item in listing.get("parquet_files", []):
        data = _fetch_bytes(str(item["url"]), timeout=60)
        table = pq.read_table(pa.BufferReader(data), columns=columns)
        for row in table.to_pylist():
            if row.get("qid") in qids:
                found[str(row["qid"])] = row
        if qids <= found.keys():
            break
    missing = qids - found.keys()
    if missing:
        raise RuntimeError(f"Extreme-When annotation rows not found: {', '.join(sorted(missing))}")
    return found


def _yt_dlp() -> str:
    override = os.environ.get("YT_DLP")
    candidates = [
        Path(override).expanduser() if override else None,
        Path(shutil.which("yt-dlp") or "/missing/yt-dlp"),
        data_root() / ".tools-venv-current/bin/yt-dlp",
        data_root() / ".tools-venv/bin/yt-dlp",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    raise RuntimeError("yt-dlp is required; see benchmarks/bench_v1/README.md for the local venv setup")


def _trash(path: Path) -> None:
    trash = shutil.which("trash")
    if path.exists() and trash:
        subprocess.run([trash, str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _download_video(video_id: str, destination: Path, *, section: tuple[float, float] | None = None, max_height: int = 480) -> None:
    if destination.is_file() and destination.stat().st_size > 100_000:
        return
    stage = Path(tempfile.mkdtemp(prefix="bench-v1-ytdlp-", dir=str(data_root())))
    try:
        output = stage / "clip.%(ext)s"
        command = [
            _yt_dlp(), "--js-runtimes", "node", "--no-warnings", "--no-playlist",
            "--format", f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}]/b",
            "--merge-output-format", "mp4", "--remux-video", "mp4",
            "--output", str(output),
        ]
        if section:
            start, end = section
            if not (0 <= start < end):
                raise ValueError("video section must have increasing non-negative bounds")
            command.extend(["--download-sections", f"*{start:.3f}-{end:.3f}", "--force-keyframes-at-cuts"])
        command.append(f"https://www.youtube.com/watch?v={video_id}")
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=1800)
        if completed.returncode:
            raise RuntimeError(f"YouTube download failed for {video_id} (yt-dlp exit {completed.returncode})")
        candidates = [path for path in stage.glob("clip.*") if path.is_file() and path.stat().st_size > 1024]
        if not candidates:
            observed = [(path.name, path.stat().st_size) for path in stage.iterdir() if path.is_file()]
            detail = (completed.stderr or completed.stdout)[-500:].replace("\n", " ")
            raise RuntimeError(f"YouTube download produced no video file for {video_id}; files={observed}; detail={detail}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidates[0], destination)
    finally:
        _trash(stage)


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"annotation field {field!r} is not numeric") from exc


def _answer_key_row(
    identifier: str, truth: Any, tolerance: dict[str, Any], notes: str, source: dict[str, Any],
    *, truth_windows: list[list[float]] | None = None,
) -> dict[str, Any]:
    row = {"id": identifier, "truth": truth, "tolerance": tolerance, "notes": notes, "source": source}
    if truth_windows:
        row["truth_windows"] = truth_windows
    return row


def _probe_duration(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to validate long-video downloads")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=45,
    )
    if completed.returncode:
        raise RuntimeError(f"could not probe downloaded video {video_path.name}")
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"downloaded video {video_path.name} has no duration") from exc


def _source_video_license(source_corpus: str | None) -> str:
    if source_corpus == "LVBench":
        return "LVBench is for academic research; commercial use is prohibited; the corpus does not own source-video copyright"
    if source_corpus == "VideoMME":
        return "VideoMME is for academic research; video copyright belongs to the source owner; its README prohibits redistribution or modification without prior approval"
    return "Source video remains under its source-corpus terms and uploader-specific copyright"


def _source_video_license_url(source_corpus: str | None) -> str:
    if source_corpus == "LVBench":
        return LVBENCH_LICENSE_URL
    if source_corpus == "VideoMME":
        return VIDEOMME_LICENSE_URL
    return EXTREME_CARD_URL


def fetch_real() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = data_root()
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    qvh = _qvh_rows()
    countix = _countix_rows()
    extreme_qids = set(EXTREME_QIDS)
    extreme_qids.update({f"{EXTREME_NEGATIVE_VIDEO_ID}_q022"})
    extreme = _extreme_rows(extreme_qids)
    questions: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []

    for number, qid in enumerate(QVH_QIDS, 1):
        row = qvh[qid]
        video_id, start_text, end_text = str(row["vid"]).rsplit("_", 2)
        source_start, source_end = _float(start_text, "vid start"), _float(end_text, "vid end")
        video_path = videos / f"qvh-{video_id}-{int(source_start)}-{int(source_end)}.mp4"
        _download_video(video_id, video_path, section=(source_start, source_end), max_height=480)
        windows = row.get("relevant_windows") or []
        if not windows:
            raise RuntimeError(f"QVHighlights row {qid} has no annotated relevant window")
        truth_windows = []
        for target in windows:
            if not isinstance(target, list) or len(target) != 2:
                raise RuntimeError(f"QVHighlights row {qid} has a malformed annotated window")
            truth_windows.append([float(target[0]), float(target[1])])
        if not truth_windows:
            raise RuntimeError(f"QVHighlights row {qid} has no usable annotated window")
        identifier = f"real_moment_{number:02d}"
        prompt = f"At what time interval, in clip-local seconds, is this moment visible? {row['query']} Return [start, end]."
        questions.append({
            "id": identifier, "track": "moment", "split": "real",
            "video_path": str(video_path), "question": prompt,
            "answer_format": "interval_seconds", "cue_type": "visual_or_speech_scene",
        })
        keys.append(_answer_key_row(identifier, truth_windows[0],
            {"primary_metric": "interval_iou", "primary_iou_threshold": 0.5, "reported_iou_thresholds": [0.5, 0.7], "midpoint_tolerance_seconds": 2.0},
            "QVHighlights validation; all annotated relevant windows are accepted by interval-IoU scoring.", {
                "dataset": "QVHighlights validation", "annotation_id": qid,
                "annotation_url": QVH_DATA_URL, "annotation_license": "CC BY-NC-SA 4.0",
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_license": "YouTube uploader-specific; not asserted by the dataset",
                "video_license_url": f"https://www.youtube.com/watch?v={video_id}",
                "clip_source_interval_seconds": [source_start, source_end],
            }, truth_windows=truth_windows))

    negative_row = qvh[QVH_NEGATIVE_QID]
    neg_video_id, neg_start_text, neg_end_text = str(negative_row["vid"]).rsplit("_", 2)
    neg_start, neg_end = _float(neg_start_text, "negative vid start"), _float(neg_end_text, "negative vid end")
    neg_video = videos / f"qvh-{neg_video_id}-{int(neg_start)}-{int(neg_end)}.mp4"
    _download_video(neg_video_id, neg_video, section=(neg_start, neg_end), max_height=480)
    identifier = "real_moment_10"
    questions.append({
        "id": identifier, "track": "moment", "split": "real", "video_path": str(neg_video),
        "question": "At what time does a polar bear swim under Antarctic ice in this clip? If none is shown, answer not_found.",
        "answer_format": "number_seconds_or_not_found", "cue_type": "negative_control",
    })
    keys.append(_answer_key_row(identifier, "not_found", {"primary_seconds": 0.5, "reported_seconds": [0.2, 0.5, 1.0]},
        "Manually selected negative control on a QVHighlights home-vlog clip; overview frames showed no polar bear or ice scene.", {
            "dataset": "QVHighlights validation video", "context_annotation_id": QVH_NEGATIVE_QID,
            "annotation_url": QVH_DATA_URL, "annotation_license": "CC BY-NC-SA 4.0",
            "video_url": f"https://www.youtube.com/watch?v={neg_video_id}",
            "video_license": "YouTube uploader-specific; not asserted by the dataset",
            "video_license_url": f"https://www.youtube.com/watch?v={neg_video_id}",
            "clip_source_interval_seconds": [neg_start, neg_end],
        }))

    for number, (video_id, class_name, target_name) in enumerate(COUNTIX_ITEMS, 1):
        row = countix.get((video_id, class_name))
        if row is None:
            raise RuntimeError(f"Countix validation row not found for {video_id} / {class_name}")
        kin_start, kin_end = _float(row["kinetics_start"], "kinetics_start"), _float(row["kinetics_end"], "kinetics_end")
        rep_start, rep_end = _float(row["repetition_start"], "repetition_start"), _float(row["repetition_end"], "repetition_end")
        if rep_start < kin_start - 0.01 and rep_end <= (kin_end - kin_start) + 0.01:
            source_start, source_end = kin_start + rep_start, kin_start + rep_end
        else:
            source_start, source_end = rep_start, rep_end
        if source_end > kin_end + 0.05 or source_start < kin_start - 0.05 or source_end <= source_start:
            raise RuntimeError(f"Countix repetition span falls outside its Kinetics clip for {video_id}")
        video_path = videos / f"countix-{number:02d}-{video_id}.mp4"
        _download_video(video_id, video_path, section=(source_start, source_end), max_height=480)
        identifier = f"real_count_{number:02d}"
        questions.append({
            "id": identifier, "track": "count", "split": "real",
            "video_path": str(video_path),
            "question": f"How many {target_name} are completed in this clip? Return an integer; answer 0 if none occur.",
            "answer_format": "integer", "cue_type": "visual_action_count",
        })
        keys.append(_answer_key_row(identifier, int(row["count"]), {"exact": True, "off_by_one": 1},
            "Countix validation repetition-count label; inspected frames from the downloaded annotated span.", {
                "dataset": "Countix validation (Kinetics-700-2020)", "class": class_name,
                "video_id": video_id, "annotation_url": COUNTIX_URL, "dataset_repository": COUNTIX_REPO,
                "annotation_license": "Not stated in the Countix archive README",
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_license": "YouTube uploader-specific; not asserted by the dataset",
                "video_license_url": f"https://www.youtube.com/watch?v={video_id}",
            }))

    # Negative count uses an independently sourced Countix training clip already
    # represented in this real split; no answer-bearing field is added to questions.
    count_negative_video = videos / f"countix-01-{COUNTIX_ITEMS[0][0]}.mp4"
    identifier = "real_count_10"
    questions.append({
        "id": identifier, "track": "count", "split": "real", "video_path": str(count_negative_video),
        "question": "How many elephants walk through the exercise studio in this clip? Return 0 if none appear.",
        "answer_format": "integer", "cue_type": "negative_control",
    })
    keys.append(_answer_key_row(identifier, 0, {"exact": True, "off_by_one": 1},
        "Manually selected negative object-count control on a Countix exercise clip; inspected the entire short clip.", {
            "dataset": "Countix validation video", "context_video_id": COUNTIX_ITEMS[0][0],
            "annotation_url": COUNTIX_URL, "annotation_license": "Not stated in the Countix archive README",
            "video_url": f"https://www.youtube.com/watch?v={COUNTIX_ITEMS[0][0]}",
            "video_license": "YouTube uploader-specific; not asserted by the dataset",
            "video_license_url": f"https://www.youtube.com/watch?v={COUNTIX_ITEMS[0][0]}",
        }))

    long_video_paths: dict[str, Path] = {}
    long_video_durations: dict[str, float] = {}
    for video_id in sorted({qid.rsplit("_q", 1)[0] for qid in EXTREME_QIDS}):
        video_path = videos / f"extreme-when-{video_id}.mp4"
        _download_video(video_id, video_path, max_height=480)
        duration = _probe_duration(video_path)
        if duration < 1800:
            raise RuntimeError(f"Extreme-When source {video_id} is only {duration:.1f}s; expected at least 30 minutes")
        long_video_paths[video_id] = video_path
        long_video_durations[video_id] = duration
    if len(long_video_paths) < 3 or not any(duration >= 7200 for duration in long_video_durations.values()):
        raise RuntimeError("Extreme-When search must use three long videos and at least one 2-hour source")

    for number, qid in enumerate(EXTREME_QIDS, 1):
        row = extreme[qid]
        video_id = str(row.get("video_id") or qid.rsplit("_q", 1)[0])
        if video_id not in long_video_paths:
            raise RuntimeError(f"Extreme-When question {qid} is not linked to a selected long video")
        declared_duration = _float(row.get("video_duration_s"), "video_duration_s")
        if declared_duration < 1800 or long_video_durations[video_id] < 1800:
            raise RuntimeError(f"Extreme-When question {qid} does not point to a long source video")
        target = row.get("correct_interval")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise RuntimeError(f"Extreme-When row {qid} has no two-point correct_interval")
        identifier = f"real_search_{number:02d}"
        original_question = str(row["question"]).strip()
        prompt = f"Find the interval in this {round(declared_duration / 60)}-minute video where this happens: {original_question} Return [start, end] in seconds from the start of the video."
        questions.append({
            "id": identifier, "track": "search", "split": "real", "video_path": str(long_video_paths[video_id]),
            "question": prompt, "answer_format": "interval_seconds",
            "cue_type": "speech_cued" if qid.endswith(("_q017",)) else "visual_only",
        })
        keys.append(_answer_key_row(identifier, [float(target[0]), float(target[1])],
            {"primary_metric": "interval_iou", "primary_iou_threshold": 0.5, "reported_iou_thresholds": [0.5, 0.7], "midpoint_tolerance_seconds": 2.0},
            "Extreme-When benchmark interval; frame samples at the annotated interval were inspected.", {
                "dataset": "Extreme-When benchmark test split", "annotation_id": qid,
                "annotation_url": EXTREME_CARD_URL, "annotation_license": "CC BY 4.0",
                "source_corpus": row.get("source_corpus"),
                "source_video_duration_seconds": declared_duration,
                "downloaded_video_duration_seconds": long_video_durations[video_id],
                "video_url": str(row.get("youtube_url") or f"https://www.youtube.com/watch?v={EXTREME_NEGATIVE_VIDEO_ID}"),
                "video_license": _source_video_license(row.get("source_corpus")),
                "video_license_url": _source_video_license_url(row.get("source_corpus")),
            }))

    negative_row = extreme[f"{EXTREME_NEGATIVE_VIDEO_ID}_q022"]
    negative_video = videos / f"extreme-when-{EXTREME_NEGATIVE_VIDEO_ID}.mp4"
    if not negative_video.is_file():
        _download_video(EXTREME_NEGATIVE_VIDEO_ID, negative_video, max_height=480)
    negative_video_duration = _probe_duration(negative_video)
    identifier = "real_search_10"
    questions.append({
        "id": identifier, "track": "search", "split": "real", "video_path": str(negative_video),
        "question": "Find the moment in this 53-minute video when a polar bear swims beneath Antarctic ice. If none is shown, answer not_found.",
        "answer_format": "number_seconds_or_not_found", "cue_type": "negative_control",
    })
    keys.append(_answer_key_row(identifier, "not_found", {"primary_seconds": 2.0, "reported_seconds": [2.0, 5.0]},
        "Manually selected negative visual control on a quantum-physics documentary; overview frames showed no polar bear or Antarctic ice scene.", {
            "dataset": "Extreme-When benchmark source video", "context_annotation_id": negative_row.get("qid"),
            "annotation_url": EXTREME_CARD_URL, "annotation_license": "CC BY 4.0",
            "source_corpus": negative_row.get("source_corpus"),
            "source_video_duration_seconds": _float(negative_row.get("video_duration_s"), "negative video duration"),
            "downloaded_video_duration_seconds": negative_video_duration,
            "video_url": str(negative_row.get("youtube_url") or f"https://www.youtube.com/watch?v={EXTREME_NEGATIVE_VIDEO_ID}"),
            "video_license": _source_video_license(negative_row.get("source_corpus")),
            "video_license_url": _source_video_license_url(negative_row.get("source_corpus")),
        }))
    return questions, keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    questions, keys = fetch_real()
    q_path, k_path = question_path(), key_path()
    prior_questions = read_jsonl(q_path) if q_path.exists() else []
    prior_keys = read_jsonl(k_path) if k_path.exists() else []
    write_jsonl_atomic(q_path, merge_by_id(prior_questions, questions))
    write_jsonl_atomic(k_path, merge_by_id(prior_keys, keys), mode=0o600)
    print(f"Fetched {len(questions)} real questions and verified media files under {data_root() / 'videos'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
