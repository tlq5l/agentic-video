"""High-level Agentic Video Engine."""
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple, Union
from src.core.types import (
    VideoMetadata,
    TranscriptResult,
    SearchResult,
    ContactSheetResult,
    InspectionWindowResult,
    EvidenceItem,
    format_timestamp
)
from src.core.config import Config
from src.core.errors import ExtractionBoundsError, SubprocessTimeoutError
from src.core.cache import MediaCache
from src.media.probe import probe_video
from src.media.frames import extract_window_frames
from src.media.sampling import sample_sparse_frames
from src.media.contact_sheet import (
    generate_video_overview,
    generate_video_overviews,
    build_contact_sheets,
    compose_contact_sheet,
    default_sheet_path,
)
from src.media.scene import detect_scenes, detect_soft_transitions, scan_scene_changes, SceneCut
from src.transcript.asr import transcribe_video, request_transcription
from src.transcript.search import search_transcript
from src.media.audio import (
    audio_onsets as measure_audio_onsets,
    loudness_timeline as measure_loudness_timeline,
    spectrogram_window as render_spectrogram_window,
    sync_check as compare_audio_video_sync,
)

# Most items (windows, scan segments) one multi-item tool call extracts side by side.
WINDOW_WORKERS = 4


def _timestamp_provenance(frame: Any) -> Dict[str, Optional[float]]:
    """Expose sampling-vs-source timing without inventing a decoded PTS."""
    requested = getattr(frame, "requested_timestamp", None)
    if requested is None:
        # ``timestamp_sec`` is the legacy/requested clock on FrameInfo and on
        # older test doubles. It is not a decoded source PTS.
        requested = getattr(frame, "timestamp_sec", None)
    decoded = getattr(frame, "decoded_timestamp", None)
    delta = getattr(frame, "timestamp_delta", None)
    requested_value = None if requested is None else round(float(requested), 6)
    decoded_value = None if decoded is None else round(float(decoded), 6)
    if delta is None and requested_value is not None and decoded_value is not None:
        delta = decoded_value - requested_value
    delta_value = None if delta is None else round(float(delta), 6)
    return {
        "requested_timestamp": requested_value,
        "decoded_timestamp": decoded_value,
        "timestamp_delta": delta_value,
    }


class AgenticVideoEngine:
    """Core local-first programmatic engine for agentic video understanding."""
    
    def __init__(self, source_path: str, cache: Optional[MediaCache] = None):
        self.source_path = source_path
        self._cache = cache or MediaCache()
        self._meta: Optional[VideoMetadata] = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._meta is None:
            self._meta = probe_video(self.source_path, cache=self._cache)
        return self._meta

    def probe(self) -> VideoMetadata:
        return self.metadata

    def transcribe(
        self,
        model_size: str = Config.DEFAULT_WHISPER_MODEL,
        subtitle_override: Optional[str] = None,
        timeout_sec: Optional[float] = None
    ) -> TranscriptResult:
        return transcribe_video(self.source_path, model_size=model_size, subtitle_override=subtitle_override, timeout_sec=timeout_sec)

    def transcribe_request(self, model_size: str = Config.DEFAULT_WHISPER_MODEL) -> Dict[str, Any]:
        """Return an MCP-ready cached transcript or bounded background status."""
        return request_transcription(self.source_path, model_size=model_size)

    def search_transcript(
        self,
        query: str,
        top_k: int = 5,
        time_range: Optional[tuple[float, float]] = None,
    ) -> Union[List[SearchResult], Dict[str, Any]]:
        return search_transcript(self.source_path, query, top_k=top_k, time_range=time_range)

    def build_visual_index(self) -> Dict[str, Any]:
        """Start/resume local indexing in a background worker; repeat to poll."""
        from src.retrieval.visual import submit
        return submit(self, "build")

    def search_visual(self, query: str, top_k: int = 5, start_sec: float = 0.0,
                      end_sec: Optional[float] = None) -> Dict[str, Any]:
        """Search visible content, asynchronously. Repeat the exact call to poll."""
        from src.retrieval.visual import submit
        return submit(self, "search", query=query, top_k=top_k,
                      start_sec=start_sec, end_sec=end_sec)

    def count_repetitions(self, start_sec: float = 0.0, end_sec: Optional[float] = None,
                          region=None, fps: float = 10.0, method: str = "auto",
                          min_period_sec: float = 0.4, max_period_sec: float = 3.0,
                          action_hint: Optional[str] = None) -> Dict[str, Any]:
        """Start or poll a bounded repetition count in a background worker."""
        from src.retrieval.repetition import submit
        return submit(self, start_sec=start_sec, end_sec=end_sec, region=region, fps=fps,
                      method=method, min_period_sec=min_period_sec,
                      max_period_sec=max_period_sec, action_hint=action_hint)

    def count_occurrences(self, description: Optional[str] = None, text: Optional[str] = None,
                          case_sensitive: bool = False, start_sec: float = 0.0,
                          end_sec: Optional[float] = None, fps: float = 5.0,
                          min_gap_sec: float = 0.3, max_spans: int = 20) -> Dict[str, Any]:
        """Start or poll a bounded count of distinct appearances of a symbol or exact text."""
        from src.retrieval.occurrence import submit
        return submit(self, description=description, text=text, case_sensitive=case_sensitive,
                      start_sec=start_sec, end_sec=end_sec, fps=fps, min_gap_sec=min_gap_sec,
                      max_spans=max_spans)

    def overview(
        self,
        samples: int = 12,
        cols: int = 4,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        deadline_sec: Optional[float] = None,
    ) -> ContactSheetResult:
        """Generate a coarse visual overview for the whole video or a temporal range."""
        return generate_video_overview(
            self.source_path,
            samples=samples,
            cols=cols,
            start_sec=start_sec,
            end_sec=end_sec,
            cache=self._cache,
            deadline_sec=deadline_sec,
        )

    def _overview_batch(self, ranges: List[Dict[str, Any]]) -> List[ContactSheetResult]:
        """Overviews for several ranges, extracted together from one shared worker pool."""
        return generate_video_overviews(self.source_path, ranges, cache=self._cache)

    def overview_chunks(
        self,
        chunk_sec: float = 60.0,
        samples_per_chunk: int = 8,
        cols: int = 4,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        max_chunks: int = 32,
    ) -> List[Dict[str, Any]]:
        """Build deterministic ordered overview sheets across a temporal span."""
        meta = self.metadata
        try:
            start = float(start_sec)
            end = meta.duration_sec if end_sec is None else float(end_sec)
            chunk = float(chunk_sec)
            max_count = int(max_chunks)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "Chunk overview bounds and chunk_sec must be numeric",
                {"start_sec": start_sec, "end_sec": end_sec, "chunk_sec": chunk_sec, "max_chunks": max_chunks},
            ) from exc
        if not all(math.isfinite(value) for value in (start, end, chunk)):
            raise ExtractionBoundsError(
                "Chunk overview bounds and chunk_sec must be finite",
                {"start_sec": start, "end_sec": end, "chunk_sec": chunk},
            )
        if start < 0 or start >= meta.duration_sec or end <= start or end > meta.duration_sec + 0.001:
            raise ExtractionBoundsError(
                "Chunk overview window is outside media bounds; use 0 <= start_sec < end_sec <= media_duration_sec, or omit end_sec",
                {"start_sec": start, "end_sec": end, "media_duration_sec": meta.duration_sec},
            )
        if chunk <= 0 or max_count < 1:
            raise ExtractionBoundsError(
                "chunk_sec and max_chunks must be positive; defaults are chunk_sec=60 and max_chunks=32",
                {"chunk_sec": chunk, "max_chunks": max_count},
            )
        end = min(end, meta.duration_sec)
        count = int(math.ceil((end - start) / chunk))
        if count > max_count:
            raise ExtractionBoundsError(
                f"Chunk overview exceeds max_chunks budget: {count} chunks of {chunk:g}s but max_chunks={max_count}; "
                f"use chunk_sec >= {math.ceil((end - start) / max_count):g}, narrow start_sec/end_sec, or raise max_chunks (at most 32)",
                {"chunk_count": count, "max_chunks": max_count, "chunk_sec": chunk},
            )

        bounds: List[Tuple[float, float]] = []
        cursor = start
        for _ in range(count):
            chunk_end = min(end, cursor + chunk)
            bounds.append((cursor, chunk_end))
            cursor = chunk_end
        sheets = self._overview_batch(
            [{"samples": samples_per_chunk, "cols": cols, "start_sec": a, "end_sec": b} for a, b in bounds]
        )
        chunks: List[Dict[str, Any]] = []
        for index, ((cursor, chunk_end), sheet) in enumerate(zip(bounds, sheets)):
            chunk_item = sheet.to_dict()
            chunk_item.update({
                "index": index + 1,
                "range_start_sec": cursor,
                "range_end_sec": chunk_end,
            })
            chunks.append(chunk_item)
        return chunks

    def refine_temporal_beam(
        self,
        windows: List[Dict[str, float]],
        fanout: int = 4,
        samples_per_child: int = 3,
        cols: int = 3,
        context_sec: float = 0.0,
        max_total_samples: int = 96,
        align_to_cuts: bool = False,
        compact_parent_sheet: bool = False,
    ) -> Dict[str, Any]:
        """Subdivide candidate windows into bounded, timestamped child overviews.

        The primitive preserves every parent branch and deliberately does not
        rank relevance. Parent/child ranges are absolute half-open video seconds.
        """
        meta = self.metadata
        if not isinstance(windows, list) or not windows:
            raise ExtractionBoundsError(
                'refine_temporal_beam requires a non-empty windows list, e.g. [{"start_sec": 10, "end_sec": 40}]',
                {"windows": windows},
            )
        try:
            fanout_value = int(fanout)
            samples_value = int(samples_per_child)
            cols_value = int(cols)
            context = float(context_sec)
            budget = int(max_total_samples)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "refine_temporal_beam options must be numeric",
                {
                    "fanout": fanout,
                    "samples_per_child": samples_per_child,
                    "cols": cols,
                    "context_sec": context_sec,
                    "max_total_samples": max_total_samples,
                },
            ) from exc
        if fanout_value < 2 or fanout_value > 8:
            raise ExtractionBoundsError(
                "refine_temporal_beam fanout must satisfy 2 <= fanout <= 8 (default 4)",
                {"fanout": fanout_value},
            )
        if samples_value < 1 or samples_value > 12:
            raise ExtractionBoundsError(
                "refine_temporal_beam samples_per_child must satisfy 1 <= samples_per_child <= 12 (default 3)",
                {"samples_per_child": samples_value},
            )
        if cols_value < 1:
            raise ExtractionBoundsError(
                "refine_temporal_beam cols must be positive",
                {"cols": cols_value},
            )
        if not math.isfinite(context) or context < 0:
            raise ExtractionBoundsError(
                "refine_temporal_beam context_sec must be finite and >= 0",
                {"context_sec": context},
            )
        if budget < 1 or budget > 300:
            raise ExtractionBoundsError(
                "refine_temporal_beam max_total_samples must satisfy 1 <= max_total_samples <= 300 (default 96)",
                {"max_total_samples": budget},
            )

        parents: List[Dict[str, Any]] = []
        for input_index, item in enumerate(windows):
            if not isinstance(item, dict) or "start_sec" not in item or "end_sec" not in item:
                raise ExtractionBoundsError(
                    'Each refine_temporal_beam window requires start_sec and end_sec, e.g. {"start_sec": 10, "end_sec": 40}',
                    {"input_index": input_index},
                )
            try:
                input_start = float(item["start_sec"])
                input_end = float(item["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ExtractionBoundsError(
                    "refine_temporal_beam timestamps must be numeric",
                    {"input_index": input_index},
                ) from exc
            if (
                not math.isfinite(input_start)
                or not math.isfinite(input_end)
                or input_start < 0
                or input_start >= meta.duration_sec
                or input_end <= input_start
                or input_end > meta.duration_sec + 0.001
            ):
                raise ExtractionBoundsError(
                    "refine_temporal_beam window is outside media bounds; use 0 <= start_sec < end_sec <= media_duration_sec",
                    {
                        "input_index": input_index,
                        "start_sec": input_start,
                        "end_sec": input_end,
                        "media_duration_sec": meta.duration_sec,
                    },
                )
            parent_start = max(0.0, input_start - context)
            parent_end = min(meta.duration_sec, input_end + context)
            parents.append({
                "parent_index": input_index + 1,
                "input_start_sec": input_start,
                "input_end_sec": min(input_end, meta.duration_sec),
                "start_sec": parent_start,
                "end_sec": parent_end,
            })

        planned_samples = len(parents) * fanout_value * samples_value
        if planned_samples > budget:
            raise ExtractionBoundsError(
                f"refine_temporal_beam exceeds max_total_samples budget: {planned_samples} planned "
                f"(windows x fanout x samples_per_child) > {budget}; pass fewer windows, lower fanout or samples_per_child, "
                "or raise max_total_samples (at most 300)",
                {
                    "planned_samples": planned_samples,
                    "max_total_samples": budget,
                    "parent_count": len(parents),
                    "fanout": fanout_value,
                    "samples_per_child": samples_value,
                },
            )

        # Collect every overview this call needs, then extract all frames from one shared pool.
        requests: List[Dict[str, Any]] = []
        for parent in parents:
            start = parent["start_sec"]
            end = parent["end_sec"]
            span = end - start
            child_width = span / fanout_value
            boundaries = [start + child_width * i for i in range(fanout_value + 1)]
            boundaries[0] = start
            boundaries[-1] = end

            if align_to_cuts and fanout_value > 1:
                cuts = self.detect_cuts(start_sec=start, end_sec=end)
                cut_times = [cut.timestamp_sec for cut in cuts if start < cut.timestamp_sec < end]
                snap_radius = min(5.0, child_width * 0.25)
                for boundary_index in range(1, fanout_value):
                    nominal = start + child_width * boundary_index
                    candidates = [ts for ts in cut_times if abs(ts - nominal) <= snap_radius]
                    if candidates:
                        boundaries[boundary_index] = min(
                            candidates,
                            key=lambda ts: (abs(ts - nominal), ts),
                        )

            if compact_parent_sheet:
                compact_samples = fanout_value * samples_value
                requests.append({
                    "parent": parent, "child_index": None,
                    "range": {"samples": compact_samples, "cols": min(4, compact_samples), "start_sec": start, "end_sec": end},
                })
            for child_index in range(fanout_value):
                child_start = boundaries[child_index]
                child_end = boundaries[child_index + 1]
                if child_end <= child_start:
                    raise ExtractionBoundsError(
                        "refine_temporal_beam produced a non-positive child range",
                        {
                            "parent_index": parent["parent_index"],
                            "child_index": child_index + 1,
                            "start_sec": child_start,
                            "end_sec": child_end,
                        },
                    )
                requests.append({
                    "parent": parent, "child_index": child_index,
                    "range": {"samples": samples_value, "cols": cols_value, "start_sec": child_start, "end_sec": child_end},
                })

        sheets = self._overview_batch([r["range"] for r in requests])
        total_samples = 0
        for request, sheet in zip(requests, sheets):
            parent = request["parent"]
            if request["child_index"] is None:
                parent["compact_contact_sheet_path"] = sheet.file_path
                parent["compact_sample_timestamps"] = list(sheet.timestamps)
                parent["compact_sample_count"] = sheet.sample_count
                continue
            total_samples += sheet.sample_count
            parent.setdefault("children", []).append({
                "child_index": request["child_index"] + 1,
                "start_sec": request["range"]["start_sec"],
                "end_sec": request["range"]["end_sec"],
                "sample_timestamps": list(sheet.timestamps),
                "contact_sheet_path": sheet.file_path,
                "sample_count": sheet.sample_count,
            })

        return {
            "parents": parents,
            "total_samples": total_samples,
            "estimated_vision_tokens": total_samples * Config.TOKENS_PER_IMAGE,
        }

    def track_region_changes(
        self,
        start_sec: float,
        end_sec: float,
        region: Dict[str, float],
        sample_fps: float = 0.25,
        frame_width: int = 640,
        min_change_score: float = 0.02,
        robust_sigma: float = 6.0,
        min_separation_sec: float = 2.0,
        max_candidates: int = 32,
        max_total_frames: int = 600,
    ) -> Dict[str, Any]:
        """Rank visual state-change candidates inside a stable normalized region."""
        meta = self.metadata
        try:
            start = float(start_sec)
            end = float(end_sec)
            fps = float(sample_fps)
            width = int(frame_width)
            floor = float(min_change_score)
            sigma = float(robust_sigma)
            separation = float(min_separation_sec)
            candidate_cap = int(max_candidates)
            frame_cap = int(max_total_frames)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError("track_region_changes options must be numeric", {}) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > meta.duration_sec + 0.001:
            raise ExtractionBoundsError("track_region_changes window is outside media bounds; use 0 <= start_sec < end_sec <= media_duration_sec (both are required)", {"start_sec": start, "end_sec": end, "media_duration_sec": meta.duration_sec})
        if not math.isfinite(fps) or fps <= 0 or fps > 10:
            raise ExtractionBoundsError("track_region_changes sample_fps must satisfy 0 < fps <= 10 (default 0.25)", {"sample_fps": fps})
        if width < 64 or width > 1920:
            raise ExtractionBoundsError("track_region_changes frame_width must satisfy 64 <= width <= 1920 (default 640)", {"frame_width": width})
        if not math.isfinite(floor) or floor < 0 or floor > 1 or not math.isfinite(sigma) or sigma < 0 or not math.isfinite(separation) or separation < 0:
            raise ExtractionBoundsError("track_region_changes thresholds are out of bounds; min_change_score must be in [0, 1], robust_sigma and min_separation_sec >= 0", {"min_change_score": floor, "robust_sigma": sigma, "min_separation_sec": separation})
        if candidate_cap < 1 or candidate_cap > 128 or frame_cap < 2 or frame_cap > 2000:
            raise ExtractionBoundsError("track_region_changes budgets are out of bounds; max_candidates must be 1-128 and max_total_frames 2-2000", {"max_candidates": candidate_cap, "max_total_frames": frame_cap})
        if not isinstance(region, dict):
            raise ExtractionBoundsError("track_region_changes region must be an object with x, y, width, height as fractions of the frame, e.g. {\"x\": 0.25, \"y\": 0.1, \"width\": 0.5, \"height\": 0.6}", {"region": region})
        try:
            rx = float(region["x"]); ry = float(region["y"]); rw = float(region["width"]); rh = float(region["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractionBoundsError("track_region_changes region requires numeric x, y, width, height (this tool uses width/height, not w/h)", {"region": region}) from exc
        if not all(math.isfinite(v) for v in (rx, ry, rw, rh)) or rx < 0 or ry < 0 or rw <= 0 or rh <= 0 or rx + rw > 1.000001 or ry + rh > 1.000001:
            raise ExtractionBoundsError("track_region_changes region must lie within normalized [0,1] bounds: x, y >= 0, width, height > 0, x+width <= 1, y+height <= 1", {"region": {"x": rx, "y": ry, "width": rw, "height": rh}})
        planned = int(math.ceil((end - start) * fps)) + 1
        if planned > frame_cap:
            raise ExtractionBoundsError(f"track_region_changes exceeds max_total_frames budget: {planned} frames planned > {frame_cap}; lower sample_fps to at most {max(frame_cap - 1, 1) / (end - start):.3g} for this window, shorten start_sec/end_sec, or raise max_total_frames (at most 2000)", {"planned_frames": planned, "max_total_frames": frame_cap})

        frames = []
        scan_segments = []
        max_segment_duration = min(
            180.0,
            Config.MAX_WINDOW_DURATION_SEC,
            max(0.1, (Config.MAX_FRAMES_PER_CALL - 2) / fps),
        )
        base_segments = []
        cursor = start
        while cursor < end - 1e-9:
            segment_end = min(end, cursor + max_segment_duration)
            base_segments.append((cursor, segment_end))
            cursor = segment_end

        # Process a LIFO worklist so that adding right then left halves keeps
        # successful receipts chronological while making timeout recovery
        # iterative and strictly bounded.
        pending_segments = list(reversed(base_segments))
        max_attempts = max(len(base_segments), (2 * frame_cap) - 1)
        attempts = 0
        while pending_segments:
            segment_start, segment_end = pending_segments.pop()
            attempts += 1
            if attempts > max_attempts:
                raise ExtractionBoundsError(
                    "track_region_changes timeout recovery exceeded bounded attempts",
                    {"attempts": attempts, "max_attempts": max_attempts, "max_total_frames": frame_cap},
                )
            try:
                segment_frames = sample_sparse_frames(
                    self.source_path, fps=fps, start_sec=segment_start, end_sec=segment_end, width=width,
                    overlay_timestamp=False, cache=self._cache,
                )
            except SubprocessTimeoutError:
                midpoint = segment_start + ((segment_end - segment_start) / 2.0)
                if midpoint <= segment_start or midpoint >= segment_end:
                    raise
                pending_segments.append((midpoint, segment_end))
                pending_segments.append((segment_start, midpoint))
                continue

            scan_segments.append({"start_sec": segment_start, "end_sec": segment_end, "sample_count": len(segment_frames)})
            for frame in segment_frames:
                if not frames or float(frame.timestamp_sec) > float(frames[-1].timestamp_sec) + 1e-6:
                    frames.append(frame)
            if len(frames) > frame_cap:
                raise ExtractionBoundsError(
                    "track_region_changes recovery exceeds max_total_frames",
                    {"sample_count": len(frames), "max_total_frames": frame_cap},
                )
        if len(frames) < 2:
            raise ExtractionBoundsError(f"track_region_changes needs at least two sampled frames but got {len(frames)}; raise sample_fps to at least {2 / (end - start):.3g} for this {end - start:.3g}s window (maximum 10) or widen start_sec/end_sec", {"sample_count": len(frames), "sample_fps": fps})

        from PIL import Image, ImageChops, ImageFilter, ImageStat
        prepared = []
        for frame in frames:
            with Image.open(frame.file_path) as source:
                image = source.convert("L")
                iw, ih = image.size
                left = max(0, min(iw - 1, int(round(rx * iw))))
                top = max(0, min(ih - 1, int(round(ry * ih))))
                right = max(left + 1, min(iw, int(round((rx + rw) * iw))))
                bottom = max(top + 1, min(ih, int(round((ry + rh) * ih))))
                crop = image.crop((left, top, right, bottom)).resize((64, 64), Image.Resampling.BILINEAR).filter(ImageFilter.BoxBlur(0.5))
                prepared.append(crop)

        samples: List[Dict[str, Any]] = []
        scores: List[float] = []
        for index in range(1, len(prepared)):
            diff = ImageChops.difference(prepared[index - 1], prepared[index])
            mean_abs = ImageStat.Stat(diff).mean[0] / 255.0
            histogram = diff.histogram()
            changed_fraction = sum(histogram[16:]) / sum(histogram)
            score = round((0.65 * mean_abs) + (0.35 * changed_fraction), 6)
            scores.append(score)
            samples.append({
                "from_timestamp_sec": float(frames[index - 1].timestamp_sec),
                "timestamp_sec": float(frames[index].timestamp_sec),
                "from_timestamp_provenance": _timestamp_provenance(frames[index - 1]),
                "timestamp_provenance": _timestamp_provenance(frames[index]),
                "change_score": score,
                "mean_abs_diff": round(mean_abs, 6),
                "changed_fraction": round(changed_fraction, 6),
            })

        sorted_scores = sorted(scores)
        count = len(sorted_scores)
        median = sorted_scores[count // 2] if count % 2 else (sorted_scores[count // 2 - 1] + sorted_scores[count // 2]) / 2.0
        deviations = sorted(abs(score - median) for score in scores)
        dcount = len(deviations)
        mad = deviations[dcount // 2] if dcount % 2 else (deviations[dcount // 2 - 1] + deviations[dcount // 2]) / 2.0
        robust_threshold = median + sigma * 1.4826 * mad
        threshold = min(1.0, max(floor, robust_threshold))

        raw_candidates = [item for item in samples if item["change_score"] >= threshold]
        raw_candidates.sort(key=lambda item: (-item["change_score"], item["timestamp_sec"]))
        selected: List[Dict[str, Any]] = []
        for item in raw_candidates:
            if all(abs(item["timestamp_sec"] - existing["timestamp_sec"]) >= separation for existing in selected):
                selected.append(dict(item))
                if len(selected) >= candidate_cap:
                    break
        for rank, item in enumerate(selected, 1):
            item["rank"] = rank

        # Slow, continuous change (a drifting background, a slow pan, grain)
        # raises every sample a little, so the robust threshold rises with it
        # and no candidate is ranked. Report it separately so "no candidates"
        # is not read as "nothing moves".
        activity_floor = 0.001
        active_fraction = sum(1 for score in scores if score >= activity_floor) / len(scores)
        steady_detected = active_fraction >= 0.6
        if steady_detected:
            steady_note = (
                "Most sample pairs change a little: the region changes continuously (slow motion, "
                "animation, grain or compression noise). Compare full-size crops 0.5-1 s apart to see which."
            )
        else:
            steady_note = "Most sample pairs are still; changes, if any, are the listed candidates."

        return {
            "region": {"x": rx, "y": ry, "width": rw, "height": rh},
            "start_sec": start,
            "end_sec": end,
            "sample_fps": fps,
            "sample_count": len(frames),
            "temporal_resolution_sec": round(1.0 / fps, 6),
            "scan_segments": scan_segments,
            "threshold": round(threshold, 6),
            "baseline_median": round(median, 6),
            "baseline_mad": round(mad, 6),
            "steady_change": {
                "detected": steady_detected,
                "active_fraction": round(active_fraction, 6),
                "activity_floor": activity_floor,
                "note": steady_note,
            },
            "samples": samples,
            "candidates": selected,
        }

    def track_region_events(
        self,
        start_sec: float,
        end_sec: float,
        region: Dict[str, float],
        sample_fps: float = 0.25,
        frame_width: int = 640,
        change_threshold: float = 0.02,
        ambiguity_margin: float = 0.01,
        min_separation_sec: float = 2.0,
        max_events: int = 32,
        max_total_frames: int = 600,
    ) -> Dict[str, Any]:
        """Return auditable visual transitions, not only a change count.

        This is deliberately a separate primitive from ``track_region_changes``.
        It treats threshold crossings as chronological transitions, preserves the
        two sampled moments that witness each transition, and refuses to report a
        precise count when the evidence is borderline, truncated, or incomplete.
        """
        meta = self.metadata
        try:
            start = float(start_sec)
            end = float(end_sec)
            fps = float(sample_fps)
            width = int(frame_width)
            threshold = float(change_threshold)
            margin = float(ambiguity_margin)
            separation = float(min_separation_sec)
            event_cap = int(max_events)
            frame_cap = int(max_total_frames)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError("track_region_events options must be numeric", {}) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > meta.duration_sec + 0.001:
            raise ExtractionBoundsError(
                "track_region_events window is outside media bounds; use 0 <= start_sec < end_sec <= media_duration_sec (both are required)",
                {"start_sec": start, "end_sec": end, "media_duration_sec": meta.duration_sec},
            )
        if not math.isfinite(fps) or fps <= 0 or fps > 10:
            raise ExtractionBoundsError("track_region_events sample_fps must satisfy 0 < fps <= 10 (default 0.25)", {"sample_fps": fps})
        if width < 64 or width > 1920:
            raise ExtractionBoundsError("track_region_events frame_width must satisfy 64 <= width <= 1920 (default 640)", {"frame_width": width})
        if (
            not math.isfinite(threshold)
            or threshold < 0
            or threshold > 1
            or not math.isfinite(margin)
            or margin < 0
            or margin > 1
            or not math.isfinite(separation)
            or separation < 0
        ):
            raise ExtractionBoundsError(
                "track_region_events thresholds are out of bounds; change_threshold and ambiguity_margin must be in [0, 1], min_separation_sec >= 0",
                {"change_threshold": threshold, "ambiguity_margin": margin, "min_separation_sec": separation},
            )
        if event_cap < 1 or event_cap > 128 or frame_cap < 2 or frame_cap > 2000:
            raise ExtractionBoundsError(
                "track_region_events budgets are out of bounds; max_events must be 1-128 and max_total_frames 2-2000",
                {"max_events": event_cap, "max_total_frames": frame_cap},
            )
        if not isinstance(region, dict):
            raise ExtractionBoundsError("track_region_events region must be an object with x, y, width, height as fractions of the frame, e.g. {\"x\": 0.25, \"y\": 0.1, \"width\": 0.5, \"height\": 0.6}", {"region": region})
        try:
            rx = float(region["x"])
            ry = float(region["y"])
            rw = float(region["width"])
            rh = float(region["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "track_region_events region requires numeric x, y, width, height (this tool uses width/height, not w/h)", {"region": region}
            ) from exc
        if not all(math.isfinite(v) for v in (rx, ry, rw, rh)) or rx < 0 or ry < 0 or rw <= 0 or rh <= 0 or rx + rw > 1.000001 or ry + rh > 1.000001:
            raise ExtractionBoundsError(
                "track_region_events region must lie within normalized [0,1] bounds: x, y >= 0, width, height > 0, x+width <= 1, y+height <= 1",
                {"region": {"x": rx, "y": ry, "width": rw, "height": rh}},
            )

        planned = max(2, int(math.ceil((end - start) * fps)))
        if planned > frame_cap:
            raise ExtractionBoundsError(
                f"track_region_events exceeds max_total_frames budget: {planned} frames planned > {frame_cap}; lower sample_fps to at most "
                f"{frame_cap / (end - start):.3g} for this window, shorten start_sec/end_sec, or raise max_total_frames (at most 2000)",
                {"planned_frames": planned, "max_total_frames": frame_cap},
            )

        max_segment_duration = min(
            180.0,
            Config.MAX_WINDOW_DURATION_SEC,
            max(0.1, (Config.MAX_FRAMES_PER_CALL - 2) / fps),
        )
        segments = []
        cursor = start
        while cursor < end - 1e-9:
            segment_end = min(end, cursor + max_segment_duration)
            segments.append((cursor, segment_end))
            cursor = segment_end

        frames = []
        scan_segments = []
        incomplete = False
        incomplete_reason = None
        for segment_start, segment_end in segments:
            try:
                segment_frames = sample_sparse_frames(
                    self.source_path,
                    fps=fps,
                    start_sec=segment_start,
                    end_sec=segment_end,
                    width=width,
                    overlay_timestamp=False,
                    cache=self._cache,
                )
            except SubprocessTimeoutError:
                incomplete = True
                incomplete_reason = "sampling timeout"
                scan_segments.append({"start_sec": segment_start, "end_sec": segment_end, "sample_count": 0})
                break

            scan_segments.append({"start_sec": segment_start, "end_sec": segment_end, "sample_count": len(segment_frames)})
            frames.extend(segment_frames)

        # Segment extraction can repeat a boundary frame.  Keep one chronological
        # sample per timestamp and preserve the requested range provenance.
        unique_frames = {}
        for frame in frames:
            timestamp = float(frame.timestamp_sec)
            if start - 1e-6 <= timestamp <= end + 1e-6:
                unique_frames.setdefault(round(timestamp, 6), frame)
        frames = [unique_frames[key] for key in sorted(unique_frames)]
        if len(frames) > frame_cap:
            raise ExtractionBoundsError(
                "track_region_events exceeds max_total_frames budget after sampling",
                {"sample_count": len(frames), "max_total_frames": frame_cap},
            )
        if len(frames) < planned:
            incomplete = True
            incomplete_reason = incomplete_reason or "sampler returned fewer frames than planned"
        if len(frames) < 2:
            reason = incomplete_reason or "insufficient sampled frames"
            return {
                "region": {"x": rx, "y": ry, "width": rw, "height": rh},
                "start_sec": start,
                "end_sec": end,
                "sample_fps": fps,
                "sample_count": len(frames),
                "temporal_resolution_sec": round(1.0 / fps, 6),
                "scan_segments": scan_segments,
                "transitions": [],
                "event_count": None,
                "uncertain": False,
                "incomplete": True,
                "uncertainty_reason": reason,
            }

        from PIL import Image, ImageChops, ImageFilter, ImageStat

        prepared = []
        for frame in frames:
            with Image.open(frame.file_path) as source:
                image = source.convert("L")
                iw, ih = image.size
                left = max(0, min(iw - 1, int(round(rx * iw))))
                top = max(0, min(ih - 1, int(round(ry * ih))))
                right = max(left + 1, min(iw, int(round((rx + rw) * iw))))
                bottom = max(top + 1, min(ih, int(round((ry + rh) * ih))))
                prepared.append(
                    image.crop((left, top, right, bottom))
                    .resize((64, 64), Image.Resampling.BILINEAR)
                    .filter(ImageFilter.BoxBlur(0.5))
                )

        samples: List[Dict[str, Any]] = []
        for index in range(1, len(prepared)):
            diff = ImageChops.difference(prepared[index - 1], prepared[index])
            mean_abs = ImageStat.Stat(diff).mean[0] / 255.0
            histogram = diff.histogram()
            changed_fraction = sum(histogram[16:]) / sum(histogram)
            score = round((0.65 * mean_abs) + (0.35 * changed_fraction), 6)
            samples.append(
                {
                    "from_timestamp_sec": float(frames[index - 1].timestamp_sec),
                    "timestamp_sec": float(frames[index].timestamp_sec),
                    "from_timestamp_provenance": _timestamp_provenance(frames[index - 1]),
                    "timestamp_provenance": _timestamp_provenance(frames[index]),
                    "change_score": score,
                    "mean_abs_diff": round(mean_abs, 6),
                    "changed_fraction": round(changed_fraction, 6),
                }
            )

        above_threshold = [item for item in samples if item["change_score"] >= threshold]
        ranked = sorted(above_threshold, key=lambda item: (-item["change_score"], item["timestamp_sec"]))
        selected = []
        for item in ranked:
            if all(abs(item["timestamp_sec"] - chosen["timestamp_sec"]) >= separation for chosen in selected):
                selected.append(item)
                if len(selected) >= event_cap:
                    break

        uncertainty_reasons = []
        if any(item["change_score"] < threshold + margin for item in selected):
            uncertainty_reasons.append("borderline change score")
        if len(above_threshold) > event_cap:
            incomplete = True
            uncertainty_reasons.append("max_events cap reached")
        if incomplete and incomplete_reason:
            uncertainty_reasons.append(incomplete_reason)
        uncertain = bool(uncertainty_reasons)

        selected.sort(key=lambda item: item["timestamp_sec"])
        transitions = []
        for rank, item in enumerate(selected, 1):
            transitions.append(
                {
                    "from_timestamp_sec": item["from_timestamp_sec"],
                    "timestamp_sec": item["timestamp_sec"],
                    "from_timestamp_provenance": item["from_timestamp_provenance"],
                    "timestamp_provenance": item["timestamp_provenance"],
                    "change_score": item["change_score"],
                    "mean_abs_diff": item["mean_abs_diff"],
                    "changed_fraction": item["changed_fraction"],
                    "evidence_moments": {
                        "before": {
                            "timestamp_sec": item["from_timestamp_sec"],
                            "timestamp_provenance": item["from_timestamp_provenance"],
                        },
                        "after": {
                            "timestamp_sec": item["timestamp_sec"],
                            "timestamp_provenance": item["timestamp_provenance"],
                        },
                    },
                    "rank": rank,
                }
            )

        return {
            "region": {"x": rx, "y": ry, "width": rw, "height": rh},
            "start_sec": start,
            "end_sec": end,
            "sample_fps": fps,
            "sample_count": len(frames),
            "temporal_resolution_sec": round(1.0 / fps, 6),
            "scan_segments": scan_segments,
            "transitions": transitions,
            "event_count": None if uncertain or incomplete else len(transitions),
            "uncertain": uncertain,
            "incomplete": incomplete,
            "uncertainty_reason": "; ".join(uncertainty_reasons) if uncertainty_reasons else None,
        }

    def inspect_moments(
        self,
        moments: List[Dict[str, Any]],
        before_sec: float = 4.0,
        after_sec: float = 4.0,
        samples_per_moment: int = 9,
        cols: int = 3,
        max_moments: int = 16,
        max_total_samples: int = 144,
    ) -> Dict[str, Any]:
        """Inspect anchored moments with explicit chronological before/anchor/after evidence."""
        meta = self.metadata
        if not isinstance(moments, list) or not moments:
            raise ExtractionBoundsError("inspect_moments requires a non-empty moments list, e.g. [{\"timestamp_sec\": 12.5, \"label\": \"door opens\"}]", {"moments": moments})
        try:
            before = float(before_sec)
            after = float(after_sec)
            samples = int(samples_per_moment)
            cols_value = int(cols)
            max_count = int(max_moments)
            budget = int(max_total_samples)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "inspect_moments options must be numeric",
                {
                    "before_sec": before_sec,
                    "after_sec": after_sec,
                    "samples_per_moment": samples_per_moment,
                    "cols": cols,
                    "max_moments": max_moments,
                    "max_total_samples": max_total_samples,
                },
            ) from exc
        if not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0 or (before == 0 and after == 0):
            raise ExtractionBoundsError("inspect_moments requires finite non-negative before_sec/after_sec with at least one above zero (defaults 4 and 4)", {"before_sec": before, "after_sec": after})
        if samples < 3 or samples > 25 or samples % 2 == 0:
            raise ExtractionBoundsError("inspect_moments samples_per_moment must be an odd integer in [3, 25] (default 9)", {"samples_per_moment": samples})
        if cols_value < 1:
            raise ExtractionBoundsError("inspect_moments cols must be positive", {"cols": cols_value})
        if max_count < 1 or max_count > 16 or budget < 1 or budget > 400:
            raise ExtractionBoundsError("inspect_moments budgets are out of bounds; max_moments must be 1-16 and max_total_samples 1-400", {"max_moments": max_count, "max_total_samples": budget})
        if len(moments) > max_count:
            raise ExtractionBoundsError(f"inspect_moments exceeds max_moments budget: {len(moments)} moments > {max_count}; call again with the remaining moments (at most 16 per call)", {"moment_count": len(moments), "max_moments": max_count})
        planned = len(moments) * samples
        if planned > budget:
            raise ExtractionBoundsError(f"inspect_moments exceeds max_total_samples budget: {planned} planned (moments x samples_per_moment) > {budget}; "
                "pass fewer moments, lower samples_per_moment, or raise max_total_samples (at most 400)", {"planned_samples": planned, "max_total_samples": budget})

        side = samples // 2
        last_valid = max(0.0, meta.duration_sec - 0.001)
        planned_moments: List[Dict[str, Any]] = []
        for input_index, item in enumerate(moments):
            if not isinstance(item, dict) or "timestamp_sec" not in item:
                raise ExtractionBoundsError("Each inspect_moments item requires timestamp_sec (seconds), e.g. {\"timestamp_sec\": 12.5}", {"input_index": input_index})
            try:
                anchor = float(item["timestamp_sec"])
            except (TypeError, ValueError) as exc:
                raise ExtractionBoundsError("inspect_moments timestamps must be numeric", {"input_index": input_index}) from exc
            if not math.isfinite(anchor) or anchor < 0 or anchor >= meta.duration_sec:
                raise ExtractionBoundsError("inspect_moments timestamp is outside media bounds; use 0 <= timestamp_sec < media_duration_sec", {"input_index": input_index, "timestamp_sec": anchor, "media_duration_sec": meta.duration_sec})
            start = max(0.0, anchor - before)
            end = min(last_valid, anchor + after)
            before_ts = [start + (anchor - start) * i / side for i in range(side)] if anchor > start else []
            after_ts = [anchor + (end - anchor) * (i + 1) / side for i in range(side)] if end > anchor else []
            timestamps = before_ts + [anchor] + after_ts
            anchor_index = len(before_ts) + 1
            label = item.get("label")
            label_text = f" [{label}]" if label is not None else ""
            planned_moments.append({
                "input_index": input_index, "anchor": anchor, "label": label, "start": start, "end": end,
                "anchor_index": anchor_index,
                "job": {
                    "timestamps": timestamps,
                    "cols": min(cols_value, len(timestamps)),
                    "title": f"Moment {input_index + 1}{label_text}: anchor #{anchor_index} @ {format_timestamp(anchor)}",
                },
            })
        sheets = build_contact_sheets(self.source_path, [m["job"] for m in planned_moments], cache=self._cache)
        items: List[Dict[str, Any]] = []
        total_samples = 0
        for plan, sheet in zip(planned_moments, sheets):
            anchor, label, start, end = plan["anchor"], plan["label"], plan["start"], plan["end"]
            anchor_index = plan["anchor_index"]
            total_samples += sheet.sample_count
            items.append({
                "index": plan["input_index"] + 1,
                "timestamp_sec": anchor,
                "label": None if label is None else str(label),
                "start_sec": start,
                "end_sec": end,
                "anchor_sample_index": anchor_index,
                "sample_timestamps": list(sheet.timestamps),
                "relative_offsets_sec": [round(ts - anchor, 6) for ts in sheet.timestamps],
                "contact_sheet_path": sheet.file_path,
                "sample_count": sheet.sample_count,
            })
        return {
            "moments": items,
            "total": len(items),
            "total_samples": total_samples,
            "estimated_vision_tokens": total_samples * Config.TOKENS_PER_IMAGE,
        }

    def inspect_windows(
        self,
        windows: List[Dict[str, float]],
        fps: float = Config.DEFAULT_DENSE_FPS,
        context_sec: float = 0.0,
        merge_overlaps: bool = True,
        max_windows: int = 16,
        max_total_frames: int = 150,
    ) -> Dict[str, Any]:
        """Inspect multiple absolute half-open windows under one bounded budget."""
        meta = self.metadata
        if not isinstance(windows, list) or not windows:
            raise ExtractionBoundsError("inspect_windows requires a non-empty windows list, e.g. [{\"start_sec\": 10, \"end_sec\": 14}]", {"windows": windows})
        try:
            fps_value = float(fps)
            context = float(context_sec)
            max_count = int(max_windows)
            max_frames = int(max_total_frames)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "inspect_windows options must be numeric",
                {"fps": fps, "context_sec": context_sec, "max_windows": max_windows, "max_total_frames": max_total_frames},
            ) from exc
        if not math.isfinite(fps_value) or fps_value <= 0 or fps_value > Config.MAX_EXTRACTION_FPS:
            raise ExtractionBoundsError("inspect_windows fps is out of bounds; use 0 < fps <= 30 (default 4)", {"fps": fps_value})
        if not math.isfinite(context) or context < 0:
            raise ExtractionBoundsError("inspect_windows context_sec must be finite and >= 0", {"context_sec": context})
        if max_count < 1 or max_frames < 1:
            raise ExtractionBoundsError("inspect_windows budgets must be positive (defaults max_windows=16, max_total_frames=150)", {"max_windows": max_count, "max_total_frames": max_frames})
        if len(windows) > max_count:
            raise ExtractionBoundsError(f"inspect_windows exceeds max_windows budget: {len(windows)} windows > {max_count}; call again with the remaining windows (at most 16 per call)", {"window_count": len(windows), "max_windows": max_count})

        normalized: List[Dict[str, Any]] = []
        for input_index, item in enumerate(windows):
            if not isinstance(item, dict) or "start_sec" not in item or "end_sec" not in item:
                raise ExtractionBoundsError("Each inspect_windows item requires start_sec and end_sec, e.g. {\"start_sec\": 10, \"end_sec\": 14}", {"input_index": input_index})
            try:
                start = float(item["start_sec"])
                end = float(item["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ExtractionBoundsError("inspect_windows timestamps must be numeric", {"input_index": input_index}) from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > meta.duration_sec + 0.001:
                raise ExtractionBoundsError(
                    "inspect_windows window is outside media bounds; use 0 <= start_sec < end_sec <= media_duration_sec",
                    {"input_index": input_index, "start_sec": start, "end_sec": end, "media_duration_sec": meta.duration_sec},
                )
            expanded_start = max(0.0, start - context)
            expanded_end = min(meta.duration_sec, end + context)
            normalized.append({
                "start_sec": expanded_start,
                "end_sec": expanded_end,
                "input_indices": [input_index],
            })

        normalized.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
        selected: List[Dict[str, Any]] = []
        for item in normalized:
            if merge_overlaps and selected and item["start_sec"] <= selected[-1]["end_sec"]:
                selected[-1]["end_sec"] = max(selected[-1]["end_sec"], item["end_sec"])
                selected[-1]["input_indices"].extend(item["input_indices"])
            else:
                selected.append({
                    "start_sec": item["start_sec"],
                    "end_sec": item["end_sec"],
                    "input_indices": list(item["input_indices"]),
                })

        predicted_frames = sum(math.ceil((item["end_sec"] - item["start_sec"]) * fps_value) for item in selected)
        if predicted_frames > max_frames:
            raise ExtractionBoundsError(
                f"inspect_windows exceeds max_total_frames budget: ~{predicted_frames} frames predicted > {max_frames}; lower fps, "
                "shorten the windows, or raise max_total_frames (at most 500); contact_sheet gives a cheaper overview of a long span",
                {"predicted_frames": predicted_frames, "max_total_frames": max_frames},
            )

        items: List[Dict[str, Any]] = []
        total_frames = 0
        total_tokens = 0
        # Windows are independent ffmpeg runs; a few at once (capped, the machine is shared), results in order.
        with ThreadPoolExecutor(max_workers=min(WINDOW_WORKERS, len(selected))) as pool:
            results = list(pool.map(
                lambda item: self.inspect_window(start_sec=item["start_sec"], end_sec=item["end_sec"], fps=fps_value),
                selected,
            ))
        for index, (item, result) in enumerate(zip(selected, results), start=1):
            payload = result.to_dict()
            frame_count = len(payload["frames"])
            total_frames += frame_count
            total_tokens += int(payload["estimated_vision_tokens"])
            items.append({
                "index": index,
                "input_indices": item["input_indices"],
                "start_sec": payload["start_sec"],
                "end_sec": payload["end_sec"],
                "fps": payload["fps"],
                "frames": payload["frames"],
                "contact_sheet_path": payload.get("contact_sheet_path"),
                "estimated_vision_tokens": payload["estimated_vision_tokens"],
            })
        return {
            "windows": items,
            "total": len(items),
            "total_frames": total_frames,
            "estimated_vision_tokens": total_tokens,
        }

    def inspect_window(
        self,
        start_sec: float,
        end_sec: float,
        fps: float = Config.DEFAULT_DENSE_FPS,
        create_sheet: bool = True
    ) -> InspectionWindowResult:
        """Densify inspection over a target time window [start_sec, end_sec]."""
        frames = extract_window_frames(
            self.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps,
            burn_timestamps=True,
            cache=self._cache,
        )
        sheet_path = None
        if create_sheet and frames:
            # Reuse the dense frames: seeking each one again with its own
            # ffmpeg run took over a minute for a few windows.
            timestamps = [f.timestamp_sec for f in frames]
            cols = min(4, len(timestamps))
            title = f"Inspection: [{format_timestamp(start_sec)} - {format_timestamp(end_sec)}] @ {fps} FPS"
            sheet = compose_contact_sheet(
                frames,
                default_sheet_path(self._cache, self.source_path, timestamps, cols),
                title=title,
                cols=cols,
            )
            sheet_path = sheet.file_path
            
        estimated_tokens = len(frames) * Config.TOKENS_PER_IMAGE
        return InspectionWindowResult(
            source_path=self.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps,
            frames=frames,
            contact_sheet_path=sheet_path,
            estimated_vision_tokens=estimated_tokens
        )

    def detect_cuts(
        self,
        threshold: float = 0.25,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[SceneCut]:
        """Detect cuts in the whole video or a bounded temporal range."""
        meta = self.metadata
        try:
            start = float(start_sec)
            end = meta.duration_sec if end_sec is None else float(end_sec)
        except (TypeError, ValueError) as exc:
            raise ExtractionBoundsError(
                "Cut-detection timestamps must be numeric",
                {"start_sec": start_sec, "end_sec": end_sec},
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ExtractionBoundsError(
                "Cut-detection timestamps must be finite",
                {"start_sec": start, "end_sec": end},
            )
        if start < 0 or start >= meta.duration_sec or end <= start or end > meta.duration_sec + 0.001:
            raise ExtractionBoundsError(
                "Cut-detection window is outside media bounds",
                {"start_sec": start, "end_sec": end, "media_duration_sec": meta.duration_sec},
            )
        duration = None if end_sec is None and start == 0.0 else min(end, meta.duration_sec) - start
        # A hard cut also shows up in the soft-transition scan and is removed
        # there by matching it to the hard-cut list. With a raised threshold
        # that list is short, so match against the default-threshold cuts:
        # a cut the caller filtered out must not come back as a transition.
        # One decode fills the cache for the three calls below.
        scan_scene_changes(
            self.source_path,
            thresholds=(threshold, 0.25) if threshold > 0.25 else (threshold,),
            start_sec=start,
            duration_sec=duration,
            cache=self._cache,
            defer_echoes=True,
        )
        known_cuts = None
        if threshold > 0.25:
            known_cuts = detect_scenes(
                self.source_path,
                threshold=0.25,
                start_sec=start,
                duration_sec=duration,
                cache=self._cache,
            )
        cuts = detect_scenes(
            self.source_path,
            threshold=threshold,
            start_sec=start,
            duration_sec=duration,
            cache=self._cache,
        )
        soft = detect_soft_transitions(
            self.source_path,
            hard_cuts=cuts if known_cuts is None else known_cuts,
            start_sec=start,
            duration_sec=duration,
            cache=self._cache,
        )
        if known_cuts is not None:
            # A cut found by the frame-step check scored below 0.25, so a
            # caller who raised the threshold has filtered it out too.
            soft = [cut for cut in soft if cut.kind != "cut"]
        cuts = sorted(list(cuts) + soft, key=lambda cut: cut.timestamp_sec)
        if end_sec is None and start == 0.0:
            return cuts
        return [cut for cut in cuts if start <= cut.timestamp_sec < end]

    def audio_onsets(
        self,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        min_strength: float = 0.12,
        band_hz: Any = None,
        compare_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect non-speech attacks in a selected band and compare optional media."""
        return measure_audio_onsets(
            self.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            min_strength=min_strength,
            cache=self._cache,
            band_hz=band_hz,
            compare_to=compare_to,
        )

    def loudness_timeline(
        self,
        window_sec: Optional[float] = None,
        bpm: Optional[float] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        ignore_ranges: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Measure loudness and spectral energy in fixed or beat-length windows."""
        return measure_loudness_timeline(
            self.source_path,
            window_sec=window_sec,
            bpm=bpm,
            start_sec=start_sec,
            end_sec=end_sec,
            cache=self._cache,
            ignore_ranges=ignore_ranges,
        )

    def spectrogram_window(
        self,
        start_sec: float,
        end_sec: float,
        marks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Render a log-frequency, timecoded PNG for a bounded audio window."""
        return render_spectrogram_window(
            self.source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            marks=marks,
            cache=self._cache,
        )

    def sync_check(
        self,
        cues: Optional[List[Dict[str, Any]]] = None,
        tolerance_ms: float = 80.0,
    ) -> Dict[str, Any]:
        """Compare detected visual cuts and audio onsets to intended cues."""
        return compare_audio_video_sync(
            self.source_path,
            cues=cues,
            tolerance_ms=tolerance_ms,
            cache=self._cache,
            cut_detector=self.detect_cuts,
            # The search window must be at least the tolerance; 500 ms is the default.
            search_window_ms=max(500.0, float(tolerance_ms)),
        )

    def build_evidence(
        self,
        start_sec: float,
        end_sec: float,
        frame_paths: List[str],
        transcript_quote: Optional[str] = None,
        visual_description: Optional[str] = None,
        evidence_basis: str = "",
        selection_reason: str = "",
        provenance: str = "",
        retrieval_method: str = ""
    ) -> EvidenceItem:
        """Package verified evidence item with timestamps, frame paths, and auditable provenance."""
        return EvidenceItem(
            timestamp_start=start_sec,
            timestamp_end=end_sec,
            timecode_start=format_timestamp(start_sec),
            timecode_end=format_timestamp(end_sec),
            frame_paths=frame_paths,
            transcript_quote=transcript_quote,
            visual_description=visual_description,
            evidence_basis=evidence_basis,
            selection_reason=selection_reason,
            provenance=provenance,
            retrieval_method=retrieval_method
        )
