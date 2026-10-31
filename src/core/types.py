"""Data structures, schemas, and evidence representations."""
from dataclasses import dataclass, field, asdict
import math
from typing import List, Dict, Any, Optional

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm timecode."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

@dataclass
class VideoMetadata:
    source_path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    format_name: str
    size_bytes: int
    has_audio: bool
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None
    fingerprint: str = ""
    # False for audio-only files, whose width, height and fps are 0.
    has_video: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FrameInfo:
    index: int
    timestamp_sec: float
    timecode: str
    file_path: str
    width: int
    height: int
    # ``timestamp_sec`` remains the requested/backwards-compatible clock.
    # These fields expose source-frame timing when ffprobe can resolve it.
    requested_timestamp: Optional[float] = None
    decoded_timestamp: Optional[float] = None
    timestamp_delta: Optional[float] = None
    # On-screen text read by OCR (read_text); kept out of the output unless it was asked for.
    text: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        text = d.pop("text")
        if text:
            d.update(text)
        return d

@dataclass
class TranscriptSegment:
    id: int
    start_sec: float
    end_sec: float
    text: str
    start_timecode: str = ""
    end_timecode: str = ""
    confidence: Optional[float] = None
    suspect: Optional[str] = None  # why the line may be invented (see src/transcript/repeats.py); None when not flagged

    def __post_init__(self):
        if not all(math.isfinite(float(value)) for value in (self.start_sec, self.end_sec)):
            raise ValueError("Transcript segment timestamps must be finite")
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("Transcript segment timestamps must satisfy 0 <= start <= end")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Transcript segment confidence must be between 0 and 1")
        if not self.start_timecode:
            self.start_timecode = format_timestamp(self.start_sec)
        if not self.end_timecode:
            self.end_timecode = format_timestamp(self.end_sec)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["suspect"] is None:
            del d["suspect"]
        return d

@dataclass
class TranscriptResult:
    source_path: str
    language: str
    duration_sec: float
    segment_count: int
    segments: List[TranscriptSegment]
    has_speech: bool = True
    notice: str = "Note: Non-speech audio (music, sound effects, alarms) is NOT transcribed."
    confidence: Optional[float] = None
    uncertainty: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["segments"] = [s.to_dict() if isinstance(s, TranscriptSegment) else s for s in self.segments]
        return d

@dataclass
class SearchResult:
    segment_id: int
    start_sec: float
    end_sec: float
    text: str
    score: float
    start_timecode: str
    end_timecode: str
    context: str = ""
    suspect: Optional[str] = None
    match: Optional[str] = None  # hybrid search only: "exact", "all_terms" or "semantic"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for optional in ("suspect", "match"):
            if d[optional] is None:
                del d[optional]
        return d

@dataclass
class ContactSheetResult:
    file_path: str
    sample_count: int
    rows: int
    cols: int
    start_sec: float
    end_sec: float
    timestamps: List[float]
    frame_width: int
    frame_height: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EvidenceItem:
    timestamp_start: float
    timestamp_end: float
    timecode_start: str
    timecode_end: str
    frame_paths: List[str] = field(default_factory=list)
    transcript_quote: Optional[str] = None
    visual_description: Optional[str] = None
    confidence: float = 1.0
    evidence_basis: str = ""  # Observable data grounding this item (e.g. 'frames 120-135 + transcript span 00:14-00:16')
    selection_reason: str = ""  # Short declarative why-this-window-was-chosen, factual not inferential
    provenance: str = ""  # Where it came from: tool name, file, cache fingerprint
    retrieval_method: str = ""  # How it was retrieved (e.g. transcript_search, cut_scan, window_inspection)
    
    def __post_init__(self):
        if not all(math.isfinite(float(value)) for value in (self.timestamp_start, self.timestamp_end)):
            raise ValueError("Evidence timestamps must be finite")
        if self.timestamp_start < 0 or self.timestamp_end < self.timestamp_start:
            raise ValueError("Evidence timestamps must satisfy 0 <= start <= end")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Evidence confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class InspectionWindowResult:
    source_path: str
    start_sec: float
    end_sec: float
    fps: float
    frames: List[FrameInfo]
    contact_sheet_path: Optional[str] = None
    estimated_vision_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["frames"] = [f.to_dict() if isinstance(f, FrameInfo) else f for f in self.frames]
        return d
