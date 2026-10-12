"""Media processing and frame extraction engine for agentic video understanding."""
from src.media.probe import probe_video
from src.media.seeking import extract_single_frame, build_seek_arguments
from src.media.sampling import sample_sparse_frames, sample_dense_window
from src.media.overlay import burn_timestamp_badge, apply_overlay_to_file
from src.media.contact_sheet import compose_contact_sheet, generate_ffmpeg_tile_sheet
from src.media.scene import detect_scenes, partition_into_scene_segments, SceneCut, SceneSegment

__all__ = [
    "probe_video",
    "extract_single_frame",
    "build_seek_arguments",
    "sample_sparse_frames",
    "sample_dense_window",
    "burn_timestamp_badge",
    "apply_overlay_to_file",
    "compose_contact_sheet",
    "generate_ffmpeg_tile_sheet",
    "detect_scenes",
    "partition_into_scene_segments",
    "SceneCut",
    "SceneSegment",
]
