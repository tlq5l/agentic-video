"""Text+Vision Model Adapter for Agentic Video.

Connects ordinary Vision-Language Models (Gemini Flash, Claude 3.5 Sonnet, GPT-4o)
to video understanding without requiring native video ingestion.
"""
from typing import Dict, Any, List, Optional
from src.retrieval.engine import AgenticVideoEngine
from src.core.types import EvidenceItem, SearchResult

class AgenticVideoModelAdapter:
    """Orchestrates programmatic video reasoning for standard Text+Vision models."""
    
    def __init__(self, source_path: str):
        self.engine = AgenticVideoEngine(source_path)

    def formulate_inspection_plan(self, question: str) -> Dict[str, Any]:
        """Determine whether query is transcript-driven or visual-driven."""
        meta = self.engine.probe()
        
        # Check transcript if speech is present
        transcript_matches = []
        transcription_status = {}
        if meta.has_audio:
            try:
                transcript_matches = self.engine.search_transcript(question, top_k=3)
                if isinstance(transcript_matches, dict):
                    if transcript_matches.get("transcription_status") in {"started", "in_progress", "busy", "failed", "start_failed"}:
                        transcription_status = {
                            key: transcript_matches[key]
                            for key in (
                                "transcription_status", "retry_after_seconds",
                                "requested_model_size", "cached_model_size",
                                "requested_model_cached", "transcription_message",
                            )
                            if key in transcript_matches
                        }
                        transcription_status["status"] = transcript_matches.get(
                            "status", transcript_matches["transcription_status"]
                        )
                    if transcription_status and not transcript_matches.get("hits"):
                        return {
                            **transcription_status,
                            "retry_after_seconds": transcription_status.get("retry_after_seconds", 0),
                            "transcription_message": transcription_status.get("transcription_message", ""),
                            "is_transcript_driven": False,
                            "transcript_matches": [],
                            "candidate_windows": [],
                            "metadata": meta.to_dict(),
                        }
                    transcript_matches = [
                        SearchResult(**match) for match in transcript_matches.get("hits", [])
                    ]
            except Exception:
                transcript_matches = []
                
        is_transcript_driven = len(transcript_matches) > 0
        candidate_windows = []
        
        if is_transcript_driven:
            for match in transcript_matches:
                # Pad window around speech timestamp
                w_start = max(0.0, match.start_sec - 1.5)
                w_end = min(meta.duration_sec, match.end_sec + 2.0)
                candidate_windows.append((w_start, w_end, f"Transcript match: '{match.text}'"))
        else:
            # Visual-driven: coarse overview across video
            candidate_windows.append((0.0, meta.duration_sec, "Full video visual search"))
            
        return {
            "is_transcript_driven": is_transcript_driven,
            "transcript_matches": [m.to_dict() for m in transcript_matches],
            "candidate_windows": candidate_windows,
            "metadata": meta.to_dict(),
            **transcription_status,
        }

    def execute_agentic_workflow(
        self,
        question: str,
        dense_fps: float = 4.0
    ) -> Dict[str, Any]:
        """Run the complete agentic video understanding pipeline."""
        plan = self.formulate_inspection_plan(question)
        if "transcription_status" in plan:
            return plan
        meta = self.engine.probe()
        
        inspected_frames = []
        contact_sheets = []
        transcript_quotes = []
        
        # 1. Overview sheet
        overview = self.engine.overview(samples=8)
        contact_sheets.append(overview.file_path)
        
        # 2. Targeted window zoom
        if plan["is_transcript_driven"] and plan["candidate_windows"]:
            # Dense inspect around highest-scoring transcript hit
            w_start, w_end, reason = plan["candidate_windows"][0]
            window_res = self.engine.inspect_window(start_sec=w_start, end_sec=w_end, fps=dense_fps)
            inspected_frames.extend([f.file_path for f in window_res.frames])
            if window_res.contact_sheet_path:
                contact_sheets.append(window_res.contact_sheet_path)
            if plan["transcript_matches"]:
                transcript_quotes.append(plan["transcript_matches"][0]["text"])
        else:
            # Coarse-to-fine: detect scenes or use overview
            scenes = self.engine.detect_cuts(threshold=0.25)
            if scenes:
                # Sample window around first prominent scene cut
                cut_ts = scenes[0].timestamp_sec
                w_start = max(0.0, cut_ts - 2.0)
                w_end = min(meta.duration_sec, cut_ts + 2.0)
                window_res = self.engine.inspect_window(start_sec=w_start, end_sec=w_end, fps=dense_fps)
                inspected_frames.extend([f.file_path for f in window_res.frames])
                if window_res.contact_sheet_path:
                    contact_sheets.append(window_res.contact_sheet_path)

        total_frames = len(inspected_frames) + 8  # 8 from overview
        vision_tokens = total_frames * 258
        
        return {
            "question": question,
            "source_path": meta.source_path,
            "duration_sec": meta.duration_sec,
            "plan": plan,
            "frames_inspected_count": total_frames,
            "estimated_vision_tokens": vision_tokens,
            "contact_sheets": contact_sheets,
            "detail_frames": inspected_frames[:12], # top bounded frames
            "transcript_evidence": transcript_quotes
        }
