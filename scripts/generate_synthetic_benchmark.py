"""
Deterministic Synthetic Benchmark Video Generator for Agentic Video Understanding.

Generates:
1. synthetic_bench_60s.mp4: A 60-second video with exact millisecond timecodes,
   clear visual events, state changes, audio tones/chirps, and subtitles.
2. synthetic_bench_60s.vtt: WebVTT transcript file aligned with audio events.
3. synthetic_bench_60s.srt: SubRip subtitle file aligned with audio events.
4. synthetic_ground_truth.json: Ground-truth metadata defining answers for all 6 qualitative tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


@dataclass
class EventInterval:
    event_id: str
    task_type: str
    name: str
    start_s: float
    end_s: float
    description: str
    visual_text: str
    visual_color: str
    audio_description: str
    ground_truth_answer: Any


GROUND_TRUTH_EVENTS: List[EventInterval] = [
    EventInterval(
        event_id="task1_supernova",
        task_type="locate_single_event",
        name="SUPERNOVA_FIRST_OCCURRENCE",
        start_s=14.0,
        end_s=16.0,
        description="First occurrence of the Supernova visual explosion event",
        visual_text="EVENT: SUPERNOVA (STAR BURST)",
        visual_color="#f59e0b",  # Amber/gold
        audio_description="Ascending high chime tone (880Hz)",
        ground_truth_answer={"timestamp_s": 14.0, "event_name": "SUPERNOVA"},
    ),
    EventInterval(
        event_id="task2_seq_pre",
        task_type="temporal_sequence",
        name="PHASE_PRE_BLUE_SQUARE",
        start_s=24.0,
        end_s=25.5,
        description="Preceding event: Blue square phase",
        visual_text="SEQ-1: BLUE SQUARE",
        visual_color="#3b82f6",  # Blue
        audio_description="Low pulse tone (220Hz)",
        ground_truth_answer={"phase": "PRE", "shape": "BLUE SQUARE", "timestamp_s": 24.0},
    ),
    EventInterval(
        event_id="task2_seq_target",
        task_type="temporal_sequence",
        name="PHASE_TARGET_GREEN_CIRCLE",
        start_s=26.0,
        end_s=27.5,
        description="Target reference event: Green circle phase",
        visual_text="SEQ-2: GREEN CIRCLE [TARGET]",
        visual_color="#10b981",  # Green
        audio_description="Mid pulse tone (440Hz)",
        ground_truth_answer={"phase": "TARGET", "shape": "GREEN CIRCLE", "timestamp_s": 26.0},
    ),
    EventInterval(
        event_id="task2_seq_post",
        task_type="temporal_sequence",
        name="PHASE_POST_RED_TRIANGLE",
        start_s=28.0,
        end_s=29.5,
        description="Succeeding event: Red triangle phase",
        visual_text="SEQ-3: RED TRIANGLE",
        visual_color="#ef4444",  # Red
        audio_description="High pulse tone (660Hz)",
        ground_truth_answer={"phase": "POST", "shape": "RED TRIANGLE", "timestamp_s": 28.0},
    ),
    EventInterval(
        event_id="task3_transcript_verification",
        task_type="transcript_driven_verification",
        name="PROJECT_CYGNUS_VERIFICATION",
        start_s=34.0,
        end_s=36.5,
        description="Spoken audio mentions 'Project Cygnus' and visual badge reveals verification code",
        visual_text="PROJECT CYGNUS: CODE [CYGNUS-VERIFIED-77]",
        visual_color="#06b6d4",  # Cyan
        audio_description="Spoken phrase / tone burst: Project Cygnus activated",
        ground_truth_answer={
            "transcript_phrase": "Project Cygnus activated",
            "timestamp_s": 34.0,
            "visual_verification_code": "CYGNUS-VERIFIED-77",
        },
    ),
    EventInterval(
        event_id="task4_visual_state_change",
        task_type="visual_only_state_change",
        name="BACKGROUND_CRIMSON_FLIP",
        start_s=39.5,  # Sub-second boundary!
        end_s=41.5,
        description="Abrupt visual-only background state flip from slate navy to bright crimson (no audio cue)",
        visual_text="STATE CHANGE: CRIMSON ALERT (SUB-SECOND ONSET)",
        visual_color="#e11d48",  # Rose/Crimson
        audio_description="Silent (no audio cue, purely visual event)",
        ground_truth_answer={
            "state_change": "slate_to_crimson",
            "onset_timestamp_s": 39.5,
            "duration_s": 2.0,
        },
    ),
    EventInterval(
        event_id="task5_recurrence_1",
        task_type="sparse_recurring_event",
        name="PURPLE_DIAMOND_OCCURRENCE_1",
        start_s=7.0,
        end_s=8.0,
        description="First occurrence of sparse flashing purple diamond",
        visual_text="BADGE: PURPLE DIAMOND [1 of 3]",
        visual_color="#a855f7",  # Purple
        audio_description="Dual chirp tone (520Hz + 780Hz)",
        ground_truth_answer={"occurrence_index": 1, "timestamp_s": 7.0},
    ),
    EventInterval(
        event_id="task5_recurrence_2",
        task_type="sparse_recurring_event",
        name="PURPLE_DIAMOND_OCCURRENCE_2",
        start_s=21.0,
        end_s=22.0,
        description="Second occurrence of sparse flashing purple diamond",
        visual_text="BADGE: PURPLE DIAMOND [2 of 3]",
        visual_color="#a855f7",
        audio_description="Dual chirp tone (520Hz + 780Hz)",
        ground_truth_answer={"occurrence_index": 2, "timestamp_s": 21.0},
    ),
    EventInterval(
        event_id="task5_recurrence_3",
        task_type="sparse_recurring_event",
        name="PURPLE_DIAMOND_OCCURRENCE_3",
        start_s=48.0,
        end_s=49.0,
        description="Third occurrence of sparse flashing purple diamond",
        visual_text="BADGE: PURPLE DIAMOND [3 of 3]",
        visual_color="#a855f7",
        audio_description="Dual chirp tone (520Hz + 780Hz)",
        ground_truth_answer={"occurrence_index": 3, "timestamp_s": 48.0},
    ),
]

NEGATIVE_CONTROL_TASKS = [
    {
        "event_id": "task6_negative_emerald_elephant",
        "task_type": "negative_control",
        "query": "Does an EMERALD ELEPHANT or CODE_ZEBRA appear in the video?",
        "ground_truth_found": False,
        "expected_verdict": "NOT_FOUND",
        "rationale": "Event was never rendered in the video. Model must reject with not found.",
    },
    {
        "event_id": "task6_negative_audio_explosion",
        "task_type": "negative_control",
        "query": "At what timestamp does the explosion sound effect occur?",
        "ground_truth_found": False,
        "expected_verdict": "NOT_FOUND",
        "rationale": "Audio contains only tones/beeps/chirps and synthetic voice, never an explosion.",
    },
]


def generate_audio_wav(
    output_wav_path: Path,
    duration_s: float = 60.0,
    sample_rate: int = 44100,
) -> None:
    """Generate deterministic audio waveform with exact frequency markers matching events."""
    num_samples = int(duration_s * sample_rate)
    audio_data = bytearray()

    for i in range(num_samples):
        t = i / sample_rate
        val = 0.0

        # Ambient low reference tone
        val += 0.03 * math.sin(2.0 * math.pi * 100.0 * t)

        # Task 1: 14.0s - 16.0s (880Hz chime)
        if 14.0 <= t <= 16.0:
            val += 0.4 * math.sin(2.0 * math.pi * 880.0 * t) * math.exp(-2.0 * ((t - 14.0) % 0.5))

        # Task 2: Sequence tones
        elif 24.0 <= t <= 25.5:
            val += 0.35 * math.sin(2.0 * math.pi * 220.0 * t)
        elif 26.0 <= t <= 27.5:
            val += 0.35 * math.sin(2.0 * math.pi * 440.0 * t)
        elif 28.0 <= t <= 29.5:
            val += 0.35 * math.sin(2.0 * math.pi * 660.0 * t)

        # Task 3: 34.0s - 36.5s (Dual chord for Cygnus)
        elif 34.0 <= t <= 36.5:
            val += 0.25 * math.sin(2.0 * math.pi * 587.33 * t) + 0.25 * math.sin(2.0 * math.pi * 880.0 * t)

        # Task 4: Visual-only at 39.5 - 41.5s: purely ambient hum, no event sound!

        # Task 5: Recurring chirps at 7.0-8.0, 21.0-22.0, 48.0-49.0
        elif (7.0 <= t <= 8.0) or (21.0 <= t <= 22.0) or (48.0 <= t <= 49.0):
            freq = 520.0 + 260.0 * math.sin(2.0 * math.pi * 5.0 * t)
            val += 0.4 * math.sin(2.0 * math.pi * freq * t)

        # Clamp and convert to 16-bit PCM mono
        val = max(-1.0, min(1.0, val))
        sample_int = int(val * 32767.0)
        audio_data.extend(struct.pack("<h", sample_int))

    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


def get_active_event(t: float) -> EventInterval | None:
    for ev in GROUND_TRUTH_EVENTS:
        if ev.start_s <= t < ev.end_s:
            return ev
    return None


def render_benchmark_frame(
    t: float,
    frame_idx: int,
    width: int = 640,
    height: int = 360,
    fps: int = 25,
) -> Image.Image:
    """Render a single high-fidelity frame with deterministic visual content."""
    if 39.5 <= t < 41.5:
        bg_color = (136, 19, 55)  # Crimson Alert
    else:
        bg_color = (15, 23, 42)  # Dark slate navy (#0f172a)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Grid background lines for spatial reference
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(30, 41, 59), width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(30, 41, 59), width=1)

    # Header bar
    draw.rectangle([(0, 0), (width, 36)], fill=(30, 41, 59))
    header_text = "AGENTIC VIDEO BENCHMARK SUITE | DETERMINISTIC V1"
    draw.text((12, 10), header_text, fill=(203, 213, 225))

    # Exact Clock Overlay (HH:MM:SS.mmm) & Frame Counter at Top Right
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis = int((t - math.floor(t)) * 1000)
    clock_str = f"TIME: {minutes:02d}:{seconds:02d}.{millis:03d} | F#{frame_idx:04d}"

    clock_w = 230
    draw.rectangle([(width - clock_w - 8, 4), (width - 8, 32)], fill=(0, 0, 0))
    draw.rectangle([(width - clock_w - 8, 4), (width - 8, 32)], outline=(59, 130, 246), width=1)
    draw.text((width - clock_w, 10), clock_str, fill=(56, 189, 248))

    # Active Event Detection
    active_ev = get_active_event(t)

    # Center Stage Area
    center_box = [(40, 50), (width - 40, height - 60)]
    draw.rectangle(center_box, outline=(51, 65, 85), width=2)

    if active_ev is not None:
        color_hex = active_ev.visual_color.lstrip("#")
        r, g, b = tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))

        # Event Banner
        draw.rectangle([(50, 60), (width - 50, 110)], fill=(r, g, b))
        draw.text((60, 75), active_ev.visual_text, fill=(255, 255, 255))

        # Distinctive Shape based on event type
        if active_ev.task_type == "locate_single_event":
            # Starburst / Gold diamond
            cx, cy = width // 2, 200
            sz = 45
            points = [
                (cx, cy - sz),
                (cx + sz // 3, cy - sz // 3),
                (cx + sz, cy),
                (cx + sz // 3, cy + sz // 3),
                (cx, cy + sz),
                (cx - sz // 3, cy + sz // 3),
                (cx - sz, cy),
                (cx - sz // 3, cy - sz // 3),
            ]
            draw.polygon(points, fill=(245, 158, 11), outline=(255, 255, 255))
            draw.text((cx - 70, cy + 55), "TARGET: SUPERNOVA", fill=(253, 224, 71))

        elif active_ev.name == "PHASE_PRE_BLUE_SQUARE":
            cx, cy = width // 2, 200
            draw.rectangle([(cx - 40, cy - 40), (cx + 40, cy + 40)], fill=(59, 130, 246), outline=(255, 255, 255), width=2)
            draw.text((cx - 45, cy + 50), "PRE-EVENT: BLUE SQUARE", fill=(147, 197, 253))

        elif active_ev.name == "PHASE_TARGET_GREEN_CIRCLE":
            cx, cy = width // 2, 200
            draw.ellipse([(cx - 45, cy - 45), (cx + 45, cy + 45)], fill=(16, 185, 129), outline=(255, 255, 255), width=2)
            draw.text((cx - 60, cy + 55), "TARGET: GREEN CIRCLE", fill=(110, 231, 183))

        elif active_ev.name == "PHASE_POST_RED_TRIANGLE":
            cx, cy = width // 2, 200
            points = [(cx, cy - 45), (cx + 50, cy + 45), (cx - 50, cy + 45)]
            draw.polygon(points, fill=(239, 68, 68), outline=(255, 255, 255))
            draw.text((cx - 50, cy + 55), "POST-EVENT: RED TRIANGLE", fill=(252, 165, 165))

        elif active_ev.task_type == "transcript_driven_verification":
            cx, cy = width // 2, 195
            draw.rectangle([(cx - 140, cy - 35), (cx + 140, cy + 35)], fill=(8, 145, 178), outline=(255, 255, 255), width=2)
            draw.text((cx - 120, cy - 20), "PROJECT CYGNUS SECURITY GATE", fill=(255, 255, 255))
            draw.text((cx - 95, cy + 5), "TOKEN: CYGNUS-VERIFIED-77", fill=(254, 240, 138))

        elif active_ev.task_type == "sparse_recurring_event":
            cx, cy = width // 2, 200
            sz = 40
            points = [(cx, cy - sz), (cx + sz, cy), (cx, cy + sz), (cx - sz, cy)]
            draw.polygon(points, fill=(168, 85, 247), outline=(255, 255, 255), width=2)
            draw.text((cx - 50, cy + 50), "PURPLE DIAMOND", fill=(216, 180, 254))

        elif active_ev.task_type == "visual_only_state_change":
            cx, cy = width // 2, 200
            draw.text((cx - 110, cy - 20), "CRIMSON STATE ACTIVE", fill=(255, 255, 255))
            draw.text((cx - 90, cy + 10), "SUB-SECOND ONSET: 39.500s", fill=(254, 202, 202))
    else:
        cx, cy = width // 2, 170
        draw.ellipse([(cx - 20, cy - 20), (cx + 20, cy + 20)], fill=(30, 41, 59), outline=(71, 85, 105))
        draw.text((cx - 45, cy + 35), "STATE: IDLE SCAN", fill=(100, 116, 139))

    # Bottom Audio Status / Subtitle Bar
    draw.rectangle([(0, height - 32), (width, height)], fill=(15, 23, 42))
    draw.line([(0, height - 32), (width, height - 32)], fill=(51, 65, 85))
    if active_ev is not None and active_ev.audio_description != "Silent":
        audio_caption = f"AUDIO STREAM: {active_ev.audio_description}"
        caption_color = (250, 204, 21)
    else:
        audio_caption = "AUDIO STREAM: [Ambient Reference Tone 100Hz]"
        caption_color = (148, 163, 184)
    draw.text((12, height - 24), audio_caption, fill=caption_color)

    return img


def generate_subtitles(output_dir: Path) -> None:
    """Generate VTT and SRT files aligned with benchmark events."""
    vtt_lines = ["WEBVTT", ""]
    srt_lines = []
    srt_idx = 1

    transcript_events = [
        (0.0, 5.0, "System startup. Deterministic video understanding benchmark initialized."),
        (7.0, 8.0, "Notice: Periodic beacon flash one."),
        (14.0, 16.0, "Alert: Supernova optical flash detected on main sensor."),
        (21.0, 22.0, "Notice: Periodic beacon flash two."),
        (24.0, 25.5, "Sequence step alpha: Blue square appearing."),
        (26.0, 27.5, "Sequence step bravo: Green circle appearing."),
        (28.0, 29.5, "Sequence step charlie: Red triangle appearing."),
        (34.0, 36.5, "Project Cygnus activated. Visual verification token is displayed on screen."),
        (48.0, 49.0, "Notice: Periodic beacon flash three."),
        (55.0, 60.0, "Benchmark run complete. Finalizing media telemetry."),
    ]

    for start_s, end_s, text in transcript_events:
        vtt_start = f"{int(start_s//3600):02d}:{int((start_s%3600)//60):02d}:{start_s%60:06.3f}"
        vtt_end = f"{int(end_s//3600):02d}:{int((end_s%3600)//60):02d}:{end_s%60:06.3f}"
        vtt_lines.append(f"{vtt_start} --> {vtt_end}")
        vtt_lines.append(text)
        vtt_lines.append("")

        srt_start = vtt_start.replace(".", ",")
        srt_end = vtt_end.replace(".", ",")
        srt_lines.append(str(srt_idx))
        srt_lines.append(f"{srt_start} --> {srt_end}")
        srt_lines.append(text)
        srt_lines.append("")
        srt_idx += 1

    (output_dir / "synthetic_bench_60s.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")
    (output_dir / "synthetic_bench_60s.srt").write_text("\n".join(srt_lines), encoding="utf-8")


def generate_benchmark_suite(
    output_dir: Path,
    duration_s: float = 60.0,
    fps: int = 25,
    width: int = 640,
    height: int = 360,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / "temp_audio.wav"
    mp4_path = output_dir / "synthetic_bench_60s.mp4"
    json_path = output_dir / "synthetic_ground_truth.json"

    print(f"[1/4] Generating synthetic audio ({duration_s}s)...")
    generate_audio_wav(wav_path, duration_s=duration_s)

    print(f"[2/4] Generating subtitles (VTT & SRT)...")
    generate_subtitles(output_dir)

    print(f"[3/4] Rendering video frames ({int(duration_s * fps)} frames @ {fps}fps) via FFmpeg pipe...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(wav_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(mp4_path),
    ]

    total_frames = int(duration_s * fps)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    assert proc.stdin is not None
    for f_idx in range(total_frames):
        t = f_idx / fps
        frame = render_benchmark_frame(t, f_idx, width=width, height=height, fps=fps)
        proc.stdin.write(frame.tobytes())

    # communicate() closes stdin itself; closing it first makes it raise on Python 3.12+.
    _, stderr_data = proc.communicate()

    if proc.returncode != 0:
        stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
        raise RuntimeError(f"FFmpeg encoding failed with code {proc.returncode}: {stderr}")

    # Remove temporary wav
    try:
        if wav_path.exists():
            wav_path.unlink()
    except OSError:
        pass

    print(f"[4/4] Writing ground-truth metadata to {json_path}...")
    ground_truth_dict = {
        "benchmark_name": "synthetic_bench_60s",
        "duration_s": duration_s,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "events": [asdict(ev) for ev in GROUND_TRUTH_EVENTS],
        "negative_controls": NEGATIVE_CONTROL_TASKS,
        "tasks": {
            "task1_locate_single_event": {
                "task_type": "locate_single_event",
                "name": "Locate Supernova Event",
                "prompt": "When does the SUPERNOVA event first occur in the video?",
                "target_event_id": "task1_supernova",
                "ground_truth_timestamp_s": 14.0,
                "timestamp_tolerance_s": 1.0,
                "expected_keywords": ["supernova", "14", "star"],
            },
            "task2_temporal_sequence": {
                "task_type": "temporal_sequence",
                "name": "Sequence Before and After Green Circle",
                "prompt": "What visual shape appears immediately before and immediately after the GREEN CIRCLE event?",
                "target_event_id": "task2_seq_target",
                "ground_truth": {
                    "before": {"shape": "BLUE SQUARE", "timestamp_s": 24.0},
                    "target": {"shape": "GREEN CIRCLE", "timestamp_s": 26.0},
                    "after": {"shape": "RED TRIANGLE", "timestamp_s": 28.0},
                },
                "expected_keywords": ["blue square", "red triangle"],
            },
            "task3_transcript_verification": {
                "task_type": "transcript_driven_verification",
                "name": "Verify Project Cygnus Token",
                "prompt": "Where is 'Project Cygnus' mentioned in audio/transcript, and what visual verification token is shown on screen?",
                "target_event_id": "task3_transcript_verification",
                "ground_truth": {
                    "mention_timestamp_s": 34.0,
                    "visual_token": "CYGNUS-VERIFIED-77",
                },
                "timestamp_tolerance_s": 1.5,
                "expected_keywords": ["CYGNUS-VERIFIED-77", "34"],
            },
            "task4_visual_state_change": {
                "task_type": "visual_only_state_change",
                "name": "Sub-second Crimson Background Onset",
                "prompt": "At what exact timestamp does the background color flip to crimson alert?",
                "target_event_id": "task4_visual_state_change",
                "ground_truth_timestamp_s": 39.5,
                "timestamp_tolerance_s": 0.6,
                "expected_keywords": ["39.5", "crimson"],
            },
            "task5_sparse_recurring_event": {
                "task_type": "sparse_recurring_event",
                "name": "Count Purple Diamond Badges",
                "prompt": "How many times does the flashing PURPLE DIAMOND badge appear in the video, and at what timestamps?",
                "target_event_ids": ["task5_recurrence_1", "task5_recurrence_2", "task5_recurrence_3"],
                "ground_truth_count": 3,
                "ground_truth_timestamps_s": [7.0, 21.0, 48.0],
                "timestamp_tolerance_s": 1.0,
            },
            "task6_negative_control": {
                "task_type": "negative_control",
                "name": "Negative Control: Emerald Elephant",
                "prompt": "Does an EMERALD ELEPHANT or CODE_ZEBRA occur at any point in the video?",
                "ground_truth_present": False,
                "expected_verdict": "NOT_FOUND",
                "acceptable_answers": ["not found", "does not occur", "never appears", "absent", "none", "no"],
            },
        },
    }

    json_path.write_text(json.dumps(ground_truth_dict, indent=2), encoding="utf-8")
    print(f"Generated synthetic benchmark suite successfully in {output_dir}")

    return {
        "mp4": mp4_path,
        "vtt": output_dir / "synthetic_bench_60s.vtt",
        "srt": output_dir / "synthetic_bench_60s.srt",
        "json": json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Synthetic Video Benchmark Generator")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "fixtures"), help="Output directory")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate")
    args = parser.parse_args()

    generate_benchmark_suite(Path(args.output_dir), duration_s=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()
