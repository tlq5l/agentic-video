"""Generated-only, blind synthetic repetition clips. No checked-in video data."""
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw


CASES = [
    # name, kind, count, period, drift, pause_after, noise, distractor
    ("ball_fast", "ball", 12, .45, 0, -1, 0, False),
    ("ball_slow", "ball", 5, 1.5, 0, -1, 0, False),
    ("ball_drift", "ball", 8, .6, .5, -1, .02, False),
    ("pendulum", "pendulum", 7, .8, 0, -1, 0, False),
    ("pendulum_pause", "pendulum", 6, 1.1, 0, 2, 0, False),
    ("pendulum_slow", "pendulum", 3, 2.5, 0, -1, .01, True),
    ("ball_very_slow", "ball", 2, 3.0, 0, -1, 0, False),
    ("pulse", "pulse", 20, .4, 0, -1, 0, False),
    ("pulse_drift", "pulse", 9, .55, .6, -1, .01, False),
    ("pulse_bg", "pulse", 4, 1.7, 0, -1, .02, True),
    ("figure", "figure", 10, .65, 0, -1, 0, False),
    ("figure_pause", "figure", 5, 1.2, -.2, 2, .02, True),
    ("no_repetition", "none", 0, 1, 0, -1, .02, True),
    # Cropped windows begin and end inside a cycle. The truth excludes both.
    ("ball_edge", "ball", 6, .8, 0, -1, 0, False, (.7, .7)),
    ("pendulum_edge", "pendulum", 5, 1.0, 0, -1, 0, False, (.7, .7)),
]


def generate(case, output: Path, fps=10):
    name, kind, count, period, drift, pause_after, noise, distractor = case[:8]
    crop_start, crop_end = case[8] if len(case) > 8 else (0, 0)
    periods = np.linspace(period, period*(1+drift), count) if count else []
    starts = []
    cursor = .5
    for i, span in enumerate(periods):
        starts.append(cursor)
        cursor += float(span)
        if i == pause_after:
            cursor += 1.3
    duration = cursor + .5 if count else 5.0
    rng = np.random.default_rng(16)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-threads", "2", "-f", "rawvideo",
                             "-pixel_format", "rgb24", "-video_size", "96x72", "-framerate", str(fps),
                             "-i", "pipe:0", "-an", "-c:v", "mpeg4", "-q:v", "3", str(output)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame_num in range(int(crop_start*fps), math.ceil((duration-crop_end)*fps)):
            t = frame_num / fps
            arr = np.full((72, 96, 3), 26, dtype=np.uint8)
            if noise:
                jitter = rng.normal(0, noise*255, (72, 96, 1))
                arr = np.clip(arr.astype(float)+jitter, 0, 255).astype(np.uint8)
                arr = np.repeat(arr[:, :, :1], 3, axis=2)
            image = Image.fromarray(arr)
            draw = ImageDraw.Draw(image)
            if distractor:
                dx = 80 + int(4*math.sin(t*2.1))
                draw.rectangle((dx, 4, dx+5, 9), fill=(115, 115, 115))
            active = next(((t-start)/span for start,span in zip(starts,periods) if start <= t <= start+span), None)
            if active is not None:
                phase = 2*math.pi*active
                if kind == "ball":
                    y = 12 + int(42*abs(math.sin(math.pi*active)))
                    draw.ellipse((43,y,55,y+12), fill=(245,245,245))
                elif kind == "pendulum":
                    x = 48 + int(27*math.sin(phase))
                    draw.line((48,7,x,42), fill=(170,170,170), width=2)
                    draw.ellipse((x-6,37,x+6,49), fill=(245,245,245))
                elif kind == "pulse":
                    radius = 6 + int(18*math.sin(math.pi*active)**2)
                    draw.ellipse((48-radius,36-radius,48+radius,36+radius), fill=(245,245,245))
                elif kind == "figure":
                    height = 14 + int(18*math.sin(math.pi*active)**2)
                    draw.ellipse((45,28-height,51,34-height), fill=(230,230,230))
                    draw.rectangle((44,35-height,52,55), fill=(230,230,230))
                    draw.line((44,42-height,35-int(8*math.sin(phase)),52), fill=(230,230,230), width=3)
                    draw.line((52,42-height,61+int(8*math.sin(phase)),52), fill=(230,230,230), width=3)
            elif kind != "none":
                if kind == "ball":
                    draw.ellipse((43,12,55,24), fill=(245,245,245))
                elif kind == "pendulum":
                    draw.line((48,7,48,42), fill=(170,170,170), width=2)
                    draw.ellipse((42,37,54,49), fill=(245,245,245))
                elif kind == "pulse":
                    draw.ellipse((42,30,54,42), fill=(245,245,245))
                elif kind == "figure":
                    draw.ellipse((45,14,51,20), fill=(230,230,230))
                    draw.rectangle((44,21,52,55), fill=(230,230,230))
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        if proc.wait(timeout=20):
            raise RuntimeError(error)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()
    return count - int(crop_start > .5) - int(crop_end > .5)
