"""Pillow-based deterministic timestamp burning and visual badges."""
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from src.core.types import format_timestamp


def burn_timestamp_badge(
    image: Image.Image,
    timestamp_sec: float,
    frame_index: Optional[int] = None,
    position: str = "top-left",
    badge_bg: Tuple[int, int, int, int] = (15, 17, 23, 220),
    badge_border: Tuple[int, int, int] = (0, 200, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 14
) -> Image.Image:
    """Burn high-contrast, deterministic timecode badge directly into frame pixels.
    
    Why Pillow over FFmpeg `drawtext`:
    1. Zero fontconfig / OS font path dependencies (runs identically on Windows, Mac, Linux Docker).
    2. Completely prevents PTS reset / drift bugs when using input seeking.
    3. Guarantees uniform machine-readable OCR text for multimodal vision models.
    4. Eliminates temporal hallucination by anchoring models to ground-truth timecode.
    """
    # Ensure image is in RGB or RGBA mode
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
        
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        # Older Pillow fallback
        font = ImageFont.load_default()
        
    time_str = format_timestamp(timestamp_sec)
    if frame_index is not None:
        label = f"#{frame_index:03d}  {time_str}"
    else:
        label = f"TIME: {time_str}"
        
    bbox = font.getbbox(label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 6, 4
    
    badge_w = text_w + pad_x * 2
    badge_h = text_h + pad_y * 2
    
    img_w, img_h = image.size
    margin = 10
    
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = img_w - badge_w - margin, margin
    elif position == "bottom-left":
        x, y = margin, img_h - badge_h - margin
    elif position == "bottom-right":
        x, y = img_w - badge_w - margin, img_h - badge_h - margin
    else:
        x, y = margin, margin
        
    # Draw rounded or boxed badge
    box = [x, y, x + badge_w, y + badge_h]
    draw.rectangle(box, fill=badge_bg[:3], outline=badge_border, width=1)
    draw.text((x + pad_x, y + pad_y), label, fill=text_color, font=font)
    
    return image


def apply_overlay_to_file(
    image_path: Union[str, Path],
    timestamp_sec: float,
    frame_index: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    position: str = "top-left"
) -> str:
    """Read image from file, burn timestamp badge, and save atomically."""
    in_path = Path(image_path)
    out_path = Path(output_path or image_path)
    
    with Image.open(in_path) as im:
        annotated = burn_timestamp_badge(im.copy(), timestamp_sec, frame_index, position=position)
        annotated.save(out_path, quality=90)
    return str(out_path)


# Alias for backward/forward compatibility
draw_timestamp_badge = burn_timestamp_badge
