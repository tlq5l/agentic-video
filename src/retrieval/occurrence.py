"""Bounded, asynchronous counting of distinct on-screen occurrences of a symbol, object or exact text.

Frames are decoded densely and judged one at a time; a frame "has" the target when
  - `text` is present: Apple Vision OCR finds the phrase as whole words (RELEASE, not RELEASED);
  - `description` names a colour: only pixels of that colour form candidate regions, so
    same-shaped look-alikes in other colours never match (CLIP alone is weak at colour);
  - `description` names a plain geometric shape (circle, square, diamond, ...): the candidate's
    outline geometry decides, because CLIP confuses diamonds with squares;
  - any other description: CLIP scores the candidate crops against contrast prompts.
Consecutive matching frames merge into one span; a gap of `min_gap_sec` or more starts a new one.
A larger `min_gap_sec` joins separate appearances, so the result reports every gap it bridged.
When MAX_SPANS_LIMIT or more spans are found the description is too broad, so the result carries no count.
"""
import collections
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.core.config import Config
from src.core.errors import as_dependency_error
from src.retrieval import ocr

ALGORITHM_VERSION = "occurrences-v7"
MAX_FRAMES = 2400
MAX_SPANS_LIMIT = 100
DEFAULT_MIN_GAP = 0.3
ANALYSIS_WIDTH = 320
DECODE_WIDTH = 1280
READ_STALL_TIMEOUT_SEC = 30.0
OCR_WORKERS = 2
CLIP_BATCH_FRAMES = 8  # frames whose tiles share one CLIP batch
STATIC_PIXEL_DELTA = 24
STATIC_PIXEL_COUNT = 4
CLIP_MIN_PROBABILITY = 0.4
MIN_COLOR_STRENGTH = .2  # saturation x value; dim, muted pixels (sensor noise, a dark tinted background) are not a coloured symbol
MIN_SYMBOL_WIDTH = 0.025  # smaller specks (video noise) are not symbols; fraction of frame width
TELEPORT_JUMP = 0.4  # fraction of the frame a match may move between two samples before it counts as a new appearance
SPAN_NOTE = "Spans are sample-based (+/- 1/fps); verify each representative_sec with inspect_window."
SHEET_TILES = 24
DIRECTIONS = ("any", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")
TILE_NOTE = (" contact_sheet is one image with a tile per listed span, in span order: a zoomed frame of the span's best "
             "match, boxed, labeled with the span number and time. Look at each tile and drop any span whose tile is not "
             "the asked-for thing; spans marked * (`weak`) are the likeliest false ones. The answer is the number of "
             "tiles that match, plus any omitted spans and respecting any merged gaps noted above.")
TILE_SIZE = (288, 192)  # one span's zoomed crop on the sheet
TILE_LABEL_HEIGHT = 24
SHEET_COLS = 4
WEAK_SPAN_SEC = .5  # a span shorter than this, or made of one detection, is weak
WEAK_SCORE_RATIO = .75  # so is one scoring below this share of the median span score
TIED_TILES = 8  # tiles kept per span while later samples may still tie its best score
_jobs = OrderedDict()
_jobs_lock = threading.Lock()
_worker = None
_worker_key = None

# Hue in degrees, then minimum saturation and value/lightness bounds (all 0-1).
_COLORS = {
    "red": ("wrap", (345, 12), .45, .3, 1), "orange": ("hue", (12, 45), .5, .55, 1),
    "yellow": ("hue", (45, 72), .4, .45, 1), "green": ("hue", (72, 165), .35, .25, 1),
    "cyan": ("hue", (165, 200), .35, .3, 1), "blue": ("hue", (200, 260), .35, .25, 1),
    "purple": ("hue", (260, 300), .3, .2, 1), "magenta": ("hue", (300, 345), .4, .3, 1),
    "pink": ("wrap", (300, 12), .12, .7, 1), "brown": ("hue", (10, 45), .4, .15, .55),
    "white": ("any", None, 0, .82, 1), "black": ("any", None, 0, 0, .15),
    "gray": ("any", None, 0, .25, .75),
}
_STRENGTH_COLORS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"}  # pink and brown are pale or dim by nature
_COLOR_WORDS = {"gold": "yellow", "golden": "yellow", "aqua": "cyan", "teal": "cyan", "violet": "purple",
                "grey": "gray", **{name: name for name in _COLORS}}
_GEOMETRIC = {"circle": "circle", "circular": "circle", "disc": "circle", "disk": "circle",
              "ring": "circle", "dot": "circle", "oval": "ellipse", "ellipse": "ellipse",
              "square": "square", "rectangle": "rectangle", "diamond": "diamond",
              "rhombus": "diamond", "triangle": "triangle"}
_OTHER_SHAPES = ["star", "cross", "arrow", "heart", "hexagon", "pentagon"]
_BACKGROUND_PROMPTS = ["an empty background", "a plain colored screen", "some text on a screen",
                       "a person", "an abstract shape", "a photograph of a room"]
_SCOPE_SPLIT = re.compile(r"\b(?:on|against|over|inside|with|behind|near|in front of)\b")


def parse_description(description):
    """Split a free-text description into the parts each check uses."""
    words = re.findall(r"[a-z]+", description.lower())
    scope = _SCOPE_SPLIT.split(description.lower(), maxsplit=1)[0]
    color = next((_COLOR_WORDS[w] for w in re.findall(r"[a-z]+", scope) if w in _COLOR_WORDS), None)
    geometric = next((_GEOMETRIC[w] for w in words if w in _GEOMETRIC), None)
    style = ("outlined" if {"outline", "outlined", "hollow", "unfilled", "ring"} & set(words)
             else "filled" if {"filled", "solid", "dot"} & set(words) else None)
    other = next((w for w in words if w in _OTHER_SHAPES), None)
    stripped = " ".join(w for w in words if _COLOR_WORDS.get(w) != color or color is None)
    return {"color": color, "shape": geometric, "style": style, "other_shape": other, "prompt": stripped}


def color_mask(image, color):
    """Boolean mask of pixels of the named colour; hue tolerances survive video compression.

    Orange ends at 45 degrees so an orange shape's compressed fringe does not read as yellow.
    """
    kind, hues, smin, vmin, vmax = _COLORS[color]
    hsv = np.asarray(image.convert("HSV"))
    hue = hsv[..., 0].astype(np.float32) * 360 / 255
    sat = hsv[..., 1].astype(np.float32) / 255
    val = hsv[..., 2].astype(np.float32) / 255
    if kind == "hue":
        in_hue = (hue >= hues[0]) & (hue < hues[1])
    elif kind == "wrap":
        in_hue = (hue >= hues[0]) | (hue < hues[1])
    else:
        in_hue = np.ones(hue.shape, dtype=bool)
    if color in ("white", "gray", "black"):
        sat_ok = sat <= .18
    elif color == "pink":
        sat_ok = (sat >= smin) & (sat <= .55)
    else:
        sat_ok = sat >= smin
    strong = sat * val >= MIN_COLOR_STRENGTH if color in _STRENGTH_COLORS else True
    return in_hue & sat_ok & (val >= vmin) & (val <= vmax) & strong


def foreground_mask(image):
    """Pixels that differ from the frame's border colour; used when no colour is named."""
    array = np.asarray(image).astype(np.int16)
    border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    background = np.median(border, axis=0)
    return np.abs(array - background).max(axis=-1) > 48


def find_blobs(mask, min_area, min_side=0, max_cover=.6, limit=6):
    """4-connected regions of a mask (joined across 2 px gaps), largest first.

    Returns dicts with a tight `mask` crop, its bbox `box`=(x0,y0,x1,y1) and pixel `area`.
    """
    height, width = mask.shape
    joined = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
    parent, runs, previous = [], [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(height):
        row = joined[y]
        if not row.any():
            previous = []
            continue
        edges = np.diff(np.r_[0, row.view(np.int8), 0])
        current = []
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            index = len(runs)
            runs.append((y, int(start), int(end)))
            parent.append(index)
            current.append(index)
            for other in previous:
                if runs[other][1] < end and start < runs[other][2]:
                    parent[find(index)] = find(other)
        previous = current
        if len(runs) > 20000:  # noise, not symbols
            return []
    groups = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)
    blobs = []
    for members in groups.values():
        x0 = min(r[1] for r in members); x1 = max(r[2] for r in members)
        y0 = min(r[0] for r in members); y1 = max(r[0] for r in members) + 1
        region = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for y, start, end in members:
            region[y - y0, start - x0:end - x0] = True
        ink = region & mask[y0:y1, x0:x1]
        area = int(ink.sum())
        if area < min_area or (x1 - x0) * (y1 - y0) > max_cover * width * height:
            continue
        ys, xs = np.nonzero(ink)
        box = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
        if min(box[2] - box[0], box[3] - box[1]) < min_side:
            continue
        blobs.append({"mask": ink[box[1] - y0:box[3] - y0, box[0] - x0:box[2] - x0], "box": box, "area": area})
    return sorted(blobs, key=lambda b: -b["area"])[:limit]


def _convex_hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def half(seq):
        chain = []
        for p in seq:
            while len(chain) >= 2 and (chain[-1][0]-chain[-2][0])*(p[1]-chain[-2][1]) - (chain[-1][1]-chain[-2][1])*(p[0]-chain[-2][0]) <= 0:
                chain.pop()
            chain.append(p)
        return chain
    lower, upper = half(points), half(reversed(points))
    return lower[:-1] + upper[:-1]


def shape_features(ink):
    """Scale-free outline descriptors of one region's pixels."""
    ys, xs = np.nonzero(ink)
    candidates = []
    for y in np.unique(ys):
        row = xs[ys == y]
        candidates += [(int(row.min()), int(y)), (int(row.max()), int(y))]
    for x in np.unique(xs):
        column = ys[xs == x]
        candidates += [(int(x), int(column.min())), (int(x), int(column.max()))]
    hull = _convex_hull(candidates)
    px, py = np.array([p[0] for p in hull], float), np.array([p[1] for p in hull], float)
    area = abs(float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))) / 2
    perimeter = float(np.hypot(px - np.roll(px, -1), py - np.roll(py, -1)).sum())
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span_x, span_y = max(float(x1 - x0), 1.0), max(float(y1 - y0), 1.0)

    def edge_mid(mask, coords, low, span):
        return float((coords[mask].mean() - low) / span)
    ink_ratio = float(ink.sum()) / max(area, 1.0)
    if ink_ratio >= .85:
        convex = True  # a filled figure with notches (a star) would leave gaps in its hull
    else:
        # An outline of a convex figure hugs its hull; a star's inner strokes sit far inside it.
        stride = max(1, len(xs) // 400)
        points = np.stack([xs[::stride], ys[::stride]], axis=1).astype(float)
        start, edge = np.stack([px, py], axis=1), np.stack([np.roll(px, -1) - px, np.roll(py, -1) - py], axis=1)
        along = np.clip(((points[:, None] - start[None]) * edge[None]).sum(-1) / np.maximum((edge ** 2).sum(-1), 1e-9), 0, 1)
        depth = np.linalg.norm(points[:, None] - (start[None] + along[..., None] * edge[None]), axis=-1).min(axis=1)
        convex = bool(np.percentile(depth, 95) <= 2 * max(ink.sum() / max(perimeter, 1.0), 1.0) + 1.5)
    return {
        "fill": area / (span_x * span_y), "circularity": 4 * math.pi * area / max(perimeter ** 2, 1e-9),
        "aspect": min(span_x, span_y) / max(span_x, span_y),
        "left_y": edge_mid(xs == x0, ys, y0, span_y), "right_y": edge_mid(xs == x1, ys, y0, span_y),
        "top_x": edge_mid(ys == y0, xs, x0, span_x), "bottom_x": edge_mid(ys == y1, xs, x0, span_x),
        "ink": ink_ratio, "convex": convex,
    }


def shape_score(shape, f):
    """0 when the outline is not the named shape; else 0.5-1.0 (higher = closer to ideal)."""
    if not f["convex"]:
        return 0.0
    mids = max(abs(f["left_y"] - .5), abs(f["right_y"] - .5), abs(f["top_x"] - .5), abs(f["bottom_x"] - .5))
    if shape == "circle":
        deviation = max((.955 - f["circularity"]) / .04, (.85 - f["aspect"]) / .15)
    elif shape == "ellipse":
        deviation = max(abs(f["fill"] - .785) / .07, (.7 - f["circularity"]) / .1, (.3 - f["aspect"]) / .1)
    elif shape == "square":
        deviation = max((.9 - f["fill"]) / .08, (.85 - f["aspect"]) / .1)
    elif shape == "rectangle":
        deviation = (.9 - f["fill"]) / .08
    elif shape == "diamond":
        deviation = max(abs(f["fill"] - .5) / .12, (mids - .2) / .1, (.4 - f["aspect"]) / .1)
    else:  # triangle: half-filled box whose base corners sit on one horizontal edge
        base = max(min(f["left_y"], f["right_y"]) - .75, .25 - max(f["left_y"], f["right_y"]))
        deviation = max(abs(f["fill"] - .5) / .12, -base / .15 if base < 0 else 0)
    return 0.0 if deviation >= 1 else .5 + .5 * (1 - max(deviation, 0))


def style_ok(style, features):
    return style is None or (features["ink"] < .6 if style == "outlined" else features["ink"] >= .75)


def _analysis_size(frame):
    """Size of the small copy that colour/shape checks run on and that boxes are normalized against."""
    return ANALYSIS_WIDTH, max(1, round(frame.height * ANALYSIS_WIDTH / frame.width))


class FrameJudge:
    """Decides, per decoded frame, whether the target is visible, and where."""

    def __init__(self, description, text, case_sensitive):
        self.case_sensitive = case_sensitive
        self.pattern = ocr.phrase_pattern(text, case_sensitive) if text else None
        self.spec = parse_description(description) if description else None
        self._clip = None

    def _prepare_clip(self):
        if self._clip is not None:
            return
        from src.retrieval.visual import get_encoder
        encoder = get_encoder()
        spec = self.spec
        prompt = spec["prompt"]
        if spec["other_shape"]:
            negatives = [prompt.replace(spec["other_shape"], other) for other in _OTHER_SHAPES + list(set(_GEOMETRIC.values()))
                         if other != spec["other_shape"]]
        else:
            negatives = list(_BACKGROUND_PROMPTS)
        vectors = np.stack([encoder.text(p) for p in [prompt] + negatives])
        self._clip = (encoder, vectors)

    def _clip_scores(self, crops):
        encoder, vectors = self._clip
        logits = 100 * encoder.pil_images(crops) @ vectors.T
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [float(p[0]) if p.argmax() == 0 else 0.0 for p in probs]

    def judge(self, frame):
        """frame: PIL RGB. Returns None (absent) or {score, box, boxes}; boxes are normalized (x, y, w, h)."""
        found_text = None
        if self.pattern is not None:
            found_text = ocr.find_phrase(ocr.recognize(frame), self.pattern, self.case_sensitive)
            if found_text is None:
                return None
            box = (found_text["x"], found_text["y"], found_text["w"], found_text["h"])
            if self.spec is None:
                return {"score": found_text["confidence"], "box": box, "boxes": [box]}
        seen = self._judge_visual(frame)
        if seen is None or found_text is None:
            return seen
        return {"score": min(seen["score"], found_text["confidence"]), "box": box, "boxes": [box]}

    def _judge_visual(self, frame):
        spec = self.spec
        small = frame.resize(_analysis_size(frame))
        width, height = small.size
        if spec["color"] or spec["shape"]:
            mask = color_mask(small, spec["color"]) if spec["color"] else foreground_mask(small)
            blobs = find_blobs(mask, max(12, int(.0002 * width * height)), min_side=MIN_SYMBOL_WIDTH * width)
            if spec["shape"]:
                matches = []
                for blob in blobs:
                    features = shape_features(blob["mask"])
                    score = shape_score(spec["shape"], features) if style_ok(spec["style"], features) else 0.0
                    if score:
                        matches.append((score, blob["box"]))
            else:
                matches = self._clip_blobs(frame, blobs, width, height)
        else:
            matches = self._clip_tiles(frame)
        return self._hit(matches, width, height)

    @staticmethod
    def _hit(matches, width, height):
        if not matches:
            return None
        boxes = [(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height) for _, (x0, y0, x1, y1) in matches]
        best = max(range(len(matches)), key=lambda i: matches[i][0])
        return {"score": matches[best][0], "box": boxes[best], "boxes": boxes}

    def _clip_blobs(self, frame, blobs, width, height):
        if not blobs:
            return []
        if not self.spec["prompt"]:  # colour alone was described; the colour check is the whole test
            return [(1.0, blob["box"]) for blob in blobs]
        self._prepare_clip()
        crops = []
        for blob in blobs:
            x0, y0, x1, y1 = blob["box"]
            cx, cy = (x0 + x1) / 2 / width * frame.width, (y0 + y1) / 2 / height * frame.height
            half = max(x1 - x0, y1 - y0) / width * frame.width * .7 + 8
            crops.append(frame.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))))
        return [(score, blob["box"]) for score, blob in zip(self._clip_scores(crops), blobs) if score >= CLIP_MIN_PROBABILITY]

    @staticmethod
    def _tile_boxes(frame):
        w, h = frame.size
        return [(0, 0, w, h)] + [(x, y, x + int(w * .6), y + int(h * .6))
                                  for x in (0, int(w * .4)) for y in (0, int(h * .4))]

    @staticmethod
    def _best_tile(scores, tiles, width):
        index = int(np.argmax(scores))
        if scores[index] < CLIP_MIN_PROBABILITY:
            return []
        scale = ANALYSIS_WIDTH / width
        return [(scores[index], tuple(v * scale for v in tiles[index]))]

    def _clip_tiles(self, frame):
        self._prepare_clip()
        tiles = self._tile_boxes(frame)
        return self._best_tile(self._clip_scores([frame.crop(t) for t in tiles]), tiles, frame.width)

    @property
    def batchable(self):
        """Only the whole-frame CLIP check judges frames independently of each other, so frames can share a batch."""
        return self.pattern is None and self.spec is not None and not (self.spec["color"] or self.spec["shape"])

    def judge_many(self, frames):
        """Same results as `judge` per frame, with all frames' tiles embedded in one CLIP batch (cheaper per crop)."""
        self._prepare_clip()
        tile_sets = [self._tile_boxes(frame) for frame in frames]
        scores = self._clip_scores([frame.crop(t) for frame, tiles in zip(frames, tile_sets) for t in tiles])
        results, offset = [], 0
        for frame, tiles in zip(frames, tile_sets):
            matches = self._best_tile(scores[offset:offset + len(tiles)], tiles, frame.width)
            offset += len(tiles)
            results.append(self._hit(matches, *_analysis_size(frame)))
        return results


def _direction(points):
    """Travel of the matched region across a span, from its first and last centres (normalized)."""
    head, tail = np.mean(points[:2], axis=0), np.mean(points[-2:], axis=0)
    dx, dy = tail - head
    if max(abs(dx), abs(dy)) < .15:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "left_to_right" if dx > 0 else "right_to_left"
    return "top_to_bottom" if dy > 0 else "bottom_to_top"


class SpanBuilder:
    """Groups matching samples into spans; keeps a small tile per sample for the first `keep` spans."""

    def __init__(self, step, min_gap_sec, keep):
        self.step, self.min_gap, self.keep = step, min_gap_sec, keep
        self.spans = []
        self.bridged = []  # (start, end) of absences the default min_gap_sec would have split but this one joined

    def add(self, t, hit, frame):
        """`frame` is None when replaying cached judgments; tiles are then filled in for the chosen samples only."""
        if hit is None:
            return
        last = self.spans[-1]["samples"][-1] if self.spans else None
        # Vanishing in one place and reappearing far away within a step (a marker wrapping around) is a new
        # appearance; a second look-alike elsewhere in the frame is not, so compare the nearest match.
        jumped = last is not None and min(max(abs(x + w / 2 - last[2][0] - last[2][2] / 2),
                                              abs(y + h / 2 - last[2][1] - last[2][3] / 2))
                                          for x, y, w, h in hit["boxes"]) > TELEPORT_JUMP
        absence = None if last is None else t - last[0] - self.step
        if last is None or jumped or absence >= self.min_gap - 1e-6:
            self.spans.append({"samples": [], "best": -1.0, "tiles": []})
        elif absence >= DEFAULT_MIN_GAP - 1e-6:
            self.bridged.append((last[0] + self.step, t))
        span = self.spans[-1]
        sample = (t, hit["score"], hit["box"])
        span["samples"].append(sample)
        if len(self.spans) > self.keep:
            return
        # Only samples within .03 of the best score can become the representative, so only those get a tile.
        span["best"] = max(span["best"], sample[1])
        span["tiles"] = [c for c in span["tiles"] if c[1] >= span["best"] - .03]
        if sample[1] >= span["best"] - .03:
            span["tiles"].append(sample + (None if frame is None else _span_tile(frame, hit["box"]),))
            if len(span["tiles"]) > TIED_TILES:
                del span["tiles"][1::2]  # thin evenly so the survivors still spread across the span

    def finish(self):
        """Fill start/end and a representative sample: best score, then nearest the span's middle."""
        for span in self.spans:
            samples = span["samples"]
            span["start"], span["end"] = samples[0][0], samples[-1][0] + self.step
            best = max(s[1] for s in samples)
            middle = (span["start"] + span["end"] - self.step) / 2
            near_best = span["tiles"] or [s + (None,) for s in samples if s[1] >= best - .03]
            span["rep"] = min(near_best, key=lambda s: abs(s[0] - middle))
            span["points"] = [(s[2][0] + s[2][2] / 2, s[2][1] + s[2][3] / 2) for s in samples]
            span["score"] = best
        median = float(np.median([s["score"] for s in self.spans])) if self.spans else 0.0
        for span in self.spans:
            # Weak spans are still counted; the flag only says which tiles to check first.
            span["weak"] = (len(span["samples"]) == 1 or span["end"] - span["start"] < WEAK_SPAN_SEC
                            or span["score"] < WEAK_SCORE_RATIO * median)
        return self.spans


def _span_tile(frame, box):
    """Zoomed crop of the frame around the matched region, boxed in neutral white/black, at the sheet's tile size."""
    width, height = frame.size
    x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    aspect = TILE_SIZE[0] / TILE_SIZE[1]
    crop_w = min(max(2 * w, 4.5 * h, .15 * width), width, height * aspect)  # room around the box to judge colour and shape
    crop_h = crop_w / aspect
    left = min(max(x + w / 2 - crop_w / 2, 0), width - crop_w)
    top = min(max(y + h / 2 - crop_h / 2, 0), height - crop_h)
    tile = frame.crop((round(left), round(top), round(left + crop_w), round(top + crop_h))).resize(TILE_SIZE, Image.BILINEAR)
    scale = TILE_SIZE[0] / crop_w
    rect = [(x - left) * scale, (y - top) * scale, (x + w - left) * scale, (y + h - top) * scale]
    draw = ImageDraw.Draw(tile)
    draw.rectangle([rect[0] - 2, rect[1] - 2, rect[2] + 2, rect[3] + 2], outline=(0, 0, 0), width=1)
    draw.rectangle([rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3], outline=(255, 255, 255), width=1)
    return tile


def _compose_sheet(tiles, path, total):
    """One labeled sheet with a tile per span in span order; `tiles` holds (label, weak, image)."""
    cols = min(SHEET_COLS, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    pad, header = 4, 30
    cell_w, cell_h = TILE_SIZE[0], TILE_SIZE[1] + TILE_LABEL_HEIGHT
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, header + rows * cell_h + (rows + 1) * pad), (20, 22, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((pad + 4, 7), f"SPANS {len(tiles)}" + (f" of {total}" if total > len(tiles) else "")
              + " | boxed = detected match | * = weak", fill=(220, 225, 235), font=font)
    for i, (label, weak, image) in enumerate(tiles):
        x, y = pad + i % cols * (cell_w + pad), header + pad + i // cols * (cell_h + pad)
        draw.text((x + 4, y + 3), label, fill=(255, 190, 60) if weak else (255, 255, 255), font=font)
        sheet.paste(image, (x, y + TILE_LABEL_HEIGHT))
    staged = path.with_name(f"{path.stem}.{os.getpid()}.tmp.jpg")
    sheet.save(staged, quality=Config.JPEG_QUALITY)
    staged.replace(path)
    return {"file_path": str(path), "sample_count": len(tiles), "rows": rows, "cols": cols,
            "frame_width": TILE_SIZE[0], "frame_height": TILE_SIZE[1], "labels": [label for label, _, _ in tiles]}


def _validate(args, duration=None):
    description, text = args.get("description"), args.get("text")
    for name, value, limit in (("description", description, 500), ("text", text, 200)):
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            raise ValueError(f"{name} must be a string of at most {limit} characters")
    if not (description and description.strip()) and not (text and text.strip()):
        raise ValueError("Give `description` (a visible object or symbol, e.g. 'yellow outlined circle') "
                         "and/or `text` (the exact on-screen phrase, e.g. 'RELEASE').")
    if description and not any(parse_description(description)[k] for k in ("color", "shape", "prompt")):
        raise ValueError("`description` needs words describing the object, e.g. 'yellow outlined circle'.")
    start, end, fps = args.get("start_sec", 0.0), args.get("end_sec"), args.get("fps", 5.0)
    gap, max_spans = args.get("min_gap_sec", .3), args.get("max_spans", 20)
    values = [start, fps, gap] + ([] if end is None else [end])
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("start_sec, end_sec, fps and min_gap_sec must be finite numbers")
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"Need 0 <= start_sec < end_sec (got start_sec={start:g}, end_sec={end!r}); omit end_sec to search to the end of the video")
    if duration is not None and (start >= duration or (end is not None and end > duration + 1.0)):
        raise ValueError(f"Window must lie within the video (duration {duration:g}s)")
    if not 0 < fps <= 15:
        raise ValueError(f"fps must be in (0, 15] (got {fps:g}); omit it for the default of 5")
    if gap <= 0:
        raise ValueError(f"min_gap_sec must be positive (got {gap:g}); omit it for the default of 0.3")
    if isinstance(max_spans, bool) or not isinstance(max_spans, int) or not 1 <= max_spans <= MAX_SPANS_LIMIT:
        raise ValueError(f"max_spans must be an integer between 1 and {MAX_SPANS_LIMIT} (got {max_spans!r}); count always includes every span")
    direction = args.get("direction", "any")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)} (got {direction!r}); omit it to count every direction")


def _needs_clip(spec):
    return bool(spec) and not spec["shape"] and bool(spec["prompt"])


def _dependencies_ok(args):
    """Fail before scheduling, with an install hint, when a needed local engine is missing."""
    import importlib.util
    if args.get("text"):
        ocr.ensure_available()
    if args.get("description") and _needs_clip(parse_description(args["description"])):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            raise RuntimeError("Describing this object needs the local CLIP model: run python -m pip install "
                               "-r requirements-visual.txt and cache the model (see README visual search setup). "
                               "A plain coloured shape (e.g. 'yellow circle') and exact `text` work without it.")


def _iter_frames(source, start, end, fps, width, height, progress):
    """Yield (timestamp, PIL RGB) decoded one at a time so memory stays bounded."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(start), "-i", str(source), "-t", str(end - start),
           "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "2", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tail, stalled = bytearray(), threading.Event()

    def drain():
        try:
            while chunk := os.read(proc.stderr.fileno(), 4096):
                tail.extend(chunk)
                del tail[:-500]
        except OSError:
            pass  # the pipe closes when the process is killed

    def kill_stalled():
        stalled.set()
        proc.kill()
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    size, index = width * height * 3, 0
    try:
        while True:
            watchdog = threading.Timer(READ_STALL_TIMEOUT_SEC, kill_stalled)
            watchdog.start()
            try:
                block = proc.stdout.read(size)
            finally:
                watchdog.cancel()
            if len(block) == size:
                yield start + index / fps, Image.frombytes("RGB", (width, height), block)
                index += 1
                progress(index)
                continue
            code = proc.wait()
            if stalled.is_set():
                raise RuntimeError(f"ffmpeg stalled for {READ_STALL_TIMEOUT_SEC:g}s decoding the source")
            if code or block:
                raise RuntimeError(f"ffmpeg decode failed: {tail.decode(errors='replace') or 'truncated frame'}")
            if index == 0:
                raise ValueError("No frames could be decoded in this window; check start_sec/end_sec against the video duration")
            break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        reader.join(timeout=1)
        proc.stderr.close()


def _judge_frames(source, judge, builder, window, total, progress):
    """Decode the window, judge every frame and feed the builder; returns (one hit per frame, frames analyzed)."""
    start, end, fps, width, height = window
    workers = OCR_WORKERS if judge.pattern is not None and judge.spec is None else 1
    if judge.batchable:
        batch_frames, judge_batch, lookahead = CLIP_BATCH_FRAMES, judge.judge_many, CLIP_BATCH_FRAMES * 3
    else:
        batch_frames, judge_batch, lookahead = 1, lambda frames: [judge.judge(frames[0])], workers * 4
    previous, last_hit, analyzed, hits = None, None, 0, []
    pending = collections.deque()  # (t, frame, slot); slot is (batch, index), or None for a static frame
    batch = None  # [frames, future]: analyzed frames judged in one call; future is None until submitted

    def flush():
        nonlocal batch
        if batch is not None:
            batch[1] = pool.submit(judge_batch, batch[0])
            batch = None

    def settle(limit):
        nonlocal last_hit
        while len(pending) > limit:
            t, frame, slot = pending.popleft()
            if slot is not None:
                owner, index = slot
                if owner[1] is None:
                    flush()
                last_hit = owner[1].result()[index]
            builder.add(t, last_hit, frame)
            hits.append(last_hit)
    # Apple Vision text recognition runs about twice as fast with two frames in flight; frames are
    # still added in order, so results match a serial run. Visual (CLIP) judging stays serial, but
    # whole-frame CLIP checks embed several frames per call.
    with ThreadPoolExecutor(workers) as pool:
        for t, frame in _iter_frames(source, start, end, fps, width, height, lambda n: progress(n, total)):
            gray = np.asarray(frame.resize((160, 90)).convert("L"), dtype=np.int16)
            # A thin symbol appearing or leaving changes only a few pixels, so count changed pixels, not the mean.
            if previous is not None and (np.abs(gray - previous) > STATIC_PIXEL_DELTA).sum() < STATIC_PIXEL_COUNT:
                slot = None
            else:
                if batch is None:
                    batch = [[], None]
                slot = (batch, len(batch[0]))
                batch[0].append(frame)
                if len(batch[0]) >= batch_frames:
                    flush()
                previous, analyzed = gray, analyzed + 1
            pending.append((t, frame, slot))
            settle(lookahead)
        settle(0)
    return hits, analyzed


def _load_judgments(path):
    """(hits, analyzed) saved by an earlier run, or None when absent or unreadable."""
    try:
        saved = json.loads(path.read_text())
        return saved["hits"], int(saved["analyzed"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fill_tiles(source, spans, window):
    """Cut the zoomed tile of each span's representative sample from a frame decoded again at that sample's time."""
    start, fps, width, height = window
    for span in spans:
        t, score, box, _ = span["rep"]
        index = round((t - start) * fps)
        first = max(index - 1, 0)  # start one sample early so the fps filter picks the same source frame as the full pass
        frames = _iter_frames(source, start + first / fps, start + (index + 1) / fps + .5 / fps, fps, width, height, lambda n: None)
        try:
            frame = next(itertools.islice(frames, index - first, None))[1]
        finally:
            frames.close()
        span["rep"] = (t, score, box, _span_tile(frame, box))


def _source_stamp(source):
    stat = Path(source).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def count_video(engine, progress=lambda done, total: None, **args):
    meta = engine.metadata
    _validate(args, meta.duration_sec)
    stamp = _source_stamp(engine.source_path)
    description = (args.get("description") or "").strip() or None
    text = (args.get("text") or "").strip() or None
    case_sensitive = bool(args.get("case_sensitive", False))
    start = float(args.get("start_sec", 0.0))
    end = meta.duration_sec if args.get("end_sec") is None else min(meta.duration_sec, float(args["end_sec"]))
    requested_fps, min_gap = float(args.get("fps", 5.0)), float(args.get("min_gap_sec", .3))
    max_spans = int(args.get("max_spans", 20))
    fps = min(requested_fps, MAX_FRAMES / (end - start))
    if fps < 1.0:
        bounds = [start + i * MAX_FRAMES for i in range(int(math.ceil((end - start) / MAX_FRAMES)))] + [end]
        windows = ", ".join(f"{a:g}-{b:g}" for a, b in zip(bounds, bounds[1:]))
        raise ValueError(f"A {end - start:g}s window cannot be sampled at 1 fps within the {MAX_FRAMES}-frame budget; "
                         f"use start_sec/end_sec windows of at most {MAX_FRAMES} seconds, one call each: {windows}")
    fps = round(fps, 3)
    judged = [ALGORITHM_VERSION, meta.fingerprint, description, text, case_sensitive, start, end, requested_fps]
    direction = args.get("direction", "any")
    filters = [] if direction == "any" else [direction]
    tag = hashlib.sha256(json.dumps(judged + [min_gap, max_spans] + filters).encode()).hexdigest()[:20]
    cache_dir = engine._cache.get_entry_dir(meta.fingerprint)
    result_path = cache_dir / f"{tag}.occurrences.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if not cached["contact_sheet"] or Path(cached["contact_sheet"]["file_path"]).exists():
            return cached
    # Only SpanBuilder reads min_gap_sec and max_spans, so the per-frame judgments are kept under a key without them.
    judgments_path = cache_dir / f"{hashlib.sha256(json.dumps(judged).encode()).hexdigest()[:20]}.judgments.json"
    judge = FrameJudge(description, text, case_sensitive)
    width = min(meta.width, DECODE_WIDTH) // 2 * 2
    height = max(2, round(meta.height * width / meta.width) // 2 * 2)
    total = int(math.ceil((end - start) * fps))
    builder = SpanBuilder(1 / fps, min_gap, min(max_spans, SHEET_TILES))
    replayed = _load_judgments(judgments_path)
    if replayed is None:
        hits, analyzed = _judge_frames(engine.source_path, judge, builder, (start, end, fps, width, height), total, progress)
    else:
        hits, analyzed = replayed
        for index, hit in enumerate(hits):
            builder.add(start + index / fps, hit, None)
    spans = builder.finish()
    capped = len(spans) >= MAX_SPANS_LIMIT
    matched = spans if direction == "any" else [span for span in spans if _direction(span["points"]) == direction]
    shown = matched[:max_spans]
    missing_tiles = [span for span in shown[:SHEET_TILES] if span["rep"][3] is None]  # replayed runs, or spans past the builder's tile limit
    if missing_tiles:
        _fill_tiles(engine.source_path, missing_tiles, (start, fps, width, height))
    entries = []
    for i, span in enumerate(shown):
        x, y, w, h = span["rep"][2]
        span_end = min(span["end"], end)
        entries.append({"index": i + 1, "start_sec": round(span["start"], 3), "end_sec": round(span_end, 3),
                        "duration_sec": round(span_end - span["start"], 3), "score": round(span["score"], 3),
                        "representative_sec": round(span["rep"][0], 3), "frame_badge": f"#{i + 1}", "weak": span["weak"],
                        "region": {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)},
                        "travel": _direction(span["points"])})
    omitted = len(matched) - len(shown)
    spec = judge.spec
    result = {"status": "complete", "count": None if capped else len(matched), "spans": entries, "spans_returned": len(shown),
              "spans_omitted": omitted, "window": {"start_sec": round(start, 3), "end_sec": round(end, 3)},
              "fps_used": fps, "frames_sampled": total, "frames_analyzed": analyzed,
              "checks": {"text": text, "description": description, "color": spec and spec["color"],
                         "shape": spec and (spec["shape"] or spec["other_shape"])},
              "contact_sheet": None, "note": SPAN_NOTE}
    if fps < requested_fps:
        result["note"] += f" fps lowered from {requested_fps:g} to {fps:g} to fit the {MAX_FRAMES}-frame budget."
    if capped:
        result.update({"capped": True, "spans_found": len(spans)})
        result["note"] = (f"NOT A COUNT: at least {MAX_SPANS_LIMIT} spans matched (count is null; the true number is "
                          f"unknown), so the description is probably too broad and is matching noise or the background. "
                          f"Do not report a number from this. Make the description more specific "
                          f"(colour and shape words), use `text` for words, or narrow start_sec/end_sec, then rerun; "
                          f"or count by eye from inspect_window/contact_sheet. {result['note']}")
    elif omitted:
        result["note"] += (f" {omitted} later spans are omitted but included in count; list them by rerunning with "
                           f"start_sec={entries[-1]['end_sec']} or a larger max_spans.")
    if builder.bridged and not capped:
        joined = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in builder.bridged[:5]) + (", ..." if len(builder.bridged) > 5 else "")
        result["gaps_merged"] = len(builder.bridged)
        if direction == "any":
            result["count_without_merges"] = len(spans) + len(builder.bridged)
            unmerged = f"without merging the count is {len(spans) + len(builder.bridged)}"
        else:
            unmerged = f"the joined pairs' direction is unknown, so no count without merging is given for direction={direction}"
        result["note"] += (f" min_gap_sec={min_gap:g} joined {len(builder.bridged)} pair(s) of appearances that the default "
                           f"{DEFAULT_MIN_GAP:g} s keeps apart (gaps at {joined}); {unmerged}. "
                           f"Lower min_gap_sec if those were separate appearances.")
    travel = collections.Counter(_direction(span["points"]) for span in spans)
    moving = {way: n for way, n in travel.items() if way != "stationary"}
    if moving:
        result["travel_counts"] = moving
        if direction == "any":
            outcome = ("The count includes every direction. Directional counting is unsupported: verify each span with "
                       "inspect_window before answering a directional question.")
        else:
            outcome = "Verify each span with inspect_window before answering."
        result["note"] += (f" {sum(moving.values())} appearance(s) show net motion across the frame ("
                           f"{', '.join(f'{n} {way}' for way, n in moving.items())}), estimated from each span's first and last "
                           f"positions; these are not verified crossings. {outcome}")
    if direction != "any" and not capped:
        stationary = travel.get("stationary", 0)
        result["note"] += (f" direction={direction}: counted the {len(matched)} of {len(spans)} appearance(s) that travel "
                           f"{direction}; {len(spans) - len(matched) - stationary} moved another way and {stationary} stayed "
                           f"still, and are left out of count and spans.")
    if shown and not capped:
        result["note"] += TILE_NOTE
    if len(shown) > SHEET_TILES:
        result["note"] += (f" The sheet shows only the first {SHEET_TILES} of {len(shown)} listed spans; judge the rest "
                           f"with inspect_window at their representative_sec.")
    if not spans:
        result["note"] += " Nothing matched at this sampling rate; raise fps to catch sub-second appearances."
        if description:
            result["note"] += (" Count 0 only means nothing passed the checks in `checks`; if the target is on screen, "
                               "name its colour and a basic shape (circle, square, diamond, triangle), or count by eye.")
    if _source_stamp(engine.source_path) != stamp:
        raise RuntimeError("Source changed during occurrence counting; retry with the current video")
    with tempfile.TemporaryDirectory(prefix="occurrences-", dir=cache_dir) as temp:
        if replayed is None:
            staged = Path(temp) / "judgments.json"
            staged.write_text(json.dumps({"hits": hits, "analyzed": analyzed}))
            staged.replace(judgments_path)
        if shown:
            tiles = [(f"#{i + 1}{'*' if span['weak'] else ''} {span['rep'][0]:.1f}s", span["weak"], span["rep"][3])
                     for i, span in enumerate(shown[:SHEET_TILES])]
            sheet_path = engine._cache.get_sheets_dir(meta.fingerprint) / f"occurrences_{tag}.jpg"
            result["contact_sheet"] = _compose_sheet(tiles, sheet_path, len(shown))
        staged = Path(temp) / "result.json"
        staged.write_text(json.dumps(result))
        staged.replace(result_path)
    return result


def submit(engine, **args):
    """Start or poll a deduplicated worker; repeat the same call until status is complete or error."""
    global _worker, _worker_key
    _validate(args)
    _dependencies_ok(args)
    source = Path(engine.source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}. Pass the path of an existing local video file as `source`.")
    stat = source.stat()
    key = (ALGORITHM_VERSION, str(engine._cache.base_dir.resolve()), str(source),
           stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, json.dumps(args, sort_keys=True))
    with _jobs_lock:
        if key in _jobs:
            result = dict(_jobs[key])
            if result["status"] == "in_progress" and (_worker_key != key or _worker is None or not _worker.is_alive()):
                result = {"status": "error", "message": "Occurrence count worker stopped unexpectedly; repeat the call."}
                _jobs[key] = result
            if result["status"] == "error":
                del _jobs[key]
            return result
        if _worker is not None and _worker.is_alive():
            return {"status": "in_progress", "retry_after_seconds": 5,
                    "message": "Another occurrence count is running; repeat this call shortly."}
        while len(_jobs) >= 32:
            _jobs.popitem(last=False)
        _jobs[key] = {"status": "in_progress", "retry_after_seconds": 5}

        def report(done, total):
            with _jobs_lock:
                _jobs[key] = {"status": "in_progress", "retry_after_seconds": 5,
                              "message": f"Analyzed {done} of {total} sampled frames; repeat this call to poll."}

        def work():
            try:
                result = count_video(engine, progress=report, **args)
            except Exception as exc:
                result = {"status": "error", "message": str(as_dependency_error(exc))}
            with _jobs_lock:
                _jobs[key] = result
        _worker = threading.Thread(target=work, name="occurrence-count", daemon=True)
        _worker_key = key
        _worker.start()
        return {"status": "in_progress", "retry_after_seconds": 5}
