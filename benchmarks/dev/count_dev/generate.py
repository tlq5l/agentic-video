#!/usr/bin/env python3
"""Generate the count dev set: occurrence videos, repetition clips, questions and truth.

Deterministic from --seed (default 5117). Media goes outside the repo (default
~/.cache/agentic-video-dev/count_dev, override with COUNT_DEV_ROOT or --out); the question
and truth JSONL files are written next to this script and may be committed. Rendering
helpers come from benchmarks/bench_v1/generate_synthetic.py; nothing of its generate()
or its fixed counts and labels is reused.

    python -m benchmarks.dev.count_dev.generate [--seed N] [--out DIR] [--only ID ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmarks.bench_v1.generate_synthetic import (
    _draw_text_filter, _ffmpeg_filter_path, _font, _quantize, _require, _run)

SEED = 5117
FPS = 24
HERE = Path(__file__).resolve().parent

# phrase -> (glyph, tool-level shape class); outlined glyphs carry "outlined" in their label
SHAPES = {
    "circle": ("●", "circle"), "outlined circle": ("○", "circle"),
    "square": ("■", "square"), "outlined square": ("□", "square"),
    "diamond": ("◆", "diamond"), "outlined diamond": ("◇", "diamond"),
    "triangle": ("▲", "triangle"), "outlined triangle": ("△", "triangle"),
    "star": ("★", "star"), "outlined star": ("☆", "star"),
    "heart": ("♥", "heart"), "hexagon": ("⬢", "hexagon"), "crescent": ("☾", "crescent"),
    "arrowhead": ("▶", "arrow"),
}
TRAIN_SHAPES = ["circle", "outlined circle", "square", "outlined square", "diamond", "outlined diamond",
                "triangle", "outlined triangle", "star"]
TEST_ONLY_SHAPES = ["heart", "hexagon", "crescent", "outlined star", "arrowhead"]
# colour word -> ffmpeg colour
COLORS = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "lime", "cyan": "cyan",
          "blue": "dodgerblue", "white": "white", "purple": "blueviolet", "magenta": "magenta", "pink": "lightpink"}
TRAIN_COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "white"]
TEST_ONLY_COLORS = ["purple", "magenta", "pink"]
TRAIN_WORDS = ["LAUNCH", "SIGNAL LOST", "TARGET LOCKED", "STANDBY", "OVERRIDE", "FUEL LOW"]
TEST_ONLY_WORDS = ["HULL BREACH", "CLEARED", "NEW ORDER", "ALARM"]
TINTS = ["navy", "maroon", "darkgreen", "darkslategray", "darkolivegreen", "darkslateblue", "indigo", "saddlebrown"]
# background cell colour per colour family; a bright background often shares a hue with the target
LIFE_COLORS = {"cyan": "0x77b7c5", "orange": "0xc59b77", "yellow": "0xc5b877", "green": "0x8fc577", "blue": "0x7788c5",
               "purple": "0xb777c5", "magenta": "0xc577b0", "pink": "0xc59aa8", "red": "0xc5777f", "white": "0xbdbdbd"}

TRAP_ARGS = {"close": {}, "crowd": {"max_spans": 100}}  # arguments an agent passed in the observed failures
# trap2 layouts: "passes" = left-to-right passes beside right-to-left ones, "rtl" = only right-to-left passes,
# "cap" = 101 short appearances (the 100-span limit); asked with the default arguments
TRAP_ARGS.update({"passes": {}, "rtl": {}, "cap": {}})

# id, split, motion blur, tint alpha (0 = no scene tints, so no cuts and a bright background), groups of (kind, layout)
VIDEOS = [
    ("occ01", "train", False, .6, [("glyph", "gaps"), ("glyph", "zero"), ("text", "cuts")]),
    ("occ02", "train", False, .5, [("glyph", "cuts"), ("text", "gaps"), ("glyph", "single")]),
    ("occ03", "train", True, .5, [("cross", "many"), ("cross", "one"), ("glyph", "gaps")]),
    ("occ04", "train", False, .7, [("text", "cuts"), ("text", "zero"), ("glyph", "long")]),
    ("occ05", "train", False, .5, [("glyph", "cut_jump"), ("glyph", "gaps"), ("text", "long")]),
    ("occ06", "train", True, .6, [("cross", "many"), ("cross", "zero")]),
    ("occ07", "train", False, 0, [("glyph", "gaps"), ("glyph", "gaps"), ("glyph", "zero")]),
    ("occ08", "train", False, .6, [("text", "gaps"), ("glyph", "single"), ("text", "cuts")]),
    ("occ09", "train", False, 0, [("glyph", "long"), ("glyph", "gaps")]),
    ("occ10", "train", True, .5, [("cross", "many"), ("glyph", "cuts"), ("text", "zero")]),
    ("occ11", "test", False, .6, [("glyph", "gaps"), ("glyph", "cuts"), ("text", "cuts")]),
    ("occ12", "test", True, .5, [("cross", "many"), ("cross", "zero"), ("glyph", "long")]),
    ("occ13", "test", False, .5, [("text", "gaps"), ("glyph", "cut_jump"), ("glyph", "zero")]),
    ("occ14", "test", False, 0, [("glyph", "gaps"), ("text", "long"), ("glyph", "gaps")]),
    # trap cases: "close" appearances are 0.4-0.9 s apart and the question carries the min_gap_sec an agent chose
    # (0.8-1.0); "crowd" hides a target of 1-4 appearances among many shapes of other colours, asked with max_spans 100
    ("occ15", "train", False, .6, [("glyph", "close"), ("text", "close")]),
    ("occ16", "train", False, .5, [("glyph", "crowd"), ("glyph", "close")]),
    ("occ17", "train", False, 0, [("glyph", "crowd"), ("text", "close")]),
    ("occ18", "train", True, .5, [("glyph", "crowd"), ("glyph", "crowd")]),
    ("occ19", "test", False, .5, [("glyph", "close"), ("glyph", "crowd")]),
    ("occ20", "test", False, .6, [("text", "close"), ("glyph", "crowd"), ("glyph", "close")]),
    # trap2 (climb set, split train2): left-to-right passes beside right-to-left ones (the tool counts every pass),
    # right-to-left passes only, close appearances with a long min_gap_sec, and 101 appearances (the span cap)
    ("trap2_occ21", "train2", False, .5, [("cross", "passes"), ("cross", "passes"), ("cross", "rtl")]),
    ("trap2_occ22", "train2", False, .6, [("glyph", "close"), ("glyph", "close"), ("text", "close")]),
    ("trap2_occ23", "train2", False, 0, [("glyph", "close"), ("glyph", "close"), ("glyph", "close")]),
    ("trap2_occ24", "train2", False, .5, [("glyph", "cap"), ("cross", "passes")]),
    ("trap2_occ25", "train2", True, .5, [("glyph", "cap"), ("glyph", "crowd")]),
]

# id, split, motion, reps, period sec, tempo drift (fraction), sensor noise, background, camera drift, pose shape
REPS = [
    ("rep01", "train", "bounce", 12, .55, 0, .02, "plain", 0, "cos"),
    ("rep02", "train", "jack", 15, .7, .25, .03, "texture", 0, "cos"),
    ("rep03", "train", "sway", 6, 1.3, 0, .02, "texture", 0, "cos"),
    ("rep04", "train", "press", 9, .95, -.3, .03, "texture", 1, "skew"),
    ("rep05", "train", "orbit", 8, .8, 0, .02, "plain", 0, "cos"),
    ("rep06", "train", "pulse", 20, .38, 0, .02, "plain", 0, "cos"),
    ("rep07", "train", "bounce", 4, 2.2, 0, .04, "texture", 1, "cos"),
    ("rep08", "train", "jack", 10, .6, .4, .02, "clutter", 0, "skew"),
    ("rep09", "test", "press", 14, .5, 0, .03, "texture", 1, "skew"),
    ("rep10", "test", "orbit", 5, 1.4, .3, .02, "texture", 0, "cos"),
    ("rep11", "test", "jack", 18, .45, -.2, .03, "clutter", 1, "cos"),
    ("rep12", "test", "bounce", 2, 3.2, 0, .02, "plain", 0, "cos"),
    # trap2: the periodic count ran one cycle over on train (rep03, rep05, rep07)
    ("trap2_rep13", "train2", "sway", 5, 1.5, 0, .02, "plain", 0, "cos"),
    ("trap2_rep14", "train2", "bounce", 7, 1.1, .2, .03, "texture", 1, "skew"),
]


def data_root(override=None) -> Path:
    return Path(override or os.environ.get("COUNT_DEV_ROOT", "~/.cache/agentic-video-dev/count_dev")).expanduser()


# ---------------------------------------------------------------- occurrence videos

def _label(color, shape):
    return " ".join([color] + shape.split())


def _tool_class(shape):
    return SHAPES[shape][1]


def _glyph_filter(ev):
    """drawtext for one glyph event: a wobbling icon in a slot, or a straight crossing."""
    start, dur = ev["start"], max(.1, ev["end"] - ev["start"])
    parts = []
    if _font():
        parts.append(f"fontfile='{_ffmpeg_filter_path(_font())}'")
    if ev["path"] == "cross_lr":
        x, y = f"40+560*(t-{start:.6f})/{dur:.6f}", str(ev["y"])
    elif ev["path"] == "cross_rl":
        x, y = f"600-560*(t-{start:.6f})/{dur:.6f}", str(ev["y"])
    else:
        x = f"{ev['x']}+25*sin(t*1.7+{ev['phase']:.3f})"
        y = f"{ev['y']}+17*cos(t*1.19+{ev['phase']:.3f})"
    parts += [f"text='{SHAPES[ev['shape']][0]}'", f"fontcolor={COLORS[ev['color']]}", f"fontsize={ev['size']}",
              "borderw=3", "bordercolor=black@0.9", f"x='{x}'", f"y='{y}'",
              f"enable='gte(t,{start:.6f})*lt(t,{ev['end']:.6f})'"]
    return "drawtext=" + ":".join(parts)


def _render(path: Path, duration, events, texts, cuts, seed, motion_blur, life_color, noise, tint):
    """Life-texture background, colour-tint scene cuts, then the events (compare bench_v1 _render_video)."""
    ffmpeg = _require("ffmpeg")
    filters = [f"noise=alls={noise}:allf=t+u:all_seed={seed}"]
    if seed % 2:
        filters.append("drawgrid=width=60:height=52:thickness=1:color=white@0.12")
    for start, end, color in cuts:
        filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color={color}@{tint}:t=fill:enable='gte(t,{start:.6f})*lt(t,{end:.6f})'")
    filters += [_glyph_filter(ev) for ev in events]
    filters += [_draw_text_filter(t["text"], "white", t["start"], t["end"], size=t["size"], y=t["y"]) for t in texts]
    if motion_blur:
        filters.append("tmix=frames=3:weights='1 2 1'")
    filters += ["crop=640:360:x='40+12*sin(t*0.16)':y='20+9*cos(t*0.11)'", "format=yuv420p"]
    dark = "0x" + "".join(f"{int(int(life_color[i:i + 2], 16) * .2):02x}" for i in (2, 4, 6))
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
          f"life=s=720x420:rate={FPS}:ratio=0.28:rule=B36/S23:seed={seed}:life_color={life_color}:"
          f"death_color={dark}:mold_color={dark}:mold=3",
          "-t", str(duration), "-vf", ",".join(filters), "-an", "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(path)])


class _Plan:
    """Builds one video's events and truth from a recipe; every random choice comes from `rng`."""

    def __init__(self, rng, split):
        self.rng, self.split = rng, split
        self.events, self.texts, self.cuts = [], [], []  # cuts: instants of hard scene changes
        self.targets, self.decoys = [], []  # (shape class, colour, outlined); a decoy never matches a target
        self.cursor = 2.0

    def pool(self, train, test_only, decoy=False):
        """Vocabulary for a choice. Test videos often draw targets from test-only words; train videos never see them."""
        if self.split != "test":
            return train
        return train + test_only if decoy or self.rng.random() >= .6 else test_only

    def slot(self, used):
        """A grid position no other symbol of the group uses (wobbling symbols never overlap a neighbour)."""
        slot = self.rng.choice([(x, y) for x in (62, 198, 334, 470) for y in (50, 165, 280) if (x, y) not in used])
        used.add(slot)
        return slot

    @staticmethod
    def _matches(target, other):
        """A target label matches its class and colour; an "outlined" label also needs an outlined glyph."""
        return target[:2] == other[:2] and (not target[2] or other[2])

    def _clashes(self, key, against):
        return any(not isinstance(t, str) and (self._matches(t, key) or self._matches(key, t)) for t in against)

    def target_glyph(self):
        while True:
            shape = self.rng.choice(self.pool(TRAIN_SHAPES, TEST_ONLY_SHAPES))
            color = self.rng.choice(self.pool(TRAIN_COLORS, TEST_ONLY_COLORS))
            key = (_tool_class(shape), color, "outlined" in shape)
            if not self._clashes(key, self.targets + self.decoys):
                self.targets.append(key)
                return shape, color

    def decoy_specs(self, shape, color, n, modes=("color", "shape", "style")):
        """Look-alikes: same shape in another colour, same colour in another shape, filled twin of an outlined target."""
        rng, out = self.rng, []
        shapes = self.pool(TRAIN_SHAPES, TEST_ONLY_SHAPES, decoy=True)
        colors = self.pool(TRAIN_COLORS, TEST_ONLY_COLORS, decoy=True)
        for _ in range(n):
            mode = rng.choice([m for m in modes if m != "style" or "outlined" in shape])
            if mode == "color":
                cand = (shape, rng.choice([c for c in colors if c != color]))
            elif mode == "shape":
                cand = (rng.choice([s for s in shapes if _tool_class(s) != _tool_class(shape)]), color)
            else:
                cand = (shape.replace("outlined ", ""), color)
            key = (_tool_class(cand[0]), cand[1], "outlined" in cand[0])
            if not self._clashes(key, self.targets):
                self.decoys.append(key)
                out.append(cand)
        return out

    def add_glyph(self, shape, color, start, end, slot, path="wobble", size=66):
        x, y = slot
        self.events.append({"shape": shape, "color": color, "start": _quantize(start), "end": _quantize(end),
                            "path": path, "x": x, "y": y, "phase": round(self.rng.uniform(0, 6), 3), "size": size})

    def group(self, kind, layout, qid):
        """Add one target's appearances plus decoys; returns the question row and truth row."""
        return getattr(self, f"_{kind}")(layout, qid)

    def _count(self, layout):
        fixed = {"zero": 0, "single": 1, "long": 1, "one": 1, "rtl": 0, "cap": 101}
        if layout in fixed:
            return fixed[layout]
        if layout == "crowd":
            return self.rng.randint(1, 4)
        if layout == "close":
            return self.rng.randint(2, 4)
        return self.rng.randint(2, 3) if layout == "cut_jump" else self.rng.randint(2, 6)

    def _spans(self, layout, n):
        """[(start, end)] of the target's appearances, and the scene-cut instants that fall between them."""
        rng, t, spans, cuts = self.rng, self.cursor, [], []
        if layout == "long":
            spans.append((t, t + rng.uniform(5.5, 8.0)))
        elif layout in ("single", "one"):
            spans.append((t, t + rng.uniform(.9, 2.0)))
        elif layout == "cap":  # short appearances, each gap longer than the default min_gap_sec keeps apart
            for _ in range(n):
                spans.append((t, t + rng.uniform(.5, .8)))
                t = spans[-1][1] + rng.uniform(.4, .6)
        elif layout == "cut_jump":  # the marker is replaced at once, at a cut, by itself in another slot
            for i in range(n):
                spans.append((t, t + rng.uniform(1.0, 1.6)))
                t = spans[-1][1]
                cuts.append(t)
            cuts.pop()
        else:
            for _ in range(n):
                spans.append((t, t + rng.uniform(.8, 1.5)))
                gap = (rng.uniform(.5, 1.4) if layout == "cuts" else rng.uniform(.4, .9) if layout == "close"
                       else rng.uniform(.45, 1.7))
                cuts.append(spans[-1][1] + gap / 2 if layout == "cuts" else None)
                t = spans[-1][1] + gap
            cuts.pop()
        return spans, [c for c in cuts if c is not None]

    def _finish(self, end):
        self.cursor = end + 1.6
        self.cuts.append(round(end + .8, 3))

    def _glyph(self, layout, qid):
        rng, n = self.rng, self._count(layout)
        shape, color = self.target_glyph()
        label = _label(color, shape)
        spans, cuts = self._spans(layout, n) if n else ([], [])
        self.cuts += [round(c, 3) for c in cuts]
        used = set()
        slot = self.slot(used)
        for i, (a, b) in enumerate(spans):
            if layout in ("cuts", "cut_jump") and i:
                slot = self.slot(used)
            self.add_glyph(shape, color, a, b, slot)
        end = spans[-1][1] if spans else self.cursor + 6.0
        if layout == "crowd":
            end = max(end, self._crowd(shape, color, used, end))
        else:
            for dshape, dcolor in self.decoy_specs(shape, color, rng.randint(2, 4)):
                a = rng.uniform(self.cursor, max(self.cursor + 1.0, end - 1.0))
                self.add_glyph(dshape, dcolor, a, a + rng.uniform(.7, 1.4), self.slot(used))
                end = max(end, self.events[-1]["end"])
        self._finish(end)
        row = {"id": qid, "tool": "count_occurrences", "target": label, "target_kind": "description",
               "mode": "appearances", "question": _glyph_question(rng, label)}
        if layout in TRAP_ARGS:
            row["args"] = self._trap_args(layout)
        return (row,
                {"id": qid, "count": len(spans), "intervals": [[_quantize(a), _quantize(b)] for a, b in spans],
                 "layout": layout})

    def _trap_args(self, layout):
        """The argument an agent chose in the observed failure: a long min_gap_sec, or the largest max_spans."""
        return {"min_gap_sec": round(self.rng.uniform(.8, 1.0), 2)} if layout == "close" else dict(TRAP_ARGS[layout])

    def _crowd(self, shape, color, used, end):
        """8-10 shapes of other colours (any shape, the target's own included) that stay on screen through the target's appearances."""
        rng = self.rng
        shapes = self.pool(TRAIN_SHAPES, TEST_ONLY_SHAPES, decoy=True)
        colors = [c for c in self.pool(TRAIN_COLORS, TEST_ONLY_COLORS, decoy=True) if c != color]
        last = end
        for _ in range(rng.randint(8, 10)):
            while True:
                dshape, dcolor = rng.choice(shapes), rng.choice(colors)
                key = (_tool_class(dshape), dcolor, "outlined" in dshape)
                if not self._clashes(key, self.targets):
                    break
            self.decoys.append(key)
            a = rng.uniform(self.cursor, max(self.cursor + .5, end - 1.5))
            self.add_glyph(dshape, dcolor, a, a + rng.uniform(2.0, 4.0), self.slot(used), size=54)
            last = max(last, self.events[-1]["end"])
        return last

    def _text(self, layout, qid):
        rng, n = self.rng, self._count(layout)
        word = rng.choice([w for w in self.pool(TRAIN_WORDS, TEST_ONLY_WORDS) if w not in self.targets])
        self.targets.append(word)
        spans, cuts = self._spans(layout, n) if n else ([], [])
        self.cuts += [round(c, 3) for c in cuts]
        y = rng.choice([40, 150, 250])
        for a, b in spans:
            self.texts.append({"text": word, "start": _quantize(a), "end": _quantize(b), "size": 42, "y": y})
        end = spans[-1][1] if spans else self.cursor + 5.0
        for text in rng.sample([word + "S", word + "ED", "RE" + word, "UN" + word], 2):
            a = rng.uniform(self.cursor, max(self.cursor + 1.0, end - 1.2))
            self.texts.append({"text": text, "start": _quantize(a), "end": _quantize(a + rng.uniform(.9, 1.4)),
                               "size": 40, "y": rng.choice([v for v in (40, 150, 250) if v != y])})
            end = max(end, self.texts[-1]["end"])
        self._finish(end)
        question = rng.choice([
            f"How many separate times does the exact on-screen phrase ‘{word}’ appear? Ignore similar phrases with extra letters.",
            f"Count how many times the text ‘{word}’ is shown on screen. Similar-looking words do not count.",
            f"How many distinct times is the phrase ‘{word}’ displayed in this video?"])
        row = {"id": qid, "tool": "count_occurrences", "target": word, "target_kind": "text", "mode": "appearances",
               "question": question}
        if layout in TRAP_ARGS:
            row["args"] = self._trap_args(layout)
        return (row,
                {"id": qid, "count": len(spans), "intervals": [[_quantize(a), _quantize(b)] for a, b in spans],
                 "layout": layout})

    def _cross(self, layout, qid):
        """Left-to-right traversals are the truth; right-to-left passes and look-alikes are decoys."""
        rng, n = self.rng, self._count(layout)
        shape, color = self.target_glyph()
        label = _label(color, shape)
        row = rng.choice([70, 150, 240])
        t, truth = self.cursor, []
        for _ in range(n):
            dur = rng.uniform(1.3, 2.4)
            self.add_glyph(shape, color, t, t + dur, (0, row), path="cross_lr")
            truth.append([_quantize(t), _quantize(t + dur)])
            t += dur + rng.uniform(.6, 1.6)
        for _ in range(rng.randint(1, 2) if n else 2):  # right to left, same marker
            self.add_glyph(shape, color, t, t + rng.uniform(1.3, 2.2), (0, row), path="cross_rl")
            t = self.events[-1]["end"] + rng.uniform(.6, 1.2)
        other_row = rng.choice([r for r in (70, 150, 240) if r != row])
        for dshape, dcolor in self.decoy_specs(shape, color, 2, modes=("color",)):  # left to right, same marker in another colour
            self.add_glyph(dshape, dcolor, t, t + rng.uniform(1.3, 2.2), (0, other_row), path="cross_lr")
            t = self.events[-1]["end"] + .5
        self._finish(t)
        question = rng.choice([
            f"How many times does the {label} marker travel left to right across the screen? Ignore right-to-left passes and other colors.",
            f"Count the left-to-right crossings of the {label} across the display. Passes in the other direction do not count.",
            f"How many times does the {label} cross the screen from left to right?"])
        row = {"id": qid, "tool": "count_occurrences", "target": label, "target_kind": "description",
               "mode": "left_to_right", "question": question}
        if layout in TRAP_ARGS:
            row["args"] = self._trap_args(layout)
        return (row, {"id": qid, "count": n, "intervals": truth, "layout": layout})


def _glyph_question(rng, label):
    return rng.choice([
        f"How many separate times does the {label} appear? Ignore look-alikes in other colors or shapes.",
        f"Count how many times the {label} shows up on screen.",
        f"How many distinct appearances of the {label} are there in this video?"])


def plan_video(spec, seed):
    vid, split, blur, tint, groups = spec
    rng = random.Random(f"{seed}-{vid}")
    plan = _Plan(rng, split)
    questions, truths = [], []
    for i, (kind, layout) in enumerate(groups, 1):
        q, t = plan.group(kind, layout, f"{vid}_q{i}")
        questions.append({**q, "video": f"{vid}.mp4", "split": split})
        truths.append({**t, "video": f"{vid}.mp4", "split": split})
    duration = math.ceil(plan.cursor + 1.0)
    # one tint per scene between consecutive cut instants: hard cuts, no fades
    tints = []
    if tint:
        bounds = [0.0] + sorted(plan.cuts) + [float(duration)]
        first = rng.randrange(len(TINTS))
        tints = [(a, b, TINTS[(first + i) % len(TINTS)]) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    # an untinted background shares the hue of the first symbol it must not confuse with the target
    hues = [k[1] for k in plan.targets if not isinstance(k, str)]
    life = LIFE_COLORS[hues[0]] if not tint and hues else rng.choice(list(LIFE_COLORS.values()))
    render = {"duration": duration, "events": plan.events, "texts": plan.texts, "cuts": tints,
              "seed": rng.randrange(1, 10000), "motion_blur": blur, "life_color": life,
              "noise": rng.randint(5, 10), "tint": tint}
    return questions, truths, render


# ---------------------------------------------------------------- repetition clips

W, H, REP_FPS = 320, 180, 30


def _pose(phase, shape):
    """0..1 pose for a phase in cycles; "skew" is slow going out and quick coming back."""
    frac = phase % 1.0
    if shape == "skew":
        frac = frac ** 1.6
    return .5 - .5 * math.cos(2 * math.pi * frac)


def _texture(rng, background):
    if background == "plain":
        return np.full((H + 40, W + 40, 3), 30, np.uint8)
    base = rng.normal(0, 1, (H // 6 + 8, W // 6 + 8))
    tex = np.asarray(Image.fromarray(((base - base.min()) / np.ptp(base) * 255).astype(np.uint8))
                     .resize((W + 40, H + 40), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    return np.stack([25 + tex * .18, 30 + tex * .16, 38 + tex * .14], axis=2).astype(np.uint8)


def _draw_pose(d, motion, phase, pose, cx, cy):
    fill = (235, 235, 235)
    if motion == "bounce":
        y = cy + 30 - int(60 * pose)
        d.ellipse((cx - 12, y - 12, cx + 12, y + 12), fill=fill)
    elif motion == "sway":
        x = cx + int(60 * math.sin(2 * math.pi * phase))
        d.line((cx, 20, x, cy + 20), fill=(170, 170, 170), width=2)
        d.ellipse((x - 11, cy + 9, x + 11, cy + 31), fill=fill)
    elif motion == "jack":
        lift = int(38 * pose)
        d.ellipse((cx - 8, cy - 44, cx + 8, cy - 28), fill=fill)
        d.line((cx, cy - 28, cx, cy + 10), fill=fill, width=6)
        for side in (-1, 1):
            d.line((cx, cy - 22, cx + side * (10 + lift // 2), cy + 2 - lift), fill=fill, width=4)
            d.line((cx, cy + 10, cx + side * (6 + lift // 3), cy + 46), fill=fill, width=4)
    elif motion == "press":
        rise = int(34 * pose)
        d.rectangle((cx - 45, cy + 22 - rise, cx + 45, cy + 34 - rise), fill=fill)
        d.ellipse((cx + 40, cy + 6 - rise, cx + 58, cy + 24 - rise), fill=fill)
        d.line((cx - 40, cy + 34 - rise, cx - 44, cy + 50), fill=fill, width=5)
    elif motion == "orbit":
        a = 2 * math.pi * phase
        x, y = cx + int(50 * math.cos(a)), cy + int(36 * math.sin(a))
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=fill)
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(120, 120, 120))
    elif motion == "pulse":
        r = 8 + int(28 * pose)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)


def render_rep(path: Path, spec, seed):
    rid, _split, motion, count, period, drift, noise, background, camera, pace = spec
    rng = np.random.default_rng([seed, int(rid.rsplit("rep", 1)[1])])
    periods = np.linspace(period, period * (1 + drift), count)
    lead, tail = 1.0 + rng.uniform(0, .6), 1.0 + rng.uniform(0, .6)
    ends = lead + np.cumsum(periods)
    texture = _texture(rng, background)
    cx, cy = W // 2 + int(rng.integers(-30, 30)), H // 2 + int(rng.integers(-10, 10))
    wander = rng.uniform(0, 6, 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                             "-video_size", f"{W}x{H}", "-framerate", str(REP_FPS), "-i", "pipe:0", "-an",
                             "-c:v", "libx264", "-crf", "24", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                             "-threads", "2", str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for n in range(int((ends[-1] + tail) * REP_FPS)):
            t = n / REP_FPS
            done = int(np.searchsorted(ends, t, side="right"))
            if t < lead:
                phase = 0.0
            elif done >= count:
                phase = float(count)
            else:
                phase = done + (t - (ends[done - 1] if done else lead)) / periods[done]
            dx = 20 + int(camera * 12 * math.sin(t * .35 + wander[0]))
            dy = 20 + int(camera * 8 * math.cos(t * .27 + wander[1]))
            frame = Image.fromarray(texture[dy:dy + H, dx:dx + W].copy())
            d = ImageDraw.Draw(frame)
            if background == "clutter":  # an unrelated, non-periodic mover
                bx = 20 + int((t * 23) % (W - 40))
                d.ellipse((bx, 12, bx + 10, 22), fill=(120, 160, 90))
            _draw_pose(d, motion, phase, _pose(phase, pace), cx, cy)
            arr = np.asarray(frame, dtype=np.float32) + rng.normal(0, noise * 255, (H, W, 1))
            proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        if proc.wait(timeout=60):
            raise RuntimeError(error)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def rep_rows(spec):
    rid, split, motion, count = spec[:4]
    return ({"id": rid, "tool": "count_repetitions", "target": motion, "target_kind": "action", "mode": "repetitions",
             "question": "How many times is the movement repeated in this video? Count each complete repetition.",
             "video": f"{rid}.mp4", "split": split},
            {"id": rid, "count": count, "video": f"{rid}.mp4", "split": split, "layout": motion})


# ---------------------------------------------------------------- driver

def build(seed=SEED, out=None, only=None, render=True):
    root = data_root(out)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    questions, truths = [], []
    for spec in VIDEOS:
        qs, ts, params = plan_video(spec, seed)
        questions += qs
        truths += ts
        if render and (not only or spec[0] in only):
            _render(root / "videos" / f"{spec[0]}.mp4", params["duration"], params["events"], params["texts"],
                    params["cuts"], params["seed"], params["motion_blur"], params["life_color"], params["noise"], params["tint"])
    for spec in REPS:
        q, t = rep_rows(spec)
        questions.append(q)
        truths.append(t)
        if render and (not only or spec[0] in only):
            render_rep(root / "videos" / f"{spec[0]}.mp4", spec, seed)
    return questions, truths


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", help="media directory (default ~/.cache/agentic-video-dev/count_dev)")
    parser.add_argument("--only", nargs="*", help="render only these video ids")
    parser.add_argument("--no-render", action="store_true", help="write the JSONL files without rendering media")
    args = parser.parse_args()
    if args.seed == 8448:
        parser.error("8448 is the bench-v1 seed; use a different one")
    questions, truths = build(args.seed, args.out, args.only, render=not args.no_render)
    write_jsonl(HERE / "questions.jsonl", questions)
    write_jsonl(HERE / "truth.jsonl", truths)
    print(f"{len(questions)} questions, media under {data_root(args.out)}")


if __name__ == "__main__":
    main()
