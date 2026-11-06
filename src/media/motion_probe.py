"""motion_ tools: is a GIF or short clip's motion right? (noise, boil, stutter, loop seam)

Overview first (motion_sheet, motion_stats, motion_profile), then zoom (motion_zoom,
motion_boil_map, motion_compare). Frames and the motion index stay in a process-wide LRU
(memory-warm) and are written under the MediaCache entry for the file's fingerprint
(disk-warm, survives a restart). motion_open starts the build in a background thread so an
agent can pre-warm a file before it asks.

Measures, on luma 0-255:
- change[t]: mean |frame t - frame t-1| over the whole frame. t=0 wraps to the last frame, so
  change[0] is the loop seam.
- still mask: 8x8 blocks whose median frame-to-frame change is below 2.0 levels.
- boil: mean |f_t - (f_{t-1} + f_{t+1}) / 2| over still pixels, all frames. Flicker where
  nothing should move. (The temporal-std version failed its planted test: it reads slow drift
  as boil.)
- speed[t]: mean |frame t - frame t-1| counting only changes above 2 levels. A stand-in for
  the moving parts' speed; there is no optical flow, cv2 is not a repo dependency.
"""
import hashlib
import json
import os
import subprocess
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.cache import MediaCache

VERSION = "motion-v1"
BLOCK = 8
STILL_T = 2.0
WIDTH_CAP = 512
MAX_FRAMES = 240
MAX_PX = 8000
RENDER_VERSION = 1  # bump when a tool's drawing changes, so saved images are not reused
BG = (128, 128, 128)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
STUTTER_RATIO = 3.0
STUTTER_FLOOR = 0.05  # fraction of the median speed; stops near-zero neighbours inflating the ratio
FREEZE_RATIO = 0.10
FREEZE_MIN_RUN = 2
DUP_LEVEL = 0.05
SEAM_RATIO = 3.0
BOIL_FAULT_DELTA = 1.0
STORE_MAX_BYTES = 384 * 1024 * 1024
ARRAYS = ("rgb", "change", "speed", "still", "flicker", "median")
DEFAULT_FIXTURE = Path(__file__).resolve().parent / "clean-sprite-256.gif"  # ships in the wheel with src/


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType sizing
        return ImageFont.load_default()


FONT = _font(13)
FONT_BIG = _font(16)


# --------------------------------------------------------------------------- decoding

def _probe_ffmpeg(path: Path) -> Tuple[int, int, float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout
        stream = json.loads(out)["streams"][0]
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name} as an image or a video") from exc
    num, _, den = str(stream["r_frame_rate"]).partition("/")
    fps = round(float(num) / float(den), 3) if den and float(den) else float(num or 0)
    return int(stream["width"]), int(stream["height"]), fps


def _decode_pillow(im) -> Tuple[np.ndarray, List[int], float]:
    n = min(getattr(im, "n_frames", 1), MAX_FRAMES)
    w0, h0 = im.size
    W = min(w0, WIDTH_CAP)
    H = round(h0 * W / w0)
    frames, durs = [], []
    for i in range(n):
        im.seek(i)
        durs.append(int(im.info.get("duration") or 100))
        rgba = im.convert("RGBA")
        pic = Image.alpha_composite(Image.new("RGBA", rgba.size, BG + (255,)), rgba).convert("RGB")
        if W != w0:
            pic = pic.resize((W, H), Image.Resampling.BILINEAR)
        frames.append(np.asarray(pic))
    total = sum(durs)
    return np.stack(frames), durs, round(1000 * n / total, 3) if total else 0.0


def _decode_ffmpeg(path: Path) -> Tuple[np.ndarray, List[int], float, int, int]:
    w0, h0, fps = _probe_ffmpeg(path)
    W = min(w0, WIDTH_CAP) // 2 * 2
    H = max(2, round(h0 * W / w0) // 2 * 2)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-frames:v", str(MAX_FRAMES),
         "-vf", f"scale={W}:{H}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    if n == 0:
        raise ValueError(f"no decodable video frames in {path.name}")
    rgb = np.frombuffer(raw, np.uint8, count=n * H * W * 3).reshape(n, H, W, 3).copy()
    per = round(1000 / fps) if fps else 100
    return rgb, [per] * n, fps, w0, h0


def decode(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Frames as uint8 (n, h, w, 3) at most WIDTH_CAP wide, plus meta. Transparent pixels go on grey."""
    try:
        im = Image.open(path)
    except Exception:
        im = None
    if im is not None and (getattr(im, "n_frames", 1) > 1 or im.format == "GIF"):
        rgb, durs, fps = _decode_pillow(im)
        w0, h0 = im.size
    else:
        rgb, durs, fps, w0, h0 = _decode_ffmpeg(path)
    n, H, W = rgb.shape[:3]
    if n < 2:
        raise ValueError(f"{path.name} has {n} frame(s); motion checks need at least 2")
    meta = {"frames": int(n), "w": int(W), "h": int(H), "fps": fps, "durations": durs,
            "source_w": int(w0), "source_h": int(h0), "truncated": bool(n >= MAX_FRAMES)}
    return rgb, meta


# --------------------------------------------------------------------------- measures (pure)

def _luma(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame.astype(np.float32) @ LUMA + 0.5, 0, 255).astype(np.uint8)


def build_index(rgb: np.ndarray) -> Dict[str, Any]:
    """Every per-frame and per-pixel number the tools need, computed once per clip."""
    n, h, w = rgb.shape[:3]
    lum = np.stack([_luma(f) for f in rgb])
    hb, wb = h // BLOCK, w // BLOCK
    if hb == 0 or wb == 0:
        raise ValueError(f"frames must be at least {BLOCK} px on each side")
    blk = np.stack([lum[t][:hb * BLOCK, :wb * BLOCK].astype(np.float32)
                    .reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3)) for t in range(n)])
    still_b = np.median(np.abs(np.diff(blk, axis=0)), axis=0) < STILL_T
    still = np.pad(np.repeat(np.repeat(still_b, BLOCK, 0), BLOCK, 1),
                   ((0, h - hb * BLOCK), (0, w - wb * BLOCK)), mode="edge")
    change = np.empty(n, np.float64)
    speed = np.empty(n, np.float64)
    flicker_acc = np.zeros((h, w), np.float32)
    for t in range(n):
        cur = lum[t].astype(np.float32)
        prev = lum[t - 1].astype(np.float32)
        nxt = lum[(t + 1) % n].astype(np.float32)
        d = np.abs(cur - prev)
        change[t] = d.mean()
        speed[t] = np.where(d > STILL_T, d, 0.0).mean()
        flicker_acc += np.abs(cur - (prev + nxt) / 2)
    flicker = flicker_acc / n
    sub = np.unique(np.linspace(0, n - 1, min(n, 9)).round().astype(int))
    k = len(sub) // 2
    median = np.partition(rgb[sub], k, axis=0)[k]
    boil = float(flicker[still].mean()) if still.any() else 0.0
    return {
        "change": change, "speed": speed, "still": still, "flicker": flicker, "median": median,
        "scalars": {"boil": boil, "still_frac": float(still.mean())},
    }


def find_flags(change: np.ndarray, speed: np.ndarray) -> List[Dict[str, Any]]:
    """stutter, freeze, duplicate and loop_seam flags, each with its frame index."""
    n = len(change)
    flags: List[Dict[str, Any]] = []
    med_speed = float(np.median(speed))
    floor = STUTTER_FLOOR * med_speed
    for t in range(n):
        neigh = [speed[(t + k) % n] for k in (-2, -1, 1, 2)]
        base = max(float(np.median(neigh)), floor)
        if base > 0 and speed[t] > STUTTER_RATIO * base:
            flags.append({"kind": "stutter", "frame": t, "value": round(float(speed[t] / base), 2)})
        if change[t] < DUP_LEVEL:
            flags.append({"kind": "duplicate", "frame": t, "value": round(float(change[t]), 3)})
    if med_speed > 0:
        low = speed < FREEZE_RATIO * med_speed
        for t in range(n):
            if low[t] and not low[t - 1]:
                run = 0
                while run < n and low[(t + run) % n]:
                    run += 1
                if FREEZE_MIN_RUN <= run < n:
                    flags.append({"kind": "freeze", "frame": t, "value": run})
    rest_med = float(np.median(change[1:]))
    if rest_med > 0 and change[0] > SEAM_RATIO * rest_med:
        flags.append({"kind": "loop_seam", "frame": 0, "value": round(float(change[0] / rest_med), 2)})
    return sorted(flags, key=lambda f: (f["frame"], f["kind"]))


def _stats(index: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    change = index["change"]
    rest = change[1:]
    med = float(np.median(rest))
    seam = float(change[0])
    return {
        "boil": round(index["scalars"]["boil"], 3),
        "still_frac": round(index["scalars"]["still_frac"], 4),
        "seam_change": round(seam, 3),
        "median_neighbour_change": round(med, 3),
        "seam_ratio": round(seam / med, 2) if med > 0 else None,
        "mean_change": round(float(rest.mean()), 3),
        "change": [round(float(c), 2) for c in change],
        "frames": meta["frames"],
        "fps": meta["fps"],
    }


# --------------------------------------------------------------------------- warm store

class _Clip:
    def __init__(self, fingerprint: str, meta: Dict[str, Any], rgb: np.ndarray, index: Dict[str, Any]):
        self.fingerprint = fingerprint
        self.meta = meta
        self.rgb = rgb
        self.index = index
        self.nbytes = rgb.nbytes + sum(np.asarray(index[k]).nbytes for k in ARRAYS if k != "rgb")


_STORE: "OrderedDict[str, _Clip]" = OrderedDict()
_STORE_LOCK = threading.Lock()
_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_PENDING: Dict[str, threading.Thread] = {}
_FAILED: Dict[str, str] = {}  # fingerprint -> decode error, so motion_open can report it


def clear_memory_store() -> None:
    """Forget every in-memory clip (the disk copy stays). Tests and timing runs use it."""
    with _STORE_LOCK:
        _STORE.clear()


def _local_path(source: Any) -> Path:
    p = Path(str(source)).expanduser()
    if not p.is_file():
        raise ValueError(f"motion tools read a local file; {source} is not one. Download a URL with contact_sheet first, then pass the cached path.")
    return p


def _entry(fingerprint: str) -> Path:
    return MediaCache().get_entry_dir(fingerprint) / VERSION


def _write_disk(entry: Path, clip: _Clip) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    arrays = {"rgb": clip.rgb, **{k: clip.index[k] for k in ARRAYS if k != "rgb"}}
    for name, arr in arrays.items():
        tmp = entry / f".{name}.{tag}.npy"
        np.save(tmp, np.asarray(arr))
        os.replace(tmp, entry / f"{name}.npy")
    meta = dict(clip.meta, version=VERSION, fingerprint=clip.fingerprint, scalars=clip.index["scalars"])
    tmp = entry / f".meta.{tag}.json"  # meta is written last: it marks a complete entry
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, entry / "meta.json")


def _read_disk(entry: Path, fingerprint: str) -> Optional[_Clip]:
    meta_path = entry / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("version") != VERSION or meta.get("fingerprint") != fingerprint:
        return None
    rgb = np.load(entry / "rgb.npy")
    index = {k: np.load(entry / f"{k}.npy") for k in ARRAYS if k != "rgb"}
    index["scalars"] = meta["scalars"]
    return _Clip(fingerprint, {k: meta[k] for k in ("frames", "w", "h", "fps", "durations", "source_w", "source_h", "truncated")}, rgb, index)


def _evict_locked() -> None:
    while len(_STORE) > 1 and sum(c.nbytes for c in _STORE.values()) > STORE_MAX_BYTES:
        _STORE.popitem(last=False)


def _clip(source: Any) -> Tuple[_Clip, str]:
    """(clip, origin) with origin "memory", "disk" or "cold". One build per file, even when threads ask at once."""
    path = _local_path(source)
    fingerprint = MediaCache().compute_fingerprint(str(path))
    with _STORE_LOCK:
        hit = _STORE.get(fingerprint)
        if hit is not None:
            _STORE.move_to_end(fingerprint)
            return hit, "memory"
        lock = _BUILD_LOCKS.setdefault(fingerprint, threading.Lock())
    with lock:
        with _STORE_LOCK:
            hit = _STORE.get(fingerprint)
            if hit is not None:
                _STORE.move_to_end(fingerprint)
                return hit, "memory"
        entry = _entry(fingerprint)
        clip = _read_disk(entry, fingerprint)
        origin = "disk"
        if clip is None:
            rgb, meta = decode(path)
            clip = _Clip(fingerprint, meta, rgb, build_index(rgb))
            _write_disk(entry, clip)
            origin = "cold"
        with _STORE_LOCK:
            _STORE[fingerprint] = clip
            _STORE.move_to_end(fingerprint)
            _evict_locked()
        return clip, origin


def _build_in_background(path: str, fingerprint: str) -> None:
    try:
        _clip(path)
    except Exception as exc:  # a thread's exception is lost; keep it for motion_open to report
        with _STORE_LOCK:
            _FAILED[fingerprint] = str(exc)


def _tool_open(args: Dict[str, Any]) -> Dict[str, Any]:
    path = _local_path(args["source"])
    fingerprint = MediaCache().compute_fingerprint(str(path))
    with _STORE_LOCK:
        if fingerprint in _STORE:
            return {"status": "ready", "fingerprint": fingerprint, "origin": "memory"}
        if fingerprint in _FAILED:
            return {"status": "failed", "fingerprint": fingerprint, "error": _FAILED[fingerprint]}
        thread = _PENDING.get(fingerprint)
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=_build_in_background, args=(str(path), fingerprint), daemon=True, name=f"motion-{fingerprint[:8]}")
            _PENDING[fingerprint] = thread
            thread.start()
    return {"status": "in_progress", "retry_after_seconds": 1, "fingerprint": fingerprint}


# --------------------------------------------------------------------------- drawing helpers

def heat(t: np.ndarray) -> np.ndarray:
    """Colormap black -> red -> yellow for t in [0, 1]; float RGB 0..255."""
    t = np.clip(t, 0, 1)
    return np.stack([np.clip(2 * t, 0, 1), np.clip(2 * t - 1, 0, 1), np.zeros_like(t)], -1) * 255


def diff_heat(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean(-1)
    return heat(d * 4 / 255).astype(np.uint8)


def _label(img: Image.Image, text: str, xy: Tuple[int, int]) -> None:
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox(xy, text, font=FONT)
    d.rectangle((l - 2, t - 2, r + 2, b + 2), fill=(0, 0, 0))
    d.text(xy, text, font=FONT, fill=(255, 255, 255))


def _parse_box(box: Any, H: int, W: int) -> Tuple[int, int, int, int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must be [x, y, width, height] in pixels of the decoded frames")
    x, y, w, h = (int(v) for v in box)
    x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"box {box} is empty for a {W}x{H} frame")
    return x, y, w, h


def _parse_frames(s: Any, n: int) -> Tuple[int, int]:
    if s in (None, ""):
        return 0, n - 1
    a, _, b = str(s).partition("-")
    a = max(int(a), 0)
    b = min(int(b) if b else a, n - 1)
    if a > b:
        raise ValueError(f"frames {s} is empty for {n} frames (0-{n - 1})")
    return a, b


def _tool_file(path: Path) -> Dict[str, Any]:
    from src.executor.mcp_server import make_tool_file  # lazy: mcp_server imports this module
    return make_tool_file(str(path))


def _out_path(clip: _Clip, kind: str, key: str) -> Path:
    out_dir = MediaCache().get_entry_dir(clip.fingerprint) / "motion-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{kind}-{hashlib.sha1(f'{RENDER_VERSION}|{key}'.encode()).hexdigest()[:10]}.png"


def _reuse_png(clip: _Clip, kind: str, key: str) -> Optional[Dict[str, Any]]:
    """The saved image for the same clip, tool and settings, or None; saving the PNG is most of a warm call."""
    path = _out_path(clip, kind, key)
    if not path.is_file():
        return None
    with Image.open(path) as img:
        w, h = img.size
    return {"file_path": str(path), "w": w, "h": h, "tool_file": _tool_file(path)}


def _png(img: Image.Image, clip: _Clip, kind: str, key: str) -> Dict[str, Any]:
    path = _out_path(clip, kind, key)
    tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")  # unique per writer: two writers never share one file
    img.save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, path)  # a half-written file is never reused
    return {"file_path": str(path), "w": img.width, "h": img.height, "tool_file": _tool_file(path)}


# --------------------------------------------------------------------------- tools

def _tool_sheet(args: Dict[str, Any]) -> Dict[str, Any]:
    clip, origin = _clip(args["source"])
    rgb, meta = clip.rgb, clip.meta
    n, H, W = meta["frames"], meta["h"], meta["w"]
    cols = int(args.get("cols", 8))
    cw = int(args.get("cell", 128))
    ch = max(1, round(cw * H / W))
    cells = [np.asarray(Image.fromarray(np.ascontiguousarray(rgb[i])).resize((cw, ch), Image.Resampling.BILINEAR))
             for i in range(n)]
    rows = -(-n // cols)
    pad, top = 6, 28
    cell_h = 2 * ch + pad
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, top + rows * (cell_h + pad) + pad), (24, 24, 24))
    name = Path(str(args["source"])).name
    ImageDraw.Draw(sheet).text((pad, 6), f"{name} | {n} frames | {meta['fps']:.1f} fps", font=FONT_BIG, fill=(235, 235, 235))
    for i in range(n):
        r, c = divmod(i, cols)
        x, y = pad + c * (cw + pad), top + pad + r * (cell_h + pad)
        sheet.paste(Image.fromarray(cells[i]), (x, y))
        sheet.paste(Image.fromarray(diff_heat(cells[i], cells[i - 1])), (x, y + ch))
        _label(sheet, str(i), (x + 3, y + 3))
    out = _png(sheet, clip, "sheet", f"{cols}|{cw}")
    out.update(frames=n, fps=meta["fps"], change=[round(float(c), 2) for c in clip.index["change"]], origin=origin)
    return out


def _tool_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    clip, origin = _clip(args["source"])
    meta = clip.meta
    if args.get("box") is not None:
        x, y, w, h = _parse_box(args["box"], meta["h"], meta["w"])
        if min(w, h) < BLOCK:
            raise ValueError(f"box {args['box']} is {w}x{h} px inside the {meta['w']}x{meta['h']} frame; "
                             f"motion_stats needs at least {BLOCK} px on each side. Use a larger box.")
        index = build_index(clip.rgb[:, y:y + h, x:x + w])
        out = _stats(index, meta)
        out.update(box=[x, y, w, h])
    else:
        out = _stats(clip.index, meta)
    out.update(w=meta["w"], h=meta["h"], origin=origin)
    return out


def _tool_boil_map(args: Dict[str, Any]) -> Dict[str, Any]:
    clip, origin = _clip(args["source"])
    gain = float(args.get("gain", 4.0))
    base = clip.index["median"].astype(np.float32) * 0.35
    flick = clip.index["flicker"]
    t = np.clip(flick / gain, 0, 1)
    mixed = base * (1 - t[..., None]) + heat(t) * t[..., None]
    still = clip.index["still"][..., None]
    img = np.clip(np.where(still, mixed, base) + 0.5, 0, 255).astype(np.uint8)
    out = _png(Image.fromarray(img), clip, "boil", f"{gain}")
    out.update(boil=round(clip.index["scalars"]["boil"], 3), still_frac=round(clip.index["scalars"]["still_frac"], 4),
               max_flicker=round(float(flick.max()), 2), gain=gain, origin=origin)
    return out


def _tool_zoom(args: Dict[str, Any]) -> Dict[str, Any]:
    clip, origin = _clip(args["source"])
    rgb, meta = clip.rgb, clip.meta
    n, H, W = meta["frames"], meta["h"], meta["w"]
    x, y, w, h = _parse_box(args["box"], H, W)
    f0, f1 = _parse_frames(args.get("frames"), n)
    idx = list(range(f0, f1 + 1))
    gap, lab = 6, 20  # gap between frame columns; label strip above the rows
    # Largest zoom that keeps the canvas within MAX_PX on each side: columns of w*z px with gaps, rows of lab + 2*h*z.
    z = min(int(args.get("zoom", 4)), (MAX_PX - (len(idx) - 1) * gap) // (len(idx) * w), (MAX_PX - lab) // (2 * h))
    if z < 1:
        too_wide = len(idx) * w + (len(idx) - 1) * gap
        raise ValueError(f"box [{x}, {y}, {w}, {h}] over frames {f0}-{f1} is too large: at zoom 1 the image would be "
                         f"{too_wide}x{lab + 2 * h} px, past the {MAX_PX} px limit. Use a smaller box or fewer frames.")
    key = f"{x},{y},{w},{h}|{f0}-{f1}|{z}"
    out = _reuse_png(clip, "zoom", key)
    if out is not None:
        out.update(box=[x, y, w, h], frames=[f0, f1], zoom=z, origin=origin)
        return out
    cw = w * z
    canvas = Image.new("RGB", (len(idx) * (cw + gap) - gap, lab + 2 * h * z), (24, 24, 24))

    def up(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, z, 0), z, 1)

    for k, f in enumerate(idx):
        cur = rgb[f, y:y + h, x:x + w]
        prev = rgb[(f - 1) % n, y:y + h, x:x + w]
        X = k * (cw + gap)
        canvas.paste(Image.fromarray(np.ascontiguousarray(up(cur))), (X, lab))
        canvas.paste(Image.fromarray(np.ascontiguousarray(up(diff_heat(cur, prev)))), (X, lab + h * z))
        _label(canvas, f"f{f}", (X + 2, 2))
    out = _png(canvas, clip, "zoom", key)
    out.update(box=[x, y, w, h], frames=[f0, f1], zoom=z, origin=origin)
    return out


def _tool_profile(args: Dict[str, Any]) -> Dict[str, Any]:
    clip, origin = _clip(args["source"])
    idx = clip.index
    flags = find_flags(idx["change"], idx["speed"])
    counts: Dict[str, int] = {}
    for f in flags:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    graph = _graph(idx["speed"], idx["change"], flags)
    out = _png(graph, clip, "profile", "v1")
    out.update(frames=clip.meta["frames"], fps=clip.meta["fps"],
               speed=[round(float(s), 3) for s in idx["speed"]],
               change=[round(float(c), 3) for c in idx["change"]],
               flags=flags[:200], flag_counts=counts, origin=origin,
               note="speed counts changes above 2 luma levels (no optical flow)")
    return out


def _graph(speed: np.ndarray, change: np.ndarray, flags: List[Dict[str, Any]], w: int = 640, h: int = 220) -> Image.Image:
    img = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    n = len(speed)
    top, bottom, left, right = 24, h - 18, 8, w - 8
    ymax = max(float(speed.max()), float(change.max()), 1e-6)

    def xy(i: int, v: float) -> Tuple[float, float]:
        return (left + (right - left) * i / max(n - 1, 1), bottom - (bottom - top) * v / ymax)

    d.line([xy(i, float(v)) for i, v in enumerate(change)], fill=(255, 170, 60), width=1)
    d.line([xy(i, float(v)) for i, v in enumerate(speed)], fill=(90, 170, 255), width=2)
    for f in flags:
        x, _ = xy(f["frame"], 0)
        d.line([(x, top), (x, bottom)], fill=(255, 60, 60))
    d.text((left, 4), f"speed (blue), change (orange), max {ymax:.2f}; red = flag", font=FONT, fill=(235, 235, 235))
    return img


def _tool_compare(args: Dict[str, Any]) -> Dict[str, Any]:
    a, _ = _clip(args["source"])
    b, _ = _clip(args["compare_to"])
    n = min(a.meta["frames"], b.meta["frames"])
    frame = int(args.get("frame", 0))
    if not 0 <= frame < n:
        raise ValueError(f"frame {frame} out of range 0..{n - 1}")
    fa = a.rgb[frame]
    fb = b.rgb[frame]
    if fa.shape != fb.shape:
        raise ValueError(f"size mismatch {fa.shape[1]}x{fa.shape[0]} vs {fb.shape[1]}x{fb.shape[0]}")
    H, W = fa.shape[:2]
    if args.get("box") is not None:
        x, y, w, h = _parse_box(args["box"], H, W)
        fa, fb = fa[y:y + h, x:x + w], fb[y:y + h, x:x + w]
    h, w = fa.shape[:2]
    gap, lab = 8, 20
    canvas = Image.new("RGB", (3 * w + 2 * gap, h + lab), (24, 24, 24))
    canvas.paste(Image.fromarray(np.ascontiguousarray(fa)), (0, lab))
    canvas.paste(Image.fromarray(np.ascontiguousarray(fb)), (w + gap, lab))
    canvas.paste(Image.fromarray(np.ascontiguousarray(diff_heat(fa, fb))), (2 * (w + gap), lab))
    _label(canvas, f"A {Path(args['source']).name} f{frame}", (2, 2))
    _label(canvas, f"B {Path(args['compare_to']).name} f{frame}", (w + gap + 2, 2))
    _label(canvas, "diff x4", (2 * (w + gap) + 2, 2))
    key = f"{args['source']}|{args['compare_to']}|{frame}|{args.get('box')}"
    out = _png(canvas, a, "compare", key)
    out.update(frame=frame, mean_abs_diff=round(float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean()), 3))
    return out


def _plant(rgb: np.ndarray, fault: str, still: np.ndarray) -> np.ndarray:
    n = len(rgb)
    k = n // 3
    if fault == "dropped_frames":
        return np.delete(rgb, [k, k + 1, k + 2], axis=0)
    if fault == "duplicated_frame":
        return np.insert(rgb, k + 1, rgb[k], axis=0)
    if fault == "frozen_hold":
        out = rgb.copy()
        out[k + 1:k + 4] = rgb[k]
        return out
    if fault == "swapped_frames":
        out = rgb.copy()
        out[[k, k + 5]] = out[[k + 5, k]]
        return out
    if fault == "noise_on_still":
        rng = np.random.default_rng(7)
        noise = rng.integers(-3, 4, size=rgb.shape, dtype=np.int16)
        mixed = rgb.astype(np.int16) + noise * still[None, :, :, None]
        return np.clip(mixed, 0, 255).astype(np.uint8)
    if fault == "loop_seam_jump":
        return rgb[:-8]
    raise ValueError(f"unknown fault {fault}")


def selftest_report(rgb: np.ndarray) -> Dict[str, Any]:
    clean = build_index(rgb)
    clean_flags = find_flags(clean["change"], clean["speed"])
    clean_boil = clean["scalars"]["boil"]
    kinds = {"dropped_frames": "stutter", "swapped_frames": "stutter", "duplicated_frame": "duplicate",
             "frozen_hold": "freeze", "loop_seam_jump": "loop_seam"}
    detectors: Dict[str, Dict[str, Any]] = {}

    def run_fault(fault: str) -> Dict[str, Any]:
        idx = build_index(_plant(rgb, fault, clean["still"]))
        flags = find_flags(idx["change"], idx["speed"])
        hits = [f for f in flags if f["kind"] == kinds[fault]]
        return {"caught": bool(hits), "evidence": hits[:2]}

    def detector(name: str, faults: Dict[str, Dict[str, Any]], clean_hits: int) -> None:
        detectors[name] = {"pass": all(f["caught"] for f in faults.values()) and clean_hits == 0,
                           "clean_flags": clean_hits, "faults": faults}

    stutter = {f: run_fault(f) for f in ("dropped_frames", "swapped_frames")}
    detector("stutter", stutter, sum(f["kind"] == "stutter" for f in clean_flags))
    detector("duplicate", {"duplicated_frame": run_fault("duplicated_frame")}, sum(f["kind"] == "duplicate" for f in clean_flags))
    detector("freeze", {"frozen_hold": run_fault("frozen_hold")}, sum(f["kind"] == "freeze" for f in clean_flags))
    detector("loop_seam", {"loop_seam_jump": run_fault("loop_seam_jump")}, sum(f["kind"] == "loop_seam" for f in clean_flags))
    noisy = build_index(_plant(rgb, "noise_on_still", clean["still"]))["scalars"]["boil"]
    delta = noisy - clean_boil
    detectors["boil"] = {"pass": delta > BOIL_FAULT_DELTA and clean_boil <= BOIL_FAULT_DELTA, "clean_flags": 0,
                         "faults": {"noise_on_still": {"caught": delta > BOIL_FAULT_DELTA,
                                                       "boil_clean": round(clean_boil, 3), "boil_planted": round(noisy, 3)}}}
    return {"pass": all(d["pass"] for d in detectors.values()) and not clean_flags,
            "clean_flag_count": len(clean_flags), "detectors": detectors}


def _tool_selftest(args: Dict[str, Any]) -> Dict[str, Any]:
    path = _local_path(args["source"]) if args.get("source") else DEFAULT_FIXTURE
    rgb, meta = decode(path)
    report = selftest_report(rgb)
    report.update(source=str(path), frames=meta["frames"])
    return report


TOOLS = {
    "motion_open": _tool_open,
    "motion_sheet": _tool_sheet,
    "motion_stats": _tool_stats,
    "motion_boil_map": _tool_boil_map,
    "motion_zoom": _tool_zoom,
    "motion_profile": _tool_profile,
    "motion_compare": _tool_compare,
    "motion_selftest": _tool_selftest,
}


def dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOLS:
        raise ValueError(f"unknown motion tool {name}")
    try:
        return {"ok": True, "data": TOOLS[name](args)}
    except ValueError as exc:  # bad input comes back as an error result whose message names the parameter
        return {"ok": False, "error": {"code": "invalid_argument", "message": str(exc)}}
