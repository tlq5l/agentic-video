"""sync_check timing and output dump for one input, run from one code tree.

    python benchmarks/speed/sync_check.py REPO CACHEDIR INPUT OUTDIR [case ...]

Cases: sync_{cues6,rich,nocues}[_bpm] call audio.sync_check directly; hc_{cues6,rich}[_bpm]
go through handle_call("sync_check") (video inputs only; cut detection needs a video stream).
"rich" is 13 cues (kick/snare/hat/zap, two with band_hz, coincident pairs); bpm is 120.
Each case writes json.dumps(result) to OUTDIR/<case>.json, so two trees can be compared with cmp.
Onset/loudness result caches are bypassed unless BENCH_WARM=1; the cut cache is warmed before
timing unless BENCH_COLD=1. Prints (wall s, CPU s incl. children) per case.
"""
import json
import os
import resource
import sys
import time

repo, cachedir, path, outdir = sys.argv[1:5]
only = sys.argv[5:]
os.environ["AGENTIC_VIDEO_CACHE_DIR"] = cachedir
os.environ["AGENTIC_VIDEO_TEMP_DIR"] = os.path.join(cachedir, "tmp")
sys.path.insert(0, repo)
from src.core.cache import MediaCache  # noqa: E402
from src.executor.mcp_server import handle_call  # noqa: E402
from src.media import audio  # noqa: E402

assert audio.__file__.startswith(os.path.realpath(repo)) or audio.__file__.startswith(repo), audio.__file__
if not os.environ.get("BENCH_WARM"):
    orig = MediaCache.load_json
    MediaCache.load_json = lambda self, fp, key: None if str(key).startswith(("audio_onsets_", "loudness_timeline_")) else orig(self, fp, key)
os.makedirs(outdir, exist_ok=True)


def cpu():
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


_t0 = time.perf_counter()
dur = len(audio._decode_pcm(path)[0]) / 48000.0
print("pcm_decode_wall", round(time.perf_counter() - _t0, 1), flush=True)
has_video = not path.endswith(".m4a")


def cut_detector():
    if not has_video:
        return []
    from src.retrieval.engine import AgenticVideoEngine
    return AgenticVideoEngine(path).detect_cuts()


def f(x):
    return round(dur * x, 2)


cues6 = [{"time_sec": f(x), "label": "hit"} for x in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]
rich = [
    {"time_sec": 0.0, "label": "kick"}, {"time_sec": f(0.08), "label": "kick"}, {"time_sec": f(0.08), "label": "snare"},
    {"time_sec": f(0.2), "label": "kick"}, {"time_sec": f(0.2) + 0.5, "label": "hat"}, {"time_sec": f(0.33), "label": "snare"},
    {"time_sec": f(0.45), "label": "kick"}, {"time_sec": f(0.5), "label": "zap", "band_hz": "high"},
    {"time_sec": f(0.6), "label": "snare"}, {"time_sec": f(0.6), "label": "kick"}, {"time_sec": f(0.72), "label": "hat"},
    {"time_sec": f(0.9), "label": "zap", "band_hz": [2000, 6000]}, {"time_sec": f(0.95), "label": "kick"},
]
cases = {}
for cname, cues in (("cues6", cues6), ("rich", rich), ("nocues", None)):
    for bpm in (None, 120.0):
        key = "sync_" + cname + ("_bpm" if bpm else "")
        cases[key] = (lambda cues=cues, bpm=bpm: audio.sync_check(path, cues=cues, cut_detector=cut_detector, bpm=bpm))
for cname, cues in (("cues6", cues6), ("rich", rich)):
    for bpm in (None, 120):
        args = {"source": path, "cues": cues}
        if bpm:
            args["bpm"] = bpm
        cases["hc_" + cname + ("_bpm" if bpm else "")] = (lambda args=args: handle_call("sync_check", args))
if has_video and not os.environ.get("BENCH_COLD"):
    cut_detector()  # warm the cut cache so timings are sound work only
timing = {}
for name, fn in cases.items():
    if only and name not in only:
        continue
    t0, c0 = time.perf_counter(), cpu()
    r = fn()
    timing[name] = (round(time.perf_counter() - t0, 1), round(cpu() - c0, 1))
    with open(os.path.join(outdir, name + ".json"), "w") as fh:
        json.dump(r, fh)
    print(name, timing[name], flush=True)
print("peak_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824, 2), flush=True)
json.dump(timing, open(os.path.join(outdir, "_timing.txt"), "w"))
