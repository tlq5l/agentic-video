import sys, time, json
sys.path.insert(0, ".")
import src.media.audio as A
import src.retrieval.engine as E
T = {}
def wrap(mod, name):
    fn = getattr(mod, name)
    def w(*a, **k):
        t = time.perf_counter(); r = fn(*a, **k); T[name] = T.get(name, 0) + time.perf_counter() - t; return r
    setattr(mod, name, w)
for n in ["audio_onsets", "_decode_pcm", "_learn_cue_signatures", "_overlap_templates", "_temporal_group_matches", "_simple_dominant_band_check"]:
    wrap(A, n)
wrap(E.AgenticVideoEngine, "detect_cuts")
from src.executor.mcp_server import handle_call
f = lambda x: round(x * 3600 * 1000) / 1000
cues = [{"time_sec": 0.0, "label": "kick"}, {"time_sec": f(0.08), "label": "kick"}, {"time_sec": f(0.08), "label": "snare"},
 {"time_sec": f(0.2), "label": "kick"}, {"time_sec": f(0.2) + 0.5, "label": "hat"}, {"time_sec": f(0.33), "label": "snare"},
 {"time_sec": f(0.45), "label": "kick"}, {"time_sec": f(0.5), "label": "zap", "band_hz": "high"},
 {"time_sec": f(0.6), "label": "snare"}, {"time_sec": f(0.6), "label": "kick"}, {"time_sec": f(0.72), "label": "hat"},
 {"time_sec": f(0.9), "label": "zap", "band_hz": [2000, 6000]}, {"time_sec": f(0.95), "label": "kick"}]
t = time.perf_counter()
r = handle_call("sync_check", {"source": sys.argv[1], "cues": cues, "bpm": 120, "tolerance_ms": 81})
T["TOTAL"] = time.perf_counter() - t
print(json.dumps({k: round(v, 1) for k, v in T.items()}), str(r)[:80])
