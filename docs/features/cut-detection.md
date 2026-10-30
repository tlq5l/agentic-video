# Cut detection

How a user reaches it: the `detect_cuts` tool, `sync_check`, and `python -m src.loop check` (its cut list).

Code: `src/media/scene.py` (`detect_scenes`, `scan_scene_changes`, `detect_soft_transitions`, `_debounce`, `_sharp_step`) and `src/retrieval/engine.py` (`detect_cuts`).

How to check it works: `pytest tests/test_cut_strongest_frame.py tests/test_media.py tests/test_soft_changes_e2e.py tests/test_scene_lazy_path.py`.

What depends on it: `src/loop.py` (cut list for the check loop), `src/media/audio.py` (`sync_check`), `src/adapters/model_adapter.py` (window around the first cut), cut-aligned boundaries in `src/retrieval/engine.py`, and `partition_into_scene_segments`.

Gotchas:
- A hard cut is a frame whose ffmpeg scene score is over the threshold (default 0.25). Frames next to each other that are all over it are one change, and the cut is reported at the strongest of them. A partly changed frame just before a cut can score just over the threshold; reporting the first such frame put those cuts one frame early.
- "Next to each other" means at most 1.5 frame lengths apart, with the frame rate from `probe_video`. If the frame rate cannot be read, no frames are joined and the first frame over the threshold is reported.
- After that, a cut closer than 0.5 s to the last kept cut is dropped, even when it is the stronger one.
- A cut found by the soft scan (score under the threshold) is timed the same way: at the stronger of its one or two frames.
- Two paths must give the same answer: `detect_scenes` on its own (ffmpeg returns only the frames over the threshold) and the one-decode scan in `scan_scene_changes` (cached scores of every frame over 0.05).
- A soft peak whose colours do not change much and whose picture is not a dissolve (blend error above 0.28) is kept only when both sides are still (under 0.015) and it is at least 0.035 strong. A weaker one is not dropped outright: it is kept only if the frame-step check (below) finds a cut.
- A soft peak under 0.06 whose blend error is between 0.1 and 0.28 is dropped when its change stays in part of the frame (area under 0.9): weak motion that is not a dissolve. A fade or a whole-frame step changes every cell and is kept. Blend alone cannot tell them apart: a short fade sampled off its midpoint has a blend error near 0.2.
- A peak that the still-sides test drops (motion on a side, or under 0.035 strong) is kept as a cut when the frame step at it is one cut (the same check as a hard cut that the scene score missed). Other dropped peaks stay dropped. On a large video the candidate carries a `rescue` marker and is settled later by `_settle_changes`, which removes it if it does not settle as a cut.
- Result cache keys are versioned (`scenes_v2_th…`, `softtrans_v13_…`); bump them when the stored answer changes. `scenescores_v1_…` holds raw scores.
