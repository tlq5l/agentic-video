# Speed pass log

Method: baseline and candidate code are unpacked side by side and run alternately
(A,B,A,B...) with `benchmarks/speed/run.py`; numbers are CPU s (user+sys, incl. ffmpeg
children) of the tool in the cold run. The machine is shared, so only same-pair
comparisons count. Keep rule: wins 3 of 4 pairs, no QUALITY_FAIL, >=10% on the target tool.

## 1. detect_cuts: gray frames only around candidate transitions (KEPT)
Idea: the scan scaled every frame to 32x18 gray, but only the +-0.6 s around each
candidate transition is read (`_sharp_step`). Scan without that branch; then read those
stretches with a short seek decode (frame times from `showinfo` are checked against the
scan; any mismatch falls back to the old all-frames scan). Stretches are fetched 4 at a time.
Result (4 pairs, quiet machine, detect_cuts CPU s, base -> new):
clip_415s 52.8/56.5/51.2/51.4 -> 46.6/45.2/45.8/46.0 (-11%);
x_post_127s 16.6/16.3/16.5/16.6 -> 13.3/13.1/13.2/13.1 (-20%). 4/4 pairs, no QUALITY_FAIL.
Wall of the scan unchanged (clip_415s about 5 s) once the stretches run in parallel
(sequentially they added 2-3 s).

## 2. detect_cuts: ffmpeg -hwaccel videotoolbox for the scan (UNDONE)
Idea: hardware H.264 decode. Output was byte-identical (md5 of soft.raw, native.raw, scene
txt on 60 s) and CPU fell about 20% (clip_415s scan 47 -> 37 CPU s). But the hardware
decoder is the bottleneck when 8 parts run at once: scan wall 5 s -> 7.5 s on a 7 min
video (1 part: 23 s), which would push an hour-long video past the 60 s Executor limit
(software 8 parts: about 43 s; hardware: about 64 s). Using hardware for only 2 of 8 parts
keeps the wall but saves under 5% CPU. Undone.
