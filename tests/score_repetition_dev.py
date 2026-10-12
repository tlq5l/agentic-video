"""Optional full synthetic scorer: python tests/score_repetition_dev.py"""
import sys
import tempfile
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.core.cache import MediaCache
from src.retrieval.engine import AgenticVideoEngine
from src.retrieval import repetition
from tests.fixtures.repetition_dev import CASES, generate


class PixelEncoder:
    """Deterministic stand-in for CLIP; measures the embedding algorithm without model downloads."""
    def images(self, paths):
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("L").resize((32,24)), dtype=np.float32).ravel() / 255
            rows.append(pixels / max(np.linalg.norm(pixels), 1e-12))
        return np.asarray(rows)


def main():
    original = repetition._embed_frames
    def pixels(frames, temp_dir):
        encoder = PixelEncoder()
        paths = []
        for i, frame in enumerate(frames):
            path = Path(temp_dir)/f"pixel_{i}.jpg"
            Image.fromarray(frame).save(path)
            paths.append(path)
        return encoder.images(paths)
    use_clip = "--clip" in sys.argv
    backend = original if use_clip else pixels
    memo = {}
    def cached_embeddings(frames, temp_dir):
        digest = hashlib.sha256()
        for frame in frames:
            digest.update(frame.tobytes())
        key = digest.digest()
        if key not in memo:
            memo[key] = backend(frames, temp_dir)
        return memo[key]
    repetition._embed_frames = cached_embeddings
    print("embedding backend:", "local CLIP" if use_clip else "deterministic pixel proxy")
    scores = {m: [] for m in ("motion", "embedding", "auto")}
    try:
        with tempfile.TemporaryDirectory(prefix="repcount-dev-") as td:
            root = Path(td)
            for case in CASES:
                memo.clear()
                source = root/f"{case[0]}.mp4"
                truth = generate(case, source)
                engine = AgenticVideoEngine(str(source), cache=MediaCache(root/"cache"))
                predictions = {}
                for method in scores:
                    result = repetition.count_video(engine, method=method)
                    predictions[method] = result["count"]
                    scores[method].append(abs(result["count"]-truth))
                print(f"{case[0]:18} truth={truth:2} " + " ".join(f"{m}={predictions[m]:2}" for m in scores))
    finally:
        repetition._embed_frames = original
    for method, errors in scores.items():
        exact = sum(e == 0 for e in errors)/len(errors)
        near = sum(e <= 1 for e in errors)/len(errors)
        print(f"{method:10} exact={exact:.1%} +/-1={near:.1%}")


if __name__ == "__main__":
    main()
