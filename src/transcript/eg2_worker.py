"""EmbeddingGemma 2 text embedder on the Apple Neural Engine (Core ML), run as a child process.

Standalone on purpose: the parent starts it with a configurable interpreter (one that has
coremltools, tokenizers and numpy), so the main environment needs none of them.

Protocol on stdin/stdout, one JSON object per line:
  request   {"id": 1, "prompt": "Document" | "SearchQuery", "texts": ["..."]}
  response  {"id": 1, "vectors": "<base64 float32, [n, 768], L2-normalised>"} or {"id": 1, "error": "..."}
The model loads on the first request, once per process (Core ML function `pack_256`; a text over 256 tokens loads `embed_512` as a second MLModel on first use). The process exits when stdin closes
(parent gone) or after AGENTIC_VIDEO_EG2_IDLE_SEC seconds without a request (default and maximum 600).
SIGTERM and SIGINT make it exit normally: Core ML leaves a 0.5 to 0.85 GB EmbeddingGemma2Text_<UUID>.mlmodelc folder in
TMPDIR for every loaded model unless the process exits normally (SIGKILL leaks it), so the loaded models are released before exit.
Text only: no GPU model is ever loaded here, so Neural Engine and GPU models never share a process.

Usage: python eg2_worker.py <model_dir>   (model_dir holds EmbeddingGemma2Text.mlpackage,
embeddings.bf16, tokenizer.json and config.json)
"""
import base64
import gc
import json
import math
import os
import re
import select
import signal
import sys
from pathlib import Path

PACK_LEN, PACK_SLOTS, MAX_TOKENS = 256, 8, 512
TEXT_SIZES = [32, 48, 64, 128, 256, 512]
MAX_IDLE_SEC = 600.0


class Embedder:
    def __init__(self, model_dir: Path):
        # coremltools imports torch when it is installed (about 225 MB resident); prediction does not need it.
        for unused in ("torch", "tensorflow"):
            sys.modules.setdefault(unused, None)  # None makes `import torch` raise ImportError, which coremltools handles
        import numpy as np
        import coremltools as ct
        from tokenizers import Tokenizer

        self.np, self.ct, self.dir = np, ct, model_dir
        self.cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.dim = self.cfg["embedding_dim"]
        self.scale = math.sqrt(self.cfg["hidden_size"])
        self.table = np.memmap(model_dir / "embeddings.bf16", dtype=np.uint16, mode="r",
                               shape=tuple(self.cfg["token_table"]["shape"]))
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.models = {}

    def _model(self, function_name):
        if function_name not in self.models:
            self.models[function_name] = self.ct.models.MLModel(
                str(self.dir / "EmbeddingGemma2Text.mlpackage"),
                function_name=function_name, compute_units=self.ct.ComputeUnit.CPU_AND_NE)
        return self.models[function_name]

    def _rows(self, ids):
        np = self.np
        bits = self.table[np.asarray(ids, dtype=np.int64)].astype(np.uint32) << 16
        return (bits.view(np.float32) * np.float32(self.scale)).astype(np.float16)

    def _ids(self, text, prefix):
        ids = self.tok.encode(prefix + text, add_special_tokens=False).ids
        return [self.cfg["bos_token_id"]] + ids[: MAX_TOKENS - 2] + [self.cfg["eos_token_id"]]

    def _pack(self, id_lists):
        """Up to 8 short texts in one pack_256 call."""
        np, pad = self.np, self.cfg["pad_token_id"]
        flat, off = [], 0
        positions = np.zeros((PACK_LEN, 1), dtype=np.float16)
        bias = np.full((1, 1, PACK_LEN, PACK_LEN), -1e4, dtype=np.float16)
        pool = np.zeros((PACK_SLOTS, PACK_LEN), dtype=np.float16)
        for j, ids in enumerate(id_lists):
            n = len(ids)
            flat.extend(ids)
            bias[0, 0, off:off + n, off:off + n] = 0
            positions[off:off + n, 0] = np.arange(n, dtype=np.float16)
            pool[j, off:off + n] = np.float16(1.0 / n)
            off += n
        flat.extend([pad] * (PACK_LEN - off))
        np.fill_diagonal(bias[0, 0, off:, off:], 0)
        out = self._model("pack_256").predict({
            "inputs_embeds": self._rows(flat)[None], "attention_bias": bias,
            "positions": positions, "pool": pool})
        return np.asarray(out["embedding"], dtype=np.float32)[: len(id_lists)]

    def _single(self, ids):
        """One long text (over 256 tokens) through the smallest embed_S that fits."""
        np = self.np
        size = next(s for s in TEXT_SIZES if s >= len(ids))
        emb = np.empty((1, size, self.cfg["hidden_size"]), dtype=np.float16)
        emb[0] = self._rows(ids + [self.cfg["pad_token_id"]] * (size - len(ids)))
        mask = np.zeros((1, size), dtype=np.float16)
        mask[0, :len(ids)] = 1
        out = self._model(f"embed_{size}").predict({"inputs_embeds": emb, "attention_mask": mask})
        return np.asarray(out["embedding"], dtype=np.float32)[0]

    def embed(self, texts, prompt_name):
        np = self.np
        id_lists = []
        if texts:
            prefix = self.cfg["prompts"][re.sub(r"(?<!^)(?=[A-Z])", "_", prompt_name).lower()]
            id_lists = [self._ids(t, prefix) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        group, used = [], 0
        for i, ids in enumerate(id_lists):
            if len(ids) > PACK_LEN:
                out[i] = self._single(ids)
                continue
            if group and (len(group) == PACK_SLOTS or used + len(ids) > PACK_LEN):
                out[group] = self._pack([id_lists[g] for g in group])
                group, used = [], 0
            group.append(i)
            used += len(ids)
        if group:
            out[group] = self._pack([id_lists[g] for g in group])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _serve(model_dir: Path, embedders: list) -> None:
    idle = min(float(os.environ.get("AGENTIC_VIDEO_EG2_IDLE_SEC", MAX_IDLE_SEC)), MAX_IDLE_SEC)
    replies = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # stray prints from libraries go to stderr, never into the protocol
    pending = b""
    while True:
        while b"\n" not in pending:
            ready, _, _ = select.select([0], [], [], idle)
            if not ready:
                return  # idle too long: exit and free the model
            chunk = os.read(0, 1 << 20)
            if not chunk:
                return  # parent closed the pipe
            pending += chunk
        line, pending = pending.split(b"\n", 1)
        request = json.loads(line)
        try:
            if not embedders:
                embedders.append(Embedder(model_dir))
            vectors = embedders[0].embed(request["texts"], request["prompt"])
            reply = {"id": request["id"], "vectors": base64.b64encode(vectors.tobytes()).decode("ascii")}
        except Exception as exc:  # reported to the parent; the worker keeps serving
            reply = {"id": request["id"], "error": f"{type(exc).__name__}: {exc}"}
        try:
            replies.write(json.dumps(reply) + "\n")
            replies.flush()
        except BrokenPipeError:
            return  # the parent is gone: exit normally


def main():
    def _stop(signum, frame):
        raise SystemExit(0)  # unwinds to the `finally` below, so the exit is a normal one

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    embedders: list = []
    try:
        _serve(Path(sys.argv[1]), embedders)
    finally:
        for embedder in embedders:
            embedder.models.clear()  # releasing the MLModels is what removes their TMPDIR folders
        embedders.clear()
        gc.collect()


if __name__ == "__main__":
    main()
