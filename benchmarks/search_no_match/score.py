"""Score how search_transcript answers queries whose subject is not in the speech.

Calls search_transcript through src.executor.mcp_server.handle_call, the way an agent does (hybrid mode,
top_k 5). Needs the EmbeddingGemma 2 Core ML model and an interpreter that has coremltools, tokenizers
and numpy, set in the environment (AGENTIC_VIDEO_EG2_DIR, AGENTIC_VIDEO_EG2_PYTHON). Without them the
semantic half is off and the script stops, because the climb metric would read zero.

    python benchmarks/search_no_match/score.py [--baseline FILE] [--out FILE] [--heldout]

Splits: list position in questions.json, even = train, odd = test. Train numbers are printed on every
run. Test numbers are printed only with --heldout (measure them once per commit, never to tune).

Metrics (METRIC lines):
  climb_nomatch_semantic_share   share of no-match queries that return any match=semantic hit (lower is better)
  guard_positive_top5            share of reworded-line queries whose top 5 hits cover the target line (higher)
  guard_lexical_dropped          exact/all_terms hits of the baseline run that are missing now (must be 0)
  guard_fields_changed           hits or responses whose field names differ from the expected shape (must be 0)
  diag_nomatch_semantic_hits     mean number of semantic hits per no-match query (diagnostic)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.executor.mcp_server import handle_call  # noqa: E402

QUESTIONS = Path(__file__).with_name("questions.json")
EXPECTED_HIT_FIELDS = {"segment_id", "start_sec", "end_sec", "text", "score",
                       "start_timecode", "end_timecode", "context", "match"}
OPTIONAL_HIT_FIELDS = {"suspect"}  # present only on lines flagged as repeated or invented at run time
EXPECTED_HIT_LABELS = {"exact", "all_terms", "semantic"}
EXPECTED_TOP_KEYS = {"hits", "total", "note", "requested_model_size", "cached_model_size",
                     "requested_model_cached", "transcription_status", "retry_after_seconds",
                     "transcription_message", "model_substitution"}
TARGET_TOLERANCE_SEC = 0.5
TOP_K = 5


def call(source, query, mode):
    """search_transcript as an agent calls it. Waits while a new video's semantic index is still building."""
    for _ in range(24):
        response = handle_call("search_transcript", {"source": source, "query": query, "top_k": TOP_K, "mode": mode})
        if not response["ok"]:
            raise SystemExit(f"search_transcript failed for {query!r}: {response}")
        data = response["data"]
        note = data.get("note", "")
        if "still building" in note:
            time.sleep(5)
            continue
        if mode == "hybrid" and "Semantic search is off" in note:
            raise SystemExit(f"the semantic half cannot start, so this run would not measure anything: {note}")
        return data
    raise SystemExit(f"the semantic index for {source} never finished building")


def covers(hit, start_sec):
    return hit["start_sec"] - TARGET_TOLERANCE_SEC <= start_sec <= hit["end_sec"] + TARGET_TOLERANCE_SEC


def measure(questions):
    source = questions["source"]
    records = []
    plan = [("no_match", i, query, None) for i, query in enumerate(questions["no_match"])]
    plan += [("positive", i, item["query"], item) for i, item in enumerate(questions["positive"])]
    for kind, index, query, item in plan:
        hybrid = call(source, query, "hybrid")
        lexical = call(source, query, "lexical")
        hits = hybrid["hits"]
        semantic_scores = [h["score"] for h in hits if h["match"] == "semantic"]
        record = {
            "kind": kind,
            "split": "train" if index % 2 == 0 else "test",
            "index": index,
            "query": query,
            "semantic_scores": semantic_scores,
            "lexical_labelled": sorted([h["segment_id"], h["match"]] for h in hits if h["match"] in ("exact", "all_terms")),
            "lexical_top": sorted(h["segment_id"] for h in lexical["hits"]),
            "top_keys": sorted(hybrid.keys()),
            "hit_keys_ok": all(set(h) - OPTIONAL_HIT_FIELDS == EXPECTED_HIT_FIELDS and h["match"] in EXPECTED_HIT_LABELS
                               for h in hits),
            "top_keys_ok": set(hybrid) <= EXPECTED_TOP_KEYS,
        }
        if item is not None:
            record["target_segment_id"] = item["segment_id"]
            record["target_start_sec"] = item["start_sec"]
            record["top5_has_target"] = any(covers(h, item["start_sec"]) for h in hits[:TOP_K])
            covering = [h for h in hybrid["hits"] if covers(h, item["start_sec"])]
            record["target_hit_scores"] = [[h["match"], h["score"]] for h in covering]
        records.append(record)
    return records


def share(rows, key):
    return sum(1 for r in rows if key(r)) / len(rows) if rows else float("nan")


def summarize(records, split, baseline):
    rows = [r for r in records if r["split"] == split]
    nomatch = [r for r in rows if r["kind"] == "no_match"]
    positive = [r for r in rows if r["kind"] == "positive"]
    base_by_query = {(b["kind"], b["index"]): b for b in baseline} if baseline else {}
    dropped = 0
    for r in rows:
        base = base_by_query.get((r["kind"], r["index"]))
        if base is not None:
            dropped += len({tuple(x) for x in base["lexical_labelled"]} - {tuple(x) for x in r["lexical_labelled"]})
    changed = sum(1 for r in rows if not r["hit_keys_ok"] or not r["top_keys_ok"])
    out = {
        "climb_nomatch_semantic_share": share(nomatch, lambda r: bool(r["semantic_scores"])),
        "guard_positive_top5": share(positive, lambda r: r["top5_has_target"]),
        "guard_lexical_dropped": dropped,
        "guard_fields_changed": changed,
        "diag_nomatch_semantic_hits": (sum(len(r["semantic_scores"]) for r in nomatch) / len(nomatch)) if nomatch else float("nan"),
        "n_nomatch": len(nomatch),
        "n_positive": len(positive),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", help="records JSON from --out of the baseline run (for the lexical and top-5 guards)")
    parser.add_argument("--out", help="write the per-query records here (scratch folder, never committed)")
    parser.add_argument("--heldout", action="store_true", help="also print the test split (once per commit)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    records = measure(questions)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")

    splits = ["train"] + (["test"] if args.heldout else [])
    for split in splits:
        s = summarize(records, split, baseline)
        tag = "" if split == "train" else "heldout_"
        print(f"{tag}n_nomatch={s['n_nomatch']} n_positive={s['n_positive']} split={split}")
        print(f"METRIC {tag}climb_nomatch_semantic_share={s['climb_nomatch_semantic_share']:.4f}")
        print(f"METRIC {tag}guard_positive_top5={s['guard_positive_top5']:.4f}")
        print(f"METRIC {tag}guard_lexical_dropped={s['guard_lexical_dropped']}")
        print(f"METRIC {tag}guard_fields_changed={s['guard_fields_changed']}")
        print(f"METRIC {tag}diag_nomatch_semantic_hits={s['diag_nomatch_semantic_hits']:.3f}")


if __name__ == "__main__":
    main()
