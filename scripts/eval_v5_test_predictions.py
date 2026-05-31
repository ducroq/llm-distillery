"""Score the cd v5 test set (and any additional articles) through the v5 inference
pipeline and persist per-article predictions for downstream agent spot-check.

Outputs a JSONL where each record has:
  - id, title, content (truncated)
  - oracle_dims  (DeepSeek-v5-prompt scores from training labels)
  - student_dims (v5 model + calibration predictions)
  - oracle_wavg, student_wavg
  - diff (student - oracle, per dim + overall)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from filters.cultural_discovery.v5.inference import CulturalDiscoveryScorer

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]


def wavg(dims):
    return sum(dims[d] * w for d, w in zip(DIMS, WEIGHTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="datasets/training/cultural-discovery_v5/test.jsonl")
    parser.add_argument("--source",
                        default="datasets/scored/cd_v5_deepseek_merged_for_training.jsonl",
                        help="Source file with oracle scores + content (training format strips these)")
    parser.add_argument("--output", default="datasets/scored/cd_v5_test_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load test article IDs
    test_ids = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            test_ids.append(r["id"])
    print(f"Test IDs: {len(test_ids)}")

    # Load source (has oracle + content)
    source_lookup = {}
    with open(args.source, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            source_lookup[r["id"]] = r
    print(f"Source records: {len(source_lookup)}")

    test_articles = [source_lookup[tid] for tid in test_ids if tid in source_lookup]
    if args.limit:
        test_articles = test_articles[:args.limit]
    print(f"Test articles to score: {len(test_articles)}")

    print("Loading v5 scorer...")
    scorer = CulturalDiscoveryScorer(use_prefilter=False)  # Skip prefilter for already-passed articles
    print("Scorer loaded. Starting scoring...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    errors = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, art in enumerate(test_articles, 1):
            try:
                af = art["cultural_discovery_analysis"]
                oracle_dims = {d: af[d]["score"] for d in DIMS}
                oracle_ct = af.get("content_type", "unknown")

                result = scorer.score_article(art)
                student_dims = {d: result["scores"][d] for d in DIMS}
                student_wavg_val = result.get("weighted_average", wavg(student_dims))

                record = {
                    "id": art["id"],
                    "title": art.get("title", "")[:200],
                    "url": art.get("url", ""),
                    "source": art.get("source", ""),
                    "oracle_dims": oracle_dims,
                    "oracle_ct": oracle_ct,
                    "oracle_wavg": round(wavg(oracle_dims), 2),
                    "student_dims": {d: round(float(student_dims[d]), 2) for d in DIMS},
                    "student_wavg": round(float(student_wavg_val), 2),
                    "diff": {d: round(float(student_dims[d]) - oracle_dims[d], 2) for d in DIMS},
                    "diff_wavg": round(float(student_wavg_val) - wavg(oracle_dims), 2),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                successes += 1
                if i % 50 == 0:
                    print(f"  {i}/{len(test_articles)} done")
            except Exception as e:
                errors += 1
                print(f"  ERROR on {art.get('id','?')}: {e}")

    print(f"\nComplete: {successes} successful, {errors} errors")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
