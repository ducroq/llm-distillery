"""Final merge of v5 training data for DS-based retrain.

Combines:
  A) `datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl` (522 calibration records)
     — needs format conversion from validate_deepseek_oracle.py shape to
       prepare_data.py shape (nested `cultural_discovery_analysis` field).
  B) `datasets/scored/cd_v5_8k_deepseek_v5_prompt.jsonl` (8029 production records)
     — already in `cultural_discovery_analysis` shape.

Dedup: prefer the 8K production record when IDs overlap (later scoring under
current prompt version). The 49 hard-negatives in (A) are NOT expected to
overlap with (B) since they were collected separately.

Output: `datasets/scored/cd_v5_deepseek_merged_for_training.jsonl`
Format: matches what `prepare_data.py` expects (cultural_discovery_analysis with
nested dim scores).
"""

import json
from pathlib import Path

DIMENSIONS = [
    "discovery_novelty",
    "heritage_significance",
    "cross_cultural_connection",
    "human_resonance",
    "evidence_quality",
]


def main():
    cal_path = Path("datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl")
    prod_path = Path("datasets/scored/cd_v5_8k_deepseek_v5_prompt.jsonl")
    out_path = Path("datasets/scored/cd_v5_deepseek_merged_for_training.jsonl")

    # We need the original article content (title/content/url/source) joined onto the calibration scores.
    # validate_deepseek_oracle.py only stored {id, title, deepseek, deepseek_content_type} — not full content.
    # Source the full content from the canonical staging file used for calibration:
    canonical_articles_path = Path("datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl")
    article_lookup = {}
    with open(canonical_articles_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            article_lookup[r["id"]] = r

    # Load calibration records (522) and convert to cultural_discovery_analysis shape
    cal_records = []
    cal_errors = 0
    with open(cal_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                cal_errors += 1
                continue
            article = article_lookup.get(r["id"])
            if not article:
                cal_errors += 1
                continue
            # Convert flat deepseek dims to nested cultural_discovery_analysis
            analysis = {
                d: {"score": r["deepseek"].get(d), "evidence": ""} for d in DIMENSIONS
            }
            analysis["content_type"] = r.get("deepseek_content_type", "unknown")
            analysis["filter_version"] = "5.0-deepseek-calibration"
            analysis["analyzed_by"] = "deepseek-chat"
            merged = {**article, "cultural_discovery_analysis": analysis}
            cal_records.append(merged)
    print(f"Calibration (522): {len(cal_records)} records ({cal_errors} dropped)")

    # Load production records (8K)
    prod_records = []
    prod_errors = 0
    with open(prod_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                prod_errors += 1
                continue
            if "cultural_discovery_analysis" not in r:
                prod_errors += 1
                continue
            prod_records.append(r)
    print(f"Production (8K): {len(prod_records)} records ({prod_errors} dropped)")

    # Merge with dedup; prefer production records on collision
    seen = set()
    merged = []
    for r in prod_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)
    cal_added = 0
    cal_overlap = 0
    for r in cal_records:
        if r["id"] in seen:
            cal_overlap += 1
            continue
        seen.add(r["id"])
        merged.append(r)
        cal_added += 1
    print(f"Calibration overlap with production: {cal_overlap}")
    print(f"Calibration added (unique): {cal_added}")
    print(f"Total merged: {len(merged)}")

    # Sanity check: all records have the required field + all 5 dims
    valid = 0
    invalid = []
    for r in merged:
        af = r.get("cultural_discovery_analysis", {})
        ok = af and all(
            isinstance(af.get(d), dict) and af.get(d).get("score") is not None
            for d in DIMENSIONS
        )
        if ok:
            valid += 1
        else:
            invalid.append(r["id"])
    print(f"Schema validation: {valid}/{len(merged)} valid")
    if invalid:
        print(f"  Invalid sample ids (first 5): {invalid[:5]}")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path} ({len(merged)} records)")


if __name__ == "__main__":
    main()
