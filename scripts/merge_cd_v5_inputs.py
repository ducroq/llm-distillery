"""
Merge raw oracle-scored sources for cultural_discovery v5 training.

Combines five scored cohorts, normalizes the analysis field name to
`cultural_discovery_analysis` (matches v5 config filter.name), and dedups
by article id with newer/higher-quality scoring winning.

Priority (highest first — wins ties):
  1. cd_v5_hard_negatives        (49, v5.0-draft prompt, 2026-05-29)
  2. active_learning_cd_v5_rescored (473, v5.0-draft prompt, today)
  3. active_learning_cd_v4       (202, pre-v3 prompt, 2026-02-20)
  4. cultural-discovery-v2       (2,919, pre-v3 prompt, 2026-01-29)
  5. cultural-discovery-v1       (9,992, pre-v3 prompt, 2026-01-27)

Output: datasets/scored/cultural_discovery_v5_merged.jsonl

Usage:
    PYTHONPATH=. python scripts/merge_cd_v5_inputs.py

After re-score of AL v5 completes, re-run; this script is idempotent.
"""

import glob
import json
from collections import Counter
from pathlib import Path

TARGET_FIELD = "cultural_discovery_analysis"
LEGACY_FIELD = "cultural-discovery_analysis"

SOURCES = [
    ("cd_v5_hard_negatives",
     "datasets/scored/cd_v5_hard_negatives/cultural_discovery/scored_batch_*.jsonl"),
    ("al_v5_rescored",
     "datasets/scored/active_learning_cd_v5_rescored/cultural_discovery/scored_batch_*.jsonl"),
    ("al_v4",
     "datasets/scored/active_learning_cd_v4/cultural-discovery/scored_batch_*.jsonl"),
    ("cd_v2",
     "datasets/scored/cultural-discovery-v2/cultural-discovery/scored_batch_*.jsonl"),
    ("cd_v1",
     "datasets/scored/cultural-discovery-v1/all_scored.jsonl"),
]

OUTPUT = Path("datasets/scored/cultural_discovery_v5_merged.jsonl")


def normalize_field(record: dict) -> dict | None:
    """Rename legacy hyphen field to underscore. Drop records with no analysis."""
    if TARGET_FIELD in record:
        return record
    if LEGACY_FIELD in record:
        record[TARGET_FIELD] = record.pop(LEGACY_FIELD)
        return record
    return None


def main():
    seen_ids: dict[str, str] = {}
    kept: Counter = Counter()
    skipped_dup: Counter = Counter()
    skipped_no_analysis: Counter = Counter()
    missing_source: list[str] = []

    merged: list[dict] = []

    for source_name, pattern in SOURCES:
        files = sorted(glob.glob(pattern))
        if not files:
            missing_source.append(f"{source_name} ({pattern})")
            continue

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    record = normalize_field(record)
                    if record is None:
                        skipped_no_analysis[source_name] += 1
                        continue

                    article_id = record.get("id")
                    if not article_id:
                        skipped_no_analysis[source_name] += 1
                        continue

                    if article_id in seen_ids:
                        skipped_dup[source_name] += 1
                        continue

                    seen_ids[article_id] = source_name
                    kept[source_name] += 1
                    merged.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    print(f"{'Source':<28} {'Kept':>8} {'Dup':>8} {'NoAnalysis':>12}")
    print("-" * 70)
    for source_name, _ in SOURCES:
        print(f"{source_name:<28} {kept[source_name]:>8} "
              f"{skipped_dup[source_name]:>8} {skipped_no_analysis[source_name]:>12}")
    print("-" * 70)
    print(f"{'TOTAL':<28} {sum(kept.values()):>8} "
          f"{sum(skipped_dup.values()):>8} {sum(skipped_no_analysis.values()):>12}")

    if missing_source:
        print("\nMISSING SOURCES (glob matched no files):")
        for ms in missing_source:
            print(f"  - {ms}")

    print(f"\nWrote {len(merged)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
