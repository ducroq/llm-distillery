"""
Apply score caps to crime articles marked for capping in crime_review.tsv.

Reads the reviewed TSV, finds articles marked with Y in the 'cap' column,
and sets all their dimension scores to values that produce weighted_avg = 2.0
(well into LOW tier, matching the content_type_cap of 3.0 with margin).

The approach: scale all dimensions proportionally so weighted_avg = 2.0.
This preserves relative dimension relationships while ensuring the article
is correctly classified as LOW.

Usage:
    python scripts/apply_crime_caps.py [--dry-run]

Input:  datasets/training/uplifting_v6/crime_review.tsv (with 'cap' column filled)
Output: datasets/training/uplifting_v6/{train,val,test}.jsonl (overwritten in-place)
        datasets/training/uplifting_v6/crime_caps_applied.json (log of changes)
"""

import argparse
import json
from pathlib import Path

WEIGHTS = [0.25, 0.15, 0.10, 0.20, 0.20, 0.10]
TARGET_WEIGHTED_AVG = 2.0  # Cap to solidly LOW tier


def weighted_avg(labels):
    return sum(s * w for s, w in zip(labels, WEIGHTS))


def cap_scores(labels, target=TARGET_WEIGHTED_AVG):
    """Scale all dimension scores proportionally to hit target weighted average."""
    current = weighted_avg(labels)
    if current <= target:
        return labels  # Already below cap

    scale = target / current
    return [round(s * scale, 2) for s in labels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    data_dir = Path("datasets/training/uplifting_v6")
    review_path = data_dir / "crime_review.tsv"

    # Read reviewed TSV
    cap_ids = set()
    with open(review_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        cap_idx = header.index("cap")
        id_idx = header.index("id")

        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > max(cap_idx, id_idx):
                cap_value = fields[cap_idx].strip().upper()
                if cap_value == "Y":
                    cap_ids.add(fields[id_idx].strip())

    print(f"Articles marked for capping: {len(cap_ids)}")

    if not cap_ids:
        print("No articles marked with Y. Nothing to do.")
        return

    # Apply caps to each split
    changes_log = []
    total_modified = 0

    for split in ["train", "val", "test"]:
        filepath = data_dir / f"{split}.jsonl"
        examples = []
        split_modified = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("id", "") in cap_ids:
                    old_labels = ex["labels"][:]
                    old_avg = weighted_avg(old_labels)
                    new_labels = cap_scores(old_labels)
                    new_avg = weighted_avg(new_labels)

                    changes_log.append({
                        "id": ex["id"],
                        "split": split,
                        "title": ex.get("title", "")[:100],
                        "old_labels": [round(x, 2) for x in old_labels],
                        "new_labels": new_labels,
                        "old_weighted_avg": round(old_avg, 2),
                        "new_weighted_avg": round(new_avg, 2),
                    })

                    ex["labels"] = new_labels
                    split_modified += 1

                examples.append(ex)

        if not args.dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        total_modified += split_modified
        print(f"  {split}: {split_modified} articles capped")

    # Show changes
    print(f"\nTotal modified: {total_modified}")
    print(f"\nChanges:")
    for c in changes_log:
        print(f"  [{c['split']}] {c['old_weighted_avg']} -> {c['new_weighted_avg']} | {c['title']}")

    # Save log
    if not args.dry_run:
        log_path = data_dir / "crime_caps_applied.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(changes_log, f, indent=2, ensure_ascii=False)
        print(f"\nChanges log saved to: {log_path}")
    else:
        print(f"\n[DRY RUN] No files modified.")


if __name__ == "__main__":
    main()
