"""
Remove duplicate IDs from training data splits.
Keeps duplicates in train, removes from val/test.
"""
import json
import sys
from pathlib import Path

def deduplicate_training_data(data_dir: Path):
    """Remove cross-split duplicates from training data."""

    # Load all splits
    train_data = []
    val_data = []
    test_data = []

    train_file = data_dir / 'train.jsonl'
    val_file = data_dir / 'val.jsonl'
    test_file = data_dir / 'test.jsonl'

    print(f"Loading data from {data_dir}...")

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))

    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Step 1: Remove within-split duplicates
    train_ids = {}
    for ex in train_data:
        train_ids[ex['id']] = ex
    train_data = list(train_ids.values())

    val_ids = {}
    for ex in val_data:
        val_ids[ex['id']] = ex
    val_data = list(val_ids.values())

    test_ids = {}
    for ex in test_data:
        test_ids[ex['id']] = ex
    test_data = list(test_ids.values())

    print(f"After within-split dedup: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Step 2: Remove cross-split duplicates (keep in train)
    train_id_set = set(ex['id'] for ex in train_data)
    val_id_set = set(ex['id'] for ex in val_data)
    test_id_set = set(ex['id'] for ex in test_data)

    train_val_overlap = train_id_set & val_id_set
    train_test_overlap = train_id_set & test_id_set
    val_test_overlap = val_id_set & test_id_set

    if train_val_overlap:
        print(f"  Train-Val overlap: {len(train_val_overlap)} IDs")
    if train_test_overlap:
        print(f"  Train-Test overlap: {len(train_test_overlap)} IDs")
    if val_test_overlap:
        print(f"  Val-Test overlap: {len(val_test_overlap)} IDs")

    # Remove overlaps
    val_data_clean = [ex for ex in val_data if ex['id'] not in train_id_set]
    test_data_clean = [ex for ex in test_data if ex['id'] not in train_id_set and ex['id'] not in val_id_set]

    total_removed = len(val_data) - len(val_data_clean) + len(test_data) - len(test_data_clean)

    print(f"\nAfter cross-split dedup:")
    print(f"  Train: {len(train_data)} (unchanged)")
    print(f"  Val: {len(val_data_clean)} (removed {len(val_data) - len(val_data_clean)})")
    print(f"  Test: {len(test_data_clean)} (removed {len(test_data) - len(test_data_clean)})")
    print(f"  Total removed: {total_removed}")

    # Write back
    print(f"\nWriting deduplicated files...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val_data_clean:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for ex in test_data_clean:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"✅ Done! New totals: {len(train_data) + len(val_data_clean) + len(test_data_clean)} examples")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python deduplicate_training_data.py <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    deduplicate_training_data(data_dir)
