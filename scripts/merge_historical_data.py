#!/usr/bin/env python3
"""Merge historical database from Google Drive into llm-distillery datasets."""

import json
from pathlib import Path
from typing import Dict, Set
from datetime import datetime
import sys

def load_existing_ids(datasets_dir: Path) -> Set[str]:
    """Load all existing article IDs from current datasets."""
    existing_ids = set()

    for jsonl_file in datasets_dir.glob("master_dataset_*.jsonl"):
        print(f"  Reading existing: {jsonl_file.name}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_ids.add(item['id'])

    print(f"  Found {len(existing_ids):,} existing article IDs")
    return existing_ids

def merge_historical_database(
    historical_db_path: Path,
    output_dir: Path,
    existing_ids: Set[str]
) -> Dict:
    """Merge all historical database JSONL files, deduplicating by ID."""

    # Find all JSONL files in historical database
    jsonl_files = sorted(historical_db_path.glob("*/content_items_*.jsonl"))

    print(f"\nFound {len(jsonl_files)} historical collection files")

    merged_items = {}
    stats = {
        'total_read': 0,
        'duplicates_within_historical': 0,
        'duplicates_with_existing': 0,
        'new_items': 0,
        'earliest_date': None,
        'latest_date': None
    }

    for jsonl_file in jsonl_files:
        print(f"  Processing: {jsonl_file.parent.name}")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                stats['total_read'] += 1
                item = json.loads(line)
                item_id = item['id']

                # Check if already in existing datasets
                if item_id in existing_ids:
                    stats['duplicates_with_existing'] += 1
                    continue

                # Check if already seen in historical DB
                if item_id in merged_items:
                    stats['duplicates_within_historical'] += 1
                    # Keep the newer one (by collected_date)
                    existing_collected = merged_items[item_id].get('collected_date', '')
                    new_collected = item.get('collected_date', '')
                    if new_collected > existing_collected:
                        merged_items[item_id] = item
                    continue

                # New item
                merged_items[item_id] = item
                stats['new_items'] += 1

                # Track date range
                pub_date = item.get('published_date', '')
                if pub_date:
                    if stats['earliest_date'] is None or pub_date < stats['earliest_date']:
                        stats['earliest_date'] = pub_date
                    if stats['latest_date'] is None or pub_date > stats['latest_date']:
                        stats['latest_date'] = pub_date

    return merged_items, stats

def write_merged_dataset(merged_items: Dict, output_file: Path):
    """Write merged items to JSONL file."""

    # Sort by published_date for consistency
    sorted_items = sorted(
        merged_items.values(),
        key=lambda x: x.get('published_date', '')
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sorted_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(sorted_items):,} items to: {output_file}")

def main():
    # Paths
    historical_db_path = Path("I:/Mijn Drive/NexusMind/historical-database/current")
    output_dir = Path("datasets/raw")

    if not historical_db_path.exists():
        print(f"ERROR: Historical database not found at: {historical_db_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MERGING HISTORICAL DATABASE INTO LLM-DISTILLERY")
    print("="*70)

    # Load existing IDs for deduplication
    print("\n1. Loading existing article IDs from current datasets...")
    existing_ids = load_existing_ids(output_dir)

    # Merge historical database
    print("\n2. Merging historical database (deduplicating)...")
    merged_items, stats = merge_historical_database(
        historical_db_path,
        output_dir,
        existing_ids
    )

    # Generate output filename with date range
    earliest = stats['earliest_date'][:10] if stats['earliest_date'] else 'unknown'
    latest = stats['latest_date'][:10] if stats['latest_date'] else 'unknown'

    # Convert dates to compact format
    earliest_compact = earliest.replace('-', '')
    latest_compact = latest.replace('-', '')

    output_file = output_dir / f"historical_dataset_{earliest_compact}_{latest_compact}.jsonl"

    # Write merged dataset
    print("\n3. Writing merged dataset...")
    write_merged_dataset(merged_items, output_file)

    # Print summary
    print("\n" + "="*70)
    print("MERGE SUMMARY")
    print("="*70)
    print(f"Total items read from historical DB:  {stats['total_read']:,}")
    print(f"Duplicates with existing datasets:    {stats['duplicates_with_existing']:,}")
    print(f"Duplicates within historical DB:      {stats['duplicates_within_historical']:,}")
    print(f"New unique items added:               {stats['new_items']:,}")
    print(f"\nDate range: {earliest} to {latest}")
    print(f"\nOutput file: {output_file}")
    print("="*70)

    # Show total dataset size
    total_items = len(existing_ids) + stats['new_items']
    print(f"\nTotal dataset size: {total_items:,} articles")
    print(f"  - Existing: {len(existing_ids):,}")
    print(f"  - New from historical DB: {stats['new_items']:,}")

if __name__ == "__main__":
    main()
