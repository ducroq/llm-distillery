#!/usr/bin/env python3
"""Consolidate all batch outputs, applying updated prefilter retroactively."""

import json
import sys
from pathlib import Path
from collections import Counter

# Import prefilter
sys.path.insert(0, str(Path('.') / 'filters' / 'uplifting' / 'v1'))
from prefilter import UpliftingPreFilterV1

def main():
    # All batch directories
    batch_dirs = [
        'datasets/ground_truth_1k_flash_validation',
        'datasets/ground_truth_batch2',
        'datasets/ground_truth_batch3',
        'datasets/ground_truth_batch4',
        'datasets/ground_truth_batch5',
        'datasets/ground_truth_batch6',
        'datasets/ground_truth_batch7',
        'datasets/ground_truth_batch8',
        'datasets/ground_truth_batch9',
        'datasets/ground_truth_batch10',
        'datasets/ground_truth_batch11',
        'datasets/ground_truth_batch12',
        'datasets/ground_truth_batch13',
        'datasets/ground_truth_batch14',
        'datasets/ground_truth_batch15',
        'datasets/ground_truth_batch16',
        'datasets/ground_truth_batch17',
        'datasets/ground_truth_batch18',
        'datasets/ground_truth_batch19',
        'datasets/ground_truth_batch20',
    ]

    # Also include the curated set from the 10k retroactive filtering
    curated_file = 'datasets/ground_truth_curated_prefiltered/labeled_articles.jsonl'

    output_dir = Path('datasets/ground_truth_combined_all')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'labeled_articles.jsonl'

    print("Consolidating all batch outputs with updated prefilter...")
    print("=" * 80)

    prefilter = UpliftingPreFilterV1()

    all_articles = []
    total_loaded = 0
    total_passed = 0
    total_blocked = 0
    block_reasons = Counter()
    batch_stats = []

    # Process each batch
    for batch_dir in batch_dirs:
        labeled_file = Path(batch_dir) / 'uplifting' / 'labeled_articles.jsonl'

        if not labeled_file.exists():
            print(f"[SKIP] {batch_dir} - file not found")
            continue

        # Load articles from this batch
        batch_articles = []
        with open(labeled_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch_articles.append(json.loads(line))

        # Apply prefilter to each article
        batch_passed = 0
        batch_blocked = 0
        batch_reasons = Counter()

        for article in batch_articles:
            should_label, reason = prefilter.should_label(article)

            if should_label:
                all_articles.append(article)
                batch_passed += 1
                total_passed += 1
            else:
                batch_blocked += 1
                total_blocked += 1
                block_reasons[reason] += 1
                batch_reasons[reason] += 1

        total_loaded += len(batch_articles)
        batch_stats.append((batch_dir, len(batch_articles), batch_passed, batch_blocked, batch_reasons))

        print(f"[OK] {batch_dir}: {batch_passed}/{len(batch_articles)} passed ({batch_passed/len(batch_articles)*100:.1f}%)")

    # Load curated articles (already prefiltered, but let's be safe)
    if Path(curated_file).exists():
        print(f"\nLoading curated articles from retroactive 10k filtering...")
        curated_count = 0
        with open(curated_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    # These are already filtered, but double-check
                    should_label, reason = prefilter.should_label(article)
                    if should_label:
                        all_articles.append(article)
                        curated_count += 1
                        total_passed += 1
        print(f"[OK] Added {curated_count} curated articles")

    # Remove duplicates (by URL)
    print(f"\nRemoving duplicates...")
    seen_urls = set()
    unique_articles = []
    duplicates = 0

    for article in all_articles:
        url = article.get('url', '')
        if url and url in seen_urls:
            duplicates += 1
            continue
        if url:
            seen_urls.add(url)
        unique_articles.append(article)

    print(f"Removed {duplicates} duplicates")

    # Write consolidated dataset
    print(f"\nWriting {len(unique_articles):,} articles to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in unique_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    # Show final statistics
    print("\n" + "=" * 80)
    print("CONSOLIDATION RESULTS")
    print("=" * 80)
    print(f"Total articles loaded:    {total_loaded:,}")
    print(f"Passed updated prefilter: {len(unique_articles):,} ({len(unique_articles)/total_loaded*100:.1f}%)")
    print(f"Blocked:                  {total_blocked:,} ({total_blocked/total_loaded*100:.1f}%)")
    print(f"Duplicates removed:       {duplicates:,}")

    print("\nBlock reasons:")
    for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason:<35} {count:>6,} ({count/total_blocked*100:.1f}%)")

    print(f"\n[DONE] Final dataset: {len(unique_articles):,} articles")
    print(f"  Location: {output_file}")

if __name__ == '__main__':
    main()
