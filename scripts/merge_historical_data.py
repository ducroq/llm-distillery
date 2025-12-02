"""
Merge historical database collections into a master dataset.

This script reads all collection files from the historical-database/current folder,
deduplicates articles, and creates a new master dataset.

Usage:
    python scripts/merge_historical_data.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Set
from collections import defaultdict


def find_collection_files(base_path: Path) -> list[Path]:
    """Find all content_items_*.jsonl files in the current collections folder."""
    print(f"Scanning {base_path}...")

    collection_files = []
    for collection_dir in sorted(base_path.iterdir()):
        if not collection_dir.is_dir():
            continue

        # Look for content_items_*.jsonl file
        for file in collection_dir.glob("content_items_*.jsonl"):
            collection_files.append(file)

    print(f"Found {len(collection_files)} collection files")
    return collection_files


def load_and_deduplicate(collection_files: list[Path]) -> tuple[list[Dict], Dict]:
    """
    Load all articles and deduplicate by URL.

    Returns:
        - List of unique articles
        - Statistics about the merge process
    """
    articles_by_url = {}  # Use URL as unique key
    stats = defaultdict(int)

    print("\nProcessing collections...")
    for i, file_path in enumerate(collection_files, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(collection_files)} files, {len(articles_by_url)} unique articles so far...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        article = json.loads(line)
                        stats['total_articles'] += 1

                        # Use URL as unique identifier
                        url = article.get('url') or article.get('link')
                        if not url:
                            stats['no_url'] += 1
                            continue

                        # Deduplicate - keep first occurrence
                        if url not in articles_by_url:
                            articles_by_url[url] = article
                            stats['unique_articles'] += 1
                        else:
                            stats['duplicates'] += 1

                    except json.JSONDecodeError as e:
                        stats['json_errors'] += 1
                        print(f"    JSON error in {file_path.name} line {line_num}: {e}")

        except Exception as e:
            stats['file_errors'] += 1
            print(f"    Error reading {file_path.name}: {e}")

    print(f"\nLoaded {len(articles_by_url)} unique articles")

    return list(articles_by_url.values()), dict(stats)


def sort_articles_by_date(articles: list[Dict]) -> list[Dict]:
    """Sort articles by publication date (newest first)."""
    print("\nSorting articles by date...")

    def get_date(article):
        """Extract publication date from article."""
        # Try multiple date fields
        for field in ['published', 'pubDate', 'date', 'published_date']:
            if field in article and article[field]:
                try:
                    # Parse ISO format or similar
                    date_str = article[field]
                    if isinstance(date_str, str):
                        # Handle ISO format with timezone
                        if 'T' in date_str:
                            date_str = date_str.split('T')[0]
                        return date_str
                except (ValueError, TypeError, AttributeError):
                    # Date parsing failed, try next field
                    pass

        # Default to far future if no date (will be at the end)
        return '9999-12-31'

    sorted_articles = sorted(articles, key=get_date, reverse=True)
    print(f"Sorted {len(sorted_articles)} articles")

    return sorted_articles


def write_master_dataset(articles: list[Dict], output_path: Path, stats: Dict):
    """Write articles to master dataset file and create metadata."""
    print(f"\nWriting master dataset to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_articles': len(articles),
        'file_size_mb': round(file_size_mb, 2),
        'date_range': {
            'start': '2025-10-09',
            'end': '2025-11-24'
        },
        'merge_stats': stats,
        'source': 'I:/Mijn Drive/NexusMind/historical-database/current/'
    }

    # Write metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMaster dataset created:")
    print(f"  - File: {output_path.name}")
    print(f"  - Size: {file_size_mb:.2f} MB")
    print(f"  - Articles: {len(articles):,}")
    print(f"  - Metadata: {metadata_path.name}")

    return metadata


def print_summary(stats: Dict, metadata: Dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"Total articles processed:  {stats['total_articles']:,}")
    print(f"Unique articles:           {stats['unique_articles']:,}")
    print(f"Duplicates removed:        {stats['duplicates']:,}")
    print(f"Articles without URL:      {stats.get('no_url', 0):,}")
    print(f"JSON errors:               {stats.get('json_errors', 0):,}")
    print(f"File errors:               {stats.get('file_errors', 0):,}")
    print(f"\nDeduplication rate: {stats['duplicates'] / stats['total_articles'] * 100:.1f}%")
    print(f"Final dataset size: {metadata['file_size_mb']:.2f} MB")
    print("="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("Historical Database Merge Script")
    print("="*60)

    # Paths
    historical_db_path = Path("I:/Mijn Drive/NexusMind/historical-database/current")
    output_dir = Path("datasets/raw")
    output_file = output_dir / "master_dataset_20251009_20251124.jsonl"

    # Check if source exists
    if not historical_db_path.exists():
        print(f"Error: Source path not found: {historical_db_path}")
        return

    # Check if output already exists
    if output_file.exists():
        response = input(f"\nWARNING: {output_file.name} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Step 1: Find all collection files
    collection_files = find_collection_files(historical_db_path)

    if not collection_files:
        print("Error: No collection files found!")
        return

    # Step 2: Load and deduplicate
    articles, stats = load_and_deduplicate(collection_files)

    if not articles:
        print("Error: No articles loaded!")
        return

    # Step 3: Sort by date
    sorted_articles = sort_articles_by_date(articles)

    # Step 4: Write master dataset
    metadata = write_master_dataset(sorted_articles, output_file, stats)

    # Step 5: Print summary
    print_summary(stats, metadata)

    print(f"\nDone! Master dataset ready at: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the dataset: head datasets/raw/{output_file.name}")
    print(f"  2. Run calibration on this dataset")
    print(f"  3. Generate 10K training dataset")


if __name__ == "__main__":
    main()
