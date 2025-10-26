#!/usr/bin/env python3
"""
Dataset Preparation for Ground Truth Generation

Merges historical database into a master dataset with:
- Deduplication by article ID
- Quality filtering
- Metadata preservation
- Incremental updates support
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List
from datetime import datetime
from collections import Counter
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare and merge datasets for ground truth generation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track seen IDs for deduplication
        self.seen_ids: Set[str] = set()
        self.stats = Counter()

    def load_existing_ids(self, master_file: Path) -> Set[str]:
        """Load existing article IDs from master dataset for incremental updates"""
        existing_ids = set()

        if master_file.exists():
            logger.info(f"Loading existing IDs from {master_file}")
            with open(master_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        if 'id' in article:
                            existing_ids.add(article['id'])
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Loaded {len(existing_ids):,} existing article IDs")

        return existing_ids

    def is_valid_article(self, article: Dict) -> bool:
        """Validate article has required fields and content"""
        # Required fields
        required = ['id', 'title', 'content']
        if not all(field in article for field in required):
            self.stats['invalid_missing_fields'] += 1
            return False

        # Must have non-empty content
        if not article.get('content', '').strip():
            self.stats['invalid_empty_content'] += 1
            return False

        # Must have non-empty title
        if not article.get('title', '').strip():
            self.stats['invalid_empty_title'] += 1
            return False

        return True

    def merge_datasets(
        self,
        source_pattern: str,
        output_file: Path,
        incremental: bool = False,
        max_articles: Optional[int] = None
    ) -> int:
        """
        Merge multiple JSONL files into a single master dataset

        Args:
            source_pattern: Glob pattern for source files (e.g., "path/to/*.jsonl")
            output_file: Path to master output file
            incremental: If True, append only new articles to existing master
            max_articles: Maximum number of articles to include (None = unlimited)

        Returns:
            Number of articles written
        """
        # Load existing IDs if incremental
        if incremental:
            self.seen_ids = self.load_existing_ids(output_file)
            mode = 'a'  # Append mode
        else:
            self.seen_ids = set()
            mode = 'w'  # Write mode

        # Find all source files
        # Handle absolute paths with drive letters (Windows) and relative paths
        import glob as glob_module
        source_files = sorted([Path(p) for p in glob_module.glob(source_pattern, recursive=True)])
        if not source_files:
            logger.error(f"No files found matching pattern: {source_pattern}")
            return 0

        logger.info(f"Found {len(source_files)} source files")
        logger.info(f"Output: {output_file}")
        logger.info(f"Mode: {'Incremental (append)' if incremental else 'Full (overwrite)'}")

        articles_written = 0

        with open(output_file, mode, encoding='utf-8') as out_f:
            for source_file in source_files:
                logger.info(f"Processing: {source_file.name}")

                try:
                    with open(source_file, 'r', encoding='utf-8') as in_f:
                        for line_num, line in enumerate(in_f, 1):
                            self.stats['total_read'] += 1

                            try:
                                article = json.loads(line)

                                # Validate article
                                if not self.is_valid_article(article):
                                    continue

                                article_id = article['id']

                                # Check for duplicates
                                if article_id in self.seen_ids:
                                    self.stats['duplicates'] += 1
                                    continue

                                # Add to master dataset
                                self.seen_ids.add(article_id)
                                out_f.write(json.dumps(article, ensure_ascii=False) + '\n')
                                articles_written += 1
                                self.stats['written'] += 1

                                # Check max articles limit
                                if max_articles and articles_written >= max_articles:
                                    logger.info(f"Reached max articles limit: {max_articles:,}")
                                    return articles_written

                                # Progress logging
                                if articles_written % 10000 == 0:
                                    logger.info(f"  Written: {articles_written:,} articles")

                            except json.JSONDecodeError as e:
                                self.stats['json_errors'] += 1
                                logger.warning(f"  JSON error in {source_file.name} line {line_num}: {e}")
                                continue

                except Exception as e:
                    logger.error(f"Error processing {source_file}: {e}")
                    continue

        return articles_written

    def print_stats(self):
        """Print processing statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("Dataset Preparation Statistics")
        logger.info("=" * 60)
        logger.info(f"Total articles read:     {self.stats['total_read']:,}")
        logger.info(f"Valid articles written:  {self.stats['written']:,}")
        logger.info(f"Duplicates skipped:      {self.stats['duplicates']:,}")
        logger.info(f"Invalid (missing fields):{self.stats['invalid_missing_fields']:,}")
        logger.info(f"Invalid (empty content): {self.stats['invalid_empty_content']:,}")
        logger.info(f"Invalid (empty title):   {self.stats['invalid_empty_title']:,}")
        logger.info(f"JSON decode errors:      {self.stats['json_errors']:,}")
        logger.info("=" * 60)

    def create_metadata_file(self, output_file: Path, source_pattern: str):
        """Create metadata file documenting the dataset"""
        metadata = {
            "created": datetime.now().isoformat(),
            "source_pattern": source_pattern,
            "total_articles": self.stats['written'],
            "unique_articles": len(self.seen_ids),
            "duplicates_removed": self.stats['duplicates'],
            "invalid_articles": sum([
                self.stats['invalid_missing_fields'],
                self.stats['invalid_empty_content'],
                self.stats['invalid_empty_title']
            ]),
            "output_file": str(output_file),
            "statistics": dict(self.stats)
        }

        metadata_file = output_file.parent / f"{output_file.stem}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata written to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ground truth generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all historical data into master dataset
  python prepare_dataset.py \\
      --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \\
      --output datasets/master_dataset.jsonl

  # Incremental update (append only new articles)
  python prepare_dataset.py \\
      --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \\
      --output datasets/master_dataset.jsonl \\
      --incremental

  # Create smaller dataset for testing (1000 articles)
  python prepare_dataset.py \\
      --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \\
      --output datasets/test_1k.jsonl \\
      --max-articles 1000
        """
    )

    parser.add_argument(
        '--source',
        required=True,
        help='Glob pattern for source JSONL files (e.g., "path/*/*.jsonl")'
    )

    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Path to output master dataset file'
    )

    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Append only new articles to existing master dataset'
    )

    parser.add_argument(
        '--max-articles',
        type=int,
        help='Maximum number of articles to include'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets'),
        help='Base output directory (default: datasets/)'
    )

    args = parser.parse_args()

    # Initialize preparer
    preparer = DatasetPreparer(args.output_dir)

    # Merge datasets
    logger.info("\n" + "=" * 60)
    logger.info("Starting Dataset Preparation")
    logger.info("=" * 60)

    articles_written = preparer.merge_datasets(
        source_pattern=args.source,
        output_file=args.output,
        incremental=args.incremental,
        max_articles=args.max_articles
    )

    # Print statistics
    preparer.print_stats()

    # Create metadata
    preparer.create_metadata_file(args.output, args.source)

    logger.info(f"\n✅ Dataset preparation complete!")
    logger.info(f"Master dataset: {args.output}")
    logger.info(f"Total articles: {articles_written:,}")


if __name__ == "__main__":
    main()
