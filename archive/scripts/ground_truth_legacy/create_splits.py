#!/usr/bin/env python3
"""
Create train/validation/test splits from master dataset

Splits the master dataset into:
- Train: 70% (~36,300 articles)
- Validation: 15% (~7,780 articles)
- Test: 15% (~7,780 articles)

With stratified sampling to ensure balanced distribution.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_articles(input_file: Path) -> List[Dict]:
    """Load all articles from JSONL file"""
    articles = []

    logger.info(f"Loading articles from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line)
                articles.append(article)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON error at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(articles):,} articles")
    return articles


def analyze_distribution(articles: List[Dict], field: str = 'source') -> Counter:
    """Analyze distribution of a field in articles"""
    distribution = Counter(article.get(field, 'unknown') for article in articles)

    logger.info(f"\nDistribution by {field}:")
    for value, count in distribution.most_common():
        percentage = (count / len(articles)) * 100
        logger.info(f"  {value}: {count:,} ({percentage:.1f}%)")

    return distribution


def create_splits(
    articles: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split articles into train/val/test sets

    Args:
        articles: List of article dictionaries
        train_ratio: Proportion for training set (default: 0.70)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        seed: Random seed for reproducibility
        stratify_by: Optional field to stratify by (e.g., 'source')

    Returns:
        Tuple of (train, val, test) article lists
    """
    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Set random seed
    random.seed(seed)

    # Shuffle articles
    shuffled = articles.copy()
    random.shuffle(shuffled)

    # Calculate split sizes
    n = len(shuffled)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    # test_size is the remainder to ensure all articles are used

    # Simple split (no stratification)
    if not stratify_by:
        train = shuffled[:train_size]
        val = shuffled[train_size:train_size + val_size]
        test = shuffled[train_size + val_size:]

        logger.info(f"\nCreated splits (random):")
        logger.info(f"  Train: {len(train):,} articles ({len(train)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val):,} articles ({len(val)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test):,} articles ({len(test)/n*100:.1f}%)")

        return train, val, test

    # Stratified split by field
    logger.info(f"\nCreating stratified splits by '{stratify_by}'")

    # Group articles by stratification field
    groups = {}
    for article in shuffled:
        value = article.get(stratify_by, 'unknown')
        if value not in groups:
            groups[value] = []
        groups[value].append(article)

    train, val, test = [], [], []

    for value, group_articles in groups.items():
        # Shuffle within group
        random.shuffle(group_articles)

        # Calculate group split sizes
        group_n = len(group_articles)
        group_train_size = int(group_n * train_ratio)
        group_val_size = int(group_n * val_ratio)

        # Split this group
        group_train = group_articles[:group_train_size]
        group_val = group_articles[group_train_size:group_train_size + group_val_size]
        group_test = group_articles[group_train_size + group_val_size:]

        # Add to overall splits
        train.extend(group_train)
        val.extend(group_val)
        test.extend(group_test)

        logger.info(f"  {value}: {len(group_train)} train, {len(group_val)} val, {len(group_test)} test")

    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    logger.info(f"\nTotal splits:")
    logger.info(f"  Train: {len(train):,} articles ({len(train)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val):,} articles ({len(val)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test):,} articles ({len(test)/n*100:.1f}%)")

    return train, val, test


def write_split(articles: List[Dict], output_file: Path):
    """Write articles to JSONL file"""
    logger.info(f"Writing {len(articles):,} articles to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from master dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split
  python create_splits.py \\
      --input datasets/raw/master_dataset.jsonl \\
      --output-dir datasets/splits

  # Custom ratios
  python create_splits.py \\
      --input datasets/raw/master_dataset.jsonl \\
      --output-dir datasets/splits \\
      --train-ratio 0.8 \\
      --val-ratio 0.1 \\
      --test-ratio 0.1

  # Stratified by source
  python create_splits.py \\
      --input datasets/raw/master_dataset.jsonl \\
      --output-dir datasets/splits \\
      --stratify-by source
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input master dataset (JSONL)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets/splits'),
        help='Output directory for splits (default: datasets/splits/)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio (default: 0.70)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--stratify-by',
        type=str,
        help='Field to stratify by (e.g., "source")'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load articles
    articles = load_articles(args.input)

    # Analyze distribution if stratifying
    if args.stratify_by:
        analyze_distribution(articles, args.stratify_by)

    # Create splits
    train, val, test = create_splits(
        articles,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by=args.stratify_by
    )

    # Write splits
    write_split(train, args.output_dir / 'train.jsonl')
    write_split(val, args.output_dir / 'val.jsonl')
    write_split(test, args.output_dir / 'test.jsonl')

    # Write metadata
    metadata = {
        "created": "2025-10-26",
        "input_file": str(args.input),
        "total_articles": len(articles),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "stratify_by": args.stratify_by
    }

    metadata_file = args.output_dir / 'splits_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nMetadata written to: {metadata_file}")
    logger.info(f"\n✅ Splits created successfully!")
    logger.info(f"  Train: {args.output_dir / 'train.jsonl'}")
    logger.info(f"  Val:   {args.output_dir / 'val.jsonl'}")
    logger.info(f"  Test:  {args.output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
