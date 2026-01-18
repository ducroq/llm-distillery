"""
Commerce Prefilter - Dataset Preparation

Prepares train/val/test splits from collected examples.

Usage:
    python -m filters.common.commerce_prefilter.training.prepare_dataset \
        --input datasets/commerce_prefilter/raw_examples.jsonl \
        --output datasets/commerce_prefilter/splits/
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_examples(input_path: Path) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def stratified_split(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split examples into train/val/test with stratification by label.

    Args:
        examples: List of examples with 'label' field
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        (train, val, test) lists
    """
    random.seed(seed)

    # Separate by label
    positive = [ex for ex in examples if ex.get('label') == 1]
    negative = [ex for ex in examples if ex.get('label') == 0]

    # Shuffle
    random.shuffle(positive)
    random.shuffle(negative)

    def split_list(lst: List, train_r: float, val_r: float) -> Tuple[List, List, List]:
        n = len(lst)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    # Split each class
    pos_train, pos_val, pos_test = split_list(positive, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(negative, train_ratio, val_ratio)

    # Combine and shuffle
    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_split(examples: List[Dict], output_path: Path):
    """Save examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def print_split_stats(name: str, examples: List[Dict]):
    """Print statistics for a split."""
    labels = Counter(ex.get('label') for ex in examples)
    sources = Counter(ex.get('source') for ex in examples)

    print(f"\n{name}: {len(examples)} examples")
    print(f"  Labels: {dict(labels)}")
    print(f"  Top sources: {dict(sources.most_common(5))}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare commerce prefilter train/val/test splits"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("datasets/commerce_prefilter/raw_examples.jsonl"),
        help="Input JSONL with collected examples"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("datasets/commerce_prefilter/splits"),
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training set"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation set"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Dataset Preparation")
    print("=" * 60)

    # Load examples
    print(f"\nLoading from {args.input}...")
    examples = load_examples(args.input)
    print(f"Loaded {len(examples)} examples")

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Split
    print(f"\nSplitting with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    train, val, test = stratified_split(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Print stats
    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Test", test)

    # Save
    print(f"\nSaving to {args.output}/...")
    save_split(train, args.output / "train.jsonl")
    save_split(val, args.output / "val.jsonl")
    save_split(test, args.output / "test.jsonl")

    print("\nDone!")
    print(f"  train.jsonl: {len(train)} examples")
    print(f"  val.jsonl: {len(val)} examples")
    print(f"  test.jsonl: {len(test)} examples")


if __name__ == "__main__":
    main()
