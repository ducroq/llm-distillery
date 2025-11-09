"""
Prepare training data for sustainability_tech_deployment model distillation.

This script:
1. Loads oracle-labeled data
2. Splits into train/val/test sets (stratified by tier)
3. Converts to simplified training format (score arrays only, matching uplifting filter)
4. Applies oversampling for minority classes
5. Exports in JSONL format for Unsloth/HuggingFace training
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import argparse


# Dimension order (must match config.yaml)
DIMENSION_NAMES = [
    "deployment_maturity",
    "technology_performance",
    "cost_trajectory",
    "scale_of_deployment",
    "market_penetration",
    "technology_readiness",
    "supply_chain_maturity",
    "proof_of_impact"
]


def load_labels(input_file: Path) -> List[Dict[str, Any]]:
    """Load oracle labels from JSONL file."""
    labels = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))
    return labels


def assign_tier(overall_score: float) -> str:
    """Assign tier based on overall score."""
    if overall_score >= 8.0:
        return 'deployed'
    elif overall_score >= 6.0:
        return 'early_commercial'
    elif overall_score >= 4.0:
        return 'pilot'
    else:
        return 'vaporware'


def stratified_split(
    labels: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into train/val/test sets with stratification by tier."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

    random.seed(seed)

    # Group by tier
    tier_groups = {}
    for label in labels:
        analysis = label.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score', 0.0)
        tier = assign_tier(overall_score)

        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(label)

    # Split each tier
    train_set = []
    val_set = []
    test_set = []

    for tier, items in tier_groups.items():
        random.shuffle(items)

        n = len(items)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_set.extend(items[:train_end])
        val_set.extend(items[train_end:val_end])
        test_set.extend(items[val_end:])

    # Shuffle final sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def oversample_minority_classes(
    train_set: List[Dict[str, Any]],
    target_ratio: float = 0.2
) -> List[Dict[str, Any]]:
    """Oversample minority classes to improve balance."""
    # Count by tier
    tier_counts = Counter()
    tier_items = {}

    for item in train_set:
        analysis = item.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score', 0.0)
        tier = assign_tier(overall_score)
        tier_counts[tier] += 1
        if tier not in tier_items:
            tier_items[tier] = []
        tier_items[tier].append(item)

    # Calculate target count for minority classes
    majority_count = tier_counts.most_common(1)[0][1]
    target_count = int(majority_count * target_ratio)

    # Oversample
    oversampled = list(train_set)  # Start with original

    for tier, count in tier_counts.items():
        if count < target_count and tier != 'vaporware':
            # Calculate how many copies we need
            multiplier = target_count // count
            remainder = target_count % count

            # Add full copies
            for _ in range(multiplier - 1):  # -1 because original already in list
                oversampled.extend(tier_items[tier])

            # Add remainder
            oversampled.extend(random.sample(tier_items[tier], remainder))

    random.shuffle(oversampled)
    return oversampled


def convert_to_training_format(
    labels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert oracle labels to simplified training format (score arrays only).

    Matches uplifting filter format:
    {
        "id": "article-123",
        "title": "...",
        "content": "...",
        "url": "...",
        "labels": [8, 7, 6, 8, 5, 7, 6, 8],  # dimension scores as array
        "dimension_names": ["deployment_maturity", "technology_performance", ...]
    }
    """
    training_data = []

    for label in labels:
        analysis = label.get('sustainability_tech_deployment_analysis', {})

        if not analysis:
            continue  # Skip if no analysis

        # Extract dimension scores in correct order
        dimensions = analysis.get('dimensions', {})
        score_array = [dimensions.get(dim, 0) for dim in DIMENSION_NAMES]

        # Truncate content to match oracle input (~800 words)
        content = label.get('content', label.get('description', ''))
        words = content.split()
        if len(words) > 800:
            content = ' '.join(words[:800]) + '...'

        training_data.append({
            'id': label.get('id', ''),
            'title': label.get('title', ''),
            'content': content,
            'url': label.get('url', ''),
            'labels': score_array,
            'dimension_names': DIMENSION_NAMES
        })

    return training_data


def save_training_data(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    output_dir: Path
):
    """Save training, validation, and test data to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train set
    train_file = output_dir / 'train.jsonl'
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save val set
    val_file = output_dir / 'val.jsonl'
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save test set
    test_file = output_dir / 'test.jsonl'
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'\nSaved training data:')
    print(f'  Train: {train_file} ({len(train_data):,} examples)')
    print(f'  Val:   {val_file} ({len(val_data):,} examples)')
    print(f'  Test:  {test_file} ({len(test_data):,} examples)')


def print_statistics(
    labels: List[Dict[str, Any]],
    train_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    train_data: List[Dict[str, Any]]
):
    """Print dataset statistics."""
    print('\n' + '=' * 70)
    print('DATASET STATISTICS')
    print('=' * 70)

    # Original distribution
    print(f'\nOriginal Dataset:')
    tier_counts = Counter()
    for label in labels:
        analysis = label.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score', 0.0)
        tier = assign_tier(overall_score)
        tier_counts[tier] += 1

    total = len(labels)
    for tier in ['vaporware', 'pilot', 'early_commercial', 'deployed']:
        count = tier_counts[tier]
        pct = count / total * 100 if total > 0 else 0
        print(f'  {tier:20s}: {count:5,} ({pct:5.1f}%)')

    # Split sizes
    print(f'\nSplit Sizes (Stratified):')
    print(f'  Train: {len(train_set):,} labels')
    print(f'  Val:   {len(val_set):,} labels')
    print(f'  Test:  {len(test_set):,} labels')

    # After oversampling
    print(f'\nAfter Oversampling (Train Only):')
    print(f'  Train: {len(train_data):,} examples')

    # Tier distribution (train after oversampling)
    train_tiers = Counter()
    for item in train_data:
        # Calculate tier from labels array
        # Use weighted average matching config.yaml weights
        weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07]
        overall_score = sum(score * weight for score, weight in zip(item['labels'], weights))
        tier = assign_tier(overall_score)
        train_tiers[tier] += 1

    print(f'\nTrain Tier Distribution (After Oversampling):')
    for tier in ['vaporware', 'pilot', 'early_commercial', 'deployed']:
        count = train_tiers[tier]
        pct = count / len(train_data) * 100 if len(train_data) > 0 else 0
        print(f'  {tier:20s}: {count:5,} ({pct:5.1f}%)')

    # Validation tier distribution (natural, no oversampling)
    val_tiers = Counter()
    for label in val_set:
        analysis = label.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score', 0.0)
        tier = assign_tier(overall_score)
        val_tiers[tier] += 1

    print(f'\nVal Tier Distribution (Natural, No Oversampling):')
    for tier in ['vaporware', 'pilot', 'early_commercial', 'deployed']:
        count = val_tiers[tier]
        pct = count / len(val_set) * 100 if len(val_set) > 0 else 0
        print(f'  {tier:20s}: {count:5,} ({pct:5.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for tech deployment distillation')
    parser.add_argument('--input', required=True, help='Input JSONL file with oracle labels')
    parser.add_argument('--output-dir', required=True, help='Output directory for training data')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--oversample-ratio', type=float, default=0.2, help='Target minority class ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load labels
    print(f'Loading labels from: {args.input}')
    labels = load_labels(Path(args.input))
    print(f'Loaded {len(labels):,} labels')

    # Split
    print(f'\nSplitting into train/val/test ({args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%})...')
    train_set, val_set, test_set = stratified_split(
        labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Oversample train set only
    print(f'\nOversampling minority classes (target ratio: {args.oversample_ratio:.1%})...')
    train_set_oversampled = oversample_minority_classes(train_set, args.oversample_ratio)

    # Convert to training format (simplified score arrays)
    print(f'\nConverting to training format (score arrays only)...')
    train_data = convert_to_training_format(train_set_oversampled)
    val_data = convert_to_training_format(val_set)
    test_data = convert_to_training_format(test_set)

    # Print statistics
    print_statistics(labels, train_set, val_set, test_set, train_data)

    # Save
    save_training_data(train_data, val_data, test_data, Path(args.output_dir))

    print('\n' + '=' * 70)
    print('TRAINING DATA PREPARATION COMPLETE')
    print('=' * 70)
    print(f'\nOutput directory: {args.output_dir}')
    print(f'Format: Simplified score arrays (matches uplifting filter)')
    print(f'Stratification: Maintains tier proportions across splits')
    print(f'Oversampling: Applied to training set only')


if __name__ == '__main__':
    main()
