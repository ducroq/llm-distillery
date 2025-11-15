"""
Generic training data preparation script for any filter.

This script:
1. Reads filter configuration to extract dimensions, tiers, analysis field
2. Loads oracle-labeled data
3. Splits into train/val/test sets (stratified by tier)
4. Converts to simplified training format (score arrays only)
5. Exports in JSONL format for Qwen training

Usage:
    python training/prepare_data.py \
        --filter filters/uplifting/v1 \
        --input datasets/labeled/uplifting/labeled_articles.jsonl \
        --output-dir datasets/training/uplifting \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import argparse


def load_filter_config(filter_dir: Path) -> Dict[str, Any]:
    """Load filter configuration from config.yaml."""
    config_file = filter_dir / 'config.yaml'

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def extract_filter_info(config: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, float]]:
    """Extract filter name, dimension names, and tier boundaries from config.

    Returns:
        (filter_name, dimension_names, tier_boundaries)
    """
    filter_name = config['filter']['name']

    # Extract dimensions in order (maintaining order from config)
    dimensions_config = config['scoring']['dimensions']
    dimension_names = list(dimensions_config.keys())

    # Extract tier boundaries
    tiers_config = config['scoring'].get('tiers', {})
    tier_boundaries = {}

    for tier_name, tier_info in tiers_config.items():
        threshold = tier_info.get('threshold', 0.0)
        tier_boundaries[tier_name] = threshold

    # Sort tiers by threshold (descending) for tier assignment
    tier_boundaries = dict(sorted(tier_boundaries.items(),
                                  key=lambda x: x[1],
                                  reverse=True))

    return filter_name, dimension_names, tier_boundaries


def get_analysis_field_name(filter_name: str) -> str:
    """Infer analysis field name from filter name.

    Examples:
        uplifting -> uplifting_analysis
        sustainability_tech_deployment -> sustainability_tech_deployment_analysis
    """
    return f"{filter_name}_analysis"


def load_labels(input_file: Path) -> List[Dict[str, Any]]:
    """Load oracle labels from JSONL file."""
    labels = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))
    return labels


def assign_tier(overall_score: float, tier_boundaries: Dict[str, float]) -> str:
    """Assign tier based on overall score and tier boundaries.

    Args:
        overall_score: The overall score
        tier_boundaries: Dict mapping tier names to minimum thresholds (sorted descending)

    Returns:
        Tier name (metadata only, not used in training)
    """
    for tier_name, threshold in tier_boundaries.items():
        if overall_score >= threshold:
            return tier_name

    # Return lowest tier if no match
    return list(tier_boundaries.keys())[-1]


def stratified_split(
    labels: List[Dict[str, Any]],
    analysis_field: str,
    tier_boundaries: Dict[str, float],
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
        analysis = label.get(analysis_field, {})

        # Try different field names for overall score
        overall_score = (analysis.get('overall_score') or
                        analysis.get('overall_uplift_score') or
                        0.0)

        tier = assign_tier(overall_score, tier_boundaries)

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


def convert_to_training_format(
    labels: List[Dict[str, Any]],
    analysis_field: str,
    dimension_names: List[str]
) -> List[Dict[str, Any]]:
    """Convert oracle labels to simplified training format (score arrays only).

    Format:
    {
        "id": "article-123",
        "title": "...",
        "content": "...",
        "url": "...",
        "labels": [7, 8, 6, 5, 7, 4, 6, 5],  # dimension scores as array
        "dimension_names": ["agency", "progress", ...]
    }
    """
    training_data = []

    for label in labels:
        analysis = label.get(analysis_field, {})

        if not analysis:
            continue  # Skip if no analysis

        # Extract dimension scores in correct order
        dimensions = analysis.get('dimensions', {})

        # Handle nested structure (dimensions can be objects with score/reasoning vs. flat score values)
        score_array = []
        for dim in dimension_names:
            dim_value = dimensions.get(dim, 0)

            # If it's a dict with 'score' field, extract score
            if isinstance(dim_value, dict):
                score_array.append(dim_value.get('score', 0))
            else:
                # Otherwise assume it's the score directly
                score_array.append(dim_value)

        # Get content (handle different field names)
        content = label.get('content', label.get('description', ''))

        training_data.append({
            'id': label.get('id', ''),
            'title': label.get('title', ''),
            'content': content,
            'url': label.get('url', ''),
            'labels': score_array,
            'dimension_names': dimension_names
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

    print(f"\nSaved training data:")
    print(f"  Train: {train_file} ({len(train_data)} examples)")
    print(f"  Val:   {val_file} ({len(val_data)} examples)")
    print(f"  Test:  {test_file} ({len(test_data)} examples)")


def print_statistics(
    labels: List[Dict[str, Any]],
    train_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    analysis_field: str,
    tier_boundaries: Dict[str, float]
):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    # Original distribution
    print("\nOriginal Dataset (Tier labels - metadata only):")
    tier_counts = Counter()
    for label in labels:
        analysis = label.get(analysis_field, {})
        overall_score = (analysis.get('overall_score') or
                        analysis.get('overall_uplift_score') or
                        0.0)
        tier = assign_tier(overall_score, tier_boundaries)
        tier_counts[tier] += 1

    total = len(labels)
    for tier_name in tier_boundaries.keys():
        count = tier_counts.get(tier_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {tier_name:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nSplit Sizes (Stratified):")
    print(f"  Train: {len(train_set)} labels")
    print(f"  Val:   {len(val_set)} labels")
    print(f"  Test:  {len(test_set)} labels")

    # Train tier distribution
    print(f"\nTrain Tier Distribution (Informational - not used in training):")
    train_tier_counts = Counter()
    for item in train_set:
        analysis = item.get(analysis_field, {})
        overall_score = (analysis.get('overall_score') or
                        analysis.get('overall_uplift_score') or
                        0.0)
        tier = assign_tier(overall_score, tier_boundaries)
        train_tier_counts[tier] += 1

    train_total = len(train_set)
    for tier_name in tier_boundaries.keys():
        count = train_tier_counts.get(tier_name, 0)
        pct = (count / train_total * 100) if train_total > 0 else 0
        print(f"  {tier_name:20s}: {count:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for any filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uplifting filter
  python scripts/prepare_training_data.py \\
      --filter filters/uplifting/v1 \\
      --input datasets/labeled/uplifting/labeled_articles.jsonl \\
      --output-dir datasets/training/uplifting

  # Tech deployment filter
  python scripts/prepare_training_data.py \\
      --filter filters/sustainability_tech_deployment/v1 \\
      --input datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \\
      --output-dir datasets/training/sustainability_tech_deployment
        """)

    parser.add_argument('--filter', type=str, required=True,
                       help='Path to filter directory (e.g., filters/uplifting/v1)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file with oracle labels')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for train/val/test splits')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    filter_dir = Path(args.filter)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    # Load filter configuration
    print(f"Loading filter configuration from: {filter_dir}")
    config = load_filter_config(filter_dir)
    filter_name, dimension_names, tier_boundaries = extract_filter_info(config)
    analysis_field = get_analysis_field_name(filter_name)

    print(f"Filter: {filter_name}")
    print(f"Dimensions: {len(dimension_names)} ({', '.join(dimension_names)})")
    print(f"Analysis field: {analysis_field}")
    print(f"Tiers: {len(tier_boundaries)} ({', '.join(tier_boundaries.keys())})")

    # Load labels
    print(f"\nLoading labels from: {input_file}")
    labels = load_labels(input_file)
    print(f"Loaded {len(labels)} labels")

    # Split data
    print(f"\nSplitting into train/val/test ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
    train_set, val_set, test_set = stratified_split(
        labels,
        analysis_field,
        tier_boundaries,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Convert to training format
    print(f"\nConverting to training format (score arrays only)...")
    train_data = convert_to_training_format(train_set, analysis_field, dimension_names)
    val_data = convert_to_training_format(val_set, analysis_field, dimension_names)
    test_data = convert_to_training_format(test_set, analysis_field, dimension_names)

    # Print statistics
    print_statistics(labels, train_set, val_set, test_set, analysis_field, tier_boundaries)

    # Save data
    save_training_data(train_data, val_data, test_data, output_dir)

    print("\n" + "="*70)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"\nFilter: {filter_name} ({len(dimension_names)} dimensions)")
    print(f"Output directory: {output_dir}")
    print(f"Format: Simplified score arrays")
    print(f"Stratification: Maintains tier proportions across splits")
    print(f"Note: Tier labels are metadata only, training uses dimensional scores")


if __name__ == '__main__':
    main()
