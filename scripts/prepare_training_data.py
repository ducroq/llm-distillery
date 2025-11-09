"""
Prepare training data for model distillation.

This script:
1. Loads oracle-labeled data
2. Splits into train/val sets (stratified by tier)
3. Converts to training format (prompt/completion pairs)
4. Applies oversampling for minority classes
5. Exports in JSONL format for Unsloth/HuggingFace training
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import argparse


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
        return 'deployed_proven'
    elif overall_score >= 6.0:
        return 'early_commercial'
    elif overall_score >= 4.0:
        return 'pilot'
    else:
        return 'vaporware'


def stratified_split(
    labels: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into train/val sets with stratification by tier."""
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

    for tier, items in tier_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - val_ratio))
        train_set.extend(items[:split_idx])
        val_set.extend(items[split_idx:])

    # Shuffle final sets
    random.shuffle(train_set)
    random.shuffle(val_set)

    return train_set, val_set


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


def build_training_prompt(article: Dict[str, Any]) -> str:
    """Build prompt for training (same format as oracle prompt)."""
    title = article.get('title', 'No title')
    content = article.get('content', article.get('description', 'No content'))

    # Truncate if too long (rough heuristic: ~2000 words max)
    words = content.split()
    if len(words) > 2000:
        content = ' '.join(words[:2000]) + '...'

    prompt = f"""Analyze this article about technology for deployment maturity.

**Title**: {title}

**Content**: {content}

Evaluate the following 8 dimensions on a scale of 1-10:

1. **deployment_maturity**: How widely deployed is this technology in real-world use?
2. **technology_performance**: How well does the technology work technically?
3. **cost_trajectory**: What is the cost trend and economic viability?
4. **scale_of_deployment**: What is the scale of current deployments?
5. **market_penetration**: How much market share or adoption exists?
6. **technology_readiness**: What is the technical maturity level?
7. **supply_chain_maturity**: How developed is the supply chain?
8. **proof_of_impact**: What evidence exists of real-world impact?

Provide your analysis in JSON format with dimension scores, reasoning, and overall score."""

    return prompt


def build_training_completion(analysis: Dict[str, Any]) -> str:
    """Build completion for training (oracle output format)."""
    # Extract dimensions
    dimensions = analysis.get('dimensions', {})
    reasoning = analysis.get('reasoning', '')
    overall_score = analysis.get('overall_score', 0.0)

    completion = {
        'dimensions': dimensions,
        'reasoning': reasoning,
        'overall_score': overall_score
    }

    return json.dumps(completion, ensure_ascii=False)


def convert_to_training_format(
    labels: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Convert oracle labels to training format (prompt/completion pairs)."""
    training_data = []

    for label in labels:
        article = label
        analysis = label.get('sustainability_tech_deployment_analysis', {})

        if not analysis:
            continue  # Skip if no analysis

        prompt = build_training_prompt(article)
        completion = build_training_completion(analysis)

        training_data.append({
            'prompt': prompt,
            'completion': completion,
            'metadata': {
                'article_id': label.get('id'),
                'overall_score': analysis.get('overall_score', 0.0),
                'tier': assign_tier(analysis.get('overall_score', 0.0))
            }
        })

    return training_data


def save_training_data(
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    output_dir: Path
):
    """Save training and validation data to JSONL files."""
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

    print(f'\nSaved training data:')
    print(f'  Train: {train_file} ({len(train_data):,} examples)')
    print(f'  Val:   {val_file} ({len(val_data):,} examples)')


def print_statistics(
    train_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]]
):
    """Print dataset statistics."""
    print('\n=== DATASET STATISTICS ===\n')

    # Original split
    print('Original Split:')
    print(f'  Train: {len(train_set):,} labels')
    print(f'  Val:   {len(val_set):,} labels')

    # After oversampling
    print(f'\nAfter Oversampling:')
    print(f'  Train: {len(train_data):,} examples')
    print(f'  Val:   {len(val_data):,} examples')

    # Tier distribution (train)
    train_tiers = Counter(item['metadata']['tier'] for item in train_data)
    print(f'\nTrain Tier Distribution:')
    for tier in ['vaporware', 'pilot', 'early_commercial', 'deployed_proven']:
        count = train_tiers[tier]
        pct = count / len(train_data) * 100
        print(f'  {tier:20s}: {count:5,} ({pct:5.1f}%)')

    # Tier distribution (val)
    val_tiers = Counter(item['metadata']['tier'] for item in val_data)
    print(f'\nVal Tier Distribution:')
    for tier in ['vaporware', 'pilot', 'early_commercial', 'deployed_proven']:
        count = val_tiers[tier]
        pct = count / len(val_data) * 100
        print(f'  {tier:20s}: {count:5,} ({pct:5.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for model distillation')
    parser.add_argument('--input', required=True, help='Input JSONL file with oracle labels')
    parser.add_argument('--output-dir', required=True, help='Output directory for training data')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--oversample-ratio', type=float, default=0.2, help='Target minority class ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load labels
    print(f'Loading labels from: {args.input}')
    labels = load_labels(Path(args.input))
    print(f'Loaded {len(labels):,} labels')

    # Split
    print(f'\nSplitting into train/val ({1-args.val_ratio:.0%}/{args.val_ratio:.0%})...')
    train_set, val_set = stratified_split(labels, args.val_ratio, args.seed)

    # Oversample train set
    print(f'\nOversampling minority classes (target ratio: {args.oversample_ratio:.1%})...')
    train_set_oversampled = oversample_minority_classes(train_set, args.oversample_ratio)

    # Convert to training format
    print(f'\nConverting to training format...')
    train_data = convert_to_training_format(train_set_oversampled)
    val_data = convert_to_training_format(val_set)

    # Save
    print_statistics(train_set, val_set, train_data, val_data)
    save_training_data(train_data, val_data, Path(args.output_dir))

    print('\n✅ Training data preparation complete!')


if __name__ == '__main__':
    main()
