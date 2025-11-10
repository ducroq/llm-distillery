"""
Find articles likely to score high on deployment dimensions.

Strategy: Use keyword matching to identify articles about deployed,
commercial, or widely-adopted technology rather than concepts/announcements.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse


def has_deployment_signals(article: Dict[str, Any], min_keywords: int = 3) -> tuple[bool, List[str]]:
    """Check if article has strong deployment signals."""

    # Keywords indicating deployed/commercial technology
    deployment_keywords = [
        # Direct deployment terms
        'deployed', 'deployment', 'rollout', 'roll out', 'rolled out',
        'in operation', 'operational', 'in production', 'producing',

        # Commercial terms
        'commercial', 'commercially available', 'on the market',
        'for sale', 'revenue', 'sales', 'sold',

        # Adoption/scale terms
        'adoption', 'adopted', 'market share', 'customers',
        'installations', 'installed capacity',
        'gigawatt', 'megawatt', 'terawatt',
        'million units', 'billion',

        # Manufacturing/supply chain
        'manufacturing', 'factory', 'production facility',
        'mass production', 'supply chain', 'shipping', 'delivered',

        # Market maturity
        'market leader', 'market penetration', 'widespread use',
        'standard', 'mainstream', 'commodity'
    ]

    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    matches = []
    for keyword in deployment_keywords:
        if keyword in text:
            matches.append(keyword)

    # Deduplicate
    matches = list(set(matches))

    return len(matches) >= min_keywords, matches


def load_already_labeled(labeled_dir: Path) -> set:
    """Load IDs of already-labeled articles to avoid duplicates."""
    labeled_ids = set()

    if labeled_dir.exists():
        for batch_file in labeled_dir.glob('labeled_batch_*.jsonl'):
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        article = json.loads(line)
                        labeled_ids.add(article.get('id'))

    return labeled_ids


def find_deployment_candidates(
    source_pattern: str,
    already_labeled: set,
    target_count: int = 1000,
    min_keywords: int = 3
) -> List[Dict[str, Any]]:
    """Find articles with deployment signals."""

    candidates = []

    source_files = list(Path('.').glob(source_pattern))
    print(f'Scanning {len(source_files)} source files...')

    for source_file in source_files:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    article_id = article.get('id')

                    # Skip already labeled
                    if article_id in already_labeled:
                        continue

                    # Check for deployment signals
                    has_signals, matches = has_deployment_signals(article, min_keywords)

                    if has_signals:
                        candidates.append({
                            'article': article,
                            'deployment_keywords': matches,
                            'keyword_count': len(matches)
                        })

                        if len(candidates) >= target_count * 2:  # Get 2x target for sampling
                            break

        if len(candidates) >= target_count * 2:
            break

    return candidates


def main():
    parser = argparse.ArgumentParser(description='Find deployment-focused articles')
    parser.add_argument('--source', required=True, help='Source JSONL pattern (e.g., "datasets/raw/*.jsonl")')
    parser.add_argument('--labeled-dir', help='Directory with already-labeled articles (to skip)')
    parser.add_argument('--target-count', type=int, default=500, help='Target number of candidates')
    parser.add_argument('--min-keywords', type=int, default=3, help='Minimum deployment keywords')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load already labeled IDs
    already_labeled = set()
    if args.labeled_dir:
        labeled_dir = Path(args.labeled_dir)
        already_labeled = load_already_labeled(labeled_dir)
        print(f'Loaded {len(already_labeled):,} already-labeled article IDs')

    # Find candidates
    print(f'\nSearching for articles with >={args.min_keywords} deployment keywords...')
    candidates = find_deployment_candidates(
        args.source,
        already_labeled,
        args.target_count,
        args.min_keywords
    )

    print(f'\nFound {len(candidates):,} candidate articles')

    # Sample target count
    random.seed(args.seed)
    if len(candidates) > args.target_count:
        # Prioritize articles with more keywords
        candidates.sort(key=lambda x: x['keyword_count'], reverse=True)

        # Take top 50% by keyword count, then sample randomly from rest
        top_half = candidates[:len(candidates)//2]
        bottom_half = candidates[len(candidates)//2:]

        # Sample evenly from both groups
        top_sample = random.sample(top_half, min(args.target_count//2, len(top_half)))
        bottom_sample = random.sample(bottom_half, min(args.target_count//2, len(bottom_half)))

        sampled = top_sample + bottom_sample
        random.shuffle(sampled)
        sampled = sampled[:args.target_count]
    else:
        sampled = candidates

    print(f'Sampled {len(sampled):,} articles for labeling')

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            # Just write the article (not the metadata)
            f.write(json.dumps(item['article'], ensure_ascii=False) + '\n')

    print(f'\nWrote {len(sampled):,} articles to: {output_path}')

    # Show statistics
    keyword_counts = [item['keyword_count'] for item in sampled]
    print(f'\nKeyword count distribution:')
    print(f'  Min:    {min(keyword_counts)}')
    print(f'  Max:    {max(keyword_counts)}')
    print(f'  Mean:   {sum(keyword_counts)/len(keyword_counts):.1f}')
    print(f'  Median: {sorted(keyword_counts)[len(keyword_counts)//2]}')


if __name__ == '__main__':
    main()
