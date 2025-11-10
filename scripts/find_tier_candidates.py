"""
Find tier-specific candidates from corpus for balanced dataset creation.

This script analyzes the corpus to find articles likely to score in specific tiers
based on source type and content signals, enabling targeted oracle labeling.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set
import argparse


# Tier-specific search patterns
TIER_1_PATTERNS = [
    # Scale indicators (deployed systems)
    (r'\d+\s*gigawatts?\s+(deployed|installed|operating|capacity)', 'gigawatt_scale', 3),
    (r'\d+\s*terawatt', 'terawatt_scale', 3),
    (r'\d+\s*million\s+(units|installations|systems)\s+(installed|deployed|operating)', 'million_units', 3),

    # Operational indicators
    (r'operating since \d{4}', 'operating_since', 2),
    (r'(generated|produced)\s+\d+.*(terawatt|gigawatt).*hours', 'energy_generated', 2),
    (r'grid(-|\s)connected', 'grid_connected', 2),

    # Market indicators
    (r'market share of \d+%', 'market_share', 2),
    (r'\d+\s*billion.*(revenue|sales)', 'billion_revenue', 3),
    (r'industry standard', 'industry_standard', 2),
    (r'widely adopted', 'widely_adopted', 2),
    (r'dominant (technology|platform|solution)', 'dominant', 2),
]

TIER_2_PATTERNS = [
    # Commercial indicators
    (r'first commercial (deployment|sale|installation|customer)', 'first_commercial', 2),
    (r'(revenue|sales).*(doubled|tripled|grew)', 'revenue_growth', 2),
    (r'\d+\s*(customers|clients|contracts)', 'customer_count', 2),
    (r'commercially available', 'commercially_available', 2),
    (r'market entry', 'market_entry', 1),
    (r'(launched|shipping|available)\s+now', 'launched_now', 1),

    # Pilot to commercial
    (r'pilot.*(completed|successful).*(commercial|production)', 'pilot_to_commercial', 2),
    (r'moving (to|into) production', 'moving_to_production', 2),
]

TIER_3_PATTERNS = [
    # Pilot/demonstration
    (r'pilot (project|program|deployment)', 'pilot_project', 2),
    (r'demonstration (project|site|facility)', 'demonstration', 2),
    (r'testing (at|in|with)', 'testing', 1),
    (r'trial (deployment|installation)', 'trial', 1),

    # Funding for pilots
    (r'(grant|funding).*(pilot|demonstration|test)', 'pilot_funding', 2),
    (r'(DoE|Department of Energy|EU|Horizon).*(awarded|grant)', 'government_grant', 2),
]

# Vaporware patterns (negative signals - avoid labeling these for high tiers)
VAPORWARE_PENALTIES = [
    (r'\bwill\b.*(deploy|launch|release)', 'future_will', -2),
    (r'\bplans to\b', 'plans_to', -2),
    (r'\baims to\b', 'aims_to', -1),
    (r'\bconcept\b', 'concept', -2),
    (r'\bproposes\b', 'proposes', -2),
    (r'\bhopes to\b', 'hopes_to', -2),
    (r'\bif successful\b', 'if_successful', -1),
]

# Source tags that are likely to have high-tier content
TIER_1_SOURCES = [
    'climate_solutions_irena',
    'energy_utilities_pv_magazine',
    'energy_utilities_utility_dive',
    'science_ieee',
    'industry_intelligence_mckinsey',
    'professional_business_forbes',
]

TIER_2_SOURCES = [
    'fintech_markets',
    'professional_business',
    'industry_intelligence',
    'energy_utilities',
    'automotive_transport',
]

TIER_3_SOURCES = [
    'venture_startup',
    'science_arxiv',
    'science_phys',
    'economics_nber',
    'community_social',
]


def score_article_for_tier(article: Dict, tier: int) -> tuple[bool, int, List[str]]:
    """
    Score an article for likelihood of being in a specific tier.

    Returns:
        (qualifies, score, signals_found)
    """
    text = f"{article.get('title', '')} {article.get('content', '')}".lower()
    source = article.get('source', '')

    score = 0
    signals = []

    # Check patterns based on tier
    if tier == 1:
        patterns = TIER_1_PATTERNS
    elif tier == 2:
        patterns = TIER_2_PATTERNS
    else:
        patterns = TIER_3_PATTERNS

    for pattern, name, points in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += points
            signals.append(name)

    # Apply vaporware penalties
    for pattern, name, penalty in VAPORWARE_PENALTIES:
        if re.search(pattern, text, re.IGNORECASE):
            score += penalty
            signals.append(f'penalty_{name}')

    # Source bonus
    if tier == 1 and any(src in source for src in TIER_1_SOURCES):
        score += 2
        signals.append('tier1_source')
    elif tier == 2 and any(src in source for src in TIER_2_SOURCES):
        score += 1
        signals.append('tier2_source')
    elif tier == 3 and any(src in source for src in TIER_3_SOURCES):
        score += 1
        signals.append('tier3_source')

    # Qualifying thresholds
    thresholds = {
        1: 5,  # Tier 1 needs strong signals (≥5 points)
        2: 3,  # Tier 2 needs moderate signals (≥3 points)
        3: 2,  # Tier 3 needs weak signals (≥2 points)
    }

    qualifies = score >= thresholds[tier]

    return qualifies, score, signals


def main():
    parser = argparse.ArgumentParser(description='Find tier-specific candidates')
    parser.add_argument('--source', required=True, help='Source files glob pattern')
    parser.add_argument('--tier', type=int, required=True, choices=[1, 2, 3],
                        help='Target tier (1=deployed, 2=early_commercial, 3=pilot)')
    parser.add_argument('--target-count', type=int, default=1000,
                        help='Target number of candidates to find')
    parser.add_argument('--labeled-dir', help='Directory with already labeled articles (to skip)')
    parser.add_argument('--output', required=True, help='Output JSONL file for candidates')

    args = parser.parse_args()

    # Load already labeled IDs
    labeled_ids: Set[str] = set()
    if args.labeled_dir:
        labeled_dir = Path(args.labeled_dir)
        if labeled_dir.exists():
            for batch_file in labeled_dir.glob('labeled_batch_*.jsonl'):
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            article = json.loads(line)
                            labeled_ids.add(article['id'])
            print(f'Already labeled: {len(labeled_ids):,} articles')

    # Scan source files
    source_files = list(Path().glob(args.source))
    print(f'\nScanning {len(source_files)} source files...')

    candidates = []
    signal_counter = Counter()
    source_counter = Counter()

    for source_file in source_files:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                article = json.loads(line)
                article_id = article.get('id')

                # Skip if already labeled
                if article_id in labeled_ids:
                    continue

                # Score for target tier
                qualifies, score, signals = score_article_for_tier(article, args.tier)

                if qualifies:
                    candidates.append({
                        'article': article,
                        'score': score,
                        'signals': signals
                    })

                    for signal in signals:
                        signal_counter[signal] += 1

                    source_counter[article.get('source', 'unknown')] += 1

    print(f'\nFound {len(candidates):,} candidates')

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Take top N
    candidates = candidates[:args.target_count]

    print(f'Selected top {len(candidates):,} by deployment score')

    # Write to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for cand in candidates:
            f.write(json.dumps(cand['article'], ensure_ascii=False) + '\n')

    print(f'\nWrote {len(candidates)} articles to: {output_path}')

    # Print statistics
    scores = [c['score'] for c in candidates]
    print(f'\nDeployment score distribution:')
    print(f'  Min:    {min(scores) if scores else 0}')
    print(f'  Max:    {max(scores) if scores else 0}')
    print(f'  Mean:   {sum(scores)/len(scores) if scores else 0:.1f}')
    print(f'  Median: {sorted(scores)[len(scores)//2] if scores else 0}')

    print(f'\nTop deployment signals:')
    for signal, count in signal_counter.most_common(10):
        print(f'  {signal:25s}: {count}')

    print(f'\nTop sources:')
    for source, count in source_counter.most_common(10):
        print(f'  {source:40s}: {count}')


if __name__ == '__main__':
    main()
