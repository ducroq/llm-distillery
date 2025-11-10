"""
Find articles about truly deployed, commercial-scale technology.

Strategy: Much stricter filtering - look for strong signals of:
1. Commercial scale (gigawatts, billions, millions of units)
2. Market data (market share, revenue figures, sales data)
3. Established infrastructure (factories, supply chains, installations)
4. Proven track record (years of operation, customer base)
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse


def has_strong_deployment_signals(article: Dict[str, Any]) -> tuple[bool, int, List[str]]:
    """Check for STRONG deployment signals - commercial scale, market data, infrastructure."""

    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    signals = []
    score = 0

    # SCALE INDICATORS (very strong signal)
    scale_patterns = [
        (r'\d+\s*gigawatts?\b', 'gigawatt_scale', 3),
        (r'\d+\s*terawatts?\b', 'terawatt_scale', 3),
        (r'\d+\s*million (units|installations|customers|users)', 'million_units', 3),
        (r'\d+\s*billion (dollars|revenue|sales)', 'billion_revenue', 3),
        (r'market share of \d+%', 'market_share_data', 2),
        (r'\d+% of (global|world|market)', 'market_percentage', 2),
    ]

    for pattern, name, points in scale_patterns:
        if re.search(pattern, text):
            signals.append(name)
            score += points

    # COMMERCIAL OPERATION (strong signal)
    commercial_patterns = [
        (r'commercially operating since \d{4}', 'operating_since', 2),
        (r'in commercial operation', 'commercial_operation', 2),
        (r'commercial deployment', 'commercial_deployment', 2),
        (r'\d+ years? of operation', 'years_operating', 2),
        (r'market leader', 'market_leader', 2),
        (r'industry standard', 'industry_standard', 2),
    ]

    for pattern, name, points in commercial_patterns:
        if re.search(pattern, text):
            signals.append(name)
            score += points

    # INFRASTRUCTURE (moderate signal)
    infrastructure_patterns = [
        (r'manufacturing facility|factory|production plant', 'manufacturing_facility', 1),
        (r'supply chain', 'supply_chain', 1),
        (r'\d+,\d+ installations', 'installations_count', 1),
        (r'installed capacity', 'installed_capacity', 1),
    ]

    for pattern, name, points in infrastructure_patterns:
        if re.search(pattern, text):
            signals.append(name)
            score += points

    # PROVEN TECHNOLOGY (moderate signal)
    proven_patterns = [
        (r'proven technology', 'proven_tech', 1),
        (r'established technology', 'established_tech', 1),
        (r'mature market', 'mature_market', 1),
        (r'widely adopted', 'widely_adopted', 1),
    ]

    for pattern, name, points in proven_patterns:
        if re.search(pattern, text):
            signals.append(name)
            score += points

    # NEGATIVE SIGNALS - reduce score for vaporware indicators
    vaporware_patterns = [
        r'\bwill\b.*\b(deploy|launch|introduce)',
        r'\bplans to\b',
        r'\baims to\b',
        r'\bexpects to\b',
        r'\bpromises\b',
        r'\bcould\b.*\btransform',
        r'\bmay\b.*\brevolutionize',
        r'\bpotential\b.*\bbreakthrough',
        r'\bannounced\b.*\bfunding',
        r'\braises \$\d+',
        r'\bseries [A-D]\b',
        r'\bpilot project\b',
        r'\bdemonstration\b',
        r'\bprototype\b',
    ]

    vaporware_count = sum(1 for p in vaporware_patterns if re.search(p, text))
    if vaporware_count >= 3:
        score -= 5  # Heavy penalty for vaporware language
    elif vaporware_count >= 2:
        score -= 2

    # Require minimum score threshold
    return score >= 5, score, signals


def main():
    parser = argparse.ArgumentParser(description='Find high-deployment articles')
    parser.add_argument('--source', required=True, help='Source JSONL pattern')
    parser.add_argument('--labeled-dir', help='Directory with already-labeled articles')
    parser.add_argument('--target-count', type=int, default=200, help='Target candidates')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load already labeled IDs
    already_labeled = set()
    if args.labeled_dir:
        labeled_dir = Path(args.labeled_dir)
        if labeled_dir.exists():
            for batch_file in labeled_dir.glob('labeled_batch_*.jsonl'):
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            article = json.loads(line)
                            already_labeled.add(article.get('id'))

    print(f'Already labeled: {len(already_labeled):,} articles')

    # Find high-deployment candidates
    candidates = []
    source_files = list(Path('.').glob(args.source))
    print(f'\nScanning {len(source_files)} source files...')

    for source_file in source_files:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    article_id = article.get('id')

                    if article_id in already_labeled:
                        continue

                    has_signals, score, signals = has_strong_deployment_signals(article)

                    if has_signals:
                        candidates.append({
                            'article': article,
                            'deployment_score': score,
                            'signals': signals
                        })

                        if len(candidates) >= args.target_count * 3:
                            break

        if len(candidates) >= args.target_count * 3:
            break

    print(f'\nFound {len(candidates):,} high-deployment candidates')

    # Sort by deployment score and take top N
    candidates.sort(key=lambda x: x['deployment_score'], reverse=True)
    selected = candidates[:args.target_count]

    print(f'Selected top {len(selected):,} by deployment score')

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item['article'], ensure_ascii=False) + '\n')

    print(f'\nWrote {len(selected):,} articles to: {output_path}')

    # Show statistics
    scores = [item['deployment_score'] for item in selected]
    print(f'\nDeployment score distribution:')
    print(f'  Min:    {min(scores)}')
    print(f'  Max:    {max(scores)}')
    print(f'  Mean:   {sum(scores)/len(scores):.1f}')
    print(f'  Median: {sorted(scores)[len(scores)//2]}')

    # Show top signals
    from collections import Counter
    all_signals = []
    for item in selected:
        all_signals.extend(item['signals'])

    signal_counts = Counter(all_signals)
    print(f'\nTop deployment signals:')
    for signal, count in signal_counts.most_common(10):
        print(f'  {signal:25s}: {count}')


if __name__ == '__main__':
    main()
