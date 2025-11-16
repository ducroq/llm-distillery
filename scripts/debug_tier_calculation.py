"""
Debug script to verify tier calculation logic for sustainability_tech_deployment.

This script checks:
1. What fields exist in the analysis
2. How dimension scores are structured
3. What the calculated average should be
"""

import json
import sys
from pathlib import Path

def debug_tier_calculation(scored_file: Path):
    """Debug tier calculation for first article."""

    print("=" * 70)
    print("TIER CALCULATION DEBUG")
    print("=" * 70)
    print()

    # Load first article
    with open(scored_file, 'r', encoding='utf-8') as f:
        article = json.loads(f.readline())

    print(f"Article ID: {article.get('id')}")
    print(f"Title: {article.get('title', '')[:80]}")
    print()

    # Get analysis
    analysis = article.get('sustainability_tech_deployment_analysis', {})

    if not analysis:
        print("ERROR: No sustainability_tech_deployment_analysis field found!")
        return

    print(f"Analysis keys: {list(analysis.keys())}")
    print()

    # Check for overall_score fields
    print("Overall score fields:")
    print(f"  overall_score: {analysis.get('overall_score')}")
    print(f"  overall_uplift_score: {analysis.get('overall_uplift_score')}")
    print()

    # Check dimensions
    dimensions = [
        'deployment_maturity',
        'technology_performance',
        'cost_trajectory',
        'scale_of_deployment',
        'market_penetration',
        'technology_readiness',
        'supply_chain_maturity',
        'proof_of_impact'
    ]

    print("Dimension structure:")
    for dim in dimensions:
        dim_data = analysis.get(dim)
        print(f"  {dim}:")
        print(f"    value: {dim_data}")
        print(f"    type: {type(dim_data)}")
        if isinstance(dim_data, dict):
            print(f"    has 'score': {'score' in dim_data}")
            if 'score' in dim_data:
                print(f"    score value: {dim_data['score']}")

    print()

    # Calculate average (mimicking prepare_data.py logic)
    print("Calculating average (prepare_data.py logic):")
    scores = []
    for dim in dimensions:
        dim_data = analysis.get(dim)
        if isinstance(dim_data, dict) and 'score' in dim_data:
            score = dim_data['score']
            scores.append(score)
            print(f"  {dim}: {score}")
        elif isinstance(dim_data, (int, float)):
            scores.append(dim_data)
            print(f"  {dim}: {dim_data} (direct value)")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nCollected {len(scores)} scores: {scores}")
        print(f"Average: {avg:.2f}")
        print()

        # Determine tier
        if avg >= 8.0:
            tier = 'mass_deployment'
        elif avg >= 6.5:
            tier = 'commercial_proven'
        elif avg >= 5.0:
            tier = 'early_commercial'
        elif avg >= 3.0:
            tier = 'pilot_stage'
        else:
            tier = 'vaporware'

        print(f"Tier assignment: {tier}")
        print()
        print("Tier thresholds:")
        print("  mass_deployment: avg >= 8.0")
        print("  commercial_proven: avg >= 6.5")
        print("  early_commercial: avg >= 5.0")
        print("  pilot_stage: avg >= 3.0")
        print("  vaporware: avg < 3.0")
    else:
        print("ERROR: No scores collected!")

    print()
    print("=" * 70)


if __name__ == '__main__':
    scored_file = Path('datasets/scored/sustainability_tech_deployment_v1/sustainability_tech_deployment/scored_batch_001.jsonl')

    if not scored_file.exists():
        print(f"ERROR: File not found: {scored_file}")
        print("Make sure you run this from the llm-distillery root directory")
        sys.exit(1)

    debug_tier_calculation(scored_file)
