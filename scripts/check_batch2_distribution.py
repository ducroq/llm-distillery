"""
Quick analysis of Batch 2 aggressive labeling distribution.

This script analyzes the tier distribution from the new batch of 2,000
randomized labels to see if we've found more high-tier examples.
"""

import json
from pathlib import Path
from collections import Counter

def analyze_batch():
    """Analyze tier distribution from batch 2."""

    label_dir = Path("datasets/labeled/tech_deployment_aggressive_labeling/sustainability_tech_deployment")

    if not label_dir.exists():
        print(f"ERROR: Label directory not found: {label_dir}")
        return

    # Load all labels
    labels = []
    batch_files = sorted(label_dir.glob("labeled_batch_*.jsonl"))

    if not batch_files:
        print(f"ERROR: No batch files found in {label_dir}")
        return

    print(f"Found {len(batch_files)} batch files")
    print(f"Files: {[f.name for f in batch_files]}\n")

    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labels.append(json.loads(line))

    print(f"Total labels loaded: {len(labels)}\n")

    # Tier distribution
    tier_counts = Counter()
    score_ranges = {
        'deployed': [],
        'early_comm': [],
        'pilot': [],
        'vaporware': []
    }

    for label in labels:
        analysis = label.get('sustainability_tech_deployment_analysis', {})
        score = analysis.get('overall_score', 0)
        tier = analysis.get('tier', 'unknown')

        tier_counts[tier] += 1

        # Categorize by tier thresholds
        if score >= 8.0:
            score_ranges['deployed'].append(score)
        elif score >= 6.0:
            score_ranges['early_comm'].append(score)
        elif score >= 4.0:
            score_ranges['pilot'].append(score)
        else:
            score_ranges['vaporware'].append(score)

    # Print distribution
    print("=" * 60)
    print("TIER DISTRIBUTION (Batch 2)")
    print("=" * 60)

    total = len(labels)

    for tier_name, tier_key in [
        ("Deployed (≥8.0)", 'deployed'),
        ("Early Commercial (6.0-7.9)", 'early_comm'),
        ("Pilot (4.0-5.9)", 'pilot'),
        ("Vaporware (<4.0)", 'vaporware')
    ]:
        count = len(score_ranges[tier_key])
        pct = (count / total * 100) if total > 0 else 0
        print(f"{tier_name:30} {count:5} ({pct:5.1f}%)")

    print(f"\n{'Total':30} {total:5}")

    # Compare to Batch 1 (from CURRENT_TASK.md)
    print("\n" + "=" * 60)
    print("COMPARISON TO BATCH 1")
    print("=" * 60)

    batch1_dist = {
        'vaporware': 81.2,
        'pilot': 10.8,
        'early_comm': 6.7,
        'deployed': 1.4
    }

    print(f"{'Tier':30} {'Batch 1':>10} {'Batch 2':>10} {'Change':>10}")
    print("-" * 60)

    for tier_name, tier_key in [
        ("Deployed", 'deployed'),
        ("Early Commercial", 'early_comm'),
        ("Pilot", 'pilot'),
        ("Vaporware", 'vaporware')
    ]:
        batch1_pct = batch1_dist[tier_key]
        batch2_pct = (len(score_ranges[tier_key]) / total * 100) if total > 0 else 0
        change = batch2_pct - batch1_pct
        change_str = f"{change:+.1f}%"

        print(f"{tier_name:30} {batch1_pct:9.1f}% {batch2_pct:9.1f}% {change_str:>10}")

    # Score statistics
    print("\n" + "=" * 60)
    print("SCORE STATISTICS (Batch 2)")
    print("=" * 60)

    all_scores = [label.get('sustainability_tech_deployment_analysis', {}).get('overall_score', 0) for label in labels]

    if all_scores:
        print(f"Min score:  {min(all_scores):.2f}")
        print(f"Max score:  {max(all_scores):.2f}")
        print(f"Mean score: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Median:     {sorted(all_scores)[len(all_scores)//2]:.2f}")

    # High-value finds
    print("\n" + "=" * 60)
    print("HIGH-VALUE EXAMPLES (score ≥ 7.0)")
    print("=" * 60)

    high_scores = [
        (label.get('id', 'unknown'),
         label.get('title', 'No title')[:60],
         label.get('sustainability_tech_deployment_analysis', {}).get('overall_score', 0))
        for label in labels
        if label.get('sustainability_tech_deployment_analysis', {}).get('overall_score', 0) >= 7.0
    ]

    high_scores.sort(key=lambda x: x[2], reverse=True)

    if high_scores:
        print(f"\nFound {len(high_scores)} examples with score ≥ 7.0:\n")
        for article_id, title, score in high_scores[:20]:  # Top 20
            print(f"{score:.1f} | {title}")
    else:
        print("\nNo examples with score ≥ 7.0 found")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    deployed_pct = (len(score_ranges['deployed']) / total * 100) if total > 0 else 0
    early_comm_pct = (len(score_ranges['early_comm']) / total * 100) if total > 0 else 0

    if deployed_pct > 3.0 or early_comm_pct > 10.0:
        print("✅ Batch 2 shows IMPROVEMENT in high-tier distribution!")
        print("   → CONTINUE labeling more batches (Batch 3, 4, etc.)")
        print(f"   → Potential: {len(labels) * (9839 / 2000):.0f} total labels from remaining corpus")
    else:
        print("⚠️  Batch 2 shows SIMILAR distribution to Batch 1")
        print("   → Corpus fundamentally imbalanced (news aggregator bias)")
        print("   → RECOMMENDATION: Accept imbalance, proceed to training with mitigation:")
        print("      • Oversampling minority classes")
        print("      • Class weighting")
        print("      • Synthetic augmentation for deployed tier only")
        print(f"   → Cost to label all remaining: ~${(9839 - 2000) * 0.001:.2f}")


if __name__ == '__main__':
    analyze_batch()
