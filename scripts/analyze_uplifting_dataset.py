"""
Analyze uplifting dataset quality before training.

Checks:
1. Dimensional score distributions (full range coverage)
2. Bin population (are all score values represented?)
3. Correlation between dimensions
4. Tier distribution
5. Comparison to investment-risk quality metrics
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import glob


def load_scored_data(pattern: str):
    """Load all scored articles from glob pattern."""
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No files found matching: {pattern}")
        sys.exit(1)

    print(f"Loading {len(files)} files...")

    articles = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))

    return articles


def analyze_dimensions(articles):
    """Analyze dimensional score distributions."""

    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']

    stats = {}

    for dim in dimensions:
        scores = []
        for article in articles:
            analysis = article.get('uplifting_analysis', {})
            if dim in analysis:
                scores.append(analysis[dim])

        if not scores:
            print(f"WARNING: No scores found for dimension '{dim}'")
            continue

        # Calculate statistics
        scores_sorted = sorted(scores)
        n = len(scores)

        # Bin population (0-10 range)
        bins = {i: 0 for i in range(11)}
        for score in scores:
            bin_idx = int(round(score))
            if 0 <= bin_idx <= 10:
                bins[bin_idx] += 1

        bins_populated = sum(1 for count in bins.values() if count > 0)

        stats[dim] = {
            'count': n,
            'mean': sum(scores) / n,
            'median': scores_sorted[n // 2],
            'min': min(scores),
            'max': max(scores),
            'std': (sum((x - sum(scores)/n)**2 for x in scores) / n)**0.5,
            'bins': bins,
            'bins_populated': bins_populated,
            'bins_populated_pct': (bins_populated / 11) * 100
        }

    return stats


def analyze_tiers(articles):
    """Analyze tier distribution based on collective_benefit and wonder."""

    tier_counts = Counter()

    for article in articles:
        analysis = article.get('uplifting_analysis', {})

        cb = analysis.get('collective_benefit', 0)
        wonder = analysis.get('wonder', 0)

        # Uplifting criteria: CB >= 5.0 OR (Wonder >= 7.0 AND CB >= 3.0)
        is_uplifting = cb >= 5.0 or (wonder >= 7.0 and cb >= 3.0)

        if not is_uplifting:
            tier = "not_uplifting"
        else:
            # Calculate impact score
            dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                         'innovation', 'justice', 'resilience', 'wonder']
            impact = sum(analysis.get(d, 0) for d in dimensions) / len(dimensions)

            if impact >= 7.0:
                tier = "impact"
            elif cb >= 4.0:
                tier = "connection"
            else:
                tier = "not_uplifting"

        tier_counts[tier] += 1

    return tier_counts


def print_report(articles, dim_stats, tier_counts):
    """Print comprehensive quality report."""

    print("\n" + "=" * 80)
    print("UPLIFTING DATASET QUALITY REPORT")
    print("=" * 80)
    print()

    print(f"Total articles scored: {len(articles)}")
    print()

    # Dimensional statistics
    print("=" * 80)
    print("DIMENSIONAL SCORE STATISTICS")
    print("=" * 80)
    print()

    print(f"{'Dimension':<25} {'Mean':<8} {'Median':<8} {'Std':<8} {'Range':<12} {'Bins':>12}")
    print("-" * 80)

    for dim, stats in dim_stats.items():
        range_str = f"{stats['min']:.1f}-{stats['max']:.1f}"
        bins_str = f"{stats['bins_populated']}/11 ({stats['bins_populated_pct']:.0f}%)"

        print(f"{dim:<25} {stats['mean']:<8.2f} {stats['median']:<8.1f} "
              f"{stats['std']:<8.2f} {range_str:<12} {bins_str:>12}")

    print()

    # Average stats
    avg_mean = sum(s['mean'] for s in dim_stats.values()) / len(dim_stats)
    avg_std = sum(s['std'] for s in dim_stats.values()) / len(dim_stats)
    avg_bins = sum(s['bins_populated'] for s in dim_stats.values()) / len(dim_stats)
    avg_bins_pct = sum(s['bins_populated_pct'] for s in dim_stats.values()) / len(dim_stats)
    min_range = min(s['min'] for s in dim_stats.values())
    max_range = max(s['max'] for s in dim_stats.values())

    range_avg_str = f"{min_range:.1f}-{max_range:.1f}"
    print(f"{'AVERAGE':<25} {avg_mean:<8.2f} {'N/A':<8} "
          f"{avg_std:<8.2f} {range_avg_str:<12} "
          f"{avg_bins:.1f}/11 ({avg_bins_pct:.0f}%)")

    print()

    # Tier distribution
    print("=" * 80)
    print("TIER DISTRIBUTION")
    print("=" * 80)
    print()

    total_articles = len(articles)

    for tier in ["impact", "connection", "not_uplifting"]:
        count = tier_counts[tier]
        pct = (count / total_articles * 100) if total_articles > 0 else 0
        print(f"{tier:20s}: {count:5d} ({pct:5.1f}%)")

    print()

    # Quality assessment
    print("=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    print()

    issues = []
    warnings = []

    # Check range coverage
    for dim, stats in dim_stats.items():
        if stats['max'] < 8.0:
            issues.append(f"FAIL {dim}: Max score only {stats['max']:.1f} (need >=8 for full range)")
        elif stats['max'] < 9.0:
            warnings.append(f"WARN {dim}: Max score only {stats['max']:.1f} (ideally >=9)")

        if stats['min'] > 0.5:
            issues.append(f"FAIL {dim}: Min score {stats['min']:.1f} (should have scores near 0)")

    # Check bin population
    for dim, stats in dim_stats.items():
        if stats['bins_populated'] < 7:
            issues.append(f"FAIL {dim}: Only {stats['bins_populated']}/11 bins populated (need >=7)")
        elif stats['bins_populated'] < 9:
            warnings.append(f"WARN {dim}: Only {stats['bins_populated']}/11 bins populated (ideally >=9)")

    # Check standard deviation (should show clear separation)
    for dim, stats in dim_stats.items():
        if stats['std'] < 1.0:
            issues.append(f"FAIL {dim}: Std dev only {stats['std']:.2f} (too uniform, need >1.0)")
        elif stats['std'] < 1.5:
            warnings.append(f"WARN {dim}: Std dev only {stats['std']:.2f} (ideally >=1.5)")

    # Check tier balance
    uplifting_count = tier_counts['impact'] + tier_counts['connection']
    uplifting_pct = (uplifting_count / total_articles * 100) if total_articles > 0 else 0

    if uplifting_pct < 10:
        issues.append(f"FAIL Only {uplifting_pct:.1f}% uplifting content (need >=10%)")
    elif uplifting_pct < 20:
        warnings.append(f"WARN Only {uplifting_pct:.1f}% uplifting content (ideally >=20%)")

    if uplifting_pct > 60:
        issues.append(f"FAIL Too much uplifting content ({uplifting_pct:.1f}% - need <=60%)")

    # Print issues
    if issues:
        print("CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print()

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
        print()

    if not issues and not warnings:
        print("[PASS] ALL CHECKS PASSED - Dataset quality is EXCELLENT")
        print()

    # Overall verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if issues:
        print("[FAIL] DATASET NOT READY FOR TRAINING")
        print()
        print("Critical issues found. Recommendations:")
        print("1. Score more diverse articles (wider score range)")
        print("2. Ensure oracle is using full 0-10 scale")
        print("3. Check if prefilter is too aggressive (blocking too much)")
        print()
    elif warnings:
        print("[WARN] DATASET USABLE BUT NOT OPTIMAL")
        print()
        print("Dataset will work but could be better. Consider:")
        print("1. Scoring additional articles for better coverage")
        print("2. Targeting edge cases (very high/low scores)")
        print()
    else:
        print("[PASS] DATASET READY FOR TRAINING")
        print()
        print("Quality metrics meet or exceed requirements.")
        print("Proceed with training preparation!")
        print()

    # Comparison to investment-risk
    print("=" * 80)
    print("COMPARISON TO INVESTMENT-RISK v2 (Reference)")
    print("=" * 80)
    print()

    print("Investment-risk v2 quality metrics:")
    print("  - Average mean: 1.62")
    print("  - Average std dev: 1.76")
    print("  - Average bins populated: 7.9/11 (71.9%)")
    print("  - Range: 0-8 (all dimensions)")
    print("  - Result: 0.67 MAE (EXCELLENT)")
    print()

    print(f"Uplifting quality metrics:")
    print(f"  - Average mean: {avg_mean:.2f}")
    print(f"  - Average std dev: {avg_std:.2f}")
    print(f"  - Average bins populated: {avg_bins:.1f}/11 ({avg_bins_pct:.0f}%)")
    print(f"  - Range: {min_range:.1f}-{max_range:.1f}")
    print(f"  - Expected result: {'0.6-0.8 MAE' if not issues else 'UNKNOWN (fix issues first)'}")
    print()


def main():
    # Default path
    pattern = "datasets/scored/uplifting_v1/uplifting/scored_batch_*.jsonl"

    if len(sys.argv) > 1:
        pattern = sys.argv[1]

    print(f"Analyzing scored data: {pattern}")

    # Load data
    articles = load_scored_data(pattern)
    print(f"Loaded {len(articles)} articles")

    # Analyze
    dim_stats = analyze_dimensions(articles)
    tier_counts = analyze_tiers(articles)

    # Report
    print_report(articles, dim_stats, tier_counts)


if __name__ == '__main__':
    main()
