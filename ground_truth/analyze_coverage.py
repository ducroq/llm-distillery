#!/usr/bin/env python3
"""
Analyze dimensional coverage in labeled training data.

This script helps you understand if your training data has sufficient
coverage across the full range (1-10) for each dimension.

Usage:
    python -m ground_truth.analyze_coverage \
        --labeled-file datasets/uplifting_training_1500/uplifting/labeled_articles.jsonl \
        --dimensions agency,progress,collective_benefit,connection,innovation,justice,resilience,wonder
"""

import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict
import sys


def analyze_dimensional_coverage(
    labeled_file: str,
    dimensions: List[str]
) -> Dict:
    """
    Analyze score distribution for each dimension.

    Returns dict with statistics and warnings about coverage gaps.
    """

    # Track scores for each dimension
    dimension_scores = defaultdict(list)

    # Load labeled data
    print(f"Loading labeled data from {labeled_file}...\n")

    article_count = 0
    with open(labeled_file, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            article_count += 1

            # Extract scores from analysis field
            if 'analysis' in article:
                analysis = article['analysis']
                for dim in dimensions:
                    if dim in analysis:
                        score = analysis[dim]
                        if isinstance(score, (int, float)) and 1 <= score <= 10:
                            dimension_scores[dim].append(int(score))

    print(f"Analyzed {article_count} labeled articles\n")
    print("="*70)

    # Analyze each dimension
    coverage_report = {}

    for dim in dimensions:
        scores = dimension_scores[dim]

        if not scores:
            print(f"\n⚠️  WARNING: No scores found for dimension '{dim}'")
            continue

        # Count occurrences of each score
        score_counts = Counter(scores)

        # Calculate statistics
        min_score = min(scores)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)

        # Identify gaps (scores 1-10 that have zero examples)
        all_scores = set(range(1, 11))
        present_scores = set(score_counts.keys())
        missing_scores = sorted(all_scores - present_scores)

        # Identify sparse regions (scores with < 5% of examples)
        total = len(scores)
        threshold = total * 0.05  # 5% threshold
        sparse_scores = [score for score in range(1, 11)
                        if score_counts.get(score, 0) < threshold and score in present_scores]

        # Print dimension report
        print(f"\n{dim.upper()}")
        print("-" * 70)
        print(f"Total examples: {len(scores)}")
        print(f"Range: {min_score} - {max_score}")
        print(f"Mean: {mean_score:.2f}")
        print(f"\nScore distribution:")

        for score in range(1, 11):
            count = score_counts.get(score, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_length

            status = ""
            if count == 0:
                status = "  ⚠️  MISSING"
            elif count < threshold:
                status = "  ⚠️  SPARSE"

            print(f"  {score:2d}: {bar:50s} {count:4d} ({pct:5.1f}%){status}")

        # Coverage warnings
        warnings = []

        if missing_scores:
            warnings.append(f"Missing scores: {missing_scores}")

        if max_score < 10:
            warnings.append(f"No examples of score 10 (max is {max_score})")

        if min_score > 1:
            warnings.append(f"No examples of score 1 (min is {min_score})")

        if sparse_scores:
            warnings.append(f"Sparse coverage at: {sparse_scores}")

        # Check for concentration in middle
        middle_range = range(4, 8)
        middle_count = sum(score_counts.get(s, 0) for s in middle_range)
        middle_pct = (middle_count / total * 100) if total > 0 else 0

        if middle_pct > 70:
            warnings.append(f"{middle_pct:.0f}% concentrated in middle (4-7)")

        # Print warnings
        if warnings:
            print(f"\n⚠️  Coverage Issues:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print(f"\n✅ Good coverage across full range")

        # Store for report
        coverage_report[dim] = {
            'total': len(scores),
            'min': min_score,
            'max': max_score,
            'mean': mean_score,
            'distribution': dict(score_counts),
            'missing_scores': missing_scores,
            'sparse_scores': sparse_scores,
            'warnings': warnings
        }

    return coverage_report


def generate_recommendations(coverage_report: Dict) -> List[str]:
    """Generate recommendations based on coverage analysis."""

    recommendations = []

    # Check for dimensions with poor coverage
    for dim, report in coverage_report.items():
        if report['missing_scores']:
            if 1 in report['missing_scores']:
                recommendations.append(
                    f"Find examples of very LOW {dim} (score 1-2) - articles that clearly lack this quality"
                )
            if 10 in report['missing_scores']:
                recommendations.append(
                    f"Find examples of very HIGH {dim} (score 9-10) - exceptional articles for this quality"
                )

        # Check for middle concentration
        middle_count = sum(report['distribution'].get(s, 0) for s in range(4, 8))
        total = report['total']
        if middle_count / total > 0.70:
            recommendations.append(
                f"Increase edge case coverage for {dim} - currently too concentrated in middle range"
            )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dimensional coverage in labeled training data'
    )
    parser.add_argument(
        '--labeled-file',
        required=True,
        help='Path to labeled_articles.jsonl'
    )
    parser.add_argument(
        '--dimensions',
        default='agency,progress,collective_benefit,connection,innovation,justice,resilience,wonder',
        help='Comma-separated list of dimensions to analyze'
    )
    parser.add_argument(
        '--output',
        help='Optional: save report as JSON'
    )

    args = parser.parse_args()

    # Parse dimensions
    dimensions = [d.strip() for d in args.dimensions.split(',')]

    # Analyze coverage
    coverage_report = analyze_dimensional_coverage(
        args.labeled_file,
        dimensions
    )

    # Generate recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = generate_recommendations(coverage_report)

    if not recommendations:
        print("\n✅ Coverage looks good! No major gaps detected.")
    else:
        print("\nTo improve training data quality:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(coverage_report, f, indent=2)
        print(f"\n📁 Detailed report saved to: {args.output}")

    print()


if __name__ == '__main__':
    main()
