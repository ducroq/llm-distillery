"""
Verify data quality for sustainability_tech_deployment scored articles.

Checks for:
1. Parse errors (malformed JSON)
2. Missing dimensions
3. All-zeros or suspicious patterns
4. Dimension score distributions
5. Score ranges (0-10)
6. Examples of high/low scorers
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def verify_sustainability_tech_data(scored_dir: Path):
    """Verify scored data quality."""

    print("=" * 70)
    print("Sustainability Tech Deployment - Data Quality Verification")
    print("=" * 70)
    print()

    # Expected dimensions
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

    # Stats collectors
    total_articles = 0
    parse_errors = 0
    missing_analysis = 0
    missing_dimensions = defaultdict(int)
    all_zeros = 0
    out_of_range = defaultdict(int)

    dimension_scores = defaultdict(list)
    primary_tech_counts = Counter()
    deployment_stage_counts = Counter()
    confidence_counts = Counter()

    examples_high = []
    examples_low = []
    examples_zeros = []

    # Process all batch files
    batch_files = sorted(scored_dir.glob("scored_batch_*.jsonl"))

    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_articles += 1

                try:
                    article = json.loads(line)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    print(f"Parse error in {batch_file.name}:{line_num}: {e}")
                    continue

                # Check for analysis
                analysis = article.get('sustainability_tech_deployment_analysis')
                if not analysis:
                    missing_analysis += 1
                    continue

                # Check dimensions
                scores = []
                has_missing = False
                for dim in dimensions:
                    dim_data = analysis.get(dim)
                    if not dim_data or 'score' not in dim_data:
                        missing_dimensions[dim] += 1
                        has_missing = True
                        continue

                    score = dim_data['score']
                    scores.append(score)
                    dimension_scores[dim].append(score)

                    # Check range (0-10)
                    if score < 0 or score > 10:
                        out_of_range[dim] += 1

                # Check for all zeros
                if scores and all(s == 0 for s in scores):
                    all_zeros += 1
                    if len(examples_zeros) < 3:
                        examples_zeros.append({
                            'id': article.get('id'),
                            'title': article.get('title', '')[:80],
                            'scores': scores
                        })

                # Collect categorical data
                primary_tech_counts[analysis.get('primary_technology', 'unknown')] += 1
                deployment_stage_counts[analysis.get('deployment_stage', 'unknown')] += 1
                confidence_counts[analysis.get('confidence', 'unknown')] += 1

                # Collect examples
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score >= 7 and len(examples_high) < 5:
                        examples_high.append({
                            'id': article.get('id'),
                            'title': article.get('title', '')[:80],
                            'avg_score': avg_score,
                            'scores': {dim: analysis[dim]['score'] for dim in dimensions if dim in analysis}
                        })
                    elif avg_score <= 3 and len(examples_low) < 5:
                        examples_low.append({
                            'id': article.get('id'),
                            'title': article.get('title', '')[:80],
                            'avg_score': avg_score,
                            'scores': {dim: analysis[dim]['score'] for dim in dimensions if dim in analysis}
                        })

    # Print results
    print(f"Total Articles: {total_articles:,}")
    print()

    # Errors
    print("ERRORS AND WARNINGS:")
    print("-" * 70)
    print(f"Parse errors: {parse_errors}")
    print(f"Missing analysis: {missing_analysis}")
    print(f"All-zeros articles: {all_zeros}")

    if missing_dimensions:
        print(f"\nMissing dimensions:")
        for dim, count in sorted(missing_dimensions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dim}: {count}")

    if out_of_range:
        print(f"\nOut-of-range scores (not 0-10):")
        for dim, count in out_of_range.items():
            print(f"  {dim}: {count}")

    print()

    # Dimension distributions
    print("DIMENSION SCORE DISTRIBUTIONS:")
    print("-" * 70)
    print(f"{'Dimension':<30} {'Mean':<8} {'Median':<8} {'Min':<6} {'Max':<6} {'StdDev':<8}")
    print("-" * 70)

    for dim in dimensions:
        if dim in dimension_scores and dimension_scores[dim]:
            scores = dimension_scores[dim]
            mean_score = statistics.mean(scores)
            median_score = statistics.median(scores)
            min_score = min(scores)
            max_score = max(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0

            print(f"{dim:<30} {mean_score:<8.2f} {median_score:<8.1f} {min_score:<6} {max_score:<6} {stdev:<8.2f}")

    print()

    # Categorical distributions
    print("CATEGORICAL DISTRIBUTIONS:")
    print("-" * 70)

    print("\nPrimary Technology:")
    for tech, count in primary_tech_counts.most_common(10):
        pct = (count / total_articles) * 100
        print(f"  {tech:<30} {count:>6} ({pct:>5.1f}%)")

    print("\nDeployment Stage:")
    for stage, count in deployment_stage_counts.most_common():
        pct = (count / total_articles) * 100
        print(f"  {stage:<30} {count:>6} ({pct:>5.1f}%)")

    print("\nConfidence:")
    for conf, count in confidence_counts.most_common():
        pct = (count / total_articles) * 100
        print(f"  {conf:<30} {count:>6} ({pct:>5.1f}%)")

    print()

    # Examples
    if examples_high:
        print("HIGH SCORERS (avg >= 7):")
        print("-" * 70)
        for ex in examples_high[:3]:
            print(f"\nID: {ex['id']}")
            print(f"Title: {ex['title']}")
            print(f"Avg Score: {ex['avg_score']:.1f}")
            print(f"Scores: {ex['scores']}")

    print()

    if examples_low:
        print("LOW SCORERS (avg <= 3):")
        print("-" * 70)
        for ex in examples_low[:3]:
            print(f"\nID: {ex['id']}")
            print(f"Title: {ex['title']}")
            print(f"Avg Score: {ex['avg_score']:.1f}")
            print(f"Scores: {ex['scores']}")

    print()

    if examples_zeros:
        print("ALL-ZEROS EXAMPLES:")
        print("-" * 70)
        for ex in examples_zeros:
            print(f"\nID: {ex['id']}")
            print(f"Title: {ex['title']}")
            print(f"Scores: {ex['scores']}")

    print()

    # Final verdict
    print("=" * 70)
    print("VERDICT:")
    print("=" * 70)

    issues = []
    if parse_errors > 0:
        issues.append(f"{parse_errors} parse errors")
    if missing_analysis > total_articles * 0.01:  # >1%
        issues.append(f"{missing_analysis} missing analysis ({missing_analysis/total_articles*100:.1f}%)")
    if all_zeros > total_articles * 0.05:  # >5%
        issues.append(f"{all_zeros} all-zeros articles ({all_zeros/total_articles*100:.1f}%)")
    if any(count > 0 for count in out_of_range.values()):
        issues.append(f"Out-of-range scores detected")

    if issues:
        print("WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("OK - No significant issues detected")

    print()
    print(f"Scored articles ready for training: {total_articles - parse_errors - missing_analysis:,}")
    print("=" * 70)


if __name__ == '__main__':
    scored_dir = Path("datasets/scored/sustainability_tech_deployment_v1/sustainability_tech_deployment")

    if not scored_dir.exists():
        print(f"Error: Directory not found: {scored_dir}")
        sys.exit(1)

    verify_sustainability_tech_data(scored_dir)
