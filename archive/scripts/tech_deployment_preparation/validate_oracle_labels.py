"""
Validate oracle label quality by analyzing a random sample.

This script:
1. Samples N random labeled articles
2. Analyzes label statistics (tier distribution, score ranges, etc.)
3. Performs quality checks (missing dimensions, outliers, reasoning quality)
4. Generates a validation report in markdown
"""

import json
import random
import statistics
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import argparse


def load_all_labels(labeled_dir: Path) -> List[Dict[str, Any]]:
    """Load all labeled articles from batch files."""
    all_labels = []

    for batch_file in sorted(labeled_dir.glob('labeled_batch_*.jsonl')):
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_labels.append(json.loads(line))

    return all_labels


def analyze_sample(sample: List[Dict[str, Any]], filter_name: str) -> Dict[str, Any]:
    """Analyze label quality metrics from sample."""

    analysis_key = f'{filter_name}_analysis'

    # Collect statistics
    tier_counts = Counter()
    content_type_counts = Counter()
    overall_scores = []
    dimension_scores = defaultdict(list)
    gatekeeper_violations = []
    missing_reasoning = []
    missing_dimensions = []
    score_dimension_mismatches = []

    for article in sample:
        analysis = article.get(analysis_key, {})
        article_id = article.get('id', 'unknown')

        # Basic fields
        tier = analysis.get('tier', 'unknown')
        tier_counts[tier] += 1

        content_type = analysis.get('content_type', 'unknown')
        content_type_counts[content_type] += 1

        overall_score = analysis.get('overall_score', 0)
        overall_scores.append(overall_score)

        reasoning = analysis.get('reasoning', '')
        if not reasoning or len(reasoning) < 50:
            missing_reasoning.append(article_id)

        # Dimension analysis
        dimensions = analysis.get('dimensions', {})
        if not dimensions:
            missing_dimensions.append(article_id)

        for dim_name, dim_data in dimensions.items():
            if isinstance(dim_data, dict):
                score = dim_data.get('score', 0)
                dim_reasoning = dim_data.get('reasoning', '')

                dimension_scores[dim_name].append(score)

                if not dim_reasoning or len(dim_reasoning) < 20:
                    # Dimension missing reasoning
                    pass
            else:
                dimension_scores[dim_name].append(dim_data)

        # Gatekeeper check (tech deployment: technical_credibility < 4 should cap at 3.9)
        tech_cred = dimensions.get('technical_credibility', {})
        if isinstance(tech_cred, dict):
            tech_cred_score = tech_cred.get('score', 0)
        else:
            tech_cred_score = tech_cred

        if tech_cred_score < 4.0 and overall_score >= 4.0:
            gatekeeper_violations.append({
                'id': article_id,
                'tech_credibility': tech_cred_score,
                'overall_score': overall_score,
                'tier': tier
            })

    # Calculate statistics
    results = {
        'sample_size': len(sample),
        'tier_distribution': dict(tier_counts),
        'content_type_distribution': dict(content_type_counts),
        'overall_score_stats': {
            'mean': statistics.mean(overall_scores) if overall_scores else 0,
            'median': statistics.median(overall_scores) if overall_scores else 0,
            'stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'min': min(overall_scores) if overall_scores else 0,
            'max': max(overall_scores) if overall_scores else 0,
        },
        'dimension_stats': {},
        'quality_issues': {
            'gatekeeper_violations': gatekeeper_violations,
            'missing_reasoning': missing_reasoning,
            'missing_dimensions': missing_dimensions,
        }
    }

    # Dimension statistics
    for dim_name, scores in dimension_scores.items():
        if scores:
            results['dimension_stats'][dim_name] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
            }

    return results


def generate_report(results: Dict[str, Any], output_path: Path, filter_name: str):
    """Generate markdown validation report."""

    report = []
    report.append(f"# Oracle Label Quality Validation Report")
    report.append(f"")
    report.append(f"**Filter**: {filter_name}")
    report.append(f"**Date**: {Path.cwd().name}")
    report.append(f"**Sample Size**: {results['sample_size']} articles")
    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Tier Distribution
    report.append(f"## Tier Distribution")
    report.append(f"")
    tier_dist = results['tier_distribution']
    total = sum(tier_dist.values())

    for tier in ['deployed_proven', 'early_commercial', 'pilot_demonstration', 'vaporware']:
        count = tier_dist.get(tier, 0)
        pct = (count / total * 100) if total > 0 else 0
        report.append(f"- **{tier}**: {count} ({pct:.1f}%)")

    report.append(f"")

    # Overall Score Statistics
    report.append(f"## Overall Score Statistics")
    report.append(f"")
    stats = results['overall_score_stats']
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Mean   | {stats['mean']:.2f} |")
    report.append(f"| Median | {stats['median']:.2f} |")
    report.append(f"| Stdev  | {stats['stdev']:.2f} |")
    report.append(f"| Min    | {stats['min']:.2f} |")
    report.append(f"| Max    | {stats['max']:.2f} |")
    report.append(f"")

    # Dimension Statistics
    report.append(f"## Dimension Score Averages")
    report.append(f"")
    report.append(f"| Dimension | Mean | Median | Stdev | Min | Max |")
    report.append(f"|-----------|------|--------|-------|-----|-----|")

    for dim_name in sorted(results['dimension_stats'].keys()):
        dim_stats = results['dimension_stats'][dim_name]
        report.append(
            f"| {dim_name} | {dim_stats['mean']:.2f} | {dim_stats['median']:.2f} | "
            f"{dim_stats['stdev']:.2f} | {dim_stats['min']:.1f} | {dim_stats['max']:.1f} |"
        )

    report.append(f"")

    # Content Type Distribution
    report.append(f"## Content Type Distribution")
    report.append(f"")
    ct_dist = results['content_type_distribution']
    total_ct = sum(ct_dist.values())

    for ctype, count in sorted(ct_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_ct * 100) if total_ct > 0 else 0
        report.append(f"- **{ctype}**: {count} ({pct:.1f}%)")

    report.append(f"")

    # Quality Issues
    report.append(f"## Quality Issues")
    report.append(f"")

    issues = results['quality_issues']

    # Gatekeeper violations
    gk_violations = issues['gatekeeper_violations']
    if gk_violations:
        report.append(f"### ⚠️ Gatekeeper Rule Violations: {len(gk_violations)}")
        report.append(f"")
        report.append(f"Articles with `technical_credibility < 4.0` but `overall_score >= 4.0`:")
        report.append(f"")
        for v in gk_violations[:10]:  # Show first 10
            report.append(
                f"- `{v['id']}`: tech_cred={v['technical_credibility']:.1f}, "
                f"overall={v['overall_score']:.1f}, tier={v['tier']}"
            )
        if len(gk_violations) > 10:
            report.append(f"- ... and {len(gk_violations) - 10} more")
        report.append(f"")
    else:
        report.append(f"### ✅ Gatekeeper Rule Violations: 0")
        report.append(f"")

    # Missing reasoning
    missing_reasoning = issues['missing_reasoning']
    if missing_reasoning:
        report.append(f"### ⚠️ Missing/Short Reasoning: {len(missing_reasoning)}")
        report.append(f"")
        report.append(f"Articles with reasoning < 50 characters:")
        for aid in missing_reasoning[:10]:
            report.append(f"- `{aid}`")
        if len(missing_reasoning) > 10:
            report.append(f"- ... and {len(missing_reasoning) - 10} more")
        report.append(f"")
    else:
        report.append(f"### ✅ Missing/Short Reasoning: 0")
        report.append(f"")

    # Missing dimensions
    missing_dims = issues['missing_dimensions']
    if missing_dims:
        report.append(f"### ⚠️ Missing Dimensions: {len(missing_dims)}")
        report.append(f"")
        for aid in missing_dims[:10]:
            report.append(f"- `{aid}`")
        if len(missing_dims) > 10:
            report.append(f"- ... and {len(missing_dims) - 10} more")
        report.append(f"")
    else:
        report.append(f"### ✅ Missing Dimensions: 0")
        report.append(f"")

    # Summary
    report.append(f"---")
    report.append(f"")
    report.append(f"## Summary")
    report.append(f"")

    total_issues = len(gk_violations) + len(missing_reasoning) + len(missing_dims)
    issue_rate = (total_issues / results['sample_size']) * 100

    if issue_rate < 5:
        report.append(f"✅ **PASS**: Label quality is excellent ({issue_rate:.1f}% issue rate)")
    elif issue_rate < 15:
        report.append(f"⚠️ **WARNING**: Label quality is acceptable but has issues ({issue_rate:.1f}% issue rate)")
    else:
        report.append(f"❌ **FAIL**: Label quality needs improvement ({issue_rate:.1f}% issue rate)")

    report.append(f"")
    report.append(f"**Recommendation**: ")
    if issue_rate < 5:
        report.append(f"Proceed to model training. Labels are high quality.")
    elif issue_rate < 15:
        report.append(f"Review flagged issues and consider re-labeling problematic articles.")
    else:
        report.append(f"Re-run oracle labeling with improved prompt or different model.")

    report.append(f"")

    # Write report
    output_path.write_text('\n'.join(report), encoding='utf-8')
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate oracle label quality')
    parser.add_argument('--labeled-dir', required=True, help='Directory containing labeled_batch_*.jsonl files')
    parser.add_argument('--filter-name', required=True, help='Filter name (e.g., sustainability_tech_deployment)')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of articles to sample')
    parser.add_argument('--output', required=True, help='Output report path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    output_path = Path(args.output)

    # Load all labels
    print(f"Loading labels from: {labeled_dir}")
    all_labels = load_all_labels(labeled_dir)
    print(f"Total labeled articles: {len(all_labels)}")

    # Sample
    random.seed(args.seed)
    sample_size = min(args.sample_size, len(all_labels))
    sample = random.sample(all_labels, sample_size)
    print(f"Sampled {sample_size} articles")

    # Analyze
    print("Analyzing sample...")
    results = analyze_sample(sample, args.filter_name)

    # Generate report
    print("Generating report...")
    generate_report(results, output_path, args.filter_name)

    print("Done!")


if __name__ == '__main__':
    main()
