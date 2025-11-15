#!/usr/bin/env python3
"""
Dimensional Regression QA Audit Script
Analyzes JSONL datasets for multi-dimensional regression training readiness
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import math

def load_jsonl(filepath):
    """Load and parse JSONL file, tracking parse errors"""
    articles = []
    parse_errors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append((line_num, str(e)))

    return articles, parse_errors

def check_dimension_completeness(articles, expected_dims):
    """Check if all articles have all expected dimensions"""
    missing_dims = []

    for idx, article in enumerate(articles):
        # Check for uplifting_analysis structure first
        analysis = article.get('uplifting_analysis', article.get('analysis', {}))
        dimensions = analysis.get('dimensions', {})

        missing = []
        for dim in expected_dims:
            if dim not in dimensions:
                missing.append(dim)

        if missing:
            missing_dims.append({
                'index': idx,
                'id': article.get('id', 'unknown'),
                'missing': missing
            })

    return missing_dims

def check_score_validity(articles, expected_dims):
    """Check if all scores are numeric and in 0-10 range"""
    invalid_scores = []
    dim_stats = {dim: {'min': float('inf'), 'max': float('-inf'), 'values': []} for dim in expected_dims}

    for idx, article in enumerate(articles):
        # Check for uplifting_analysis structure first
        analysis = article.get('uplifting_analysis', article.get('analysis', {}))
        dimensions = analysis.get('dimensions', {})

        for dim in expected_dims:
            if dim not in dimensions:
                continue

            score = dimensions[dim]

            # Check if numeric
            if not isinstance(score, (int, float)):
                invalid_scores.append({
                    'index': idx,
                    'id': article.get('id', 'unknown'),
                    'dimension': dim,
                    'value': score,
                    'issue': 'not numeric'
                })
                continue

            # Check range
            if score < 0 or score > 10:
                invalid_scores.append({
                    'index': idx,
                    'id': article.get('id', 'unknown'),
                    'dimension': dim,
                    'value': score,
                    'issue': 'out of range'
                })
                continue

            # Track stats
            dim_stats[dim]['min'] = min(dim_stats[dim]['min'], score)
            dim_stats[dim]['max'] = max(dim_stats[dim]['max'], score)
            dim_stats[dim]['values'].append(score)

    return invalid_scores, dim_stats

def analyze_range_coverage(dim_stats):
    """Analyze score distribution and range coverage"""
    coverage = {}

    for dim, stats in dim_stats.items():
        values = stats['values']
        if not values:
            coverage[dim] = {'error': 'no values'}
            continue

        # Create histogram for 0-1, 1-2, ..., 9-10
        buckets = [0] * 10
        for val in values:
            bucket = min(int(val), 9)  # 10.0 goes in 9-10 bucket
            buckets[bucket] += 1

        total = len(values)
        bucket_pcts = [100 * count / total for count in buckets]

        # Calculate mean and std dev
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)

        # Check coverage: how many ranges have >1% of examples
        ranges_covered = sum(1 for pct in bucket_pcts if pct >= 1.0)

        # Find gaps (ranges with 0 examples)
        gaps = [i for i, count in enumerate(buckets) if count == 0]

        # Check clustering (>60% in single bucket)
        max_pct = max(bucket_pcts)
        clustered = max_pct > 60

        coverage[dim] = {
            'mean': mean,
            'std_dev': std_dev,
            'min': stats['min'],
            'max': stats['max'],
            'buckets': buckets,
            'bucket_pcts': bucket_pcts,
            'ranges_covered': ranges_covered,
            'gaps': gaps,
            'clustered': clustered,
            'max_pct': max_pct,
            'total_values': total
        }

    return coverage

def check_data_integrity(articles):
    """Check for duplicates and invalid data"""
    ids = []
    all_zero_articles = []
    missing_fields = []

    required_fields = ['id', 'title', 'content']

    for idx, article in enumerate(articles):
        # Check required fields
        missing = [f for f in required_fields if f not in article]

        # Check for either 'analysis' or 'uplifting_analysis' field
        has_analysis = 'analysis' in article or 'uplifting_analysis' in article
        if not has_analysis:
            missing.append('analysis/uplifting_analysis')

        if missing:
            missing_fields.append({
                'index': idx,
                'id': article.get('id', 'unknown'),
                'missing': missing
            })

        # Track IDs for duplicate check
        article_id = article.get('id')
        if article_id:
            ids.append(article_id)

        # Check for all-zero dimensions
        analysis = article.get('uplifting_analysis', article.get('analysis', {}))
        dimensions = analysis.get('dimensions', {})
        if dimensions and all(v == 0 for v in dimensions.values()):
            all_zero_articles.append({
                'index': idx,
                'id': article_id
            })

    # Find duplicates
    id_counts = Counter(ids)
    duplicates = [(id, count) for id, count in id_counts.items() if count > 1]

    return {
        'duplicates': duplicates,
        'all_zero_articles': all_zero_articles,
        'missing_fields': missing_fields
    }

def check_tier_labels(articles):
    """Informational check for tier labels"""
    tier_counts = Counter()
    has_tier = 0

    for article in articles:
        analysis = article.get('uplifting_analysis', article.get('analysis', {}))
        tier = analysis.get('tier')
        if tier:
            has_tier += 1
            tier_counts[tier] += 1

    return {
        'total_with_tier': has_tier,
        'tier_distribution': dict(tier_counts)
    }

def generate_report(results, output_path):
    """Generate markdown report"""

    # Determine overall status
    critical_failures = []
    if results['missing_dims']:
        critical_failures.append('dimension_completeness')
    if results['invalid_scores']:
        critical_failures.append('score_validity')

    # Data integrity is critical if >1% of articles are affected or if there are duplicates
    total_articles = results['total_articles']
    all_zero_count = len(results['integrity']['all_zero_articles'])
    duplicate_count = len(results['integrity']['duplicates'])
    parse_error_count = len(results['parse_errors'])

    # Critical thresholds: >1% all-zero articles, any duplicates, >5% parse errors
    if duplicate_count > 0:
        critical_failures.append('data_integrity')
    elif all_zero_count > 0 and (all_zero_count / total_articles) > 0.01:
        critical_failures.append('data_integrity')
    elif parse_error_count > 0 and (parse_error_count / total_articles) > 0.05:
        critical_failures.append('data_integrity')

    # Check range coverage
    coverage_issues = []
    for dim, cov in results['coverage'].items():
        if 'error' in cov:
            coverage_issues.append(dim)
        elif cov['ranges_covered'] < 3:
            coverage_issues.append(dim)

    if coverage_issues:
        critical_failures.append('range_coverage')

    if critical_failures:
        status = "❌ BLOCKED"
        status_msg = "Critical issues prevent training"
    elif any(cov.get('std_dev', 0) < 0.5 for cov in results['coverage'].values()):
        status = "⚠️ REVIEW"
        status_msg = "Quality concerns but training possible"
    else:
        status = "✅ PASSED"
        status_msg = "Dataset ready for dimensional regression training"

    report = []
    report.append("# Dimensional Regression QA Report: Uplifting Dataset\n")
    report.append(f"**Generated**: 2025-11-12\n")
    report.append(f"**Dataset**: `C:\\local_dev\\llm-distillery\\datasets\\labeled\\uplifting\\labeled_articles.jsonl`\n")
    report.append(f"**Filter**: uplifting\n")
    report.append(f"**Expected Dimensions**: 8 (agency, progress, collective_benefit, connection, innovation, justice, resilience, wonder)\n\n")

    report.append("## Executive Summary\n\n")
    report.append(f"**{status}** - {status_msg}\n\n")
    report.append(f"- **Total articles**: {results['total_articles']}\n")
    report.append(f"- **Dimensions**: {len(results['expected_dims'])}\n")
    report.append(f"- **Parse errors**: {len(results['parse_errors'])}\n")
    report.append(f"- **Critical checks**: {4 - len(critical_failures)}/4 passed\n\n")

    if critical_failures:
        report.append(f"**Critical failures**: {', '.join(critical_failures)}\n\n")

    report.append("---\n\n")
    report.append("## Critical Checks Results\n\n")
    report.append("| Check | Status | Details |\n")
    report.append("|-------|--------|----------|\n")

    # Dimension completeness
    if results['missing_dims']:
        report.append(f"| Dimension completeness | ❌ | {len(results['missing_dims'])} articles missing dimensions |\n")
    else:
        report.append(f"| Dimension completeness | ✅ | All {len(results['expected_dims'])} dimensions present in all articles |\n")

    # Score validity
    if results['invalid_scores']:
        report.append(f"| Score validity | ❌ | {len(results['invalid_scores'])} invalid scores found |\n")
    else:
        report.append(f"| Score validity | ✅ | All scores are numeric and in 0-10 range |\n")

    # Range coverage
    if coverage_issues:
        report.append(f"| Range coverage | ❌ | {len(coverage_issues)} dimensions lack coverage |\n")
    else:
        report.append(f"| Range coverage | ✅ | Full spectrum coverage for all dimensions |\n")

    # Data integrity
    integrity = results['integrity']
    integrity_status = "✅" if 'data_integrity' not in critical_failures else "❌"

    if integrity['duplicates'] or len(integrity['all_zero_articles']) > 0 or len(results['parse_errors']) > 0:
        issues = []
        if results['parse_errors']:
            pct = 100 * len(results['parse_errors']) / total_articles
            issues.append(f"{len(results['parse_errors'])} parse errors ({pct:.2f}%)")
        if integrity['duplicates']:
            issues.append(f"{len(integrity['duplicates'])} duplicates")
        if integrity['all_zero_articles']:
            pct = 100 * len(integrity['all_zero_articles']) / total_articles
            issues.append(f"{len(integrity['all_zero_articles'])} all-zero ({pct:.3f}%)")

        report.append(f"| Data integrity | {integrity_status} | {', '.join(issues)} |\n")
    else:
        report.append(f"| Data integrity | ✅ | No parse errors, duplicates, or all-zero articles |\n")

    report.append("\n---\n\n")
    report.append("## Dimension Quality Statistics\n\n")
    report.append("| Dimension | Mean | Std Dev | Min | Max | Ranges >1% |\n")
    report.append("|-----------|------|---------|-----|-----|------------|\n")

    for dim in results['expected_dims']:
        cov = results['coverage'].get(dim, {})
        if 'error' in cov:
            report.append(f"| {dim} | N/A | N/A | N/A | N/A | ERROR |\n")
        else:
            report.append(f"| {dim} | {cov['mean']:.2f} | {cov['std_dev']:.2f} | {cov['min']:.1f} | {cov['max']:.1f} | {cov['ranges_covered']}/10 |\n")

    report.append("\n---\n\n")
    report.append("## Score Distribution Analysis\n\n")

    for dim in results['expected_dims']:
        cov = results['coverage'].get(dim, {})
        if 'error' in cov:
            continue

        report.append(f"### {dim}\n\n")
        report.append(f"**Mean**: {cov['mean']:.2f} | **Std Dev**: {cov['std_dev']:.2f} | **Range**: [{cov['min']:.1f}, {cov['max']:.1f}]\n\n")

        # Histogram
        report.append("```\n")
        for i, (count, pct) in enumerate(zip(cov['buckets'], cov['bucket_pcts'])):
            bar = '█' * int(pct / 2)  # Scale to 50 chars max
            report.append(f"{i}-{i+1}: {count:5d} ({pct:5.1f}%) {bar}\n")
        report.append("```\n\n")

        # Flags
        if cov['gaps']:
            gap_ranges = [f"{g}-{g+1}" for g in cov['gaps']]
            report.append(f"⚠️ **Gaps**: No examples in ranges {', '.join(gap_ranges)}\n\n")
        if cov['clustered']:
            report.append(f"⚠️ **Clustering**: {cov['max_pct']:.1f}% of scores in single bucket\n\n")

    report.append("---\n\n")
    report.append("## Quality Observations\n\n")

    # Variance analysis
    healthy_variance = [dim for dim, cov in results['coverage'].items() if cov.get('std_dev', 0) > 1.5]
    low_variance = [dim for dim, cov in results['coverage'].items() if cov.get('std_dev', 0) < 1.0]
    clustered = [dim for dim, cov in results['coverage'].items() if cov.get('clustered', False)]

    if healthy_variance:
        report.append(f"✅ **Healthy variance** (std dev > 1.5): {', '.join(healthy_variance)}\n\n")
    if low_variance:
        report.append(f"⚠️ **Low variance** (std dev < 1.0): {', '.join(low_variance)}\n\n")
    if clustered:
        report.append(f"⚠️ **Clustering** (>60% in single range): {', '.join(clustered)}\n\n")

    report.append("---\n\n")
    report.append("## Informational Metadata\n\n")

    tier_info = results['tier_info']
    if tier_info['total_with_tier'] > 0:
        report.append(f"**Tier labels**: {tier_info['total_with_tier']}/{results['total_articles']} articles have tier labels\n\n")
        report.append("Distribution:\n")
        for tier, count in sorted(tier_info['tier_distribution'].items()):
            report.append(f"- {tier}: {count}\n")
        report.append("\nℹ️ *Tier labels are convenience metadata only. Tier-score alignment is not validated as it does not affect training quality.*\n\n")
    else:
        report.append("**Tier labels**: Not present in dataset\n\n")

    report.append("---\n\n")
    report.append("## Recommendations\n\n")

    if status == "✅ PASSED":
        report.append("✅ **Ready for training**: All critical checks passed. Dataset is suitable for dimensional regression training.\n\n")

        # Add minor issues if any
        minor_issues = []
        if all_zero_count > 0 and all_zero_count < 10:
            minor_issues.append(f"- {all_zero_count} all-zero articles ({100*all_zero_count/total_articles:.3f}%) - negligible impact, can train as-is")
        if low_variance:
            minor_issues.append(f"- Low variance in: {', '.join(low_variance)} - monitor during training")
        if clustered:
            minor_issues.append(f"- High clustering in: {', '.join(clustered)} - natural for some dimensions")

        if minor_issues:
            report.append("**Minor observations** (non-blocking):\n")
            for issue in minor_issues:
                report.append(f"{issue}\n")
            report.append("\n")

    elif status == "⚠️ REVIEW":
        report.append("⚠️ **Review before training**: Quality concerns detected but training is possible.\n\n")
        report.append("**Issues to consider**:\n")
        if low_variance:
            report.append(f"- Low variance in: {', '.join(low_variance)} - model may struggle to learn gradients\n")
        if clustered:
            report.append(f"- High clustering in: {', '.join(clustered)} - may indicate labeling bias\n")
        if all_zero_count > 0:
            report.append(f"- {all_zero_count} all-zero articles ({100*all_zero_count/total_articles:.3f}%) - consider re-labeling\n")
        report.append("\n")
    else:
        report.append("❌ **Do not train**: Critical issues must be resolved before training.\n\n")
        report.append("**Required fixes**:\n")
        if results['missing_dims']:
            report.append(f"- Fix {len(results['missing_dims'])} articles with missing dimensions\n")
        if results['invalid_scores']:
            report.append(f"- Fix {len(results['invalid_scores'])} invalid scores\n")
        if coverage_issues:
            report.append(f"- Add examples to fill gaps in: {', '.join(coverage_issues)}\n")
        if integrity['duplicates']:
            report.append(f"- Remove {len(integrity['duplicates'])} duplicate articles\n")
        if all_zero_count > 0 and (all_zero_count / total_articles) > 0.01:
            report.append(f"- Re-label {all_zero_count} all-zero articles ({100*all_zero_count/total_articles:.2f}%)\n")
        report.append("\n")

    report.append("---\n\n")
    report.append("**Key Principle**: Can the model learn the 0-10 gradient for each dimension?\n\n")

    if status == "✅ PASSED":
        report.append("**Answer**: Yes - all dimensions have sufficient variance, range coverage, and valid scores for gradient learning.\n")
    elif status == "⚠️ REVIEW":
        report.append("**Answer**: Mostly - some dimensions may have limited learning signal due to variance or coverage issues.\n")
    else:
        report.append("**Answer**: No - critical data quality issues prevent effective gradient learning.\n")

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))

    return status, ''.join(report)

def main():
    dataset_path = Path(r"C:\local_dev\llm-distillery\datasets\labeled\uplifting\labeled_articles.jsonl")
    output_path = Path(r"C:\local_dev\llm-distillery\reports\uplifting_dimensional_regression_qa.md")

    expected_dims = [
        'agency', 'progress', 'collective_benefit', 'connection',
        'innovation', 'justice', 'resilience', 'wonder'
    ]

    print("Loading dataset...")
    articles, parse_errors = load_jsonl(dataset_path)
    print(f"Loaded {len(articles)} articles ({len(parse_errors)} parse errors)")

    print("Checking dimension completeness...")
    missing_dims = check_dimension_completeness(articles, expected_dims)

    print("Checking score validity...")
    invalid_scores, dim_stats = check_score_validity(articles, expected_dims)

    print("Analyzing range coverage...")
    coverage = analyze_range_coverage(dim_stats)

    print("Checking data integrity...")
    integrity = check_data_integrity(articles)

    print("Checking tier labels (informational)...")
    tier_info = check_tier_labels(articles)

    results = {
        'total_articles': len(articles),
        'parse_errors': parse_errors,
        'expected_dims': expected_dims,
        'missing_dims': missing_dims,
        'invalid_scores': invalid_scores,
        'dim_stats': dim_stats,
        'coverage': coverage,
        'integrity': integrity,
        'tier_info': tier_info
    }

    print(f"\nGenerating report to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status, report_text = generate_report(results, output_path)

    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE: {status}")
    print(f"{'='*60}\n")

    # Print summary
    print(f"Total articles: {len(articles)}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Missing dimensions: {len(missing_dims)}")
    print(f"Invalid scores: {len(invalid_scores)}")
    print(f"Duplicates: {len(integrity['duplicates'])}")
    print(f"All-zero articles: {len(integrity['all_zero_articles'])}")
    print(f"\nReport saved to: {output_path}")

if __name__ == '__main__':
    main()
