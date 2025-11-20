"""
Generate prefilter validation report from scored training data.

This retroactively validates the prefilter by analyzing which articles
passed/blocked and their oracle scores.
"""
import json
import sys
import argparse
from pathlib import Path
from collections import Counter
import importlib.util


def load_prefilter(filter_path: Path):
    """Load prefilter module from filter directory."""
    prefilter_path = filter_path / "prefilter.py"
    spec = importlib.util.spec_from_file_location("prefilter", prefilter_path)
    prefilter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefilter)
    return prefilter


def load_scored_articles(data_dir: Path):
    """Load all scored articles from training directory."""
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_file = data_dir / "test.jsonl"

    articles = []
    for file in [train_file, val_file, test_file]:
        if file.exists():
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        articles.append(json.loads(line))

    return articles


def test_prefilter(prefilter_module, articles):
    """Test prefilter on articles and categorize results."""
    passed = []
    blocked = []
    block_reasons = Counter()

    for article in articles:
        # Test prefilter
        result = prefilter_module.should_send_to_oracle({
            'title': article['title'],
            'content': article['content']
        })

        if result['send_to_oracle']:
            passed.append(article)
        else:
            blocked.append(article)
            block_reasons[result.get('reason', 'unknown')] += 1

    return passed, blocked, block_reasons


def calculate_statistics(articles, dimension_names):
    """Calculate mean scores per dimension."""
    if not articles:
        return {}

    stats = {}
    for i, dim in enumerate(dimension_names):
        scores = [a['labels'][i] for a in articles if i < len(a['labels'])]
        if scores:
            stats[dim] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }

    return stats


def generate_report(filter_name: str, version: str, articles: list,
                   passed: list, blocked: list, block_reasons: Counter,
                   dimension_names: list):
    """Generate prefilter validation report."""

    total = len(articles)
    pass_rate = (len(passed) / total * 100) if total > 0 else 0

    # Calculate score statistics for passed vs blocked
    passed_stats = calculate_statistics(passed, dimension_names)
    blocked_stats = calculate_statistics(blocked, dimension_names)

    # Calculate overall means for passed articles
    passed_overall_means = []
    for article in passed:
        if article['labels']:
            passed_overall_means.append(sum(article['labels']) / len(article['labels']))

    avg_passed_score = sum(passed_overall_means) / len(passed_overall_means) if passed_overall_means else 0

    # Calculate overall means for blocked articles
    blocked_overall_means = []
    for article in blocked:
        if article['labels']:
            blocked_overall_means.append(sum(article['labels']) / len(article['labels']))

    avg_blocked_score = sum(blocked_overall_means) / len(blocked_overall_means) if blocked_overall_means else 0

    # False negatives: blocked articles with high scores (>6.0 average)
    false_negatives = [a for a in blocked if sum(a['labels']) / len(a['labels']) > 6.0]

    report = f"""# {filter_name} {version} - Prefilter Validation Report

**Date:** 2025-11-20
**Method:** Retroactive analysis of scored training data
**Sample Size:** {total:,} articles

## Executive Summary

**Pass Rate:** {pass_rate:.1f}% ({len(passed):,} passed / {len(blocked):,} blocked)

**Prefilter Performance:**
- ✅ Average score of passed articles: {avg_passed_score:.2f}/10
- ✅ Average score of blocked articles: {avg_blocked_score:.2f}/10
- {'✅' if len(false_negatives) == 0 else '⚠️'} False negatives (blocked with score >6.0): {len(false_negatives)} ({len(false_negatives)/len(blocked)*100:.1f}% of blocked)

**Assessment:** {'PASS - Prefilter effectively blocks low-value content' if len(false_negatives) < len(blocked) * 0.1 else 'REVIEW - High false negative rate'}

## Block Reason Distribution

"""

    for reason, count in block_reasons.most_common():
        pct = (count / len(blocked) * 100) if len(blocked) > 0 else 0
        report += f"- **{reason}**: {count:,} articles ({pct:.1f}%)\n"

    report += f"""

## Score Distribution Analysis

### Passed Articles (n={len(passed):,})

Average score by dimension:

"""

    for dim, stats in passed_stats.items():
        report += f"- **{dim}**: {stats['mean']:.2f} (range: {stats['min']:.1f}-{stats['max']:.1f})\n"

    if blocked_stats:
        report += f"""

### Blocked Articles (n={len(blocked):,})

Average score by dimension:

"""
        for dim, stats in blocked_stats.items():
            report += f"- **{dim}**: {stats['mean']:.2f} (range: {stats['min']:.1f}-{stats['max']:.1f})\n"

    if false_negatives:
        report += f"""

## False Negatives Analysis

Found {len(false_negatives)} articles blocked by prefilter but scored >6.0 by oracle:

"""
        for i, article in enumerate(false_negatives[:5], 1):
            avg_score = sum(article['labels']) / len(article['labels'])
            report += f"""
### {i}. {article['title'][:80]}

- **Average Score:** {avg_score:.2f}/10
- **Article ID:** {article['id']}
"""

    report += f"""

## Recommendations

"""

    if len(false_negatives) == 0:
        report += "✅ Prefilter is working well - no high-scoring articles blocked\n"
    elif len(false_negatives) < len(blocked) * 0.05:
        report += f"✅ Prefilter has low false negative rate ({len(false_negatives)}/{len(blocked)} = {len(false_negatives)/len(blocked)*100:.1f}%)\n"
    else:
        report += f"⚠️ Review prefilter rules - blocking {len(false_negatives)} high-quality articles ({len(false_negatives)/len(blocked)*100:.1f}% of blocked)\n"

    if pass_rate < 30:
        report += "⚠️ Low pass rate - prefilter may be too aggressive\n"
    elif pass_rate > 90:
        report += "⚠️ High pass rate - prefilter may not be filtering enough\n"
    else:
        report += f"✅ Pass rate ({pass_rate:.1f}%) is reasonable\n"

    report += """
✅ Prefilter validated retroactively using training data

---

*Generated from scored training data analysis*
"""

    return report


def main():
    parser = argparse.ArgumentParser(description='Generate prefilter validation report')
    parser.add_argument('--filter', type=str, required=True,
                       help='Path to filter directory')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--filter-name', type=str, required=True,
                       help='Filter name')
    parser.add_argument('--version', type=str, required=True,
                       help='Filter version')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')

    args = parser.parse_args()

    filter_path = Path(args.filter)
    data_dir = Path(args.data_dir)

    print(f"Loading prefilter from {filter_path}...")
    prefilter = load_prefilter(filter_path)

    print(f"Loading scored articles from {data_dir}...")
    articles = load_scored_articles(data_dir)
    print(f"Loaded {len(articles):,} articles")

    print("Testing prefilter...")
    passed, blocked, block_reasons = test_prefilter(prefilter, articles)

    print(f"Results: {len(passed)} passed, {len(blocked)} blocked")

    # Get dimension names from first article
    dimension_names = articles[0]['dimension_names'] if articles else []

    print("Generating report...")
    report = generate_report(
        args.filter_name,
        args.version,
        articles,
        passed,
        blocked,
        block_reasons,
        dimension_names
    )

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to {output_path}")


if __name__ == '__main__':
    main()
