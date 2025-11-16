"""
Generate validation report from manual review results.

Usage:
    python scripts/generate_validation_report.py \
        --review filters/uplifting/v4/validation_review.json \
        --output filters/uplifting/v4/validation_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter


def load_review_data(review_file: Path) -> Dict[str, Any]:
    """Load manual review results."""
    with open(review_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_agreement_stats(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate oracle-reviewer agreement statistics."""

    reviewed = [a for a in articles if a['manual_review']['reviewed']]

    if not reviewed:
        return {
            'total_reviewed': 0,
            'agreement_rate': 0,
            'by_category': {}
        }

    correct = sum(1 for a in reviewed if a['manual_review']['oracle_correct'])
    agreement_rate = (correct / len(reviewed)) * 100

    # By category
    by_category = {}
    for category in ['high_scorers', 'edge_cases', 'low_scorers']:
        cat_articles = [a for a in reviewed if a['category'] == category]
        if cat_articles:
            cat_correct = sum(1 for a in cat_articles if a['manual_review']['oracle_correct'])
            by_category[category] = {
                'reviewed': len(cat_articles),
                'correct': cat_correct,
                'agreement_rate': (cat_correct / len(cat_articles)) * 100
            }

    return {
        'total_reviewed': len(reviewed),
        'correct': correct,
        'incorrect': len(reviewed) - correct,
        'agreement_rate': agreement_rate,
        'by_category': by_category
    }


def generate_markdown_report(review_data: Dict[str, Any], output_file: Path):
    """Generate markdown validation report."""

    articles = review_data['articles']
    stats = calculate_agreement_stats(articles)

    report = []

    # Header
    report.append("# Uplifting v4 - Oracle Validation Report")
    report.append("")
    report.append("**Date:** 2025-11-16")
    report.append("**Oracle Model:** Gemini Flash 1.5")
    report.append("**Validation Corpus:** master_dataset_20251026_20251029.jsonl (fresh corpus)")
    report.append("**Sample Size:** 100 articles (random sample, seed=2025)")
    report.append("**Manual Review:** 30 articles (10 high, 10 edge, 10 low)")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")

    if stats['total_reviewed'] > 0:
        report.append(f"**Oracle-Reviewer Agreement: {stats['agreement_rate']:.1f}%**")
        report.append("")
        report.append(f"- Reviewed: {stats['total_reviewed']}/30 articles")
        report.append(f"- Correct: {stats['correct']}")
        report.append(f"- Incorrect: {stats['incorrect']}")
        report.append("")

        # By category
        if stats['by_category']:
            report.append("**Agreement by Category:**")
            for category, data in stats['by_category'].items():
                cat_name = category.replace('_', ' ').title()
                report.append(f"- {cat_name}: {data['agreement_rate']:.1f}% ({data['correct']}/{data['reviewed']})")
            report.append("")
    else:
        report.append("**Status:** ⚠️ Manual review not yet completed")
        report.append("")
        report.append("This report contains 30 selected articles for manual review. Please review each article and update the `manual_review` fields in `validation_review.json`, then regenerate this report.")
        report.append("")

    report.append("---")
    report.append("")

    # High Scorers
    report.append("## High Scorers (Collective Benefit ≥7 or Avg ≥7)")
    report.append("")

    high_scorers = [a for a in articles if a['category'] == 'high_scorers']
    report.append(f"**Selected:** {len(high_scorers)} articles")
    report.append("")

    for i, article in enumerate(high_scorers, 1):
        scores = article['oracle_scores']
        meta = article['metadata']

        report.append(f"### {i}. {article['title'][:80]}")
        report.append("")
        report.append(f"**ID:** `{article['id']}`")
        report.append(f"**URL:** {article['url']}")
        report.append("")
        report.append(f"**Oracle Scores:**")
        report.append(f"- Collective Benefit: {scores['collective_benefit']} (gatekeeper)")
        report.append(f"- Average Score: {meta['avg_score']:.1f}")
        report.append(f"- Agency: {scores['agency']}, Progress: {scores['progress']}, Connection: {scores['connection']}")
        report.append(f"- Innovation: {scores['innovation']}, Justice: {scores['justice']}, Resilience: {scores['resilience']}, Wonder: {scores['wonder']}")
        report.append("")
        report.append(f"**Oracle Reasoning:** {article['oracle_reasoning']}")
        report.append("")

        if article['manual_review']['reviewed']:
            correct = "✅ Correct" if article['manual_review']['oracle_correct'] else "❌ Incorrect"
            report.append(f"**Manual Review:** {correct}")
            if article['manual_review']['reviewer_notes']:
                report.append(f"**Notes:** {article['manual_review']['reviewer_notes']}")
            report.append("")

    report.append("---")
    report.append("")

    # Edge Cases
    report.append("## Edge Cases (Collective Benefit 4-6, Mixed Scores)")
    report.append("")

    edge_cases = [a for a in articles if a['category'] == 'edge_cases']
    report.append(f"**Selected:** {len(edge_cases)} articles")
    report.append("")

    for i, article in enumerate(edge_cases, 1):
        scores = article['oracle_scores']
        meta = article['metadata']

        report.append(f"### {i}. {article['title'][:80]}")
        report.append("")
        report.append(f"**ID:** `{article['id']}`")
        report.append(f"**URL:** {article['url']}")
        report.append("")
        report.append(f"**Oracle Scores:**")
        report.append(f"- Collective Benefit: {scores['collective_benefit']} (gatekeeper)")
        report.append(f"- Average Score: {meta['avg_score']:.1f}")
        report.append(f"- Score Variance: {meta['variance']:.2f} (higher = more mixed)")
        report.append(f"- Agency: {scores['agency']}, Progress: {scores['progress']}, Connection: {scores['connection']}")
        report.append(f"- Innovation: {scores['innovation']}, Justice: {scores['justice']}, Resilience: {scores['resilience']}, Wonder: {scores['wonder']}")
        report.append("")
        report.append(f"**Oracle Reasoning:** {article['oracle_reasoning']}")
        report.append("")

        if article['manual_review']['reviewed']:
            correct = "✅ Correct" if article['manual_review']['oracle_correct'] else "❌ Incorrect"
            report.append(f"**Manual Review:** {correct}")
            if article['manual_review']['reviewer_notes']:
                report.append(f"**Notes:** {article['manual_review']['reviewer_notes']}")
            report.append("")

    report.append("---")
    report.append("")

    # Low Scorers
    report.append("## Low Scorers (Collective Benefit ≤3 or Avg ≤3)")
    report.append("")

    low_scorers = [a for a in articles if a['category'] == 'low_scorers']
    report.append(f"**Selected:** {len(low_scorers)} articles")
    report.append("")

    for i, article in enumerate(low_scorers, 1):
        scores = article['oracle_scores']
        meta = article['metadata']

        report.append(f"### {i}. {article['title'][:80]}")
        report.append("")
        report.append(f"**ID:** `{article['id']}`")
        report.append(f"**URL:** {article['url']}")
        report.append("")
        report.append(f"**Oracle Scores:**")
        report.append(f"- Collective Benefit: {scores['collective_benefit']} (gatekeeper)")
        report.append(f"- Average Score: {meta['avg_score']:.1f}")
        report.append(f"- Agency: {scores['agency']}, Progress: {scores['progress']}, Connection: {scores['connection']}")
        report.append(f"- Innovation: {scores['innovation']}, Justice: {scores['justice']}, Resilience: {scores['resilience']}, Wonder: {scores['wonder']}")
        report.append("")
        report.append(f"**Oracle Reasoning:** {article['oracle_reasoning']}")
        report.append("")

        if article['manual_review']['reviewed']:
            correct = "✅ Correct" if article['manual_review']['oracle_correct'] else "❌ Incorrect"
            report.append(f"**Manual Review:** {correct}")
            if article['manual_review']['reviewer_notes']:
                report.append(f"**Notes:** {article['manual_review']['reviewer_notes']}")
            report.append("")

    report.append("---")
    report.append("")

    # Conclusion
    report.append("## Conclusion")
    report.append("")

    if stats['total_reviewed'] > 0:
        if stats['agreement_rate'] >= 90:
            verdict = "✅ EXCELLENT"
            recommendation = "Oracle quality is excellent. Proceed with confidence."
        elif stats['agreement_rate'] >= 80:
            verdict = "✅ GOOD"
            recommendation = "Oracle quality is good. Minor issues may exist but acceptable for production."
        elif stats['agreement_rate'] >= 70:
            verdict = "⚠️ ACCEPTABLE"
            recommendation = "Oracle quality is acceptable but consider reviewing prompt for improvements."
        else:
            verdict = "❌ NEEDS IMPROVEMENT"
            recommendation = "Oracle quality is below acceptable threshold. Review and revise prompt before production use."

        report.append(f"**Verdict:** {verdict}")
        report.append("")
        report.append(f"**Agreement Rate:** {stats['agreement_rate']:.1f}%")
        report.append("")
        report.append(f"**Recommendation:** {recommendation}")
        report.append("")

        # Issues found
        incorrect_articles = [a for a in articles if a['manual_review']['reviewed'] and not a['manual_review']['oracle_correct']]
        if incorrect_articles:
            report.append("**Issues Found:**")
            for article in incorrect_articles:
                report.append(f"- {article['category'].replace('_', ' ').title()}: {article['title'][:60]}... - {article['manual_review']['reviewer_notes']}")
            report.append("")

    else:
        report.append("**Next Steps:**")
        report.append("")
        report.append("1. Review the 30 selected articles in `validation_review.json`")
        report.append("2. For each article, update:")
        report.append("   - `reviewed`: true")
        report.append("   - `oracle_correct`: true/false")
        report.append("   - `reviewer_notes`: explanation if incorrect")
        report.append("3. Regenerate this report with: `python scripts/generate_validation_report.py --review validation_review.json --output validation_report.md`")
        report.append("")

    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nValidation report generated: {output_file}")

    if stats['total_reviewed'] > 0:
        print(f"\nAgreement rate: {stats['agreement_rate']:.1f}%")
        print(f"Reviewed: {stats['total_reviewed']}/30 articles")


def main():
    parser = argparse.ArgumentParser(description='Generate validation report from manual review')
    parser.add_argument('--review', type=Path, required=True, help='Review JSON file')
    parser.add_argument('--output', type=Path, required=True, help='Output markdown file')

    args = parser.parse_args()

    print(f"Loading review data from: {args.review}")
    review_data = load_review_data(args.review)

    print(f"\nGenerating validation report...")
    generate_markdown_report(review_data, args.output)


if __name__ == '__main__':
    main()
