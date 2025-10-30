#!/usr/bin/env python3
"""
Generic model calibration: Compare Claude vs Gemini on ANY semantic filter prompt.

This script labels a small sample (default 100 articles) with both Claude and Gemini,
then generates a detailed comparison report to help you decide:
- Which LLM to use for large-scale labeling (cost vs quality tradeoff)
- Whether tier distributions match expectations
- Whether the prompt framework rules are being followed correctly

Usage:
    python -m ground_truth.calibrate_models \
        --prompt prompts/sustainability.md \
        --source ../content-aggregator/data/collected/articles.jsonl \
        --sample-size 100 \
        --output reports/sustainability_calibration.md
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from .batch_labeler import GenericBatchLabeler


def select_calibration_sample(
    source_file: Path,
    sample_size: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Select random articles for calibration."""
    random.seed(seed)

    articles = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line.strip())
            articles.append(article)

    if len(articles) > sample_size:
        articles = random.sample(articles, sample_size)

    print(f"Selected {len(articles)} articles for calibration")
    return articles


def label_with_provider(
    articles: List[Dict],
    prompt_path: str,
    provider: str,
    filter_name: str,
    cache_dir: Path = None
) -> List[Dict]:
    """Label articles with specified LLM provider."""
    print(f"\n{'='*70}")
    print(f"Labeling {len(articles)} articles with {provider.upper()}")
    print(f"{'='*70}\n")

    labeler = GenericBatchLabeler(
        prompt_path=prompt_path,
        llm_provider=provider,
        filter_name=filter_name
    )

    labeled_articles = []

    for i, article in enumerate(articles, 1):
        print(f"  [{i}/{len(articles)}] Analyzing with {provider}...", end=' ')

        try:
            analysis = labeler.analyze_article(article)

            if analysis:
                # Create a copy to avoid modifying original
                article_copy = article.copy()
                article_copy[f'{filter_name}_analysis'] = analysis
                labeled_articles.append(article_copy)

                # Get tier/score (they may have different keys depending on prompt)
                tier = analysis.get('tier', 'unknown')
                score_key = f'{filter_name}_score'
                score = analysis.get(score_key, analysis.get('overall_uplift_score', 0))

                print(f"OK ({tier}, score: {score:.2f})")
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nSuccessfully labeled: {len(labeled_articles)}/{len(articles)} articles")

    # Cache results if cache_dir provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{provider}_labels.jsonl"

        with open(cache_file, 'w', encoding='utf-8') as f:
            for article in labeled_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"Cached {len(labeled_articles)} labeled articles to: {cache_file}")

    return labeled_articles


def extract_tier_and_score(analysis: Dict, filter_name: str) -> Tuple[str, float]:
    """Extract tier and score from analysis (generic for any filter)."""
    tier = analysis.get('tier', 'unknown')

    # Try multiple score keys in order of preference
    score_keys = [
        'overall_uplift_score',  # Uplifting filter
        'sustainability_score',  # Sustainability filter
        f'{filter_name}_score',  # Generic pattern
        'overall_score',         # Fallback
    ]

    score = 0.0
    for key in score_keys:
        if key in analysis and analysis[key] is not None:
            score = float(analysis[key])
            break

    return tier, score


def analyze_batch(articles: List[Dict], provider_name: str, filter_name: str) -> Dict:
    """Analyze labeled batch and extract statistics."""
    if not articles:
        return None

    # Extract analyses
    analyses = [a[f'{filter_name}_analysis'] for a in articles]

    # Extract tiers and scores
    tiers = []
    scores = []
    content_types = []

    for analysis in analyses:
        tier, score = extract_tier_and_score(analysis, filter_name)
        tiers.append(tier)
        scores.append(score)

        # Content type if available
        content_type = analysis.get('content_type', 'unknown')
        content_types.append(content_type)

    # Calculate stats
    tier_counts = Counter(tiers)
    tier_pcts = {k: v/len(articles)*100 for k, v in tier_counts.items()}

    stats = {
        'provider': provider_name,
        'total': len(articles),
        'tier_counts': tier_counts,
        'tier_pcts': tier_pcts,
        'avg_score': sum(scores)/len(scores) if scores else 0,
        'median_score': sorted(scores)[len(scores)//2] if scores else 0,
        'score_range': (min(scores), max(scores)) if scores else (0, 0),
        'content_type_counts': Counter(content_types)
    }

    return stats


def print_stats_comparison(stats1: Dict, stats2: Dict):
    """Print comparison between two LLM providers."""
    print(f"\n{'='*70}")
    print(f"CALIBRATION COMPARISON: {stats1['provider']} vs {stats2['provider']}")
    print(f"{'='*70}\n")

    # Tier distribution comparison
    print("TIER DISTRIBUTION:")
    print(f"  {'Tier':<20} {stats1['provider']:>15} {stats2['provider']:>15} {'Difference':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    all_tiers = set(stats1['tier_pcts'].keys()) | set(stats2['tier_pcts'].keys())
    for tier in sorted(all_tiers):
        pct1 = stats1['tier_pcts'].get(tier, 0)
        pct2 = stats2['tier_pcts'].get(tier, 0)
        diff = pct2 - pct1
        print(f"  {tier:<20} {pct1:14.1f}% {pct2:14.1f}% {diff:+14.1f}%")

    # Score comparison
    print(f"\nSCORE STATISTICS:")
    print(f"  Average:  {stats1['avg_score']:.2f} vs {stats2['avg_score']:.2f} (diff: {stats2['avg_score']-stats1['avg_score']:+.2f})")
    print(f"  Median:   {stats1['median_score']:.2f} vs {stats2['median_score']:.2f} (diff: {stats2['median_score']-stats1['median_score']:+.2f})")
    print(f"  Range:    {stats1['score_range'][0]:.2f}-{stats1['score_range'][1]:.2f} vs {stats2['score_range'][0]:.2f}-{stats2['score_range'][1]:.2f}")

    # Cost analysis
    print(f"\nCOST COMPARISON (for 5,000 articles):")
    claude_cost = 5000 * 0.009
    gemini_cost = 5000 * 0.00018
    print(f"  Claude:  ${claude_cost:.2f}")
    print(f"  Gemini:  ${gemini_cost:.2f} (50x cheaper)")
    print(f"  Savings: ${claude_cost - gemini_cost:.2f} by using Gemini")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    tier_diff = sum(abs(stats1['tier_pcts'].get(t, 0) - stats2['tier_pcts'].get(t, 0)) for t in all_tiers)
    score_diff = abs(stats1['avg_score'] - stats2['avg_score'])

    if tier_diff < 10 and score_diff < 0.5:
        print(f"  Distributions are very similar (tier diff: {tier_diff:.1f}%, score diff: {score_diff:.2f})")
        print(f"  RECOMMENDED: Use Gemini for large-scale labeling (50x cheaper, similar quality)")
    elif tier_diff < 20 and score_diff < 1.0:
        print(f"  Distributions are somewhat similar (tier diff: {tier_diff:.1f}%, score diff: {score_diff:.2f})")
        print(f"  RECOMMENDED: Use Gemini but spot-check 100 articles with Claude to validate")
    else:
        print(f"  Distributions differ significantly (tier diff: {tier_diff:.1f}%, score diff: {score_diff:.2f})")
        print(f"  RECOMMENDED: Use Claude for ground truth, or refine your prompt for better Gemini consistency")


def generate_calibration_report(
    claude_articles: List[Dict],
    gemini_articles: List[Dict],
    claude_stats: Dict,
    gemini_stats: Dict,
    filter_name: str,
    prompt_path: str,
    output_file: Path
):
    """Generate markdown calibration report."""
    lines = []

    # Header
    lines.append(f"# Model Calibration Report: {filter_name.replace('_', ' ').title()}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Prompt**: `{prompt_path}`")
    lines.append(f"**Sample Size**: {claude_stats['total']} articles (labeled by both models)")
    lines.append(f"**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    all_tiers = set(claude_stats['tier_pcts'].keys()) | set(gemini_stats['tier_pcts'].keys())
    tier_diff = sum(abs(claude_stats['tier_pcts'].get(t, 0) - gemini_stats['tier_pcts'].get(t, 0)) for t in all_tiers)
    score_diff = abs(claude_stats['avg_score'] - gemini_stats['avg_score'])

    lines.append(f"- **Tier Distribution Difference**: {tier_diff:.1f}%")
    lines.append(f"- **Average Score Difference**: {score_diff:.2f}")
    lines.append(f"- **Claude Average Score**: {claude_stats['avg_score']:.2f}")
    lines.append(f"- **Gemini Average Score**: {gemini_stats['avg_score']:.2f}")
    lines.append("")

    if tier_diff < 10 and score_diff < 0.5:
        lines.append("**Recommendation**: Models show very similar results. **Use Gemini** for large-scale labeling (50x cheaper).")
    elif tier_diff < 20 and score_diff < 1.0:
        lines.append("**Recommendation**: Models show acceptable similarity. **Use Gemini** but validate with Claude spot-checks.")
    else:
        lines.append("**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Tier Distribution Comparison
    lines.append("## Tier Distribution Comparison")
    lines.append("")
    lines.append("| Tier | Claude | Gemini | Difference |")
    lines.append("|------|--------|--------|------------|")

    for tier in sorted(all_tiers):
        claude_pct = claude_stats['tier_pcts'].get(tier, 0)
        gemini_pct = gemini_stats['tier_pcts'].get(tier, 0)
        diff = gemini_pct - claude_pct
        lines.append(f"| {tier.replace('_', ' ').title()} | {claude_pct:.1f}% | {gemini_pct:.1f}% | {diff:+.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Score Statistics
    lines.append("## Score Statistics")
    lines.append("")
    lines.append("| Metric | Claude | Gemini | Difference |")
    lines.append("|--------|--------|--------|------------|")
    lines.append(f"| Average | {claude_stats['avg_score']:.2f} | {gemini_stats['avg_score']:.2f} | {gemini_stats['avg_score']-claude_stats['avg_score']:+.2f} |")
    lines.append(f"| Median | {claude_stats['median_score']:.2f} | {gemini_stats['median_score']:.2f} | {gemini_stats['median_score']-claude_stats['median_score']:+.2f} |")
    lines.append(f"| Min | {claude_stats['score_range'][0]:.2f} | {gemini_stats['score_range'][0]:.2f} | {gemini_stats['score_range'][0]-claude_stats['score_range'][0]:+.2f} |")
    lines.append(f"| Max | {claude_stats['score_range'][1]:.2f} | {gemini_stats['score_range'][1]:.2f} | {gemini_stats['score_range'][1]-claude_stats['score_range'][1]:+.2f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Cost Analysis
    lines.append("## Cost Analysis (5,000 articles)")
    lines.append("")
    lines.append("| Model | Cost per Article | Total Cost | Savings |")
    lines.append("|-------|------------------|------------|---------|")

    claude_total = 5000 * 0.009
    gemini_total = 5000 * 0.00018
    savings = claude_total - gemini_total

    lines.append(f"| Claude 3.5 Sonnet | $0.009 | ${claude_total:.2f} | - |")
    lines.append(f"| Gemini 1.5 Pro | $0.00018 | ${gemini_total:.2f} | ${savings:.2f} (96%) |")
    lines.append("")
    lines.append("*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sample Comparisons (show 5 articles with biggest disagreement)
    lines.append("## Sample Article Comparisons")
    lines.append("")
    lines.append(f"Comparing {len(claude_articles)} Claude articles vs {len(gemini_articles)} Gemini articles")
    lines.append("")

    # Find articles with biggest score disagreement
    # Create dictionaries keyed by article ID to match articles correctly
    claude_by_id = {art.get('id', art.get('url', '')): art for art in claude_articles}
    gemini_by_id = {art.get('id', art.get('url', '')): art for art in gemini_articles}

    # Find common articles (analyzed by both models)
    common_ids = set(claude_by_id.keys()) & set(gemini_by_id.keys())

    lines.append(f"**Matched articles**: {len(common_ids)} (analyzed successfully by both models)")
    lines.append("")
    lines.append("Showing 5 articles with largest score disagreement:")
    lines.append("")

    disagreements = []
    for article_id in common_ids:
        claude_art = claude_by_id[article_id]
        gemini_art = gemini_by_id[article_id]

        claude_analysis = claude_art[f'{filter_name}_analysis']
        gemini_analysis = gemini_art[f'{filter_name}_analysis']

        _, claude_score = extract_tier_and_score(claude_analysis, filter_name)
        _, gemini_score = extract_tier_and_score(gemini_analysis, filter_name)

        diff = abs(claude_score - gemini_score)
        disagreements.append((diff, claude_art, gemini_art, claude_score, gemini_score))

    # Sort by disagreement (highest first)
    disagreements.sort(reverse=True, key=lambda x: x[0])

    for i, (diff, claude_art, gemini_art, claude_score, gemini_score) in enumerate(disagreements[:5], 1):
        claude_analysis = claude_art[f'{filter_name}_analysis']
        gemini_analysis = gemini_art[f'{filter_name}_analysis']

        claude_tier, _ = extract_tier_and_score(claude_analysis, filter_name)
        gemini_tier, _ = extract_tier_and_score(gemini_analysis, filter_name)

        lines.append(f"### Sample {i} - Disagreement: {diff:.2f}")
        lines.append("")
        lines.append(f"**Title**: {claude_art.get('title', 'No title')[:100]}")
        lines.append("")
        lines.append(f"**Excerpt**: {claude_art.get('content', '')[:200]}...")
        lines.append("")
        lines.append("| Model | Score | Tier | Reasoning |")
        lines.append("|-------|-------|------|-----------|")

        claude_reasoning = claude_analysis.get('reasoning', '')[:100]
        gemini_reasoning = gemini_analysis.get('reasoning', '')[:100]

        lines.append(f"| Claude | {claude_score:.2f} | {claude_tier} | {claude_reasoning}... |")
        lines.append(f"| Gemini | {gemini_score:.2f} | {gemini_tier} | {gemini_reasoning}... |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Next Steps
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Review the tier distribution and score statistics")
    lines.append("2. Examine sample articles with large disagreements")
    lines.append("3. If distributions are similar, proceed with Gemini for cost savings")
    lines.append("4. If distributions differ, consider:")
    lines.append("   - Refining the prompt for better clarity")
    lines.append("   - Adding more examples to the prompt")
    lines.append("   - Using Claude for ground truth (higher cost but higher quality)")
    lines.append("")

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nCalibration report written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate Claude vs Gemini for any semantic filter prompt'
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='Path to prompt markdown file (e.g., prompts/sustainability.md)'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Source JSONL file with articles'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of articles to test (default: 100)'
    )
    parser.add_argument(
        '--output',
        help='Output markdown report file (default: reports/<filter>_calibration.md)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )

    args = parser.parse_args()

    # Auto-detect filter name from prompt path
    prompt_path = Path(args.prompt)
    filter_name = prompt_path.stem  # e.g., "sustainability" from "sustainability.md"

    # Set default output path if not specified
    if args.output:
        output_file = Path(args.output)
    else:
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        output_file = reports_dir / f'{filter_name}_calibration.md'

    print("="*70)
    print(f"MODEL CALIBRATION: {filter_name.upper()}")
    print("="*70)
    print(f"Prompt:      {args.prompt}")
    print(f"Source:      {args.source}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output:      {output_file}")
    print(f"Random seed: {args.seed}")
    print("="*70)
    print()

    # Select sample
    source_file = Path(args.source)
    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        return 1

    articles = select_calibration_sample(source_file, args.sample_size, args.seed)

    if not articles:
        print("ERROR: No articles available for calibration!")
        return 1

    # Set up cache directory: calibrations/<filter_name>/
    cache_dir = Path('calibrations') / filter_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}\n")

    # Label with Claude
    claude_articles = label_with_provider(articles, args.prompt, 'claude', filter_name, cache_dir)

    # Label with Gemini
    gemini_articles = label_with_provider(articles, args.prompt, 'gemini', filter_name, cache_dir)

    if len(claude_articles) == 0 or len(gemini_articles) == 0:
        print("\nERROR: One or both models failed to label articles!")
        return 1

    # Analyze results
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS")
    print(f"{'='*70}\n")

    claude_stats = analyze_batch(claude_articles, 'Claude', filter_name)
    gemini_stats = analyze_batch(gemini_articles, 'Gemini', filter_name)

    # Print comparison
    print_stats_comparison(claude_stats, gemini_stats)

    # Generate report
    print(f"\n{'='*70}")
    print("GENERATING CALIBRATION REPORT")
    print(f"{'='*70}\n")

    generate_calibration_report(
        claude_articles,
        gemini_articles,
        claude_stats,
        gemini_stats,
        filter_name,
        args.prompt,
        output_file
    )

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Report: {output_file}")
    print()
    print("Review the report to decide which model to use for large-scale labeling.")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
