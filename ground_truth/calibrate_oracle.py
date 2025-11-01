#!/usr/bin/env python3
"""
Oracle Calibration: Compare LLM models for semantic filters with random sampling.

This script selects a RANDOM sample (default 100 articles) and labels it with
multiple LLM models, then generates a detailed comparison report to help you decide:
- Which LLM to use for large-scale labeling (cost vs quality tradeoff)
- Whether dimension-level judgments align between models
- Whether tier distributions match expectations
- How much the prefilter reduces labeling costs

Features:
- Random sampling with reproducible seed (--seed)
- Automatic comprehensive text cleaning (Unicode, HTML, BiDi marks, etc.)
- Prefilter support (filter packages only)
- Dimension-level correlation analysis
- Cost-saving estimates

Usage (Recommended - Filter Package):
    python -m ground_truth.calibrate_oracle \
        --filter filters/uplifting/v1 \
        --source "datasets/raw/master_dataset_2025*.jsonl" \
        --models gemini-flash,gemini-pro \
        --sample-size 100 \
        --seed 42

Usage (Legacy - Direct Prompt):
    python -m ground_truth.calibrate_oracle \
        --prompt prompts/sustainability.md \
        --source ../content-aggregator/data/collected/articles.jsonl \
        --models claude,gemini-flash \
        --sample-size 100

Key Parameters:
    --filter          : Filter package path (includes prefilter + prompt)
    --source          : JSONL file(s) - supports glob patterns
    --models          : Comma-separated list of models to compare
    --sample-size     : Number of articles to randomly sample (default: 100)
    --seed            : Random seed for reproducibility (default: 42)

Output:
    - Markdown report with statistics and comparisons
    - Dimension-level correlation analysis (if matplotlib available)
    - Visualization plots (scatter plots, correlation heatmap)
    - Prefilter efficiency statistics
"""

import json
import random
import argparse
import sys
import glob as glob_module
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from .batch_labeler import GenericBatchLabeler

# Visualization imports (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print("Warning: matplotlib not available. Skipping visualizations.")


def load_filter_package(filter_path: Path) -> Tuple:
    """
    Load filter package components.

    Returns:
        (prefilter_instance, prompt_path, config_dict)
    """
    print(f"Loading filter package: {filter_path}")

    # Load prefilter
    prefilter_module_path = filter_path / "prefilter.py"
    prefilter = None

    if prefilter_module_path.exists():
        spec = importlib.util.spec_from_file_location("prefilter", prefilter_module_path)
        prefilter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefilter_module)

        # Get the prefilter class (defined in this module, not imported)
        prefilter_classes = [
            obj for name, obj in vars(prefilter_module).items()
            if isinstance(obj, type) and 'PreFilter' in name
            and obj.__module__ == prefilter_module.__name__  # Exclude imported classes
        ]

        if prefilter_classes:
            prefilter_class = prefilter_classes[0]
            prefilter = prefilter_class()
            print(f"  Loaded: {prefilter_class.__name__} v{prefilter.VERSION}")

    # Find prompt file (try compressed first, then regular)
    prompt_path = filter_path / "prompt-compressed.md"
    if not prompt_path.exists():
        prompt_path = filter_path / "prompt.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"No prompt file found in {filter_path}")

    print(f"  Prompt: {prompt_path.name}")

    # Load config (for reference)
    config_path = filter_path / "config.yaml"
    config_dict = {}
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

    return prefilter, prompt_path, config_dict


def select_calibration_sample(
    source_pattern: str,
    sample_size: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Select random articles for calibration. Supports glob patterns."""
    random.seed(seed)

    # Handle glob patterns
    source_files = glob_module.glob(source_pattern)
    if not source_files:
        # Try as literal path
        source_files = [source_pattern]

    print(f"Loading articles from {len(source_files)} file(s)...")

    articles = []
    for source_file in source_files:
        source_path = Path(source_file)
        if not source_path.exists():
            continue

        print(f"  Reading: {source_path.name}")
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line.strip())
                articles.append(article)

    print(f"Loaded {len(articles)} total articles")

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

    # Check if cached labels exist
    if cache_dir:
        cache_file = cache_dir / f"{provider}_labels.jsonl"
        if cache_file.exists():
            print(f"Found cached labels at {cache_file}")
            print(f"Loading cached results...")
            labeled_articles = []
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line.strip())
                    labeled_articles.append(article)

            print(f"Loaded {len(labeled_articles)} cached labeled articles")
            return labeled_articles

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

    # Extract dimension scores
    dimension_names = ['agency', 'progress', 'collective_benefit', 'connection',
                       'innovation', 'justice', 'resilience', 'wonder']
    dimensions_data = {dim: [] for dim in dimension_names}

    for analysis in analyses:
        tier, score = extract_tier_and_score(analysis, filter_name)
        tiers.append(tier)
        scores.append(score)

        # Content type if available (handle both string and list)
        content_type = analysis.get('content_type', 'unknown')
        # Convert list to tuple for hashability, or use string as-is
        if isinstance(content_type, list):
            content_type = tuple(content_type) if content_type else 'unknown'
        content_types.append(content_type)

        # Extract dimension scores
        for dim in dimension_names:
            dim_value = analysis.get(dim, 0)
            dimensions_data[dim].append(dim_value)

    # Calculate stats
    tier_counts = Counter(tiers)
    tier_pcts = {k: v/len(articles)*100 for k, v in tier_counts.items()}

    # Calculate dimension statistics
    dimension_stats = {}
    for dim, values in dimensions_data.items():
        if values:
            dimension_stats[dim] = {
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values)//2],
                'min': min(values),
                'max': max(values),
                'values': values  # Keep for correlation analysis
            }

    stats = {
        'provider': provider_name,
        'total': len(articles),
        'tier_counts': tier_counts,
        'tier_pcts': tier_pcts,
        'avg_score': sum(scores)/len(scores) if scores else 0,
        'median_score': sorted(scores)[len(scores)//2] if scores else 0,
        'score_range': (min(scores), max(scores)) if scores else (0, 0),
        'content_type_counts': Counter(content_types),
        'dimensions': dimension_stats
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
    print(f"\nOVERALL SCORE STATISTICS:")
    print(f"  Average:  {stats1['avg_score']:.2f} vs {stats2['avg_score']:.2f} (diff: {stats2['avg_score']-stats1['avg_score']:+.2f})")
    print(f"  Median:   {stats1['median_score']:.2f} vs {stats2['median_score']:.2f} (diff: {stats2['median_score']-stats1['median_score']:+.2f})")
    print(f"  Range:    {stats1['score_range'][0]:.2f}-{stats1['score_range'][1]:.2f} vs {stats2['score_range'][0]:.2f}-{stats2['score_range'][1]:.2f}")

    # Dimension comparison
    if 'dimensions' in stats1 and 'dimensions' in stats2:
        print(f"\nDIMENSION SCORE COMPARISON:")
        print(f"  {'Dimension':<20} {stats1['provider']:>15} {stats2['provider']:>15} {'Difference':>15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        all_dims = set(stats1['dimensions'].keys()) | set(stats2['dimensions'].keys())
        for dim in ['agency', 'progress', 'collective_benefit', 'connection',
                    'innovation', 'justice', 'resilience', 'wonder']:
            if dim in all_dims:
                mean1 = stats1['dimensions'][dim]['mean']
                mean2 = stats2['dimensions'][dim]['mean']
                diff = mean2 - mean1
                print(f"  {dim:<20} {mean1:15.2f} {mean2:15.2f} {diff:+15.2f}")

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


def calculate_correlation(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient and linear regression slope.

    Returns (correlation, slope)
    """
    if not values1 or not values2 or len(values1) != len(values2):
        return 0.0, 0.0

    if VISUALIZATIONS_AVAILABLE:
        corr = np.corrcoef(values1, values2)[0, 1]
        # Linear regression: slope
        if np.std(values1) > 0:
            slope = np.cov(values1, values2)[0, 1] / np.var(values1)
        else:
            slope = 0.0
        return corr, slope
    else:
        # Manual calculation without numpy
        n = len(values1)
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n

        std1 = (sum((x - mean1) ** 2 for x in values1) / n) ** 0.5
        std2 = (sum((x - mean2) ** 2 for x in values2) / n) ** 0.5

        if std1 == 0 or std2 == 0:
            return 0.0, 0.0

        covariance = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n)) / n
        corr = covariance / (std1 * std2)
        slope = covariance / (std1 ** 2) if std1 > 0 else 0.0

        return corr, slope


def analyze_correlations(stats1: Dict, stats2: Dict, filter_name: str) -> Dict:
    """
    Analyze correlations between two models' dimension scores.

    Returns dict with correlation coefficients and slopes for each dimension.
    """
    correlations = {}

    if 'dimensions' not in stats1 or 'dimensions' not in stats2:
        return correlations

    dims1 = stats1['dimensions']
    dims2 = stats2['dimensions']

    for dim in dims1.keys():
        if dim in dims2:
            values1 = dims1[dim]['values']
            values2 = dims2[dim]['values']

            # Ensure same length (match by article order)
            min_len = min(len(values1), len(values2))
            values1 = values1[:min_len]
            values2 = values2[:min_len]

            corr, slope = calculate_correlation(values1, values2)
            correlations[dim] = {
                'correlation': corr,
                'slope': slope,
                'values1': values1,
                'values2': values2
            }

    return correlations


def generate_correlation_visualizations(
    correlations: Dict,
    stats1: Dict,
    stats2: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Generate scatter plots for dimension correlations.

    Returns list of generated image paths.
    """
    if not VISUALIZATIONS_AVAILABLE:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    # Dimension names in order
    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, dim in enumerate(dimensions):
        if dim not in correlations:
            continue

        ax = axes[idx]
        corr_data = correlations[dim]
        values1 = corr_data['values1']
        values2 = corr_data['values2']
        corr = corr_data['correlation']
        slope = corr_data['slope']

        # Scatter plot
        ax.scatter(values1, values2, alpha=0.5, s=30)

        # Regression line
        if len(values1) > 0:
            x_range = np.linspace(min(values1), max(values1), 100)
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            y_range = mean2 + slope * (x_range - mean1)
            ax.plot(x_range, y_range, 'r--', linewidth=2, alpha=0.7)

        # Diagonal reference line (perfect agreement)
        max_val = max(max(values1) if values1 else 10, max(values2) if values2 else 10)
        ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1, alpha=0.3, label='y=x')

        ax.set_xlabel(f"{stats1['provider']}", fontsize=10)
        ax.set_ylabel(f"{stats2['provider']}", fontsize=10)
        ax.set_title(f"{dim}\nr={corr:.3f}, slope={slope:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # Remove extra subplot
    if len(dimensions) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()

    # Save figure
    img_path = output_dir / 'dimension_correlations.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths.append(img_path)

    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    corr_matrix = []
    labels = []
    for dim in dimensions:
        if dim in correlations:
            corr_matrix.append(correlations[dim]['correlation'])
            labels.append(dim)

    if corr_matrix:
        y_pos = np.arange(len(labels))
        colors = plt.cm.RdYlGn([(c + 1) / 2 for c in corr_matrix])  # Map -1..1 to 0..1

        bars = ax.barh(y_pos, corr_matrix, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Pearson Correlation')
        ax.set_title(f"Dimension Correlations\n{stats1['provider']} vs {stats2['provider']}")
        ax.set_xlim(-1, 1)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, corr_matrix)):
            ax.text(val + 0.02 if val > 0 else val - 0.02, i, f'{val:.3f}',
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)

    plt.tight_layout()
    img_path = output_dir / 'correlation_summary.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths.append(img_path)

    return image_paths


def print_correlation_analysis(correlations: Dict, stats1: Dict, stats2: Dict):
    """Print correlation analysis to console."""
    if not correlations:
        return

    print(f"\nCORRELATION ANALYSIS:")
    print(f"  {'Dimension':<20} {'Correlation':>12} {'Slope':>12} {'Interpretation'}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*30}")

    for dim in ['agency', 'progress', 'collective_benefit', 'connection',
                'innovation', 'justice', 'resilience', 'wonder']:
        if dim in correlations:
            corr = correlations[dim]['correlation']
            slope = correlations[dim]['slope']

            # Interpretation
            if corr > 0.9:
                interp = "Excellent agreement"
            elif corr > 0.7:
                interp = "Good agreement"
            elif corr > 0.5:
                interp = "Moderate agreement"
            elif corr > 0.3:
                interp = "Weak agreement"
            else:
                interp = "Poor agreement"

            print(f"  {dim:<20} {corr:12.3f} {slope:12.3f} {interp}")

    # Overall interpretation
    avg_corr = sum(correlations[d]['correlation'] for d in correlations) / len(correlations)
    print(f"\n  Average correlation: {avg_corr:.3f}")

    if avg_corr > 0.7:
        print(f"  → Models show strong agreement on rankings (different scales)")
        print(f"  → Consider: normalize {stats1['provider']} scores to match {stats2['provider']}'s scale")
    elif avg_corr > 0.5:
        print(f"  → Models show moderate agreement")
        print(f"  → Some semantic differences exist beyond scale")
    else:
        print(f"  → Models show weak agreement")
        print(f"  → Significant semantic disagreement, choose oracle carefully")


def generate_calibration_report(
    claude_articles: List[Dict],
    gemini_articles: List[Dict],
    claude_stats: Dict,
    gemini_stats: Dict,
    filter_name: str,
    prompt_path: str,
    output_file: Path,
    prefilter_stats: Dict = None
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

    # Pre-filter Statistics (if available)
    if prefilter_stats and prefilter_stats['total_sampled'] > 0:
        lines.append("## Pre-filter Statistics")
        lines.append("")
        lines.append(f"- **Total Articles Sampled**: {prefilter_stats['total_sampled']}")
        lines.append(f"- **Passed Pre-filter**: {prefilter_stats['passed']} ({prefilter_stats['pass_rate']:.1f}%)")
        lines.append(f"- **Blocked by Pre-filter**: {prefilter_stats['blocked']} ({100-prefilter_stats['pass_rate']:.1f}%)")

        if prefilter_stats['block_reasons']:
            lines.append("")
            lines.append("**Block Reasons**:")
            lines.append("")
            for reason, count in prefilter_stats['block_reasons'].most_common():
                pct = count / prefilter_stats['blocked'] * 100 if prefilter_stats['blocked'] > 0 else 0
                lines.append(f"- {reason}: {count} ({pct:.1f}%)")

        lines.append("")

        # Cost savings from pre-filter
        if prefilter_stats['pass_rate'] < 100:
            api_calls_saved = prefilter_stats['blocked']
            lines.append(f"**Cost Savings**: Pre-filter blocked {api_calls_saved} articles, saving expensive LLM API calls.")
            lines.append(f"For ground truth generation (5,000 articles), this would save ~{int(5000 * (100-prefilter_stats['pass_rate'])/100)} API calls.")

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
        description='Compare multiple LLM models for any semantic filter'
    )

    # Support both --filter (new) and --prompt (legacy)
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument(
        '--filter',
        help='Path to filter package directory (e.g., filters/uplifting/v1)'
    )
    filter_group.add_argument(
        '--prompt',
        help='Path to prompt markdown file [DEPRECATED: use --filter instead]'
    )

    parser.add_argument(
        '--source',
        required=True,
        help='Source JSONL file(s) with articles (supports glob patterns)'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='gemini-flash,gemini-pro',
        help='Comma-separated list of models to compare (default: gemini-flash,gemini-pro). Options: gemini-flash, gemini-pro, claude-sonnet'
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

    # Parse models list
    model_list = [m.strip() for m in args.models.split(',')]
    if len(model_list) < 2:
        print("ERROR: Must specify at least 2 models to compare")
        return 1

    # Load filter package or just prompt
    prefilter = None
    prompt_path = None

    if args.filter:
        filter_path = Path(args.filter)
        filter_name = filter_path.parent.name  # e.g., "uplifting" from "filters/uplifting/v1"
        prefilter, prompt_path, config = load_filter_package(filter_path)
    else:
        # Legacy --prompt mode
        prompt_path = Path(args.prompt)
        filter_name = prompt_path.stem
        print("WARNING: --prompt is deprecated. Use --filter for pre-filter support.")
        print()

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
    print(f"Prompt:      {prompt_path}")
    print(f"Source:      {args.source}")
    print(f"Models:      {', '.join(model_list)}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output:      {output_file}")
    print(f"Random seed: {args.seed}")
    if prefilter:
        print(f"Pre-filter:  {prefilter.__class__.__name__} v{prefilter.VERSION}")
    print("="*70)
    print()

    # Select sample (now supports glob patterns)
    articles = select_calibration_sample(args.source, args.sample_size, args.seed)

    if not articles:
        print("ERROR: No articles available for calibration!")
        return 1

    # Apply pre-filter if available
    prefilter_stats = {
        'total_sampled': len(articles),
        'passed': 0,
        'blocked': 0,
        'pass_rate': 0.0,
        'block_reasons': Counter()
    }

    if prefilter:
        print(f"\n{'='*70}")
        print(f"APPLYING PRE-FILTER: {prefilter.__class__.__name__}")
        print(f"{'='*70}\n")

        passed_articles = []
        blocked_articles = []

        for i, article in enumerate(articles, 1):
            should_label, reason = prefilter.should_label(article)

            if should_label:
                passed_articles.append(article)
            else:
                blocked_articles.append((article, reason))
                prefilter_stats['block_reasons'][reason] += 1

            if i % 10 == 0:
                print(f"  Processed {i}/{len(articles)} articles...", end='\r')

        print(f"  Processed {len(articles)}/{len(articles)} articles... DONE")
        print()

        prefilter_stats['passed'] = len(passed_articles)
        prefilter_stats['blocked'] = len(blocked_articles)
        prefilter_stats['pass_rate'] = (len(passed_articles) / len(articles) * 100) if articles else 0

        print(f"Pre-filter Results:")
        print(f"  Total sampled:  {len(articles)}")
        print(f"  Passed:         {len(passed_articles)} ({prefilter_stats['pass_rate']:.1f}%)")
        print(f"  Blocked:        {len(blocked_articles)} ({100-prefilter_stats['pass_rate']:.1f}%)")

        if blocked_articles:
            print(f"\n  Block reasons:")
            for reason, count in prefilter_stats['block_reasons'].most_common():
                pct = count / len(blocked_articles) * 100
                print(f"    - {reason}: {count} ({pct:.1f}%)")

        # Use only passed articles for labeling
        articles = passed_articles

        if not articles:
            print("\nERROR: Pre-filter blocked all articles! No articles to label.")
            return 1

        print(f"\nProceeding with {len(articles)} passed articles for oracle labeling...")
    else:
        print("No pre-filter found - labeling all sampled articles")
        prefilter_stats['passed'] = len(articles)
        prefilter_stats['pass_rate'] = 100.0

    # Set up cache directory: calibrations/<filter_name>/
    cache_dir = Path('calibrations') / filter_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCache directory: {cache_dir}\n")

    # Label with each model
    labeled_results = {}
    for model in model_list:
        # Map model names to provider names
        provider_map = {
            'gemini-flash': 'gemini',
            'gemini-pro': 'gemini-pro',
            'claude-sonnet': 'claude'
        }
        provider = provider_map.get(model, model)

        labeled_articles = label_with_provider(articles, str(prompt_path), provider, filter_name, cache_dir)

        if len(labeled_articles) == 0:
            print(f"\nERROR: {model} failed to label articles!")
            return 1

        labeled_results[model] = labeled_articles

    # Analyze results for each model
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS")
    print(f"{'='*70}\n")

    stats_results = {}
    for model, labeled_articles in labeled_results.items():
        stats = analyze_batch(labeled_articles, model, filter_name)
        stats_results[model] = stats

    # Print comparison (first two models)
    models = list(model_list)
    print_stats_comparison(stats_results[models[0]], stats_results[models[1]])

    # Correlation analysis
    correlations = analyze_correlations(stats_results[models[0]], stats_results[models[1]], filter_name)
    print_correlation_analysis(correlations, stats_results[models[0]], stats_results[models[1]])

    # Generate visualizations
    if VISUALIZATIONS_AVAILABLE and correlations:
        print(f"\n{'='*70}")
        print("GENERATING CORRELATION VISUALIZATIONS")
        print(f"{'='*70}\n")

        viz_dir = output_file.parent / f"{output_file.stem}_visualizations"
        image_paths = generate_correlation_visualizations(
            correlations,
            stats_results[models[0]],
            stats_results[models[1]],
            viz_dir
        )

        if image_paths:
            print(f"Generated {len(image_paths)} visualization(s):")
            for img_path in image_paths:
                print(f"  - {img_path}")
            print()

    # Generate report
    print(f"\n{'='*70}")
    print("GENERATING CALIBRATION REPORT")
    print(f"{'='*70}\n")

    generate_calibration_report(
        labeled_results[models[0]],
        labeled_results[models[1]],
        stats_results[models[0]],
        stats_results[models[1]],
        filter_name,
        str(prompt_path),
        output_file,
        prefilter_stats
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
