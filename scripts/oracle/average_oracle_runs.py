"""
Average multiple oracle scoring runs for continuous training targets.

Joins oracle runs by article URL, averages per-dimension scores, and reports
distribution statistics. Designed to combat the discrete score problem where
single oracle runs produce only 15-17 unique values per dimension.

With 3 runs, expect ~45+ unique values per dimension → better training signal.

Usage:
    python scripts/oracle/average_oracle_runs.py \
        --runs datasets/scored/thriving_v1_run1 \
               datasets/scored/thriving_v1_run2 \
               datasets/scored/thriving_v1_run3 \
        --output datasets/scored/thriving_v1_averaged \
        --filter-name thriving
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ground_truth import analysis_field_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_scored_articles(run_dir: Path, filter_name: str) -> Dict[str, dict]:
    """Load scored articles from a run directory, keyed by URL.

    Looks for JSONL files in the directory and extracts articles that have
    the expected analysis field.
    """
    field_name = analysis_field_name(filter_name)
    articles = {}

    jsonl_files = list(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {run_dir}")
        return articles

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at {jsonl_file}:{line_num}")
                    continue

                url = article.get("url")
                if not url:
                    continue

                if field_name not in article:
                    continue

                articles[url] = article

    logger.info(f"Loaded {len(articles)} scored articles from {run_dir}")
    return articles


def extract_dimension_scores(analysis: dict, dimensions: Optional[List[str]] = None) -> Dict[str, float]:
    """Extract numeric scores from an analysis dict.

    Handles both flat format (dimension: score) and nested format
    (dimension: {score: X, evidence: Y}).
    """
    scores = {}
    for key, value in analysis.items():
        if key in ("content_type", "weighted_average", "tier", "tier_description"):
            continue
        if dimensions and key not in dimensions:
            continue

        if isinstance(value, dict) and "score" in value:
            scores[key] = float(value["score"])
        elif isinstance(value, (int, float)):
            scores[key] = float(value)

    return scores


def average_runs(
    run_dirs: List[Path],
    filter_name: str,
    dimensions: Optional[List[str]] = None,
) -> List[dict]:
    """Average oracle scores across multiple runs.

    Args:
        run_dirs: List of directories containing scored JSONL files
        filter_name: Filter name (for analysis field lookup)
        dimensions: Optional list of dimension names to average

    Returns:
        List of articles with averaged scores
    """
    field_name = analysis_field_name(filter_name)

    # Load all runs
    runs = []
    for run_dir in run_dirs:
        articles = load_scored_articles(run_dir, filter_name)
        if not articles:
            logger.warning(f"No articles loaded from {run_dir}")
        runs.append(articles)

    if not runs:
        logger.error("No runs loaded")
        return []

    # Find articles present in ALL runs
    all_urls = set(runs[0].keys())
    for run in runs[1:]:
        all_urls &= set(run.keys())

    logger.info(f"Articles present in all {len(runs)} runs: {len(all_urls)}")

    # Report coverage
    for i, run in enumerate(runs):
        total = len(run)
        in_common = len(all_urls & set(run.keys()))
        logger.info(f"  Run {i+1}: {total} total, {in_common} in common ({in_common/total*100:.1f}%)")

    # Average scores
    averaged_articles = []
    dimension_values = defaultdict(list)  # For distribution stats

    for url in sorted(all_urls):
        # Use first run's article as base (preserving metadata)
        base_article = runs[0][url].copy()

        # Collect scores from all runs
        all_scores = []
        for run in runs:
            analysis = run[url].get(field_name, {})
            scores = extract_dimension_scores(analysis, dimensions)
            if scores:
                all_scores.append(scores)

        if not all_scores:
            continue

        # Average each dimension
        averaged_analysis = {}
        dim_names = set()
        for scores in all_scores:
            dim_names.update(scores.keys())

        for dim in sorted(dim_names):
            values = [s[dim] for s in all_scores if dim in s]
            if values:
                avg = sum(values) / len(values)
                averaged_analysis[dim] = round(avg, 4)
                dimension_values[dim].append(avg)

        base_article[field_name] = averaged_analysis
        averaged_articles.append(base_article)

    logger.info(f"Averaged {len(averaged_articles)} articles across {len(runs)} runs")

    # Report distribution statistics
    logger.info("\nDistribution statistics (per dimension):")
    logger.info(f"{'Dimension':<30} {'Unique':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    logger.info("-" * 80)

    for dim in sorted(dimension_values.keys()):
        values = dimension_values[dim]
        unique = len(set(round(v, 4) for v in values))
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        logger.info(
            f"{dim:<30} {unique:>8} {mean:>8.3f} {std:>8.3f} "
            f"{min(values):>8.3f} {max(values):>8.3f}"
        )

    return averaged_articles


def main():
    parser = argparse.ArgumentParser(
        description="Average multiple oracle scoring runs for continuous training targets"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing scored JSONL files (one per run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for averaged JSONL",
    )
    parser.add_argument(
        "--filter-name",
        type=str,
        required=True,
        help="Filter name (e.g., 'thriving', 'uplifting')",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        type=str,
        default=None,
        help="Specific dimensions to average (default: all found in data)",
    )

    args = parser.parse_args()

    # Validate run directories
    for run_dir in args.runs:
        if not run_dir.is_dir():
            logger.error(f"Run directory not found: {run_dir}")
            sys.exit(1)

    # Average runs
    averaged_articles = average_runs(args.runs, args.filter_name, args.dimensions)

    if not averaged_articles:
        logger.error("No articles to write")
        sys.exit(1)

    # Write output
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"{args.filter_name}_v1_averaged.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for article in averaged_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {len(averaged_articles)} averaged articles to {output_file}")

    # Summary
    field_name = analysis_field_name(args.filter_name)
    sample = averaged_articles[0].get(field_name, {})
    logger.info(f"Sample averaged scores: {json.dumps(sample, indent=2)}")


if __name__ == "__main__":
    main()
