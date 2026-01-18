"""
Commerce Prefilter - Backtesting Script

Run the commerce SLM on historical scored articles to find commerce content
that currently slips through the prefilter.

Usage:
    python backtest.py --input "I:/Mijn Drive/NexusMind/filtered" --output backtest_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[4]))

from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM


def load_jsonl(path: Path) -> list:
    """Load articles from JSONL file."""
    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return articles


def load_all_scored_articles(nexusmind_path: Path, filter_name: str) -> list:
    """Load all scored articles (high, medium, low tiers) for a filter."""
    filter_path = nexusmind_path / "filtered" / filter_name
    articles = []

    # Define tier directories based on filter
    if filter_name == "investment_risk":
        tier_dirs = ["BLUE", "YELLOW", "NOISE"]
    else:
        tier_dirs = ["high", "medium", "low"]

    for tier in tier_dirs:
        tier_path = filter_path / tier
        if not tier_path.exists():
            continue

        for jsonl_file in tier_path.glob("*.jsonl"):
            tier_articles = load_jsonl(jsonl_file)
            for article in tier_articles:
                article['_source_tier'] = tier
                article['_source_file'] = str(jsonl_file.name)
            articles.extend(tier_articles)
            print(f"  Loaded {len(tier_articles)} from {tier}/{jsonl_file.name}")

    return articles


def run_backtest(
    nexusmind_path: Path,
    filter_names: list,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> dict:
    """Run backtest on scored articles."""

    print("=" * 60, flush=True)
    print("Commerce Prefilter - Backtest", flush=True)
    print("=" * 60, flush=True)
    print(f"Threshold: {threshold}", flush=True)
    print(f"NexusMind path: {nexusmind_path}", flush=True)

    # Load model
    print("\nLoading commerce detector...", flush=True)
    detector = CommercePrefilterSLM(threshold=threshold)
    print(f"Device: {detector.device}", flush=True)

    results = {
        "run_at": datetime.now().isoformat(),
        "threshold": threshold,
        "filters": {},
        "flagged_articles": [],
    }

    total_flagged = 0
    total_articles = 0

    for filter_name in filter_names:
        print(f"\n--- {filter_name} ---", flush=True)

        # Load articles
        articles = load_all_scored_articles(nexusmind_path, filter_name)
        if not articles:
            print(f"  No articles found for {filter_name}", flush=True)
            continue

        print(f"  Total articles: {len(articles)}", flush=True)
        total_articles += len(articles)

        # Run predictions
        print("  Running predictions...", flush=True)
        predictions = detector.batch_predict(articles, batch_size=batch_size)

        # Analyze results
        flagged = []
        scores_by_tier = defaultdict(list)

        for article, pred in zip(articles, predictions):
            tier = article.get('_source_tier', 'unknown')
            scores_by_tier[tier].append(pred['score'])

            if pred['is_commerce']:
                flagged.append({
                    "id": article.get('id', 'unknown'),
                    "title": article.get('title', '')[:100],
                    "url": article.get('url', ''),
                    "score": pred['score'],
                    "tier": tier,
                    "filter": filter_name,
                    "source": article.get('source', ''),
                })

        # Store results
        filter_results = {
            "total_articles": len(articles),
            "flagged_count": len(flagged),
            "flagged_pct": len(flagged) / len(articles) * 100 if articles else 0,
            "by_tier": {},
        }

        for tier, scores in scores_by_tier.items():
            tier_flagged = sum(1 for s in scores if s >= threshold)
            filter_results["by_tier"][tier] = {
                "count": len(scores),
                "flagged": tier_flagged,
                "flagged_pct": tier_flagged / len(scores) * 100 if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            }

        results["filters"][filter_name] = filter_results
        results["flagged_articles"].extend(flagged)
        total_flagged += len(flagged)

        # Print summary
        print(f"  Flagged as commerce: {len(flagged)} ({len(flagged)/len(articles)*100:.1f}%)")
        for tier, stats in filter_results["by_tier"].items():
            print(f"    {tier}: {stats['flagged']}/{stats['count']} ({stats['flagged_pct']:.1f}%) avg={stats['avg_score']:.3f}")

    results["summary"] = {
        "total_articles": total_articles,
        "total_flagged": total_flagged,
        "flagged_pct": total_flagged / total_articles * 100 if total_articles else 0,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total articles analyzed: {total_articles}")
    print(f"Flagged as commerce: {total_flagged} ({total_flagged/total_articles*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest commerce prefilter on historical data")
    parser.add_argument(
        "--nexusmind-path",
        type=Path,
        default=Path("I:/Mijn Drive/NexusMind"),
        help="Path to NexusMind directory",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["sustainability_technology", "uplifting", "investment_risk"],
        help="Filters to backtest",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Commerce classification threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backtest_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(
        nexusmind_path=args.nexusmind_path,
        filter_names=args.filters,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Print top flagged articles
    if results["flagged_articles"]:
        print("\n--- Top 10 Flagged Articles ---")
        sorted_flagged = sorted(results["flagged_articles"], key=lambda x: x['score'], reverse=True)
        for i, article in enumerate(sorted_flagged[:10], 1):
            print(f"\n{i}. [{article['score']:.3f}] {article['title']}")
            print(f"   Filter: {article['filter']}, Tier: {article['tier']}")
            print(f"   URL: {article['url'][:80]}...")


if __name__ == "__main__":
    main()
