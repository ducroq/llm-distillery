"""
Commerce Prefilter - Oracle Scoring

Runs the oracle on articles and generates analysis report.
Supports resume capability for long-running batch jobs.

Usage:
    # Calibration (small sample)
    python -m filters.common.commerce_prefilter.training.run_calibration

    # Full training set (with resume)
    python -m filters.common.commerce_prefilter.training.run_calibration \
        --input datasets/training/commerce_prefilter_v1/training_sample.jsonl \
        --output-dir datasets/training/commerce_prefilter_v1 \
        --model models/gemini-2.5-flash \
        --resume
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

from filters.common.commerce_prefilter.v1.oracle import CommerceOracle


def load_articles(path: Path) -> list:
    """Load articles from JSONL file."""
    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def load_existing_results(path: Path) -> dict:
    """Load already-scored results, keyed by article_id."""
    results = {}
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('article_id'):
                        results[r['article_id']] = r
                except json.JSONDecodeError:
                    continue
    return results


def analyze_results(results: list) -> dict:
    """Analyze calibration results."""
    # Filter out errors
    valid = [r for r in results if r.get('commerce_score', -1) >= 0]
    errors = [r for r in results if r.get('commerce_score', -1) < 0]

    scores = [r['commerce_score'] for r in valid]

    # Score distribution
    buckets = Counter()
    for s in scores:
        if s <= 2:
            buckets['0-2 (journalism)'] += 1
        elif s <= 4:
            buckets['3-4 (mostly journalism)'] += 1
        elif s <= 6:
            buckets['5-6 (mixed)'] += 1
        elif s <= 8:
            buckets['7-8 (mostly commerce)'] += 1
        else:
            buckets['9-10 (commerce)'] += 1

    # By sample bucket
    by_bucket = {}
    for r in valid:
        bucket = r.get('_sample_bucket', 'unknown')
        if bucket not in by_bucket:
            by_bucket[bucket] = []
        by_bucket[bucket].append(r['commerce_score'])

    bucket_stats = {}
    for bucket, bucket_scores in by_bucket.items():
        bucket_stats[bucket] = {
            'count': len(bucket_scores),
            'mean': round(mean(bucket_scores), 2),
            'median': round(median(bucket_scores), 2),
            'min': round(min(bucket_scores), 2),
            'max': round(max(bucket_scores), 2),
        }

    # Key signals frequency
    all_signals = []
    for r in valid:
        all_signals.extend(r.get('key_signals', []))
    signal_counts = Counter(all_signals).most_common(20)

    return {
        'total': len(results),
        'valid': len(valid),
        'errors': len(errors),
        'success_rate': round(len(valid) / len(results) * 100, 1) if results else 0,
        'score_stats': {
            'mean': round(mean(scores), 2) if scores else 0,
            'median': round(median(scores), 2) if scores else 0,
            'stdev': round(stdev(scores), 2) if len(scores) > 1 else 0,
            'min': round(min(scores), 2) if scores else 0,
            'max': round(max(scores), 2) if scores else 0,
        },
        'distribution': dict(buckets),
        'by_sample_bucket': bucket_stats,
        'top_signals': signal_counts,
        'error_samples': [{'title': r['title'], 'error': r.get('error')} for r in errors[:5]],
    }


def generate_report(analysis: dict, results: list, output_dir: Path) -> str:
    """Generate markdown calibration report."""

    # Get samples for manual review
    valid = [r for r in results if r.get('commerce_score', -1) >= 0]
    valid_sorted = sorted(valid, key=lambda x: x['commerce_score'])

    # Select samples: low, medium, high scores
    low_samples = [r for r in valid_sorted if r['commerce_score'] <= 2][:3]
    mid_samples = [r for r in valid_sorted if 4 <= r['commerce_score'] <= 6][:3]
    high_samples = [r for r in valid_sorted if r['commerce_score'] >= 8][:3]

    report = f"""# Oracle Calibration Report - Commerce Prefilter v1

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Oracle Model:** Gemini 1.5 Pro
**Sample Size:** {analysis['total']} articles
**Status:** [TO BE DETERMINED AFTER MANUAL REVIEW]

---

## Executive Summary

[TO BE FILLED AFTER MANUAL REVIEW]

---

## Calibration Results

### Completeness
- Total articles: {analysis['total']}
- Successfully scored: {analysis['valid']} ({analysis['success_rate']}%)
- Errors: {analysis['errors']}

### Score Statistics
| Metric | Value |
|--------|-------|
| Mean | {analysis['score_stats']['mean']} |
| Median | {analysis['score_stats']['median']} |
| Std Dev | {analysis['score_stats']['stdev']} |
| Min | {analysis['score_stats']['min']} |
| Max | {analysis['score_stats']['max']} |

### Score Distribution
| Range | Count | Percentage |
|-------|-------|------------|
"""
    total = analysis['valid']
    for range_name, count in sorted(analysis['distribution'].items()):
        pct = round(count / total * 100, 1) if total else 0
        report += f"| {range_name} | {count} | {pct}% |\n"

    report += """
### By Sample Bucket
| Bucket | Count | Mean | Median | Min | Max |
|--------|-------|------|--------|-----|-----|
"""
    for bucket, stats in analysis['by_sample_bucket'].items():
        report += f"| {bucket} | {stats['count']} | {stats['mean']} | {stats['median']} | {stats['min']} | {stats['max']} |\n"

    report += """
### Top Key Signals Detected
| Signal | Count |
|--------|-------|
"""
    for signal, count in analysis['top_signals'][:15]:
        report += f"| {signal} | {count} |\n"

    report += """
---

## Samples for Manual Review

### LOW Scores (0-2) - Should be Journalism
"""
    for r in low_samples:
        report += f"""
**Title:** {r['title']}
**Score:** {r['commerce_score']}
**Reasoning:** {r['reasoning']}
**Signals:** {', '.join(r.get('key_signals', []))}
**Manual Assessment:** [ ] Correct [ ] Incorrect
"""

    report += """
### MEDIUM Scores (4-6) - Mixed/Ambiguous
"""
    for r in mid_samples:
        report += f"""
**Title:** {r['title']}
**Score:** {r['commerce_score']}
**Reasoning:** {r['reasoning']}
**Signals:** {', '.join(r.get('key_signals', []))}
**Manual Assessment:** [ ] Correct [ ] Incorrect
"""

    report += """
### HIGH Scores (8-10) - Should be Commerce
"""
    for r in high_samples:
        report += f"""
**Title:** {r['title']}
**Score:** {r['commerce_score']}
**Reasoning:** {r['reasoning']}
**Signals:** {', '.join(r.get('key_signals', []))}
**Manual Assessment:** [ ] Correct [ ] Incorrect
"""

    if analysis['errors'] > 0:
        report += """
---

## Errors

"""
        for err in analysis['error_samples']:
            report += f"- **{err['title']}**: {err['error']}\n"

    report += """
---

## Decision Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Success rate | > 95% | {success_rate}% | {success_status} |
| Score variance | std > 1.0 | {stdev} | {variance_status} |
| Commerce bucket separation | commerce_source mean > journalism mean | {separation} | {sep_status} |
| Manual review agreement | > 80% | [PENDING] | [PENDING] |

---

## Recommendation

**[TO BE DETERMINED AFTER MANUAL REVIEW]**

Options:
- ✅ **READY** - Proceed with full batch scoring using Gemini Flash
- ⚠️ **REVIEW** - Minor issues, may need prompt adjustments
- ❌ **BLOCK** - Significant issues, do not proceed

---

## Next Steps

1. [ ] Complete manual review of samples above
2. [ ] Update status based on review
3. [ ] If READY: Run full scoring with `--model gemini-flash`
4. [ ] If REVIEW/BLOCK: Adjust prompt and re-calibrate

---

*Report generated: {datetime}*
""".format(
        success_rate=analysis['success_rate'],
        success_status='✅' if analysis['success_rate'] >= 95 else '❌',
        stdev=analysis['score_stats']['stdev'],
        variance_status='✅' if analysis['score_stats']['stdev'] >= 1.0 else '⚠️',
        separation=f"commerce={analysis['by_sample_bucket'].get('commerce_source', {}).get('mean', 'N/A')}, journalism={analysis['by_sample_bucket'].get('journalism', {}).get('mean', 'N/A')}",
        sep_status='✅' if analysis['by_sample_bucket'].get('commerce_source', {}).get('mean', 0) > analysis['by_sample_bucket'].get('journalism', {}).get('mean', 10) else '⚠️',
        datetime=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run commerce prefilter oracle scoring"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("datasets/calibration/commerce_prefilter_v1/calibration_sample.jsonl"),
        help="Input sample file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("datasets/calibration/commerce_prefilter_v1"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model (use gemini-2.5-flash for production)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of articles (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results (skip already-scored articles)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="scored.jsonl",
        help="Output filename (default: scored.jsonl)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Oracle Scoring")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Resume: {args.resume}")

    # Setup output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / args.output_file

    # Load articles
    articles = load_articles(args.input)
    if args.limit:
        articles = articles[:args.limit]
    print(f"Total articles: {len(articles)}")

    # Check for resume
    existing_results = {}
    if args.resume:
        existing_results = load_existing_results(results_path)
        print(f"Already scored: {len(existing_results)}")

    # Filter to articles needing scoring
    articles_to_score = []
    for article in articles:
        article_id = article.get('article_id', article.get('id', ''))
        if article_id not in existing_results:
            articles_to_score.append(article)

    print(f"Articles to score: {len(articles_to_score)}")

    if not articles_to_score:
        print("All articles already scored!")
    else:
        # Initialize oracle
        oracle = CommerceOracle(model_name=args.model)

        # Run scoring with incremental saves
        print(f"\nScoring articles (delay={args.delay}s)...")
        print("Progress will be saved incrementally.\n")

        # Open file in append mode for incremental saves
        with open(results_path, 'a', encoding='utf-8') as f:
            for i, article in enumerate(articles_to_score):
                try:
                    result = oracle.score_article(article)
                    result['_sample_bucket'] = article.get('_sample_bucket', 'unknown')

                    # Save immediately
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                    # Progress update
                    if (i + 1) % 50 == 0 or i == 0:
                        total_done = len(existing_results) + i + 1
                        print(f"  [{total_done}/{len(articles)}] Scored: {article.get('title', '')[:50]}...")

                    # Delay between calls
                    if i < len(articles_to_score) - 1:
                        import time
                        time.sleep(args.delay)

                except Exception as e:
                    print(f"  ERROR scoring {article.get('title', '')[:30]}: {e}")
                    # Save error result
                    error_result = {
                        'article_id': article.get('article_id', article.get('id', '')),
                        'title': article.get('title', ''),
                        'commerce_score': -1,
                        'error': str(e),
                        '_sample_bucket': article.get('_sample_bucket', 'unknown'),
                    }
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()

    # Load all results for analysis
    print("\nLoading all results for analysis...")
    all_results = load_existing_results(results_path)
    results_list = list(all_results.values())

    # Analyze
    print("Analyzing results...")
    analysis = analyze_results(results_list)

    # Save analysis
    analysis_path = args.output_dir / "analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")

    # Generate report
    report = generate_report(analysis, results_list, args.output_dir)
    report_path = args.output_dir / "SCORING_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total scored: {analysis['valid']} / {analysis['total']}")
    print(f"Success rate: {analysis['success_rate']}%")
    print(f"Score mean: {analysis['score_stats']['mean']}, stdev: {analysis['score_stats']['stdev']}")
    print("\nBy sample bucket:")
    for bucket, stats in analysis['by_sample_bucket'].items():
        print(f"  {bucket}: mean={stats['mean']}, n={stats['count']}")

    print(f"\nResults saved to: {results_path}")
    if args.resume:
        print("(Resume mode - results were appended incrementally)")


if __name__ == "__main__":
    main()
