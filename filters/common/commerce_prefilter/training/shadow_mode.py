"""
Commerce Prefilter - Shadow Mode Comparison

Runs the SLM commerce detector alongside existing regex patterns to compare decisions.
Use this to calibrate the threshold and validate model performance before production.

Usage:
    python -m filters.common.commerce_prefilter.training.shadow_mode \
        --input datasets/scored/sustainability_technology.jsonl \
        --output reports/commerce_prefilter_shadow.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Import existing regex patterns from sustainability_technology prefilter
COMMERCE_PATTERNS = [
    r'\b(Black Friday|Prime Day|Cyber Monday|holiday deal)\b',
    r'\b(discount code|coupon code|promo code)\b',
    r'\b(save \$\d+|percent off|\d+% off)\b',
    r'\bgift guide\b',
    r'\bGreen Deals\b',
    r'\$\d[\d,]*\s*(savings|discount|off)\b',
    r'\bdeals?\b.{0,30}(ending|expire|tonight|today only)',
    r'\b(exclusive|limited).{0,15}(low|price|deal|offer)\b',
    r'\b(new|all.time|record)\s+low.{0,10}(price|from|\$)',
    r'\bstarting\s+(at|from)\s+\$\d',
]


def regex_is_commerce(article: Dict) -> Tuple[bool, str]:
    """Check if article matches commerce regex patterns."""
    title = article.get('title', '')
    content = article.get('content', article.get('text', ''))[:2000]
    combined = f"{title} {content}".lower()

    for pattern in COMMERCE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return (True, pattern[:30])

    return (False, "")


def load_articles(input_path: Path) -> List[Dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return articles


def run_shadow_mode(
    articles: List[Dict],
    slm_detector,
    threshold: float = 0.7
) -> Dict:
    """
    Run shadow mode comparison.

    Args:
        articles: List of articles to process
        slm_detector: CommercePrefilterSLM instance
        threshold: Classification threshold

    Returns:
        Comparison report dict
    """
    results = {
        'total': len(articles),
        'agreement': 0,
        'disagreement': 0,
        'slm_only': [],  # SLM detects, regex doesn't
        'regex_only': [],  # Regex detects, SLM doesn't
        'both_detect': [],  # Both agree it's commerce
        'both_pass': [],  # Both agree it's journalism
        'scores': [],
    }

    for article in articles:
        # Regex detection
        regex_commerce, regex_pattern = regex_is_commerce(article)

        # SLM detection
        slm_result = slm_detector.is_commerce(article)
        slm_commerce = slm_result['is_commerce']
        slm_score = slm_result['score']

        results['scores'].append(slm_score)

        # Categorize
        title = article.get('title', '')[:80]

        if regex_commerce and slm_commerce:
            results['agreement'] += 1
            results['both_detect'].append({
                'title': title,
                'regex_pattern': regex_pattern,
                'slm_score': slm_score,
            })
        elif not regex_commerce and not slm_commerce:
            results['agreement'] += 1
            results['both_pass'].append({
                'title': title,
                'slm_score': slm_score,
            })
        elif slm_commerce and not regex_commerce:
            results['disagreement'] += 1
            results['slm_only'].append({
                'title': title,
                'slm_score': slm_score,
            })
        else:  # regex_commerce and not slm_commerce
            results['disagreement'] += 1
            results['regex_only'].append({
                'title': title,
                'regex_pattern': regex_pattern,
                'slm_score': slm_score,
            })

    # Summary stats
    results['agreement_rate'] = results['agreement'] / len(articles) if articles else 0
    results['slm_only_count'] = len(results['slm_only'])
    results['regex_only_count'] = len(results['regex_only'])

    # Score distribution
    scores = results['scores']
    if scores:
        results['score_stats'] = {
            'min': min(scores),
            'max': max(scores),
            'mean': sum(scores) / len(scores),
            'median': sorted(scores)[len(scores) // 2],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run commerce prefilter shadow mode comparison"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL file with articles"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("reports/commerce_prefilter_shadow.json"),
        help="Output JSON report"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="SLM classification threshold"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to SLM model directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of articles to process"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Shadow Mode Comparison")
    print("=" * 60)

    # Load articles
    print(f"\nLoading articles from {args.input}...")
    articles = load_articles(args.input)
    if args.limit:
        articles = articles[:args.limit]
    print(f"Loaded {len(articles)} articles")

    # Load SLM detector
    print("\nLoading SLM detector...")
    from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM
    slm_detector = CommercePrefilterSLM(
        model_path=args.model_path,
        threshold=args.threshold,
    )
    print(f"Threshold: {slm_detector.threshold}")

    # Run comparison
    print("\nRunning comparison...")
    results = run_shadow_mode(articles, slm_detector, args.threshold)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total articles: {results['total']}")
    print(f"Agreement: {results['agreement']} ({results['agreement_rate']:.1%})")
    print(f"Disagreement: {results['disagreement']}")
    print(f"  SLM only: {results['slm_only_count']} (SLM detects, regex doesn't)")
    print(f"  Regex only: {results['regex_only_count']} (Regex detects, SLM doesn't)")

    if 'score_stats' in results:
        stats = results['score_stats']
        print(f"\nScore distribution:")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")

    # Sample disagreements
    if results['slm_only']:
        print("\n--- SLM Only (sample) ---")
        for item in results['slm_only'][:5]:
            print(f"  [{item['slm_score']:.3f}] {item['title']}")

    if results['regex_only']:
        print("\n--- Regex Only (sample) ---")
        for item in results['regex_only'][:5]:
            print(f"  [{item['slm_score']:.3f}] {item['title']} (pattern: {item['regex_pattern']})")

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Remove large lists for JSON output (keep samples)
    report = {
        'config': {
            'input': str(args.input),
            'threshold': args.threshold,
            'total_articles': results['total'],
        },
        'summary': {
            'agreement': results['agreement'],
            'agreement_rate': results['agreement_rate'],
            'disagreement': results['disagreement'],
            'slm_only_count': results['slm_only_count'],
            'regex_only_count': results['regex_only_count'],
        },
        'score_stats': results.get('score_stats', {}),
        'samples': {
            'slm_only': results['slm_only'][:20],
            'regex_only': results['regex_only'][:20],
            'both_detect': results['both_detect'][:10],
        },
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
