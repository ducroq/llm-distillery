"""
Analyze uplifting filter calibration sample systematically.
Following Prompt Calibration Agent template criteria.
"""

import json
import statistics
from pathlib import Path

def load_calibration_sample(filepath):
    """Load JSONL calibration data."""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles

def calculate_variance(scores):
    """Calculate variance of dimensional scores."""
    if len(scores) < 2:
        return 0
    return statistics.variance(scores)

def analyze_article(article):
    """Extract uplifting analysis from article."""
    analysis = article.get('uplifting_analysis', {})
    dims = analysis.get('dimensions', {})

    return {
        'id': article['id'],
        'title': article['title'],
        'overall_score': analysis.get('overall_uplift_score', 0),
        'tier': analysis.get('tier', 'unknown'),
        'content_type': analysis.get('content_type', 'unknown'),
        'reasoning': analysis.get('reasoning', ''),
        'dimensions': dims,
        'variance': calculate_variance(list(dims.values())) if dims else 0,
        'sentiment_category': article.get('metadata', {}).get('sentiment_category', 'unknown')
    }

def classify_expected_category(article):
    """
    Manually classify what we expect based on content.
    This is human judgment to validate oracle performance.
    """
    title = article['title'].lower()
    content = article['content'][:500].lower()  # First 500 chars

    # Doom/negative framing indicators
    doom_indicators = [
        'crash', 'faces', 'slashed', 'shutdown', 'dry', 'trap',
        'deadly', 'crisis'
    ]

    # Uplifting/progress indicators
    progress_indicators = [
        'comeback', 'restored', 'solutions', 'blueprint', 'warriors',
        'revival', 'alternatives', 'helps', 'benefits'
    ]

    # Corporate/tech without clear benefit
    corporate_indicators = [
        'api gateway', 'budgeting app', 'rust and go', 'programming',
        'quietly rewriting', 'billion'
    ]

    # Count matches
    doom_score = sum(1 for ind in doom_indicators if ind in title or ind in content)
    progress_score = sum(1 for ind in progress_indicators if ind in title or ind in content)
    corporate_score = sum(1 for ind in corporate_indicators if ind in title or ind in content)

    # Classify
    if corporate_score >= 2:
        return 'off_topic_corporate'
    elif doom_score > progress_score:
        return 'doom_framed'
    elif progress_score > 0:
        return 'uplifting'
    else:
        return 'neutral'

def main():
    filepath = Path(r'C:\local_dev\llm-distillery\datasets\working\uplifting_calibration_labeled.jsonl')
    articles = load_calibration_sample(filepath)

    print("=" * 80)
    print("UPLIFTING FILTER CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal articles: {len(articles)}")
    print(f"Sample size: SMALL (target: 50, actual: {len(articles)})")
    print()

    # Analyze each article
    analyzed = []
    for article in articles:
        analyzed.append({
            **analyze_article(article),
            'expected_category': classify_expected_category(article)
        })

    # CRITICAL METRIC 1: Off-Topic Rejection Rate
    print("=" * 80)
    print("CRITICAL METRIC 1: OFF-TOPIC REJECTION RATE")
    print("=" * 80)

    off_topic = [a for a in analyzed if a['expected_category'] in ['off_topic_corporate', 'doom_framed']]
    off_topic_high_scores = [a for a in off_topic if a['overall_score'] >= 5.0]
    off_topic_very_high = [a for a in off_topic if a['overall_score'] >= 7.0]

    print(f"\nOff-topic articles identified: {len(off_topic)}")
    print(f"Scored >= 5.0 (FALSE POSITIVES): {len(off_topic_high_scores)}")
    print(f"Scored >= 7.0 (SEVERE FALSE POSITIVES): {len(off_topic_very_high)}")

    if len(off_topic) > 0:
        fp_rate = len(off_topic_high_scores) / len(off_topic) * 100
        severe_fp_rate = len(off_topic_very_high) / len(off_topic) * 100
        print(f"\nFalse positive rate: {fp_rate:.1f}% (target: <10%)")
        print(f"Severe false positive rate: {severe_fp_rate:.1f}% (target: <5%)")

        if fp_rate > 20:
            print("\nFAIL: >20% false positive rate")
        elif fp_rate > 10:
            print("\nREVIEW: 10-20% false positive rate")
        else:
            print("\nPASS: <10% false positive rate")

    if off_topic_high_scores:
        print("\n--- FALSE POSITIVE EXAMPLES ---")
        for a in off_topic_high_scores[:5]:
            print(f"\n{a['title'][:80]}")
            print(f"  Score: {a['overall_score']:.2f} | Tier: {a['tier']}")
            print(f"  Expected: {a['expected_category']}")
            print(f"  Oracle reasoning: {a['reasoning'][:150]}...")

    # CRITICAL METRIC 2: On-Topic Recognition Rate
    print("\n" + "=" * 80)
    print("CRITICAL METRIC 2: ON-TOPIC RECOGNITION RATE")
    print("=" * 80)

    on_topic = [a for a in analyzed if a['expected_category'] == 'uplifting']
    on_topic_low_scores = [a for a in on_topic if a['overall_score'] < 5.0]
    on_topic_high_scores = [a for a in on_topic if a['overall_score'] >= 7.0]

    print(f"\nOn-topic articles identified: {len(on_topic)}")
    print(f"Scored < 5.0 (FALSE NEGATIVES): {len(on_topic_low_scores)}")
    print(f"Scored >= 7.0 (correctly recognized): {len(on_topic_high_scores)}")

    if len(on_topic) > 0:
        fn_rate = len(on_topic_low_scores) / len(on_topic) * 100
        print(f"\nFalse negative rate: {fn_rate:.1f}% (target: <20%)")

        if fn_rate > 30:
            print("FAIL: >30% false negative rate")
        elif fn_rate > 20:
            print("REVIEW: 20-30% false negative rate")
        else:
            print("PASS: <20% false negative rate")

    if on_topic_low_scores:
        print("\n--- FALSE NEGATIVE EXAMPLES ---")
        for a in on_topic_low_scores[:3]:
            print(f"\n{a['title'][:80]}")
            print(f"  Score: {a['overall_score']:.2f} | Tier: {a['tier']}")
            print(f"  Oracle reasoning: {a['reasoning'][:150]}...")

    # CRITICAL METRIC 3: Dimensional Consistency
    print("\n" + "=" * 80)
    print("CRITICAL METRIC 3: DIMENSIONAL CONSISTENCY")
    print("=" * 80)

    variances = [a['variance'] for a in analyzed if a['variance'] > 0]
    avg_variance = statistics.mean(variances) if variances else 0
    low_variance_articles = [a for a in analyzed if a['variance'] < 0.5 and a['variance'] > 0]

    print(f"\nAverage dimensional variance: {avg_variance:.2f} (target: >1.0)")
    print(f"Articles with variance < 0.5: {len(low_variance_articles)} ({len(low_variance_articles)/len(analyzed)*100:.1f}%)")

    if avg_variance > 1.0:
        print("PASS: Good dimensional differentiation")
    elif avg_variance > 0.5:
        print("REVIEW: Moderate dimensional differentiation")
    else:
        print("FAIL: Poor dimensional differentiation")

    if low_variance_articles:
        print("\n--- LOW VARIANCE EXAMPLES ---")
        for a in low_variance_articles[:3]:
            dims = a['dimensions']
            print(f"\n{a['title'][:80]}")
            print(f"  Dimensions: {list(dims.values())}")
            print(f"  Variance: {a['variance']:.2f}")

    # SCORE DISTRIBUTION
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)

    score_ranges = {
        '0-2': [a for a in analyzed if 0 <= a['overall_score'] < 2],
        '3-4': [a for a in analyzed if 3 <= a['overall_score'] < 5],
        '5-6': [a for a in analyzed if 5 <= a['overall_score'] < 7],
        '7-8': [a for a in analyzed if 7 <= a['overall_score'] < 9],
        '9-10': [a for a in analyzed if 9 <= a['overall_score'] <= 10]
    }

    for range_label, articles_in_range in score_ranges.items():
        print(f"{range_label}: {len(articles_in_range)} articles")

    # TIER DISTRIBUTION
    print("\n--- TIER DISTRIBUTION ---")
    tiers = {}
    for a in analyzed:
        tier = a['tier']
        tiers[tier] = tiers.get(tier, 0) + 1

    for tier, count in sorted(tiers.items()):
        print(f"{tier}: {count} articles")

    # DETAILED ARTICLE LIST
    print("\n" + "=" * 80)
    print("DETAILED ARTICLE ANALYSIS")
    print("=" * 80)

    for i, a in enumerate(analyzed, 1):
        print(f"\n{i}. {a['title'][:80]}")
        print(f"   Overall: {a['overall_score']:.2f} | Tier: {a['tier']} | Type: {a['content_type']}")
        print(f"   Expected: {a['expected_category']} | Variance: {a['variance']:.2f}")
        dims = a['dimensions']
        print(f"   Dimensions: agency={dims.get('agency', 0)}, progress={dims.get('progress', 0)}, "
              f"collective={dims.get('collective_benefit', 0)}, connection={dims.get('connection', 0)}")
        print(f"   Reasoning: {a['reasoning'][:150]}...")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSAMPLE SIZE WARNING: Only {len(articles)} articles (target: 50)")
    print("Results may not be statistically reliable.")
    print("\nRecommendation: Increase sample size before making PASS/FAIL decision.")

if __name__ == '__main__':
    main()
