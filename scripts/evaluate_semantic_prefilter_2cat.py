"""
Evaluate semantic prefilter with simplified 2-category setup.

This tests whether simpler categorization (sustainability vs other) performs
as well as or better than the more complex 6-category setup.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from filters.sustainability_technology.v1.semantic_prefilter import SemanticPreFilter
from filters.sustainability_technology.v1.prefilter import SustainabilityTechnologyPreFilterV1


def load_scored_articles(data_dir: Path) -> List[Dict]:
    """Load all scored articles from JSONL files."""
    articles = []
    for batch_file in sorted(data_dir.glob("scored_batch_*.jsonl")):
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                articles.append(json.loads(line))
    return articles


def calculate_avg_score(article: Dict) -> float:
    """Calculate average score across all dimensions."""
    analysis = article['sustainability_technology_analysis']
    dimensions = [
        'technology_readiness_level',
        'technical_performance',
        'economic_competitiveness',
        'life_cycle_environmental_impact',
        'social_equity_impact',
        'governance_systemic_impact'
    ]
    scores = [analysis[d]['score'] for d in dimensions]
    return sum(scores) / len(scores)


def evaluate_filter(filter_obj, articles: List[Dict], filter_name: str) -> Dict:
    """Evaluate filter performance on articles."""
    print(f"\nEvaluating {filter_name}...")

    passed_articles = []
    blocked_articles = []
    false_positives = []
    false_negatives = []

    start_time = time.time()

    for i, article in enumerate(articles):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")

        avg_score = calculate_avg_score(article)

        try:
            should_pass, reason = filter_obj.apply_filter(article)

            if should_pass:
                passed_articles.append((article, avg_score))
                if avg_score <= 2.0:
                    false_positives.append((article, avg_score, reason))
            else:
                blocked_articles.append((article, avg_score))
                if avg_score > 5.0:
                    false_negatives.append((article, avg_score, reason))

        except Exception as e:
            print(f"  Error processing article {article.get('id', 'unknown')}: {e}")
            blocked_articles.append((article, avg_score))

    processing_time = time.time() - start_time

    # Calculate metrics
    total = len(articles)
    passed = len(passed_articles)
    blocked = len(blocked_articles)
    pass_rate = (passed / total * 100) if total > 0 else 0

    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    fp_rate = (fp_count / passed * 100) if passed > 0 else 0
    fn_rate = (fn_count / blocked * 100) if blocked > 0 else 0

    relevant_passed = len([a for a, score in passed_articles if score > 3.0])
    precision = (relevant_passed / passed * 100) if passed > 0 else 0

    return {
        'filter': filter_name,
        'total_articles': total,
        'passed': passed,
        'blocked': blocked,
        'pass_rate': pass_rate,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'precision': precision,
        'processing_time': processing_time,
        'fp_examples': false_positives[:5],
        'fn_examples': false_negatives[:5]
    }


def main():
    print("=" * 80)
    print("Semantic Prefilter Evaluation - 2 Category Configuration")
    print("=" * 80)

    # Load scored articles
    data_dir = Path("sandbox/semantic_evaluation_1k/sustainability_technology")
    print(f"\nLoading scored articles from: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    articles = load_scored_articles(data_dir)
    print(f"Loaded {len(articles)} scored articles")

    if len(articles) == 0:
        print("ERROR: No articles found.")
        return

    # Score distribution
    scores = [calculate_avg_score(a) for a in articles]
    high_scores = len([s for s in scores if s > 5.0])
    medium_scores = len([s for s in scores if 3.0 < s <= 5.0])
    low_scores = len([s for s in scores if s <= 3.0])

    print(f"\nScore distribution:")
    print(f"  High (>5.0):      {high_scores:4d} ({high_scores/len(scores)*100:.1f}%)")
    print(f"  Medium (3.0-5.0): {medium_scores:4d} ({medium_scores/len(scores)*100:.1f}%)")
    print(f"  Low (<=3.0):      {low_scores:4d} ({low_scores/len(scores)*100:.1f}%)")

    # Test with 2 categories
    print("\n" + "=" * 80)
    print("Testing 2-Category Semantic Prefilter")
    print("=" * 80)

    two_cat_labels = [
        "sustainability technology, renewable energy, climate solutions, and environmental innovation",
        "other topics including sports, entertainment, politics, and general news"
    ]

    semantic_results = []
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for threshold in thresholds:
        filter_obj = SemanticPreFilter(
            confidence_threshold=threshold,
            device=-1
        )
        filter_obj.set_candidate_labels(two_cat_labels)

        result = evaluate_filter(
            filter_obj,
            articles,
            f"Semantic-2cat (threshold={threshold:.2f})"
        )
        semantic_results.append(result)

    # Generate comparison report
    print("\n" + "=" * 80)
    print("Generating Comparison Report")
    print("=" * 80)

    report_path = Path("filters/sustainability_technology/v1/reports/SEMANTIC_2CAT_EVALUATION.md")
    generate_report(semantic_results, articles, report_path)

    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 80)
    print("2-Category Evaluation Complete!")
    print("=" * 80)


def generate_report(semantic_results: List[Dict], articles: List[Dict], output_path: Path):
    """Generate evaluation report for 2-category configuration."""

    best_semantic = min(semantic_results, key=lambda r: r['fp_rate'] + r['fn_rate'])

    report = f"""# Semantic Prefilter Evaluation - 2 Category Configuration

**Date**: {time.strftime('%Y-%m-%d')}
**Articles Evaluated**: {len(articles)}
**Configuration**: 2 categories (sustainability vs other)

## Category Setup

**Simplified 2-category approach**:
1. "sustainability technology, renewable energy, climate solutions, and environmental innovation"
2. "other topics including sports, entertainment, politics, and general news"

## Rationale

Simpler categorization should provide:
- Clearer decision boundary (sustainability vs not-sustainability)
- Easier threshold tuning
- Less arbitrary category selection
- Potentially better performance (fewer categories to confuse the model)

---

## Results Summary

### Best Threshold: {best_semantic['filter'].split('=')[1].strip(')')}

- **False Positive Rate**: {best_semantic['fp_rate']:.1f}%
- **False Negative Rate**: {best_semantic['fn_rate']:.1f}%
- **Precision**: {best_semantic['precision']:.1f}%
- **Pass Rate**: {best_semantic['pass_rate']:.1f}%

### Performance Table

| Threshold | Pass Rate | FP Rate | FN Rate | Precision | Combined Error | Speed |
|-----------|-----------|---------|---------|-----------|----------------|-------|
"""

    for result in semantic_results:
        threshold = result['filter'].split('=')[1].strip(')')
        combined_error = result['fp_rate'] + result['fn_rate']
        marker = " **<- BEST**" if result == best_semantic else ""
        report += f"| {threshold} | {result['pass_rate']:.1f}% | "
        report += f"{result['fp_rate']:.1f}% | {result['fn_rate']:.1f}% | "
        report += f"{result['precision']:.1f}% | {combined_error:.1f}% | "
        report += f"{result['processing_time']:.1f}s |{marker}\n"

    report += f"""

---

## Comparison with 6-Category Setup

See `SEMANTIC_EVALUATION_REPORT.md` for 6-category results.

**Key Question**: Does 2-category perform as well as 6-category?
- If yes: Use simpler 2-category (less arbitrary, easier to explain)
- If 6-category significantly better: Keep 6 categories

---

## False Positive Examples (Top 3)

Articles that passed but scored <=2.0:

"""

    for i, (article, score, reason) in enumerate(best_semantic.get('fp_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason: {reason}\n"

    report += f"""

---

## False Negative Examples (Top 3)

Articles that were blocked but scored >5.0:

"""

    for i, (article, score, reason) in enumerate(best_semantic.get('fn_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason: {reason}\n"

    report += """

---

## Recommendation

Compare this report with the 6-category results to determine:
1. Which configuration has lower combined error rate?
2. Is the performance difference significant?
3. Which is easier to explain and maintain?

**Default recommendation**: Use 2-category unless 6-category shows significant improvement (>5% lower combined error).
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    main()
