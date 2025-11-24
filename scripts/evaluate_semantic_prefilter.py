"""
Evaluate semantic prefilter performance on 1000 scored articles.

Compares:
- Semantic prefilter (multiple confidence thresholds)
- Keyword prefilter (baseline)

Metrics:
- False Positive Rate: % of low-scoring articles (≤2.0) that passed
- False Negative Rate: % of high-scoring articles (>5.0) that were blocked
- Pass Rate: % of articles that passed the filter
- Precision: % of passed articles that are actually relevant (>3.0)
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
    """
    Evaluate filter performance on articles.

    Returns:
        {
            'filter': str,
            'total_articles': int,
            'passed': int,
            'blocked': int,
            'pass_rate': float,
            'false_positives': int,  # Passed but scored ≤2.0
            'false_negatives': int,  # Blocked but scored >5.0
            'fp_rate': float,
            'fn_rate': float,
            'precision': float,      # % of passed that are relevant (>3.0)
            'processing_time': float
        }
    """
    print(f"\nEvaluating {filter_name}...")

    passed_articles = []
    blocked_articles = []
    false_positives = []  # Low scores that passed
    false_negatives = []  # High scores that were blocked

    start_time = time.time()

    for i, article in enumerate(articles):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")

        # Calculate oracle score
        avg_score = calculate_avg_score(article)

        # Apply filter
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
            # Count as blocked on error
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

    # Precision: % of passed articles that are actually relevant (>3.0)
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
        'fp_examples': false_positives[:5],  # Top 5 examples
        'fn_examples': false_negatives[:5]
    }


def main():
    print("=" * 80)
    print("Semantic Prefilter Evaluation")
    print("=" * 80)

    # Load scored articles
    data_dir = Path("sandbox/semantic_evaluation_1k/sustainability_technology")
    print(f"\nLoading scored articles from: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run batch_scorer first to generate scored articles.")
        return

    articles = load_scored_articles(data_dir)
    print(f"Loaded {len(articles)} scored articles")

    if len(articles) == 0:
        print("ERROR: No articles found. Is scoring still running?")
        return

    # Score distribution
    scores = [calculate_avg_score(a) for a in articles]
    high_scores = len([s for s in scores if s > 5.0])
    medium_scores = len([s for s in scores if 3.0 < s <= 5.0])
    low_scores = len([s for s in scores if s <= 3.0])

    print(f"\nScore distribution:")
    print(f"  High (>5.0):     {high_scores:4d} ({high_scores/len(scores)*100:.1f}%)")
    print(f"  Medium (3.0-5.0): {medium_scores:4d} ({medium_scores/len(scores)*100:.1f}%)")
    print(f"  Low (<=3.0):      {low_scores:4d} ({low_scores/len(scores)*100:.1f}%)")

    # Evaluate semantic prefilter with different thresholds
    print("\n" + "=" * 80)
    print("Testing Semantic Prefilter (multiple thresholds)")
    print("=" * 80)

    semantic_results = []
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for threshold in thresholds:
        filter_obj = SemanticPreFilter(
            confidence_threshold=threshold,
            device=-1  # CPU
        )
        result = evaluate_filter(
            filter_obj,
            articles,
            f"Semantic (threshold={threshold:.2f})"
        )
        semantic_results.append(result)

    # Evaluate keyword prefilter (baseline)
    print("\n" + "=" * 80)
    print("Testing Keyword Prefilter (baseline)")
    print("=" * 80)

    keyword_filter = SustainabilityTechnologyPreFilterV1()
    keyword_result = evaluate_filter(keyword_filter, articles, "Keyword")

    # Generate report
    print("\n" + "=" * 80)
    print("Generating Report")
    print("=" * 80)

    report_path = Path("filters/sustainability_technology/v1/reports/SEMANTIC_EVALUATION_REPORT.md")
    generate_report(semantic_results, keyword_result, articles, report_path)

    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


def generate_report(semantic_results: List[Dict], keyword_result: Dict,
                   articles: List[Dict], output_path: Path):
    """Generate comprehensive evaluation report."""

    # Find best threshold
    best_semantic = min(semantic_results, key=lambda r: r['fp_rate'] + r['fn_rate'])

    report = f"""# Semantic Prefilter Evaluation Report

**Date**: {time.strftime('%Y-%m-%d')}
**Articles Evaluated**: {len(articles)}
**Oracle**: gemini-flash

## Executive Summary

{'✅ **SEMANTIC PREFILTER RECOMMENDED**' if best_semantic['fp_rate'] < keyword_result['fp_rate'] else '⚠️ **FURTHER TUNING NEEDED**'}

**Best Configuration**: Semantic filter with confidence threshold = {best_semantic['filter'].split('=')[1].strip(')')}
- False Positive Rate: {best_semantic['fp_rate']:.1f}% (vs {keyword_result['fp_rate']:.1f}% keyword baseline)
- False Negative Rate: {best_semantic['fn_rate']:.1f}% (vs {keyword_result['fn_rate']:.1f}% keyword baseline)
- Precision: {best_semantic['precision']:.1f}% (vs {keyword_result['precision']:.1f}% keyword baseline)

---

## Performance Comparison

### Summary Table

| Filter | Pass Rate | FP Rate | FN Rate | Precision | Speed |
|--------|-----------|---------|---------|-----------|-------|
"""

    # Add keyword baseline
    report += f"| Keyword (baseline) | {keyword_result['pass_rate']:.1f}% | "
    report += f"{keyword_result['fp_rate']:.1f}% | {keyword_result['fn_rate']:.1f}% | "
    report += f"{keyword_result['precision']:.1f}% | {keyword_result['processing_time']:.1f}s |\n"

    # Add semantic results
    for result in semantic_results:
        threshold = result['filter'].split('=')[1].strip(')')
        marker = " **←**" if result == best_semantic else ""
        report += f"| Semantic (t={threshold}) | {result['pass_rate']:.1f}% | "
        report += f"{result['fp_rate']:.1f}% | {result['fn_rate']:.1f}% | "
        report += f"{result['precision']:.1f}% | {result['processing_time']:.1f}s |{marker}\n"

    report += """

### Metrics Explanation

- **Pass Rate**: % of articles that passed the prefilter
- **FP Rate**: % of passed articles with oracle scores ≤2.0 (false positives)
- **FN Rate**: % of blocked articles with oracle scores >5.0 (false negatives)
- **Precision**: % of passed articles with oracle scores >3.0 (actually relevant)
- **Speed**: Total processing time for 1000 articles

---

## Threshold Analysis

### False Positive vs False Negative Tradeoff

"""

    report += "| Threshold | FP Rate | FN Rate | Combined Error | Recommendation |\n"
    report += "|-----------|---------|---------|----------------|----------------|\n"

    for result in semantic_results:
        threshold = result['filter'].split('=')[1].strip(')')
        combined_error = result['fp_rate'] + result['fn_rate']
        marker = " **← BEST**" if result == best_semantic else ""

        if float(threshold) <= 0.30:
            rec = "Too permissive"
        elif float(threshold) >= 0.45:
            rec = "Too restrictive"
        else:
            rec = "Balanced"

        report += f"| {threshold} | {result['fp_rate']:.1f}% | {result['fn_rate']:.1f}% | "
        report += f"{combined_error:.1f}% | {rec}{marker} |\n"

    report += f"""

**Recommendation**: Use threshold = **{best_semantic['filter'].split('=')[1].strip(')')}**
- Lowest combined error rate ({best_semantic['fp_rate'] + best_semantic['fn_rate']:.1f}%)
- Good balance between precision and recall
- {'Significantly better' if best_semantic['fp_rate'] < keyword_result['fp_rate'] * 0.7 else 'Comparable'} to keyword baseline

---

## False Positive Examples

Articles that passed the filter but scored ≤2.0 (should have been blocked):

### Keyword Prefilter
"""

    for i, (article, score, reason) in enumerate(keyword_result.get('fp_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason passed: {reason}\n"

    report += f"\n### Semantic Prefilter (threshold={best_semantic['filter'].split('=')[1].strip(')')})\n"

    for i, (article, score, reason) in enumerate(best_semantic.get('fp_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason passed: {reason}\n"

    report += """

---

## False Negative Examples

Articles that were blocked but scored >5.0 (should have passed):

### Keyword Prefilter
"""

    for i, (article, score, reason) in enumerate(keyword_result.get('fn_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason blocked: {reason}\n"

    report += f"\n### Semantic Prefilter (threshold={best_semantic['filter'].split('=')[1].strip(')')})\n"

    for i, (article, score, reason) in enumerate(best_semantic.get('fn_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason blocked: {reason}\n"

    report += """

---

## Decision

"""

    if best_semantic['fp_rate'] < keyword_result['fp_rate'] * 0.8:
        report += f"""✅ **APPROVE SEMANTIC PREFILTER**

**Rationale**:
1. False positive rate: {best_semantic['fp_rate']:.1f}% vs {keyword_result['fp_rate']:.1f}% (keyword)
2. Semantic understanding prevents 'oil in turmoil' type errors
3. Processing time acceptable for batch processing ({best_semantic['processing_time']/len(articles)*1000:.1f}ms per article)

**Recommended Implementation**:
- Use semantic prefilter for 10K training data generation
- Confidence threshold: {best_semantic['filter'].split('=')[1].strip(')')}
- Expected processing time: ~{best_semantic['processing_time']/len(articles)*10000/3600:.1f} hours for 10K articles
- Expected cost savings: ~${(keyword_result['fp_rate'] - best_semantic['fp_rate'])/100 * 10000 * 0.0015:.2f} (fewer false positives to oracle)
"""
    else:
        report += f"""⚠️ **FURTHER TUNING NEEDED**

**Current Results**:
- Semantic FP rate: {best_semantic['fp_rate']:.1f}%
- Keyword FP rate: {keyword_result['fp_rate']:.1f}%
- Improvement: {keyword_result['fp_rate'] - best_semantic['fp_rate']:.1f}% (not significant)

**Recommendations**:
1. Refine candidate labels for better discrimination
2. Test with more specific positive category definition
3. Consider hybrid approach (keyword + semantic for ambiguous cases)
4. Collect more edge case examples for label tuning
"""

    report += """

---

## Next Steps

1. Review false positive/negative examples above
2. If approved, integrate semantic prefilter into batch_scorer
3. Generate 10K training dataset using approved configuration
4. Train student model on filtered data

---

## References

- Semantic prefilter implementation: `filters/sustainability_technology/v1/semantic_prefilter.py`
- Integration guide: `filters/sustainability_technology/v1/SEMANTIC_INTEGRATION.md`
- Raw evaluation data: `sandbox/semantic_evaluation_1k/`
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    main()
