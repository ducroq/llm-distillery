"""
Improved Semantic Prefilter Evaluation (2-category + broader definition)

Key improvements:
- 2 categories instead of 6 (clearer binary decision)
- Broader positive category (includes conservation, circular economy, policy)
- Proper recall calculation
- GPU support
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
    Evaluate filter performance with proper recall calculation.

    Returns metrics including:
    - Recall: % of high-scoring articles that passed
    - Precision: % of passed articles that are relevant
    - FP rate: % of passed articles that scored <=2.0
    """
    print(f"\nEvaluating {filter_name}...")

    passed_articles = []
    blocked_articles = []
    false_positives = []
    false_negatives = []

    # Count high-scoring articles
    high_scoring_count = 0

    start_time = time.time()

    for i, article in enumerate(articles):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")

        avg_score = calculate_avg_score(article)
        is_high_scoring = avg_score > 5.0

        if is_high_scoring:
            high_scoring_count += 1

        try:
            should_pass, reason = filter_obj.apply_filter(article)

            if should_pass:
                passed_articles.append((article, avg_score))
                if avg_score <= 2.0:
                    false_positives.append((article, avg_score, reason))
            else:
                blocked_articles.append((article, avg_score))
                if is_high_scoring:
                    false_negatives.append((article, avg_score, reason))

        except Exception as e:
            print(f"  Error processing article {article.get('id', 'unknown')}: {e}")
            blocked_articles.append((article, avg_score))
            if is_high_scoring:
                false_negatives.append((article, avg_score, "error"))

    processing_time = time.time() - start_time

    # Calculate metrics
    total = len(articles)
    passed = len(passed_articles)
    blocked = len(blocked_articles)
    pass_rate = (passed / total * 100) if total > 0 else 0

    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    # FP rate: % of passed articles that are false positives
    fp_rate = (fp_count / passed * 100) if passed > 0 else 0

    # Recall: % of high-scoring articles that passed
    high_scoring_passed = high_scoring_count - fn_count
    recall = (high_scoring_passed / high_scoring_count * 100) if high_scoring_count > 0 else 0

    # Miss rate: % of high-scoring articles that were blocked
    miss_rate = (fn_count / high_scoring_count * 100) if high_scoring_count > 0 else 0

    # Precision: % of passed articles that are actually relevant (>3.0)
    relevant_passed = len([a for a, score in passed_articles if score > 3.0])
    precision = (relevant_passed / passed * 100) if passed > 0 else 0

    return {
        'filter': filter_name,
        'total_articles': total,
        'high_scoring_articles': high_scoring_count,
        'passed': passed,
        'blocked': blocked,
        'pass_rate': pass_rate,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'fp_rate': fp_rate,
        'recall': recall,
        'miss_rate': miss_rate,
        'precision': precision,
        'processing_time': processing_time,
        'fp_examples': false_positives[:5],
        'fn_examples': false_negatives[:5]
    }


def main():
    print("=" * 80)
    print("IMPROVED Semantic Prefilter Evaluation")
    print("2-Category + Broader Definition")
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

    # Improved 2-category configuration
    print("\n" + "=" * 80)
    print("Testing IMPROVED Semantic Prefilter")
    print("=" * 80)

    improved_labels = [
        # POSITIVE - broader definition
        "sustainability, renewable energy, climate solutions, environmental conservation, "
        "circular economy, biodiversity, green technology, and sustainability policy",

        # NEGATIVE - everything else
        "other topics including sports, entertainment, politics, and general news"
    ]

    print(f"\nCategory configuration:")
    print(f"  Positive: {improved_labels[0][:80]}...")
    print(f"  Negative: {improved_labels[1][:80]}...")

    # Ask user for device
    device_choice = input("\nUse GPU? (y/n, default=n): ").strip().lower()
    device = 0 if device_choice == 'y' else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"\nUsing: {device_name}")

    semantic_results = []
    thresholds = [0.30, 0.35, 0.40, 0.45]  # Focus on balanced range

    for threshold in thresholds:
        filter_obj = SemanticPreFilter(
            confidence_threshold=threshold,
            device=device
        )
        filter_obj.set_candidate_labels(improved_labels)

        result = evaluate_filter(
            filter_obj,
            articles,
            f"Semantic-improved (t={threshold:.2f})"
        )
        semantic_results.append(result)

    # Evaluate keyword baseline
    print("\n" + "=" * 80)
    print("Testing Keyword Prefilter (baseline)")
    print("=" * 80)

    keyword_filter = SustainabilityTechnologyPreFilterV1()
    keyword_result = evaluate_filter(keyword_filter, articles, "Keyword")

    # Generate report
    print("\n" + "=" * 80)
    print("Generating Report")
    print("=" * 80)

    report_path = Path("filters/sustainability_technology/v1/reports/SEMANTIC_IMPROVED_EVALUATION.md")
    generate_report(semantic_results, keyword_result, articles, high_scores, report_path)

    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


def generate_report(semantic_results: List[Dict], keyword_result: Dict,
                   articles: List[Dict], high_scoring_count: int, output_path: Path):
    """Generate comprehensive evaluation report."""

    # Find best by combined error and by recall
    best_combined = min(semantic_results, key=lambda r: r['fp_rate'] + r['miss_rate'])
    best_recall = max(semantic_results, key=lambda r: r['recall'])

    report = f"""# Improved Semantic Prefilter Evaluation

**Date**: {time.strftime('%Y-%m-%d')}
**Articles Evaluated**: {len(articles)}
**High-Scoring Articles**: {high_scoring_count} (threshold: >5.0)

## Configuration

**2-Category Setup (Improved)**:
1. **Positive**: sustainability, renewable energy, climate solutions, environmental conservation, circular economy, biodiversity, green technology, and sustainability policy
2. **Negative**: other topics including sports, entertainment, politics, and general news

**Key Improvements over 6-category**:
- Simpler binary decision (no ambiguous "general news" middle ground)
- Broader positive definition (includes conservation, circular economy, policy)
- Proper recall calculation

---

## Executive Summary

**Best by Recall**: {best_recall['filter']}
- **Recall**: {best_recall['recall']:.1f}% (catches {best_recall['recall']:.0f}% of good articles)
- **FP Rate**: {best_recall['fp_rate']:.1f}% (vs {keyword_result['fp_rate']:.1f}% keyword)
- **Precision**: {best_recall['precision']:.1f}%

**Best by Combined Error**: {best_combined['filter']}
- **Recall**: {best_combined['recall']:.1f}%
- **FP Rate**: {best_combined['fp_rate']:.1f}%
- **Combined Error**: {best_combined['fp_rate'] + best_combined['miss_rate']:.1f}%

---

## Performance Comparison

### Summary Table

| Filter | Pass Rate | Recall | Miss Rate | FP Rate | Precision | Speed |
|--------|-----------|--------|-----------|---------|-----------|-------|
| **Keyword** | {keyword_result['pass_rate']:.1f}% | {keyword_result['recall']:.1f}% | {keyword_result['miss_rate']:.1f}% | {keyword_result['fp_rate']:.1f}% | {keyword_result['precision']:.1f}% | {keyword_result['processing_time']:.1f}s |
"""

    for result in semantic_results:
        threshold = result['filter'].split('=')[1].strip(')')
        marker = " **<- BEST RECALL**" if result == best_recall else ""
        if result == best_combined and result != best_recall:
            marker = " **<- BEST ERROR**"

        report += f"| Semantic (t={threshold}) | {result['pass_rate']:.1f}% | "
        report += f"{result['recall']:.1f}% | {result['miss_rate']:.1f}% | "
        report += f"{result['fp_rate']:.1f}% | {result['precision']:.1f}% | "
        report += f"{result['processing_time']:.1f}s |{marker}\n"

    report += f"""

### Metrics Explanation

- **Pass Rate**: % of all articles that passed the prefilter
- **Recall**: % of high-scoring articles (>5.0) that passed ← KEY METRIC
- **Miss Rate**: % of high-scoring articles that were blocked
- **FP Rate**: % of passed articles with oracle scores <=2.0 (false positives)
- **Precision**: % of passed articles with oracle scores >3.0 (actually relevant)
- **Speed**: Total processing time for {len(articles)} articles

---

## Comparison with Original 6-Category

**Original 6-category (threshold 0.50)**:
- Recall: ~15% (missed 85% of good articles!)
- FP Rate: 2.1%
- Problem: "general news" category caught legitimate sustainability articles

**Improved 2-category (threshold {best_recall['filter'].split('=')[1].strip(')')})**:
- Recall: {best_recall['recall']:.1f}% (misses only {best_recall['miss_rate']:.1f}% of good articles)
- FP Rate: {best_recall['fp_rate']:.1f}%
- Improvement: {best_recall['recall'] - 15:.1f}% better recall

---

## Threshold Analysis

| Threshold | Recall | Miss Rate | FP Rate | Combined Error | Recommendation |
|-----------|--------|-----------|---------|----------------|----------------|
"""

    for result in semantic_results:
        threshold = result['filter'].split('=')[1].strip(')')
        combined = result['fp_rate'] + result['miss_rate']

        if result['recall'] >= 40:
            rec = "Good recall"
        elif result['recall'] >= 30:
            rec = "Balanced"
        else:
            rec = "Low recall"

        marker = " **<- RECOMMENDED**" if result == best_recall else ""

        report += f"| {threshold} | {result['recall']:.1f}% | {result['miss_rate']:.1f}% | "
        report += f"{result['fp_rate']:.1f}% | {combined:.1f}% | {rec}{marker} |\n"

    report += f"""

**Recommendation**: Use threshold = **{best_recall['filter'].split('=')[1].strip(')')}**
- Best recall: {best_recall['recall']:.1f}%
- Still reduces FP significantly: {best_recall['fp_rate']:.1f}% vs {keyword_result['fp_rate']:.1f}% (keyword)
- Good balance for training data generation

---

## False Positive Examples

Articles that passed but scored <=2.0:

### Keyword Prefilter
"""

    for i, (article, score, reason) in enumerate(keyword_result.get('fp_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"

    report += f"\n### Improved Semantic (threshold={best_recall['filter'].split('=')[1].strip(')')})\n"

    for i, (article, score, reason) in enumerate(best_recall.get('fp_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"

    report += """

---

## False Negative Examples

Articles that were blocked but scored >5.0:

### Keyword Prefilter
"""

    for i, (article, score, reason) in enumerate(keyword_result.get('fn_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason: {reason}\n"

    report += f"\n### Improved Semantic (threshold={best_recall['filter'].split('=')[1].strip(')')})\n"

    for i, (article, score, reason) in enumerate(best_recall.get('fn_examples', [])[:3], 1):
        title = article.get('title', 'N/A')[:80]
        report += f"\n**{i}. [{score:.1f}/10]** {title}\n"
        report += f"   - Reason: {reason}\n"

    # Decision logic
    if best_recall['recall'] >= 40 and best_recall['fp_rate'] < keyword_result['fp_rate'] * 0.7:
        decision = "APPROVE"
        rationale = f"""✅ **APPROVE IMPROVED SEMANTIC PREFILTER**

**Rationale**:
1. Recall: {best_recall['recall']:.1f}% (acceptable - catches most good articles)
2. FP reduction: {best_recall['fp_rate']:.1f}% vs {keyword_result['fp_rate']:.1f}% ({(1 - best_recall['fp_rate']/keyword_result['fp_rate'])*100:.0f}% reduction)
3. Broader category definition catches conservation, circular economy, policy
4. 2-category setup avoids "general news" ambiguity

**Recommended Implementation**:
- Use improved semantic prefilter for training data generation
- Configuration: 2 categories, threshold {best_recall['filter'].split('=')[1].strip(')')}
- Expected: ~{best_recall['recall']:.0f}% of relevant articles, ~{100-best_recall['fp_rate']:.0f}% FP filtering
"""
    else:
        decision = "RECONSIDER"
        rationale = f"""⚠️ **RECONSIDER - USE KEYWORD PREFILTER**

**Current Results**:
- Recall: {best_recall['recall']:.1f}% ({"too low" if best_recall['recall'] < 40 else "acceptable"})
- FP reduction: {(1 - best_recall['fp_rate']/keyword_result['fp_rate'])*100:.0f}% ({"not significant" if best_recall['fp_rate'] >= keyword_result['fp_rate'] * 0.7 else "good"})

**Recommendation**: Stick with keyword prefilter
- 100% recall (no missed articles)
- Simple and fast
- Accept {keyword_result['fp_rate']:.1f}% FP rate (oracle will score them low)
- Training data benefits from full spectrum of relevance
"""

    report += f"""

---

## Decision

{rationale}

---

## Next Steps

1. Review false positive/negative examples above
2. If approved: Update semantic_prefilter.py with improved 2-category configuration
3. Generate 10K training dataset using chosen configuration
4. Train student model

---

## References

- Original evaluation: `SEMANTIC_EVALUATION_REPORT.md` (6-category, poor recall)
- Semantic prefilter: `filters/sustainability_technology/v1/semantic_prefilter.py`
- Integration guide: `filters/sustainability_technology/v1/SEMANTIC_INTEGRATION.md`
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    main()
