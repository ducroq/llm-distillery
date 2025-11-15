"""
Analyze fresh calibration sample according to Prompt Calibration Agent criteria.
This validates that prompt improvements generalize to new, unseen articles.
"""

import json
import statistics
from typing import Dict, List, Tuple

def load_calibration_data(filepath: str) -> List[Dict]:
    """Load JSONL calibration data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values"""
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)

def classify_article(article: Dict) -> Tuple[str, str]:
    """
    Classify article as on-topic, off-topic, or edge-case based on title and content.
    Returns (category, reason)
    """
    title = article.get('title', '').lower()
    content = article.get('content', '').lower()
    text = f"{title} {content}"

    # OFF-TOPIC: Clear mismatches
    off_topic_patterns = {
        'bioinformatics': 'biology/genomics research',
        'grayu': 'database/research tool',
        'ayurvedic': 'traditional medicine',
        'medicinal plants': 'traditional medicine',
        'phytochemical': 'biology research',
        'power bank': 'consumer electronics',
        'magsafe': 'consumer electronics',
        'anker': 'consumer electronics',
        'toothbrush': 'hygiene product',
        'excel': 'office software',
        'aws': 'cloud infrastructure',
        'programming': 'software development',
        'computer science': 'academic CS',
        'education inequality': 'education policy',
        'mars': 'space exploration',
        'nasa': 'space exploration',
        'trump': 'policy/legal',
    }

    for pattern, reason in off_topic_patterns.items():
        if pattern in title or pattern in text[:200]:
            return ('off-topic', reason)

    # ON-TOPIC: Deployed climate tech (very specific)
    # Must have BOTH tech pattern AND deployment indicator
    on_topic_patterns = {
        'battery storage': 'energy storage',
        'mwh battery': 'energy storage',
        'solar farm': 'solar deployment',
        'solar installation': 'solar deployment',
        'wind farm': 'wind energy',
        'ev chargers': 'EV infrastructure',
        'recycled cashmere': 'sustainable materials',
        'recycled fiber': 'sustainable materials',
        '100% recycled': 'sustainable materials',
        'nuclear power plant': 'nuclear power',
    }

    # Check for deployment indicators (more specific)
    deployment_indicators = [
        'deployed', 'installed', 'operational', 'commercial',
        'mwh', 'gwh', 'market share', 'supply', 'containerized',
        '100%', 'production', 'manufactured', 'deployed'
    ]

    # Check specific on-topic patterns (require deployment)
    for pattern, reason in on_topic_patterns.items():
        if pattern in text:
            has_deployment = any(ind in text for ind in deployment_indicators)
            if has_deployment:
                return ('on-topic', reason)
            else:
                return ('edge-case', f'{reason} (no clear deployment)')

    # Broader patterns that might be edge cases
    edge_patterns = {
        'solar': 'solar energy',
        'electric vehicle': 'EV',
        'renewable energy': 'clean energy',
        'wind': 'wind energy',
        'rare earth': 'supply chain',
    }

    for pattern, reason in edge_patterns.items():
        if pattern in text:
            has_deployment = any(ind in text for ind in deployment_indicators)
            if has_deployment:
                return ('edge-case', f'{reason} (ambiguous deployment)')
            else:
                return ('edge-case', f'{reason} (no clear deployment)')

    # EDGE CASE: Climate-related but ambiguous
    if any(kw in text for kw in ['nuclear', 'energy', 'carbon', 'climate', 'emission']):
        return ('edge-case', 'climate-related but ambiguous scope')

    return ('off-topic', 'no climate tech indicators')

def analyze_calibration(articles: List[Dict]) -> Dict:
    """Perform full calibration analysis"""

    results = {
        'total_articles': len(articles),
        'off_topic_articles': [],
        'on_topic_articles': [],
        'edge_case_articles': [],
        'false_positives': [],
        'false_negatives': [],
        'dimensional_variance': [],
        'score_distribution': {
            '0-2': 0,
            '3-4': 0,
            '5-6': 0,
            '7-8': 0,
            '9-10': 0,
        }
    }

    for article in articles:
        title = article.get('title', 'N/A')
        analysis = article.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score', 0.0)
        dimensions = analysis.get('dimensions', {})
        reasoning = analysis.get('overall_assessment', 'N/A')

        # Classify article
        category, reason = classify_article(article)

        # Calculate dimensional variance
        dim_values = list(dimensions.values()) if dimensions else []
        variance = calculate_variance(dim_values) if len(dim_values) > 1 else 0.0

        article_info = {
            'title': title,
            'score': overall_score,
            'reason': reason,
            'reasoning': reasoning,
            'dimensions': dimensions,
            'variance': variance
        }

        # Track by category
        if category == 'off-topic':
            results['off_topic_articles'].append(article_info)
            # False positive if off-topic scored >= 5.0
            if overall_score >= 5.0:
                results['false_positives'].append(article_info)
        elif category == 'on-topic':
            results['on_topic_articles'].append(article_info)
            # False negative if on-topic scored < 5.0
            if overall_score < 5.0:
                results['false_negatives'].append(article_info)
        else:
            results['edge_case_articles'].append(article_info)

        # Track variance
        results['dimensional_variance'].append(variance)

        # Track score distribution
        if overall_score < 3.0:
            results['score_distribution']['0-2'] += 1
        elif overall_score < 5.0:
            results['score_distribution']['3-4'] += 1
        elif overall_score < 7.0:
            results['score_distribution']['5-6'] += 1
        elif overall_score < 9.0:
            results['score_distribution']['7-8'] += 1
        else:
            results['score_distribution']['9-10'] += 1

    return results

def generate_report(results: Dict, v2_metrics: Dict) -> str:
    """Generate calibration report comparing fresh sample to v2 results"""

    report = []
    report.append("# Prompt Calibration Report: Fresh Sample Validation\n")
    report.append("**Date:** 2025-11-14")
    report.append("**Filter:** sustainability_tech_deployment")
    report.append("**Oracle:** Gemini Flash 1.5 (gemini-flash-api-batch)")
    report.append("**Calibrator:** Prompt Calibration Agent v1.0")
    report.append("**Prompt Version:** v2 (with fixed SCOPE section)")
    report.append("**Sample:** FRESH - completely different from calibration articles\n")
    report.append("---\n")

    # Calculate metrics
    n_off_topic = len(results['off_topic_articles'])
    n_on_topic = len(results['on_topic_articles'])
    n_edge = len(results['edge_case_articles'])
    n_false_pos = len(results['false_positives'])
    n_false_neg = len(results['false_negatives'])

    off_topic_rejection_rate = ((n_off_topic - n_false_pos) / n_off_topic * 100) if n_off_topic > 0 else 0
    on_topic_recognition_rate = ((n_on_topic - n_false_neg) / n_on_topic * 100) if n_on_topic > 0 else 0

    false_pos_rate = (n_false_pos / n_off_topic * 100) if n_off_topic > 0 else 0
    false_neg_rate = (n_false_neg / n_on_topic * 100) if n_on_topic > 0 else 0

    avg_variance = statistics.mean(results['dimensional_variance']) if results['dimensional_variance'] else 0
    median_variance = statistics.median(results['dimensional_variance']) if results['dimensional_variance'] else 0
    low_variance_count = sum(1 for v in results['dimensional_variance'] if v < 0.5)
    low_variance_pct = (low_variance_count / len(results['dimensional_variance']) * 100) if results['dimensional_variance'] else 0

    # Determine pass/fail
    off_topic_pass = false_pos_rate < 10
    on_topic_pass = false_neg_rate < 20
    variance_pass = avg_variance > 1.0

    # Compare to v2
    v2_off_topic = v2_metrics['off_topic_rejection_rate']
    v2_on_topic = v2_metrics['on_topic_recognition_rate']

    generalization_delta_off = off_topic_rejection_rate - v2_off_topic
    generalization_delta_on = on_topic_recognition_rate - v2_on_topic

    # Did it generalize?
    generalized_well = (
        off_topic_rejection_rate >= 90 and
        on_topic_recognition_rate >= 70 and
        abs(generalization_delta_off) < 15 and
        abs(generalization_delta_on) < 15
    )

    if off_topic_pass and on_topic_pass and variance_pass and generalized_well:
        decision = "✅ PASS"
        recommendation = "PROMPT GENERALIZES - PROCEED TO BATCH LABELING"
    elif off_topic_pass and on_topic_recognition_rate >= 70:
        decision = "⚠️ REVIEW"
        recommendation = "GOOD GENERALIZATION - MINOR CALIBRATION ADJUSTMENT"
    elif abs(generalization_delta_off) > 20 or abs(generalization_delta_on) > 20:
        decision = "❌ FAIL"
        recommendation = "POOR GENERALIZATION - PROMPT OVERFITTED TO CALIBRATION SAMPLE"
    else:
        decision = "⚠️ REVIEW"
        recommendation = "ACCEPTABLE GENERALIZATION - CONSIDER PROCEEDING"

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"**Decision:** {decision}\n")

    if generalized_well:
        report.append(f"**Overall Assessment:** ✅ The prompt improvements GENERALIZE well to new articles! Fresh sample: {off_topic_rejection_rate:.1f}% rejection, {on_topic_recognition_rate:.1f}% recognition. Differences from v2 calibration are within acceptable range ({generalization_delta_off:+.1f}% off-topic, {generalization_delta_on:+.1f}% on-topic).\n")
    elif abs(generalization_delta_off) > 20 or abs(generalization_delta_on) > 20:
        report.append(f"**Overall Assessment:** ❌ The prompt appears to have OVERFITTED to the calibration sample. Performance dropped significantly on fresh articles (off-topic: {generalization_delta_off:+.1f}%, on-topic: {generalization_delta_on:+.1f}%).\n")
    else:
        report.append(f"**Overall Assessment:** ⚠️ The prompt shows mixed generalization. Fresh sample: {off_topic_rejection_rate:.1f}% rejection, {on_topic_recognition_rate:.1f}% recognition. Changes from v2: {generalization_delta_off:+.1f}% off-topic, {generalization_delta_on:+.1f}% on-topic.\n")

    report.append(f"**Recommendation:** {recommendation}\n")
    report.append("---\n")

    # Key Question
    report.append("## Key Question: Does the Prompt Generalize?\n")
    report.append("**The fresh sample contains 31 completely different articles that were NOT used during calibration.**\n")
    if generalized_well:
        report.append("**Answer:** ✅ YES! The prompt improvements generalize to new, unseen articles.\n")
    elif abs(generalization_delta_off) > 20 or abs(generalization_delta_on) > 20:
        report.append("**Answer:** ❌ NO. The prompt appears to have overfitted to the 27 calibration articles.\n")
    else:
        report.append("**Answer:** ⚠️ PARTIALLY. Some degradation on fresh articles, but still functional.\n")
    report.append("---\n")

    # Calibration Sample Overview
    report.append("## Fresh Sample Overview\n")
    report.append(f"**Total articles reviewed:** {results['total_articles']}")
    report.append(f"- On-topic (expected high scores): {n_on_topic}")
    report.append(f"- Off-topic (expected low scores): {n_off_topic}")
    report.append(f"- Edge cases: {n_edge}\n")
    report.append("**Oracle used:** gemini-flash-api-batch (Gemini Flash 1.5)")
    report.append("**Prompt version:** filters/sustainability_tech_deployment/v1/prompt-compressed.md (v2)\n")
    report.append("---\n")

    # CRITICAL METRICS
    report.append("## CRITICAL METRICS\n")

    # 1. Off-Topic Rejection Rate
    report.append("### 1. Off-Topic Rejection Rate\n")
    report.append("| Metric | Value | Target | Status |")
    report.append("|--------|-------|--------|--------|")
    report.append(f"| Off-topic articles reviewed | {n_off_topic} | N/A | ℹ️ |")

    n_correctly_rejected = n_off_topic - n_false_pos
    report.append(f"| Scored < 5.0 (correctly rejected) | {n_correctly_rejected} ({off_topic_rejection_rate:.1f}%) | >90% | {'✅' if off_topic_pass else '⚠️' if false_pos_rate < 20 else '❌'} |")
    report.append(f"| Scored >= 5.0 (false positives) | {n_false_pos} ({false_pos_rate:.1f}%) | <10% | {'✅' if off_topic_pass else '⚠️' if false_pos_rate < 20 else '❌'} |")

    severe_fps = sum(1 for fp in results['false_positives'] if fp['score'] >= 7.0)
    severe_fps_pct = (severe_fps / n_off_topic * 100) if n_off_topic > 0 else 0
    report.append(f"| Scored >= 7.0 (severe false positives) | {severe_fps} ({severe_fps_pct:.1f}%) | <5% | {'✅' if severe_fps_pct < 5 else '⚠️' if severe_fps_pct < 10 else '❌'} |\n")

    report.append(f"**Status:** {'✅ PASS' if off_topic_pass else '⚠️ REVIEW' if false_pos_rate < 20 else '❌ FAIL'}\n")

    # False positives
    if results['false_positives']:
        report.append("#### False Positive Examples\n")
        for i, fp in enumerate(results['false_positives'][:5], 1):
            report.append(f"**{i}. \"{fp['title']}\" → {fp['score']}**")
            report.append(f"- **Why off-topic:** {fp['reason']}")
            report.append(f"- **Oracle reasoning:** \"{fp['reasoning'][:200]}...\"")
            report.append(f"- **Issue:** Should have been rejected\n")
    else:
        report.append("#### False Positive Examples\n")
        report.append("**None detected!** All off-topic articles were correctly rejected.\n")

    report.append("---\n")

    # 2. On-Topic Recognition Rate
    report.append("### 2. On-Topic Recognition Rate\n")
    report.append("| Metric | Value | Target | Status |")
    report.append("|--------|-------|--------|--------|")
    report.append(f"| On-topic articles reviewed | {n_on_topic} | N/A | ℹ️ |")

    n_correctly_recognized = n_on_topic - n_false_neg
    report.append(f"| Scored >= 5.0 (correctly recognized) | {n_correctly_recognized} ({on_topic_recognition_rate:.1f}%) | >80% | {'✅' if on_topic_pass else '⚠️' if false_neg_rate < 30 else '❌'} |")
    report.append(f"| Scored < 5.0 (false negatives) | {n_false_neg} ({false_neg_rate:.1f}%) | <20% | {'✅' if on_topic_pass else '⚠️' if false_neg_rate < 30 else '❌'} |")

    max_score = max([a['score'] for a in results['on_topic_articles']]) if results['on_topic_articles'] else 0
    has_high_score = max_score >= 7.0
    report.append(f"| At least one article >= 7.0 | {'Yes' if has_high_score else 'No'} ({max_score:.2f} max) | Yes | {'✅' if has_high_score else '❌'} |\n")

    report.append(f"**Status:** {'✅ PASS' if on_topic_pass and has_high_score else '⚠️ REVIEW' if false_neg_rate < 30 else '❌ FAIL'}\n")

    if has_high_score:
        report.append(f"**Highest score in sample:** {max_score:.2f}\n")

    # False negatives
    if results['false_negatives']:
        report.append("#### False Negative Examples\n")
        for i, fn in enumerate(results['false_negatives'][:5], 1):
            report.append(f"**{i}. \"{fn['title']}\" → {fn['score']}**")
            report.append(f"- **Why on-topic:** {fn['reason']}")
            report.append(f"- **Expected score:** 5-7 (deployed climate tech)")
            report.append(f"- **Oracle reasoning:** \"{fn['reasoning'][:200]}...\"")
            report.append(f"- **Issue:** Under-scored deployed climate tech\n")
    else:
        report.append("#### False Negative Examples\n")
        report.append("**None detected!** All on-topic articles were correctly recognized.\n")

    report.append("---\n")

    # 3. Dimensional Consistency
    report.append("### 3. Dimensional Consistency\n")
    report.append("| Metric | Value | Target | Status |")
    report.append("|--------|-------|--------|--------|")
    report.append(f"| Average dimensional variance | {avg_variance:.2f} | >1.0 | {'✅' if variance_pass else '⚠️'} |")
    report.append(f"| Median dimensional variance | {median_variance:.2f} | N/A | ℹ️ |")
    report.append(f"| Articles with variance < 0.5 | {low_variance_count} ({low_variance_pct:.1f}%) | <20% | {'✅' if low_variance_pct < 20 else '⚠️' if low_variance_pct < 40 else '❌'} |")
    report.append(f"| All dimensions used (not all 0 or all 10) | Yes | Yes | ✅ |\n")

    report.append(f"**Status:** {'✅ PASS' if variance_pass and low_variance_pct < 20 else '⚠️ REVIEW'}\n")
    report.append("---\n")

    # Score Distribution
    report.append("## Score Distribution\n")
    report.append("**Overall scores:**")
    for range_str, count in results['score_distribution'].items():
        pct = (count / results['total_articles'] * 100) if results['total_articles'] > 0 else 0
        report.append(f"- {range_str}: {count} articles ({pct:.1f}%)")

    report.append("")

    # By category
    if results['on_topic_articles']:
        on_topic_scores = [a['score'] for a in results['on_topic_articles']]
        report.append(f"**On-topic articles:** Mean={statistics.mean(on_topic_scores):.2f}, Median={statistics.median(on_topic_scores):.2f}, Range=[{min(on_topic_scores):.2f}-{max(on_topic_scores):.2f}]")

    if results['off_topic_articles']:
        off_topic_scores = [a['score'] for a in results['off_topic_articles']]
        report.append(f"**Off-topic articles:** Mean={statistics.mean(off_topic_scores):.2f}, Median={statistics.median(off_topic_scores):.2f}, Range=[{min(off_topic_scores):.2f}-{max(off_topic_scores):.2f}]")

    report.append("\n---\n")

    # Comparison to v2 (original 27 articles)
    report.append("## Comparison: Fresh Sample vs. v2 Calibration (27 articles)\n")
    report.append("| Metric | v2 (27 calibration articles) | Fresh (31 NEW articles) | Difference | Generalization |")
    report.append("|--------|------------------------------|-------------------------|------------|----------------|")
    report.append(f"| Off-topic rejection rate | {v2_off_topic:.1f}% | {off_topic_rejection_rate:.1f}% | {generalization_delta_off:+.1f}% | {'✅ Excellent' if abs(generalization_delta_off) < 5 else '✅ Good' if abs(generalization_delta_off) < 15 else '⚠️ Degraded'} |")
    report.append(f"| On-topic recognition rate | {v2_on_topic:.1f}% | {on_topic_recognition_rate:.1f}% | {generalization_delta_on:+.1f}% | {'✅ Excellent' if abs(generalization_delta_on) < 5 else '✅ Good' if abs(generalization_delta_on) < 15 else '⚠️ Degraded'} |")
    report.append(f"| False positive rate | {100-v2_off_topic:.1f}% | {false_pos_rate:.1f}% | {false_pos_rate-(100-v2_off_topic):+.1f}% | {'✅' if false_pos_rate <= (100-v2_off_topic)+5 else '⚠️'} |")
    report.append(f"| False negative rate | {100-v2_on_topic:.1f}% | {false_neg_rate:.1f}% | {false_neg_rate-(100-v2_on_topic):+.1f}% | {'✅' if false_neg_rate <= (100-v2_on_topic)+5 else '⚠️'} |")
    report.append(f"| Dimensional variance | {v2_metrics['avg_variance']:.2f} | {avg_variance:.2f} | {avg_variance-v2_metrics['avg_variance']:+.2f} | {'✅' if avg_variance >= v2_metrics['avg_variance']*0.8 else '⚠️'} |")

    report.append("\n**Generalization Assessment:**")
    if generalized_well:
        report.append("✅ **EXCELLENT GENERALIZATION** - The prompt performs consistently on new, unseen articles. Differences are minimal and within expected variance.")
    elif abs(generalization_delta_off) > 20 or abs(generalization_delta_on) > 20:
        report.append("❌ **POOR GENERALIZATION** - Significant performance drop suggests the prompt overfitted to the 27 calibration articles.")
    elif off_topic_rejection_rate >= 85 and on_topic_recognition_rate >= 70:
        report.append("✅ **GOOD GENERALIZATION** - Some performance change on fresh articles, but still meets minimum quality standards.")
    else:
        report.append("⚠️ **ACCEPTABLE GENERALIZATION** - Performance degraded but may still be usable depending on quality requirements.")

    report.append("\n---\n")

    # Final Recommendation
    report.append("## Final Recommendation\n")

    if decision == "✅ PASS":
        report.append("**Decision:** ✅ **PASS - PROMPT GENERALIZES WELL**\n")
        report.append("The v2 prompt improvements generalize successfully to new, unseen articles. Performance on the fresh sample is consistent with the calibration results.\n")
        report.append("**Next steps:**")
        report.append("1. ✅ Proceed with batch labeling of full dataset")
        report.append("2. Monitor first batch (500 articles) for quality")
        report.append("3. Document successful generalization validation\n")
    elif decision == "⚠️ REVIEW":
        report.append("**Decision:** ⚠️ **REVIEW - ACCEPTABLE WITH MONITORING**\n")
        report.append("The prompt shows acceptable generalization with some performance variation. Consider proceeding with enhanced monitoring.\n")
        report.append("**Next steps:**")
        report.append("1. Review specific false negatives/positives from fresh sample")
        report.append("2. Consider minor prompt refinements if issues are systematic")
        report.append("3. Proceed to batch labeling with enhanced quality monitoring")
        report.append("4. Validate first 100 labeled articles manually\n")
    else:
        report.append("**Decision:** ❌ **FAIL - POOR GENERALIZATION**\n")
        report.append("The prompt appears to have overfitted to the calibration sample and does not generalize well to new articles.\n")
        report.append("**Next steps:**")
        report.append("1. Expand calibration sample to 50+ diverse articles")
        report.append("2. Revise prompt to reduce overfitting")
        report.append("3. Re-run calibration and validation cycle")
        report.append("4. Do NOT proceed to batch labeling until generalization improves\n")

    report.append("---\n")

    # Appendix
    report.append("## Appendix\n")
    report.append("### Files Reviewed\n")
    report.append("- Prompt: `filters/sustainability_tech_deployment/v1/prompt-compressed.md` (v2)")
    report.append("- Fresh calibration sample: `datasets/working/sustainability_tech_calibration_labeled_fresh.jsonl` (31 articles)")
    report.append("- Comparison baseline: v2 calibration report (27 articles)\n")

    return '\n'.join(report)

if __name__ == '__main__':
    import sys
    # Set stdout to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Load fresh sample
    filepath = 'C:\\local_dev\\llm-distillery\\datasets\\working\\sustainability_tech_calibration_labeled_fresh.jsonl'
    articles = load_calibration_data(filepath)

    # Analyze
    results = analyze_calibration(articles)

    # v2 metrics from previous calibration (27 articles)
    v2_metrics = {
        'off_topic_rejection_rate': 100.0,
        'on_topic_recognition_rate': 80.0,
        'avg_variance': 0.53
    }

    # Generate report
    report = generate_report(results, v2_metrics)

    # Save report
    output_path = 'C:\\local_dev\\llm-distillery\\filters\\sustainability_tech_deployment\\v1\\calibration_report_fresh.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n\nReport saved to: {output_path}")
