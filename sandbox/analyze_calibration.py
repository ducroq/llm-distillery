#!/usr/bin/env python3
"""
Analyze sustainability_tech_deployment calibration sample.
Systematically review 27 labeled articles for:
- Off-topic false positives (scored >= 5.0 when should be low)
- On-topic false negatives (scored < 5.0 when should be high)
- Dimensional consistency (variance should be >1.0)
- Oracle reasoning quality
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any

def load_calibration_data(filepath: str) -> List[Dict]:
    """Load JSONL calibration data."""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def calculate_dimensional_variance(dimensions: Dict[str, float]) -> float:
    """Calculate variance of dimensional scores."""
    scores = list(dimensions.values())
    if len(scores) < 2:
        return 0.0
    return statistics.variance(scores)

def is_off_topic(article: Dict) -> bool:
    """
    Determine if article is clearly OFF-TOPIC for sustainability tech deployment.

    OFF-TOPIC indicators:
    - Generic IT/cloud/programming
    - Office productivity software
    - Healthcare (unless sustainability-related)
    - Social media, gaming, entertainment
    - Policy/legal discussions without tech deployment
    - Education inequality
    """
    title = article.get('title', '').lower()
    content = article.get('content', '').lower()

    # Keywords that indicate OFF-TOPIC
    off_topic_keywords = [
        # Generic IT
        'aws', 'azure', 'cloud infrastructure', 'database', 'api', 'programming',
        'excel', 'microsoft office', 'software development', 'coding',
        # Policy/Legal without tech
        'court', 'lawsuit', 'policy', 'regulation', 'legal',
        # Education/Social issues
        'university', 'education', 'graduates', 'students', 'inequality',
        # Healthcare
        'medical', 'health', 'hospital', 'disease',
        # Generic tech
        'toothbrush', 'consumer electronics',
        # Computing research without climate relevance
        'energy efficiency' if 'climate' not in content and 'carbon' not in content and 'renewable' not in content else ''
    ]

    # Keywords that indicate IN-SCOPE
    in_scope_keywords = [
        'solar', 'wind', 'renewable', 'battery', 'electric vehicle', 'ev ', 'heat pump',
        'hydrogen', 'carbon capture', 'emissions', 'climate tech', 'sustainability',
        'geothermal', 'hydro', 'grid', 'decarboniz', 'clean energy'
    ]

    # Check if clearly in-scope
    in_scope_count = sum(1 for kw in in_scope_keywords if kw in title or kw in content)
    if in_scope_count >= 2:
        return False  # Clearly in-scope

    # Check if clearly off-topic
    off_topic_count = sum(1 for kw in off_topic_keywords if kw and (kw in title or kw in content))
    if off_topic_count >= 1 and in_scope_count == 0:
        return True

    # Check analysis assessment
    analysis = article.get('sustainability_tech_deployment_analysis', {})
    assessment = analysis.get('overall_assessment', '').lower()

    if 'out of scope' in assessment or 'not related' in assessment:
        return True

    return False

def is_on_topic(article: Dict) -> bool:
    """
    Determine if article is clearly ON-TOPIC for sustainability tech deployment.
    """
    title = article.get('title', '').lower()
    content = article.get('content', '').lower()

    # Keywords that indicate IN-SCOPE
    in_scope_keywords = [
        'solar', 'wind', 'renewable', 'battery', 'electric vehicle', 'ev ', 'evs ',
        'heat pump', 'hydrogen', 'carbon capture', 'emissions', 'climate tech',
        'sustainability tech', 'geothermal', 'hydroelectric', 'grid modern',
        'decarboniz', 'clean energy', 'charging infrastructure'
    ]

    in_scope_count = sum(1 for kw in in_scope_keywords if kw in title or kw in content)

    # Need strong signals for on-topic
    return in_scope_count >= 2

def analyze_calibration(filepath: str) -> Dict:
    """Perform full calibration analysis."""
    articles = load_calibration_data(filepath)

    results = {
        'total_articles': len(articles),
        'off_topic_articles': [],
        'on_topic_articles': [],
        'false_positives': [],  # Off-topic but scored high
        'false_negatives': [],  # On-topic but scored low
        'dimensional_variances': [],
        'low_variance_articles': [],
        'score_distribution': {
            '0-2': 0, '3-4': 0, '5-6': 0, '7-8': 0, '9-10': 0
        }
    }

    for article in articles:
        analysis = article.get('sustainability_tech_deployment_analysis', {})
        dimensions = analysis.get('dimensions', {})
        overall_score = analysis.get('overall_score', 0)

        # Calculate variance
        variance = calculate_dimensional_variance(dimensions)
        results['dimensional_variances'].append(variance)

        if variance < 0.5:
            results['low_variance_articles'].append({
                'title': article['title'],
                'score': overall_score,
                'variance': variance,
                'dimensions': dimensions
            })

        # Score distribution
        if overall_score <= 2:
            results['score_distribution']['0-2'] += 1
        elif overall_score <= 4:
            results['score_distribution']['3-4'] += 1
        elif overall_score <= 6:
            results['score_distribution']['5-6'] += 1
        elif overall_score <= 8:
            results['score_distribution']['7-8'] += 1
        else:
            results['score_distribution']['9-10'] += 1

        # Categorize articles
        off_topic = is_off_topic(article)
        on_topic = is_on_topic(article)

        if off_topic:
            results['off_topic_articles'].append({
                'title': article['title'],
                'score': overall_score,
                'reasoning': analysis.get('overall_assessment', ''),
                'url': article.get('url', '')
            })

            # Check for false positives (off-topic scored high)
            if overall_score >= 5.0:
                results['false_positives'].append({
                    'title': article['title'],
                    'score': overall_score,
                    'reasoning': analysis.get('overall_assessment', ''),
                    'dimensions': dimensions
                })

        if on_topic:
            results['on_topic_articles'].append({
                'title': article['title'],
                'score': overall_score,
                'reasoning': analysis.get('overall_assessment', ''),
                'deployment_stage': analysis.get('deployment_stage', ''),
                'url': article.get('url', '')
            })

            # Check for false negatives (on-topic scored low)
            # Only flag if it describes actual deployment
            deployment_stage = analysis.get('deployment_stage', '')
            if overall_score < 5.0 and deployment_stage in ['mass_deployment', 'commercial_proven', 'early_commercial']:
                results['false_negatives'].append({
                    'title': article['title'],
                    'score': overall_score,
                    'expected_score': '7-8' if deployment_stage == 'mass_deployment' else '5-7',
                    'reasoning': analysis.get('overall_assessment', ''),
                    'deployment_stage': deployment_stage
                })

    # Calculate summary statistics
    if results['dimensional_variances']:
        results['avg_variance'] = statistics.mean(results['dimensional_variances'])
        results['median_variance'] = statistics.median(results['dimensional_variances'])

    results['off_topic_rejection_rate'] = (
        (len(results['off_topic_articles']) - len(results['false_positives'])) /
        len(results['off_topic_articles'])
    ) * 100 if results['off_topic_articles'] else 0

    results['on_topic_recognition_rate'] = (
        (len(results['on_topic_articles']) - len(results['false_negatives'])) /
        len(results['on_topic_articles'])
    ) * 100 if results['on_topic_articles'] else 0

    return results

def print_report(results: Dict):
    """Print calibration report to console."""
    import sys
    # Ensure UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 80)
    print("SUSTAINABILITY TECH DEPLOYMENT - CALIBRATION ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total articles reviewed: {results['total_articles']}")
    print(f"- Off-topic articles: {len(results['off_topic_articles'])}")
    print(f"- On-topic articles: {len(results['on_topic_articles'])}")
    print(f"- Edge cases: {results['total_articles'] - len(results['off_topic_articles']) - len(results['on_topic_articles'])}")
    print()

    print("=" * 80)
    print("CRITICAL METRICS")
    print("=" * 80)
    print()

    # Off-topic rejection
    print("1. OFF-TOPIC REJECTION RATE")
    print("-" * 80)
    off_topic_count = len(results['off_topic_articles'])
    fp_count = len(results['false_positives'])
    fp_high_count = len([fp for fp in results['false_positives'] if fp['score'] >= 7.0])

    print(f"Off-topic articles reviewed: {off_topic_count}")
    print(f"Scored < 5.0 (correctly rejected): {off_topic_count - fp_count} ({results['off_topic_rejection_rate']:.1f}%)")
    print(f"Scored >= 5.0 (false positives): {fp_count} ({fp_count/off_topic_count*100 if off_topic_count else 0:.1f}%)")
    print(f"Scored >= 7.0 (severe false positives): {fp_high_count} ({fp_high_count/off_topic_count*100 if off_topic_count else 0:.1f}%)")
    print()

    # Status determination
    if fp_count / off_topic_count < 0.10 if off_topic_count else True:
        status = "✅ PASS"
    elif fp_count / off_topic_count < 0.20 if off_topic_count else True:
        status = "⚠️ REVIEW"
    else:
        status = "❌ FAIL"

    print(f"Status: {status}")
    print()

    if results['false_positives']:
        print("FALSE POSITIVE EXAMPLES:")
        print()
        for i, fp in enumerate(results['false_positives'][:5], 1):
            print(f"{i}. \"{fp['title']}\" → {fp['score']:.2f}")
            print(f"   Oracle reasoning: \"{fp['reasoning'][:150]}...\"")
            print(f"   Issue: Article is off-topic for climate/sustainability tech")
            print()

    # On-topic recognition
    print("2. ON-TOPIC RECOGNITION RATE")
    print("-" * 80)
    on_topic_count = len(results['on_topic_articles'])
    fn_count = len(results['false_negatives'])
    high_score_count = len([a for a in results['on_topic_articles'] if a['score'] >= 7.0])

    print(f"On-topic articles reviewed: {on_topic_count}")
    print(f"Scored >= 5.0 (correctly recognized): {on_topic_count - fn_count} ({results['on_topic_recognition_rate']:.1f}%)")
    print(f"Scored < 5.0 (false negatives): {fn_count} ({fn_count/on_topic_count*100 if on_topic_count else 0:.1f}%)")
    print(f"At least one article >= 7.0: {'Yes' if high_score_count > 0 else 'No'} ({high_score_count} articles)")
    print()

    # Status determination
    if fn_count / on_topic_count < 0.20 if on_topic_count else True:
        status = "✅ PASS"
    elif fn_count / on_topic_count < 0.30 if on_topic_count else True:
        status = "⚠️ REVIEW"
    else:
        status = "❌ FAIL"

    print(f"Status: {status}")
    print()

    if results['false_negatives']:
        print("FALSE NEGATIVE EXAMPLES:")
        print()
        for i, fn in enumerate(results['false_negatives'][:3], 1):
            print(f"{i}. \"{fn['title']}\" → {fn['score']:.2f}")
            print(f"   Expected: {fn['expected_score']}")
            print(f"   Oracle reasoning: \"{fn['reasoning'][:150]}...\"")
            print()

    # Dimensional consistency
    print("3. DIMENSIONAL CONSISTENCY")
    print("-" * 80)
    avg_var = results.get('avg_variance', 0)
    low_var_count = len(results['low_variance_articles'])
    low_var_pct = (low_var_count / results['total_articles'] * 100) if results['total_articles'] else 0

    print(f"Average dimensional variance: {avg_var:.2f}")
    print(f"Median dimensional variance: {results.get('median_variance', 0):.2f}")
    print(f"Articles with variance < 0.5: {low_var_count} ({low_var_pct:.1f}%)")
    print()

    # Status determination
    if avg_var > 1.0 and low_var_pct < 20:
        status = "✅ PASS"
    elif avg_var > 0.5:
        status = "⚠️ REVIEW"
    else:
        status = "❌ FAIL"

    print(f"Status: {status}")
    print()

    if results['low_variance_articles'][:3]:
        print("LOW VARIANCE EXAMPLES:")
        print()
        for i, lv in enumerate(results['low_variance_articles'][:3], 1):
            print(f"{i}. \"{lv['title']}\" → Variance: {lv['variance']:.2f}, Score: {lv['score']:.2f}")
            dim_str = ', '.join([f"{k.split('_')[0]}: {v}" for k, v in lv['dimensions'].items()])
            print(f"   Dimensions: {dim_str}")
            print()

    print("=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)
    print()
    for range_str, count in results['score_distribution'].items():
        pct = (count / results['total_articles'] * 100) if results['total_articles'] else 0
        bar = '█' * int(pct / 2)
        print(f"{range_str:6s}: {count:2d} articles ({pct:5.1f}%) {bar}")
    print()

    print("=" * 80)
    print("DECISION")
    print("=" * 80)
    print()

    # Overall decision
    critical_failures = 0
    if fp_count / off_topic_count >= 0.20 if off_topic_count else False:
        critical_failures += 1
    if fn_count / on_topic_count >= 0.30 if on_topic_count else False:
        critical_failures += 1
    if avg_var < 1.0:
        critical_failures += 1

    if critical_failures == 0 and fp_count / off_topic_count < 0.10 if off_topic_count else True:
        decision = "✅ PASS"
        recommendation = "PROCEED TO BATCH LABELING"
    elif critical_failures == 0:
        decision = "⚠️ REVIEW"
        recommendation = "FIX MINOR ISSUES BEFORE BATCH LABELING"
    else:
        decision = "❌ FAIL"
        recommendation = "MAJOR PROMPT REVISION REQUIRED"

    print(f"Decision: {decision}")
    print(f"Recommendation: {recommendation}")
    print()

if __name__ == '__main__':
    filepath = r'C:\local_dev\llm-distillery\datasets\working\sustainability_tech_calibration_labeled.jsonl'
    results = analyze_calibration(filepath)
    print_report(results)

    # Save detailed results
    output_file = r'C:\local_dev\llm-distillery\sandbox\calibration_analysis_detailed.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
