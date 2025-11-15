#!/usr/bin/env python3
"""
Comprehensive QA analysis of the uplifting dataset.
Generates detailed statistics, validation checks, and quality report.
"""

import json
import statistics
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

# Expected tier thresholds from config
TIER_THRESHOLDS = {
    'impact': 7.0,
    'connection': 4.0,
    'not_uplifting': 0.0
}

# Expected dimension weights from config
DIMENSION_WEIGHTS = {
    'agency': 0.14,
    'progress': 0.19,
    'collective_benefit': 0.38,
    'connection': 0.10,
    'innovation': 0.08,
    'justice': 0.04,
    'resilience': 0.02,
    'wonder': 0.05
}

EXPECTED_DIMENSIONS = set(DIMENSION_WEIGHTS.keys())

def load_dataset(filepath):
    """Load JSONL dataset and return articles with parse tracking."""
    articles = []
    parse_errors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
                articles.append(article)
            except json.JSONDecodeError as e:
                parse_errors.append({
                    'line': line_num,
                    'error': str(e),
                    'content': line[:100]
                })

    return articles, parse_errors

def validate_structure(articles):
    """Validate data structure and required fields."""
    issues = {
        'missing_id': [],
        'missing_title': [],
        'missing_content': [],
        'missing_uplifting_analysis': [],
        'missing_dimensions': [],
        'incomplete_dimensions': [],
        'invalid_scores': [],
        'missing_overall_score': [],
        'missing_tier': []
    }

    for idx, article in enumerate(articles):
        article_id = article.get('id', f'article_{idx}')

        # Check required top-level fields
        if 'id' not in article:
            issues['missing_id'].append(idx)
        if 'title' not in article or not article.get('title', '').strip():
            issues['missing_title'].append(article_id)
        if 'content' not in article or not article.get('content', '').strip():
            issues['missing_content'].append(article_id)

        # Check uplifting_analysis structure
        if 'uplifting_analysis' not in article:
            issues['missing_uplifting_analysis'].append(article_id)
            continue

        analysis = article['uplifting_analysis']

        # Check for dimensions - they can be stored in two ways:
        # 1. As nested dict: dimensions['agency']['score']
        # 2. As flat in analysis: analysis['agency'] (direct score)
        # First check if dimensions dict exists
        if 'dimensions' in analysis:
            dimensions = analysis['dimensions']
            found_dims = set(dimensions.keys())

            # Check for incomplete dimensions
            missing_dims = EXPECTED_DIMENSIONS - found_dims
            if missing_dims:
                issues['incomplete_dimensions'].append({
                    'id': article_id,
                    'missing': list(missing_dims)
                })

            # Validate dimension scores (handle both nested dict and flat number)
            for dim_name, dim_data in dimensions.items():
                if isinstance(dim_data, dict):
                    # Nested structure with score field
                    if 'score' not in dim_data:
                        issues['invalid_scores'].append({
                            'id': article_id,
                            'dimension': dim_name,
                            'reason': 'missing score field in dict'
                        })
                        continue
                    score = dim_data['score']
                elif isinstance(dim_data, (int, float)):
                    # Direct score
                    score = dim_data
                else:
                    issues['invalid_scores'].append({
                        'id': article_id,
                        'dimension': dim_name,
                        'reason': f'unexpected type: {type(dim_data)}'
                    })
                    continue

                if not isinstance(score, (int, float)) or score < 0 or score > 10:
                    issues['invalid_scores'].append({
                        'id': article_id,
                        'dimension': dim_name,
                        'score': score,
                        'reason': 'score out of range [0-10]'
                    })
        else:
            # Check if dimensions are stored flat in analysis
            missing_dims = []
            for dim_name in EXPECTED_DIMENSIONS:
                if dim_name not in analysis:
                    missing_dims.append(dim_name)

            if missing_dims:
                issues['missing_dimensions'].append({
                    'id': article_id,
                    'missing': missing_dims
                })

        # Check overall_score and tier
        if 'overall_uplift_score' not in analysis:
            issues['missing_overall_score'].append(article_id)
        if 'tier' not in analysis:
            issues['missing_tier'].append(article_id)

    return issues

def check_duplicates(articles):
    """Check for duplicate IDs and URLs."""
    id_counts = Counter(a['id'] for a in articles if 'id' in a)
    url_counts = Counter(a.get('url', '') for a in articles if a.get('url'))

    duplicate_ids = [(id_, count) for id_, count in id_counts.items() if count > 1]
    duplicate_urls = [(url, count) for url, count in url_counts.items() if count > 1 and url]

    return duplicate_ids, duplicate_urls

def check_failed_labeling(articles):
    """Find articles with all zero scores (failed labeling)."""
    failed = []

    for article in articles:
        if 'uplifting_analysis' not in article:
            continue

        analysis = article['uplifting_analysis']
        if 'dimensions' not in analysis:
            continue

        dimensions = analysis['dimensions']
        scores = [dim.get('score', 0) for dim in dimensions.values() if isinstance(dim, dict)]

        if scores and all(s == 0 for s in scores):
            failed.append({
                'id': article.get('id'),
                'title': article.get('title', '')[:100],
                'overall_score': analysis.get('overall_uplift_score', 0)
            })

    return failed

def calculate_statistics(articles):
    """Calculate comprehensive statistics."""
    stats = {
        'overall_scores': [],
        'dimension_scores': defaultdict(list),
        'tiers': Counter(),
        'content_types': Counter(),
        'sources': Counter()
    }

    for article in articles:
        # Overall scores
        if 'uplifting_analysis' in article:
            analysis = article['uplifting_analysis']
            if 'overall_uplift_score' in analysis:
                stats['overall_scores'].append(analysis['overall_uplift_score'])

            # Tier distribution
            if 'tier' in analysis:
                stats['tiers'][analysis['tier']] += 1

            # Content type distribution
            if 'content_type' in analysis:
                stats['content_types'][analysis['content_type']] += 1

            # Dimension scores - handle both nested dict and flat format
            if 'dimensions' in analysis:
                for dim_name, dim_data in analysis['dimensions'].items():
                    if isinstance(dim_data, dict) and 'score' in dim_data:
                        stats['dimension_scores'][dim_name].append(dim_data['score'])
                    elif isinstance(dim_data, (int, float)):
                        # Direct score format
                        stats['dimension_scores'][dim_name].append(dim_data)

        # Source distribution
        if 'source' in article:
            stats['sources'][article['source']] += 1

    return stats

def compute_summary_stats(scores):
    """Compute summary statistics for a list of scores."""
    if not scores:
        return {
            'min': 0, 'max': 0, 'mean': 0, 'median': 0,
            'std_dev': 0, 'count': 0
        }

    return {
        'min': round(min(scores), 2),
        'max': round(max(scores), 2),
        'mean': round(statistics.mean(scores), 2),
        'median': round(statistics.median(scores), 2),
        'std_dev': round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
        'count': len(scores)
    }

def validate_tier_assignments(articles):
    """Validate that tier assignments match score thresholds."""
    mismatches = []

    for article in articles:
        if 'uplifting_analysis' not in article:
            continue

        analysis = article['uplifting_analysis']
        score = analysis.get('overall_uplift_score')
        tier = analysis.get('tier')

        if score is None or tier is None:
            continue

        # Determine expected tier based on config thresholds
        if score >= TIER_THRESHOLDS['impact']:
            expected_tier = 'impact'
        elif score >= TIER_THRESHOLDS['connection']:
            expected_tier = 'connection'
        else:
            expected_tier = 'not_uplifting'

        if tier != expected_tier:
            mismatches.append({
                'id': article.get('id'),
                'title': article.get('title', '')[:80],
                'score': score,
                'assigned_tier': tier,
                'expected_tier': expected_tier
            })

    return mismatches

def analyze_score_distribution(scores, bins=10):
    """Analyze score distribution in bins."""
    if not scores:
        return []

    bin_width = 10.0 / bins
    distribution = [0] * bins

    for score in scores:
        bin_idx = min(int(score / bin_width), bins - 1)
        distribution[bin_idx] += 1

    bin_data = []
    for i, count in enumerate(distribution):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        percentage = (count / len(scores)) * 100
        bin_data.append({
            'range': f'{bin_start:.1f}-{bin_end:.1f}',
            'count': count,
            'percentage': round(percentage, 2)
        })

    return bin_data

def main():
    # File paths
    dataset_path = Path(r'C:\local_dev\llm-distillery\datasets\labeled\uplifting\labeled_articles.jsonl')

    print("=" * 80)
    print("UPLIFTING DATASET QA ANALYSIS")
    print("=" * 80)
    print()

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    articles, parse_errors = load_dataset(dataset_path)
    print(f"Loaded {len(articles)} articles")
    if parse_errors:
        print(f"WARNING: {len(parse_errors)} parse errors found")
    print()

    # Validate structure
    print("Validating data structure...")
    issues = validate_structure(articles)

    # Check duplicates
    print("Checking for duplicates...")
    duplicate_ids, duplicate_urls = check_duplicates(articles)

    # Check failed labeling
    print("Checking for failed labeling...")
    failed_articles = check_failed_labeling(articles)

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(articles)

    # Validate tier assignments
    print("Validating tier assignments...")
    tier_mismatches = validate_tier_assignments(articles)

    # Generate summary statistics
    overall_stats = compute_summary_stats(stats['overall_scores'])
    dimension_stats = {
        dim: compute_summary_stats(scores)
        for dim, scores in stats['dimension_scores'].items()
    }

    # Score distribution
    score_distribution = analyze_score_distribution(stats['overall_scores'])

    print("\nAnalysis complete!")
    print()

    # Output results as JSON for processing
    results = {
        'metadata': {
            'total_articles': len(articles),
            'analysis_date': datetime.now().isoformat(),
            'dataset_path': str(dataset_path)
        },
        'parse_errors': parse_errors,
        'structure_issues': {k: len(v) for k, v in issues.items()},
        'structure_issues_detail': issues,
        'duplicates': {
            'duplicate_ids': len(duplicate_ids),
            'duplicate_urls': len(duplicate_urls),
            'duplicate_ids_detail': duplicate_ids[:10],
            'duplicate_urls_detail': duplicate_urls[:10]
        },
        'failed_labeling': {
            'count': len(failed_articles),
            'examples': failed_articles[:10]
        },
        'overall_stats': overall_stats,
        'dimension_stats': dimension_stats,
        'tier_distribution': dict(stats['tiers']),
        'content_type_distribution': dict(stats['content_types']),
        'source_distribution': dict(stats['sources'].most_common(20)),
        'score_distribution': score_distribution,
        'tier_mismatches': {
            'count': len(tier_mismatches),
            'examples': tier_mismatches[:20]
        }
    }

    # Save results
    output_path = Path(r'C:\local_dev\llm-distillery\reports\uplifting_qa_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total articles: {len(articles)}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Structure issues: {sum(len(v) for v in issues.values())}")
    print(f"Duplicate IDs: {len(duplicate_ids)}")
    print(f"Failed labeling: {len(failed_articles)}")
    print(f"Tier mismatches: {len(tier_mismatches)}")
    print()
    print(f"Overall score: {overall_stats['mean']} ± {overall_stats['std_dev']}")
    print(f"  Range: {overall_stats['min']} - {overall_stats['max']}")
    print()
    print("Tier distribution:")
    for tier, count in sorted(stats['tiers'].items(), key=lambda x: -x[1]):
        percentage = (count / len(articles)) * 100
        print(f"  {tier}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
