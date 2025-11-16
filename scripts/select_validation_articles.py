"""
Select articles for manual validation review.

Selects:
- 10 high scorers (collective_benefit >=7 or avg score >=7)
- 10 edge cases (collective_benefit 4-6 with mixed dimension scores)
- 10 low scorers (collective_benefit <=3 or avg score <=3)

Usage:
    python scripts/select_validation_articles.py \
        --input filters/uplifting/v4/validation_scored/uplifting/scored_batch_001.jsonl \
        --output filters/uplifting/v4/validation_review.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def calculate_avg_score(analysis: Dict[str, Any], dimensions: List[str]) -> float:
    """Calculate average score across all dimensions."""
    scores = [analysis.get(dim, 0) for dim in dimensions]
    return sum(scores) / len(scores) if scores else 0


def calculate_score_variance(analysis: Dict[str, Any], dimensions: List[str]) -> float:
    """Calculate variance in scores (edge cases have mixed scores)."""
    scores = [analysis.get(dim, 0) for dim in dimensions]
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    return variance ** 0.5  # Return standard deviation


def load_scored_articles(input_file: Path) -> List[Dict[str, Any]]:
    """Load scored articles from JSONL."""
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def categorize_articles(articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize articles into high/edge/low scorers."""

    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']

    high_scorers = []
    edge_cases = []
    low_scorers = []

    for article in articles:
        analysis = article.get('uplifting_analysis', {})
        cb = analysis.get('collective_benefit', 0)
        avg_score = calculate_avg_score(analysis, dimensions)
        variance = calculate_score_variance(analysis, dimensions)

        # Add metadata for sorting
        article['_meta'] = {
            'collective_benefit': cb,
            'avg_score': avg_score,
            'variance': variance
        }

        # Categorize
        if cb >= 7 or avg_score >= 7:
            high_scorers.append(article)
        elif cb <= 3 or avg_score <= 3:
            low_scorers.append(article)
        elif 4 <= cb <= 6:
            edge_cases.append(article)

    return {
        'high': high_scorers,
        'edge': edge_cases,
        'low': low_scorers
    }


def select_representative_sample(
    categories: Dict[str, List[Dict[str, Any]]],
    n_per_category: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Select representative articles from each category."""

    selected = {}

    # High scorers: Sort by avg_score descending, pick top 10
    high = sorted(categories['high'], key=lambda x: x['_meta']['avg_score'], reverse=True)
    selected['high_scorers'] = high[:n_per_category]

    # Edge cases: Sort by variance descending (most mixed scores), pick top 10
    edge = sorted(categories['edge'], key=lambda x: x['_meta']['variance'], reverse=True)
    selected['edge_cases'] = edge[:n_per_category]

    # Low scorers: Sort by avg_score ascending, pick bottom 10
    low = sorted(categories['low'], key=lambda x: x['_meta']['avg_score'])
    selected['low_scorers'] = low[:n_per_category]

    return selected


def generate_review_document(selected: Dict[str, List[Dict[str, Any]]], output_file: Path):
    """Generate JSON file for manual review."""

    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']

    review_data = {
        'metadata': {
            'total_articles': sum(len(v) for v in selected.values()),
            'categories': {k: len(v) for k, v in selected.items()},
            'purpose': 'Manual validation of oracle scoring quality'
        },
        'articles': []
    }

    for category, articles in selected.items():
        for article in articles:
            analysis = article.get('uplifting_analysis', {})

            review_article = {
                'id': article.get('id', 'unknown'),
                'category': category,
                'title': article.get('title', ''),
                'content': article.get('content', '')[:500] + '...',  # First 500 chars
                'url': article.get('url', ''),
                'oracle_scores': {dim: analysis.get(dim, 0) for dim in dimensions},
                'oracle_reasoning': analysis.get('reasoning', ''),
                'metadata': {
                    'avg_score': article['_meta']['avg_score'],
                    'collective_benefit': article['_meta']['collective_benefit'],
                    'variance': article['_meta']['variance']
                },
                'manual_review': {
                    'reviewed': False,
                    'oracle_correct': None,
                    'reviewer_notes': ''
                }
            }

            review_data['articles'].append(review_article)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    print(f"\nReview document saved: {output_file}")
    print(f"\nArticles selected for manual review:")
    print(f"  High scorers:  {len(selected['high_scorers'])}")
    print(f"  Edge cases:    {len(selected['edge_cases'])}")
    print(f"  Low scorers:   {len(selected['low_scorers'])}")
    print(f"  Total:         {len(review_data['articles'])}")


def main():
    parser = argparse.ArgumentParser(description='Select validation articles for manual review')
    parser.add_argument('--input', type=Path, required=True, help='Scored JSONL file')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--n-per-category', type=int, default=10, help='Articles per category')

    args = parser.parse_args()

    print(f"Loading scored articles from: {args.input}")
    articles = load_scored_articles(args.input)
    print(f"Loaded {len(articles)} articles")

    print(f"\nCategorizing articles...")
    categories = categorize_articles(articles)
    print(f"  High scorers:  {len(categories['high'])}")
    print(f"  Edge cases:    {len(categories['edge'])}")
    print(f"  Low scorers:   {len(categories['low'])}")

    print(f"\nSelecting representative sample ({args.n_per_category} per category)...")
    selected = select_representative_sample(categories, args.n_per_category)

    print(f"\nGenerating review document...")
    generate_review_document(selected, args.output)

    print(f"\nNext step: Manually review {args.output} and update 'manual_review' fields")


if __name__ == '__main__':
    main()
