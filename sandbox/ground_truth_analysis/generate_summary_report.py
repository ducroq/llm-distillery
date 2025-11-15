#!/usr/bin/env python3
"""Generate comprehensive summary report of ground truth dataset."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_labeled_data(file_path: str) -> List[Dict]:
    """Load labeled articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def calculate_impact_score(dimensions: Dict[str, int]) -> float:
    """Calculate weighted average impact score."""
    # Weights from the original filter definition
    weights = {
        'agency': 1.0,
        'progress': 1.0,
        'collective_benefit': 1.5,
        'connection': 0.8,
        'innovation': 1.2,
        'justice': 1.3,
        'resilience': 1.0,
        'wonder': 0.9
    }

    total_weighted = 0
    total_weight = 0

    for dim, score in dimensions.items():
        if dim in weights:
            total_weighted += score * weights[dim]
            total_weight += weights[dim]

    return total_weighted / total_weight if total_weight > 0 else 0

def get_coverage_stats(articles: List[Dict]) -> Dict:
    """Calculate coverage statistics for all dimensions."""
    stats = defaultdict(lambda: {
        'total': 0,
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'distribution': defaultdict(int)
    })

    for article in articles:
        analysis = None
        if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
            analysis = article['uplifting_analysis']['dimensions']
        elif 'analysis' in article:
            analysis = article['analysis']

        if not analysis:
            continue

        for dim, data in analysis.items():
            # Handle both flat integers and nested dicts
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
            elif isinstance(data, int):
                score = data
            else:
                continue

            stats[dim]['total'] += 1
            stats[dim]['sum'] += score
            stats[dim]['min'] = min(stats[dim]['min'], score)
            stats[dim]['max'] = max(stats[dim]['max'], score)
            stats[dim]['distribution'][score] += 1

    # Calculate means
    for dim in stats:
        if stats[dim]['total'] > 0:
            stats[dim]['mean'] = stats[dim]['sum'] / stats[dim]['total']

    return dict(stats)

def get_top_articles_by_dimension(articles: List[Dict], dimension: str, n: int = 2) -> List[Tuple[Dict, int]]:
    """Get top N articles for a specific dimension."""
    scored_articles = []

    for article in articles:
        analysis = None
        if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
            analysis = article['uplifting_analysis']['dimensions']
        elif 'analysis' in article:
            analysis = article['analysis']

        if analysis and dimension in analysis:
            dim_data = analysis[dimension]
            # Handle both flat integers and nested dicts
            if isinstance(dim_data, dict) and 'score' in dim_data:
                scored_articles.append((article, dim_data['score']))
            elif isinstance(dim_data, int):
                scored_articles.append((article, dim_data))

    # Sort by score descending
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    return scored_articles[:n]

def get_top_articles_by_impact(articles: List[Dict], n: int = 5) -> List[Tuple[Dict, float]]:
    """Get top N articles by overall impact score."""
    scored_articles = []

    for article in articles:
        analysis = None
        if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
            analysis = article['uplifting_analysis']['dimensions']
        elif 'analysis' in article:
            analysis = article['analysis']

        if not analysis:
            continue

        # Extract dimension scores
        dimensions = {}
        for dim, data in analysis.items():
            # Handle both flat integers and nested dicts
            if isinstance(data, dict) and 'score' in data:
                dimensions[dim] = data['score']
            elif isinstance(data, int):
                dimensions[dim] = data

        if dimensions:
            impact = calculate_impact_score(dimensions)
            scored_articles.append((article, impact))

    # Sort by impact descending
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    return scored_articles[:n]

def summarize_text(text: str, max_words: int = 150) -> str:
    """Create a summary of text if it's too long."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '... [summarized]'

def format_article_content(article: Dict, max_words: int = 800) -> Tuple[str, bool]:
    """Format article content, summarizing if needed."""
    content = article.get('content', article.get('text', ''))
    words = content.split()
    word_count = len(words)

    if word_count <= max_words:
        return content, False
    else:
        # Summarize to ~200 words
        summary = ' '.join(words[:200]) + f'\n\n...[Article summarized - original was {word_count} words]'
        return summary, True

def get_analysis_data(article: Dict) -> Dict:
    """Extract analysis data from article."""
    if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
        return article['uplifting_analysis']['dimensions']
    elif 'analysis' in article:
        return article['analysis']
    return {}

def generate_markdown_report(articles: List[Dict], output_file: str):
    """Generate comprehensive markdown report."""

    # Get statistics
    stats = get_coverage_stats(articles)

    # Get top articles
    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']

    top_by_dimension = {}
    for dim in dimensions:
        top_by_dimension[dim] = get_top_articles_by_dimension(articles, dim, n=2)

    top_by_impact = get_top_articles_by_impact(articles, n=5)

    # Generate report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Ground Truth Dataset Summary Report\n\n")
        f.write(f"**Dataset**: Filtered 10k Uplifting News Articles (Gemini Flash Labeled)\n\n")
        f.write(f"**Filtering Applied**: Content ≥300 characters, framework leakage removed\n\n")
        f.write(f"**Total Articles Analyzed**: {len(articles):,}\n\n")
        f.write("---\n\n")

        # Overall Statistics
        f.write("## Overall Statistics\n\n")

        for dim in dimensions:
            if dim not in stats:
                continue

            s = stats[dim]
            f.write(f"### {dim.replace('_', ' ').title()}\n\n")
            f.write(f"- **Total examples**: {s['total']:,}\n")
            f.write(f"- **Range**: {s['min']} - {s['max']}\n")
            f.write(f"- **Mean**: {s['mean']:.2f}\n\n")

            f.write("**Distribution**:\n\n")
            for score in range(1, 11):
                count = s['distribution'].get(score, 0)
                pct = (count / s['total'] * 100) if s['total'] > 0 else 0
                bar = '#' * int(pct / 2)
                f.write(f"- Score {score:2d}: {count:5d} ({pct:5.1f}%) {bar}\n")
            f.write("\n")

        f.write("---\n\n")

        # Top articles by dimension
        f.write("## Top Articles by Dimension\n\n")

        for dim in dimensions:
            f.write(f"### {dim.replace('_', ' ').title()}\n\n")

            top_articles = top_by_dimension.get(dim, [])

            for idx, (article, score) in enumerate(top_articles, 1):
                f.write(f"#### Example {idx} (Score: {score}/10)\n\n")

                # Article metadata
                f.write(f"**Title**: {article.get('title', 'N/A')}\n\n")
                if 'url' in article:
                    f.write(f"**URL**: {article['url']}\n\n")
                if 'published_date' in article:
                    f.write(f"**Published**: {article['published_date']}\n\n")

                # Article content
                content, was_summarized = format_article_content(article)
                f.write("**Content**:\n\n")
                f.write(f"> {content}\n\n")

                # Analysis reasoning (overall, not per-dimension in batch-labeled data)
                if 'uplifting_analysis' in article and 'reasoning' in article['uplifting_analysis']:
                    f.write(f"**Overall Analysis Reasoning**:\n\n")
                    f.write(f"{article['uplifting_analysis']['reasoning']}\n\n")
                else:
                    # Legacy format with per-dimension reasoning
                    analysis = get_analysis_data(article)
                    if dim in analysis and isinstance(analysis[dim], dict):
                        dim_data = analysis[dim]
                        if 'reasoning' in dim_data:
                            f.write(f"**Reasoning for {dim} score ({score}/10)**:\n\n")
                            f.write(f"{dim_data['reasoning']}\n\n")

                f.write("---\n\n")

        # Top articles by overall impact
        f.write("## Top Articles by Overall Impact\n\n")
        f.write("These articles scored highest using the weighted average across all 8 dimensions.\n\n")

        f.write("### Impact Score Weights\n\n")
        f.write("- **agency**: 1.0\n")
        f.write("- **progress**: 1.0\n")
        f.write("- **collective_benefit**: 1.5 (highest weight)\n")
        f.write("- **connection**: 0.8\n")
        f.write("- **innovation**: 1.2\n")
        f.write("- **justice**: 1.3\n")
        f.write("- **resilience**: 1.0\n")
        f.write("- **wonder**: 0.9\n\n")

        f.write("---\n\n")

        for idx, (article, impact) in enumerate(top_by_impact, 1):
            f.write(f"### Top Article #{idx} (Impact Score: {impact:.2f}/10)\n\n")

            # Article metadata
            f.write(f"**Title**: {article.get('title', 'N/A')}\n\n")
            if 'url' in article:
                f.write(f"**URL**: {article['url']}\n\n")
            if 'published_date' in article:
                f.write(f"**Published**: {article['published_date']}\n\n")

            # Article content
            content, was_summarized = format_article_content(article)
            f.write("**Content**:\n\n")
            f.write(f"> {content}\n\n")

            # All dimension scores
            analysis = get_analysis_data(article)

            f.write("#### Dimensional Scores\n\n")

            for dim in dimensions:
                if dim in analysis:
                    # Handle both flat integers and nested dicts
                    if isinstance(analysis[dim], dict):
                        score = analysis[dim].get('score', 'N/A')
                    elif isinstance(analysis[dim], int):
                        score = analysis[dim]
                    else:
                        continue

                    f.write(f"- **{dim.replace('_', ' ').title()}**: {score}/10\n")

            f.write("\n")

            # Overall reasoning (batch-labeled data has one overall reasoning, not per-dimension)
            if 'uplifting_analysis' in article and 'reasoning' in article['uplifting_analysis']:
                f.write("#### Overall Analysis Reasoning\n\n")
                f.write(f"{article['uplifting_analysis']['reasoning']}\n\n")

            f.write("---\n\n")

        # Key Insights
        f.write("## Key Insights\n\n")

        # Find dimensions with missing score 9
        missing_9 = []
        for dim in dimensions:
            if dim in stats and 9 not in stats[dim]['distribution']:
                missing_9.append(dim)

        if missing_9:
            f.write(f"### Missing Score 9 Examples\n\n")
            f.write(f"The following dimensions have **ZERO examples** of score 9 across all 20k articles:\n\n")
            for dim in missing_9:
                f.write(f"- **{dim.replace('_', ' ').title()}**\n")
            f.write("\nThis indicates these are extremely rare in real-world news content.\n\n")

        # Dimensions with good coverage
        good_coverage = []
        for dim in dimensions:
            if dim in stats:
                count_9 = stats[dim]['distribution'].get(9, 0)
                count_10 = stats[dim]['distribution'].get(10, 0)
                if count_9 >= 50:
                    good_coverage.append((dim, count_9, count_10))

        if good_coverage:
            f.write(f"### Dimensions with Excellent Extreme Score Coverage\n\n")
            f.write("Dimensions with 50+ examples of score 9:\n\n")
            for dim, count_9, count_10 in good_coverage:
                f.write(f"- **{dim.replace('_', ' ').title()}**: {count_9} score-9 examples, {count_10} score-10 examples\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("*Report generated from 10k filtered labeled articles using Gemini Flash (content ≥300 chars, framework leaks removed)*\n")

if __name__ == '__main__':
    input_file = 'datasets/ground_truth_filtered_10k/labeled_articles.jsonl'
    output_file = 'reports/ground_truth_10k_filtered_summary.md'

    print(f"Loading data from {input_file}...")
    articles = load_labeled_data(input_file)
    print(f"Loaded {len(articles):,} articles")

    print(f"Generating report to {output_file}...")
    generate_markdown_report(articles, output_file)
    print("Report generated successfully!")
