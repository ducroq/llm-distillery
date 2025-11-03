#!/usr/bin/env python3
"""Generate top articles report in the visual story-focused format."""

import json
import sys
from pathlib import Path
from collections import Counter

# Import post-classifier
sys.path.insert(0, str(Path('.') / 'filters' / 'uplifting' / 'v1'))
from post_classifier import UpliftingPostClassifierV1

def calculate_impact_score(dimensions):
    """Calculate weighted average impact score."""
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

def get_category_label(category):
    """Convert category to display label."""
    labels = {
        'inspiring_through_adversity': 'Inspiring Through Adversity',
        'celebrating_progress': 'Celebrating Progress',
        'mixed_emotions': 'Mixed Emotions',
        'unknown': 'Mixed Emotions'
    }
    return labels.get(category, 'Mixed Emotions')

def get_category_marker(category):
    """Get markdown marker for category."""
    markers = {
        'inspiring_through_adversity': '●',  # Orange circle
        'celebrating_progress': '●',  # Green circle
        'mixed_emotions': '●',  # Gray circle
    }
    return markers.get(category, '●')

def extract_summary(article):
    """Extract or generate a summary of why the article matters."""
    # Use the article content to create a meaningful summary
    content = article.get('content', '').strip()
    title = article.get('title', '').strip()

    # If we have content, use first 300 characters as summary
    if content and len(content) > 100:
        summary = content[:400]
        # Find the last sentence boundary
        last_period = summary.rfind('.')
        last_exclamation = summary.rfind('!')
        last_question = summary.rfind('?')
        last_sentence = max(last_period, last_exclamation, last_question)

        if last_sentence > 100:  # Make sure we have a reasonable amount
            summary = summary[:last_sentence + 1]
        elif len(summary) > 300:
            summary = summary[:300] + '...'

        return summary

    # Fallback to title-based summary
    if title and len(title) > 10:
        return f"This article covers important developments related to: {title}."

    return "This article demonstrates positive progress and impact in its domain."

def main():
    dataset_file = 'datasets/uplifting_ground_truth_v1/labeled_articles.jsonl'
    output_file = 'reports/uplifting_ground_truth_v1_top_articles.md'

    print(f"Analyzing dataset: {dataset_file}")
    print("=" * 80)

    # Load articles
    articles = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Loaded {len(articles):,} articles\n")

    # Score all articles
    classifier = UpliftingPostClassifierV1()
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
            if isinstance(data, dict) and 'score' in data:
                dimensions[dim] = data['score']
            elif isinstance(data, int):
                dimensions[dim] = data

        if dimensions:
            base_impact = calculate_impact_score(dimensions)
            weighted_score, _ = classifier.calculate_weighted_score(article, base_impact)
            category, _ = classifier.classify_emotional_tone(article)

            title = article.get('title', 'Untitled')
            url = article.get('url', '')
            summary = extract_summary(article)

            scored_articles.append({
                'title': title,
                'url': url,
                'weighted_score': weighted_score,
                'base_impact': base_impact,
                'category': category,
                'summary': summary,
                'dimensions': dimensions
            })

    # Sort by weighted score
    scored_articles.sort(key=lambda x: x['weighted_score'], reverse=True)

    # Get top 50
    top_articles = scored_articles[:50]

    # Calculate category distribution
    category_counts = Counter(a['category'] for a in top_articles)

    # Generate report
    Path('reports').mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Title
        f.write("# Top 50 Uplifting News Articles\n\n")
        f.write("**Selected by Weighted Impact Score with Emotional Tone Classification**\n\n")

        # Introduction
        f.write(f"These articles were selected from a curated dataset of {len(articles):,} high-quality news articles, ")
        f.write("filtered for content quality and analyzed using AI across 8 dimensions. Each article's base impact score is ")
        f.write("weighted by emotional tone category to prioritize emotionally uplifting and actionable content. ")
        f.write("Technical/academic articles receive 0.5x weight multiplier.\n\n")

        # Category distribution
        cat_summary = []
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            cat_label = get_category_label(cat)
            cat_summary.append(f"{cat_label} ({count})")
        f.write(f"**Category distribution**: {', '.join(cat_summary)}\n\n")
        f.write("---\n\n")

        # Each article
        for rank, article in enumerate(top_articles, 1):
            category_label = get_category_label(article['category'])
            marker = get_category_marker(article['category'])

            f.write(f"## Article #{rank} - Weighted Score: {article['weighted_score']:.2f}/10\n\n")
            f.write(f"{marker} **{category_label}**\n\n")
            f.write(f"### {article['title']}\n\n")
            f.write(f"{article['url']}\n\n")
            f.write(f"**Why this article matters**: {article['summary']}\n\n")
            f.write("---\n\n")

        # Footer
        f.write("*Generated with Post-Classifier v1.0 - Uplifting Ground Truth v1 Dataset - Gemini Flash Analysis*\n")

    print(f"\nReport generated: {output_file}")
    print(f"\nTo convert to PDF with better formatting, use:")
    print(f"  pandoc {output_file} --pdf-engine=xelatex -V geometry:margin=1in -o reports/uplifting_ground_truth_v1_top_articles.pdf")

if __name__ == '__main__':
    main()
