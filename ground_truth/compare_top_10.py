#!/usr/bin/env python3
"""Compare old vs new top-10 article selection to show post-classifier impact."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import post-classifier
sys.path.insert(0, str(Path(__file__).parent.parent / 'filters' / 'uplifting' / 'v1'))
from post_classifier import UpliftingPostClassifierV1

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

def get_article_scores(articles: List[Dict]) -> List[Tuple[str, str, float, float, str]]:
    """Get all articles with base and weighted scores."""
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
            # Calculate base impact score
            base_impact = calculate_impact_score(dimensions)

            # Apply post-classifier for weighted score and category
            weighted_score, _ = classifier.calculate_weighted_score(article, base_impact)
            category, _ = classifier.classify_emotional_tone(article)

            title = article.get('title', 'Untitled')
            url = article.get('url', 'No URL')

            scored_articles.append((title, url, base_impact, weighted_score, category))

    return scored_articles

def main():
    input_file = 'datasets/ground_truth_filtered_10k/labeled_articles.jsonl'

    print("Loading dataset...")
    articles = load_labeled_data(input_file)
    print(f"Loaded {len(articles):,} articles\n")

    print("Calculating scores...")
    scored = get_article_scores(articles)

    # Get old top-10 (base impact score)
    old_top10 = sorted(scored, key=lambda x: x[2], reverse=True)[:10]

    # Get new top-10 (weighted score)
    new_top10 = sorted(scored, key=lambda x: x[3], reverse=True)[:10]

    print("=" * 100)
    print("COMPARISON: Old Top-10 (Base Impact) vs New Top-10 (Weighted with Post-Classifier)")
    print("=" * 100)

    # Count categories in both
    print("\nCATEGORY DISTRIBUTION:\n")

    old_categories = {}
    for _, _, _, _, cat in old_top10:
        old_categories[cat] = old_categories.get(cat, 0) + 1

    new_categories = {}
    for _, _, _, _, cat in new_top10:
        new_categories[cat] = new_categories.get(cat, 0) + 1

    all_cats = set(list(old_categories.keys()) + list(new_categories.keys()))

    print(f"{'Category':<30} {'Old Top-10':<12} {'New Top-10':<12} {'Change':<8}")
    print("-" * 68)
    for cat in sorted(all_cats):
        cat_name = cat.replace('_', ' ').title()
        old_count = old_categories.get(cat, 0)
        new_count = new_categories.get(cat, 0)
        if old_count == new_count:
            change = "="
        elif new_count > old_count:
            change = f"+{new_count - old_count}"
        else:
            change = f"-{old_count - new_count}"
        print(f"{cat_name:<30} {old_count:<12} {new_count:<12} {change:<8}")

    # Show dropped articles
    old_urls = {url for _, url, _, _, _ in old_top10}
    new_urls = {url for _, url, _, _, _ in new_top10}

    dropped = old_urls - new_urls
    added = new_urls - old_urls

    if dropped:
        print(f"\n\nARTICLES DROPPED FROM TOP-10 ({len(dropped)}):\n")
        print("-" * 100)
        for title, url, base, weighted, cat in old_top10:
            if url in dropped:
                cat_name = cat.replace('_', ' ').title()
                print(f"* {title[:70]}")
                print(f"  Category: {cat_name}")
                print(f"  Base Score: {base:.2f} -> Weighted: {weighted:.2f} (penalty: {base - weighted:.2f})")
                print(f"  URL: {url}\n")

    if added:
        print(f"\n\nNEW ARTICLES IN TOP-10 ({len(added)}):\n")
        print("-" * 100)
        for title, url, base, weighted, cat in new_top10:
            if url in added:
                cat_name = cat.replace('_', ' ').title()
                print(f"* {title[:70]}")
                print(f"  Category: {cat_name}")
                print(f"  Base Score: {base:.2f} -> Weighted: {weighted:.2f} (boost: {weighted - base:.2f})")
                print(f"  URL: {url}\n")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Articles replaced: {len(dropped)}/{len(old_top10)}")
    print(f"Neutral/Technical articles dropped: {old_categories.get('neutral_technical', 0) - new_categories.get('neutral_technical', 0)}")
    print(f"Heavy But Inspiring articles dropped: {old_categories.get('heavy_but_inspiring', 0) - new_categories.get('heavy_but_inspiring', 0)}")
    print(f"Celebrating Progress articles: {old_categories.get('celebrating_progress', 0)} -> {new_categories.get('celebrating_progress', 0)}")
    print(f"Inspiring Through Adversity: {old_categories.get('inspiring_through_adversity', 0)} -> {new_categories.get('inspiring_through_adversity', 0)}")

if __name__ == '__main__':
    main()
