#!/usr/bin/env python3
"""Analyze the checkpoint_1000 curated dataset."""

import json
from urllib.parse import urlparse
from collections import Counter
from pathlib import Path
import sys

# Import post-classifier
sys.path.insert(0, str(Path('.') / 'filters' / 'uplifting' / 'v1'))
from post_classifier import UpliftingPostClassifierV1

def extract_domain(url):
    """Extract clean domain from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

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

def main():
    checkpoint_file = 'datasets/ground_truth_curated_prefiltered/checkpoint_1000.jsonl'

    print(f"Analyzing checkpoint: {checkpoint_file}")
    print("=" * 80)

    # Load articles
    articles = []
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"\nTotal articles in checkpoint: {len(articles):,}\n")

    # Source distribution
    domain_counts = Counter()
    for article in articles:
        url = article.get('url', '')
        if url:
            domain = extract_domain(url)
            domain_counts[domain] += 1

    print("TOP 20 SOURCES IN CHECKPOINT:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Domain':<40} {'Count':<8} {'%':<8}")
    print("-" * 80)

    for rank, (domain, count) in enumerate(domain_counts.most_common(20), 1):
        pct = (count / len(articles)) * 100
        print(f"{rank:<6} {domain:<40} {count:<8} {pct:>6.2f}%")

    # Analyze top-10 by weighted score
    print("\n" + "=" * 80)
    print("TOP-10 ARTICLES BY WEIGHTED SCORE:")
    print("=" * 80)

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
            domain = extract_domain(url) if url else 'unknown'

            scored_articles.append((title, url, domain, base_impact, weighted_score, category))

    # Sort by weighted score
    scored_articles.sort(key=lambda x: x[4], reverse=True)
    top_10 = scored_articles[:10]

    print(f"\n{'Rank':<6} {'Domain':<30} {'Score':<8} {'Category':<30}")
    print("-" * 80)
    for rank, (title, url, domain, base, weighted, category) in enumerate(top_10, 1):
        cat_label = category.replace('_', ' ').title()
        print(f"{rank:<6} {domain:<30} {weighted:>6.2f}  {cat_label:<30}")
        print(f"       {title[:70]}")
        print()

    # Domain distribution in top-10
    top10_domains = Counter(domain for _, _, domain, _, _, _ in top_10)

    print("=" * 80)
    print("DOMAIN DIVERSITY IN TOP-10:")
    print("-" * 80)
    for domain, count in top10_domains.most_common():
        print(f"  {domain:<40} {count} article(s)")

    print("\n" + "=" * 80)
    print("DIVERSITY METRICS:")
    print("-" * 80)

    top_20 = scored_articles[:20]
    top_50 = scored_articles[:50]

    top20_domains = Counter(domain for _, _, domain, _, _, _ in top_20)
    top50_domains = Counter(domain for _, _, domain, _, _, _ in top_50)

    print(f"Unique sources in top-10: {len(top10_domains)}")
    print(f"Unique sources in top-20: {len(top20_domains)}")
    print(f"Unique sources in top-50: {len(top50_domains)}")
    print(f"Unique sources in full checkpoint: {len(domain_counts)}")

    if len(top10_domains) > 0:
        top_source = top10_domains.most_common(1)[0]
        print(f"\nTop source in top-10: {top_source[0]} ({top_source[1]}/10 = {top_source[1]/10*100:.0f}%)")

    # Category distribution
    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTION IN TOP-10:")
    print("-" * 80)

    category_counts = Counter(category for _, _, _, _, _, category in top_10)
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        cat_label = category.replace('_', ' ').title()
        print(f"  {cat_label:<40} {count}")

if __name__ == '__main__':
    main()
