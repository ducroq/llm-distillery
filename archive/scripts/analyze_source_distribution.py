#!/usr/bin/env python3
"""Analyze source distribution in the filtered 10k dataset."""

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
    # Remove www. prefix
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
    input_file = 'datasets/ground_truth_filtered_10k/labeled_articles.jsonl'

    print(f"Analyzing source distribution in {input_file}...")
    print("=" * 80)

    # Load all articles
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"\nTotal articles: {len(articles):,}\n")

    # Count by domain
    domain_counts = Counter()
    for article in articles:
        url = article.get('url', '')
        if url:
            domain = extract_domain(url)
            domain_counts[domain] += 1

    # Show top 20 sources
    print("TOP 20 SOURCES (by article count):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Domain':<40} {'Count':<8} {'%':<8}")
    print("-" * 80)

    for rank, (domain, count) in enumerate(domain_counts.most_common(20), 1):
        pct = (count / len(articles)) * 100
        print(f"{rank:<6} {domain:<40} {count:<8} {pct:>6.2f}%")

    # Analyze top-10 articles
    print("\n" + "=" * 80)
    print("TOP-10 ARTICLES (by weighted score):")
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

    # Show top-10 with domains
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
    print("DOMAIN DISTRIBUTION IN TOP-10:")
    print("-" * 80)
    for domain, count in top10_domains.most_common():
        print(f"  {domain:<40} {count} article(s)")

    # Compare to top-20, top-50
    top_20 = scored_articles[:20]
    top_50 = scored_articles[:50]

    top20_domains = Counter(domain for _, _, domain, _, _, _ in top_20)
    top50_domains = Counter(domain for _, _, domain, _, _, _ in top_50)

    print("\n" + "=" * 80)
    print("DIVERSITY ANALYSIS:")
    print("-" * 80)
    print(f"Unique sources in top-10: {len(top10_domains)}")
    print(f"Unique sources in top-20: {len(top20_domains)}")
    print(f"Unique sources in top-50: {len(top50_domains)}")
    print(f"Unique sources in full dataset: {len(domain_counts)}")

    print("\n" + "=" * 80)
    print("TOP SOURCE CONCENTRATION:")
    print("-" * 80)
    top_source = top10_domains.most_common(1)[0]
    print(f"Top source in top-10: {top_source[0]} ({top_source[1]}/10 = {top_source[1]/10*100:.0f}%)")

    if len(top20_domains) > 0:
        top_source_20 = top20_domains.most_common(1)[0]
        print(f"Top source in top-20: {top_source_20[0]} ({top_source_20[1]}/20 = {top_source_20[1]/20*100:.0f}%)")

    if len(top50_domains) > 0:
        top_source_50 = top50_domains.most_common(1)[0]
        print(f"Top source in top-50: {top_source_50[0]} ({top_source_50[1]}/50 = {top_source_50[1]/50*100:.0f}%)")

if __name__ == '__main__':
    main()
