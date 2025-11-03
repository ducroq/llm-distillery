#!/usr/bin/env python3
"""Clean the expanded dataset by removing all blocked domain articles."""

import json
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

def extract_domain(url):
    """Extract clean domain from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def main():
    # Blocked domains from prefilter
    ACADEMIC_DOMAINS = [
        'arxiv.org',
        'biorxiv.org',
        'eprint.iacr.org',
        'mdpi.com',
        'medrxiv.org',
        'journals.plos.org',
        'frontiersin.org',
        'link.aps.org',
    ]

    CODE_HOSTING_DOMAINS = [
        'github.com',
        'gitlab.com',
    ]

    VC_STARTUP_DOMAINS = [
        'techcrunch.com',
        'crunchbase.com',
        'producthunt.com',
    ]

    DEFENSE_DOMAINS = [
        'defensenews.com',
        'janes.com',
        'defense.gov',
    ]

    ALL_BLOCKED = set(ACADEMIC_DOMAINS + CODE_HOSTING_DOMAINS + VC_STARTUP_DOMAINS + DEFENSE_DOMAINS)

    input_file = Path('datasets/ground_truth_combined_expanded/labeled_articles.jsonl')
    output_dir = Path('datasets/ground_truth_combined_expanded')
    output_file = output_dir / 'labeled_articles_cleaned.jsonl'

    print("Cleaning expanded dataset...")
    print("=" * 80)

    # Load all articles
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Loaded {len(articles):,} articles")

    # Check for blocked domains
    blocked_articles = []
    clean_articles = []
    blocked_by_domain = Counter()

    for article in articles:
        url = article.get('url', '')
        if url:
            domain = extract_domain(url)
            if domain in ALL_BLOCKED:
                blocked_articles.append(article)
                blocked_by_domain[domain] += 1
            else:
                clean_articles.append(article)
        else:
            # No URL, keep it
            clean_articles.append(article)

    print(f"\nBlocked articles found: {len(blocked_articles)}")
    if blocked_articles:
        print("\nBreakdown by domain:")
        for domain, count in sorted(blocked_by_domain.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain:<40} {count:>6} articles")

    print(f"\nClean articles: {len(clean_articles):,}")

    # Check for prompt leakage patterns
    print("\n" + "=" * 80)
    print("Checking for prompt leakage patterns...")
    print("=" * 80)

    leakage_patterns = [
        'Here is my analysis',
        'Based on the article',
        'Let me analyze',
        'I will evaluate',
        'My assessment',
        'Here are the scores',
        'Dimension scores:',
        'Reasoning:',
    ]

    leakage_found = []
    for article in clean_articles:
        # Check title and content
        title = article.get('title', '')
        content = article.get('content', article.get('text', ''))

        for pattern in leakage_patterns:
            if pattern.lower() in title.lower() or pattern.lower() in content[:500].lower():
                leakage_found.append({
                    'id': article.get('id'),
                    'title': title[:100],
                    'pattern': pattern,
                    'url': article.get('url', '')
                })
                break

    if leakage_found:
        print(f"\nWARNING: Found {len(leakage_found)} articles with potential prompt leakage:")
        for item in leakage_found[:10]:  # Show first 10
            print(f"  - {item['id']}: {item['pattern']}")
            print(f"    Title: {item['title']}")
    else:
        print("\n[OK] No obvious prompt leakage patterns detected")

    # Save cleaned dataset
    print("\n" + "=" * 80)
    print(f"Saving cleaned dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in clean_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\n[DONE] Cleaned dataset saved!")
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"  Original articles:     {len(articles):,}")
    print(f"  Blocked domains:       {len(blocked_articles):,}")
    print(f"  Prompt leakage:        {len(leakage_found):,}")
    print(f"  Final clean articles:  {len(clean_articles):,}")
    print(f"\n  Output: {output_file}")

if __name__ == '__main__':
    main()
