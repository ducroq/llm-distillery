#!/usr/bin/env python3
"""Apply prefilter to existing ground_truth_filtered_10k dataset and create curated set."""

import json
import sys
from pathlib import Path
from collections import Counter

# Import prefilter
sys.path.insert(0, str(Path('.') / 'filters' / 'uplifting' / 'v1'))
from prefilter import UpliftingPreFilterV1

def main():
    input_file = 'datasets/ground_truth_filtered_10k/labeled_articles.jsonl'
    output_dir = Path('datasets/ground_truth_curated_prefiltered')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'labeled_articles.jsonl'

    print("Applying prefilter to existing 10k dataset...")
    print("=" * 80)

    prefilter = UpliftingPreFilterV1()

    # Load all articles
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Loaded {len(articles):,} articles from {input_file}\n")

    # Apply prefilter
    passed = []
    blocked = []
    block_reasons = Counter()

    for idx, article in enumerate(articles, 1):
        should_label, reason = prefilter.should_label(article)

        if should_label:
            passed.append(article)
        else:
            blocked.append((article, reason))
            block_reasons[reason] += 1

        if idx % 1000 == 0:
            print(f"Processed {idx:,}/{len(articles):,} articles...")

    print(f"\nProcessed {len(articles):,} articles... DONE\n")

    # Show results
    print("=" * 80)
    print("PREFILTER RESULTS")
    print("=" * 80)
    print(f"Total articles:  {len(articles):,}")
    print(f"Passed:          {len(passed):,} ({len(passed)/len(articles)*100:.1f}%)")
    print(f"Blocked:         {len(blocked):,} ({len(blocked)/len(articles)*100:.1f}%)")

    print("\nBlock reasons:")
    for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason:<30} {count:>6,} ({count/len(blocked)*100:.1f}%)")

    # Write passed articles to output
    print(f"\nWriting {len(passed):,} passed articles to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in passed:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print("Done!")

    # Create checkpoint file with first 1000
    checkpoint_file = output_dir / 'checkpoint_1000.jsonl'
    checkpoint_size = min(1000, len(passed))

    print(f"\nCreating checkpoint with first {checkpoint_size:,} articles...")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        for article in passed[:checkpoint_size]:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"Checkpoint saved to {checkpoint_file}")

    # Analyze source distribution in passed articles
    from urllib.parse import urlparse

    def extract_domain(url):
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    print("\n" + "=" * 80)
    print("SOURCE DISTRIBUTION IN CURATED SET")
    print("=" * 80)

    domain_counts = Counter()
    for article in passed:
        url = article.get('url', '')
        if url:
            domain = extract_domain(url)
            domain_counts[domain] += 1

    print(f"\nTop 20 sources in curated set ({len(passed):,} articles):\n")
    print(f"{'Rank':<6} {'Domain':<40} {'Count':<8} {'%':<8}")
    print("-" * 68)

    for rank, (domain, count) in enumerate(domain_counts.most_common(20), 1):
        pct = (count / len(passed)) * 100
        print(f"{rank:<6} {domain:<40} {count:<8} {pct:>6.2f}%")

    # Show some blocked articles
    print("\n" + "=" * 80)
    print("SAMPLE BLOCKED ARTICLES (first 5 of each type)")
    print("=" * 80)

    samples_shown = Counter()
    for article, reason in blocked:
        if samples_shown[reason] < 5:
            print(f"\n[{reason}]")
            print(f"  Title: {article.get('title', 'Untitled')[:70]}")
            print(f"  URL: {article.get('url', 'No URL')[:70]}")
            samples_shown[reason] += 1

if __name__ == '__main__':
    main()
