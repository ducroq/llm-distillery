"""
Commerce Prefilter - Stratified Sampling for Calibration

Samples articles for oracle calibration with stratified approach:
- Commerce candidates: Electrek, URLs with "deal" patterns
- Journalism candidates: Reuters, BBC, Nature, etc.

Usage:
    python -m filters.common.commerce_prefilter.training.collect_examples \
        --output datasets/calibration/commerce_prefilter_v1/calibration_sample.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List


# Sources likely to contain commerce/deals content
COMMERCE_SOURCES = [
    'electrek',
    'engadget',
    'techcrunch',
    'theverge',
    'cnet',
]

# URL patterns indicating commerce content (exclude dealer/dealing)
COMMERCE_URL_PATTERNS = [
    r'/deals?[/-]',      # /deal/ or /deals/
    r'deal[s]?$',        # ends with deal or deals
    r'green-deals',
    r'best-.*-deals',
    r'prime-day',
    r'black-friday',
    r'cyber-monday',
]

# Quality journalism sources (negative examples)
JOURNALISM_SOURCES = [
    'reuters',
    'bbc',
    'nature',
    'science_arxiv',
    'science_biorxiv',
    'science_phys_org',
    'deutsche_welle',
    'le_monde',
    'el_pais',
    'nrc',
    'spiegel',
]


def is_commerce_source(article: Dict) -> bool:
    """Check if article is from known commerce source."""
    source = article.get('source', '').lower()
    return any(cs in source for cs in COMMERCE_SOURCES)


def has_commerce_url(article: Dict) -> bool:
    """Check if article URL indicates commerce content."""
    url = article.get('url', '').lower()
    # Exclude "dealer" and "dealing"
    if 'dealer' in url or 'dealing' in url:
        return False
    return any(re.search(pattern, url) for pattern in COMMERCE_URL_PATTERNS)


def is_journalism_source(article: Dict) -> bool:
    """Check if article is from quality journalism source."""
    source = article.get('source', '').lower()
    return any(js in source for js in JOURNALISM_SOURCES)


def load_and_categorize(input_path: Path) -> Dict[str, List[Dict]]:
    """
    Load articles and categorize into buckets.

    Returns dict with keys: commerce_source, commerce_url, journalism, other
    """
    buckets = {
        'commerce_source': [],
        'commerce_url': [],
        'journalism': [],
        'other': [],
    }

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip articles without content
            if not article.get('content') or len(article.get('content', '')) < 200:
                continue

            # Categorize
            if is_commerce_source(article):
                buckets['commerce_source'].append(article)
            elif has_commerce_url(article):
                buckets['commerce_url'].append(article)
            elif is_journalism_source(article):
                buckets['journalism'].append(article)
            else:
                buckets['other'].append(article)

    return buckets


def stratified_sample(
    buckets: Dict[str, List[Dict]],
    n_commerce_source: int = 150,
    n_commerce_url: int = 150,
    n_journalism: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Create stratified sample from categorized buckets.
    """
    random.seed(seed)

    samples = []

    # Sample from each bucket
    for bucket_name, target_n in [
        ('commerce_source', n_commerce_source),
        ('commerce_url', n_commerce_url),
        ('journalism', n_journalism),
    ]:
        available = buckets[bucket_name]
        n = min(target_n, len(available))

        if n < target_n:
            print(f"  Warning: Only {len(available)} available in {bucket_name}, wanted {target_n}")

        sampled = random.sample(available, n)

        # Tag with sampling source
        for article in sampled:
            article['_sample_bucket'] = bucket_name

        samples.extend(sampled)
        print(f"  {bucket_name}: {n} articles")

    # Shuffle final sample
    random.shuffle(samples)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified sample for commerce prefilter calibration"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("datasets/raw/master_dataset_20251009_20251124.jsonl"),
        help="Input raw dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("datasets/calibration/commerce_prefilter_v1/calibration_sample.jsonl"),
        help="Output sample file"
    )
    parser.add_argument(
        "--n-commerce-source",
        type=int,
        default=150,
        help="Number of articles from commerce sources (Electrek, etc.)"
    )
    parser.add_argument(
        "--n-commerce-url",
        type=int,
        default=150,
        help="Number of articles with commerce URL patterns"
    )
    parser.add_argument(
        "--n-journalism",
        type=int,
        default=200,
        help="Number of articles from journalism sources"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Stratified Sampling")
    print("=" * 60)

    # Load and categorize
    buckets = load_and_categorize(args.input)

    print(f"\nCategorized articles:")
    for bucket, articles in buckets.items():
        print(f"  {bucket}: {len(articles)}")

    # Sample
    print(f"\nSampling (seed={args.seed}):")
    samples = stratified_sample(
        buckets,
        n_commerce_source=args.n_commerce_source,
        n_commerce_url=args.n_commerce_url,
        n_journalism=args.n_journalism,
        seed=args.seed,
    )

    print(f"\nTotal sample: {len(samples)} articles")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for article in samples:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"Saved to: {args.output}")

    # Show sample titles
    print("\n--- Sample titles ---")
    for bucket in ['commerce_source', 'commerce_url', 'journalism']:
        print(f"\n{bucket}:")
        bucket_samples = [a for a in samples if a.get('_sample_bucket') == bucket][:3]
        for a in bucket_samples:
            print(f"  - {a.get('title', '')[:70]}...")


if __name__ == "__main__":
    main()
