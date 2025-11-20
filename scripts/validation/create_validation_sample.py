"""
Create a random validation sample for oracle quality testing.

Usage:
    python scripts/create_validation_sample.py \
        --input datasets/raw/master_dataset_20251026_20251029.jsonl \
        --output filters/uplifting/v4/validation_sample.jsonl \
        --sample-size 100 \
        --seed 2025
"""

import json
import random
import argparse
from pathlib import Path


def load_articles(input_file: Path) -> list:
    """Load all articles from input file."""
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def create_random_sample(articles: list, sample_size: int, seed: int) -> list:
    """Create random sample of articles."""
    random.seed(seed)
    return random.sample(articles, min(sample_size, len(articles)))


def main():
    parser = argparse.ArgumentParser(description='Create validation sample for oracle testing')
    parser.add_argument('--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of articles to sample')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')

    args = parser.parse_args()

    print(f"Loading articles from: {args.input}")
    articles = load_articles(args.input)
    print(f"Loaded {len(articles)} articles")

    print(f"\nCreating random sample of {args.sample_size} articles (seed={args.seed})...")
    sample = create_random_sample(articles, args.sample_size, args.seed)

    # Save sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for article in sample:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\n✓ Saved {len(sample)} articles to: {args.output}")
    print(f"\nNext step: Score with oracle using:")
    print(f"  python -m ground_truth.batch_labeler \\")
    print(f"    --filter filters/uplifting/v4 \\")
    print(f"    --source {args.output} \\")
    print(f"    --output-dir filters/uplifting/v4/validation_scored \\")
    print(f"    --llm gemini-flash \\")
    print(f"    --batch-size 100 \\")
    print(f"    --max-batches 1")


if __name__ == '__main__':
    main()
