"""
Create a random sample from a JSONL dataset.

Simple utility for validation and calibration workflows.
"""

import json
import random
import argparse
from pathlib import Path


def create_random_sample(
    source_path: str,
    output_path: str,
    sample_size: int,
    seed: int = 42
):
    """
    Create a random sample from a JSONL file.

    Args:
        source_path: Path to source JSONL file
        output_path: Path to output JSONL file
        sample_size: Number of articles to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load all articles
    print(f"Loading articles from {source_path}...")
    articles = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(articles):,} articles")

    # Sample
    if len(articles) <= sample_size:
        print(f"WARNING: Only {len(articles)} articles available, using all")
        sample = articles
    else:
        sample = random.sample(articles, sample_size)
        print(f"Sampled {sample_size} articles (seed={seed})")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for article in sample:
            f.write(json.dumps(article, ensure_ascii=False, separators=(',', ':')) + '\n')

    print(f"Wrote {len(sample)} articles to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create random sample from JSONL dataset')
    parser.add_argument('--source', required=True, help='Source JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--sample-size', type=int, required=True, help='Number of articles to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    create_random_sample(
        source_path=args.source,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed
    )
