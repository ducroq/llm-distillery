"""
Create random training sample from master datasets.

Simple random sampling - matches your production RSS feed distribution.

Usage:
    python create_sample.py [--size 2500] [--train-ratio 0.9]
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


def create_training_sample(sample_size=2500, train_ratio=0.9, quality_threshold=0.7):
    """
    Create random sample from all master datasets.

    Why random? Your data already reflects production reality:
    - 32% academic papers (ArXiv)
    - 27% news sources
    - 41% other sources

    Training on this mix = model handles production well.
    """

    print("="*70)
    print("CREATING TRAINING SAMPLE")
    print("="*70)
    print(f"Target size: {sample_size:,} articles")
    print(f"Train/val split: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print(f"Quality threshold: {quality_threshold}")
    print("="*70)
    print()

    # Load all articles
    all_articles = []
    raw_dir = Path("datasets/raw")

    for filepath in sorted(raw_dir.glob("master_dataset_*.jsonl")):
        print(f"Loading: {filepath.name}")
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    # Basic quality filter
                    quality = article.get('metadata', {}).get('quality_score', 1.0)
                    if quality >= quality_threshold:
                        all_articles.append(article)
                        count += 1
                except:
                    continue
        print(f"  Loaded: {count:,} articles (quality >= {quality_threshold})")

    print(f"\nTotal articles available: {len(all_articles):,}")

    if len(all_articles) < sample_size:
        print(f"WARNING: Only {len(all_articles):,} articles available, using all")
        sample_size = len(all_articles)

    # Random sample
    print(f"\nCreating random sample of {sample_size:,} articles...")
    random.seed(42)  # Reproducible
    sample = random.sample(all_articles, sample_size)

    # Shuffle and split
    random.shuffle(sample)
    split_idx = int(len(sample) * train_ratio)

    train = sample[:split_idx]
    val = sample[split_idx:]

    # Save
    output_dir = Path("datasets/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "sample_train.jsonl"
    val_file = output_dir / "sample_val.jsonl"

    print(f"\nSaving...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for article in train:
            f.write(json.dumps(article, ensure_ascii=False, separators=(',', ':')) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for article in val:
            f.write(json.dumps(article, ensure_ascii=False, separators=(',', ':')) + '\n')

    print(f"\n{'='*70}")
    print("SAMPLE CREATED")
    print(f"{'='*70}")
    print(f"Training set: {len(train):,} articles")
    print(f"  -> {train_file}")
    print(f"\nValidation set: {len(val):,} articles")
    print(f"  -> {val_file}")
    print(f"\nTotal: {len(sample):,} articles")

    # Quick stats
    sources = [a.get('source', 'unknown') for a in sample]
    word_counts = [a.get('metadata', {}).get('word_count', 0) for a in sample]
    languages = [a.get('language', 'unknown') for a in sample]

    print(f"\n{'='*70}")
    print("SAMPLE CHARACTERISTICS")
    print(f"{'='*70}")

    print(f"\nTop 10 sources:")
    for i, (source, count) in enumerate(Counter(sources).most_common(10), 1):
        pct = count/len(sample)*100
        print(f"  {i:2d}. {source:30s} {count:4,} ({pct:5.1f}%)")

    print(f"\nContent length:")
    print(f"  Median: {sorted(word_counts)[len(word_counts)//2]} words")
    short = sum(1 for w in word_counts if w < 100)
    medium = sum(1 for w in word_counts if 100 <= w < 500)
    long = sum(1 for w in word_counts if w >= 500)
    print(f"  Short (<100): {short:,} ({short/len(sample)*100:.1f}%)")
    print(f"  Medium (100-500): {medium:,} ({medium/len(sample)*100:.1f}%)")
    print(f"  Long (500+): {long:,} ({long/len(sample)*100:.1f}%)")

    print(f"\nLanguages:")
    for lang, count in Counter(languages).most_common(5):
        print(f"  {lang}: {count:,} ({count/len(sample)*100:.1f}%)")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("""
1. Label the training set with Gemini Flash:
   python -m ground_truth.batch_labeler \\
     --prompt prompts/uplifting.md \\
     --source datasets/training/sample_train.jsonl \\
     --output-dir datasets/labeled/uplifting \\
     --llm gemini-flash

2. Label the validation set:
   python -m ground_truth.batch_labeler \\
     --prompt prompts/uplifting.md \\
     --source datasets/training/sample_val.jsonl \\
     --output-dir datasets/labeled/uplifting_val \\
     --llm gemini-flash

3. See: docs/guides/ground-truth-generation.md for full workflow
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training sample')
    parser.add_argument('--size', type=int, default=2500,
                       help='Sample size (default: 2500)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Training proportion (default: 0.9)')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                       help='Minimum quality score (default: 0.7)')

    args = parser.parse_args()

    create_training_sample(
        sample_size=args.size,
        train_ratio=args.train_ratio,
        quality_threshold=args.quality_threshold
    )
