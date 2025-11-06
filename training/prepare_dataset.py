"""
Dataset preparation script for LLM Distillery.

Splits labeled ground truth data into train/val/test sets per filter configuration.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_filter_config(filter_path: Path) -> Dict:
    """Load filter configuration from config.yaml."""
    config_path = filter_path / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labeled_data(dataset_path: Path) -> List[Dict]:
    """Load labeled articles from JSONL file."""
    articles = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def extract_features_and_labels(
    article: Dict, filter_name: str, dimension_names: List[str]
) -> Tuple[Dict, List[float]]:
    """
    Extract features (article content) and labels (dimension scores) from article.

    Args:
        article: Article dictionary with content and analysis
        filter_name: Name of filter (e.g., 'uplifting')
        dimension_names: List of dimension names to extract

    Returns:
        features: Dict with id, title, content
        labels: List of dimension scores in order
    """
    # Extract text features
    features = {
        "id": article["id"],
        "title": article.get("title", ""),
        "content": article.get("content", ""),
        "url": article.get("url", ""),
    }

    # Extract dimension scores from filter analysis
    analysis_key = f"{filter_name}_analysis"
    if analysis_key not in article:
        raise ValueError(f"Article {article['id']} missing {analysis_key}")

    analysis = article[analysis_key]
    dimensions = analysis.get("dimensions", {})

    # Extract scores in consistent order
    labels = []
    for dim_name in dimension_names:
        if dim_name not in dimensions:
            raise ValueError(f"Article {article['id']} missing dimension {dim_name}")
        labels.append(float(dimensions[dim_name]))

    return features, labels


def split_dataset(
    articles: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split articles into train/val/test sets.

    Args:
        articles: List of article dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        train_articles, val_articles, test_articles
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled = articles.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split
    train_articles = shuffled[:n_train]
    val_articles = shuffled[n_train : n_train + n_val]
    test_articles = shuffled[n_train + n_val :]

    return train_articles, val_articles, test_articles


def prepare_split_file(
    articles: List[Dict],
    output_path: Path,
    filter_name: str,
    dimension_names: List[str],
) -> None:
    """
    Prepare split file with features and labels in JSONL format.

    Each line contains:
    {
        "id": "article_id",
        "title": "Article title",
        "content": "Article text",
        "labels": [score1, score2, ...],
        "dimension_names": ["dim1", "dim2", ...]
    }
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            features, labels = extract_features_and_labels(
                article, filter_name, dimension_names
            )

            # Combine into training format
            training_example = {
                **features,
                "labels": labels,
                "dimension_names": dimension_names,
            }

            f.write(json.dumps(training_example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset splits for filter training"
    )
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter directory (e.g., filters/uplifting/v1)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to labeled dataset JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Load filter config
    print(f"Loading filter config from {args.filter}")
    config = load_filter_config(args.filter)
    filter_name = config["filter"]["name"]

    # Extract dimension names from config (preserve order for consistency)
    dimension_names = list(config["scoring"]["dimensions"].keys())
    print(f"Filter: {filter_name}")
    print(f"Dimensions: {dimension_names}")

    # Load labeled data
    print(f"\nLoading labeled data from {args.dataset}")
    articles = load_labeled_data(args.dataset)
    print(f"Loaded {len(articles)} articles")

    # Split dataset
    print(f"\nSplitting dataset (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})")
    train_articles, val_articles, test_articles = split_dataset(
        articles,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_articles)} articles")
    print(f"Val: {len(val_articles)} articles")
    print(f"Test: {len(test_articles)} articles")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    print(f"\nSaving splits to {args.output_dir}")
    prepare_split_file(
        train_articles,
        args.output_dir / "train.jsonl",
        filter_name,
        dimension_names,
    )
    prepare_split_file(
        val_articles,
        args.output_dir / "val.jsonl",
        filter_name,
        dimension_names,
    )
    prepare_split_file(
        test_articles,
        args.output_dir / "test.jsonl",
        filter_name,
        dimension_names,
    )

    # Save split metadata
    metadata = {
        "filter_name": filter_name,
        "filter_version": config["filter"]["version"],
        "dimension_names": dimension_names,
        "total_articles": len(articles),
        "train_articles": len(train_articles),
        "val_articles": len(val_articles),
        "test_articles": len(test_articles),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }

    metadata_path = args.output_dir / "split_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset preparation complete!")
    print(f"Splits saved to: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
