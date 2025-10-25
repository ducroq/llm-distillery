"""Stratified sampling strategies for ground truth generation."""

import json
import random
from pathlib import Path
from typing import Dict, List


class StratifiedSampler:
    """
    Stratified sampling to ensure representative ground truth dataset.

    Samples across:
    - Source categories
    - Time periods
    - Content characteristics (length, language)
    - Edge cases
    """

    def __init__(self, data_dir: str = "../content-aggregator/data/collected"):
        self.data_dir = Path(data_dir)

    def load_articles(self, max_articles: int = 100000) -> List[Dict]:
        """Load articles from JSONL files."""
        articles = []

        for jsonl_file in self.data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            articles.append(article)

                            if len(articles) >= max_articles:
                                return articles
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {jsonl_file}: {e}")
                continue

        return articles

    def stratified_sample(
        self,
        num_samples: int,
        categories: List[str],
        max_articles: int = 100000,
    ) -> List[Dict]:
        """
        Perform stratified sampling.

        Args:
            num_samples: Total number of samples needed
            categories: List of source categories to sample from
            max_articles: Maximum articles to load

        Returns:
            List of sampled articles
        """
        print(f"Loading articles from {self.data_dir}...")
        all_articles = self.load_articles(max_articles)
        print(f"Loaded {len(all_articles):,} articles")

        # Filter to relevant categories
        relevant_articles = [
            a for a in all_articles
            if a.get('metadata', {}).get('source_category') in categories
        ]
        print(f"Found {len(relevant_articles):,} articles in target categories")

        samples = []

        # 1. Category-stratified sampling (70%)
        category_samples = int(num_samples * 0.70)
        samples_per_category = category_samples // len(categories)

        for category in categories:
            category_articles = [
                a for a in relevant_articles
                if a.get('metadata', {}).get('source_category') == category
            ]

            if len(category_articles) > 0:
                n = min(samples_per_category, len(category_articles))
                samples.extend(random.sample(category_articles, n))

        # 2. Edge cases (20%)
        edge_samples = int(num_samples * 0.20)

        # Very short articles
        short = [a for a in relevant_articles if len(a.get('content', '')) < 500]
        if short:
            samples.extend(random.sample(short, min(edge_samples // 4, len(short))))

        # Very long articles
        long = [a for a in relevant_articles if len(a.get('content', '')) > 5000]
        if long:
            samples.extend(random.sample(long, min(edge_samples // 4, len(long))))

        # High sentiment (potential edge cases)
        high_sent = [
            a for a in relevant_articles
            if abs(a.get('metadata', {}).get('sentiment', {}).get('compound', 0)) > 0.8
        ]
        if high_sent:
            samples.extend(random.sample(high_sent, min(edge_samples // 4, len(high_sent))))

        # Random from remaining pool
        remaining = [a for a in relevant_articles if a not in samples]
        if remaining:
            samples.extend(random.sample(remaining, min(edge_samples // 4, len(remaining))))

        # 3. Random sampling (10%)
        random_samples = int(num_samples * 0.10)
        remaining = [a for a in relevant_articles if a not in samples]
        if remaining:
            samples.extend(random.sample(remaining, min(random_samples, len(remaining))))

        # Deduplicate and truncate
        samples = list({a['id']: a for a in samples}.values())[:num_samples]

        # Shuffle to avoid ordering bias
        random.shuffle(samples)

        print(f"Created stratified sample of {len(samples):,} articles")
        return samples
