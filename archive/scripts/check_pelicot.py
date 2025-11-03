import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.') / 'filters' / 'uplifting' / 'v1'))
from post_classifier import UpliftingPostClassifierV1

# Find the Pelicot article
with open('datasets/ground_truth_filtered_10k/labeled_articles.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            article = json.loads(line)
            if 'pelicot' in article.get('title', '').lower():
                print('FOUND PELICOT ARTICLE:')
                print('='*80)
                print(f"Title: {article.get('title', '')}")
                print(f"URL: {article.get('url', '')}\n")

                # Get metadata
                metadata = article.get('metadata', {})
                print('METADATA:')
                print(f"  Sentiment: {metadata.get('sentiment_raw_score', 0):.3f}")
                print(f"  Sentiment category: {metadata.get('sentiment_category', 'N/A')}")
                emotions = metadata.get('raw_emotions', {})
                print(f"  Emotions:")
                for emotion, score in emotions.items():
                    print(f"    {emotion}: {score:.3f}")

                # Get dimensions
                dimensions = article.get('uplifting_analysis', {}).get('dimensions', {})
                print(f"\nDIMENSIONS:")
                for dim, score in dimensions.items():
                    print(f"  {dim}: {score}")

                # Classify
                classifier = UpliftingPostClassifierV1()
                category, details = classifier.classify_emotional_tone(article)
                print(f"\nPOST-CLASSIFIER RESULT:")
                print(f"  Category: {category}")
                print(f"  Details: {details}")

                # Calculate impact
                weights = {
                    'agency': 1.0, 'progress': 1.0, 'collective_benefit': 1.5,
                    'connection': 0.8, 'innovation': 1.2, 'justice': 1.3,
                    'resilience': 1.0, 'wonder': 0.9
                }
                total_weighted = sum(dimensions.get(dim, 0) * w for dim, w in weights.items())
                total_weight = sum(weights.values())
                base = total_weighted / total_weight
                weighted, reason = classifier.calculate_weighted_score(article, base)
                print(f"\nSCORES:")
                print(f"  Base impact: {base:.2f}")
                print(f"  Weighted: {weighted:.2f}")
                print(f"  Reason: {reason}")

                break
