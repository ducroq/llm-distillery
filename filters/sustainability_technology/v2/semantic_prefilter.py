"""
Semantic (Zero-Shot) Pre-Filter for sustainability_technology v1

Uses transformer-based zero-shot classification instead of keyword matching.
More accurate but slower than keyword-based filtering.

Usage:
    - For batch processing: Use keyword prefilter (fast)
    - For high-precision filtering: Use semantic prefilter (accurate)
    - For production: Use semantic with caching
"""

from typing import Dict, Tuple, Optional
from transformers import pipeline
import torch


class SemanticPreFilter:
    """
    Semantic prefilter using zero-shot classification.

    Uses facebook/bart-large-mnli for content categorization.
    """

    def __init__(self,
                 model_name: str = "facebook/bart-large-mnli",
                 device: int = -1,  # -1 = CPU, 0 = GPU
                 confidence_threshold: float = 0.35):
        """
        Initialize semantic prefilter.

        Args:
            model_name: HuggingFace model ID
            device: -1 for CPU, 0 for GPU
            confidence_threshold: Minimum confidence for positive category (0.0-1.0)
                                 Lower = more permissive (fewer false negatives)
                                 Higher = more restrictive (fewer false positives)
        """
        print(f"Loading semantic classifier: {model_name}...")
        print(f"  Device: {'CPU' if device == -1 else f'GPU {device}'}")
        print(f"  Confidence threshold: {confidence_threshold}")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        self.confidence_threshold = confidence_threshold

        # Define candidate labels
        # These can be customized based on your data
        self.candidate_labels = self._get_default_labels()

        print(f"  Categories: {len(self.candidate_labels)}")
        print("Semantic classifier ready!\n")

    def _get_default_labels(self):
        """
        Default candidate labels for sustainability technology classification.

        Design principles:
        1. Positive category is specific and context-rich
        2. Negative categories cover common false positive sources
        3. Labels are mutually exclusive
        4. Natural language phrasing (helps model understand)

        Tune these based on your false positive/negative analysis.
        """
        return [
            # POSITIVE - what we want (be very specific)
            "sustainability technology, renewable energy, climate solutions, and environmental innovation",

            # NEGATIVE - what we don't want (common false positive sources)
            "professional sports, athletics, and sporting events",
            "entertainment, celebrities, pop culture, and show business",
            "military operations, warfare, conflict, and defense",
            "personal lifestyle, fashion, weddings, and consumer goods",
            "general news and current events",  # Catch-all for other topics
        ]

    def set_candidate_labels(self, labels: list):
        """
        Override default candidate labels.

        Args:
            labels: List of category labels (first should be positive category)

        Example:
            filter.set_candidate_labels([
                "climate technology and sustainable energy",
                "sports",
                "entertainment"
            ])
        """
        self.candidate_labels = labels
        print(f"Updated candidate labels ({len(labels)} categories)")

    def classify(self, text: str, max_length: int = 512) -> Dict:
        """
        Classify text into categories.

        Args:
            text: Article text (title + content)
            max_length: Truncate text to this length (for speed)

        Returns:
            {
                'top_category': str,      # Highest scoring category
                'confidence': float,      # Score for top category (0-1)
                'all_scores': dict,       # Scores for all categories
                'is_relevant': bool       # True if passes threshold
            }
        """
        # Truncate for speed (BART has 1024 token limit)
        text_truncated = text[:max_length]

        # Run classification
        result = self.classifier(
            text_truncated,
            candidate_labels=self.candidate_labels,
            multi_label=False  # Single best category
        )

        # Parse results
        top_category = result['labels'][0]
        top_confidence = result['scores'][0]

        # Check if it's the positive category (first in list)
        positive_category = self.candidate_labels[0]
        is_relevant = (top_category == positive_category and
                      top_confidence >= self.confidence_threshold)

        return {
            'top_category': top_category,
            'confidence': top_confidence,
            'all_scores': dict(zip(result['labels'], result['scores'])),
            'is_relevant': is_relevant
        }

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Apply semantic filter to article (compatible with BasePreFilter interface).

        Args:
            article: Article dict with 'title' and 'content' keys

        Returns:
            (should_score, reason):
                - (True, "passed_semantic"): Article is relevant
                - (False, "blocked_category:X"): Article is about category X
        """
        # Combine title and content
        title = article.get('title', '')
        content = article.get('content', '')
        text = f"{title} {content}".lower()

        # Classify
        result = self.classify(text)

        if result['is_relevant']:
            return (True, "passed_semantic")
        else:
            # Return the category it was classified as
            category = result['top_category'].split(',')[0].split(' ')[0]  # First word
            return (False, f"blocked_category:{category}")


# ============================================================================
# Configuration Guide
# ============================================================================
#
# ## Tuning Confidence Threshold
#
# confidence_threshold controls the tradeoff:
#   - 0.25-0.30: Very permissive (fewer false negatives, more false positives)
#   - 0.35-0.40: Balanced (recommended starting point)
#   - 0.45-0.50: Restrictive (fewer false positives, more false negatives)
#
# Test on 100 articles and adjust based on results.
#
# ## Customizing Labels
#
# 1. Start with defaults
# 2. Run on 100 sample articles
# 3. Check which category false positives get classified as
# 4. Refine label wording to be more discriminative
# 5. Re-test
#
# Example refinements:
#   - "sustainability technology" → "sustainability technology and climate solutions"
#   - "sports" → "professional sports games and athletic competitions"
#   - "news" → "general news and current events"
#
# More specific = better discrimination
#
# ## Speed Optimization
#
# - Use GPU (device=0): ~0.1 sec per article
# - Use CPU (device=-1): ~0.5-1 sec per article
# - Truncate text (max_length=256): Faster with minimal accuracy loss
# - Batch processing: Classify multiple articles at once (advanced)
#
# ============================================================================


if __name__ == "__main__":
    # Example usage
    filter = SemanticPreFilter(confidence_threshold=0.35)

    # Test articles
    test_articles = [
        {
            "title": "New Solar Panel Technology Achieves Record Efficiency",
            "content": "Researchers at MIT have developed a new type of solar panel..."
        },
        {
            "title": "Lakers Beat Warriors in Overtime Thriller",
            "content": "LeBron James scored 45 points as the Lakers defeated..."
        },
        {
            "title": "Celebrity Wedding: Star Marries in Lavish Ceremony",
            "content": "The actress wore a custom designer gown..."
        }
    ]

    print("=" * 60)
    print("Testing Semantic Prefilter")
    print("=" * 60)

    for article in test_articles:
        result = filter.classify(article['title'] + ' ' + article['content'])
        status = "✓ PASS" if result['is_relevant'] else "✗ BLOCK"

        print(f"\n{status} | {article['title'][:50]}")
        print(f"  Category: {result['top_category'][:50]}")
        print(f"  Confidence: {result['confidence']:.3f}")
