"""
Uplifting Post-Classifier v1.0

Categorizes and weights labeled articles based on emotional tone.
Uses existing VADER sentiment and emotion scores from metadata.

Categories:
- "Celebrating Progress": Pure uplifting content (joy > 0.15 or sentiment > 0.3)
- "Inspiring Through Adversity": Heavy but empowering (sadness/fear/anger + high justice/resilience)
- "Neutral/Technical": Academic/technical content (neutral > 0.85, joy < 0.05, gets 0.5x weight)

Purpose: Ensure final content is emotionally uplifting and actionable.
"""

from typing import Dict, Tuple, Optional


class UpliftingPostClassifierV1:
    """Post-classifier for categorizing and weighting uplifting content by emotional tone"""

    VERSION = "1.0"

    # Thresholds for emotional tone categorization
    CELEBRATING_JOY_THRESHOLD = 0.15
    CELEBRATING_SENTIMENT_THRESHOLD = 0.3

    NEUTRAL_THRESHOLD = 0.85
    NEUTRAL_MAX_JOY = 0.05

    ADVERSITY_NEGATIVE_THRESHOLD = 0.1
    ADVERSITY_MIN_JUSTICE_OR_RESILIENCE = 7

    # Weight multipliers
    TECHNICAL_WEIGHT_MULTIPLIER = 0.5
    HEAVY_BUT_INSPIRING_MULTIPLIER = 0.8  # For traumatic/heavy topics with high justice/resilience

    # Academic domains that should always be classified as neutral/technical
    ACADEMIC_DOMAINS = [
        'arxiv.org',
        'biorxiv.org',
        'eprint.iacr.org',
    ]

    def classify_emotional_tone(self, article: Dict) -> Tuple[str, Dict]:
        """
        Classify article's emotional tone using existing sentiment/emotion data.

        Args:
            article: Dict with 'metadata' and 'uplifting_analysis' keys

        Returns:
            (category, details)
            - category: "celebrating_progress", "inspiring_through_adversity", "neutral_technical", or "unknown"
            - details: Dict with classification reasoning and scores
        """
        # Check URL domain first - academic articles are always neutral/technical
        url = article.get('url', '').lower()
        for domain in self.ACADEMIC_DOMAINS:
            if domain in url:
                return "neutral_technical", {
                    'url': url,
                    'domain': domain,
                    'reasoning': f"Academic domain ({domain}) - automatically classified as technical"
                }

        metadata = article.get('metadata', {})
        emotions = metadata.get('raw_emotions', {})
        sentiment = metadata.get('sentiment_raw_score', 0)

        # Extract emotion scores
        joy = emotions.get('joy', 0)
        sadness = emotions.get('sadness', 0)
        fear = emotions.get('fear', 0)
        anger = emotions.get('anger', 0)
        neutral = emotions.get('neutral', 0)

        # Extract dimension scores if available
        dimensions = {}
        if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
            dimensions = article['uplifting_analysis']['dimensions']
        elif 'analysis' in article:
            dimensions = article['analysis']

        # Extract justice and resilience scores
        justice = dimensions.get('justice', 0)
        resilience = dimensions.get('resilience', 0)

        # Category 1: Celebrating Progress (pure uplift)
        if joy > self.CELEBRATING_JOY_THRESHOLD or sentiment > self.CELEBRATING_SENTIMENT_THRESHOLD:
            return "celebrating_progress", {
                'joy': joy,
                'sentiment': sentiment,
                'reasoning': f"High joy ({joy:.3f}) or positive sentiment ({sentiment:.3f})"
            }

        # Category 2: Neutral/Technical (gets weight penalty)
        if neutral > self.NEUTRAL_THRESHOLD and joy < self.NEUTRAL_MAX_JOY:
            return "neutral_technical", {
                'neutral': neutral,
                'joy': joy,
                'reasoning': f"High neutral emotion ({neutral:.3f}), low joy ({joy:.3f})"
            }

        # Category 3: Inspiring Through Adversity (heavy but empowering)
        max_negative = max(sadness, fear, anger)
        if (max_negative > self.ADVERSITY_NEGATIVE_THRESHOLD and
            (justice >= self.ADVERSITY_MIN_JUSTICE_OR_RESILIENCE or
             resilience >= self.ADVERSITY_MIN_JUSTICE_OR_RESILIENCE)):
            return "inspiring_through_adversity", {
                'sadness': sadness,
                'fear': fear,
                'anger': anger,
                'justice': justice,
                'resilience': resilience,
                'reasoning': f"Elevated negative emotions (max={max_negative:.3f}) with high justice/resilience ({justice}/{resilience})"
            }

        # Category 4: Heavy But Inspiring (traumatic topics with some negative emotion, high justice/resilience)
        # This catches articles like rape trials that have high justice scores but aren't emotionally uplifting
        if (max_negative > 0.05 and  # Some negative emotion present
            max_negative <= self.ADVERSITY_NEGATIVE_THRESHOLD and  # But below adversity threshold
            (justice >= self.ADVERSITY_MIN_JUSTICE_OR_RESILIENCE or
             resilience >= self.ADVERSITY_MIN_JUSTICE_OR_RESILIENCE)):
            return "heavy_but_inspiring", {
                'sadness': sadness,
                'fear': fear,
                'anger': anger,
                'justice': justice,
                'resilience': resilience,
                'reasoning': f"Moderate negative emotions (max={max_negative:.3f}) with high justice/resilience ({justice}/{resilience}) - likely heavy/traumatic topic"
            }

        # Default: Unknown (doesn't fit clear pattern)
        return "unknown", {
            'joy': joy,
            'sentiment': sentiment,
            'neutral': neutral,
            'reasoning': "Doesn't match clear emotional tone pattern"
        }

    def calculate_weighted_score(self, article: Dict, base_impact_score: float) -> Tuple[float, str]:
        """
        Calculate weighted impact score based on emotional tone category.

        Args:
            article: Dict with metadata and uplifting_analysis
            base_impact_score: The original impact score from LLM dimensions

        Returns:
            (weighted_score, applied_weight_reason)
        """
        category, details = self.classify_emotional_tone(article)

        # Apply weight penalty for neutral/technical content
        if category == "neutral_technical":
            weighted_score = base_impact_score * self.TECHNICAL_WEIGHT_MULTIPLIER
            if 'domain' in details:
                reason = f"Applied 0.5x weight to academic domain ({details['domain']})"
            elif 'neutral' in details:
                reason = f"Applied 0.5x weight to neutral/technical content (neutral={details['neutral']:.3f})"
            else:
                reason = "Applied 0.5x weight to neutral/technical content"
            return weighted_score, reason

        # Apply weight penalty for heavy but inspiring content
        if category == "heavy_but_inspiring":
            weighted_score = base_impact_score * self.HEAVY_BUT_INSPIRING_MULTIPLIER
            justice = details.get('justice', 0)
            resilience = details.get('resilience', 0)
            reason = f"Applied 0.8x weight to heavy/traumatic content (justice={justice}, resilience={resilience})"
            return weighted_score, reason

        # All other categories keep original score
        return base_impact_score, "No weight adjustment applied"

    def should_include(self, article: Dict, base_impact_score: float,
                       min_weighted_score: Optional[float] = None,
                       exclude_categories: Optional[list] = None) -> Tuple[bool, str]:
        """
        Determine if article should be included in final dataset.

        Args:
            article: Dict with metadata and uplifting_analysis
            base_impact_score: The original impact score from LLM dimensions
            min_weighted_score: Optional minimum weighted score threshold
            exclude_categories: Optional list of categories to exclude

        Returns:
            (should_include, reason)
        """
        category, details = self.classify_emotional_tone(article)
        weighted_score, weight_reason = self.calculate_weighted_score(article, base_impact_score)

        # Check category exclusions
        if exclude_categories and category in exclude_categories:
            return False, f"Excluded category: {category}"

        # Check minimum weighted score
        if min_weighted_score is not None and weighted_score < min_weighted_score:
            return False, f"Weighted score {weighted_score:.2f} below threshold {min_weighted_score:.2f}"

        return True, f"Included (category={category}, weighted_score={weighted_score:.2f})"

    def get_statistics(self) -> Dict:
        """Return classifier statistics and thresholds"""
        return {
            'version': self.VERSION,
            'celebrating_joy_threshold': self.CELEBRATING_JOY_THRESHOLD,
            'celebrating_sentiment_threshold': self.CELEBRATING_SENTIMENT_THRESHOLD,
            'neutral_threshold': self.NEUTRAL_THRESHOLD,
            'neutral_max_joy': self.NEUTRAL_MAX_JOY,
            'adversity_negative_threshold': self.ADVERSITY_NEGATIVE_THRESHOLD,
            'adversity_min_justice_or_resilience': self.ADVERSITY_MIN_JUSTICE_OR_RESILIENCE,
            'technical_weight_multiplier': self.TECHNICAL_WEIGHT_MULTIPLIER,
        }


def test_post_classifier():
    """Test the post-classifier with sample articles"""

    classifier = UpliftingPostClassifierV1()

    test_cases = [
        # Case 1: Celebrating Progress
        {
            'name': 'High Joy - Community Garden Success',
            'article': {
                'metadata': {
                    'sentiment_raw_score': 0.45,
                    'raw_emotions': {
                        'joy': 0.25,
                        'neutral': 0.60,
                        'sadness': 0.05,
                        'fear': 0.03,
                        'anger': 0.02,
                        'disgust': 0.05
                    }
                },
                'uplifting_analysis': {
                    'dimensions': {
                        'agency': 8,
                        'progress': 7,
                        'collective_benefit': 9,
                        'connection': 8,
                        'innovation': 6,
                        'justice': 6,
                        'resilience': 7,
                        'wonder': 5
                    }
                }
            },
            'base_impact_score': 7.2,
            'expected_category': 'celebrating_progress'
        },

        # Case 2: Neutral/Technical (ArXiv paper)
        {
            'name': 'Neutral/Technical - Academic Paper',
            'article': {
                'metadata': {
                    'sentiment_raw_score': 0.05,
                    'raw_emotions': {
                        'joy': 0.01,
                        'neutral': 0.92,
                        'sadness': 0.02,
                        'fear': 0.02,
                        'anger': 0.01,
                        'disgust': 0.02
                    }
                },
                'uplifting_analysis': {
                    'dimensions': {
                        'agency': 6,
                        'progress': 8,
                        'collective_benefit': 7,
                        'connection': 4,
                        'innovation': 9,
                        'justice': 5,
                        'resilience': 5,
                        'wonder': 7
                    }
                }
            },
            'base_impact_score': 6.8,
            'expected_category': 'neutral_technical',
            'expected_weighted_score': 3.4  # 6.8 * 0.5
        },

        # Case 3: Inspiring Through Adversity
        {
            'name': 'Inspiring Through Adversity - Justice/Resilience',
            'article': {
                'metadata': {
                    'sentiment_raw_score': -0.15,
                    'raw_emotions': {
                        'joy': 0.08,
                        'neutral': 0.45,
                        'sadness': 0.20,
                        'fear': 0.12,
                        'anger': 0.10,
                        'disgust': 0.05
                    }
                },
                'uplifting_analysis': {
                    'dimensions': {
                        'agency': 7,
                        'progress': 6,
                        'collective_benefit': 8,
                        'connection': 7,
                        'innovation': 5,
                        'justice': 9,
                        'resilience': 8,
                        'wonder': 6
                    }
                }
            },
            'base_impact_score': 7.5,
            'expected_category': 'inspiring_through_adversity'
        },

        # Case 4: Unknown
        {
            'name': 'Unknown - Mixed Emotions',
            'article': {
                'metadata': {
                    'sentiment_raw_score': 0.10,
                    'raw_emotions': {
                        'joy': 0.12,
                        'neutral': 0.65,
                        'sadness': 0.08,
                        'fear': 0.05,
                        'anger': 0.05,
                        'disgust': 0.05
                    }
                },
                'uplifting_analysis': {
                    'dimensions': {
                        'agency': 6,
                        'progress': 6,
                        'collective_benefit': 7,
                        'connection': 5,
                        'innovation': 6,
                        'justice': 5,
                        'resilience': 6,
                        'wonder': 5
                    }
                }
            },
            'base_impact_score': 6.0,
            'expected_category': 'unknown'
        },
    ]

    print("Testing Uplifting Post-Classifier v1.0")
    print("=" * 80)

    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 80)

        article = test['article']
        base_score = test['base_impact_score']

        # Test classification
        category, details = classifier.classify_emotional_tone(article)
        status = "[PASS]" if category == test['expected_category'] else "[FAIL]"
        print(f"{status} Category: {category} (expected: {test['expected_category']})")
        print(f"  Reasoning: {details['reasoning']}")

        # Test weighted score
        weighted_score, weight_reason = classifier.calculate_weighted_score(article, base_score)
        print(f"  Base score: {base_score:.2f} -> Weighted score: {weighted_score:.2f}")
        print(f"  Weight reason: {weight_reason}")

        if 'expected_weighted_score' in test:
            score_status = "[PASS]" if abs(weighted_score - test['expected_weighted_score']) < 0.01 else "[FAIL]"
            print(f"{score_status} Expected weighted score: {test['expected_weighted_score']:.2f}")

        # Test inclusion
        should_include, include_reason = classifier.should_include(article, base_score)
        print(f"  Should include: {should_include} - {include_reason}")

    print("\n" + "=" * 80)
    print("Post-Classifier Statistics:")
    for key, value in classifier.get_statistics().items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_post_classifier()
