"""
Uplifting v4 - Postfilter Implementation

This postfilter applies tightened thresholds based on validation results
(75% oracle agreement with tendency to overcredit technical/commercial content).

Key Decision: Raise collective_benefit threshold from 5.0 to 6.5 to reduce
false positives (technical tools, academic papers, commercial products).

Usage:
    from filters.uplifting.v4.postfilter import UpliftingPostFilter

    postfilter = UpliftingPostFilter()
    tier = postfilter.classify(scores)
    is_uplifting = postfilter.is_uplifting(scores)
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class UpliftingTier:
    """Uplifting tier classification result."""
    name: str  # 'impact', 'connection', or 'not_uplifting'
    is_uplifting: bool
    collective_benefit: float
    avg_score: float
    reason: str


class UpliftingPostFilter:
    """
    Postfilter for uplifting v4 with tightened thresholds.

    Based on validation results:
    - Oracle agreement: 75% (9/12 correct)
    - Pattern: Overcredits CB=6 technical/commercial content
    - Solution: Raise connection threshold from CB≥5.0 to CB≥6.5

    Tier Classification:
    - impact: avg_score ≥ 7.0
    - connection: collective_benefit ≥ 6.5 OR (wonder ≥ 7.0 AND collective_benefit ≥ 3.0)
    - not_uplifting: below connection threshold
    """

    # Dimension names in order
    DIMENSIONS = [
        'agency',
        'progress',
        'collective_benefit',
        'connection',
        'innovation',
        'justice',
        'resilience',
        'wonder'
    ]

    # Dimension weights (from config.yaml)
    WEIGHTS = {
        'agency': 0.14,
        'progress': 0.19,
        'collective_benefit': 0.38,  # Gatekeeper dimension
        'connection': 0.10,
        'innovation': 0.08,
        'justice': 0.03,
        'resilience': 0.03,
        'wonder': 0.05
    }

    # Tier thresholds (TIGHTENED from validation)
    IMPACT_AVG_THRESHOLD = 7.0
    CONNECTION_CB_THRESHOLD = 6.5  # ← TIGHTENED from 5.0
    WONDER_EXCEPTION_THRESHOLD = 7.0
    WONDER_EXCEPTION_CB_MIN = 3.0

    def __init__(self, use_original_thresholds: bool = False):
        """
        Initialize postfilter.

        Args:
            use_original_thresholds: If True, use original CB≥5.0 threshold
                                    (for comparison/rollback)
        """
        self.use_original = use_original_thresholds
        if use_original_thresholds:
            self.connection_threshold = 5.0  # Original
        else:
            self.connection_threshold = self.CONNECTION_CB_THRESHOLD  # Tightened (6.5)

    def classify(self, scores: Dict[str, float]) -> UpliftingTier:
        """
        Classify article into uplifting tier.

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            UpliftingTier with classification result

        Example:
            >>> scores = {
            ...     'agency': 7, 'progress': 7, 'collective_benefit': 8,
            ...     'connection': 5, 'innovation': 6, 'justice': 5,
            ...     'resilience': 6, 'wonder': 6
            ... }
            >>> tier = postfilter.classify(scores)
            >>> tier.name
            'impact'
            >>> tier.is_uplifting
            True
        """
        cb = scores['collective_benefit']
        wonder = scores['wonder']
        avg_score = sum(scores.values()) / len(scores)

        # Impact tier: High overall impact
        if avg_score >= self.IMPACT_AVG_THRESHOLD:
            return UpliftingTier(
                name='impact',
                is_uplifting=True,
                collective_benefit=cb,
                avg_score=avg_score,
                reason=f'High impact (avg={avg_score:.1f} >= {self.IMPACT_AVG_THRESHOLD})'
            )

        # Connection tier: Moderate collective benefit
        if cb >= self.connection_threshold:
            return UpliftingTier(
                name='connection',
                is_uplifting=True,
                collective_benefit=cb,
                avg_score=avg_score,
                reason=f'Moderate uplifting (CB={cb:.1f} >= {self.connection_threshold})'
            )

        # Wonder exception: High wonder with minimum CB
        if wonder >= self.WONDER_EXCEPTION_THRESHOLD and cb >= self.WONDER_EXCEPTION_CB_MIN:
            return UpliftingTier(
                name='connection',
                is_uplifting=True,
                collective_benefit=cb,
                avg_score=avg_score,
                reason=f'Wonder exception (wonder={wonder:.1f} >= {self.WONDER_EXCEPTION_THRESHOLD}, CB={cb:.1f} >= {self.WONDER_EXCEPTION_CB_MIN})'
            )

        # Not uplifting
        return UpliftingTier(
            name='not_uplifting',
            is_uplifting=False,
            collective_benefit=cb,
            avg_score=avg_score,
            reason=f'Below threshold (CB={cb:.1f} < {self.connection_threshold}, avg={avg_score:.1f} < {self.IMPACT_AVG_THRESHOLD})'
        )

    def is_uplifting(self, scores: Dict[str, float]) -> bool:
        """
        Simple boolean check: is this article uplifting?

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            True if article passes uplifting threshold (impact or connection tier)

        Example:
            >>> scores = {'collective_benefit': 7, 'wonder': 3, ...}
            >>> postfilter.is_uplifting(scores)
            True
        """
        tier = self.classify(scores)
        return tier.is_uplifting

    def calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted average score using dimension weights.

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            Weighted average score (0-10)

        Note: This is different from simple average used in tier classification.
              Use this for ranking articles within a tier.
        """
        weighted_sum = sum(scores[dim] * self.WEIGHTS[dim] for dim in self.DIMENSIONS)
        return weighted_sum

    def get_tier_stats(self, scores: Dict[str, float]) -> Dict:
        """
        Get detailed statistics for an article.

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            Dictionary with tier, scores, and statistics
        """
        tier = self.classify(scores)

        return {
            'tier': tier.name,
            'is_uplifting': tier.is_uplifting,
            'reason': tier.reason,
            'scores': {
                'collective_benefit': tier.collective_benefit,
                'avg_score': tier.avg_score,
                'weighted_score': self.calculate_weighted_score(scores),
                'wonder': scores['wonder']
            },
            'thresholds': {
                'connection_cb': self.connection_threshold,
                'impact_avg': self.IMPACT_AVG_THRESHOLD,
                'wonder_exception': self.WONDER_EXCEPTION_THRESHOLD
            }
        }


# Convenience functions for single-call usage

def classify_article(scores: Dict[str, float], use_original_thresholds: bool = False) -> str:
    """
    Classify a single article (convenience function).

    Args:
        scores: Dictionary mapping dimension names to scores (0-10)
        use_original_thresholds: Use original CB≥5.0 threshold (for comparison)

    Returns:
        Tier name: 'impact', 'connection', or 'not_uplifting'

    Example:
        >>> from filters.uplifting.v4.postfilter import classify_article
        >>> scores = {'collective_benefit': 7, ...}
        >>> classify_article(scores)
        'connection'
    """
    postfilter = UpliftingPostFilter(use_original_thresholds=use_original_thresholds)
    tier = postfilter.classify(scores)
    return tier.name


def is_uplifting(scores: Dict[str, float], use_original_thresholds: bool = False) -> bool:
    """
    Check if article is uplifting (convenience function).

    Args:
        scores: Dictionary mapping dimension names to scores (0-10)
        use_original_thresholds: Use original CB≥5.0 threshold (for comparison)

    Returns:
        True if article is uplifting (impact or connection tier)

    Example:
        >>> from filters.uplifting.v4.postfilter import is_uplifting
        >>> scores = {'collective_benefit': 7, ...}
        >>> is_uplifting(scores)
        True
    """
    postfilter = UpliftingPostFilter(use_original_thresholds=use_original_thresholds)
    return postfilter.is_uplifting(scores)


# Example usage and testing
if __name__ == '__main__':
    # Example 1: High impact article
    print("Example 1: Indigenous cultural preservation (from validation)")
    scores_high = {
        'agency': 8,
        'progress': 7,
        'collective_benefit': 8,
        'connection': 7,
        'innovation': 6,
        'justice': 7,
        'resilience': 7,
        'wonder': 6
    }

    postfilter = UpliftingPostFilter()
    tier = postfilter.classify(scores_high)
    print(f"  Tier: {tier.name}")
    print(f"  Uplifting: {tier.is_uplifting}")
    print(f"  Reason: {tier.reason}")
    print()

    # Example 2: Overcredited technical article (from validation)
    print("Example 2: i18next translation tool (overcredited by oracle)")
    scores_technical = {
        'agency': 6,
        'progress': 6,
        'collective_benefit': 6,  # ← Oracle gave 7, should be 3-4
        'connection': 5,
        'innovation': 6,
        'justice': 3,
        'resilience': 4,
        'wonder': 3
    }

    tier_tech = postfilter.classify(scores_technical)
    print(f"  Tier: {tier_tech.name}")
    print(f"  Uplifting: {tier_tech.is_uplifting}")
    print(f"  Reason: {tier_tech.reason}")
    print(f"  Note: CB=6 is now filtered (would pass with original CB>=5.0)")
    print()

    # Example 3: Low scoring article
    print("Example 3: Black Friday promotion (from validation)")
    scores_low = {
        'agency': 2,
        'progress': 2,
        'collective_benefit': 3,
        'connection': 0,
        'innovation': 2,
        'justice': 0,
        'resilience': 0,
        'wonder': 0
    }

    tier_low = postfilter.classify(scores_low)
    print(f"  Tier: {tier_low.name}")
    print(f"  Uplifting: {tier_low.is_uplifting}")
    print(f"  Reason: {tier_low.reason}")
    print()

    # Comparison with original thresholds
    print("Comparison: Tightened (CB>=6.5) vs Original (CB>=5.0)")
    print("-" * 60)

    test_cases = [
        ("High impact (CB=8)", {'collective_benefit': 8, 'wonder': 6, 'agency': 7, 'progress': 7, 'connection': 7, 'innovation': 6, 'justice': 7, 'resilience': 7}),
        ("Technical tool (CB=6)", {'collective_benefit': 6, 'wonder': 3, 'agency': 6, 'progress': 6, 'connection': 5, 'innovation': 6, 'justice': 3, 'resilience': 4}),
        ("Low score (CB=3)", {'collective_benefit': 3, 'wonder': 0, 'agency': 2, 'progress': 2, 'connection': 0, 'innovation': 2, 'justice': 0, 'resilience': 0})
    ]

    postfilter_tight = UpliftingPostFilter(use_original_thresholds=False)
    postfilter_orig = UpliftingPostFilter(use_original_thresholds=True)

    for name, scores in test_cases:
        tier_tight = postfilter_tight.classify(scores)
        tier_orig = postfilter_orig.classify(scores)

        print(f"{name}:")
        print(f"  Tightened (CB>=6.5): {tier_tight.name} ({'PASS' if tier_tight.is_uplifting else 'FAIL'})")
        print(f"  Original (CB>=5.0):  {tier_orig.name} ({'PASS' if tier_orig.is_uplifting else 'FAIL'})")
        if tier_tight.name != tier_orig.name:
            print(f"  -> DIFFERENCE: Tightened filters out CB=6 technical content")
        print()
