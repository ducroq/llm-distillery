"""
Investment-Risk v3 - Postfilter Implementation

This postfilter implements the tier classification logic for the investment-risk
filter, based on dimensional scores from oracle or student model.

Tiers:
- RED: Act now - reduce risk immediately
- YELLOW: Monitor closely - prepare for defense
- GREEN: Consider buying - value emerging
- BLUE: Educational content - no immediate action
- NOISE: Ignore - no actionable signal

Usage:
    from filters.investment_risk.v2.postfilter import InvestmentRiskPostFilter

    postfilter = InvestmentRiskPostFilter()
    tier = postfilter.classify(scores)
    is_actionable = postfilter.is_actionable(scores)
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class InvestmentRiskTier:
    """Investment risk tier classification result."""
    name: str  # 'RED', 'YELLOW', 'GREEN', 'BLUE', or 'NOISE'
    is_actionable: bool  # True for RED, YELLOW, GREEN
    signal_strength: float  # Weighted score (0-10)
    reason: str
    scores: Dict[str, float]


class InvestmentRiskPostFilter:
    """
    Postfilter for investment-risk v2.

    Based on training results:
    - Validation MAE: 0.67 (excellent)
    - Training MAE: 0.62
    - Model: Qwen 2.5-1.5B with LoRA

    Tier Classification:
    - RED: High risk signals requiring immediate action
    - YELLOW: Warning signals requiring monitoring
    - GREEN: Value opportunities
    - BLUE: Educational content
    - NOISE: Low quality or non-actionable
    """

    # Dimension names in order
    DIMENSIONS = [
        'macro_risk_severity',
        'credit_market_stress',
        'market_sentiment_extremes',
        'valuation_risk',
        'policy_regulatory_risk',
        'systemic_risk',
        'evidence_quality',
        'actionability'
    ]

    # Dimension weights for signal strength calculation
    WEIGHTS = {
        'macro_risk_severity': 0.25,
        'credit_market_stress': 0.20,
        'market_sentiment_extremes': 0.15,
        'valuation_risk': 0.15,
        'policy_regulatory_risk': 0.10,
        'systemic_risk': 0.15,
        'evidence_quality': 0.00,  # Gatekeeper, not in signal strength
        'actionability': 0.00  # Used in action_priority, not signal strength
    }

    # Tier thresholds
    RED_MACRO_THRESHOLD = 7
    RED_CREDIT_THRESHOLD = 7
    RED_SYSTEMIC_THRESHOLD = 8
    RED_EVIDENCE_MIN = 5
    RED_ACTIONABILITY_MIN = 5

    YELLOW_RISK_MIN = 5
    YELLOW_RISK_MAX = 6
    YELLOW_VALUATION_MIN = 7
    YELLOW_EVIDENCE_MIN = 5
    YELLOW_ACTIONABILITY_MIN = 4

    GREEN_SENTIMENT_MIN = 7  # Fear level
    GREEN_VALUATION_MAX = 3  # Cheap
    GREEN_EVIDENCE_MIN = 6
    GREEN_ACTIONABILITY_MIN = 5

    NOISE_EVIDENCE_MAX = 4  # Below this is noise (< 4 per config)

    def classify(self, scores: Dict[str, float]) -> InvestmentRiskTier:
        """
        Classify article into investment risk tier.

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            InvestmentRiskTier with classification result

        Example:
            >>> scores = {
            ...     'macro_risk_severity': 8,
            ...     'credit_market_stress': 7,
            ...     'systemic_risk': 7,
            ...     'evidence_quality': 6,
            ...     'actionability': 6,
            ...     ...
            ... }
            >>> tier = postfilter.classify(scores)
            >>> tier.name
            'RED'
        """
        macro = scores['macro_risk_severity']
        credit = scores['credit_market_stress']
        systemic = scores['systemic_risk']
        sentiment = scores['market_sentiment_extremes']
        valuation = scores['valuation_risk']
        evidence = scores['evidence_quality']
        actionability = scores['actionability']

        signal_strength = self.calculate_signal_strength(scores)

        # NOISE tier: Low evidence quality
        if evidence < self.NOISE_EVIDENCE_MAX:
            return InvestmentRiskTier(
                name='NOISE',
                is_actionable=False,
                signal_strength=0.0,
                reason=f'Low evidence quality ({evidence:.1f} < {self.NOISE_EVIDENCE_MAX})',
                scores=scores
            )

        # RED tier: Act now - reduce risk immediately
        # Condition: (macro >= 7 OR credit >= 7 OR systemic >= 8) AND evidence >= 5 AND actionability >= 5
        red_risk_trigger = (
            macro >= self.RED_MACRO_THRESHOLD or
            credit >= self.RED_CREDIT_THRESHOLD or
            systemic >= self.RED_SYSTEMIC_THRESHOLD
        )

        if red_risk_trigger and evidence >= self.RED_EVIDENCE_MIN and actionability >= self.RED_ACTIONABILITY_MIN:
            return InvestmentRiskTier(
                name='RED',
                is_actionable=True,
                signal_strength=signal_strength,
                reason=f'High risk signal (macro={macro:.1f}, credit={credit:.1f}, systemic={systemic:.1f})',
                scores=scores
            )

        # YELLOW tier: Monitor closely - prepare for defense
        # Condition: (macro 5-6 OR credit 5-6 OR valuation 7-8) AND evidence >= 5 AND actionability >= 4
        yellow_risk_trigger = (
            (self.YELLOW_RISK_MIN <= macro <= self.YELLOW_RISK_MAX) or
            (self.YELLOW_RISK_MIN <= credit <= self.YELLOW_RISK_MAX) or
            (valuation >= self.YELLOW_VALUATION_MIN)
        )

        if yellow_risk_trigger and evidence >= self.YELLOW_EVIDENCE_MIN and actionability >= self.YELLOW_ACTIONABILITY_MIN:
            return InvestmentRiskTier(
                name='YELLOW',
                is_actionable=True,
                signal_strength=signal_strength,
                reason=f'Warning signal (macro={macro:.1f}, credit={credit:.1f}, valuation={valuation:.1f})',
                scores=scores
            )

        # GREEN tier: Consider buying - value emerging
        # Condition: sentiment >= 7 (fear) AND valuation <= 3 (cheap) AND evidence >= 6 AND actionability >= 5
        green_opportunity = (
            sentiment >= self.GREEN_SENTIMENT_MIN and
            valuation <= self.GREEN_VALUATION_MAX and
            evidence >= self.GREEN_EVIDENCE_MIN and
            actionability >= self.GREEN_ACTIONABILITY_MIN
        )

        if green_opportunity:
            return InvestmentRiskTier(
                name='GREEN',
                is_actionable=True,
                signal_strength=signal_strength,
                reason=f'Value opportunity (sentiment={sentiment:.1f}, valuation={valuation:.1f})',
                scores=scores
            )

        # BLUE tier: Educational content - no immediate action
        # Articles that have some value but don't trigger action tiers
        if evidence >= 5 and actionability >= 3:
            return InvestmentRiskTier(
                name='BLUE',
                is_actionable=False,
                signal_strength=signal_strength,
                reason=f'Educational content (evidence={evidence:.1f}, actionability={actionability:.1f})',
                scores=scores
            )

        # NOISE tier: Default for low quality/non-actionable
        return InvestmentRiskTier(
            name='NOISE',
            is_actionable=False,
            signal_strength=0.0,
            reason=f'Low actionability or evidence (evidence={evidence:.1f}, actionability={actionability:.1f})',
            scores=scores
        )

    def is_actionable(self, scores: Dict[str, float]) -> bool:
        """
        Check if article requires action (RED, YELLOW, or GREEN tier).

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            True if article is actionable (RED/YELLOW/GREEN tier)
        """
        tier = self.classify(scores)
        return tier.is_actionable

    def calculate_signal_strength(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted signal strength (0-10).

        Uses dimension weights from config (excludes evidence_quality and actionability).

        Args:
            scores: Dictionary mapping dimension names to scores (0-10)

        Returns:
            Weighted signal strength (0-10)
        """
        weighted_sum = sum(
            scores[dim] * self.WEIGHTS[dim]
            for dim in self.DIMENSIONS
            if self.WEIGHTS[dim] > 0
        )
        # Normalize by sum of weights (should be 1.0)
        total_weight = sum(w for w in self.WEIGHTS.values() if w > 0)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

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
            'is_actionable': tier.is_actionable,
            'signal_strength': tier.signal_strength,
            'reason': tier.reason,
            'key_dimensions': {
                'macro_risk_severity': scores['macro_risk_severity'],
                'credit_market_stress': scores['credit_market_stress'],
                'systemic_risk': scores['systemic_risk'],
                'evidence_quality': scores['evidence_quality'],
                'actionability': scores['actionability']
            },
            'thresholds': {
                'RED': {
                    'macro': self.RED_MACRO_THRESHOLD,
                    'credit': self.RED_CREDIT_THRESHOLD,
                    'systemic': self.RED_SYSTEMIC_THRESHOLD,
                    'evidence_min': self.RED_EVIDENCE_MIN
                },
                'YELLOW': {
                    'risk_range': f'{self.YELLOW_RISK_MIN}-{self.YELLOW_RISK_MAX}',
                    'evidence_min': self.YELLOW_EVIDENCE_MIN
                },
                'GREEN': {
                    'sentiment_min': self.GREEN_SENTIMENT_MIN,
                    'valuation_max': self.GREEN_VALUATION_MAX,
                    'evidence_min': self.GREEN_EVIDENCE_MIN
                }
            }
        }


# Convenience functions

def classify_article(scores: Dict[str, float]) -> str:
    """
    Classify a single article (convenience function).

    Args:
        scores: Dictionary mapping dimension names to scores (0-10)

    Returns:
        Tier name: 'RED', 'YELLOW', 'GREEN', 'BLUE', or 'NOISE'

    Example:
        >>> from filters.investment_risk.v2.postfilter import classify_article
        >>> scores = {'macro_risk_severity': 8, ...}
        >>> classify_article(scores)
        'RED'
    """
    postfilter = InvestmentRiskPostFilter()
    tier = postfilter.classify(scores)
    return tier.name


def is_actionable(scores: Dict[str, float]) -> bool:
    """
    Check if article is actionable (convenience function).

    Args:
        scores: Dictionary mapping dimension names to scores (0-10)

    Returns:
        True if article requires action (RED/YELLOW/GREEN)

    Example:
        >>> from filters.investment_risk.v2.postfilter import is_actionable
        >>> scores = {'macro_risk_severity': 8, ...}
        >>> is_actionable(scores)
        True
    """
    postfilter = InvestmentRiskPostFilter()
    return postfilter.is_actionable(scores)


# Example usage and testing
if __name__ == '__main__':
    # Example 1: RED tier - High risk signal
    print("Example 1: High macro risk (RED tier)")
    scores_red = {
        'macro_risk_severity': 8,
        'credit_market_stress': 7,
        'market_sentiment_extremes': 6,
        'valuation_risk': 5,
        'policy_regulatory_risk': 4,
        'systemic_risk': 7,
        'evidence_quality': 6,
        'actionability': 6
    }

    postfilter = InvestmentRiskPostFilter()
    tier_red = postfilter.classify(scores_red)
    print(f"  Tier: {tier_red.name}")
    print(f"  Actionable: {tier_red.is_actionable}")
    print(f"  Signal Strength: {tier_red.signal_strength:.2f}")
    print(f"  Reason: {tier_red.reason}")
    print()

    # Example 2: YELLOW tier - Warning signal
    print("Example 2: Moderate risk (YELLOW tier)")
    scores_yellow = {
        'macro_risk_severity': 6,
        'credit_market_stress': 5,
        'market_sentiment_extremes': 4,
        'valuation_risk': 5,
        'policy_regulatory_risk': 3,
        'systemic_risk': 4,
        'evidence_quality': 6,
        'actionability': 5
    }

    tier_yellow = postfilter.classify(scores_yellow)
    print(f"  Tier: {tier_yellow.name}")
    print(f"  Actionable: {tier_yellow.is_actionable}")
    print(f"  Signal Strength: {tier_yellow.signal_strength:.2f}")
    print(f"  Reason: {tier_yellow.reason}")
    print()

    # Example 3: GREEN tier - Value opportunity
    print("Example 3: Value opportunity (GREEN tier)")
    scores_green = {
        'macro_risk_severity': 3,
        'credit_market_stress': 2,
        'market_sentiment_extremes': 8,  # High fear
        'valuation_risk': 2,  # Cheap
        'policy_regulatory_risk': 2,
        'systemic_risk': 2,
        'evidence_quality': 7,
        'actionability': 6
    }

    tier_green = postfilter.classify(scores_green)
    print(f"  Tier: {tier_green.name}")
    print(f"  Actionable: {tier_green.is_actionable}")
    print(f"  Signal Strength: {tier_green.signal_strength:.2f}")
    print(f"  Reason: {tier_green.reason}")
    print()

    # Example 4: NOISE tier - Low evidence quality
    print("Example 4: Low quality (NOISE tier)")
    scores_noise = {
        'macro_risk_severity': 5,
        'credit_market_stress': 4,
        'market_sentiment_extremes': 3,
        'valuation_risk': 4,
        'policy_regulatory_risk': 2,
        'systemic_risk': 3,
        'evidence_quality': 2,  # Too low
        'actionability': 3
    }

    tier_noise = postfilter.classify(scores_noise)
    print(f"  Tier: {tier_noise.name}")
    print(f"  Actionable: {tier_noise.is_actionable}")
    print(f"  Signal Strength: {tier_noise.signal_strength:.2f}")
    print(f"  Reason: {tier_noise.reason}")
    print()

    # Example 5: BLUE tier - Educational
    print("Example 5: Educational content (BLUE tier)")
    scores_blue = {
        'macro_risk_severity': 4,
        'credit_market_stress': 3,
        'market_sentiment_extremes': 3,
        'valuation_risk': 4,
        'policy_regulatory_risk': 3,
        'systemic_risk': 3,
        'evidence_quality': 6,  # Good quality
        'actionability': 4  # But not urgent
    }

    tier_blue = postfilter.classify(scores_blue)
    print(f"  Tier: {tier_blue.name}")
    print(f"  Actionable: {tier_blue.is_actionable}")
    print(f"  Signal Strength: {tier_blue.signal_strength:.2f}")
    print(f"  Reason: {tier_blue.reason}")
    print()

    # Summary
    print("="*60)
    print("Tier Summary")
    print("="*60)
    print("RED:    High risk - act now (reduce exposure)")
    print("YELLOW: Warning - monitor closely (prepare defense)")
    print("GREEN:  Opportunity - consider buying (value emerging)")
    print("BLUE:   Educational - understand (no action)")
    print("NOISE:  Ignore - low quality/non-actionable")
