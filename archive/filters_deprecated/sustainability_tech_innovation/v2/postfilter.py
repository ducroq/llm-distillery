"""
Sustainability Tech Innovation v1 - Postfilter Implementation

This postfilter implements the tier classification logic for the
sustainability_tech_innovation filter, based on dimensional scores.

Tiers:
- BREAKTHROUGH: Mass deployment OR breakthrough innovation (>= 8.0)
- VALIDATED: Commercial deployment OR validated pilots (>= 6.0)
- PROMISING: Working pilots OR research with validation (>= 4.0)
- EARLY_STAGE: Lab-scale with some real data (>= 2.0)
- VAPORWARE: Theory only, no real results (< 2.0)

Gatekeeper Rules:
- IF deployment_maturity < 3.0 → cap overall score to 2.9 (early_stage max)
- IF proof_of_impact < 3.0 → cap overall score to 2.9 (early_stage max)

Usage:
    from filters.sustainability_tech_innovation.v1.postfilter import SustainabilityTechPostFilter

    postfilter = SustainabilityTechPostFilter()
    tier = postfilter.classify(scores)
    is_newsletter_worthy = postfilter.is_newsletter_worthy(scores)
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SustainabilityTechTier:
    """Sustainability tech tier classification result."""
    name: str  # 'BREAKTHROUGH', 'VALIDATED', 'PROMISING', 'EARLY_STAGE', or 'VAPORWARE'
    is_newsletter_worthy: bool  # True for BREAKTHROUGH, VALIDATED, PROMISING
    overall_score: float  # Weighted average score (0-10)
    reason: str
    scores: Dict[str, float]
    gatekeeper_capped: bool  # True if gatekeeper rule applied


class SustainabilityTechPostFilter:
    """
    Postfilter for sustainability_tech_innovation v1.

    Implements tier classification based on weighted dimensional scores
    with gatekeeper enforcement.

    Philosophy: "Tech that works" - deployed, pilots, or validated research.
    Not vaporware or pure theory.
    """

    # Dimension names in order
    DIMENSIONS = [
        'deployment_maturity',
        'technology_performance',
        'cost_trajectory',
        'scale_of_deployment',
        'market_penetration',
        'technology_readiness',
        'supply_chain_maturity',
        'proof_of_impact'
    ]

    # Dimension weights (from config.yaml)
    WEIGHTS = {
        'deployment_maturity': 0.20,
        'technology_performance': 0.15,
        'cost_trajectory': 0.15,
        'scale_of_deployment': 0.15,
        'market_penetration': 0.15,
        'technology_readiness': 0.10,
        'supply_chain_maturity': 0.05,
        'proof_of_impact': 0.05
    }

    # Tier thresholds
    BREAKTHROUGH_THRESHOLD = 8.0
    VALIDATED_THRESHOLD = 6.0
    PROMISING_THRESHOLD = 4.0
    EARLY_STAGE_THRESHOLD = 2.0

    # Gatekeeper thresholds
    DEPLOYMENT_GATEKEEPER = 3.0
    PROOF_GATEKEEPER = 3.0
    GATEKEEPER_MAX_SCORE = 2.9  # Caps to early_stage tier

    def calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted average of dimensional scores.

        Args:
            scores: Dictionary of dimension -> score (0-10)

        Returns:
            Weighted average score (0-10)
        """
        if not scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for dim, weight in self.WEIGHTS.items():
            if dim in scores:
                weighted_sum += scores[dim] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def apply_gatekeeper_rules(
        self,
        scores: Dict[str, float],
        weighted_score: float
    ) -> tuple[float, bool, str]:
        """
        Apply gatekeeper rules to cap overall score if needed.

        Gatekeeper Rules:
        - IF deployment_maturity < 3.0 → cap to 2.9 (early_stage max)
        - IF proof_of_impact < 3.0 → cap to 2.9 (early_stage max)

        Args:
            scores: Dictionary of dimension -> score
            weighted_score: Uncapped weighted average score

        Returns:
            (capped_score, was_capped, cap_reason)
        """
        deployment = scores.get('deployment_maturity', 0.0)
        proof = scores.get('proof_of_impact', 0.0)

        # Check gatekeeper rules
        if deployment < self.DEPLOYMENT_GATEKEEPER:
            return (
                min(weighted_score, self.GATEKEEPER_MAX_SCORE),
                True,
                f"Deployment maturity too low ({deployment:.1f} < {self.DEPLOYMENT_GATEKEEPER})"
            )

        if proof < self.PROOF_GATEKEEPER:
            return (
                min(weighted_score, self.GATEKEEPER_MAX_SCORE),
                True,
                f"Proof of impact too low ({proof:.1f} < {self.PROOF_GATEKEEPER})"
            )

        # No capping needed
        return (weighted_score, False, "No gatekeeper restrictions")

    def classify_tier(self, overall_score: float) -> str:
        """
        Classify tier based on overall score.

        Args:
            overall_score: Weighted average score (0-10)

        Returns:
            Tier name
        """
        if overall_score >= self.BREAKTHROUGH_THRESHOLD:
            return 'BREAKTHROUGH'
        elif overall_score >= self.VALIDATED_THRESHOLD:
            return 'VALIDATED'
        elif overall_score >= self.PROMISING_THRESHOLD:
            return 'PROMISING'
        elif overall_score >= self.EARLY_STAGE_THRESHOLD:
            return 'EARLY_STAGE'
        else:
            return 'VAPORWARE'

    def is_newsletter_worthy(self, tier_name: str) -> bool:
        """
        Determine if article is worthy of newsletter inclusion.

        Args:
            tier_name: Tier classification

        Returns:
            True if BREAKTHROUGH, VALIDATED, or PROMISING
        """
        return tier_name in ['BREAKTHROUGH', 'VALIDATED', 'PROMISING']

    def classify(self, scores: Dict[str, float]) -> SustainabilityTechTier:
        """
        Classify article into tier based on dimensional scores.

        Args:
            scores: Dictionary of dimension -> score (0-10)

        Returns:
            SustainabilityTechTier with classification results
        """
        # Calculate weighted score
        weighted_score = self.calculate_weighted_score(scores)

        # Apply gatekeeper rules
        overall_score, was_capped, cap_reason = self.apply_gatekeeper_rules(
            scores, weighted_score
        )

        # Classify tier
        tier_name = self.classify_tier(overall_score)

        # Determine newsletter worthiness
        newsletter_worthy = self.is_newsletter_worthy(tier_name)

        # Build reason
        if was_capped:
            reason = f"{tier_name} (capped: {cap_reason})"
        else:
            reason = f"{tier_name} (score: {overall_score:.1f})"

        return SustainabilityTechTier(
            name=tier_name,
            is_newsletter_worthy=newsletter_worthy,
            overall_score=overall_score,
            reason=reason,
            scores=scores,
            gatekeeper_capped=was_capped
        )

    def format_result(self, tier: SustainabilityTechTier) -> Dict:
        """
        Format tier result as dictionary for output.

        Args:
            tier: SustainabilityTechTier result

        Returns:
            Dictionary with tier classification and metadata
        """
        return {
            'tier': tier.name,
            'is_newsletter_worthy': tier.is_newsletter_worthy,
            'overall_score': round(tier.overall_score, 2),
            'reason': tier.reason,
            'gatekeeper_capped': tier.gatekeeper_capped,
            'scores': {k: round(v, 2) for k, v in tier.scores.items()}
        }


# Example usage
if __name__ == "__main__":
    postfilter = SustainabilityTechPostFilter()

    # Example 1: High-scoring breakthrough technology
    example_breakthrough = {
        'deployment_maturity': 9.0,
        'technology_performance': 8.5,
        'cost_trajectory': 8.0,
        'scale_of_deployment': 9.0,
        'market_penetration': 7.0,
        'technology_readiness': 9.0,
        'supply_chain_maturity': 8.0,
        'proof_of_impact': 9.0
    }

    # Example 2: Validated pilot with good data
    example_validated = {
        'deployment_maturity': 6.0,
        'technology_performance': 7.0,
        'cost_trajectory': 5.0,
        'scale_of_deployment': 4.0,
        'market_penetration': 3.0,
        'technology_readiness': 6.0,
        'supply_chain_maturity': 4.0,
        'proof_of_impact': 6.0
    }

    # Example 3: Early pilot but gatekeeper blocked (low deployment)
    example_early_capped = {
        'deployment_maturity': 2.5,  # Below gatekeeper threshold
        'technology_performance': 7.0,
        'cost_trajectory': 6.0,
        'scale_of_deployment': 2.0,
        'market_penetration': 1.0,
        'technology_readiness': 5.0,
        'supply_chain_maturity': 2.0,
        'proof_of_impact': 5.0
    }

    # Example 4: Pure vaporware
    example_vaporware = {
        'deployment_maturity': 1.0,
        'technology_performance': 2.0,
        'cost_trajectory': 1.0,
        'scale_of_deployment': 0.0,
        'market_penetration': 0.0,
        'technology_readiness': 2.0,
        'supply_chain_maturity': 1.0,
        'proof_of_impact': 1.0
    }

    print("=== Sustainability Tech Innovation Postfilter Examples ===\n")

    for name, scores in [
        ("Breakthrough Example", example_breakthrough),
        ("Validated Example", example_validated),
        ("Early Stage (Gatekeeper Capped)", example_early_capped),
        ("Vaporware Example", example_vaporware)
    ]:
        tier = postfilter.classify(scores)
        result = postfilter.format_result(tier)

        print(f"{name}:")
        print(f"  Tier: {result['tier']}")
        print(f"  Newsletter Worthy: {result['is_newsletter_worthy']}")
        print(f"  Overall Score: {result['overall_score']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Gatekeeper Capped: {result['gatekeeper_capped']}")
        print()
