"""
Sustainability Technology v3 - Base Scorer Class

Inherits all shared logic from FilterBaseScorer.
Defines filter-specific constants and prefilter loading.
"""

from filters.common.filter_base_scorer import FilterBaseScorer


class BaseSustainabilityTechnologyScorer(FilterBaseScorer):
    """
    Abstract base class for sustainability technology scoring.

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    FILTER_NAME = "sustainability_technology"
    FILTER_VERSION = "3.0"

    DIMENSION_NAMES = [
        "technology_readiness_level",
        "technical_performance",
        "economic_competitiveness",
        "life_cycle_environmental_impact",
        "social_equity_impact",
        "governance_systemic_impact",
    ]

    DIMENSION_WEIGHTS = {
        "technology_readiness_level": 0.15,
        "technical_performance": 0.15,
        "economic_competitiveness": 0.20,
        "life_cycle_environmental_impact": 0.30,
        "social_equity_impact": 0.10,
        "governance_systemic_impact": 0.10,
    }

    TIER_THRESHOLDS = [
        ("high_sustainability", 7.0, "Mass deployed, proven sustainable, competitive"),
        ("medium_high", 5.0, "Commercial deployment, good sustainability"),
        ("medium", 3.0, "Pilot/early commercial, mixed profile"),
        ("low", 0.0, "Lab stage or poor sustainability performance"),
    ]

    GATEKEEPER_DIMENSION = "technology_readiness_level"
    GATEKEEPER_MIN = 3.0
    GATEKEEPER_CAP = 2.9

    def _load_prefilter(self):
        from filters.sustainability_technology.v3.prefilter import SustainabilityTechnologyPreFilterV2
        self.prefilter = SustainabilityTechnologyPreFilterV2()
