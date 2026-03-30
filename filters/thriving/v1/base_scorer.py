"""
Thriving Content Filter v1 - Base Scorer Class

Renamed from uplifting (ADR-012: lens-aligned naming).
Dimension change: removed social_cohesion_impact (overlaps with Belonging).

Inherits all shared logic from FilterBaseScorer.
Defines filter-specific constants and prefilter loading.
"""

from filters.common.filter_base_scorer import FilterBaseScorer


class BaseThrivingScorer(FilterBaseScorer):
    """
    Abstract base class for thriving content scoring.

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    FILTER_NAME = "thriving"
    FILTER_VERSION = "1.0"

    DIMENSION_NAMES = [
        "human_wellbeing_impact",
        "justice_rights_impact",
        "evidence_level",
        "benefit_distribution",
        "change_durability",
    ]

    DIMENSION_WEIGHTS = {
        "human_wellbeing_impact": 0.40,
        "justice_rights_impact": 0.25,
        "evidence_level": 0.10,
        "benefit_distribution": 0.10,
        "change_durability": 0.15,
    }

    TIER_THRESHOLDS = [
        ("high", 7.0, "Verified, broadly beneficial, lasting positive change"),
        ("medium", 4.0, "Documented benefits with moderate reach or durability"),
        ("low", 0.0, "Speculation, elite-only benefits, or no documented wellbeing impact"),
    ]

    GATEKEEPER_DIMENSION = "evidence_level"
    GATEKEEPER_MIN = 3.0
    GATEKEEPER_CAP = 3.0

    def _load_prefilter(self):
        from filters.thriving.v1.prefilter import ThrivingPreFilterV1
        self.prefilter = ThrivingPreFilterV1()
