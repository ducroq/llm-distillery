"""
Belonging Filter v1 - Base Scorer Class

Inherits all shared logic from FilterBaseScorer.
Defines filter-specific constants and prefilter loading.
"""

from filters.common.filter_base_scorer import FilterBaseScorer


class BaseBelongingScorer(FilterBaseScorer):
    """
    Abstract base class for belonging scoring.

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    FILTER_NAME = "belonging"
    FILTER_VERSION = "1.0"

    DIMENSION_NAMES = [
        "intergenerational_bonds",
        "community_fabric",
        "reciprocal_care",
        "rootedness",
        "purpose_beyond_self",
        "slow_presence",
    ]

    DIMENSION_WEIGHTS = {
        "intergenerational_bonds": 0.25,
        "community_fabric": 0.25,
        "reciprocal_care": 0.10,
        "rootedness": 0.15,
        "purpose_beyond_self": 0.15,
        "slow_presence": 0.10,
    }

    TIER_THRESHOLDS = [
        ("high", 7.0, "Strong evidence of genuine belonging - thick social fabric, rootedness, intergenerational bonds"),
        ("medium", 4.0, "Some belonging elements present, moderate community connection"),
        ("low", 0.0, "Minimal belonging evidence, isolated/transactional relationships, or commercially-framed content"),
    ]

    GATEKEEPER_DIMENSION = "community_fabric"
    GATEKEEPER_MIN = 3.0
    GATEKEEPER_CAP = 3.42

    def _load_prefilter(self):
        import importlib.util
        prefilter_path = self._get_filter_dir() / "prefilter.py"
        spec = importlib.util.spec_from_file_location("prefilter", prefilter_path)
        prefilter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefilter_module)
        self.prefilter = prefilter_module.BelongingPreFilterV1()
