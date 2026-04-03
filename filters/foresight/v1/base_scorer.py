"""
Foresight Filter v1 - Base Scorer Class

Inherits all shared logic from FilterBaseScorer.
Defines filter-specific constants and prefilter loading.
"""

from filters.common.filter_base_scorer import FilterBaseScorer


class BaseForesightScorer(FilterBaseScorer):
    """
    Abstract base class for foresight scoring.

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    FILTER_NAME = "foresight"
    FILTER_VERSION = "1.0"

    DIMENSION_NAMES = [
        "time_horizon",
        "systems_awareness",
        "course_correction",
        "intergenerational_investment",
        "institutional_durability",
        "evidence_foundation",
    ]

    DIMENSION_WEIGHTS = {
        "time_horizon": 0.25,
        "systems_awareness": 0.20,
        "course_correction": 0.20,
        "intergenerational_investment": 0.15,
        "institutional_durability": 0.10,
        "evidence_foundation": 0.10,
    }

    TIER_THRESHOLDS = [
        ("high", 7.0, "Landmark foresight — generational decisions, paradigm shifts, institutional transformation"),
        ("medium", 4.0, "Clear foresight signals — some long-term thinking, partial institutional embedding"),
        ("low", 0.0, "No foresight signals, short-term thinking, or rhetoric without action"),
    ]

    GATEKEEPER_DIMENSION = "evidence_foundation"
    GATEKEEPER_MIN = 3.0
    GATEKEEPER_CAP = 3.0

    def _load_prefilter(self):
        import importlib.util
        prefilter_path = self._get_filter_dir() / "prefilter.py"
        spec = importlib.util.spec_from_file_location("prefilter", prefilter_path)
        prefilter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefilter_module)
        self.prefilter = prefilter_module.ForesightPreFilterV1()
