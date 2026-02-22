"""
Investment Risk Filter v6 - Base Scorer Class

Inherits all shared logic from FilterBaseScorer.
Defines filter-specific constants and prefilter loading.
"""

from filters.common.filter_base_scorer import FilterBaseScorer


class BaseInvestmentRiskScorer(FilterBaseScorer):
    """
    Abstract base class for investment risk scoring.

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    FILTER_NAME = "investment-risk"
    FILTER_VERSION = "6.0"

    DIMENSION_NAMES = [
        "risk_domain_type",
        "severity_magnitude",
        "materialization_timeline",
        "evidence_quality",
        "impact_breadth",
        "retail_actionability",
    ]

    DIMENSION_WEIGHTS = {
        "risk_domain_type": 0.20,
        "severity_magnitude": 0.25,
        "materialization_timeline": 0.15,
        "evidence_quality": 0.15,
        "impact_breadth": 0.15,
        "retail_actionability": 0.10,
    }

    TIER_THRESHOLDS = [
        ("high", 7.0, "Critical risk signal - act now to reduce exposure"),
        ("medium_high", 5.0, "Elevated risk - monitor closely and prepare defense"),
        ("medium", 3.0, "Moderate signal - worth tracking, limited immediate action"),
        ("low", 0.0, "Low signal - noise, not investment-relevant, or already priced in"),
    ]

    GATEKEEPER_DIMENSION = "evidence_quality"
    GATEKEEPER_MIN = 4.0
    GATEKEEPER_CAP = 2.9

    def _load_prefilter(self):
        import importlib.util
        prefilter_path = self._get_filter_dir() / "prefilter.py"
        spec = importlib.util.spec_from_file_location("prefilter", prefilter_path)
        prefilter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefilter_module)
        self.prefilter = prefilter_module.InvestmentRiskPreFilterV5()
