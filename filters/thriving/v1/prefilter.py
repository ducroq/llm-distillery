"""
Thriving Pre-Filter v1.0

Identical to uplifting v7 prefilter. The dimension change (removing
social_cohesion_impact) only affects oracle scoring and model training,
not the prefilter logic.

Renamed from UpliftingPreFilterV7 to ThrivingPreFilterV1 (ADR-012).

Blocks obvious low-value content before LLM labeling:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)
- Pure speculation articles (no documented outcomes)
- Code repositories and developer tutorials
"""

from filters.uplifting.v7.prefilter import UpliftingPreFilterV7


class ThrivingPreFilterV1(UpliftingPreFilterV7):
    """Fast rule-based pre-filter for thriving content v1.

    Inherits all logic from uplifting v7 prefilter — same rules apply.
    Only the class name changes for consistency with the thriving filter.
    """

    VERSION = "1.0"
