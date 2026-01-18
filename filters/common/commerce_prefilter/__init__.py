"""
Commerce Prefilter SLM

ML-based classifier for detecting commerce/promotional content.
Cross-cutting prefilter that benefits all domain filters.

Usage:
    from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

    detector = CommercePrefilterSLM()
    result = detector.is_commerce(article)
    # {"is_commerce": True, "score": 0.87, "inference_time_ms": 23}
"""

from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

__all__ = ["CommercePrefilterSLM"]
