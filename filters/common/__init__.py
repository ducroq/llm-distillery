"""
Common utilities for all semantic filters.

This module provides shared functionality:
- BasePreFilter: Base class for all prefilters
- Text cleaning utilities
"""

from filters.common.base_prefilter import BasePreFilter
from filters.common.text_cleaning import (
    sanitize_text_comprehensive,
    clean_article,
    clean_article_for_labeling,
    batch_clean_articles,
)

__all__ = [
    'BasePreFilter',
    'sanitize_text_comprehensive',
    'clean_article',
    'clean_article_for_labeling',
    'batch_clean_articles',
]
