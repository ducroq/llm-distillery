"""
Base PreFilter Class

Provides common functionality for all semantic prefilters:
- Comprehensive text cleaning (Unicode, HTML, invisible chars, security)
- Standard interface for should_label()
- Shared utilities

All filter prefilters should inherit from this base class.
"""

from typing import Dict, Tuple
import sys
from pathlib import Path

# Import text cleaning from ground_truth module
sys.path.insert(0, str(Path(__file__).parent.parent / 'ground_truth'))
from text_cleaning import sanitize_text_comprehensive, clean_article as clean_article_comprehensive


class BasePreFilter:
    """
    Base class for semantic prefilters.

    Provides automatic Unicode sanitization and standard interface.
    Subclasses should implement should_label() method.
    """

    VERSION = "0.0"  # Override in subclass
    MIN_CONTENT_LENGTH = 300  # Minimum content length to prevent framework leakage

    @staticmethod
    def sanitize_unicode(text: str) -> str:
        """
        Remove surrogate characters and other invalid Unicode sequences.

        DEPRECATED: Use sanitize_text_comprehensive() for better cleaning.
        This method is kept for backward compatibility.

        Args:
            text: String that may contain invalid Unicode

        Returns:
            Cleaned string with invalid Unicode removed
        """
        if not isinstance(text, str):
            return str(text)
        # Encode with errors='ignore' to drop surrogates, then decode
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

    @staticmethod
    def sanitize_text_comprehensive(text: str) -> str:
        """
        Comprehensively clean text for LLM processing.

        Removes:
        - Invalid Unicode (surrogates)
        - HTML entities and tags
        - Zero-width characters (invisible text)
        - Bidirectional marks (security issue)
        - Normalizes whitespace

        This is the RECOMMENDED method for cleaning text.

        Args:
            text: String to clean

        Returns:
            Comprehensively cleaned string
        """
        return sanitize_text_comprehensive(text)

    @staticmethod
    def clean_article(article: Dict) -> Dict:
        """
        Recursively clean all text fields in an article.

        Applies comprehensive text cleaning to all strings:
        - Invalid Unicode removal
        - HTML cleaning
        - Invisible character removal
        - Security (BiDi marks)
        - Whitespace normalization

        Safe to call on already-clean articles (idempotent).

        Args:
            article: Article dict with potentially problematic text

        Returns:
            New dict with all text fields comprehensively cleaned
        """
        return clean_article_comprehensive(article)

    @staticmethod
    def check_content_length(article: Dict, min_length: int = None) -> Tuple[bool, str]:
        """
        Check if article content meets minimum length requirement.

        Short articles (<300 chars) often cause LLMs to analyze the evaluation
        framework/prompt instead of the article content itself (framework leakage).

        Args:
            article: Dict with 'title' and 'text'/'content' keys
            min_length: Minimum content length in characters (defaults to MIN_CONTENT_LENGTH)

        Returns:
            (should_label, reason)
            - (True, "passed"): Content is long enough
            - (False, "content_too_short"): Content below minimum threshold
        """
        if min_length is None:
            min_length = BasePreFilter.MIN_CONTENT_LENGTH

        content = article.get('text', article.get('content', ''))
        content_length = len(content)

        if content_length < min_length:
            return (False, f"content_too_short_{content_length}chars")

        return (True, "passed")

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Subclasses MUST implement this method.

        Args:
            article: Dict with 'title' and 'text'/'content' keys

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, "reason"): Block with reason string
        """
        raise NotImplementedError("Subclasses must implement should_label()")
