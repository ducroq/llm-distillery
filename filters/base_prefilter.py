"""
Base PreFilter Class

Provides common functionality for all semantic prefilters:
- Unicode sanitization (removes surrogates and invalid UTF-8)
- Standard interface for should_label()
- Shared utilities

All filter prefilters should inherit from this base class.
"""

from typing import Dict, Tuple


class BasePreFilter:
    """
    Base class for semantic prefilters.

    Provides automatic Unicode sanitization and standard interface.
    Subclasses should implement should_label() method.
    """

    VERSION = "0.0"  # Override in subclass

    @staticmethod
    def sanitize_unicode(text: str) -> str:
        """
        Remove surrogate characters and other invalid Unicode sequences.

        Prevents encoding errors when processing articles scraped from the web.
        Invalid Unicode is silently dropped using errors='ignore'.

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
    def clean_article(article: Dict) -> Dict:
        """
        Recursively sanitize all text fields in an article.

        Removes invalid Unicode characters that cause encoding errors during
        processing or storage. Safe to call on already-clean articles.

        Args:
            article: Article dict with potentially invalid Unicode

        Returns:
            New dict with all text fields sanitized
        """
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean(item) for item in obj]
            elif isinstance(obj, str):
                return BasePreFilter.sanitize_unicode(obj)
            return obj

        return _clean(article)

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
