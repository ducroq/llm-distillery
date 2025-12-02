"""
Base PreFilter Class

Provides common functionality for all semantic prefilters:
- Comprehensive text cleaning (Unicode, HTML, invisible chars, security)
- Article structure validation
- Standard interface for apply_filter()
- Shared utilities

All filter prefilters should inherit from this base class.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import text cleaning from ground_truth module
sys.path.insert(0, str(Path(__file__).parent.parent / 'ground_truth'))
from text_cleaning import sanitize_text_comprehensive, clean_article as clean_article_comprehensive


class BasePreFilter:
    """
    Base class for semantic prefilters.

    Provides automatic Unicode sanitization and standard interface.
    Subclasses should implement apply_filter() method.
    """

    VERSION = "0.0"  # Override in subclass
    MIN_CONTENT_LENGTH = 300  # Minimum content length to prevent framework leakage
    MAX_PREFILTER_CONTENT = 2000  # Content chars to analyze in prefilter (for efficiency)

    @staticmethod
    def validate_article(article) -> Tuple[bool, str]:
        """
        Validate article structure before processing.

        Checks:
        - Article is a dict
        - Has required fields (title, content/text)
        - Fields are non-empty strings

        Args:
            article: Object to validate

        Returns:
            (is_valid, reason)
            - (True, "valid"): Article structure is valid
            - (False, "reason"): Why validation failed
        """
        # Check type
        if not isinstance(article, dict):
            return (False, f"invalid_type_{type(article).__name__}")

        # Check for title
        if 'title' not in article:
            return (False, "missing_title")

        title = article.get('title')
        if not isinstance(title, str) or not title.strip():
            return (False, "empty_title")

        # Check for content (supports both 'content' and 'text' keys)
        content = article.get('content') or article.get('text')
        if content is None:
            return (False, "missing_content")

        if not isinstance(content, str) or not content.strip():
            return (False, "empty_content")

        return (True, "valid")

    @staticmethod
    def has_any_pattern(text: str, patterns: List[re.Pattern]) -> bool:
        """
        Check if text matches any regex pattern in the list.

        Args:
            text: Text to search
            patterns: List of compiled regex patterns

        Returns:
            True if any pattern matches
        """
        return any(pattern.search(text) for pattern in patterns)

    @staticmethod
    def count_pattern_matches(text: str, patterns: List[re.Pattern]) -> int:
        """
        Count total matches across all patterns.

        Args:
            text: Text to search
            patterns: List of compiled regex patterns

        Returns:
            Total number of matches across all patterns
        """
        return sum(len(pattern.findall(text)) for pattern in patterns)

    @staticmethod
    def has_any_keyword(
        text: str,
        keywords: List[str],
        case_sensitive: bool = False
    ) -> bool:
        """
        Check if text contains any keyword from the list.

        Args:
            text: Text to search
            keywords: List of keywords to find
            case_sensitive: Whether to match case (default: False)

        Returns:
            True if any keyword found
        """
        if not case_sensitive:
            text = text.lower()
            keywords = [k.lower() for k in keywords]
        return any(kw in text for kw in keywords)

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
            (apply_filter, reason)
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

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for scoring.

        Subclasses MUST implement this method.

        Args:
            article: Dict with 'title' and 'text'/'content' keys

        Returns:
            (should_score, reason)
            - (True, "passed"): Send to LLM
            - (False, "reason"): Block with reason string
        """
        raise NotImplementedError("Subclasses must implement apply_filter()")
    
    def _get_combined_text(self, article: Dict) -> str:
        """Combine title + description + content for analysis"""
        parts = []

        if 'title' in article:
            parts.append(article['title'])

        if 'description' in article:
            parts.append(article['description'])

        if 'content' in article:
            # Limit content for pre-filter efficiency
            parts.append(article['content'][:self.MAX_PREFILTER_CONTENT])

        return ' '.join(parts)
    
    def _get_combined_clean_text(self, article: Dict) -> str:
        """Combine and comprehensively clean title + description + content for analysis"""
        combined_text = self._get_combined_text(article)    
        return self.sanitize_text_comprehensive(combined_text.lower())

