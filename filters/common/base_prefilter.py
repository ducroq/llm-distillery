"""
Base PreFilter Class

Provides common functionality for all semantic prefilters:
- Comprehensive text cleaning (Unicode, HTML, invisible chars, security)
- Article structure validation
- Standard interface for apply_filter()
- Shared utilities for pattern matching
- Optional ML-based commerce content detection

All filter prefilters should inherit from this base class.
"""

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from filters.common.text_cleaning import sanitize_text_comprehensive, clean_article as clean_article_comprehensive

if TYPE_CHECKING:
    from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

logger = logging.getLogger(__name__)


class BasePreFilter:
    """
    Base class for semantic prefilters.

    Provides automatic Unicode sanitization and standard interface.
    Subclasses should implement apply_filter() method.

    Optional commerce prefilter:
        Set use_commerce_prefilter=True to enable ML-based commerce detection.
        The commerce detector is loaded lazily and shared across all instances (singleton).
    """

    VERSION = "0.0"  # Override in subclass
    MIN_CONTENT_LENGTH = 300  # Minimum content length to prevent framework leakage
    MAX_PREFILTER_CONTENT = 2000  # Content chars to analyze in prefilter (for efficiency)

    # Singleton commerce detector (shared across all instances)
    _commerce_detector: Optional["CommercePrefilterSLM"] = None
    _commerce_detector_lock = threading.Lock()

    def __init__(
        self,
        use_commerce_prefilter: bool = False,
        commerce_threshold: float = 0.7,
    ):
        """
        Initialize the prefilter.

        Args:
            use_commerce_prefilter: Enable ML-based commerce content detection.
                When enabled, articles identified as commerce/promotional content
                will be blocked before domain-specific filtering.
            commerce_threshold: Threshold for commerce classification (0-1).
                Higher values are more strict (fewer false positives).
        """
        self.use_commerce_prefilter = use_commerce_prefilter
        self.commerce_threshold = commerce_threshold

        # Lazy load commerce detector if enabled
        if use_commerce_prefilter:
            self._ensure_commerce_detector_loaded()

    @classmethod
    def _ensure_commerce_detector_loaded(cls):
        """
        Lazy load the commerce detector singleton.

        Thread-safe loading using double-checked locking pattern.
        """
        if cls._commerce_detector is not None:
            return

        with cls._commerce_detector_lock:
            # Double-check after acquiring lock
            if cls._commerce_detector is not None:
                return

            try:
                from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM
                logger.info("Loading commerce prefilter SLM...")
                cls._commerce_detector = CommercePrefilterSLM()
                logger.info("Commerce prefilter loaded successfully")
            except FileNotFoundError:
                logger.warning(
                    "Commerce prefilter model not found. "
                    "Train it first with: python -m filters.common.commerce_prefilter.training.train"
                )
                cls._commerce_detector = None
            except Exception as e:
                logger.error(f"Failed to load commerce prefilter: {e}")
                cls._commerce_detector = None

    def check_commerce(self, article: Dict) -> Tuple[bool, str, Optional[float]]:
        """
        Check if article is commerce/promotional content.

        Args:
            article: Dict with 'title' and 'content'/'text' keys

        Returns:
            (is_commerce, reason, score)
            - (True, "commerce_content_0.87", 0.87): Detected as commerce
            - (False, "", None): Not commerce or detector not available
        """
        if not self.use_commerce_prefilter or self._commerce_detector is None:
            return (False, "", None)

        result = self._commerce_detector.is_commerce(article)
        score = result["score"]

        if result["is_commerce"]:
            return (True, f"commerce_content_{score:.2f}", score)

        return (False, "", score)

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
        if not isinstance(article, dict):
            return (False, f"invalid_type_{type(article).__name__}")

        if 'title' not in article:
            return (False, "missing_title")

        title = article.get('title')
        if not isinstance(title, str) or not title.strip():
            return (False, "empty_title")

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
