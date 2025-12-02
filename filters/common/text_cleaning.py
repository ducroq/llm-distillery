"""
Comprehensive text cleaning utilities for semantic filters.

Handles:
- Invalid Unicode (surrogates)
- Zero-width characters (invisible text)
- Bidirectional marks (security issue)
- Whitespace normalization
- HTML content cleaning

Use sanitize_text_comprehensive() for all text before LLM labeling.
"""

import re
import html
from typing import Dict, List, Any, Union


def remove_zero_width_characters(text: str) -> str:
    """
    Remove zero-width characters that can break text matching and hide content.

    Removes:
    - Zero-width space (U+200B)
    - Zero-width non-joiner (U+200C)
    - Zero-width joiner (U+200D)
    - Byte Order Mark (U+FEFF)
    - Word joiner (U+2060)
    - Mongolian vowel separator (U+180E)

    Args:
        text: String that may contain zero-width characters

    Returns:
        Cleaned string with zero-width characters removed
    """
    if not isinstance(text, str):
        return str(text)

    zero_width_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Byte Order Mark (BOM) / Zero-width no-break space
        '\u2060',  # Word joiner
        '\u180e',  # Mongolian vowel separator (deprecated but still used)
    ]

    for char in zero_width_chars:
        text = text.replace(char, '')

    return text


def remove_bidi_marks(text: str) -> str:
    """
    Remove bidirectional text marks that can be used to hide malicious content.

    Removes:
    - Left-to-right mark/embedding/override
    - Right-to-left mark/embedding/override
    - Pop directional formatting

    These can be used to manipulate text display and hide malicious URLs or content.
    Critical for security when processing untrusted web content.

    Args:
        text: String that may contain BiDi marks

    Returns:
        Cleaned string with BiDi marks removed
    """
    if not isinstance(text, str):
        return str(text)

    bidi_chars = [
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
        '\u2066',  # Left-to-right isolate
        '\u2067',  # Right-to-left isolate
        '\u2068',  # First strong isolate
        '\u2069',  # Pop directional isolate
    ]

    for char in bidi_chars:
        text = text.replace(char, '')

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive whitespace while preserving paragraph breaks.

    - Converts tabs to spaces
    - Reduces multiple spaces to single space
    - Reduces excessive newlines (>2) to double newline
    - Strips leading/trailing whitespace

    Args:
        text: String with potentially excessive whitespace

    Returns:
        String with normalized whitespace
    """
    if not isinstance(text, str):
        return str(text)

    text = text.replace('\t', ' ')
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()


def clean_html_entities(text: str) -> str:
    """
    Decode HTML entities to their Unicode equivalents.

    Converts:
    - &nbsp; → space
    - &amp; → &
    - &lt; → <
    - &gt; → >
    - &quot; → "
    - &#39; → '
    - And all other HTML entities

    Args:
        text: String that may contain HTML entities

    Returns:
        String with HTML entities decoded
    """
    if not isinstance(text, str):
        return str(text)

    try:
        text = html.unescape(text)
    except Exception:
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

    return text


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags using regex.

    This is a lightweight alternative to BeautifulSoup for simple HTML removal.
    For complex HTML parsing, use content-aggregator's ContentCleaner instead.

    Args:
        text: String that may contain HTML tags

    Returns:
        String with HTML tags removed
    """
    if not isinstance(text, str):
        return str(text)

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)

    return text


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

    return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


def sanitize_text_comprehensive(text: str) -> str:
    """
    Apply all text cleaning operations for comprehensive sanitization.

    Removes:
    - Invalid Unicode (surrogates)
    - HTML entities
    - HTML tags
    - Zero-width characters
    - Bidirectional marks (security)
    - Normalizes whitespace

    This is the RECOMMENDED function for cleaning text before LLM labeling.
    Use this for all article text in the distillery pipeline.

    Args:
        text: Text to clean

    Returns:
        Comprehensively cleaned text
    """
    if not isinstance(text, str):
        text = str(text)

    text = sanitize_unicode(text)
    text = clean_html_entities(text)
    text = remove_html_tags(text)
    text = remove_zero_width_characters(text)
    text = remove_bidi_marks(text)
    text = normalize_whitespace(text)

    return text


def clean_article(article: Union[Dict, List, str, Any]) -> Union[Dict, List, str, Any]:
    """
    Recursively sanitize all text fields in an article or data structure.

    Applies comprehensive cleaning to all strings in the data structure.
    Safe to call on already-clean articles (idempotent).

    Args:
        article: Article dict/list/str with potentially problematic text

    Returns:
        New structure with all text fields comprehensively cleaned
    """
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(item) for item in obj]
        elif isinstance(obj, str):
            return sanitize_text_comprehensive(obj)
        return obj

    return _clean(article)


def clean_article_for_labeling(article: Dict) -> Dict:
    """
    Clean an article specifically for LLM labeling.

    Focuses on cleaning 'title' and 'text'/'content' fields while
    preserving metadata integrity.

    Args:
        article: Article dict with at least 'title' and 'text' or 'content'

    Returns:
        Article dict with cleaned text fields
    """
    cleaned = article.copy()

    if 'title' in cleaned:
        cleaned['title'] = sanitize_text_comprehensive(cleaned['title'])

    if 'text' in cleaned:
        cleaned['text'] = sanitize_text_comprehensive(cleaned['text'])
    elif 'content' in cleaned:
        cleaned['content'] = sanitize_text_comprehensive(cleaned['content'])

    return cleaned


def batch_clean_articles(articles: List[Dict]) -> List[Dict]:
    """
    Clean a batch of articles for LLM labeling.

    Args:
        articles: List of article dictionaries

    Returns:
        List of cleaned article dictionaries
    """
    return [clean_article_for_labeling(article) for article in articles]
