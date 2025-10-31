"""
Uplifting Pre-Filter v1.0

Blocks obvious low-value content before LLM labeling:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)

Purpose: Reduce LLM costs and improve training data quality.
"""

import re
from typing import Dict, Tuple


class UpliftingPreFilterV1:
    """Fast rule-based pre-filter for uplifting content"""

    VERSION = "1.0"

    # A) Corporate Finance Indicators
    CORPORATE_FINANCE_PATTERNS = [
        # Stock market
        r'\b(stock price|share price|market cap|trading|nasdaq|nyse|s&p)\b',
        r'\b(earnings|revenue|profit|quarterly results|financial performance)\b',
        r'\b(eps|ebitda|market valuation)\b',

        # Funding
        r'\b(funding round|series [a-e]|seed round|venture capital|vc)\b',
        r'\b(raised \$\d+|million in funding|billion in funding)\b',
        r'\b(valuation of \$|valued at \$)\b',

        # Corporate events
        r'\b(ipo|initial public offering|going public)\b',
        r'\b(m&a|merger|acquisition|buyout|takeover)\b',
        r'\b(investor|shareholder|dividend)\b',
    ]

    CORPORATE_FINANCE_EXCEPTIONS = [
        r'\b(worker cooperative|worker-owned|employee-owned)\b',
        r'\b(public benefit|b corp|benefit corporation)\b',
        r'\b(open source|open access|freely available)\b',
        r'\b(affordable access|community ownership|commons)\b',
        r'\b(non-profit|nonprofit|ngo|charity)\b',
    ]

    # C) Military/Security Indicators
    MILITARY_SECURITY_PATTERNS = [
        # Military operations
        r'\b(military buildup|defense spending|armed forces|deployment)\b',
        r'\b(weapons|arms|ammunition|missiles|fighter jets)\b',
        r'\b(nato|military alliance|defense pact)\b',

        # Security measures
        r'\b(border defense|border security|border wall)\b',
        r'\b(security measures|surveillance|military exercise)\b',
        r'\b(defense budget|military spending|arms deal)\b',
        r'\b(troops|soldiers|battalion|regiment)\b',
    ]

    MILITARY_SECURITY_EXCEPTIONS = [
        r'\b(peace|peace process|peace agreement|peace talks)\b',
        r'\b(demilitarization|disarmament|arms reduction)\b',
        r'\b(conflict resolution|reconciliation|ceasefire)\b',
        r'\b(peace keeping|peacekeeping|un peace)\b',
        r'\b(truth commission|war crimes tribunal)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        self.corporate_finance_regex = [re.compile(p, re.IGNORECASE) for p in self.CORPORATE_FINANCE_PATTERNS]
        self.corporate_finance_exceptions_regex = [re.compile(p, re.IGNORECASE) for p in self.CORPORATE_FINANCE_EXCEPTIONS]

        self.military_security_regex = [re.compile(p, re.IGNORECASE) for p in self.MILITARY_SECURITY_PATTERNS]
        self.military_security_exceptions_regex = [re.compile(p, re.IGNORECASE) for p in self.MILITARY_SECURITY_EXCEPTIONS]

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
                return UpliftingPreFilterV1.sanitize_unicode(obj)
            return obj

        return _clean(article)

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Args:
            article: Dict with 'title' and 'text' keys

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, "corporate_finance"): Block - corporate finance without exceptions
            - (False, "military_security"): Block - military buildup without peace context
        """
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Check A) Corporate Finance
        if self._has_corporate_finance(combined_text):
            if not self._has_exception(combined_text, self.corporate_finance_exceptions_regex):
                return False, "corporate_finance"

        # Check C) Military/Security
        if self._has_military_security(combined_text):
            if not self._has_exception(combined_text, self.military_security_exceptions_regex):
                return False, "military_security"

        # Passed all filters
        return True, "passed"

    def _has_corporate_finance(self, text: str) -> bool:
        """Check if text contains corporate finance indicators"""
        return any(pattern.search(text) for pattern in self.corporate_finance_regex)

    def _has_military_security(self, text: str) -> bool:
        """Check if text contains military/security indicators"""
        return any(pattern.search(text) for pattern in self.military_security_regex)

    def _has_exception(self, text: str, exception_patterns: list) -> bool:
        """Check if text contains exception keywords"""
        return any(pattern.search(text) for pattern in exception_patterns)

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        return {
            'version': self.VERSION,
            'corporate_finance_patterns': len(self.CORPORATE_FINANCE_PATTERNS),
            'corporate_finance_exceptions': len(self.CORPORATE_FINANCE_EXCEPTIONS),
            'military_security_patterns': len(self.MILITARY_SECURITY_PATTERNS),
            'military_security_exceptions': len(self.MILITARY_SECURITY_EXCEPTIONS),
        }


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = UpliftingPreFilterV1()

    test_cases = [
        # Should BLOCK - Corporate Finance
        {
            'title': 'Tech Unicorn Raises $500M Series C',
            'text': 'The startup announced a funding round led by major venture capital firms...',
            'expected': (False, 'corporate_finance')
        },

        # Should PASS - Worker Cooperative
        {
            'title': 'Worker Cooperative Expands Solar Panel Manufacturing',
            'text': 'The employee-owned company secured funding to expand production...',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Military Buildup
        {
            'title': 'Finland Increases Defense Spending After NATO Accession',
            'text': 'Military buildup includes new fighter jets and border fortifications...',
            'expected': (False, 'military_security')
        },

        # Should PASS - Peace Process
        {
            'title': 'Historic Peace Agreement Signed After 20 Years of Conflict',
            'text': 'Former combatants agreed to disarmament and truth commission...',
            'expected': (True, 'passed')
        },

        # Should PASS - General Article
        {
            'title': 'Community Garden Project Feeds 500 Families',
            'text': 'Residents transformed vacant lot into thriving urban farm...',
            'expected': (True, 'passed')
        },
    ]

    print("Testing Uplifting Pre-Filter v1.0")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        result = prefilter.should_label(test)
        expected = test['expected']
        status = "[PASS]" if result == expected else "[FAIL]"

        print(f"\nTest {i}: {status}")
        print(f"Title: {test['title']}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 60)
    print("Pre-filter Statistics:")
    for key, value in prefilter.get_statistics().items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_prefilter()
