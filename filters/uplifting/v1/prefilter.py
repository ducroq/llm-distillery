"""
Uplifting Pre-Filter v1.0

Blocks obvious low-value content before LLM labeling:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)

Purpose: Reduce LLM costs and improve training data quality.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

# Import base prefilter
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_prefilter import BasePreFilter


class UpliftingPreFilterV1(BasePreFilter):
    """Fast rule-based pre-filter for uplifting content"""

    VERSION = "1.0"

    # Domain Exclusions
    ACADEMIC_DOMAINS = [
        'arxiv.org',
        'biorxiv.org',
        'eprint.iacr.org',
        'mdpi.com',
        'medrxiv.org',
        'journals.plos.org',
        'frontiersin.org',
        'link.aps.org',
    ]

    VC_STARTUP_DOMAINS = [
        'sifted.eu',
        'tech.eu',
        'techcrunch.com',
        'crunchbase.com',
        'producthunt.com',
        'sequoiacap.com',
        'blog.ycombinator.com',
    ]

    DEFENSE_DOMAINS = [
        'defensenews.com',
    ]

    CODE_HOSTING_DOMAINS = [
        'github.com',
        'gitlab.com',
    ]

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

    # Unicode sanitization methods inherited from BasePreFilter:
    # - sanitize_unicode(text: str) -> str
    # - clean_article(article: Dict) -> Dict

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Args:
            article: Dict with 'title' and 'text' keys

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, "content_too_short"): Block - content below minimum length
            - (False, "excluded_domain_*"): Block - academic/VC/defense domain
            - (False, "corporate_finance"): Block - corporate finance without exceptions
            - (False, "military_security"): Block - military buildup without peace context
        """
        # Check content length first to prevent framework leakage
        passed, reason = self.check_content_length(article)
        if not passed:
            return False, reason

        # Check domain exclusions
        url = article.get('url', '')
        if url:
            if self._is_excluded_domain(url, self.ACADEMIC_DOMAINS):
                return False, "excluded_domain_academic"
            if self._is_excluded_domain(url, self.VC_STARTUP_DOMAINS):
                return False, "excluded_domain_vc_startup"
            if self._is_excluded_domain(url, self.DEFENSE_DOMAINS):
                return False, "excluded_domain_defense"
            if self._is_excluded_domain(url, self.CODE_HOSTING_DOMAINS):
                return False, "excluded_domain_code_hosting"

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

    def _is_excluded_domain(self, url: str, domains: list) -> bool:
        """Check if URL belongs to an excluded domain"""
        url_lower = url.lower()
        return any(domain in url_lower for domain in domains)

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
            'academic_domains': len(self.ACADEMIC_DOMAINS),
            'vc_startup_domains': len(self.VC_STARTUP_DOMAINS),
            'defense_domains': len(self.DEFENSE_DOMAINS),
            'code_hosting_domains': len(self.CODE_HOSTING_DOMAINS),
            'corporate_finance_patterns': len(self.CORPORATE_FINANCE_PATTERNS),
            'corporate_finance_exceptions': len(self.CORPORATE_FINANCE_EXCEPTIONS),
            'military_security_patterns': len(self.MILITARY_SECURITY_PATTERNS),
            'military_security_exceptions': len(self.MILITARY_SECURITY_EXCEPTIONS),
        }


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = UpliftingPreFilterV1()

    test_cases = [
        # Should BLOCK - Content too short
        {
            'title': 'Short Article',
            'text': 'This article is too short and will be blocked by the content length filter.',
            'expected': (False, 'content_too_short_75chars')
        },

        # Should BLOCK - Academic Domain (ArXiv)
        {
            'title': 'Novel Neural Architecture Achieves State-of-the-Art Performance',
            'url': 'https://arxiv.org/abs/2401.12345',
            'text': 'We present a novel neural architecture that achieves state-of-the-art performance on multiple benchmark datasets. Our approach combines attention mechanisms with hierarchical feature extraction to improve model efficiency and accuracy. Experimental results demonstrate significant improvements over baseline methods across various tasks, with particular gains in few-shot learning scenarios and transfer learning applications.',
            'expected': (False, 'excluded_domain_academic')
        },

        # Should BLOCK - VC/Startup Domain
        {
            'title': 'European Startup Raises €50M Series B',
            'url': 'https://sifted.eu/articles/startup-funding-series-b',
            'text': 'The fintech startup announced a €50M Series B funding round led by major European venture capital firms. The company plans to use the capital to expand into new markets and double its engineering team. This brings total funding to €75M since inception. Investors cited strong growth metrics and market opportunity as key factors in their decision to back the company.',
            'expected': (False, 'excluded_domain_vc_startup')
        },

        # Should BLOCK - Defense Domain
        {
            'title': 'Pentagon Awards $2B Contract for Next-Gen Fighter',
            'url': 'https://www.defensenews.com/air/2024/fighter-contract',
            'text': 'The Department of Defense announced a $2 billion contract award for the development of next-generation fighter aircraft. The contract includes provisions for advanced avionics systems, stealth technology, and enhanced weapons capabilities. Defense officials emphasized the importance of maintaining air superiority and technological edge over potential adversaries in future conflicts.',
            'expected': (False, 'excluded_domain_defense')
        },

        # Should BLOCK - Corporate Finance
        {
            'title': 'Tech Unicorn Raises $500M Series C',
            'text': 'The startup announced a major funding round led by venture capital firms, with the company valued at $2 billion after this Series C investment. The funding will be used to expand operations and hire more engineers. Investors are excited about the market cap potential and the company plans to go public within two years through an IPO process.',
            'expected': (False, 'corporate_finance')
        },

        # Should PASS - Worker Cooperative
        {
            'title': 'Worker Cooperative Expands Solar Panel Manufacturing',
            'text': 'The employee-owned company secured community funding to expand solar panel production capacity by 50%. As a worker cooperative, all employees have voting rights on major decisions and share in the profits. The expansion will create 30 new jobs in the region and help make renewable energy more affordable for local residents through their community ownership model.',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Military Buildup
        {
            'title': 'Finland Increases Defense Spending After NATO Accession',
            'text': 'Following its NATO membership, Finland announced a significant military buildup including the purchase of new fighter jets, increased troops deployment along the eastern border, and enhanced border fortifications. The defense budget will increase by 40% over the next three years to fund these military expansion projects and weapons procurement from allied nations.',
            'expected': (False, 'military_security')
        },

        # Should PASS - Peace Process
        {
            'title': 'Historic Peace Agreement Signed After 20 Years of Conflict',
            'text': 'After two decades of armed conflict, former combatants signed a comprehensive peace agreement that includes provisions for disarmament, the establishment of a truth commission to investigate war crimes, and a detailed reconciliation process. The peace process brings hope to millions of civilians affected by the war and marks a new chapter in the region.',
            'expected': (True, 'passed')
        },

        # Should PASS - General Article
        {
            'title': 'Community Garden Project Feeds 500 Families',
            'text': 'Local residents transformed a two-acre vacant lot into a thriving urban farm that now provides fresh vegetables to over 500 families in the neighborhood. The community-led initiative offers free gardening workshops, creates green space in the urban environment, and builds social connections among diverse community members. Volunteers of all ages participate in maintaining the garden throughout the growing season.',
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
