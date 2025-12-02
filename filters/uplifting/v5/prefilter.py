"""
Uplifting Pre-Filter v5.0

Blocks obvious low-value content before LLM labeling:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)
- Pure speculation articles (no documented outcomes)
- Academic preprints and research papers
- Code repositories and developer tutorials

Purpose: Reduce LLM costs and improve training data quality.

Changes from v4:
- Added speculation detection (heavy "could/might/may" language)
- Added academic domain filtering
- Improved pattern matching for corporate finance
"""

import re
from typing import Dict, List, Tuple

from filters.common.base_prefilter import BasePreFilter


class UpliftingPreFilterV5(BasePreFilter):
    """Fast rule-based pre-filter for uplifting content v5"""

    VERSION = "5.0"

    # === DOMAIN EXCLUSIONS ===

    ACADEMIC_DOMAINS = [
        'arxiv.org',
        'biorxiv.org',
        'medrxiv.org',
        'ssrn.com',
        'eprint.iacr.org',
        'mdpi.com',
        'journals.plos.org',
        'frontiersin.org',
        'link.aps.org',
        'nature.com/articles',  # Research articles
        'sciencedirect.com',
    ]

    VC_STARTUP_DOMAINS = [
        'techcrunch.com',
        'sifted.eu',
        'tech.eu',
        'crunchbase.com',
        'producthunt.com',
        'sequoiacap.com',
        'blog.ycombinator.com',
        'venturebeat.com',
    ]

    DEFENSE_DOMAINS = [
        'defensenews.com',
        'janes.com',
        'defensepriorities.org',
        'breakingdefense.com',
    ]

    CODE_HOSTING_DOMAINS = [
        'github.com',
        'gitlab.com',
        'bitbucket.org',
        'stackoverflow.com',
        'dev.to',
        'medium.com/tag/programming',
    ]

    # === CORPORATE FINANCE INDICATORS ===

    CORPORATE_FINANCE_PATTERNS = [
        # Stock market
        r'\b(stock price|share price|market cap|trading|nasdaq|nyse|s&p 500)\b',
        r'\b(earnings|quarterly results|financial performance|fiscal year)\b',
        r'\b(eps|ebitda|market valuation|price target)\b',

        # Funding
        r'\b(funding round|series [a-e]|seed round|venture capital|vc funding)\b',
        r'\braised \$[\d,]+\s*(million|billion|m|b)\b',
        r'\b(valuation of \$|valued at \$|unicorn status)\b',

        # Corporate events
        r'\b(ipo|initial public offering|going public|public listing)\b',
        r'\b(m&a|merger|acquisition|buyout|takeover|acqui-?hire)\b',
        r'\b(investor relations|shareholder value|dividend|stock buyback)\b',
    ]

    CORPORATE_FINANCE_EXCEPTIONS = [
        r'\b(worker cooperative|worker-owned|employee-owned|coop)\b',
        r'\b(public benefit|b corp|benefit corporation|social enterprise)\b',
        r'\b(open source|open access|freely available|creative commons)\b',
        r'\b(affordable access|community ownership|commons|mutual aid)\b',
        r'\b(non-?profit|nonprofit|ngo|charity|foundation)\b',
    ]

    # === MILITARY/SECURITY INDICATORS ===

    MILITARY_SECURITY_PATTERNS = [
        # Military operations
        r'\b(military buildup|defense spending|armed forces|troop deployment)\b',
        r'\b(weapons system|arms deal|ammunition|missiles|fighter jets|tanks)\b',
        r'\b(nato expansion|military alliance|defense pact|security agreement)\b',

        # Security measures
        r'\b(border defense|border security|border wall|border patrol)\b',
        r'\b(surveillance|military exercise|war games|defense budget)\b',
        r'\b(troops|soldiers|battalion|regiment|special forces)\b',
        r'\b(military spending|arms procurement|defense contract)\b',
    ]

    MILITARY_SECURITY_EXCEPTIONS = [
        r'\b(peace|peace process|peace agreement|peace talks|peace treaty)\b',
        r'\b(demilitarization|disarmament|arms reduction|denuclearization)\b',
        r'\b(conflict resolution|reconciliation|ceasefire|armistice)\b',
        r'\b(peacekeeping|peace keeping|un peace|humanitarian)\b',
        r'\b(truth commission|war crimes tribunal|justice|accountability)\b',
        r'\b(veterans? (support|services|care|mental health))\b',
    ]

    # === SPECULATION INDICATORS ===

    SPECULATION_PATTERNS = [
        # Future tense speculation
        r'\bcould (potentially |significantly |dramatically )?(transform|revolutionize|disrupt|change)\b',
        r'\bmight (help|enable|allow|lead to|result in)\b',
        r'\bmay (become|lead|result|transform|enable)\b',
        r'\b(promises to|aims to|hopes to|seeks to|plans to)\b',
        r'\b(potential to|poised to|set to|expected to)\b',

        # Hype language without evidence
        r'\b(game-?changer|breakthrough|revolutionary|disruptive)\b',
        r'\b(next big thing|future of|will transform)\b',
    ]

    # Evidence of actual outcomes (negates speculation flag)
    OUTCOME_EVIDENCE_PATTERNS = [
        r'\b(resulted in|achieved|delivered|produced|created)\b',
        r'\b(increased by|reduced by|improved by|saved)\b',
        r'\b\d+%\s*(increase|decrease|improvement|reduction)\b',
        r'\b(documented|verified|confirmed|measured|studied)\b',
        r'\b(according to|study found|research shows|data shows)\b',
        r'\b(now serves?|currently provides?|has helped)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        # Corporate finance
        self.corporate_finance_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CORPORATE_FINANCE_PATTERNS
        ]
        self.corporate_finance_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CORPORATE_FINANCE_EXCEPTIONS
        ]

        # Military security
        self.military_security_regex = [
            re.compile(p, re.IGNORECASE) for p in self.MILITARY_SECURITY_PATTERNS
        ]
        self.military_security_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.MILITARY_SECURITY_EXCEPTIONS
        ]

        # Speculation
        self.speculation_regex = [
            re.compile(p, re.IGNORECASE) for p in self.SPECULATION_PATTERNS
        ]
        self.outcome_evidence_regex = [
            re.compile(p, re.IGNORECASE) for p in self.OUTCOME_EVIDENCE_PATTERNS
        ]

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should pass to oracle for scoring.

        Args:
            article: Dict with 'title' and 'text' (or 'content') keys

        Returns:
            (passed, reason)
            - (True, "passed"): Send to oracle
            - (False, reason): Block with specific reason
        """
        # Check content length first
        passed, reason = self.check_content_length(article)
        if not passed:
            return False, reason

        # Check domain exclusions
        url = article.get('url', '')
        if url:
            domain_result = self._check_domain_exclusions(url)
            if domain_result:
                return False, domain_result

        # Get text content
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Check corporate finance
        if self._has_pattern(combined_text, self.corporate_finance_regex):
            if not self._has_pattern(combined_text, self.corporate_finance_exceptions_regex):
                return False, "corporate_finance"

        # Check military/security
        if self._has_pattern(combined_text, self.military_security_regex):
            if not self._has_pattern(combined_text, self.military_security_exceptions_regex):
                return False, "military_security"

        # Check heavy speculation (only block if NO outcome evidence)
        speculation_count = self._count_matches(combined_text, self.speculation_regex)
        outcome_count = self._count_matches(combined_text, self.outcome_evidence_regex)

        # Block if heavy speculation (3+ matches) with no outcomes
        if speculation_count >= 3 and outcome_count == 0:
            return False, "pure_speculation"

        # Passed all filters
        return True, "passed"

    def _check_domain_exclusions(self, url: str) -> str:
        """Check if URL belongs to an excluded domain. Returns reason or empty string."""
        url_lower = url.lower()

        for domain in self.ACADEMIC_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_academic"

        for domain in self.VC_STARTUP_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_vc_startup"

        for domain in self.DEFENSE_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_defense"

        for domain in self.CODE_HOSTING_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_code"

        return ""

    def _has_pattern(self, text: str, patterns: List[re.Pattern]) -> bool:
        """Check if text matches any pattern in the list."""
        return any(pattern.search(text) for pattern in patterns)

    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count total matches across all patterns."""
        count = 0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count

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
            'speculation_patterns': len(self.SPECULATION_PATTERNS),
            'outcome_evidence_patterns': len(self.OUTCOME_EVIDENCE_PATTERNS),
        }

    def classify_content_type(self, article: Dict) -> str:
        """
        Classify article content type (for oracle pre-classification).

        Returns one of:
        - "corporate_finance"
        - "military_security"
        - "speculation"
        - "peace_process"
        - "general"
        """
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Check peace process first (exception to military)
        if self._has_pattern(combined_text, self.military_security_exceptions_regex):
            if self._has_pattern(combined_text, self.military_security_regex):
                return "peace_process"

        # Check corporate finance
        if self._has_pattern(combined_text, self.corporate_finance_regex):
            if not self._has_pattern(combined_text, self.corporate_finance_exceptions_regex):
                return "corporate_finance"

        # Check military/security
        if self._has_pattern(combined_text, self.military_security_regex):
            return "military_security"

        # Check speculation
        speculation_count = self._count_matches(combined_text, self.speculation_regex)
        outcome_count = self._count_matches(combined_text, self.outcome_evidence_regex)
        if speculation_count >= 2 and outcome_count <= 1:
            return "speculation"

        return "general"


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = UpliftingPreFilterV5()

    # Note: MIN_CONTENT_LENGTH is 300 chars, so test content must be longer

    test_cases = [
        # Should BLOCK - Content too short
        {
            'title': 'Short Article',
            'text': 'This is too short to pass the minimum content length filter.',
            'expected': (False, 'content_too_short'),
            'description': 'Content too short'
        },

        # Should BLOCK - Academic Domain
        {
            'title': 'Novel Neural Architecture for Climate Modeling',
            'url': 'https://arxiv.org/abs/2401.12345',
            'text': 'We present a novel approach to climate modeling using transformer architectures. Our method achieves state-of-the-art performance on multiple benchmark datasets. The model uses attention mechanisms to capture long-range dependencies in climate data. We evaluate on three major climate prediction benchmarks and demonstrate significant improvements over baseline methods. Our code and trained models are available for reproducibility. This work contributes to the growing body of research applying deep learning to environmental science challenges.',
            'expected': (False, 'excluded_domain_academic'),
            'description': 'Academic preprint (arxiv)'
        },

        # Should BLOCK - VC/Startup Domain
        {
            'title': 'Startup Raises $50M to Transform Healthcare',
            'url': 'https://techcrunch.com/2024/startup-funding',
            'text': 'The AI startup announced a major Series B funding round led by top venture capital firms including Sequoia and Andreessen Horowitz. The company, valued at $500 million, plans to revolutionize healthcare diagnostics with its proprietary platform. CEO Jane Smith said the funds will accelerate product development and market expansion. The startup has grown 300% year-over-year and now employs 150 people across three offices.',
            'expected': (False, 'excluded_domain_vc_startup'),
            'description': 'VC news (TechCrunch)'
        },

        # Should BLOCK - Corporate Finance
        {
            'title': 'Tech Giant Reports Record Q4 Earnings',
            'text': 'The company announced quarterly results exceeding analyst expectations, with EPS of $2.50 and revenue growth of 15%. Stock price surged 8% in after-hours trading on NASDAQ. The CEO highlighted strong market cap growth and shareholder value creation during the earnings call. Analysts raised price targets following the announcement. The board also approved a $10 billion stock buyback program to return value to investors. Dividend payouts will increase by 10% starting next quarter.',
            'expected': (False, 'corporate_finance'),
            'description': 'Corporate finance news'
        },

        # Should PASS - Worker Cooperative (exception)
        {
            'title': 'Worker Cooperative Expands Solar Manufacturing',
            'text': 'The employee-owned cooperative secured community funding to expand solar panel production capacity by 50%. As a worker cooperative, all 200 employees share equally in profits and participate in democratic decision-making. The expansion will create 50 new worker-owner positions in the region. Production costs have decreased 20% through efficiency improvements led by worker teams. The cooperative model has proven more resilient than traditional corporate structures during economic downturns.',
            'expected': (True, 'passed'),
            'description': 'Worker cooperative (exception)'
        },

        # Should BLOCK - Military Buildup
        {
            'title': 'NATO Increases Defense Spending',
            'text': 'Following recent geopolitical tensions, NATO announced a major military buildup including procurement of new weapons systems, expanded troop deployments to Eastern Europe, and a 25% increase in defense spending across member nations. The alliance will acquire 500 new fighter jets and advanced missile defense systems. Defense ministers agreed to strengthen border security and increase military exercises. The defense budget will reach $1.2 trillion annually by 2026.',
            'expected': (False, 'military_security'),
            'description': 'Military buildup'
        },

        # Should PASS - Peace Process (exception)
        {
            'title': 'Historic Peace Agreement Signed After Decades of Conflict',
            'text': 'After 25 years of armed conflict that claimed over 50,000 lives, former combatants signed a comprehensive peace agreement in a historic ceremony. The peace process includes provisions for disarmament of all armed groups, establishment of a truth commission to document wartime atrocities, and reconciliation programs for affected communities. Both sides committed to demilitarization of border regions and the peaceful resolution of remaining disputes. International observers praised the agreement as a model for conflict resolution.',
            'expected': (True, 'passed'),
            'description': 'Peace process (exception)'
        },

        # Should BLOCK - Pure Speculation
        {
            'title': 'New Technology Could Transform Energy Production',
            'text': 'Scientists say the experimental breakthrough could potentially revolutionize global energy production within the next decade. The technology might help address climate change and may become the future of clean energy. Experts believe it promises to transform the entire industry and is poised to disrupt existing fossil fuel markets. The innovation could democratize access to power and might enable communities to achieve energy independence. Researchers aim to begin pilot testing next year.',
            'expected': (False, 'pure_speculation'),
            'description': 'Pure speculation (no outcomes)'
        },

        # Should PASS - Documented Outcomes
        {
            'title': 'Community Garden Project Feeds 500 Families Weekly',
            'text': 'The urban farm initiative now serves over 500 families weekly with fresh produce, according to verified program data released this month. Yields increased by 40% compared to last year through improved composting techniques. The initiative has helped reduce food insecurity in the neighborhood by 25%, with documented improvements in nutrition outcomes measured by local health clinics. Volunteers from 12 community organizations participate in the garden, which has become a model for urban agriculture.',
            'expected': (True, 'passed'),
            'description': 'Documented community impact'
        },

        # Should PASS - Mixed speculation with outcomes
        {
            'title': 'Renewable Energy Expansion Shows Measurable Results',
            'text': 'The community solar program could expand to additional neighborhoods next year pending city approval. However, it has already delivered measurable results in its first two years of operation: household energy costs reduced by an average of 30%, with verified savings documented by independent auditors from the state energy office. The project currently provides clean energy to 10,000 homes and has created 75 local installation jobs. Carbon emissions in the service area decreased by 15,000 tons annually.',
            'expected': (True, 'passed'),
            'description': 'Speculation + documented outcomes'
        },
    ]

    print("Testing Uplifting Pre-Filter v5.0")
    print("=" * 70)

    passed_tests = 0
    failed_tests = 0

    for i, test in enumerate(test_cases, 1):
        result = prefilter.apply_filter(test)
        expected = test['expected']

        # Handle partial matching for content_too_short
        if expected[1].startswith('content_too_short') and result[1].startswith('content_too_short'):
            match = True
        else:
            match = (result[0] == expected[0] and result[1] == expected[1])

        status = "[PASS]" if match else "[FAIL]"
        if match:
            passed_tests += 1
        else:
            failed_tests += 1

        print(f"\nTest {i}: {status} - {test['description']}")
        print(f"  Title: {test['title'][:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

    print("\n" + "=" * 70)
    print(f"Results: {passed_tests}/{passed_tests + failed_tests} tests passed")
    print("\nPre-filter Statistics:")
    for key, value in prefilter.get_statistics().items():
        print(f"  {key}: {value}")

    # Test content type classification
    print("\n" + "=" * 70)
    print("Content Type Classification Test:")
    for test in test_cases[3:]:  # Skip short content tests
        content_type = prefilter.classify_content_type(test)
        print(f"  {test['description']}: {content_type}")


if __name__ == '__main__':
    test_prefilter()
