"""
Investment Risk v5 Prefilter

Blocks content that is almost never investment-relevant to save oracle costs.
Target: Block 40-60% of input, <5% false negative rate on actual investment content.

Philosophy: When in doubt, let it through. Better to score some noise than miss a signal.

Combines:
- Source-based filtering (blocked/allowed sources, investment keywords)
- Content-pattern filtering (FOMO, stock picking, affiliate, clickbait)
"""

import re
from typing import Dict, List, Tuple

from filters.common.base_prefilter import BasePreFilter


class InvestmentRiskPreFilterV5(BasePreFilter):
    """Fast rule-based pre-filter for investment risk content v5"""

    VERSION = "5.0"

    # === SOURCE-BASED FILTERING ===

    # Sources to BLOCK (almost never investment-relevant)
    BLOCKED_SOURCE_PATTERNS = [
        # Developer/tech content
        r'github',
        r'dev_to',
        r'hackaday',
        r'hackernews',
        r'nvidia_developer',

        # Science news (non-financial)
        r'phys_org',
        r'nasa',
        r'smithsonian',
        r'science_news',

        # Positive/lifestyle news (uplifting filter's domain)
        r'upworthy',
        r'optimist_daily',
        r'good_news',
        r'greenme',

        # Entertainment/gaming
        r'tom_hardware',
        r'numerama',
        r'canaltech',
        r'olhar_digital',
        r'tecnoblog',
        r'punto_informatico',
    ]

    # Sources to ALWAYS ALLOW (high investment signal probability)
    ALLOWED_SOURCE_PATTERNS = [
        # Financial press
        r'reuters',
        r'bloomberg',
        r'wsj',
        r'financial_times',
        r'ft_',
        r'economist',

        # Business news
        r'business_insider',
        r'forbes',
        r'cnbc',
        r'marketwatch',
        r'barrons',

        # European financial
        r'handelsblatt',
        r'expansion',
        r'fd_',
        r'les_echos',
        r'capital',

        # Investment-specific
        r'econbrowser',
        r'investor_',
        r'seeking_alpha',
        r'motley_fool',
        r'zerohedge',

        # Central banks / official
        r'fed_',
        r'ecb_',
        r'bis_',
        r'imf_',
    ]

    # Title keywords that suggest investment relevance (override blocks)
    INVESTMENT_KEYWORDS = [
        r'\bfed\b',
        r'\becb\b',
        r'interest rate',
        r'inflation',
        r'recession',
        r'gdp\b',
        r'unemployment',
        r'stock market',
        r'bond market',
        r'yield curve',
        r'credit',
        r'default',
        r'bankruptcy',
        r'bank\s+(failure|crisis|run)',
        r'systemic',
        r'contagion',
        r'liquidity',
        r'central bank',
        r'monetary policy',
        r'fiscal',
        r'treasury',
        r'sovereign debt',
        r'currency',
        r'forex',
        r'commodities',
        r'oil price',
        r'gold price',
        r'tariff',
        r'trade war',
        r'sanctions',
    ]

    # === CONTENT-PATTERN FILTERING ===

    # FOMO and speculation patterns (block)
    FOMO_SPECULATION_PATTERNS = [
        r'\b(hot stock|stocks to buy now|buy now|don\'t miss out|act fast)\b',
        r'\b(meme stock|moonshot|to the moon|rocket|lambo)\b',
        r'\b(next big thing|hidden gem|secret stock|insider tip)\b',
        r'\b(crypto pump|shitcoin|altcoin season|crypto bull run)\b',
        r'\b(get rich quick|easy money|guaranteed returns)\b',
        r'\b(100x|10 bagger|massive gains|explosive growth)\b',
        r'\b(penny stock|otc stock|pink sheet)\b',
        r'\b(yolo|diamond hands|paper hands|hodl|wen moon)\b',
    ]

    # Stock picking patterns (block unless macro context)
    STOCK_PICKING_PATTERNS = [
        r'\b(top \d+ stocks|best stocks|stocks we like)\b',
        r'\b(buy recommendation|strong buy|price target \$)\b',
        r'\b(earnings beat|earnings miss|eps of \$)\b',
        r'\b(technical analysis|chart pattern|support level|resistance)\b',
        r'\b(bullish on|bearish on|long position|short position)\b',
        r'\b(analyst rating|upgrade to buy|downgrade to sell)\b',
    ]

    # Exceptions for stock picking (pass if has macro context)
    MACRO_CONTEXT_PATTERNS = [
        r'\b(systemic risk|macro|recession|yield curve|credit crisis)\b',
        r'\b(federal reserve|fed|ecb|central bank|monetary policy)\b',
        r'\b(banking sector|financial system|contagion|leverage)\b',
        r'\b(credit spread|credit market|bond market)\b',
        r'\b(unemployment|gdp|inflation|pmi|leading indicators)\b',
        r'(portfolio|asset allocation|diversification|risk management)',
        # Dutch/German/French common financial terms
        r'(systemisch risico|recessie|kredietcrisis|rentecurve)',
        r'(werkloosheid|inflatie|economische groei|centrale bank)',
        r'(systemisches risiko|rezession|kreditkrise|zinsstruktur)',
        r'(arbeitslosigkeit|inflation|wirtschaftswachstum|zentralbank)',
        r'(risque systémique|récession|crise du crédit|courbe des taux)',
        r'(chômage|inflation|croissance économique|banque centrale)',
    ]

    # Affiliate and conflict patterns (block)
    AFFILIATE_CONFLICT_PATTERNS = [
        r'\b(sign up|join.*link|use code|promo code|discount code)\b',
        r'\b(through my link|affiliate|sponsored|paid promotion)\b',
        r'\b(my broker|my platform|get bonus|free trial)\b',
        r'\b(discord.*join|telegram.*channel|exclusive.*group)\b',
    ]

    # Clickbait patterns (block)
    CLICKBAIT_PATTERNS = [
        r'\b(crash coming|market crash|stock market collapse)\b.*\!',
        r'\b(this one stock|the one stock|secret stock)\b',
        r'warren buffett.*secret',
        r'\b(what.*don\'t want you to know|they.*hiding)\b',
        r'🚀|💎|🌙|💰|🤑',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        super().__init__()

        # Content-pattern filters (compiled for performance)
        self.fomo_regex = re.compile(
            '|'.join(self.FOMO_SPECULATION_PATTERNS), re.IGNORECASE
        )
        self.stock_picking_regex = re.compile(
            '|'.join(self.STOCK_PICKING_PATTERNS), re.IGNORECASE
        )
        self.macro_context_regex = re.compile(
            '|'.join(self.MACRO_CONTEXT_PATTERNS), re.IGNORECASE
        )
        self.affiliate_regex = re.compile(
            '|'.join(self.AFFILIATE_CONFLICT_PATTERNS), re.IGNORECASE
        )
        self.clickbait_regex = re.compile(
            '|'.join(self.CLICKBAIT_PATTERNS), re.IGNORECASE
        )

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be scored by oracle.

        Flow:
        1. Content length check
        2. Allowed source? -> pass
        3. Investment keyword in title/content? -> pass
        4. Blocked source? -> block
        5. FOMO/speculation content? -> block
        6. Stock picking (unless macro context)? -> block
        7. Affiliate/conflict content? -> block
        8. Clickbait in title? -> block
        9. Default -> allow

        Args:
            article: Dict with 'id', 'title', 'content'/'text', 'source', 'source_type'

        Returns:
            Tuple of (should_score: bool, reason: str)
        """
        # 1. Check minimum content length
        length_ok, length_reason = self.check_content_length(article)
        if not length_ok:
            return False, length_reason

        source = article.get('source', '').lower()
        source_type = article.get('source_type', '').lower()
        article_id = article.get('id', '').lower()
        title = article.get('title', '').lower()
        content = (article.get('content', '') or article.get('text', ''))[:1000].lower()

        source_combined = f"{source} {source_type} {article_id}"
        text_combined = f"{title} {content}"

        # 2. Check if explicitly allowed source
        for pattern in self.ALLOWED_SOURCE_PATTERNS:
            if re.search(pattern, source_combined, re.IGNORECASE):
                return True, f"allowed_source:{pattern}"

        # 3. Check for investment keywords in title/content (override blocks)
        for keyword in self.INVESTMENT_KEYWORDS:
            if re.search(keyword, text_combined, re.IGNORECASE):
                return True, f"investment_keyword:{keyword}"

        # 4. Check if blocked source
        for pattern in self.BLOCKED_SOURCE_PATTERNS:
            if re.search(pattern, source_combined, re.IGNORECASE):
                return False, f"blocked_source:{pattern}"

        # 5. Check for FOMO/speculation content
        if self.fomo_regex.search(text_combined):
            return False, "fomo_speculation"

        # 6. Check for stock picking (but allow if macro context present)
        if self.stock_picking_regex.search(text_combined):
            if not self.macro_context_regex.search(text_combined):
                return False, "stock_picking"

        # 7. Check for affiliate/conflict content
        if self.affiliate_regex.search(text_combined):
            return False, "affiliate_conflict"

        # 8. Check for clickbait in title
        if self.clickbait_regex.search(title):
            return False, "clickbait"

        # 9. Default: allow (when in doubt, score it)
        return True, "default_allow"

    def get_stats(self) -> Dict:
        """Return prefilter configuration stats."""
        return {
            "blocked_source_patterns": len(self.BLOCKED_SOURCE_PATTERNS),
            "allowed_source_patterns": len(self.ALLOWED_SOURCE_PATTERNS),
            "investment_keywords": len(self.INVESTMENT_KEYWORDS),
            "fomo_speculation_patterns": len(self.FOMO_SPECULATION_PATTERNS),
            "stock_picking_patterns": len(self.STOCK_PICKING_PATTERNS),
            "macro_context_patterns": len(self.MACRO_CONTEXT_PATTERNS),
            "affiliate_conflict_patterns": len(self.AFFILIATE_CONFLICT_PATTERNS),
            "clickbait_patterns": len(self.CLICKBAIT_PATTERNS),
            "philosophy": "When in doubt, let it through"
        }


# Legacy function interface for backward compatibility
def prefilter(article: Dict) -> Tuple[bool, str]:
    """Legacy function interface - use InvestmentRiskPreFilterV5 class instead."""
    _prefilter = InvestmentRiskPreFilterV5()
    return _prefilter.apply_filter(article)


def get_stats() -> Dict:
    """Legacy function interface - use InvestmentRiskPreFilterV5 class instead."""
    _prefilter = InvestmentRiskPreFilterV5()
    return _prefilter.get_stats()


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter_instance = InvestmentRiskPreFilterV5()

    # Padding to exceed MIN_CONTENT_LENGTH (300 chars)
    pad = " This is additional content to ensure the article exceeds the minimum content length threshold for the prefilter. " * 3

    test_cases = [
        # === SOURCE-BASED TESTS ===

        # Should BLOCK - blocked source (dev/tech)
        {
            "id": "github_789", "title": "New ML framework", "source": "github",
            "content": "A fast library for machine learning." + pad,
            "expected": (False, "blocked_source:github"),
            "description": "Blocked source (github)",
        },

        # Should PASS - allowed source (reuters)
        {
            "id": "reuters_456", "title": "Fed raises rates", "source": "reuters",
            "content": "The Federal Reserve announced today." + pad,
            "expected": (True, "allowed_source:reuters"),
            "description": "Allowed source (reuters)",
        },

        # Should PASS - investment keyword overrides unknown source
        {
            "id": "science_xyz", "title": "Climate change could trigger recession",
            "source": "nature",
            "content": "New study shows economic impact of climate change." + pad,
            "expected": (True, "investment_keyword:recession"),
            "description": "Investment keyword override (recession)",
        },

        # Should PASS - general tech article, default allow
        {
            "id": "tech_news_abc", "title": "Apple launches iPhone",
            "source": "techcrunch_main",
            "content": "Apple today unveiled a new device with many features." + pad,
            "expected": (True, "default_allow"),
            "description": "Default allow (unknown source, no triggers)",
        },

        # === CONTENT-PATTERN TESTS ===

        # Should PASS - macro risk analysis
        {
            "id": "macro_001",
            "title": "Yield Curve Inversion Signals Recession Risk",
            "source": "analysis_site",
            "content": "The Treasury yield curve has inverted, historically a reliable recession indicator. With unemployment rising and credit spreads widening, economists warn of systemic risks." + pad,
            "expected": (True, "investment_keyword:recession"),
            "description": "Macro risk analysis (passes via keyword)",
        },

        # Should BLOCK - FOMO/speculation
        {
            "id": "fomo_001",
            "title": "Diamond hands! This meme stock is headed for a 100x!",
            "source": "reddit_sub",
            "content": "YOLO into this penny stock! To the moon! Lambo time!" + pad,
            "expected": (False, "fomo_speculation"),
            "description": "FOMO/speculation content",
        },

        # Should BLOCK - stock picking without macro context
        {
            "id": "pick_001",
            "title": "Analyst Strong Buy Ratings for Tech Sector",
            "source": "stock_tips",
            "content": "Our analysts have upgraded these stocks to strong buy. Price target $150. Earnings beat expected." + pad,
            "expected": (False, "stock_picking"),
            "description": "Stock picking without macro context",
        },

        # Should PASS - stock analysis WITH macro context
        {
            "id": "macro_stock_001",
            "title": "Bank Stocks Plunge as Credit Crisis Spreads",
            "source": "fin_news",
            "content": "Financial sector stocks fell sharply as credit spreads widened to crisis levels. The systemic risk to the banking sector raises concerns about contagion." + pad,
            "expected": (True, "investment_keyword:credit"),
            "description": "Stock analysis with macro context (passes via keyword)",
        },

        # Should BLOCK - affiliate marketing
        {
            "id": "aff_001",
            "title": "Best Trading Platform Review",
            "source": "review_blog",
            "content": "Sign up through my link for a bonus! Use promo code TRADE100 for $100 free. Affiliate disclosure." + pad,
            "expected": (False, "affiliate_conflict"),
            "description": "Affiliate marketing content",
        },

        # Should BLOCK - clickbait in title
        {
            "id": "click_001",
            "title": "MARKET CRASH COMING! This one stock will save you!",
            "source": "clickbait_site",
            "content": "What Wall Street doesn't want you to know about this secret method for protecting your portfolio." + pad,
            "expected": (False, "clickbait"),
            "description": "Clickbait title",
        },

        # Should BLOCK - content too short
        {
            "id": "short_001", "title": "Short Article", "source": "unknown",
            "content": "Too short.",
            "expected": (False, "content_too_short"),
            "description": "Content too short",
        },
    ]

    print(f"Testing Investment Risk Pre-Filter v{InvestmentRiskPreFilterV5.VERSION}")
    print("=" * 80)

    passed_tests = 0
    failed_tests = 0

    for i, test in enumerate(test_cases, 1):
        article = {k: v for k, v in test.items() if k not in ('expected', 'description')}
        result = prefilter_instance.apply_filter(article)
        expected = test['expected']

        # Handle partial matching for content_too_short and variable reasons
        if expected[1].startswith('content_too_short') and result[1].startswith('content_too_short'):
            match = True
        else:
            match = (result[0] == expected[0] and result[1] == expected[1])

        status = "[PASS]" if match else "[FAIL]"
        if match:
            passed_tests += 1
        else:
            failed_tests += 1

        # Safe title for console output
        title_safe = test['title'].encode('ascii', 'ignore').decode('ascii')
        print(f"\nTest {i}: {status} - {test['description']}")
        print(f"  Title: {title_safe[:60]}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

    print("\n" + "=" * 80)
    print(f"Results: {passed_tests}/{passed_tests + failed_tests} tests passed")

    print("\nPre-filter Stats:")
    for key, value in prefilter_instance.get_stats().items():
        print(f"  {key}: {value}")

    return passed_tests == len(test_cases)


if __name__ == "__main__":
    success = test_prefilter()
    exit(0 if success else 1)
