"""
Investment Risk Pre-filter v2.0

Fast rule-based filter to block obvious NOISE content before LLM evaluation.
Focuses on blocking FOMO, stock picking, affiliate marketing, and clickbait.

Philosophy: "You can't predict crashes, but you can prepare for them."
This filter BLOCKS speculation and PASSES macro risk analysis.
"""

import re
from typing import Dict, Tuple


class InvestmentRiskPreFilterV2:
    """Fast rule-based pre-filter for investment risk content"""

    VERSION = "2.0"

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
        r'\b(portfolio|asset allocation|diversification|risk management)\b',
    ]

    # Affiliate and conflict patterns (block)
    AFFILIATE_CONFLICT_PATTERNS = [
        r'\b(sign up|join.*link|use code|promo code|discount code)\b',
        r'\b(through my link|affiliate|sponsored|paid promotion)\b',
        r'\b(my broker|my platform|get bonus|free trial)\b',
        r'\b(discord.*join|telegram.*channel|exclusive.*group)\b',
    ]

    # Clickbait patterns (block unless substantive)
    CLICKBAIT_PATTERNS = [
        r'\b(crash coming|market crash|stock market collapse)\b.*\!',
        r'\b(this one stock|the one stock|secret stock)\b',
        r'warren buffett.*secret',
        r'\b(what.*don\'t want you to know|they.*hiding)\b',
        r'🚀|💎|🌙|💰|🤑',  # Common FOMO emojis
    ]

    # Academic/research patterns (block - not actionable for hobby investors)
    ACADEMIC_PATTERNS = [
        r'\b(arxiv|arxiv\.org|doi\.org)\b',
        r'\b(proceedings of|conference on|symposium on)\b',
        r'\b(journal of|published in|research paper)\b',
        r'\b(abstract:.*introduction.*methodology)\b',
        r'\b(ieee|acm|springer|elsevier|mdpi)\b',
        r'\b(theoretical.*framework|statistical.*model|simulation)\b',
    ]

    def __init__(self):
        # Compile patterns for performance
        self.fomo_regex = re.compile('|'.join(self.FOMO_SPECULATION_PATTERNS), re.IGNORECASE)
        self.stock_picking_regex = re.compile('|'.join(self.STOCK_PICKING_PATTERNS), re.IGNORECASE)
        self.macro_context_regex = re.compile('|'.join(self.MACRO_CONTEXT_PATTERNS), re.IGNORECASE)
        self.affiliate_regex = re.compile('|'.join(self.AFFILIATE_CONFLICT_PATTERNS), re.IGNORECASE)
        self.clickbait_regex = re.compile('|'.join(self.CLICKBAIT_PATTERNS), re.IGNORECASE)
        self.academic_regex = re.compile('|'.join(self.ACADEMIC_PATTERNS), re.IGNORECASE)

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be labeled by LLM.

        Returns:
            (True, "passed") - Send to LLM
            (False, "fomo_speculation") - Block FOMO/speculation
            (False, "stock_picking") - Block stock picking
            (False, "affiliate_conflict") - Block affiliate marketing
            (False, "clickbait") - Block clickbait
            (False, "academic_research") - Block academic papers
        """
        title = article.get('title', '').lower()
        text = article.get('text', '').lower()
        combined = f"{title} {text}"

        # Check for affiliate/conflict first (highest confidence block)
        if self.affiliate_regex.search(combined):
            return (False, "affiliate_conflict")

        # Check for clickbait in title (before other patterns)
        if self.clickbait_regex.search(title):
            return (False, "clickbait")

        # Check for academic papers (not actionable for hobby investors)
        if self.academic_regex.search(combined):
            return (False, "academic_research")

        # Check for stock picking (but allow if macro context present)
        if self.stock_picking_regex.search(combined):
            if not self.macro_context_regex.search(combined):
                return (False, "stock_picking")

        # Check for FOMO/speculation (last, as it's broadest)
        if self.fomo_regex.search(combined):
            return (False, "fomo_speculation")

        # Passed all filters
        return (True, "passed")


# Test the pre-filter
def test_prefilter():
    """Test investment risk pre-filter"""
    prefilter = InvestmentRiskPreFilterV2()

    test_cases = [
        # PASS: Macro risk analysis
        {
            "title": "Yield Curve Inversion Signals Recession Risk as Fed Maintains Tight Policy",
            "text": "The Treasury yield curve has inverted for the third consecutive month, historically a reliable recession indicator. With unemployment rising and credit spreads widening, economists warn of systemic risks...",
            "expected": True,
            "reason": "passed"
        },
        # BLOCK: Clickbait (exclamation marks trigger clickbait first)
        {
            "title": "This Hot Stock is Going to the Moon! Buy Now! 🚀",
            "text": "Don't miss out on this moonshot opportunity! Get rich quick with this hidden gem!",
            "expected": False,
            "reason": "clickbait"
        },
        # BLOCK: Stock picking without macro context
        {
            "title": "Top 10 Stocks to Buy Now - Analyst Strong Buy Ratings",
            "text": "Our analysts have upgraded these stocks to strong buy. Price target $150. Earnings beat expected...",
            "expected": False,
            "reason": "stock_picking"
        },
        # PASS: Stock analysis WITH macro context
        {
            "title": "Bank Stocks Plunge as Credit Crisis Spreads",
            "text": "Financial sector stocks fell sharply as credit spreads widened to crisis levels. The systemic risk to the banking sector raises concerns about contagion...",
            "expected": True,
            "reason": "passed"
        },
        # BLOCK: Affiliate marketing
        {
            "title": "Best Trading Platform - Sign Up Through My Link for Bonus",
            "text": "Join this platform using my promo code for $100 free! Affiliate link in bio...",
            "expected": False,
            "reason": "affiliate_conflict"
        },
        # BLOCK: Clickbait
        {
            "title": "MARKET CRASH COMING! This One Stock Will Save You!",
            "text": "What Wall Street doesn't want you to know about the secret stock Warren Buffett is hiding...",
            "expected": False,
            "reason": "clickbait"
        },
        # PASS: Policy risk analysis
        {
            "title": "Fed Emergency Meeting Raises Policy Error Concerns",
            "text": "The Federal Reserve's unexpected emergency meeting has spooked markets. Policy uncertainty and potential rate hike errors could trigger credit market stress...",
            "expected": True,
            "reason": "passed"
        },
        # BLOCK: Meme stock discussion
        {
            "title": "Diamond hands! This meme stock is headed for a 100x!",
            "text": "YOLO into this penny stock! To the moon! Lambo time! 🚀💎",
            "expected": False,
            "reason": "fomo_speculation"
        },
        # BLOCK: Academic research paper (arxiv)
        {
            "title": "Correlation Networks in Chinese Stock Markets",
            "text": "Abstract: This paper presents a statistical model for analyzing correlation structures in equity markets. Published in arxiv.org. Methodology section discusses simulation approaches...",
            "expected": False,
            "reason": "academic_research"
        },
        # BLOCK: Academic research paper (journal)
        {
            "title": "LLM Inference Reproducibility Analysis",
            "text": "Published in Journal of Computer Science. We present a theoretical framework for analyzing non-deterministic behavior in large language models...",
            "expected": False,
            "reason": "academic_research"
        },
        # BLOCK: Academic research paper (conference)
        {
            "title": "Novel Electrical Sensors for Industrial Applications",
            "text": "Proceedings of the IEEE Conference on Sensors 2024. This research paper presents experimental results from MDPI laboratory testing...",
            "expected": False,
            "reason": "academic_research"
        },
    ]

    print(f"\nTesting Investment Risk Pre-filter v{InvestmentRiskPreFilterV2.VERSION}\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        article = {"title": test["title"], "text": test["text"]}
        should_label, reason = prefilter.should_label(article)

        expected_result = test["expected"]
        expected_reason = test["reason"]

        if should_label == expected_result and reason == expected_reason:
            status = "[PASS]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1

        print(f"\nTest {i}: {status}")
        # Remove emoji from title for Windows console
        title_safe = test['title'].encode('ascii', 'ignore').decode('ascii')
        print(f"  Title: {title_safe[:70]}...")
        print(f"  Expected: should_label={expected_result}, reason='{expected_reason}'")
        print(f"  Got:      should_label={should_label}, reason='{reason}'")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_prefilter()
    exit(0 if success else 1)
