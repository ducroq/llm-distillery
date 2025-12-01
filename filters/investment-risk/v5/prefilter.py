"""
Investment Risk v5 Prefilter

Blocks content that is almost never investment-relevant to save oracle costs.
Target: Block 40-60% of input, <5% false negative rate on actual investment content.

Philosophy: When in doubt, let it through. Better to score some noise than miss a signal.
"""

import re
from typing import Dict, Tuple

# Sources to BLOCK (almost never investment-relevant)
BLOCKED_SOURCE_PATTERNS = [
    # Academic papers (unless explicitly financial)
    r'arxiv',
    r'biorxiv',
    r'medrxiv',
    r'plos_one',
    r'scientific_reports',
    r'frontiers_',
    r'mdpi_',
    r'iacr_eprint',  # Cryptography papers

    # Developer/tech content
    r'github',
    r'dev_to',
    r'hackaday',
    r'hackernews',  # Mostly tech, but some finance - consider allowing
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
    r'greenme',  # Environmental lifestyle

    # Entertainment/gaming
    r'tom_hardware',  # Hardware reviews
    # r'wired_italia',  # Tech/business - sometimes has policy news, allow it
    r'numerama',  # French tech
    r'canaltech',  # Portuguese tech
    r'olhar_digital',  # Portuguese tech
    r'tecnoblog',  # Portuguese tech
    r'punto_informatico',  # Italian tech
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
    r'expansion',  # Spanish financial
    r'fd_',  # Dutch financial
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


def prefilter(article: Dict) -> Tuple[bool, str]:
    """
    Determine if article should be scored by oracle.

    Args:
        article: Dict with 'id', 'title', 'content', 'source', 'source_type'

    Returns:
        Tuple of (should_score: bool, reason: str)
        - True means article should be scored
        - False means article should be blocked
    """
    source = article.get('source', '').lower()
    source_type = article.get('source_type', '').lower()
    article_id = article.get('id', '').lower()
    title = article.get('title', '').lower()
    content = article.get('content', '')[:1000].lower()  # First 1000 chars

    # Combine for pattern matching
    source_combined = f"{source} {source_type} {article_id}"
    text_combined = f"{title} {content}"

    # 1. Check if explicitly allowed source
    for pattern in ALLOWED_SOURCE_PATTERNS:
        if re.search(pattern, source_combined, re.IGNORECASE):
            return True, f"allowed_source:{pattern}"

    # 2. Check for investment keywords in title/content (override blocks)
    for keyword in INVESTMENT_KEYWORDS:
        if re.search(keyword, text_combined, re.IGNORECASE):
            return True, f"investment_keyword:{keyword}"

    # 3. Check if blocked source
    for pattern in BLOCKED_SOURCE_PATTERNS:
        if re.search(pattern, source_combined, re.IGNORECASE):
            return False, f"blocked_source:{pattern}"

    # 4. Default: allow (when in doubt, score it)
    return True, "default_allow"


def get_stats() -> Dict:
    """Return prefilter configuration stats."""
    return {
        "blocked_patterns": len(BLOCKED_SOURCE_PATTERNS),
        "allowed_patterns": len(ALLOWED_SOURCE_PATTERNS),
        "investment_keywords": len(INVESTMENT_KEYWORDS),
        "philosophy": "When in doubt, let it through"
    }


if __name__ == "__main__":
    # Test with sample articles
    test_cases = [
        {"id": "arxiv_cs_123", "title": "Deep Learning for NLP", "source": "arxiv", "content": "We propose..."},
        {"id": "reuters_456", "title": "Fed raises rates", "source": "reuters", "content": "The Federal Reserve..."},
        {"id": "github_789", "title": "New ML framework", "source": "github", "content": "A fast library for..."},
        {"id": "tech_news_abc", "title": "Apple launches iPhone", "source": "techcrunch", "content": "Apple today..."},
        {"id": "science_xyz", "title": "Climate change could trigger recession", "source": "nature", "content": "New study shows..."},
    ]

    print("Prefilter Test Results:")
    print("-" * 60)
    for article in test_cases:
        should_score, reason = prefilter(article)
        status = "SCORE" if should_score else "BLOCK"
        print(f"{status:6s} | {article['title'][:40]:40s} | {reason}")
