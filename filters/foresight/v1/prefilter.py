"""
Foresight Pre-Filter v1.0

Passes content likely to contain foresighted decision-making:
- Government policy with long-term framing
- Institutional reform and governance changes
- Education system redesigns, curriculum reform
- Course corrections (admitting failure, policy reversals)
- Intergenerational contracts (climate, pensions, infrastructure)
- Indigenous governance adoption

Blocks content unlikely to contain decision-making context:
- Sports, entertainment, celebrity news
- Pure technology/product announcements without governance context
- Crime reporting, breaking news events
- Personal finance, individual advice
- Social media chatter, listicles
"""

import re
from typing import Dict, Tuple

from filters.common.base_prefilter import BasePreFilter


class ForesightPreFilterV1(BasePreFilter):
    """Fast rule-based pre-filter for foresight content"""

    VERSION = "1.0"

    # === DOMAIN EXCLUSIONS (unlikely to contain governance decisions) ===

    SPORTS_ENTERTAINMENT_DOMAINS = [
        'espn.com',
        'bleacherreport.com',
        'tmz.com',
        'eonline.com',
        'people.com',
        'variety.com',
        'deadline.com',
        'hollywoodreporter.com',
    ]

    PERSONAL_FINANCE_DOMAINS = [
        'nerdwallet.com',
        'bankrate.com',
        'investopedia.com',
        'thebalancemoney.com',
        'creditkarma.com',
    ]

    # === BLOCK PATTERNS (content without decision-making context) ===

    SPORTS_PATTERNS = [
        r'\b(scored a goal|final score|match result|championship game)\b',
        r'\b(transfer window|transfer fee|signed a contract|free agent)\b',
        r'\b(premier league|champions league|world cup|olympics|nba|nfl|mlb)\b',
        r'\b(playoff|tournament bracket|relegation|promotion race)\b',
    ]

    ENTERTAINMENT_PATTERNS = [
        r'\b(box office|opening weekend|movie review|film review)\b',
        r'\b(album release|concert tour|grammy|oscar nomination)\b',
        r'\b(celebrity|paparazzi|red carpet|gossip)\b',
        r'\b(streaming premiere|netflix|disney\+|hulu)\b.*\b(release|premiere|season)\b',
    ]

    CRIME_PATTERNS = [
        r'\b(mugshot|police chase|manhunt|crime scene)\b',
        r'\b(murder suspect|robbery conviction|homicide investigation)\b',
        r'\b(prison sentence|sentenced to \d+ years)\b',
    ]

    PRODUCT_LAUNCH_PATTERNS = [
        r'\b(unboxing|specs reveal|benchmark results?|hands-on review)\b',
        r'\b(price starts at \$|available in stores|buy now|add to cart)\b',
        r'\b(iphone \d|galaxy s\d|pixel \d)\b.*\b(release|launch|review)\b',
        r'\b(product launch)\b.*\b(device|gadget|app|software|hardware)\b',
    ]

    LISTICLE_PATTERNS = [
        r'\b\d+ (best|top|ways to|tips for|reasons why|things you)\b',
        r"\b(you won't believe|everything you need to know about|here's why you should)\b",
        r'\b(ultimate guide to|definitive guide to)\b',
    ]

    SOCIAL_MEDIA_PATTERNS = [
        r'\b(went viral|trending on|twitter reacts|tiktok trend)\b',
        r'\b(followers|likes and shares|engagement rate)\b',
        r'\b(meme|viral video|social media storm)\b',
    ]

    # === POSITIVE SIGNAL PATTERNS (foresight indicators) ===

    POLICY_GOVERNANCE_PATTERNS = [
        # Policy and governance
        r'\b(government policy|public policy|policy reform|policy change)\b',
        r'\b(legislation|regulatory reform|constitutional|amendment)\b',
        r'\b(parliament|congress|senate|cabinet|ministry)\b',
        r'\b(governance|institutional reform|institutional change)\b',

        # Long-term framing
        r'\b(long-term|long term|decade-long|multi-year|multi-decade)\b',
        r'\b(generational|intergenerational|future generations)\b',
        r'\b(2030|2040|2050|2060|2070|2080|2100)\b',
        r'\b(\d+ year plan|\d+-year strategy|\d+ year horizon)\b',
    ]

    COURSE_CORRECTION_PATTERNS = [
        r'\b(reversed|reversal|U-turn|u-turn|policy shift)\b',
        r'\b(admitted failure|acknowledged mistake|changed course)\b',
        r'\b(lessons learned|learned from|evidence showed)\b',
        r'\b(reformed|overhauled|redesigned|restructured)\b',
    ]

    EDUCATION_PATTERNS = [
        r'\b(curriculum reform|education reform|education policy)\b',
        r'\b(school system|education system|national curriculum)\b',
        r'\b(teacher training|pedagogy|educational research)\b',
        r'\b(knowledge transfer|capacity building|skills for the future)\b',
    ]

    INTERGENERATIONAL_PATTERNS = [
        r'\b(pension reform|pension system|retirement system)\b',
        r'\b(sovereign wealth fund|trust fund|endowment)\b',
        r'\b(climate agreement|climate policy|emissions target)\b',
        r'\b(infrastructure investment|infrastructure plan)\b',
        r'\b(debt brake|fiscal rule|balanced budget)\b',
        r'\b(future generations commissioner|ombudsman)\b',
    ]

    INDIGENOUS_GOVERNANCE_PATTERNS = [
        r'\b(indigenous governance|indigenous knowledge|traditional management)\b',
        r'\b(aboriginal|first nations|native title)\b',
        r'\b(land rights|land management|stewardship)\b',
    ]

    # === MULTILINGUAL POSITIVE PATTERNS ===

    MULTILINGUAL_POSITIVE_PATTERNS = [
        # Dutch
        r'\b(overheidsbeleid|beleidshervorming|wetgeving|grondwet)\b',
        r'\b(lange termijn|generaties|toekomstige generaties)\b',
        r'\b(onderwijshervorming|curriculum|pensioenstelsel)\b',

        # German
        r'\b(regierungspolitik|politikreform|gesetzgebung|verfassung)\b',
        r'\b(langfristig|generationen|zukünftige generationen)\b',
        r'\b(bildungsreform|lehrplan|rentensystem|schuldenbremse)\b',

        # French
        r'\b(politique gouvernementale|réforme|législation|constitution)\b',
        r'\b(long terme|générations|générations futures)\b',
        r'\b(réforme éducative|programme scolaire|système de retraite)\b',

        # Portuguese
        r'\b(política governamental|reforma|legislação|constituição)\b',
        r'\b(longo prazo|gerações|gerações futuras)\b',
        r'\b(reforma educacional|currículo|sistema previdenciário)\b',

        # Spanish
        r'\b(política gubernamental|reforma|legislación|constitución)\b',
        r'\b(largo plazo|generaciones|generaciones futuras)\b',
        r'\b(reforma educativa|currículo|sistema de pensiones)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        super().__init__()

        # Block patterns
        self.sports_regex = [re.compile(p, re.IGNORECASE) for p in self.SPORTS_PATTERNS]
        self.entertainment_regex = [re.compile(p, re.IGNORECASE) for p in self.ENTERTAINMENT_PATTERNS]
        self.crime_regex = [re.compile(p, re.IGNORECASE) for p in self.CRIME_PATTERNS]
        self.product_launch_regex = [re.compile(p, re.IGNORECASE) for p in self.PRODUCT_LAUNCH_PATTERNS]
        self.listicle_regex = [re.compile(p, re.IGNORECASE) for p in self.LISTICLE_PATTERNS]
        self.social_media_regex = [re.compile(p, re.IGNORECASE) for p in self.SOCIAL_MEDIA_PATTERNS]

        # Positive patterns
        self.policy_regex = [re.compile(p, re.IGNORECASE) for p in self.POLICY_GOVERNANCE_PATTERNS]
        self.correction_regex = [re.compile(p, re.IGNORECASE) for p in self.COURSE_CORRECTION_PATTERNS]
        self.education_regex = [re.compile(p, re.IGNORECASE) for p in self.EDUCATION_PATTERNS]
        self.intergenerational_regex = [re.compile(p, re.IGNORECASE) for p in self.INTERGENERATIONAL_PATTERNS]
        self.indigenous_regex = [re.compile(p, re.IGNORECASE) for p in self.INDIGENOUS_GOVERNANCE_PATTERNS]
        self.multilingual_positive_regex = [re.compile(p, re.IGNORECASE) for p in self.MULTILINGUAL_POSITIVE_PATTERNS]

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for scoring.

        Strategy: Block clear noise, pass anything with policy/governance signals.
        Designed to be permissive — the oracle prompt handles false positives
        via soft content-type caps.

        Args:
            article: Dict with 'title' and 'text'/'content' keys

        Returns:
            (should_score, reason)
            - (True, "passed"): Send to LLM
            - (False, "reason"): Block with reason
        """
        # Check article structure
        is_valid, validation_reason = self.validate_article(article)
        if not is_valid:
            return False, validation_reason

        # Check content length
        passed, reason = self.check_content_length(article)
        if not passed:
            return False, reason

        # Check domain exclusions
        url = article.get('url', '')
        if url:
            domain_block = self._check_domain_exclusions(url)
            if domain_block:
                return False, domain_block

        # Get combined text for analysis
        combined_text = self._get_combined_clean_text(article)

        # Count positive signals
        positive_count = self._count_positive_signals(combined_text)

        # If strong positive signals, always pass (even if block patterns match)
        if positive_count >= 3:
            return True, "passed_positive_signals"

        # Check block patterns — only block if NO positive signals
        if positive_count == 0:
            if self.has_any_pattern(combined_text, self.sports_regex):
                return False, "sports"
            if self.has_any_pattern(combined_text, self.entertainment_regex):
                return False, "entertainment"
            if self.has_any_pattern(combined_text, self.crime_regex):
                return False, "crime"
            if self.has_any_pattern(combined_text, self.product_launch_regex):
                return False, "product_launch"
            if self.has_any_pattern(combined_text, self.listicle_regex):
                return False, "listicle"
            if self.has_any_pattern(combined_text, self.social_media_regex):
                return False, "social_media"

        # Pass everything else — the oracle handles nuance
        return True, "passed"

    def _check_domain_exclusions(self, url: str) -> str:
        """Check if URL is from an excluded domain"""
        url_lower = url.lower()

        for domain in self.SPORTS_ENTERTAINMENT_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_sports_entertainment"

        for domain in self.PERSONAL_FINANCE_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_personal_finance"

        return ""

    def _count_positive_signals(self, text: str) -> int:
        """Count how many foresight-positive signal CATEGORIES are present.

        Returns the number of distinct pattern groups that fired (0-6),
        not the total number of individual matches. This prevents a single
        repeated keyword from inflating the count.
        """
        count = 0
        for group in [
            self.policy_regex,
            self.correction_regex,
            self.education_regex,
            self.intergenerational_regex,
            self.indigenous_regex,
            self.multilingual_positive_regex,
        ]:
            if self.has_any_pattern(text, group):
                count += 1
        return count

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        return {
            'version': self.VERSION,
            'block_patterns': {
                'sports': len(self.SPORTS_PATTERNS),
                'entertainment': len(self.ENTERTAINMENT_PATTERNS),
                'crime': len(self.CRIME_PATTERNS),
                'product_launch': len(self.PRODUCT_LAUNCH_PATTERNS),
                'listicle': len(self.LISTICLE_PATTERNS),
                'social_media': len(self.SOCIAL_MEDIA_PATTERNS),
            },
            'positive_patterns': {
                'policy_governance': len(self.POLICY_GOVERNANCE_PATTERNS),
                'course_correction': len(self.COURSE_CORRECTION_PATTERNS),
                'education': len(self.EDUCATION_PATTERNS),
                'intergenerational': len(self.INTERGENERATIONAL_PATTERNS),
                'indigenous_governance': len(self.INDIGENOUS_GOVERNANCE_PATTERNS),
                'multilingual': len(self.MULTILINGUAL_POSITIVE_PATTERNS),
            },
            'domain_exclusions': {
                'sports_entertainment': len(self.SPORTS_ENTERTAINMENT_DOMAINS),
                'personal_finance': len(self.PERSONAL_FINANCE_DOMAINS),
            },
        }


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = ForesightPreFilterV1()

    test_cases = [
        # Should BLOCK - Sports
        {
            'title': 'Manchester United wins Premier League title',
            'text': 'In an exciting championship game, Manchester United scored a goal in the final '
                    'minutes to clinch the Premier League title. The match result was celebrated by '
                    'millions of fans worldwide. The transfer window will now begin with clubs preparing '
                    'their bids for key players as the transfer fee market heats up significantly.',
            'expected': (False, 'sports')
        },

        # Should BLOCK - Entertainment
        {
            'title': 'Marvel breaks box office records with new release',
            'text': 'The latest Marvel film broke box office records in its opening weekend, earning '
                    '$350 million globally. Critics gave the movie review mixed reactions, while fans '
                    'celebrated on the red carpet premiere. The film is expected to dominate the awards '
                    'season with potential Oscar nominations. Celebrity appearances at the premiere drew '
                    'massive crowds and paparazzi attention.',
            'expected': (False, 'entertainment')
        },

        # Should BLOCK - Product launch
        {
            'title': 'Apple announces iPhone 17 with specs reveal',
            'text': 'Apple has announced the iPhone 17 at a product launch event in Cupertino. The device '
                    'features impressive benchmark results and is now available for pre-order. Price starts at '
                    '$999 and will be available in stores next week. Early hands-on review feedback has been '
                    'positive, with users praising the new camera specs reveal and performance improvements.',
            'expected': (False, 'product_launch')
        },

        # Should PASS - Government policy reform
        {
            'title': 'EU passes landmark climate legislation with 2050 targets',
            'text': 'The European Parliament approved comprehensive legislation requiring member states to '
                    'achieve net-zero emissions by 2050. The regulatory reform includes binding targets for '
                    '2030 and 2040, with penalties for non-compliance. The long-term policy change was developed '
                    'over five years of negotiation across 27 member states. Future generations will benefit from '
                    'the institutional reform that embeds climate targets into constitutional frameworks.',
            'expected': (True, 'passed')
        },

        # Should PASS - Education reform
        {
            'title': 'Singapore overhauls national curriculum for 21st century skills',
            'text': 'Singapore announced a comprehensive curriculum reform designed to prepare students for '
                    'the economy of 2040. The education system redesign, backed by a decade of educational '
                    'research, introduces cross-disciplinary learning and systems thinking. Teacher training '
                    'programs have been restructured to support the new pedagogy. The education policy represents '
                    'a deliberate shift toward long-term capacity building for future generations.',
            'expected': (True, 'passed')
        },

        # Should PASS - Course correction
        {
            'title': 'Netherlands reverses nitrogen policy after decade of farmer protests',
            'text': 'The Dutch government acknowledged mistakes in its nitrogen policy and announced a major '
                    'policy shift after evidence showed the previous approach was failing. The reformed policy '
                    'includes a 15-year transition plan developed through lessons learned from a decade of '
                    'conflict. The government admitted failure on the previous aggressive approach and '
                    'restructured the institutional framework for agricultural governance.',
            'expected': (True, 'passed')
        },

        # Should PASS - Multilingual (Dutch governance article)
        {
            'title': 'Kabinet presenteert hervormingsplan pensioenstelsel',
            'text': 'Het kabinet heeft vandaag een ingrijpend hervormingsplan voor het pensioenstelsel '
                    'gepresenteerd. De lange termijn visie richt zich op toekomstige generaties en bevat '
                    'nieuwe wetgeving die het stelsel duurzamer moet maken. Het overheidsbeleid wordt '
                    'aangepast om de houdbaarheid voor de komende decennia te garanderen. De beleidshervorming '
                    'gaat gepaard met een overgangsperiode van tien jaar.',
            'expected': (True, 'passed')
        },

        # Should PASS - Policy article even with some entertainment overlap
        {
            'title': 'Government launches long-term arts funding reform',
            'text': 'The government announced a decade-long institutional reform of arts funding, moving from '
                    'annual grants to multi-year commitments. The policy reform acknowledges that long-term '
                    'planning is essential for cultural institutions to thrive. The legislation includes '
                    'constitutional protections for minimum funding levels. Celebrity endorsement of the '
                    'policy drew attention but the governance changes are structural.',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Listicle
        {
            'title': '10 Best Ways to Invest Your Money in 2026',
            'text': 'Here are the 10 best ways to invest your money this year. You need to know about these '
                    'exciting opportunities. From stocks to crypto, this ultimate guide to investing covers '
                    'everything you need to know about building wealth. You won\'t believe how easy it is to '
                    'get started with just $100. These tips for beginners will help you maximize returns quickly.',
            'expected': (False, 'listicle')
        },

        # Should BLOCK - Content too short
        {
            'title': 'Brief note',
            'text': 'This is too short to analyze.',
            'expected': (False, 'content_too_short')
        },
    ]

    print("Testing Foresight Pre-Filter v1.0")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        result = prefilter.apply_filter(test)
        expected = test['expected']

        result_matches = (
            result[0] == expected[0] and
            (result[1] == expected[1] or result[1].startswith(expected[1]))
        )

        status = "[PASS]" if result_matches else "[FAIL]"
        if result_matches:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {i}: {status}")
        print(f"Title: {test['title'][:60]}...")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("\nPre-filter Statistics:")
    import json
    print(json.dumps(prefilter.get_statistics(), indent=2))


if __name__ == '__main__':
    test_prefilter()
