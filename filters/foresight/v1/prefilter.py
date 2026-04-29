"""
Foresight Pre-Filter v1.0

ADR-018 declarative shape (data-only): the six text-pattern exclusion
categories live in EXCLUSION_PATTERNS dict (compiled by BasePreFilter.__init__),
and the six positive-signal pattern groups live in POSITIVE_PATTERN_GROUPS
(compiled locally — semantics differ from base's POSITIVE_PATTERNS, see below).

apply_filter stays custom because:

- The override mechanism is "distinct positive *categories* with at least one
  match >= 3", not base's "total POSITIVE_PATTERNS match count >= POSITIVE_THRESHOLD".
  A single repeated keyword would inflate base's count but counts as one category
  here. Different semantics, so POSITIVE_PATTERN_GROUPS is a custom slot rather
  than shadowing base's POSITIVE_PATTERNS.
- Two distinct pass reasons: "passed_positive_signals" (>=3 categories fire)
  and "passed" (no block patterns triggered, no strong positive signals
  either). Base pipeline returns just "passed".
- URL-based domain exclusions run before content checks.

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

History:
- v1.0 (2026-04-29): migrated to declarative BasePreFilter shape (#52, ADR-018).
  No behavior change — pattern set, override semantics ("distinct positive
  categories >= 3"), iteration order, and pass-reason distinction all preserved.
  Self-test 10/10 passes; pattern counts identical (4/4/3/4/3/3 block;
  8/4/4/6/3/15 positive; 8/5 domain).
- v1.0 prior: hand-crafted seeds + soft caps + anti-hallucination prompt rule
  (foresight-v1-lessons memory entry).
"""

import re
from typing import Dict, List, Tuple

from filters.common.base_prefilter import BasePreFilter


class ForesightPreFilterV1(BasePreFilter):
    """Fast rule-based pre-filter for foresight content.

    v1.0 (declarative-shape exclusion data per ADR-018; custom apply_filter
    retained for distinct-categories-fired override semantics, two pass
    reasons, and URL-based domain exclusions).
    """

    VERSION = "1.0"

    # === DOMAIN EXCLUSIONS (URL-based) ===
    # Unlikely to contain governance decisions.

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

    # Mapping consumed by BasePreFilter._check_domain_exclusions (hoisted
    # from per-filter copies). Iteration order = legacy check order.
    DOMAIN_EXCLUSIONS = {
        "excluded_domain_sports_entertainment": SPORTS_ENTERTAINMENT_DOMAINS,
        "excluded_domain_personal_finance": PERSONAL_FINANCE_DOMAINS,
    }

    # === ADR-018 EXCLUSION_PATTERNS ===
    # Six block categories without per-category exceptions. Iteration order
    # matches the legacy apply_filter() check order: sports, entertainment,
    # crime, product_launch, listicle, social_media. Category keys match the
    # (False, "<reason>") tuples this filter emits — no "excluded_" prefix.
    EXCLUSION_PATTERNS = {
        'sports': [
            r'\b(scored a goal|final score|match result|championship game)\b',
            r'\b(transfer window|transfer fee|signed a contract|free agent)\b',
            r'\b(premier league|champions league|world cup|olympics|nba|nfl|mlb)\b',
            r'\b(playoff|tournament bracket|relegation|promotion race)\b',
        ],
        'entertainment': [
            r'\b(box office|opening weekend|movie review|film review)\b',
            r'\b(album release|concert tour|grammy|oscar nomination)\b',
            r'\b(celebrity|paparazzi|red carpet|gossip)\b',
            r'\b(streaming premiere|netflix|disney\+|hulu)\b.*\b(release|premiere|season)\b',
        ],
        'crime': [
            r'\b(mugshot|police chase|manhunt|crime scene)\b',
            r'\b(murder suspect|robbery conviction|homicide investigation)\b',
            r'\b(prison sentence|sentenced to \d+ years)\b',
        ],
        'product_launch': [
            r'\b(unboxing|specs reveal|benchmark results?|hands-on review)\b',
            r'\b(price starts at \$|available in stores|buy now|add to cart)\b',
            r'\b(iphone \d|galaxy s\d|pixel \d)\b.*\b(release|launch|review)\b',
            r'\b(product launch)\b.*\b(device|gadget|app|software|hardware)\b',
        ],
        'listicle': [
            r'\b\d+ (best|top|ways to|tips for|reasons why|things you)\b',
            r"\b(you won't believe|everything you need to know about|here's why you should)\b",
            r'\b(ultimate guide to|definitive guide to)\b',
        ],
        'social_media': [
            r'\b(went viral|trending on|twitter reacts|tiktok trend)\b',
            r'\b(followers|likes and shares|engagement rate)\b',
            r'\b(meme|viral video|social media storm)\b',
        ],
    }

    # === POSITIVE PATTERN GROUPS ===
    # Six categories of foresight-positive signals. The override fires when
    # at least three distinct categories have any match (NOT when the total
    # match count is >=3 — see module docstring). Compiled locally because
    # the semantics differ from BasePreFilter.POSITIVE_PATTERNS (which counts
    # total matches via POSITIVE_THRESHOLD). Custom slot avoids confusion.
    POSITIVE_PATTERN_GROUPS = {
        'policy_governance': [
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
        ],
        'course_correction': [
            r'\b(reversed|reversal|U-turn|u-turn|policy shift)\b',
            r'\b(admitted failure|acknowledged mistake|changed course)\b',
            r'\b(lessons learned|learned from|evidence showed)\b',
            r'\b(reformed|overhauled|redesigned|restructured)\b',
        ],
        'education': [
            r'\b(curriculum reform|education reform|education policy)\b',
            r'\b(school system|education system|national curriculum)\b',
            r'\b(teacher training|pedagogy|educational research)\b',
            r'\b(knowledge transfer|capacity building|skills for the future)\b',
        ],
        'intergenerational': [
            r'\b(pension reform|pension system|retirement system)\b',
            r'\b(sovereign wealth fund|trust fund|endowment)\b',
            r'\b(climate agreement|climate policy|emissions target)\b',
            r'\b(infrastructure investment|infrastructure plan)\b',
            r'\b(debt brake|fiscal rule|balanced budget)\b',
            r'\b(future generations commissioner|ombudsman)\b',
        ],
        'indigenous_governance': [
            r'\b(indigenous governance|indigenous knowledge|traditional management)\b',
            r'\b(aboriginal|first nations|native title)\b',
            r'\b(land rights|land management|stewardship)\b',
        ],
        'multilingual_positive': [
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
        ],
    }

    # Number of distinct positive-signal categories required to bypass blocks.
    # 3 of 6 → meaningful signal density without requiring all categories.
    POSITIVE_CATEGORIES_THRESHOLD = 3

    def __init__(self):
        """Compile POSITIVE_PATTERN_GROUPS; base compiles EXCLUSION_PATTERNS
        into self._compiled_exclusions."""
        super().__init__()
        self._compiled_positive_groups: Dict[str, List[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.POSITIVE_PATTERN_GROUPS.items()
        }

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for scoring.

        Strategy: Block clear noise, pass anything with policy/governance signals.
        Designed to be permissive — the oracle prompt handles false positives
        via soft content-type caps.

        Custom flow (not BasePreFilter.apply_filter): the distinct-categories-fired
        override + two-tier pass reasons + URL domain exclusions don't fit the
        standard pipeline.

        Returns:
            (should_score, reason)
            - (True, "passed_positive_signals"): >=3 distinct positive-signal
                  categories fired — pass even if blocks would also fire
            - (True, "passed"): No blocks triggered, weak/no positive signal
            - (False, "<category>"): Block with reason (sports / entertainment
                  / crime / product_launch / listicle / social_media /
                  excluded_domain_*)
        """
        # Validate article structure
        is_valid, validation_reason = self.validate_article(article)
        if not is_valid:
            return False, validation_reason

        # Content length
        passed, reason = self.check_content_length(article)
        if not passed:
            return False, reason

        # Domain exclusions
        url = article.get('url', '')
        if url:
            domain_block = self._check_domain_exclusions(url)
            if domain_block:
                return False, domain_block

        # Combined cleaned text
        combined_text = self._get_combined_clean_text(article)

        # Count distinct positive-signal CATEGORIES that fire (not total matches).
        # Important: a single repeated keyword in one category counts as 1 here,
        # not as N. Differs from BasePreFilter.POSITIVE_THRESHOLD semantics.
        positive_categories = sum(
            1
            for compiled in self._compiled_positive_groups.values()
            if self.has_any_pattern(combined_text, compiled)
        )

        # Strong positive signals — pass with distinct reason, even if blocks
        # would also fire.
        if positive_categories >= self.POSITIVE_CATEGORIES_THRESHOLD:
            return True, "passed_positive_signals"

        # Block patterns — only consulted when there are NO positive signals
        # at all. Articles with 1-2 categories of positive signals pass through
        # to the oracle for nuanced judgment.
        if positive_categories == 0:
            for category, compiled_patterns in self._compiled_exclusions.items():
                if self.has_any_pattern(combined_text, compiled_patterns):
                    return False, category

        # Default: pass. The oracle handles nuance.
        return True, "passed"

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        return {
            'version': self.VERSION,
            'positive_categories_threshold': self.POSITIVE_CATEGORIES_THRESHOLD,
            'block_patterns': {
                category: len(patterns)
                for category, patterns in self.EXCLUSION_PATTERNS.items()
            },
            'positive_patterns': {
                # Reported with the legacy stats key name (`multilingual` for
                # the multilingual_positive group) for backward-compat with
                # any consumer that reads stats dicts. Internal class-attr
                # name is `multilingual_positive` for clarity.
                ('multilingual' if cat == 'multilingual_positive' else cat): len(patterns)
                for cat, patterns in self.POSITIVE_PATTERN_GROUPS.items()
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
