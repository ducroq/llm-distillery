"""
Belonging Pre-Filter v1.0

Blocks content that commodifies or commercializes belonging:
- Wellness industry (longevity hacks, Blue Zone diet tips, biohacking)
- Professional networking (LinkedIn, conferences, career social capital)
- Tourism (Blue Zone tourism, experiencing "authentic" community)
- Self-help (find your tribe, build community as individual project)
- Corporate (company culture, team building, workplace belonging)

Passes content showing organic community bonds, intergenerational ties, and rootedness.
"""

import re
from typing import Dict, Tuple

from filters.common.base_prefilter import BasePreFilter


class BelongingPreFilterV1(BasePreFilter):
    """Fast rule-based pre-filter for belonging content"""

    VERSION = "1.0"

    # === DOMAIN EXCLUSIONS ===

    WELLNESS_DOMAINS = [
        'mindbodygreen.com',
        'wellandgood.com',
        'goop.com',
        'healthline.com',
        'verywellmind.com',
        'psychologytoday.com',
        'longevity.technology',
        'lifeextension.com',
    ]

    PROFESSIONAL_NETWORKING_DOMAINS = [
        'linkedin.com',
        'forbes.com/sites/forbescoaches',
        'entrepreneur.com',
        'inc.com',
        'fastcompany.com',
        'hbr.org',
    ]

    TRAVEL_TOURISM_DOMAINS = [
        'lonelyplanet.com',
        'tripadvisor.com',
        'travelandleisure.com',
        'cntraveler.com',
        'afar.com',
        'fodors.com',
    ]

    SELF_HELP_DOMAINS = [
        'tinybuddha.com',
        'marcandangel.com',
        'pickthebrain.com',
        'lifehack.org',
        'zenhabits.net',
    ]

    # === WELLNESS INDUSTRY PATTERNS ===
    # Note: Patterns should be specific enough to avoid false negatives

    WELLNESS_PATTERNS = [
        # Longevity/biohacking (specific phrases)
        r'\b(longevity hacks?|longevity secrets?|longevity tips?)\b',
        r'\b(biohacks?|biohacking|life extension)\b',
        r'\b(anti-aging|anti aging|reverse aging)\b',
        r'\b(blue zone diet|okinawa diet|mediterranean diet secrets)\b',
        r'\b(centenarian secrets|secrets of the longest-lived|secrets to living longer)\b',

        # Health optimization (specific phrases, not generic words)
        r'\b(health hack|wellness hack|wellness routine|wellness protocol)\b',
        r'\b(fasting protocol|caloric restriction for longevity)\b',
        r'\b(sleep hack|sleep optimization protocol)\b',
        r'\b(nootropic|peptide therapy|supplement stack)\b',
    ]

    # === PROFESSIONAL NETWORKING PATTERNS ===

    NETWORKING_PATTERNS = [
        # Career networking
        r'\b(build your network|networking tips|networking event)\b',
        r'\b(linkedin|professional network|business network)\b',
        r'\b(social capital|career capital|professional connections)\b',
        r'\b(mastermind group|accountability partner|mentor network)\b',

        # Business community
        r'\b(startup ecosystem|founder community|entrepreneur community)\b',
        r'\b(coworking|co-working|wework)\b',
        r'\b(industry conference|networking conference)\b',
    ]

    # === TOURISM PATTERNS ===
    # Note: Avoid "authentic community" - appears in positive belonging articles too

    TOURISM_PATTERNS = [
        # Blue Zone tourism (specific)
        r'\b(visit okinawa|travel to sardinia|nicoya tourism)\b',
        r'\b(blue zone tour|blue zone travel|blue zone destination)\b',
        r'\b(ikaria greece travel|loma linda visit)\b',

        # Experience tourism (specific tourism phrases)
        r'\b(cultural tourism|wellness tourism|wellness retreat)\b',
        r'\b(destination wellness|retreat center|wellness resort)\b',
        r'\b(immersive travel experience|experiential travel)\b',

        # Travel guide language (combined with location context)
        r'\b(travel guide|travel itinerary|travel tips)\b',
        r'\b(best time to visit|where to stay in)\b',
        r'\b(bucket list destination|must-visit destination)\b',
    ]

    # === SELF-HELP PATTERNS ===

    SELF_HELP_PATTERNS = [
        # Find your tribe
        r'\b(find your tribe|build your tribe|create your tribe)\b',
        r'\b(build community|building community|create community)\b',
        r'\b(combat loneliness|overcome loneliness|loneliness epidemic)\b',
        r'\b(make friends|how to make friends|finding friends)\b',

        # Personal development
        r'\b(life design|lifestyle design|design your life)\b',
        r'\b(personal development|self-improvement|self-help)\b',
        r'\b(intentional living|intentional community)\b.*\b(tips|guide|how to)\b',

        # Individualistic framing
        r'\b(your ikigai|find your ikigai|discover your ikigai)\b',
        r'\b(your purpose|find your purpose|discover your purpose)\b',
    ]

    # === CORPORATE PATTERNS ===

    CORPORATE_PATTERNS = [
        # Workplace belonging
        r'\b(company culture|corporate culture|workplace culture)\b',
        r'\b(employee engagement|employee experience|employee belonging)\b',
        r'\b(team building|team bonding|offsite)\b',
        r'\b(psychological safety|workplace community)\b',

        # HR/management
        r'\b(hr strategy|people strategy|talent management)\b',
        r'\b(retention strategy|engagement strategy)\b',
        r'\b(dei initiative|diversity initiative|inclusion program)\b',
    ]

    # === ONLINE-ONLY PATTERNS ===

    ONLINE_ONLY_PATTERNS = [
        # Virtual communities
        r'\b(discord server|discord community|slack community)\b',
        r'\b(online community|virtual community|digital community)\b',
        r'\b(facebook group|subreddit|reddit community)\b',
        r'\b(online tribe|virtual tribe|digital tribe)\b',

        # Remote/digital lifestyle
        r'\b(digital nomad community|remote work community)\b',
        r'\b(online membership|virtual membership)\b',
    ]

    # === MULTILINGUAL PATTERNS (Dutch, German, French) ===

    MULTILINGUAL_BLOCK_PATTERNS = [
        # Dutch wellness/self-help
        r'\b(levensgeluk tips|zelfhulp|persoonlijke ontwikkeling)\b',
        r'\b(netwerken voor je carrière|zakelijk netwerk)\b',

        # German wellness/self-help
        r'\b(langlebigkeit|selbsthilfe|persönlichkeitsentwicklung)\b',
        r'\b(netzwerken|karriere netzwerk|berufliches netzwerk)\b',

        # French wellness/self-help
        r'\b(longévité secrets|développement personnel|aide personnelle)\b',
        r'\b(réseautage professionnel|réseau professionnel)\b',
    ]

    MULTILINGUAL_POSITIVE_PATTERNS = [
        # Dutch belonging
        r'\b(gemeenschap|buurt|dorpsleven|buren)\b',
        r'\b(grootouders|oma|opa|generaties)\b',
        r'\b(thuisdorp|geboortedorp|familieboerderij)\b',

        # German belonging
        r'\b(gemeinschaft|nachbarschaft|dorfgemeinschaft|heimat)\b',
        r'\b(großeltern|oma|opa|generationen)\b',
        r'\b(heimatdorf|familienhof|verwurzelt)\b',

        # French belonging
        r'\b(communauté|voisinage|vie de village|quartier)\b',
        r'\b(grands-parents|mamie|papi|générations)\b',
        r'\b(village natal|ferme familiale|enracinement)\b',
    ]

    # === EXCEPTION PATTERNS (allow through despite triggers) ===

    EXCEPTION_PATTERNS = [
        # Genuine community organizing
        r'\b(mutual aid|mutual-aid)\b',
        r'\b(community organizing|grassroots)\b',
        r'\b(worker cooperative|worker-owned)\b',
        r'\b(village council|town meeting|parish)\b',

        # Intergenerational
        r'\b(grandparent|grandmother|grandfather|elder)\b',
        r'\b(generation|generational|multigenerational)\b',
        r'\b(traditional knowledge|ancestral|heritage)\b',

        # Place-based
        r'\b(hometown|home village|native village)\b',
        r'\b(family farm|homestead|ancestral home)\b',
        r'\b(decades in|years in the same|generations of)\b',
    ]

    # === POSITIVE SIGNAL PATTERNS (boost pass likelihood) ===

    POSITIVE_PATTERNS = [
        # Community bonds
        r'\b(neighbor|neighbours|neighborly|neighbourhood)\b',
        r'\b(piazza|town square|village square|community center)\b',
        r'\b(church community|parish|congregation|mosque|temple|synagogue)\b',
        r'\b(mutual aid|helping each other|look after each other)\b',

        # Intergenerational
        r'\b(grandchildren|grandparents|great-grandmother|great-grandfather)\b',
        r'\b(passed down|handed down|taught me|learned from)\b',
        r'\b(family recipe|traditional craft|elder wisdom)\b',

        # Rootedness
        r'\b(born and raised|grew up here|never left)\b',
        r'\b(same house|same street|same village)\b',
        r'\b(family land|ancestral|for generations)\b',

        # Slow presence
        r'\b(sunday dinner|family meal|shared meal|gathering)\b',
        r'\b(ritual|tradition|every week|every day)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        # Block patterns (English)
        self.wellness_regex = [re.compile(p, re.IGNORECASE) for p in self.WELLNESS_PATTERNS]
        self.networking_regex = [re.compile(p, re.IGNORECASE) for p in self.NETWORKING_PATTERNS]
        self.tourism_regex = [re.compile(p, re.IGNORECASE) for p in self.TOURISM_PATTERNS]
        self.self_help_regex = [re.compile(p, re.IGNORECASE) for p in self.SELF_HELP_PATTERNS]
        self.corporate_regex = [re.compile(p, re.IGNORECASE) for p in self.CORPORATE_PATTERNS]
        self.online_only_regex = [re.compile(p, re.IGNORECASE) for p in self.ONLINE_ONLY_PATTERNS]

        # Multilingual block patterns
        self.multilingual_block_regex = [re.compile(p, re.IGNORECASE) for p in self.MULTILINGUAL_BLOCK_PATTERNS]

        # Exception and positive patterns
        self.exception_regex = [re.compile(p, re.IGNORECASE) for p in self.EXCEPTION_PATTERNS]
        self.positive_regex = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]
        self.multilingual_positive_regex = [re.compile(p, re.IGNORECASE) for p in self.MULTILINGUAL_POSITIVE_PATTERNS]

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Args:
            article: Dict with 'title' and 'text'/'content' keys

        Returns:
            (should_label, reason)
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

        # Count positive signals (English + multilingual)
        positive_count = self._count_positive_signals(combined_text)
        positive_count += self._count_multilingual_positive_signals(combined_text)

        # Check for exceptions that override blocks
        has_exception = self.has_any_pattern(combined_text, self.exception_regex)

        # Check block patterns (in order of priority)
        # Block if pattern matches AND no exception AND insufficient positive signals

        # Wellness industry
        if self._has_wellness_patterns(combined_text):
            if not has_exception and positive_count < 3:
                return False, "wellness_industry"

        # Professional networking
        if self._has_networking_patterns(combined_text):
            if not has_exception and positive_count < 3:
                return False, "networking_professional"

        # Tourism
        if self._has_tourism_patterns(combined_text):
            if not has_exception and positive_count < 3:
                return False, "tourism_consumption"

        # Self-help (slightly lower threshold - more likely to have borderline content)
        if self._has_self_help_patterns(combined_text):
            if not has_exception and positive_count < 2:
                return False, "self_help"

        # Corporate
        if self._has_corporate_patterns(combined_text):
            if not has_exception and positive_count < 3:
                return False, "corporate_belonging"

        # Online-only (block unless exception or strong positive signals)
        if self._has_online_only_patterns(combined_text):
            if not has_exception and positive_count < 2:
                return False, "online_only"

        # Multilingual block patterns
        if self.has_any_pattern(combined_text, self.multilingual_block_regex):
            if not has_exception and positive_count < 2:
                return False, "multilingual_block"

        # Passed all filters
        return True, "passed"

    def _check_domain_exclusions(self, url: str) -> str:
        """Check if URL is from an excluded domain, return reason or empty string"""
        url_lower = url.lower()

        for domain in self.WELLNESS_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_wellness"

        for domain in self.PROFESSIONAL_NETWORKING_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_networking"

        for domain in self.TRAVEL_TOURISM_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_tourism"

        for domain in self.SELF_HELP_DOMAINS:
            if domain in url_lower:
                return "excluded_domain_self_help"

        return ""

    def _has_wellness_patterns(self, text: str) -> bool:
        """Check if text contains wellness industry patterns"""
        return self.has_any_pattern(text, self.wellness_regex)

    def _has_networking_patterns(self, text: str) -> bool:
        """Check if text contains professional networking patterns"""
        return self.has_any_pattern(text, self.networking_regex)

    def _has_tourism_patterns(self, text: str) -> bool:
        """Check if text contains tourism patterns"""
        return self.has_any_pattern(text, self.tourism_regex)

    def _has_self_help_patterns(self, text: str) -> bool:
        """Check if text contains self-help patterns"""
        return self.has_any_pattern(text, self.self_help_regex)

    def _has_corporate_patterns(self, text: str) -> bool:
        """Check if text contains corporate belonging patterns"""
        return self.has_any_pattern(text, self.corporate_regex)

    def _has_online_only_patterns(self, text: str) -> bool:
        """Check if text contains online-only community patterns"""
        return self.has_any_pattern(text, self.online_only_regex)

    def _count_positive_signals(self, text: str) -> int:
        """Count how many positive belonging signals are present (English)"""
        return self.count_pattern_matches(text, self.positive_regex)

    def _count_multilingual_positive_signals(self, text: str) -> int:
        """Count multilingual positive belonging signals (Dutch, German, French)"""
        return self.count_pattern_matches(text, self.multilingual_positive_regex)

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        return {
            'version': self.VERSION,
            'wellness_domains': len(self.WELLNESS_DOMAINS),
            'networking_domains': len(self.PROFESSIONAL_NETWORKING_DOMAINS),
            'tourism_domains': len(self.TRAVEL_TOURISM_DOMAINS),
            'self_help_domains': len(self.SELF_HELP_DOMAINS),
            'wellness_patterns': len(self.WELLNESS_PATTERNS),
            'networking_patterns': len(self.NETWORKING_PATTERNS),
            'tourism_patterns': len(self.TOURISM_PATTERNS),
            'self_help_patterns': len(self.SELF_HELP_PATTERNS),
            'corporate_patterns': len(self.CORPORATE_PATTERNS),
            'online_only_patterns': len(self.ONLINE_ONLY_PATTERNS),
            'multilingual_block_patterns': len(self.MULTILINGUAL_BLOCK_PATTERNS),
            'exception_patterns': len(self.EXCEPTION_PATTERNS),
            'positive_patterns': len(self.POSITIVE_PATTERNS),
            'multilingual_positive_patterns': len(self.MULTILINGUAL_POSITIVE_PATTERNS),
        }


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = BelongingPreFilterV1()

    test_cases = [
        # Should BLOCK - Content too short
        {
            'title': 'Short Article',
            'text': 'This is too short.',
            'expected': (False, 'content_too_short')
        },

        # Should BLOCK - Wellness industry
        {
            'title': '5 Longevity Hacks from Blue Zone Centenarians',
            'text': 'Researchers have discovered the longevity secrets of the world\'s longest-lived communities. '
                    'These biohacking tips can help you optimize your lifespan and extend your healthspan significantly. '
                    'Supplements and intermittent fasting are key strategies used by centenarians in Okinawa. '
                    'Learn how to use anti-aging techniques and proven longevity hacks to reach 100. '
                    'The Blue Zone diet research shows remarkable results for those who follow these protocols.',
            'expected': (False, 'wellness_industry')
        },

        # Should BLOCK - Professional networking
        {
            'title': 'How to Build Your Professional Network for Career Success',
            'text': 'Building your network is essential for career growth in today\'s competitive marketplace. '
                    'LinkedIn connections and networking events can help you develop valuable social capital. '
                    'Join a mastermind group or find an accountability partner to accelerate your professional development. '
                    'The startup ecosystem thrives on these connections between founders and investors. '
                    'Strategic networking is the key to unlocking new opportunities and advancing your career.',
            'expected': (False, 'networking_professional')
        },

        # Should BLOCK - Tourism
        {
            'title': 'Visit Okinawa: A Travel Guide to Blue Zone Living',
            'text': 'Planning to visit Okinawa? This comprehensive travel guide covers the best time to visit, '
                    'where to stay, and must-visit destinations in this beautiful island paradise. '
                    'Experience cultural tourism at its finest in this Blue Zone destination known for longevity. '
                    'Our detailed itinerary includes wellness retreats, authentic local experiences, and travel tips '
                    'for making the most of your bucket list destination trip to Japan\'s southern islands.',
            'expected': (False, 'tourism_consumption')
        },

        # Should BLOCK - Self-help
        {
            'title': 'How to Find Your Tribe and Combat Loneliness',
            'text': 'Finding your tribe is essential for personal development and living your best life. '
                    'Learn how to build community and overcome loneliness with these practical tips and strategies. '
                    'Life design experts recommend intentional living and discovering your ikigai for fulfillment. '
                    'Here\'s how to make friends, create your tribe, and build meaningful connections in modern life. '
                    'Personal development starts with surrounding yourself with the right people.',
            'expected': (False, 'self_help')
        },

        # Should BLOCK - Corporate
        {
            'title': 'Building Workplace Belonging: HR Strategy Guide',
            'text': 'Company culture is key to employee engagement and organizational success in modern business. '
                    'This comprehensive guide covers workplace belonging initiatives, team building strategies, '
                    'and psychological safety best practices for creating an inclusive environment. '
                    'Learn how to improve your retention strategy through better employee experience programs. '
                    'DEI initiatives and engagement strategies are essential for building a thriving workplace.',
            'expected': (False, 'corporate_belonging')
        },

        # Should PASS - Genuine intergenerational community
        {
            'title': 'In This Village, Four Generations Share Sunday Dinner',
            'text': 'Every Sunday, Maria\'s family gathers at her grandmother\'s house for the weekly meal. '
                    'Her great-grandmother, now 94, still makes the family recipe pasta she learned from her mother. '
                    'The tradition has passed down through generations. Neighbors often join, as they have for decades. '
                    'The village has remained largely unchanged, with families who have lived here for generations '
                    'still gathering in the town square each evening.',
            'expected': (True, 'passed')
        },

        # Should PASS - Mutual aid
        {
            'title': 'After the Flood, Neighbors Became Family',
            'text': 'When the flood hit, neighbors who had barely spoken began checking on each other daily. '
                    'Rosa organized a mutual aid network for elderly residents. Three months later, the group still '
                    'meets every week for shared meals. The parish church became a gathering point. Grandparents '
                    'who had lived in the area for decades led the organizing efforts.',
            'expected': (True, 'passed')
        },

        # Should PASS - Rooted family
        {
            'title': 'Five Generations on the Same Farm',
            'text': 'The Johansen family has worked this land for five generations. Great-grandmother Elsa, 96, '
                    'still lives in the ancestral home where she was born. Every harvest, the extended family '
                    'gathers to help, a tradition handed down since her grandfather\'s time. The neighboring families '
                    'have known each other for over a century, helping each other through hardship.',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Online only community (may also match self_help due to "build community")
        {
            'title': 'Join Our Thriving Discord Server Community',
            'text': 'Our online community has grown to 5,000 members in the Discord server over the past year. '
                    'We host virtual events every week and have a Slack community for discussions with members. '
                    'Join our Facebook group for daily engagement with like-minded individuals from around the world. '
                    'Our digital community is thriving with members connecting virtually through video calls. '
                    'The virtual community offers courses, webinars, and networking opportunities for all members.',
            'expected': (False, 'online_only')
        },
    ]

    print("Testing Belonging Pre-Filter v1.0")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        result = prefilter.apply_filter(test)
        expected = test['expected']

        # Check if result matches expected (handle partial match for reasons)
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
        print(f"Title: {test['title'][:50]}...")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("\nPre-filter Statistics:")
    for key, value in prefilter.get_statistics().items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_prefilter()
