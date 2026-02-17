"""
Cultural Discovery Pre-Filter v1.0

Blocks obvious low-value content before LLM labeling:
- Political conflict framing (unless reconciliation/peace/dialogue)
- Tourism fluff (unless UNESCO/preservation/archaeological)
- Celebrity art news (unless philanthropy/repatriation/public donation)
- Cultural appropriation debates (unless respectful exchange/collaboration)
- Code repositories, VC/startup, and defense domains

Purpose: Reduce LLM costs and improve training data quality for cultural discovery filter.
Expected pass rate: ~15% of random articles
"""

import re
from typing import Dict, List, Tuple

from filters.common.base_prefilter import BasePreFilter


class CulturalDiscoveryPreFilterV1(BasePreFilter):
    """Fast rule-based pre-filter for cultural discovery content v1"""

    VERSION = "1.0"

    # === DOMAIN EXCLUSIONS ===

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

    # === POLITICAL CONFLICT INDICATORS ===

    POLITICAL_CONFLICT_PATTERNS = [
        # Culture war framing
        r'\b(culture war|kulturkampf|cultural battle|cultural conflict)\b',
        r'\b(cancel culture|cancellation|cancelled|canceled)\b',
        r'\b(woke|anti-woke|wokeism|wokeness)\b',
        r'\b(identity politics|political correctness|pc culture)\b',

        # Us vs them framing
        r'\b(us vs them|us versus them|our culture vs)\b',
        r'\b(cultural invasion|cultural erasure|cultural genocide)\b',
        r'\b(threat to our (culture|heritage|tradition|identity))\b',

        # Polarizing debate language
        r'\b(outrage|backlash|controversy|slammed|blasted)\b',
        r'\b(critics say|critics argue|sparked debate)\b',

        # Nationalist framing
        r'\b(defend our (culture|heritage|traditions))\b',
        r'\b(foreign influence|cultural imperialism)\b',

        # Dutch patterns
        r'\b(cultuurstrijd|woke-agenda|identiteitspolitiek)\b',
        r'\b(onze cultuur verdedigen|buitenlandse invloed)\b',

        # German patterns
        r'\b(kulturkampf|identitätspolitik|cancel culture)\b',
        r'\b(unsere kultur verteidigen|kulturelle überfremdung)\b',

        # French patterns
        r'\b(guerre culturelle|identitarisme|cancel culture)\b',
        r'\b(défendre notre culture|influence étrangère)\b',
    ]

    POLITICAL_CONFLICT_EXCEPTIONS = [
        # Reconciliation and peace
        r'\b(reconciliation|peace process|dialogue|healing)\b',
        r'\b(bridge-building|understanding|coming together)\b',
        r'\b(peacemaking|mediation|resolution)\b',
        r'\b(truth and reconciliation|transitional justice)\b',

        # Historical analysis (not advocacy)
        r'\b(historical analysis|historical perspective|historians note)\b',
        r'\b(academic study|scholarly research|research shows)\b',

        # Dutch patterns
        r'\b(verzoening|vredesproces|dialoog|heling)\b',
        r'\b(bruggen bouwen|begrip kweken)\b',

        # German patterns
        r'\b(versöhnung|friedensprozess|dialog|heilung)\b',
        r'\b(brücken bauen|verständigung)\b',

        # French patterns
        r'\b(réconciliation|processus de paix|dialogue|guérison)\b',
        r'\b(construire des ponts|compréhension mutuelle)\b',
    ]

    # === TOURISM FLUFF INDICATORS ===

    TOURISM_FLUFF_PATTERNS = [
        # Listicle patterns
        r'\b(top \d+|best \d+|\d+ must-see|\d+ things to)\b',
        r'\b(bucket list|must-visit|must-see|don\'t miss)\b',
        r'\b(hidden gems|off the beaten path|insider tips)\b',

        # Marketing language
        r'\b(breathtaking|stunning views|instagram-worthy|picture-perfect)\b',
        r'\b(unforgettable experience|trip of a lifetime|magical destination)\b',
        r'\b(book now|reserve today|limited availability)\b',

        # Travel planning
        r'\b(travel guide|visitor guide|tourist guide)\b',
        r'\b(where to stay|best hotels|accommodation|restaurants near)\b',
        r'\b(how to get there|getting around|transportation)\b',

        # Dutch patterns
        r'\b(top \d+ bezienswaardigheden|must-see attracties)\b',
        r'\b(verborgen pareltjes|bucketlist|reistips)\b',

        # German patterns
        r'\b(top \d+ sehenswürdigkeiten|must-see attraktionen)\b',
        r'\b(geheimtipps|reisetipps|urlaubstipps)\b',

        # French patterns
        r'\b(top \d+ (à voir|incontournables)|lieux à visiter)\b',
        r'\b(conseils de voyage|guide touristique)\b',
    ]

    TOURISM_FLUFF_EXCEPTIONS = [
        # UNESCO and preservation
        r'\b(unesco|world heritage|patrimoine mondial|welterbe)\b',
        r'\b(preservation effort|conservation project|restoration work|protected site)\b',
        r'\b(endangered heritage|at risk|under threat)\b',

        # Archaeological significance (require more specific context)
        r'\b(archaeological (site|find|dig|excavation|discovery))\b',
        r'\b(excavation|artifact found|artefact discovered)\b',
        r'\b(unearthed|discovered by (researchers|archaeologists))\b',

        # Expert/institutional sources
        r'\b(museum curator|university research|research institute|foundation grant)\b',
        r'\b(archaeologist|historian says|researcher found|expert analysis)\b',

        # Dutch patterns
        r'\b(beschermd erfgoed|restauratie|behoud)\b',
        r'\b(archeologisch|opgravingen|ontdekking)\b',

        # German patterns
        r'\b(denkmalschutz|restaurierung|erhaltung)\b',
        r'\b(archäologisch|ausgrabungen|entdeckung)\b',

        # French patterns
        r'\b(patrimoine protégé|restauration|conservation)\b',
        r'\b(archéologique|fouilles|découverte)\b',
    ]

    # === CELEBRITY ART INDICATORS ===

    CELEBRITY_ART_PATTERNS = [
        # Auction/market language
        r'\b(auction|sold for|fetched|hammer price)\b',
        r'\b(sotheby|christie|bonham|phillips)\b',
        r'\b(art market|art sale|collection sale)\b',
        r'\$[\d,]+\s*(million|billion|m|b)\b',

        # Celebrity collection focus
        r'\b(celebrity collection|private collection|personal art)\b',
        r'\b(billionaire|millionaire|wealthy collector)\b',
        r'\b(art investment|art portfolio|appreciating asset)\b',

        # Market speculation
        r'\b(estimated value|appraised at|worth millions)\b',
        r'\b(record price|record sale|highest ever)\b',

        # Dutch patterns
        r'\b(veiling|geveild voor|hamerprijs)\b',
        r'\b(kunstmarkt|kunstverzameling|miljoenair)\b',

        # German patterns
        r'\b(auktion|versteigert für|hammerpreis)\b',
        r'\b(kunstmarkt|kunstsammlung|milliardär)\b',

        # French patterns
        r'\b(vente aux enchères|adjugé|prix marteau)\b',
        r'\b(marché de l\'art|collection d\'art|milliardaire)\b',
    ]

    CELEBRITY_ART_EXCEPTIONS = [
        # Philanthropy
        r'\b(philanthropy|donated|donation|gift to museum)\b',
        r'\b(public access|public benefit|educational purpose)\b',
        r'\b(charitable|foundation|endowment)\b',

        # Repatriation
        r'\b(repatriation|returned|restitution|provenance)\b',
        r'\b(colonial|looted|stolen|illicit)\b',
        r'\b(origin country|rightful owner|cultural property)\b',

        # Public donation
        r'\b(public donation|museum acquisition|gift to nation)\b',
        r'\b(permanent collection|public display|open access)\b',

        # Dutch patterns
        r'\b(filantropie|schenking|donatie|museumgift)\b',
        r'\b(repatriëring|teruggave|herkomst)\b',

        # German patterns
        r'\b(philanthropie|spende|schenkung|museumsgeschenk)\b',
        r'\b(repatriierung|rückgabe|provenienz)\b',

        # French patterns
        r'\b(philanthropie|don|donation|legs)\b',
        r'\b(rapatriement|restitution|provenance)\b',
    ]

    # === APPROPRIATION DEBATE INDICATORS ===

    APPROPRIATION_DEBATE_PATTERNS = [
        # Core appropriation language (most specific)
        r'\b(cultural appropriation|appropriating culture|appropriated from)\b',
        r'\b(accused of appropriat|called out for appropriat)\b',
        r'\b(stealing (culture|tradition|heritage))\b',

        # Identity gatekeeping (specific to appropriation)
        r'\b(not your culture|stay in your lane|not for you to)\b',
        r'\b(who gets to (wear|use|perform|practice))\b',

        # Dutch patterns (specific)
        r'\b(culturele toe-eigening|cultuur stelen)\b',
        r'\b(niet jouw cultuur om te)\b',

        # German patterns (specific)
        r'\b(kulturelle aneignung|kultur stehlen)\b',
        r'\b(nicht deine kultur zu)\b',

        # French patterns (specific)
        r'\b(appropriation culturelle|voler la culture)\b',
        r'\b(pas ta culture à)\b',
    ]

    APPROPRIATION_DEBATE_EXCEPTIONS = [
        # Respectful exchange
        r'\b(respectful exchange|cultural exchange|collaboration)\b',
        r'\b(working together|partnership|co-creation)\b',
        r'\b(invitation|invited|welcomed)\b',

        # Acknowledgment and credit
        r'\b(acknowledgment|credit|attribution|honor)\b',
        r'\b(with permission|with blessing|with approval|given permission)\b',
        r'\b(with consent|in partnership|alongside)\b',

        # Educational context
        r'\b(learning|education|understanding|appreciation)\b',
        r'\b(history of|origins of|tradition of)\b',

        # Dutch patterns
        r'\b(respectvolle uitwisseling|culturele samenwerking)\b',
        r'\b(met toestemming|in samenwerking)\b',

        # German patterns
        r'\b(respektvoller austausch|kulturelle zusammenarbeit)\b',
        r'\b(mit erlaubnis|in partnerschaft)\b',

        # French patterns
        r'\b(échange respectueux|collaboration culturelle)\b',
        r'\b(avec permission|en partenariat)\b',
    ]

    # === POSITIVE CULTURAL DISCOVERY INDICATORS ===
    # Used to boost articles that clearly have cultural discovery value

    CULTURAL_DISCOVERY_BOOST_PATTERNS = [
        # Archaeological discoveries
        r'\b(discovered|unearthed|excavated|found)\b',
        r'\b(archaeological find|ancient site|artifact)\b',
        r'\b(breakthrough|revelation|new evidence)\b',

        # Cross-cultural connections
        r'\b(cross-cultural|intercultural|multicultural)\b',
        r'\b(cultural bridge|cultural exchange|cultural dialogue)\b',
        r'\b(shared heritage|common ancestry|connected civilizations)\b',

        # Heritage preservation
        r'\b(unesco|world heritage|cultural heritage)\b',
        r'\b(preservation|restoration|conservation)\b',
        r'\b(repatriation|returned to|cultural property)\b',

        # Research and expertise
        r'\b(archaeologist|historian|researcher|anthropologist)\b',
        r'\b(study (reveals|shows|finds)|research (shows|indicates))\b',
        r'\b(peer-reviewed|published in|academic journal)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        super().__init__()

        # Political conflict
        self.political_conflict_regex = [
            re.compile(p, re.IGNORECASE) for p in self.POLITICAL_CONFLICT_PATTERNS
        ]
        self.political_conflict_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.POLITICAL_CONFLICT_EXCEPTIONS
        ]

        # Tourism fluff
        self.tourism_fluff_regex = [
            re.compile(p, re.IGNORECASE) for p in self.TOURISM_FLUFF_PATTERNS
        ]
        self.tourism_fluff_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.TOURISM_FLUFF_EXCEPTIONS
        ]

        # Celebrity art
        self.celebrity_art_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CELEBRITY_ART_PATTERNS
        ]
        self.celebrity_art_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CELEBRITY_ART_EXCEPTIONS
        ]

        # Appropriation debate
        self.appropriation_debate_regex = [
            re.compile(p, re.IGNORECASE) for p in self.APPROPRIATION_DEBATE_PATTERNS
        ]
        self.appropriation_debate_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.APPROPRIATION_DEBATE_EXCEPTIONS
        ]

        # Cultural discovery boost (for classification)
        self.cultural_discovery_boost_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CULTURAL_DISCOVERY_BOOST_PATTERNS
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

        # Check appropriation debate FIRST (more specific than political conflict)
        if self._has_pattern(combined_text, self.appropriation_debate_regex):
            if not self._has_pattern(combined_text, self.appropriation_debate_exceptions_regex):
                return False, "appropriation_debate"

        # Check political conflict
        if self._has_pattern(combined_text, self.political_conflict_regex):
            if not self._has_pattern(combined_text, self.political_conflict_exceptions_regex):
                return False, "political_conflict"

        # Check tourism fluff
        if self._has_pattern(combined_text, self.tourism_fluff_regex):
            if not self._has_pattern(combined_text, self.tourism_fluff_exceptions_regex):
                return False, "tourism_fluff"

        # Check celebrity art
        if self._has_pattern(combined_text, self.celebrity_art_regex):
            if not self._has_pattern(combined_text, self.celebrity_art_exceptions_regex):
                return False, "celebrity_art"

        # Passed all filters
        return True, "passed"

    def _check_domain_exclusions(self, url: str) -> str:
        """Check if URL belongs to an excluded domain. Returns reason or empty string."""
        url_lower = url.lower()

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
            'vc_startup_domains': len(self.VC_STARTUP_DOMAINS),
            'defense_domains': len(self.DEFENSE_DOMAINS),
            'code_hosting_domains': len(self.CODE_HOSTING_DOMAINS),
            'political_conflict_patterns': len(self.POLITICAL_CONFLICT_PATTERNS),
            'political_conflict_exceptions': len(self.POLITICAL_CONFLICT_EXCEPTIONS),
            'tourism_fluff_patterns': len(self.TOURISM_FLUFF_PATTERNS),
            'tourism_fluff_exceptions': len(self.TOURISM_FLUFF_EXCEPTIONS),
            'celebrity_art_patterns': len(self.CELEBRITY_ART_PATTERNS),
            'celebrity_art_exceptions': len(self.CELEBRITY_ART_EXCEPTIONS),
            'appropriation_debate_patterns': len(self.APPROPRIATION_DEBATE_PATTERNS),
            'appropriation_debate_exceptions': len(self.APPROPRIATION_DEBATE_EXCEPTIONS),
        }

    def classify_content_type(self, article: Dict) -> str:
        """
        Classify article content type (for oracle pre-classification).

        Returns one of:
        - "cultural_discovery" (has positive indicators)
        - "political_conflict"
        - "tourism_fluff"
        - "celebrity_art"
        - "appropriation_debate"
        - "general"
        """
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Check for cultural discovery boost first
        discovery_matches = self._count_matches(combined_text, self.cultural_discovery_boost_regex)
        if discovery_matches >= 2:
            return "cultural_discovery"

        # Check appropriation debate FIRST (more specific than political conflict)
        if self._has_pattern(combined_text, self.appropriation_debate_regex):
            if not self._has_pattern(combined_text, self.appropriation_debate_exceptions_regex):
                return "appropriation_debate"

        # Check political conflict
        if self._has_pattern(combined_text, self.political_conflict_regex):
            if not self._has_pattern(combined_text, self.political_conflict_exceptions_regex):
                return "political_conflict"

        # Check tourism fluff
        if self._has_pattern(combined_text, self.tourism_fluff_regex):
            if not self._has_pattern(combined_text, self.tourism_fluff_exceptions_regex):
                return "tourism_fluff"

        # Check celebrity art
        if self._has_pattern(combined_text, self.celebrity_art_regex):
            if not self._has_pattern(combined_text, self.celebrity_art_exceptions_regex):
                return "celebrity_art"

        return "general"


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = CulturalDiscoveryPreFilterV1()

    # Note: MIN_CONTENT_LENGTH is 300 chars, so test content must be longer

    test_cases = [
        # Should BLOCK - Content too short
        {
            'title': 'Short Article',
            'text': 'This is too short to pass the minimum content length filter.',
            'expected': (False, 'content_too_short'),
            'description': 'Content too short'
        },

        # Should BLOCK - Political Conflict
        {
            'title': 'The Culture War Over Museum Collections',
            'text': 'Critics slammed the museum\'s decision in a viral backlash on social media. The controversy has sparked debate about woke culture and identity politics in arts institutions. Cancel culture is threatening our cultural heritage, according to opponents. The outrage continued to build as more critics weighed in on the politically correct decision. This represents a cultural battle that will define our times.',
            'expected': (False, 'political_conflict'),
            'description': 'Culture war framing'
        },

        # Should PASS - Reconciliation (exception to political conflict)
        {
            'title': 'Truth and Reconciliation Through Art',
            'text': 'The museum\'s new exhibition brings together artists from communities that were once in conflict. Through dialogue and bridge-building, the collaborative project explores shared heritage and promotes understanding. The reconciliation process has been supported by historians and cultural researchers. The exhibition represents years of healing and coming together across cultural divides.',
            'expected': (True, 'passed'),
            'description': 'Reconciliation (exception)'
        },

        # Should BLOCK - Tourism Fluff
        {
            'title': 'Top 10 Must-See Ancient Temples in Asia',
            'text': 'Planning your bucket list trip? These hidden gems offer breathtaking views and Instagram-worthy photo opportunities. Don\'t miss these stunning must-visit destinations! Our travel guide covers where to stay, best hotels, and insider tips for getting around. Book now for the trip of a lifetime to these picture-perfect ancient sites.',
            'expected': (False, 'tourism_fluff'),
            'description': 'Tourism listicle'
        },

        # Should PASS - UNESCO Heritage (exception to tourism)
        {
            'title': 'UNESCO Adds New Sites to World Heritage List',
            'text': 'The World Heritage Committee has recognized several endangered heritage sites for preservation. Conservation efforts are underway to protect these archaeological treasures. Researchers and archaeologists have documented the historical significance of these ancient sites. The museum and university teams are working on restoration projects to preserve these protected sites for future generations.',
            'expected': (True, 'passed'),
            'description': 'UNESCO heritage (exception)'
        },

        # Should BLOCK - Celebrity Art
        {
            'title': 'Billionaire Collector\'s Painting Sells for $50 Million',
            'text': 'The private collection went under the hammer at Christie\'s auction house yesterday. The art market record was smashed as wealthy collectors bid on the appreciating asset. Sotheby\'s experts had estimated the value at $40 million before the sale. The billionaire\'s art investment portfolio continues to grow with this latest acquisition.',
            'expected': (False, 'celebrity_art'),
            'description': 'Celebrity art auction'
        },

        # Should PASS - Repatriation (exception to celebrity art)
        {
            'title': 'Museum Returns Looted Artifacts to Origin Country',
            'text': 'After years of provenance research, the museum has agreed to repatriation of colonial-era artifacts. The cultural property will be returned to its rightful owners following revelations about its illicit acquisition. This donation to the nation represents a shift in how museums approach stolen heritage. The public benefit of returning these items has been praised by cultural property experts.',
            'expected': (True, 'passed'),
            'description': 'Repatriation (exception)'
        },

        # Should BLOCK - Appropriation Debate
        {
            'title': 'Designer Accused of Cultural Appropriation',
            'text': 'The designer was called out on Twitter for appropriating traditional patterns without permission. Critics say this is problematic and offensive to the culture being appropriated. The viral backlash led to accusations and social media outrage. "Not your culture, stay in your lane" was the trending hashtag as the controversy grew.',
            'expected': (False, 'appropriation_debate'),
            'description': 'Appropriation debate'
        },

        # Should PASS - Respectful Collaboration (exception)
        {
            'title': 'Cross-Cultural Fashion Collaboration Launches',
            'text': 'The respectful exchange between designers celebrates cultural heritage through collaboration. Working together in partnership, the co-creation project was developed with invitation from community elders. The collaboration includes acknowledgment and credit to traditional artisans. Educational materials explain the history of the tradition and honor its origins.',
            'expected': (True, 'passed'),
            'description': 'Respectful collaboration (exception)'
        },

        # Should PASS - Archaeological Discovery
        {
            'title': 'Ancient Maya Temple Discovered with Unique Murals',
            'text': 'Archaeologists have unearthed a 3,000-year-old temple containing murals depicting cross-cultural connections. The breakthrough discovery was published in a peer-reviewed journal after excavation revealed new evidence of ancient trade networks. The university research team includes historians and anthropologists who study shared heritage across civilizations. This UNESCO world heritage site candidate shows cultural exchange between distant peoples.',
            'expected': (True, 'passed'),
            'description': 'Archaeological discovery'
        },

        # Should PASS - Cultural Heritage Restoration
        {
            'title': 'Restored Fresco Reveals Hidden Cultural Connections',
            'text': 'Conservation experts at the museum have completed a five-year restoration project. The preservation effort revealed previously unknown cultural heritage connections between Mediterranean civilizations. Researchers discovered evidence of cross-cultural artistic exchange hidden beneath later paint layers. The archaeological find was documented by university historians studying ancient intercultural trade routes.',
            'expected': (True, 'passed'),
            'description': 'Heritage restoration'
        },
    ]

    print("Testing Cultural Discovery Pre-Filter v1.0")
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
    for test in test_cases[1:]:  # Skip short content test
        content_type = prefilter.classify_content_type(test)
        print(f"  {test['description']}: {content_type}")


if __name__ == '__main__':
    test_prefilter()
