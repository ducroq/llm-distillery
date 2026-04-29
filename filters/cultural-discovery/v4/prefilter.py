"""
Cultural Discovery Pre-Filter v4.0

ADR-018 declarative shape (partial): exclusion patterns are declared as the
EXCLUSION_PATTERNS category dict and compiled by BasePreFilter.__init__.
Per-category exception lists live in a parallel EXCEPTION_PATTERNS_PER_CATEGORY
dict because each exclusion category has its own escape hatches (e.g.
political_conflict has reconciliation/peace exceptions; celebrity_art has
philanthropy/repatriation exceptions). The single OVERRIDE_KEYWORDS slot on
BasePreFilter doesn't fit; apply_filter stays custom.

Blocks obvious low-value content before LLM labeling:
- Political conflict framing (unless reconciliation/peace/dialogue)
- Tourism fluff (unless UNESCO/preservation/archaeological)
- Celebrity art news (unless philanthropy/repatriation/public donation)
- Cultural appropriation debates (unless respectful exchange/collaboration)
- Code repositories, VC/startup, and defense domains

Purpose: Reduce LLM costs and improve training data quality for cultural discovery filter.
Expected pass rate: ~15% of random articles

History:
- v4.0 (2026-04-29): migrated to declarative BasePreFilter shape (#52, ADR-018).
  No behavior change vs the prior v4.0 — pattern set, per-category exceptions,
  iteration order, and classify_content_type semantics preserved verbatim.
  Self-test (10/10) passes; pattern counts identical to baseline.
- v4.0 prior: per-category exception lists, four exclusion categories,
  classify_content_type method for oracle pre-classification.

Known regression vs v3 (NOT addressed in this migration — preserves current v4
behavior so the migration commit has zero behavior delta): v4's apply_filter
does not call check_content_length() before pattern matching, while v3 did.
Short articles bypass length validation here. Documented in TODO under
"Prefilter Quality" as a follow-up.
"""

import re
from typing import Dict, List, Tuple

from filters.common.base_prefilter import BasePreFilter


class CulturalDiscoveryPreFilterV4(BasePreFilter):
    """Fast rule-based pre-filter for cultural discovery content v4.

    v4.0 (declarative-shape exclusion data per ADR-018; custom apply_filter
    retained for per-category exceptions and URL-based domain blocking).
    """

    VERSION = "4.0"

    # === DOMAIN EXCLUSIONS (URL-based) ===

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

    # === ADR-018 EXCLUSION_PATTERNS ===
    # Iteration order matches the legacy apply_filter() order: appropriation
    # FIRST (more specific than political conflict), then political_conflict,
    # tourism_fluff, celebrity_art. Category keys match the (False, "<reason>")
    # tuples this filter emits — no "excluded_" prefix because callers match
    # these strings directly.
    EXCLUSION_PATTERNS = {
        # Core appropriation language + identity gatekeeping (specific). Most
        # specific category, checked first. Multilingual: NL/DE/FR variants.
        'appropriation_debate': [
            r'\b(cultural appropriation|appropriating culture|appropriated from)\b',
            r'\b(accused of appropriat|called out for appropriat)\b',
            r'\b(stealing (culture|tradition|heritage))\b',
            r'\b(not your culture|stay in your lane|not for you to)\b',
            r'\b(who gets to (wear|use|perform|practice))\b',
            r'\b(culturele toe-eigening|cultuur stelen)\b',
            r'\b(niet jouw cultuur om te)\b',
            r'\b(kulturelle aneignung|kultur stehlen)\b',
            r'\b(nicht deine kultur zu)\b',
            r'\b(appropriation culturelle|voler la culture)\b',
            r'\b(pas ta culture à)\b',
        ],
        # Culture-war framing, us-vs-them, polarizing-debate language,
        # nationalist framing. Multilingual: NL/DE/FR.
        'political_conflict': [
            r'\b(culture war|kulturkampf|cultural battle|cultural conflict)\b',
            r'\b(cancel culture|cancellation|cancelled|canceled)\b',
            r'\b(woke|anti-woke|wokeism|wokeness)\b',
            r'\b(identity politics|political correctness|pc culture)\b',
            r'\b(us vs them|us versus them|our culture vs)\b',
            r'\b(cultural invasion|cultural erasure|cultural genocide)\b',
            r'\b(threat to our (culture|heritage|tradition|identity))\b',
            r'\b(outrage|backlash|controversy|slammed|blasted)\b',
            r'\b(critics say|critics argue|sparked debate)\b',
            r'\b(defend our (culture|heritage|traditions))\b',
            r'\b(foreign influence|cultural imperialism)\b',
            r'\b(cultuurstrijd|woke-agenda|identiteitspolitiek)\b',
            r'\b(onze cultuur verdedigen|buitenlandse invloed)\b',
            r'\b(kulturkampf|identitätspolitik|cancel culture)\b',
            r'\b(unsere kultur verteidigen|kulturelle überfremdung)\b',
            r'\b(guerre culturelle|identitarisme|cancel culture)\b',
            r'\b(défendre notre culture|influence étrangère)\b',
        ],
        # Listicle framing, marketing language, travel-planning copy.
        # Multilingual: NL/DE/FR.
        'tourism_fluff': [
            r'\b(top \d+|best \d+|\d+ must-see|\d+ things to)\b',
            r'\b(bucket list|must-visit|must-see|don\'t miss)\b',
            r'\b(hidden gems|off the beaten path|insider tips)\b',
            r'\b(breathtaking|stunning views|instagram-worthy|picture-perfect)\b',
            r'\b(unforgettable experience|trip of a lifetime|magical destination)\b',
            r'\b(book now|reserve today|limited availability)\b',
            r'\b(travel guide|visitor guide|tourist guide)\b',
            r'\b(where to stay|best hotels|accommodation|restaurants near)\b',
            r'\b(how to get there|getting around|transportation)\b',
            r'\b(top \d+ bezienswaardigheden|must-see attracties)\b',
            r'\b(verborgen pareltjes|bucketlist|reistips)\b',
            r'\b(top \d+ sehenswürdigkeiten|must-see attraktionen)\b',
            r'\b(geheimtipps|reisetipps|urlaubstipps)\b',
            r'\b(top \d+ (à voir|incontournables)|lieux à visiter)\b',
            r'\b(conseils de voyage|guide touristique)\b',
        ],
        # Auction/market language, celebrity-collection focus, market
        # speculation. Multilingual: NL/DE/FR.
        'celebrity_art': [
            r'\b(auction|sold for|fetched|hammer price)\b',
            r'\b(sotheby|christie|bonham|phillips)\b',
            r'\b(art market|art sale|collection sale)\b',
            r'\$[\d,]+\s*(million|billion|m|b)\b',
            r'\b(celebrity collection|private collection|personal art)\b',
            r'\b(billionaire|millionaire|wealthy collector)\b',
            r'\b(art investment|art portfolio|appreciating asset)\b',
            r'\b(estimated value|appraised at|worth millions)\b',
            r'\b(record price|record sale|highest ever)\b',
            r'\b(veiling|geveild voor|hamerprijs)\b',
            r'\b(kunstmarkt|kunstverzameling|miljoenair)\b',
            r'\b(auktion|versteigert für|hammerpreis)\b',
            r'\b(kunstmarkt|kunstsammlung|milliardär)\b',
            r'\b(vente aux enchères|adjugé|prix marteau)\b',
            r'\b(marché de l\'art|collection d\'art|milliardaire)\b',
        ],
    }

    # Per-category exception lists. Each key mirrors an EXCLUSION_PATTERNS key:
    # if the category's exclusion fires AND any exception in the parallel list
    # also matches, the article is allowed through that category. This is
    # category-scoped, unlike belonging's global EXCEPTION_PATTERNS or sustech's
    # global OVERRIDE_KEYWORDS — neither base slot fits, so apply_filter stays
    # custom (per ADR-018).
    EXCEPTION_PATTERNS_PER_CATEGORY = {
        # Respectful exchange, acknowledgment/credit, educational context.
        # Multilingual: NL/DE/FR.
        'appropriation_debate': [
            r'\b(respectful exchange|cultural exchange|collaboration)\b',
            r'\b(working together|partnership|co-creation)\b',
            r'\b(invitation|invited|welcomed)\b',
            r'\b(acknowledgment|credit|attribution|honor)\b',
            r'\b(with permission|with blessing|with approval|given permission)\b',
            r'\b(with consent|in partnership|alongside)\b',
            r'\b(learning|education|understanding|appreciation)\b',
            r'\b(history of|origins of|tradition of)\b',
            r'\b(respectvolle uitwisseling|culturele samenwerking)\b',
            r'\b(met toestemming|in samenwerking)\b',
            r'\b(respektvoller austausch|kulturelle zusammenarbeit)\b',
            r'\b(mit erlaubnis|in partnerschaft)\b',
            r'\b(échange respectueux|collaboration culturelle)\b',
            r'\b(avec permission|en partenariat)\b',
        ],
        # Reconciliation/peace, historical analysis (not advocacy).
        # Multilingual: NL/DE/FR.
        'political_conflict': [
            r'\b(reconciliation|peace process|dialogue|healing)\b',
            r'\b(bridge-building|understanding|coming together)\b',
            r'\b(peacemaking|mediation|resolution)\b',
            r'\b(truth and reconciliation|transitional justice)\b',
            r'\b(historical analysis|historical perspective|historians note)\b',
            r'\b(academic study|scholarly research|research shows)\b',
            r'\b(verzoening|vredesproces|dialoog|heling)\b',
            r'\b(bruggen bouwen|begrip kweken)\b',
            r'\b(versöhnung|friedensprozess|dialog|heilung)\b',
            r'\b(brücken bauen|verständigung)\b',
            r'\b(réconciliation|processus de paix|dialogue|guérison)\b',
            r'\b(construire des ponts|compréhension mutuelle)\b',
        ],
        # UNESCO/preservation, archaeological significance, expert/institutional
        # sources. Multilingual: NL/DE/FR.
        'tourism_fluff': [
            r'\b(unesco|world heritage|patrimoine mondial|welterbe)\b',
            r'\b(preservation effort|conservation project|restoration work|protected site)\b',
            r'\b(endangered heritage|at risk|under threat)\b',
            r'\b(archaeological (site|find|dig|excavation|discovery))\b',
            r'\b(excavation|artifact found|artefact discovered)\b',
            r'\b(unearthed|discovered by (researchers|archaeologists))\b',
            r'\b(museum curator|university research|research institute|foundation grant)\b',
            r'\b(archaeologist|historian says|researcher found|expert analysis)\b',
            r'\b(beschermd erfgoed|restauratie|behoud)\b',
            r'\b(archeologisch|opgravingen|ontdekking)\b',
            r'\b(denkmalschutz|restaurierung|erhaltung)\b',
            r'\b(archäologisch|ausgrabungen|entdeckung)\b',
            r'\b(patrimoine protégé|restauration|conservation)\b',
            r'\b(archéologique|fouilles|découverte)\b',
        ],
        # Philanthropy, repatriation, public donation. Multilingual: NL/DE/FR.
        'celebrity_art': [
            r'\b(philanthropy|donated|donation|gift to museum)\b',
            r'\b(public access|public benefit|educational purpose)\b',
            r'\b(charitable|foundation|endowment)\b',
            r'\b(repatriation|returned|restitution|provenance)\b',
            r'\b(colonial|looted|stolen|illicit)\b',
            r'\b(origin country|rightful owner|cultural property)\b',
            r'\b(public donation|museum acquisition|gift to nation)\b',
            r'\b(permanent collection|public display|open access)\b',
            r'\b(filantropie|schenking|donatie|museumgift)\b',
            r'\b(repatriëring|teruggave|herkomst)\b',
            r'\b(philanthropie|spende|schenkung|museumsgeschenk)\b',
            r'\b(repatriierung|rückgabe|provenienz)\b',
            r'\b(philanthropie|don|donation|legs)\b',
            r'\b(rapatriement|restitution|provenance)\b',
        ],
    }

    # === POSITIVE CULTURAL DISCOVERY INDICATORS ===
    # Used by classify_content_type() to flag articles with strong discovery
    # value (count >= 2 -> "cultural_discovery").
    #
    # Note: this attribute name shadows BasePreFilter.POSITIVE_PATTERNS (the
    # ADR-018 single-bypass slot). CD consumes the count via classify_content_type
    # rather than the base POSITIVE_THRESHOLD bypass — and POSITIVE_THRESHOLD
    # stays at 0, so base's _has_override never reads it. Patterns get compiled
    # into self._compiled_positives by super().__init__(); CD reads them from
    # there directly.
    POSITIVE_PATTERNS = [
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
        """Compile per-category exceptions; base compiles EXCLUSION_PATTERNS
        and POSITIVE_PATTERNS into self._compiled_exclusions / _compiled_positives."""
        super().__init__()
        self._compiled_exceptions_per_category: Dict[str, List[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.EXCEPTION_PATTERNS_PER_CATEGORY.items()
        }

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should pass to oracle for scoring.

        Custom flow (not BasePreFilter.apply_filter): per-category exception
        lists don't fit the base's single OVERRIDE_KEYWORDS slot. ADR-018
        explicitly permits custom apply_filter() for filters whose control
        flow diverges from the standard pipeline.

        Args:
            article: Dict with 'title' and 'text' (or 'content') keys

        Returns:
            (passed, reason)
            - (True, "passed"): Send to oracle
            - (False, reason): Block with specific reason
        """
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

        # Iterate exclusions in declared order; first blocking category wins.
        # Each category has its own parallel exception list — match suppresses
        # the block for that category only.
        for category, compiled_patterns in self._compiled_exclusions.items():
            if not self.has_any_pattern(combined_text, compiled_patterns):
                continue
            exceptions = self._compiled_exceptions_per_category.get(category, [])
            if self.has_any_pattern(combined_text, exceptions):
                continue
            return False, category

        return True, "passed"

    def classify_content_type(self, article: Dict) -> str:
        """
        Classify article content type (for oracle pre-classification).

        Returns one of:
        - "cultural_discovery" (>=2 positive boost matches)
        - "appropriation_debate" / "political_conflict" / "tourism_fluff"
          / "celebrity_art" (exclusion category fired, no exception bypass)
        - "general"
        """
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Cultural discovery boost takes precedence over exclusion classification.
        if self.count_pattern_matches(combined_text, self._compiled_positives) >= 2:
            return "cultural_discovery"

        for category, compiled_patterns in self._compiled_exclusions.items():
            if not self.has_any_pattern(combined_text, compiled_patterns):
                continue
            exceptions = self._compiled_exceptions_per_category.get(category, [])
            if self.has_any_pattern(combined_text, exceptions):
                continue
            return category

        return "general"

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

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        stats = {
            'version': self.VERSION,
            'vc_startup_domains': len(self.VC_STARTUP_DOMAINS),
            'defense_domains': len(self.DEFENSE_DOMAINS),
            'code_hosting_domains': len(self.CODE_HOSTING_DOMAINS),
            'positive_patterns': len(self.POSITIVE_PATTERNS),
        }
        for category, patterns in self.EXCLUSION_PATTERNS.items():
            stats[f'{category}_patterns'] = len(patterns)
            stats[f'{category}_exceptions'] = len(
                self.EXCEPTION_PATTERNS_PER_CATEGORY.get(category, [])
            )
        return stats


def test_prefilter():
    """Self-test mirrors the cases lifted from CD v3's test suite (#52
    migration baseline). The content-too-short case from v3 is intentionally
    omitted here — v4's apply_filter does not call check_content_length
    (see module docstring; tracked as a separate follow-up)."""

    prefilter = CulturalDiscoveryPreFilterV4()

    test_cases = [
        # Should BLOCK - Political Conflict
        {
            'title': 'The Culture War Over Museum Collections',
            'text': "Critics slammed the museum's decision in a viral backlash on social media. "
                    "The controversy has sparked debate about woke culture and identity politics in arts institutions. "
                    "Cancel culture is threatening our cultural heritage, according to opponents. "
                    "The outrage continued to build as more critics weighed in on the politically correct decision. "
                    "This represents a cultural battle that will define our times.",
            'expected': (False, 'political_conflict'),
            'description': 'Culture war framing'
        },

        # Should PASS - Reconciliation (exception to political conflict)
        {
            'title': 'Truth and Reconciliation Through Art',
            'text': "The museum's new exhibition brings together artists from communities that were once in conflict. "
                    "Through dialogue and bridge-building, the collaborative project explores shared heritage and promotes understanding. "
                    "The reconciliation process has been supported by historians and cultural researchers. "
                    "The exhibition represents years of healing and coming together across cultural divides.",
            'expected': (True, 'passed'),
            'description': 'Reconciliation (exception)'
        },

        # Should BLOCK - Tourism Fluff
        {
            'title': 'Top 10 Must-See Ancient Temples in Asia',
            'text': "Planning your bucket list trip? These hidden gems offer breathtaking views and Instagram-worthy photo opportunities. "
                    "Don't miss these stunning must-visit destinations! Our travel guide covers where to stay, best hotels, and insider tips for getting around. "
                    "Book now for the trip of a lifetime to these picture-perfect ancient sites.",
            'expected': (False, 'tourism_fluff'),
            'description': 'Tourism listicle'
        },

        # Should PASS - UNESCO Heritage (exception to tourism)
        {
            'title': 'UNESCO Adds New Sites to World Heritage List',
            'text': "The World Heritage Committee has recognized several endangered heritage sites for preservation. "
                    "Conservation efforts are underway to protect these archaeological treasures. "
                    "Researchers and archaeologists have documented the historical significance of these ancient sites. "
                    "The museum and university teams are working on restoration projects to preserve these protected sites.",
            'expected': (True, 'passed'),
            'description': 'UNESCO heritage (exception)'
        },

        # Should BLOCK - Celebrity Art
        {
            'title': "Billionaire Collector's Painting Sells for $50 Million",
            'text': "The private collection went under the hammer at Christie's auction house yesterday. "
                    "The art market record was smashed as wealthy collectors bid on the appreciating asset. "
                    "Sotheby's experts had estimated the value at $40 million before the sale. "
                    "The billionaire's art investment portfolio continues to grow with this latest acquisition.",
            'expected': (False, 'celebrity_art'),
            'description': 'Celebrity art auction'
        },

        # Should PASS - Repatriation (exception to celebrity art)
        {
            'title': 'Museum Returns Looted Artifacts to Origin Country',
            'text': "After years of provenance research, the museum has agreed to repatriation of colonial-era artifacts. "
                    "The cultural property will be returned to its rightful owners following revelations about its illicit acquisition. "
                    "This donation to the nation represents a shift in how museums approach stolen heritage. "
                    "The public benefit of returning these items has been praised by cultural property experts.",
            'expected': (True, 'passed'),
            'description': 'Repatriation (exception)'
        },

        # Should BLOCK - Appropriation Debate
        {
            'title': 'Designer Accused of Cultural Appropriation',
            'text': "The designer was called out on Twitter for appropriating traditional patterns without permission. "
                    "Critics say this is problematic and offensive to the culture being appropriated. "
                    "The viral backlash led to accusations and social media outrage. "
                    "\"Not your culture, stay in your lane\" was the trending hashtag as the controversy grew.",
            'expected': (False, 'appropriation_debate'),
            'description': 'Appropriation debate'
        },

        # Should PASS - Respectful Collaboration (exception)
        {
            'title': 'Cross-Cultural Fashion Collaboration Launches',
            'text': "The respectful exchange between designers celebrates cultural heritage through collaboration. "
                    "Working together in partnership, the co-creation project was developed with invitation from community elders. "
                    "The collaboration includes acknowledgment and credit to traditional artisans. "
                    "Educational materials explain the history of the tradition and honor its origins.",
            'expected': (True, 'passed'),
            'description': 'Respectful collaboration (exception)'
        },

        # Should PASS - Archaeological Discovery
        {
            'title': 'Ancient Maya Temple Discovered with Unique Murals',
            'text': "Archaeologists have unearthed a 3,000-year-old temple containing murals depicting cross-cultural connections. "
                    "The breakthrough discovery was published in a peer-reviewed journal after excavation revealed new evidence of ancient trade networks. "
                    "The university research team includes historians and anthropologists who study shared heritage across civilizations. "
                    "This UNESCO world heritage site candidate shows cultural exchange between distant peoples.",
            'expected': (True, 'passed'),
            'description': 'Archaeological discovery'
        },

        # Should PASS - Cultural Heritage Restoration
        {
            'title': 'Restored Fresco Reveals Hidden Cultural Connections',
            'text': "Conservation experts at the museum have completed a five-year restoration project. "
                    "The preservation effort revealed previously unknown cultural heritage connections between Mediterranean civilizations. "
                    "Researchers discovered evidence of cross-cultural artistic exchange hidden beneath later paint layers. "
                    "The archaeological find was documented by university historians studying ancient intercultural trade routes.",
            'expected': (True, 'passed'),
            'description': 'Heritage restoration'
        },
    ]

    print("Testing Cultural Discovery Pre-Filter v4.0")
    print("=" * 70)

    passed_tests = 0
    failed_tests = 0

    for i, test in enumerate(test_cases, 1):
        result = prefilter.apply_filter(test)
        expected = test['expected']
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

    print("\n" + "=" * 70)
    print("Content Type Classification (sanity check):")
    for test in test_cases:
        content_type = prefilter.classify_content_type(test)
        print(f"  {test['description']}: {content_type}")


if __name__ == '__main__':
    test_prefilter()
