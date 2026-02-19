"""
Uplifting Pre-Filter v6.0

Blocks obvious low-value content before LLM labeling:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)
- Pure speculation articles (no documented outcomes)
- Code repositories and developer tutorials

Purpose: Reduce LLM costs and improve training data quality.

Identical to v5 prefilter. Crime content-type cap is enforced in training data
and config.yaml, not in the prefilter (prefilter blocks obvious crime, but
borderline crime articles are handled by label correction in training data).
"""

import re
from typing import Dict, Tuple, List

from filters.common.base_prefilter import BasePreFilter


class UpliftingPreFilterV6(BasePreFilter):
    """Fast rule-based pre-filter for uplifting content v6"""

    VERSION = "6.0"

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

    # === CRIME/VIOLENCE INDICATORS ===

    CRIME_VIOLENCE_PATTERNS = [
        # Violent crimes - specific terms only
        r'\b(murder|murdered|murderer|homicide|manslaughter)\b',
        r'\b(rape|raped|rapist|sexual assault|sexually assaulted|molest|molestation)\b',
        r'\b(assault|assaulted|stabbing|stabbed|shot dead)\b',  # removed 'battery' (solar), 'shooting' (photo)
        r'\b(child abuse|domestic violence|human trafficking)\b',  # more specific

        # Criminal justice (perpetrator focus) - require context
        r'\b(sentenced to|guilty verdict|prison sentence)\b',  # more specific than just 'sentenced'
        r'\b(convicted of|charged with murder|charged with rape|charged with assault)\b',
        r'\b(perpetrator|sex offender|violent offender)\b',
        r'\b(life sentence|death penalty|death row)\b',
        r'\b(tbs met|terbeschikkingstelling)\b',  # Dutch: more specific

        # Specific crime events
        r'\b(armed robbery|violent robbery|kidnapping|abduction)\b',  # removed generic 'theft', 'fraud'
        r'\b(terrorist attack|terrorism|mass shooting|massacre)\b',

        # Dutch crime terms - more specific
        r'\b(verkracht|verkrachting|mishandeling|doodslag)\b',  # removed 'misbruik' (too broad), 'cel' (biology)
        r'\b(gevangenisstraf|levenslang)\b',
        # === DUTCH (NL) ===
        r'(moord|vermoord|doodslag|levensdelict)',
        r'(verkracht|verkrachting|aanranding|zedendelict)',
        r'(mishandeling|steekpartij|neergeschoten)',
        r'(kindermishandeling|huiselijk geweld|mensenhandel)',
        r'(veroordeeld tot|gevangenisstraf|levenslang)',
        r'(tbs met|terbeschikkingstelling|dader|zedendelinquent)',
        r'(gewapende overval|gijzeling|ontvoering)',
        r'(terroristische aanslag|schietpartij|bloedbad)',

        # === GERMAN (DE) ===
        r'(mord|ermordet|totschlag|tötungsdelikt)',
        r'(vergewaltigung|vergewaltigt|sexuelle nötigung)',
        r'(körperverletzung|messerstecherei|erschossen)',
        r'(kindesmisshandlung|häusliche gewalt|menschenhandel)',
        r'(verurteilt zu|gefängnisstrafe|lebenslänglich)',
        r'(täter|sexualstraftäter|gewalttäter)',
        r'(bewaffneter überfall|entführung|geiselnahme)',
        r'(terroranschlag|amoklauf|massaker)',

        # === FRENCH (FR) ===
        r'(meurtre|assassinat|homicide|tué)',
        r'(viol|violée|agression sexuelle)',
        r'(agression|poignardé|abattu)',
        r'(maltraitance|violence domestique|traite des êtres humains)',
        r'(condamné à|peine de prison|perpétuité)',
        r'(auteur|agresseur sexuel|délinquant violent)',
        r"(braquage|enlèvement|prise d'otage)",
        r'(attentat terroriste|fusillade|massacre)',
    ]

    CRIME_VIOLENCE_EXCEPTIONS = [
        # Reform and rehabilitation focus
        r'\b(rehabilitation|restorative justice|reintegration|reform|reforming)\b',
        r'\b(crime prevention|violence prevention|intervention program)\b',
        r'\b(survivor support|victim support|healing|recovery program)\b',
        r'\b(recidivism reduction|second chance|reentry program)\b',

        # Policy/systemic reform
        r'\b(prison reform|criminal justice reform|decarceration|abolition)\b',
        r'\b(diversion program|alternative sentencing|community service)\b',

        # Positive outcomes from addressing past wrongs
        r'\b(law reform|legal reform|revamp.{0,10}law|new legislation)\b',
        r'\b(accountability|justice served|landmark ruling)\b',
        r'\b(survivor|survivors|overcame|healing from|reclaim)\b',

        # Fighting/combating issues (not perpetrating)
        r'\b(combat.{0,15}trafficking|fight.{0,15}trafficking|anti-trafficking)\b',
        r'\b(combat.{0,15}violence|fight.{0,15}violence|end.{0,15}violence)\b',
        r'\b(changemaker|women empowerment|silent revolution)\b',

        # Positive resolution
        r'\b(released|freed|liberated|rescued)\b',
        r'\b(vrijgelaten|bevrijd)\b',  # Dutch: released/freed
        # === DUTCH (NL) ===
        r'(rehabilitatie|resocialisatie|re-integratie|hervorming)',
        r'(gevangenishervorming|strafrechtshervorming)',
        r'(slachtofferhulp|slachtofferondersteuning)',

        # === GERMAN (DE) ===
        r'(freigelassen|befreit|gerettet)',
        r'(rehabilitation|resozialisierung|wiedereingliederung|reform)',
        r'(gefängnisreform|strafrechtsreform)',
        r'(opferhilfe|opferunterstützung)',

        # === FRENCH (FR) ===
        r'(libéré|libérée|sauvé|sauvée)',
        r'(réhabilitation|réinsertion|réforme)',
        r'(réforme pénitentiaire|réforme judiciaire)',
        r'(aide aux victimes|soutien aux victimes)',
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
        r'(investor relations|shareholder value|dividend|stock buyback)',
        # === DUTCH (NL) ===
        r'(aandelenkoers|beurskoers|marktkapitalisatie)',
        r'(kwartaalcijfers|kwartaalresultaten|boekjaar)',
        r'(investeringsronde|durfkapitaal|risicokapitaal)',
        r'(beursgang|fusie|overname)',

        # === GERMAN (DE) ===
        r'(aktienkurs|börsenkurs|marktkapitalisierung)',
        r'(quartalszahlen|geschäftsjahr|finanzperformance)',
        r'(finanzierungsronde|risikokapital|wagniskapital)',
        r'(börsengang|fusion|übernahme)',

        # === FRENCH (FR) ===
        r'(cours de bourse|capitalisation boursière)',
        r'(résultats trimestriels|exercice fiscal)',
        r'(levée de fonds|capital-risque)',
        r'(introduction en bourse|fusion|acquisition)',
    ]

    CORPORATE_FINANCE_EXCEPTIONS = [
        r'\b(worker cooperative|worker-owned|employee-owned|coop)\b',
        r'\b(public benefit|b corp|benefit corporation|social enterprise)\b',
        r'\b(open source|open access|freely available|creative commons)\b',
        r'\b(affordable access|community ownership|commons|mutual aid)\b',
        r'(non-?profit|nonprofit|ngo|charity|foundation)',
        # === DUTCH (NL) ===
        r'(werknemerscoöperatie|coöperatie|sociaal ondernemen)',
        r'(maatschappelijke onderneming|stichting|goed doel)',

        # === GERMAN (DE) ===
        r'(genossenschaft|sozialunternehmen|gemeinnützig)',
        r'(stiftung|wohltätigkeit)',

        # === FRENCH (FR) ===
        r'(coopérative|entreprise sociale|économie sociale)',
        r'(association|fondation|but non lucratif)',
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
        r'(military spending|arms procurement|defense contract)',
        # === DUTCH (NL) ===
        r'(militaire opbouw|defensie-uitgaven|krijgsmacht|troepenmacht)',
        r'(wapensysteem|wapenhandel|munitie|raketten|gevechtsvliegtuigen)',
        r'(navo-uitbreiding|militaire alliantie|defensieverdrag)',
        r'(grensbeveiliging|grensbewaking|militaire oefening)',

        # === GERMAN (DE) ===
        r'(militärischer aufbau|verteidigungsausgaben|streitkräfte)',
        r'(waffensystem|waffenhandel|munition|raketen|kampfflugzeuge)',
        r'(nato-erweiterung|militärbündnis|verteidigungspakt)',
        r'(grenzschutz|militärübung|verteidigungshaushalt)',

        # === FRENCH (FR) ===
        r'(renforcement militaire|dépenses de défense|forces armées)',
        r"(système d'armes|vente d'armes|munitions|missiles)",
        r"(élargissement de l'otan|alliance militaire|pacte de défense)",
        r'(sécurité aux frontières|exercice militaire|budget de la défense)',
    ]

    MILITARY_SECURITY_EXCEPTIONS = [
        r'\b(peace|peace process|peace agreement|peace talks|peace treaty)\b',
        r'\b(demilitarization|disarmament|arms reduction|denuclearization)\b',
        r'\b(conflict resolution|reconciliation|ceasefire|armistice)\b',
        r'\b(peacekeeping|peace keeping|un peace|humanitarian)\b',
        r'\b(truth commission|war crimes tribunal|justice|accountability)\b',
        r'(veterans? (support|services|care|mental health))',
        # === DUTCH (NL) ===
        r'(vrede|vredesproces|vredesakkoord|vredesbesprekingen)',
        r'(demilitarisering|ontwapening|wapenreductie)',
        r'(conflictoplossing|verzoening|staakt-het-vuren|wapenstilstand)',
        r'(vredesmissie|humanitair)',

        # === GERMAN (DE) ===
        r'(frieden|friedensprozess|friedensabkommen|friedensgespräche)',
        r'(demilitarisierung|abrüstung|waffenreduzierung)',
        r'(konfliktlösung|versöhnung|waffenstillstand)',
        r'(friedensmission|humanitär)',

        # === FRENCH (FR) ===
        r'(paix|processus de paix|accord de paix|négociations de paix)',
        r'(démilitarisation|désarmement|réduction des armes)',
        r'(résolution des conflits|réconciliation|cessez-le-feu|armistice)',
        r'(maintien de la paix|humanitaire)',
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

        # Crime/violence
        self.crime_violence_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CRIME_VIOLENCE_PATTERNS
        ]
        self.crime_violence_exceptions_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CRIME_VIOLENCE_EXCEPTIONS
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

        # Check crime/violence
        if self._has_pattern(combined_text, self.crime_violence_regex):
            if not self._has_pattern(combined_text, self.crime_violence_exceptions_regex):
                return False, "crime_violence"

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
            'corporate_finance_patterns': len(self.CORPORATE_FINANCE_PATTERNS),
            'corporate_finance_exceptions': len(self.CORPORATE_FINANCE_EXCEPTIONS),
            'military_security_patterns': len(self.MILITARY_SECURITY_PATTERNS),
            'military_security_exceptions': len(self.MILITARY_SECURITY_EXCEPTIONS),
            'crime_violence_patterns': len(self.CRIME_VIOLENCE_PATTERNS),
            'crime_violence_exceptions': len(self.CRIME_VIOLENCE_EXCEPTIONS),
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
