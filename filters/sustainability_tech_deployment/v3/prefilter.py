"""
Sustainability Tech Deployment Pre-Filter v3.0

CRITICAL FIXES FROM REVALIDATION:
- Block arXiv/bioRxiv research papers unless explicit deployment
- Block social media posts without strong deployment signals
- Block infrastructure disruption (protests, service outages)
- Require explicit deployment language (deployed/installed/operational)

Expected pass rate: ~5-10% (EXTREMELY selective for deployed tech only)
"""

import re
from typing import Dict, List, Optional
from filters.base_prefilter import BasePreFilter


class TechDeploymentPreFilterV3(BasePreFilter):
    """Pre-filter for deployed climate technology (blocks vaporware and research)"""

    VERSION = "3.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_deployment"
        self.version = "3.0"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Not climate/sustainability related at all
        if not self._is_sustainability_related(text_lower):
            return (False, "not_sustainability_topic")

        # BLOCK: Research papers (arXiv, bioRxiv, medRxiv) unless explicit deployment
        if self._is_research_paper(text_lower):
            return (False, "research_paper")

        # BLOCK: Social media posts without strong deployment signals
        if self._is_social_media_without_deployment(text_lower):
            return (False, "social_media_no_deployment")

        # BLOCK: Infrastructure disruption (protests, strikes, service outages)
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # BLOCK: Vaporware and prototypes
        vaporware_patterns = [
            r'\b(concept|theoretical|could potentially)\b',
            r'\bbreakthrough (announced|unveiled)\b',
            r'\b(prototype|demo|proof-of-concept) (unveiled|announced|demonstrated)\b',
            r'\bunveils? (revolutionary|breakthrough|game-changing)',
            r'\b(will revolutionize|promises to transform)\b',
        ]

        for pattern in vaporware_patterns:
            if re.search(pattern, text_lower):
                # Check if there's deployment data to override
                if not self._has_deployment_evidence(text_lower):
                    return (False, "vaporware_announcement")

        # BLOCK: Future-only (no current deployment)
        future_only_patterns = [
            r'\b(plans to|aiming to|will deploy|will build) .{0,50}\bby (20\d{2}|next year)\b',
            r'\b(committed to|pledges to|targets?) .{0,50}\bby (20\d{2})\b',
            r'\baims? for commercial (deployment|operation) .{0,30}\bin (20\d{2})\b',
            r'\bexpects? to (launch|deploy|begin) .{0,30}\bin (20\d{2}|next year|coming years)\b',
        ]

        for pattern in future_only_patterns:
            if re.search(pattern, text_lower):
                # Check if there's ALSO current deployment
                if not self._has_deployment_evidence(text_lower):
                    return (False, "future_only_no_deployment")

        # BLOCK: Lab/research only
        lab_only_patterns = [
            r'\b(laboratory|lab) (results?|experiment|study|test)\b',
            r'\bbench-?scale\b',
            r'\bresearch (published|shows?|demonstrates?|finds?)\b(?!.{0,100}\b(deployed|commercial|operational)\b)',
            r'\bpeer-reviewed study (shows?|finds?|demonstrates?)\b(?!.{0,100}\b(deployed|commercial)\b)',
            r'\bin (controlled|ideal) conditions?\b',
        ]

        for pattern in lab_only_patterns:
            if re.search(pattern, text_lower):
                # Allow if there's commercial/deployment language
                if not self._has_deployment_evidence(text_lower):
                    return (False, "lab_only_no_commercialization")

        # BLOCK: Pure R&D announcements
        if self._is_pure_research(text_lower):
            return (False, "pure_research_no_deployment")

        # BLOCK: Token scale (negligible deployment)
        if self._is_token_scale(text_lower):
            return (False, "token_scale_negligible")

        # CRITICAL: Require explicit deployment language
        # This is the KEY fix from revalidation - don't pass articles without clear deployment signals
        if not self._has_deployment_evidence(text_lower):
            return (False, "no_deployment_language")

        # PASS: Has deployment evidence AND passed all other checks
        return (True, "passed")

    def _is_sustainability_related(self, text_lower: str) -> bool:
        """Check if article is about climate/sustainability/clean energy (PERMISSIVE)"""

        # Single keyword mentions - very broad to capture tangential content
        keywords = [
            'climate', 'carbon', 'emission', 'greenhouse', 'warming',
            'renewable', 'solar', 'wind', 'geothermal', 'hydro', 'nuclear',
            'battery', 'electric vehicle', ' ev ', 'bev', 'phev',
            'fossil fuel', 'coal', 'oil', 'gas',
            'sustainability', 'sustainable', 'green energy', 'clean energy',
            'net-zero', 'net zero', 'carbon neutral', 'decarboniz',
            'energy storage', 'grid storage', 'hydrogen',
            'heat pump', 'energy efficiency',
            'circular economy', 'recycl', 'waste reduction',
            'ecosystem', 'biodiversity', 'conservation', 'reforestation',
            'pollution', 'air quality', 'water quality',
            'paris agreement', 'cop27', 'cop28', 'unfccc',
        ]

        # Very permissive - just needs ONE keyword mention
        return any(kw in text_lower for kw in keywords)

    def _has_deployment_evidence(self, text_lower: str) -> bool:
        """Check if article has evidence of actual deployment"""

        # Strong deployment signals
        deployment_keywords = [
            r'\b(operational|online|generating|producing)\b',
            r'\b(deployed|installed|commissioned)\b',
            r'\bnow (operational|online|generating)\b',
            r'\bhas (deployed|installed|generated)\b',
            r'\b\d+[\s,]*(mw|gw|megawatt|gigawatt)s?\b',
            r'\b\d+[\s,]*(tons?|tonnes?) (co2|carbon)\b',
            r'\b\d{1,3}(,\d{3})+ (units?|vehicles?|installations?)\b',
            r'\bmarket share\b',
            r'\bcapacity factor\b',
            r'\b(operational|deployed) since \d{4}\b',
        ]

        for pattern in deployment_keywords:
            if re.search(pattern, text_lower):
                return True

        # Specific scale indicators
        if re.search(r'\b\d+\.?\d*\s*(gw|gigawatt)s?\b', text_lower):
            return True  # GW scale is always significant

        if re.search(r'\b(hundreds?|thousands?) of (mw|megawatt)s?\b', text_lower):
            return True

        # Market penetration signals
        if re.search(r'\b\d+%\s*(of|market|share)\b', text_lower):
            return True

        return False

    def _is_pure_research(self, text_lower: str) -> bool:
        """Detect pure research announcements with no commercialization"""

        research_signals = [
            r'\bresearch(?:ers)? (at|from|published)',
            r'\buniversity (of|announces|publishes)',
            r'\b(paper|study) published in\b',
            r'\bjournal of\b',
            r'\bdoi:\s*10\.\d+',
        ]

        commercial_signals = [
            r'\b(commercial|deployed|operational|installed)\b',
            r'\b(company|startup|manufacturer)\b',
            r'\bmarket\b',
            r'\bcustomers?\b',
        ]

        has_research = any(re.search(pattern, text_lower) for pattern in research_signals)
        has_commercial = any(re.search(pattern, text_lower) for pattern in commercial_signals)

        # Block if research signals but NO commercial signals
        return has_research and not has_commercial

    def _is_token_scale(self, text_lower: str) -> bool:
        """Detect token/negligible scale deployments"""

        token_patterns = [
            r'\bsingle (building|office|facility)\b',
            r'\bone (pilot|demonstration|test)\b(?!.{0,50}\b(hundreds?|thousands?|many)\b)',
            r'\b(pilot|demonstration) (project|facility|installation)\b(?!.{0,50}\b\d+\s*(mw|gw)\b)',
            r'\bless than \d+\s*kw\b',
            r'\b\d+\s*kw(?!h)\b(?!.{0,20}\b(hundreds?|thousands?)\b)',  # kW scale (not kWh)
        ]

        for pattern in token_patterns:
            if re.search(pattern, text_lower):
                # Check if there's evidence of larger scale
                if not re.search(r'\b\d+\s*(mw|gw)\b', text_lower):
                    if not re.search(r'\b\d{3,}(,\d{3})*\s*(units?|installations?)\b', text_lower):
                        return True

        return False

    def _is_research_paper(self, text_lower: str) -> bool:
        """Detect research papers (arXiv, bioRxiv, medRxiv) unless explicit deployment"""

        research_patterns = [
            r'\b(arxiv|biorxiv|medrxiv)\b',
            r'\barxiv\.org\b',
            r'\bdoi:\s*10\.\d+',
            r'\bpreprint\b',
            r'\bpaper published in\b',
            r'\bresearch paper\b',
            r'\bstudy finds\b',
            r'\bscientists discover\b',
            r'\btheoretical (model|framework|approach)\b',
            r'\bsimulation (shows|demonstrates|predicts)\b',
            r'\bmodel predicts\b',
        ]

        has_research = any(re.search(pattern, text_lower) for pattern in research_patterns)

        if not has_research:
            return False

        # Allow if there's STRONG deployment evidence (not just mentions)
        strong_deployment = [
            r'\b\d+\s*(mw|gw|megawatt|gigawatt)s?\s+(deployed|installed|operational|generating)\b',
            r'\b(commercial|deployed|operational)\s+since\s+\d{4}\b',
            r'\b\d{1,3}(,\d{3})+\s+(units|vehicles|installations)\s+(deployed|installed|sold)\b',
        ]

        has_strong_deployment = any(re.search(pattern, text_lower) for pattern in strong_deployment)

        # Block if research paper WITHOUT strong deployment evidence
        return has_research and not has_strong_deployment

    def _is_social_media_without_deployment(self, text_lower: str) -> bool:
        """Detect social media posts unless they have strong deployment signals"""

        social_media_patterns = [
            r'\b(reddit|hacker news|hackernews|twitter|social media)\b',
            r'\br/\w+\b',  # Reddit subreddit format
            r'\bhn:\b',  # Hacker News prefix
        ]

        has_social_media = any(re.search(pattern, text_lower) for pattern in social_media_patterns)

        if not has_social_media:
            return False

        # Require VERY strong deployment evidence for social media
        strong_deployment = [
            r'\b\d+\s*(mw|gw)\s+(deployed|installed|operational)\b',
            r'\b(commercial|operational)\s+since\s+\d{4}\b',
            r'\bmarket share\b',
        ]

        has_strong_deployment = any(re.search(pattern, text_lower) for pattern in strong_deployment)

        # Block if social media WITHOUT strong deployment
        return has_social_media and not has_strong_deployment

    def _is_infrastructure_disruption(self, text_lower: str) -> bool:
        """Detect infrastructure disruption (protests, strikes, service outages)"""

        disruption_patterns = [
            r'\b(protest|protesters?|demonstrat(ion|ors?))\b',
            r'\b(strike|strikes|striking)\b',
            r'\b(disruption|disrupted|blocked|halted)\b',
            r'\b(service outage|power outage|blackout)\b',
            r'\bextinction rebellion\b',
            r'\b(activists?|campaigners?)\s+(block|disrupt|halt)\b',
            r'\btraffic\s+(blocked|halted|stopped)\b',
        ]

        has_disruption = any(re.search(pattern, text_lower) for pattern in disruption_patterns)

        # Block all disruption articles - these are NOT about deployment
        return has_disruption

    def get_pass_indicators(self, article: Dict) -> List[str]:
        """Return list of indicators that led to passing (for debugging)"""

        text = self._get_combined_text(article)
        text_lower = text.lower()
        indicators = []

        # Deployment indicators
        if re.search(r'\b(operational|online|generating|producing)\b', text_lower):
            indicators.append("operational_language")

        if re.search(r'\b(deployed|installed|commissioned)\b', text_lower):
            indicators.append("deployment_language")

        if re.search(r'\b\d+[\s,]*(mw|gw|megawatt|gigawatt)s?\b', text_lower):
            indicators.append("has_mw_gw_scale")

        if re.search(r'\b\d+[\s,]*(tons?|tonnes?) (co2|carbon)\b', text_lower):
            indicators.append("has_emissions_data")

        if re.search(r'\b\d{1,3}(,\d{3})+ (units?|vehicles?|installations?)\b', text_lower):
            indicators.append("has_unit_count")

        if re.search(r'\bmarket share\b', text_lower):
            indicators.append("has_market_share")

        if re.search(r'\bcapacity factor\b', text_lower):
            indicators.append("has_capacity_factor")

        return indicators

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title + description + content for analysis"""
        parts = []

        if 'title' in article:
            parts.append(article['title'])

        if 'description' in article:
            parts.append(article['description'])

        if 'content' in article:
            # Limit content to first 2000 chars for pre-filter efficiency
            parts.append(article['content'][:2000])

        return ' '.join(parts)
