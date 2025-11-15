"""
Sustainability Tech Innovation Pre-Filter v1.0

PIVOT FROM v3 (deployed only):
- NOW INCLUDES: Working pilots, validated research with real results
- STILL BLOCKS: Pure theory, simulations without validation, vaporware
- GOAL: Capture cool sustainable tech that WORKS (not just mass deployment)

Expected pass rate: ~5-20% (more permissive than v3's 2-5%)
"""

import re
from typing import Dict, List, Optional
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV1(BasePreFilter):
    """Pre-filter for sustainable tech innovation (deployed + pilots + validated research)"""

    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "1.0"

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

        # BLOCK: Infrastructure disruption (protests, strikes, service outages)
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # BLOCK: Social media posts without strong deployment signals
        if self._is_social_media_without_deployment(text_lower):
            return (False, "social_media_no_deployment")

        # NEW: Research papers - ALLOW if has validation/results
        if self._is_research_paper(text_lower):
            if self._has_validation_evidence(text_lower):
                return (True, "research_with_validation")
            else:
                return (False, "research_without_results")

        # BLOCK: Pure vaporware (no evidence of ANY real work)
        vaporware_patterns = [
            r'\b(concept|theoretical|could potentially)\b',
            r'\bbreakthrough (announced|unveiled)\b(?!.{0,100}\b(deployed|operational|pilot)\b)',
            r'\bunveils? (revolutionary|breakthrough|game-changing)(?!.{0,100}\b(deployed|pilot|validated)\b)',
            r'\b(will revolutionize|promises to transform)\b(?!.{0,100}\b(pilot|demonstration)\b)',
        ]

        for pattern in vaporware_patterns:
            if re.search(pattern, text_lower):
                # Check if there's ANY evidence of real work (deployment, pilot, validation)
                if not self._has_any_evidence_of_real_work(text_lower):
                    return (False, "vaporware_announcement")

        # BLOCK: Future-only (no current work)
        future_only_patterns = [
            r'\b(plans to|aiming to|will deploy|will build) .{0,50}\bby (20\d{2}|next year)\b',
            r'\b(committed to|pledges to|targets?) .{0,50}\bby (20\d{2})\b',
            r'\baims? for commercial (deployment|operation) .{0,30}\bin (20\d{2})\b',
            r'\bexpects? to (launch|deploy|begin) .{0,30}\bin (20\d{2}|next year|coming years)\b',
        ]

        for pattern in future_only_patterns:
            if re.search(pattern, text_lower):
                # Check if there's ALSO current work (deployment, pilot, or research)
                if not self._has_any_evidence_of_real_work(text_lower):
                    return (False, "future_only_no_current_work")

        # LESS STRICT than v3: Allow lab/research IF has validation
        lab_only_patterns = [
            r'\b(laboratory|lab) (results?|experiment|study|test)\b',
            r'\bbench-?scale\b',
            r'\bresearch (published|shows?|demonstrates?|finds?)\b',
            r'\bpeer-reviewed study (shows?|finds?|demonstrates?)\b',
        ]

        for pattern in lab_only_patterns:
            if re.search(pattern, text_lower):
                # Allow if there's validation/performance data OR deployment/pilot language
                if not self._has_validation_evidence(text_lower):
                    return (False, "lab_only_no_validation")

        # BLOCK: Pure simulations without real-world validation
        if self._is_pure_simulation(text_lower):
            return (False, "simulation_only_no_validation")

        # LESS STRICT than v3: Don't require deployment language
        # Now require EITHER deployment OR pilot OR validation evidence
        if not self._has_any_evidence_of_real_work(text_lower):
            return (False, "no_validation_evidence")

        # PASS: Has evidence of real work (deployment, pilot, or validation)
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
            'paris agreement', 'cop27', 'cop28', 'cop29', 'unfccc',
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

    def _has_pilot_evidence(self, text_lower: str) -> bool:
        """Check if article has evidence of working pilot/demonstration"""

        pilot_patterns = [
            r'\bpilot (project|plant|facility|installation)\s+.{0,100}\b(generating|produced|achieved|demonstrated)\b',
            r'\bdemonstration (project|plant|facility)\s+.{0,100}\b(performance|results|data)\b',
            r'\bprototype (tested|validated|demonstrated)\s+.{0,100}\b(in|with|at)\b',
            r'\bfield (test|trial|demonstration)\s+.{0,100}\b(achieved|demonstrated|showed)\b',
            r'\b(pilot|demonstration)\s+.{0,100}\b\d+\s*(kw|mw)\b',
            r'\b(pilot|demonstration)\s+.{0,100}\b(six months|year|months)\b',
            r'\bworking (prototype|pilot|demonstration)\b',
            r'\bsuccessful (pilot|demonstration|field test)\b',
        ]

        for pattern in pilot_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _has_validation_evidence(self, text_lower: str) -> bool:
        """Check if article has evidence of real-world validation (for research)"""

        validation_patterns = [
            r'\b(validated|proven|achieved|demonstrated)\s+.{0,50}\b(in|with|using)\s+(real|field|actual)\b',
            r'\bperformance (data|results|metrics)\s+.{0,50}\b(from|in)\s+(real|field|actual)\b',
            r'\breal-world (validation|results|data|performance)\b',
            r'\bfield (validation|testing|data|results)\b',
            r'\b(achieved|demonstrated|showed)\s+\d+%\s+(efficiency|accuracy|improvement)\b',
            r'\btested (in|on|with)\s+(real|actual|field)\b',
            r'\b(actual|measured|observed)\s+(performance|efficiency|results)\b',
            r'\bdata (from|collected)\s+(real|field|actual|operational)\b',
            r'\b\d+\s+(kw|mw)\s+(generated|produced|achieved)\b',
            r'\b(reduced|avoided|saved)\s+\d+\s*(tons?|kg|tonnes?)\s+(co2|carbon|emissions)\b',
        ]

        for pattern in validation_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _has_any_evidence_of_real_work(self, text_lower: str) -> bool:
        """Check if article has ANY evidence of real work (deployment, pilot, or validation)"""

        # Check all three types of evidence
        if self._has_deployment_evidence(text_lower):
            return True

        if self._has_pilot_evidence(text_lower):
            return True

        if self._has_validation_evidence(text_lower):
            return True

        return False

    def _is_pure_simulation(self, text_lower: str) -> bool:
        """Detect pure simulations/models without real-world validation"""

        simulation_patterns = [
            r'\bsimulation (shows|demonstrates|predicts)\b',
            r'\bmodel (predicts|shows|demonstrates)\b',
            r'\btheoretical (model|framework|approach)\b',
            r'\bcomputational (model|simulation)\b',
            r'\b(monte carlo|agent-based|discrete event) simulation\b',
        ]

        has_simulation = any(re.search(pattern, text_lower) for pattern in simulation_patterns)

        if not has_simulation:
            return False

        # Allow if there's validation against real data
        validation_signals = [
            r'\bvalidated (against|with|using)\s+(real|actual|field|measured)\b',
            r'\b(real|actual|field|measured)\s+(data|results|performance)\b',
            r'\b(achieved|demonstrated|tested)\s+in\s+(field|real-world|actual)\b',
        ]

        has_validation = any(re.search(pattern, text_lower) for pattern in validation_signals)

        # Block if simulation WITHOUT validation
        return has_simulation and not has_validation

    def _is_research_paper(self, text_lower: str) -> bool:
        """Detect research papers (arXiv, bioRxiv, journals)"""

        research_patterns = [
            r'\b(arxiv|biorxiv|medrxiv)\b',
            r'\barxiv\.org\b',
            r'\bdoi:\s*10\.\d+',
            r'\bpreprint\b',
            r'\bpaper published in\b',
            r'\bresearch paper\b',
            r'\bjournal of\b',
        ]

        has_research = any(re.search(pattern, text_lower) for pattern in research_patterns)
        return has_research

    def _is_social_media_without_deployment(self, text_lower: str) -> bool:
        """Detect social media posts unless they have strong signals"""

        social_media_patterns = [
            r'\b(reddit|hacker news|hackernews|twitter|social media)\b',
            r'\br/\w+\b',  # Reddit subreddit format
            r'\bhn:\b',  # Hacker News prefix
        ]

        has_social_media = any(re.search(pattern, text_lower) for pattern in social_media_patterns)

        if not has_social_media:
            return False

        # Require evidence of real work for social media
        has_evidence = self._has_any_evidence_of_real_work(text_lower)

        # Block if social media WITHOUT evidence of real work
        return has_social_media and not has_evidence

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

        # Block all disruption articles - these are NOT about innovation
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

        # Pilot indicators
        if re.search(r'\bpilot (project|plant|facility)\b', text_lower):
            indicators.append("has_pilot")

        if re.search(r'\bdemonstration (project|plant)\b', text_lower):
            indicators.append("has_demonstration")

        # Validation indicators
        if re.search(r'\b(validated|proven|achieved)\b', text_lower):
            indicators.append("has_validation")

        if re.search(r'\breal-world (validation|results|data)\b', text_lower):
            indicators.append("has_real_world_data")

        if re.search(r'\bperformance (data|results)\b', text_lower):
            indicators.append("has_performance_data")

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
