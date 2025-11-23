"""
Sustainability Tech Innovation Pre-Filter v1.1 - OPTION B: Balanced

STRATEGY: Balance coverage vs quality - require substantive evidence
- Climate/energy focus (not general sustainability)
- Require SUBSTANTIVE evidence (not just mentions)
- Target: 5-10% pass rate

Expected FP rate: 5-10% (balanced trade-off)
"""

import re
from typing import Dict, List
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV1_OptionB(BasePreFilter):
    """Pre-filter Option B: Balanced - Quality vs coverage trade-off"""

    VERSION = "1.1-option-b"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "1.1-option-b"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Not climate/energy related
        if not self._is_climate_energy_related(text_lower):
            return (False, "not_climate_energy")

        # BLOCK: Out-of-scope content
        if self._is_out_of_scope(text_lower):
            return (False, "out_of_scope")

        # BLOCK: Infrastructure disruption
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # BLOCK: Pure vaporware
        if self._is_pure_vaporware(text_lower):
            return (False, "pure_vaporware")

        # BLOCK: Future-only proposals
        if self._is_future_only_proposal(text_lower):
            return (False, "future_only_proposal")

        # REQUIRE: Substantive evidence (deployed OR working pilot OR validated research)
        if not self._has_substantive_evidence(text_lower):
            return (False, "no_substantive_evidence")

        # PASS: Has climate/energy relevance AND substantive evidence
        return (True, "passed")

    def _is_climate_energy_related(self, text_lower: str) -> bool:
        """STRICT: Climate/energy/clean tech only"""

        climate_energy_keywords = [
            # Climate
            'climate', 'carbon', 'emission', 'greenhouse', 'warming',
            'net-zero', 'net zero', 'carbon neutral', 'decarboniz',

            # Renewable energy
            'renewable', 'solar', 'wind', 'geothermal', 'hydro', 'nuclear',
            'clean energy', 'green energy',

            # Electric transport
            'electric vehicle', ' ev ', 'bev', 'phev', 'battery',

            # Energy infrastructure
            'energy storage', 'grid storage', 'hydrogen', 'fuel cell',
            'heat pump', 'energy efficiency', 'smart grid',

            # Fossil (in context of transition)
            'fossil fuel', 'coal phase', 'oil transition',
        ]

        return any(kw in text_lower for kw in climate_energy_keywords)

    def _is_out_of_scope(self, text_lower: str) -> bool:
        """Block IT infrastructure, healthcare, generic business"""

        out_of_scope_patterns = [
            # IT infrastructure
            r'\b(kubernetes|docker|aws|azure|terraform|microservices?)\b(?!.{0,100}\b(energy|solar|wind|renewable)\b)',
            r'\bdevops|ci/cd|api endpoint\b',

            # Healthcare/medicine (unless climate-related)
            r'\b(medicinal plants?|pharmaceutical|clinical trial|disease)\b(?!.{0,100}\b(climate|sustainable)\b)',

            # Pure biodiversity (unless climate adaptation)
            r'\b(ethnobotanical|species conservation)\b(?!.{0,100}\b(climate|carbon)\b)',

            # Generic business
            r'\b(employee benefits?|mental health|workplace culture)\b',
            r'\b(interview questions?|hiring practices?)\b',

            # Finance (unless green/climate finance)
            r'\b(stock trading|forex|investment strategy)\b(?!.{0,100}\b(climate|green|sustainable|renewable)\b)',
        ]

        return any(re.search(p, text_lower) for p in out_of_scope_patterns)

    def _is_infrastructure_disruption(self, text_lower: str) -> bool:
        """Detect protests, strikes"""

        disruption_patterns = [
            r'\b(protest|protesters?)\s+(block|disrupt|shut down)\b',
            r'\bstrike\s+(halts?|stops?|disrupts?)\b',
            r'\bextinction rebellion\s+(blocks?|disrupts?)\b',
            r'\bactivists?\s+(chain|glue|block)\b',
        ]

        return any(re.search(p, text_lower) for p in disruption_patterns)

    def _is_pure_vaporware(self, text_lower: str) -> bool:
        """Announcements without evidence"""

        vaporware_indicators = [
            r'\bunveils? (revolutionary|breakthrough)',
            r'\bpromises? to revolutionize',
            r'\bcould potentially transform',
        ]

        has_vaporware = any(re.search(p, text_lower) for p in vaporware_indicators)

        if not has_vaporware:
            return False

        # Require STRONG evidence to override
        strong_evidence = [
            r'\b(deployed|operational) since \d{4}\b',
            r'\b\d+\s*gw\s+(deployed|operational|generating)\b',
            r'\bpilot .{0,50}\b(months|year)\s+(of operation|operational)\b',
        ]

        has_strong_evidence = any(re.search(p, text_lower) for p in strong_evidence)

        return has_vaporware and not has_strong_evidence

    def _is_future_only_proposal(self, text_lower: str) -> bool:
        """Proposals without current work"""

        # Detect proposals
        proposal_patterns = [
            r'\bproposes? (to )?(build|deploy|install)\s+.{0,50}\b\d+\s*(mw|gw)\b',
            r'\bplans? to .{0,50}\bby 20\d{2}\b',
            r'\bexpects? (to )?(deploy|complete|launch)\s+.{0,50}\bin 20\d{2}\b',
            r'\bapproved .{0,50}\bproject .{0,50}\bdelivery (in|by) 20\d{2}\b',
        ]

        has_proposal = any(re.search(p, text_lower) for p in proposal_patterns)

        if not has_proposal:
            return False

        # Allow if has CURRENT pilot or deployment
        current_work_patterns = [
            r'\b(currently|already)\s+(operational|deployed|testing)\b',
            r'\bhas (deployed|installed|built)\s+\d+\s*(mw|gw)\b',
            r'\boperational since \d{4}\b',
            r'\bpilot (operational|running|generating)\b',
        ]

        has_current_work = any(re.search(p, text_lower) for p in current_work_patterns)

        # Block if proposal WITHOUT current work
        return has_proposal and not has_current_work

    def _has_substantive_evidence(self, text_lower: str) -> bool:
        """Require SUBSTANTIVE evidence (not just mentions)"""

        # 1. Deployment evidence (operational systems)
        deployment_evidence = [
            r'\b(operational|generating|producing)\s+.{0,50}\b\d+\s*(mw|gw|gwh)\b',
            r'\bdeployed .{0,50}\b\d+\s*(mw|gw)\b',
            r'\b\d+\s*gw\s+(capacity|installed|deployed)\b',
            r'\bmarket share\s+\d+%\b',
            r'\boperational since \d{4}\b',
        ]

        if any(re.search(p, text_lower) for p in deployment_evidence):
            return True

        # 2. Pilot evidence (working pilots with data/duration)
        pilot_evidence = [
            r'\bpilot .{0,100}\b(generated?|produced|achieved|demonstrated)\s+\d+\s*(kw|mw)\b',
            r'\bpilot .{0,100}\b(six months?|year|months?)\s+(of operation|operational)\b',
            r'\bdemonstration (project|plant) .{0,100}\b(performance|results|data)\b',
            r'\bfield test .{0,100}\b(achieved|demonstrated|showed)\s+\d+%\b',
            r'\bsuccessful pilot .{0,100}\b\d+\s*(kw|mw)\b',
        ]

        if any(re.search(p, text_lower) for p in pilot_evidence):
            return True

        # 3. Research validation (real-world validation with metrics)
        validation_evidence = [
            r'\bvalidated .{0,50}\bon\s+\d+\s+(vehicles?|systems?|installations?)\b',
            r'\breal-world (data|testing|validation) .{0,100}\bover\s+\d+\s+(months?|years?)\b',
            r'\bfield (test|validation) .{0,100}\b(achieved|demonstrated)\s+\d+%\s+(efficiency|accuracy)\b',
            r'\b(measured|observed|actual)\s+performance .{0,100}\b\d+\s*(kw|mw|%)\b',
        ]

        if any(re.search(p, text_lower) for p in validation_evidence):
            return True

        return False

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title + description + content"""
        parts = []
        if 'title' in article:
            parts.append(article['title'])
        if 'description' in article:
            parts.append(article['description'])
        if 'content' in article:
            parts.append(article['content'][:2000])
        return ' '.join(parts)
