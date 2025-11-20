"""
Sustainability Tech Innovation Pre-Filter v1.1 - OPTION A: Relaxed

STRATEGY: More permissive - capture more pilots/research, accept higher FP rate
- Loosen sustainability check (climate/energy focus, but broader)
- Lower evidence threshold (ANY mention of pilots/data counts)
- Target: 10-20% pass rate (vs 2.3% in v1.0)

Expected FP rate: 15-20% (acceptable trade-off for broader coverage)
"""

import re
from typing import Dict, List
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV1_OptionA(BasePreFilter):
    """Pre-filter Option A: Relaxed - Broader coverage, accept some FPs"""

    VERSION = "1.1-option-a"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "1.1-option-a"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Not climate/energy related (TIGHTENED but still broad)
        if not self._is_climate_energy_related(text_lower):
            return (False, "not_climate_energy")

        # BLOCK: Obvious out-of-scope (IT infrastructure, medicinal plants)
        if self._is_out_of_scope(text_lower):
            return (False, "out_of_scope")

        # BLOCK: Infrastructure disruption
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # BLOCK: Pure vaporware (announcements with no evidence)
        if self._is_pure_vaporware(text_lower):
            return (False, "pure_vaporware")

        # BLOCK: Future-only proposals (no current work)
        if self._is_future_only(text_lower):
            return (False, "future_only")

        # RELAXED: Pass if has ANY tech signal (deployed, pilot, or research mention)
        if not self._has_any_tech_signal(text_lower):
            return (False, "no_tech_signal")

        # PASS: Has climate/energy relevance AND some tech signal
        return (True, "passed")

    def _is_climate_energy_related(self, text_lower: str) -> bool:
        """Climate/energy focus (TIGHTENED from v1.0)"""

        # Core climate/energy keywords
        core_keywords = [
            'climate', 'carbon', 'emission', 'greenhouse', 'warming',
            'renewable', 'solar', 'wind', 'geothermal', 'hydro', 'nuclear',
            'battery', 'electric vehicle', ' ev ', 'bev', 'phev',
            'fossil fuel', 'coal', 'oil', 'gas', 'natural gas',
            'sustainable', 'green energy', 'clean energy',
            'net-zero', 'net zero', 'carbon neutral', 'decarboniz',
            'energy storage', 'grid storage', 'hydrogen', 'fuel cell',
            'heat pump', 'energy efficiency', 'smart grid',
            'circular economy', 'recycl', 'waste-to-energy',
            'paris agreement', 'cop27', 'cop28', 'cop29',
        ]

        # Require at least ONE core keyword
        return any(kw in text_lower for kw in core_keywords)

    def _is_out_of_scope(self, text_lower: str) -> bool:
        """Block obvious out-of-scope content"""

        out_of_scope_patterns = [
            # IT infrastructure (Kubernetes, AWS, DevOps)
            r'\b(kubernetes|docker|aws|azure|devops|terraform)\b',
            r'\bapi (development|deployment|endpoint)\b',
            r'\bmicroservices?\b',

            # Medicinal/health (unless paired with climate)
            r'\b(medicinal plants?|traditional medicine|ethnobotanical)\b(?!.{0,100}\b(climate|carbon|sustainable)\b)',

            # Pure biodiversity (unless paired with climate)
            r'\b(biodiversity|species conservation|ecosystem)\b(?!.{0,100}\b(climate|carbon|resilience|adaptation)\b)',

            # Generic business/HR
            r'\b(mental health|employee benefits?|workplace)\b',

            # Pure finance (unless climate finance)
            r'\b(stock market|forex|cryptocurrency)\b(?!.{0,100}\b(climate|green|sustainable)\b)',
        ]

        for pattern in out_of_scope_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _is_infrastructure_disruption(self, text_lower: str) -> bool:
        """Detect protests, strikes, outages"""

        disruption_patterns = [
            r'\b(protest|protesters?|demonstration)\b',
            r'\b(strike|strikes|striking)\b',
            r'\b(service outage|power outage|blackout)\b(?!.{0,100}\b(prevention|solution|technology)\b)',
            r'\bextinction rebellion\b',
            r'\bactivists?\s+(block|disrupt|halt)\b',
        ]

        return any(re.search(p, text_lower) for p in disruption_patterns)

    def _is_pure_vaporware(self, text_lower: str) -> bool:
        """Pure announcements with no evidence of work"""

        vaporware_indicators = [
            r'\bunveils? (revolutionary|breakthrough|game-changing)',
            r'\bpromises? to (revolutionize|transform)',
            r'\bcould potentially',
            r'\btheoretical breakthrough',
        ]

        has_vaporware = any(re.search(p, text_lower) for p in vaporware_indicators)

        if not has_vaporware:
            return False

        # Allow if has ANY evidence of real work
        evidence_patterns = [
            r'\b(deployed|operational|pilot|demonstration|tested|validated|achieved)\b',
            r'\b\d+\s*(kw|mw|gw)\b',
            r'\bperformance (data|results)\b',
        ]

        has_evidence = any(re.search(p, text_lower) for p in evidence_patterns)

        return has_vaporware and not has_evidence

    def _is_future_only(self, text_lower: str) -> bool:
        """Proposals/plans without current work"""

        future_patterns = [
            r'\b(proposes?|plans? to|will deploy|aiming to)\s+.{0,50}\b(by 20\d{2}|next year|in 20\d{2})\b',
            r'\bexpects? to (deploy|launch)\s+.{0,50}\bin 20\d{2}\b',
            r'\btargets? .{0,50}\bby 20\d{2}\b',
        ]

        has_future = any(re.search(p, text_lower) for p in future_patterns)

        if not has_future:
            return False

        # Allow if has CURRENT work mentioned
        current_work = [
            r'\b(currently|already|now|today)\s+(operational|deployed|testing|piloting)\b',
            r'\bhas (deployed|installed|achieved|demonstrated)\b',
            r'\boperational since \d{4}\b',
            r'\bpilot (running|operating|generating)\b',
        ]

        has_current = any(re.search(p, text_lower) for p in current_work)

        return has_future and not has_current

    def _has_any_tech_signal(self, text_lower: str) -> bool:
        """RELAXED: ANY mention of deployment, pilots, or research"""

        tech_signals = [
            # Deployment
            r'\b(deployed|operational|installed|commissioned|generating|producing)\b',
            r'\b\d+\s*(kw|mw|gw)\b',
            r'\bmarket share\b',

            # Pilots (RELAXED - just mention is enough)
            r'\bpilot (project|plant|facility|program|demonstration)\b',
            r'\bdemonstration (project|plant|facility)\b',
            r'\bfield (test|trial)\b',

            # Research/validation (RELAXED)
            r'\b(validated|proven|tested|achieved|demonstrated)\b',
            r'\bperformance (data|results|metrics)\b',
            r'\breal-world (data|results|testing)\b',
            r'\bstudy (shows?|finds?|demonstrates?)\s+.{0,50}\b(efficiency|performance|improvement)\b',
        ]

        return any(re.search(p, text_lower) for p in tech_signals)

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
