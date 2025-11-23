"""
Sustainability Tech Innovation Pre-Filter v1.1 - OPTION C: Strict

STRATEGY: Maximize quality, minimize false positives
- Strict climate/energy tech focus
- Require STRONG evidence with specifics (numbers, duration, metrics)
- Target: 3-5% pass rate

Expected FP rate: <5% (quality over quantity)
"""

import re
from typing import Dict, List
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV1_OptionC(BasePreFilter):
    """Pre-filter Option C: Strict - Quality over quantity"""

    VERSION = "1.1-option-c"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "1.1-option-c"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Not climate/energy tech
        if not self._is_climate_energy_tech(text_lower):
            return (False, "not_climate_energy_tech")

        # BLOCK: Out-of-scope (strict check)
        if self._is_out_of_scope_strict(text_lower):
            return (False, "out_of_scope")

        # BLOCK: Infrastructure disruption
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # BLOCK: Vaporware or future-only
        if self._is_vaporware_or_future_only(text_lower):
            return (False, "vaporware_or_future_only")

        # STRICT: Require STRONG evidence with specifics
        if not self._has_strong_evidence_with_specifics(text_lower):
            return (False, "no_strong_evidence")

        # PASS: Climate/energy tech with strong evidence
        return (True, "passed")

    def _is_climate_energy_tech(self, text_lower: str) -> bool:
        """STRICT: Must be about climate tech or clean energy tech"""

        # Core technologies
        tech_keywords = [
            # Renewable energy tech
            'solar', 'wind', 'geothermal', 'hydro', 'tidal', 'wave energy',
            'renewable energy', 'clean energy',

            # Storage
            'battery', 'energy storage', 'grid storage',
            'hydrogen', 'fuel cell', 'ammonia',

            # Electric transport
            'electric vehicle', ' ev ', 'bev', 'phev',

            # Efficiency tech
            'heat pump', 'energy efficiency', 'smart grid',

            # Carbon removal
            'carbon capture', 'direct air capture', 'dac',
            'carbon neutral', 'net-zero', 'decarboniz',
        ]

        # Require tech keyword PLUS climate context
        has_tech = any(kw in text_lower for kw in tech_keywords)

        climate_context = [
            'climate', 'carbon', 'emission', 'greenhouse',
            'renewable', 'sustainable', 'clean energy',
            'paris agreement', 'cop27', 'cop28', 'cop29',
        ]

        has_climate_context = any(kw in text_lower for kw in climate_context)

        # Require BOTH tech AND climate context
        return has_tech and has_climate_context

    def _is_out_of_scope_strict(self, text_lower: str) -> bool:
        """STRICT: Block anything that's not climate tech"""

        out_of_scope_patterns = [
            # IT/Software (unless specifically for climate tech)
            r'\b(kubernetes|docker|devops|api|microservices?)\b(?!.{0,200}\b(solar|wind|renewable|energy|climate)\b)',

            # Generic software
            r'\b(python|javascript|react|typescript|programming)\b(?!.{0,200}\b(solar|wind|renewable|energy|climate)\b)',

            # Healthcare/medicine
            r'\b(medicinal|pharmaceutical|clinical|medical|disease|health)\b(?!.{0,200}\b(climate|sustainable)\b)',

            # Biodiversity/ecology (unless climate adaptation)
            r'\b(species|biodiversity|ecosystem|conservation)\b(?!.{0,200}\b(climate adaptation|climate resilience|carbon)\b)',

            # Business/HR
            r'\b(employee|workplace|benefits?|hiring|interview)\b',

            # Finance (unless climate finance)
            r'\b(trading|investment|portfolio)\b(?!.{0,200}\b(climate|green bond|renewable|sustainable)\b)',

            # Agriculture (unless sustainable ag tech)
            r'\b(crop|farming|agriculture)\b(?!.{0,200}\b(precision|sustainable|climate-smart|vertical farm)\b)',
        ]

        return any(re.search(p, text_lower) for p in out_of_scope_patterns)

    def _is_infrastructure_disruption(self, text_lower: str) -> bool:
        """Protests, strikes, outages"""

        disruption_patterns = [
            r'\b(protest|strike|disruption|outage|blackout)\b',
            r'\bextinction rebellion\b',
            r'\bactivists?\s+(block|chain|glue)\b',
        ]

        return any(re.search(p, text_lower) for p in disruption_patterns)

    def _is_vaporware_or_future_only(self, text_lower: str) -> bool:
        """Block vaporware, announcements, future proposals"""

        vaporware_future_patterns = [
            # Vaporware
            r'\bunveils? (revolutionary|breakthrough|game-changing)',
            r'\bpromises? to (revolutionize|transform)',
            r'\bcould potentially',

            # Future proposals (strict - block ANY future language without operational evidence)
            r'\b(proposes?|plans?|will|expects? to|aims? to)\s+.{0,50}\b(deploy|build|install|launch)\b',
            r'\bplans? for\s+\d+\s*(mw|gw)\b',
            r'\bdelivery (in|by) 20\d{2}\b',
            r'\bexpected to (begin|start|launch)\b',
        ]

        has_vaporware_future = any(re.search(p, text_lower) for p in vaporware_future_patterns)

        if not has_vaporware_future:
            return False

        # Allow ONLY if has OPERATIONAL evidence
        operational_override = [
            r'\boperational since \d{4}\b',
            r'\bcurrently (operational|generating|producing)\s+\d+\s*(mw|gw)\b',
            r'\bhas deployed\s+\d+\s*(mw|gw)\b',
        ]

        has_operational = any(re.search(p, text_lower) for p in operational_override)

        return has_vaporware_future and not has_operational

    def _has_strong_evidence_with_specifics(self, text_lower: str) -> bool:
        """STRICT: Require evidence with numbers/duration/metrics"""

        # 1. Deployment with scale (MUST have numbers)
        deployment_with_scale = [
            r'\b(operational|generating|producing)\s+.{0,30}\b\d+\s*(mw|gw)\b',
            r'\b(deployed|installed)\s+.{0,30}\b\d+\s*(mw|gw)\b',
            r'\b\d+\s*(gw|mw)\s+(capacity|installed|deployed|operational)\b',
            r'\b\d+\s*(gw|mw)\s+.{0,50}\b(generating|producing)\b',
            r'\bmarket share.{0,30}\b\d+%\b',
        ]

        if any(re.search(p, text_lower) for p in deployment_with_scale):
            return True

        # 2. Pilots with data (MUST have numbers AND duration OR performance)
        pilots_with_data = [
            r'\bpilot .{0,100}\b(generated?|produced)\s+\d+\s*(kw|mw|gwh)\b',
            r'\bpilot .{0,100}\b(operational|running)\s+(for\s+)?\d+\s+(months?|years?)\b',
            r'\bdemonstration .{0,100}\b(achieved|showed|demonstrated)\s+\d+%\s+(efficiency|performance)\b',
            r'\bfield test .{0,100}\b\d+\s*(kw|mw)\s+.{0,50}\b(months?|years?)\b',
        ]

        if any(re.search(p, text_lower) for p in pilots_with_data):
            return True

        # 3. Research validation with metrics (MUST have real-world + numbers)
        research_with_metrics = [
            r'\bvalidated .{0,50}\b(on|with)\s+\d+\s+(real|actual)\s+(vehicles?|systems?|installations?)\b',
            r'\breal-world .{0,100}\b(achieved|demonstrated|showed)\s+\d+%\b',
            r'\bfield (test|validation) .{0,100}\bover\s+\d+\s+(months?|years?)\b',
            r'\b(measured|actual|observed)\s+(performance|efficiency) .{0,100}\b\d+\s*(kw|mw|%|gwh)\b',
        ]

        if any(re.search(p, text_lower) for p in research_with_metrics):
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
