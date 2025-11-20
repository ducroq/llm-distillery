"""
Sustainability Tech Innovation Pre-Filter v2.0 - Narrowed Scope + Pilot-Friendly

STRATEGY: Climate/energy focus + pilot-friendly patterns
- REQUIRE climate/energy keywords (not generic sustainability/biodiversity)
- ALLOW pilot-specific language ("pilot project", "demonstration", "field test")
- BLOCK obvious out-of-scope (IT, medicine, finance, airline pilots)
- BLOCK infrastructure disruption (protests, strikes)
- BLOCK generic biodiversity research (unless climate-focused)
- Target: 5-20% pass rate on climate/energy articles

Philosophy: Narrow sustainability scope to climate tech, loosen deployment requirements for pilots.

CHANGELOG v2.0 (2025-11-18):
- NARROWED sustainability scope: Require climate/energy/carbon keywords
- ADDED pilot-specific patterns: "pilot project", "demonstration", "field test", "MW generated"
- BLOCKED generic biodiversity: "medicinal plants", "traditional knowledge" (unless climate-paired)
- Expected: Reduce FP rate from 85.7% → <10%, maintain/improve pass rate

CHANGELOG v1.1 (2025-11-17):
- Switched to Option D (Minimal Filtering)
- 68% pass rate on climate tech articles (vs 16% for v1.0)
- 62% improvement in false negative rate (84 → 32 blocked articles)
"""

import re
from typing import Dict
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV2(BasePreFilter):
    """Pre-filter v2.0: Climate/energy focus + pilot-friendly"""

    VERSION = "2.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "2.0"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        """Minimal filtering - only block obvious out-of-scope"""
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Obvious out-of-scope (IT, medicine, finance, airline pilots)
        if self._is_obvious_out_of_scope(text_lower):
            return (False, "obvious_out_of_scope")

        # BLOCK: Infrastructure disruption (protests, strikes)
        if self._is_infrastructure_disruption(text_lower):
            return (False, "infrastructure_disruption")

        # REQUIRE: At least some climate/energy relevance
        if not self._has_climate_energy_mention(text_lower):
            return (False, "not_climate_energy_related")

        # PASS: Has climate/energy mention and not obviously out-of-scope
        return (True, "passed")

    def _is_obvious_out_of_scope(self, text_lower: str) -> bool:
        """Block ONLY obvious non-climate content"""

        # IT infrastructure (unless paired with climate/energy)
        it_patterns = [
            r'\b(kubernetes|docker|devops|api gateway|microservices)\b(?!.{0,200}\b(solar|wind|renewable|energy|climate|battery|ev)\b)',
            r'\b(github copilot|windows container|cloud migration)\b(?!.{0,200}\b(solar|wind|renewable|energy|climate)\b)',
        ]

        # Medicine/healthcare (unless paired with climate)
        medical_patterns = [
            r'\b(medicinal plants?|pharmaceutical|clinical trial|medical device)\b(?!.{0,200}\b(climate|sustainable|renewable)\b)',
            r'\b(surgery|patient|diagnosis|therapeutic)\b(?!.{0,200}\b(climate adaptation)\b)',
        ]

        # Finance/banking (unless climate finance)
        finance_patterns = [
            r'\b(operational resilience|banking sector|financial institution)\b(?!.{0,200}\b(climate|renewable|green bond|sustainable)\b)',
            r'\b(cryptocurrency|blockchain|defi)\b(?!.{0,200}\b(renewable energy|carbon credit)\b)',
        ]

        # Airline pilots (aviation NOT climate-related)
        aviation_patterns = [
            r'\bpilots? (lose contact|emergency landing|lufthansa|airline)\b',
            r'\b(flight attendant|cockpit|aviation safety)\b(?!.{0,200}\b(sustainable aviation fuel|electric aircraft)\b)',
        ]

        # Pure astronomy/physics (unless climate science)
        science_patterns = [
            r'\b(spectroscopic|quantum process|thermodynamic)\b(?!.{0,200}\b(solar cell|energy conversion|climate)\b)',
            r'\b(ca ii resonance|solar atmosphere|magnetic sensitivity)\b(?!.{0,100}\b(photovoltaic|solar panel)\b)',
        ]

        all_patterns = it_patterns + medical_patterns + finance_patterns + aviation_patterns + science_patterns

        return any(re.search(p, text_lower) for p in all_patterns)

    def _is_infrastructure_disruption(self, text_lower: str) -> bool:
        """Block infrastructure disruption (protests, strikes, export bans)"""
        disruption_patterns = [
            r'\b(protest|strike|labor dispute|walkout)\b',
            r'\b(export (ban|curb|restriction)|trade war|embargo)\b',
            r'\b(supply chain disruption|blocked shipment)\b',
        ]

        return any(re.search(p, text_lower) for p in disruption_patterns)

    def _has_climate_energy_mention(self, text_lower: str) -> bool:
        """Check if article mentions climate/energy keywords (NARROWED SCOPE for v2)"""

        # CORE climate/energy keywords (REQUIRED - narrowed from v1.1)
        core_climate_energy = [
            # Climate & carbon (NEW REQUIREMENT)
            r'\b(climate|carbon|decarboniz|decarbonis)\b',

            # Energy sources
            r'\b(solar|wind|geothermal|hydroelectric|tidal|wave energy)\b',
            r'\b(renewable|clean energy|green energy)\b',
            r'\b(photovoltaic|pv module|solar panel|wind turbine)\b',

            # Storage & EVs
            r'\b(battery|energy storage|bess|lithium.?ion)\b',
            r'\b(electric vehicle|ev charging|ev |e-mobility)\b',
            r'\b(heat pump|thermal storage)\b',

            # Fuels
            r'\b(hydrogen|biofuel|sustainable aviation fuel|green ammonia)\b',
            r'\b(biogas|biomass|bioenergy)\b',

            # Carbon tech
            r'\b(carbon capture|ccs|ccus|direct air capture)\b',
            r'\b(climate tech|climate innovation|climate solution)\b',
            r'\b(net.?zero|carbon.?neutral|carbon.?negative)\b',
            r'\b(greenhouse gas|ghg emission|co2 reduction|emissions reduction)\b',

            # Scale indicators (energy-related)
            r'\b\d+\s*(kw|mw|gw|kwh|mwh|gwh)\b',

            # Grid & efficiency
            r'\b(smart grid|grid.?scale|energy efficiency)\b',
            r'\b(demand response|virtual power plant)\b',

            # Pilot & validation indicators (NEW - allow pilot language)
            r'\b(pilot (project|plant|program|installation|facility))\b',
            r'\b(demonstration (plant|project|facility))\b',
            r'\b(field (test|trial|validation))\b',
            r'\b(real.?world (test|validation|performance))\b',
            r'\b(\d+\s*mw (generated|pilot|demonstration))\b',
        ]

        # Block generic biodiversity unless climate-paired (NEW for v2)
        biodiversity_without_climate = [
            r'\b(medicinal plants?|ethnobotanical|traditional (knowledge|ecological))\b(?!.{0,200}\b(climate|carbon|renewable)\b)',
            r'\b(biodiversity|conservation|ecosystem)\b(?!.{0,200}\b(climate|carbon|sequestration|nature.?based solution)\b)',
        ]

        # Block if generic biodiversity (not climate-focused)
        if any(re.search(p, text_lower) for p in biodiversity_without_climate):
            return False

        return any(re.search(kw, text_lower) for kw in core_climate_energy)

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title, description, and content for analysis"""
        parts = []

        if 'title' in article and article['title']:
            parts.append(article['title'])
        if 'description' in article and article['description']:
            parts.append(article['description'])
        if 'content' in article and article['content']:
            # Truncate content to first 1000 chars (prefilter doesn't need full text)
            parts.append(article['content'][:1000])

        return ' '.join(parts)
