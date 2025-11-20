"""
Sustainability Tech Innovation Pre-Filter v1.1 - OPTION D: Minimal Filtering

STRATEGY: Trust the oracle - minimal prefiltering, let oracle do the work
- ONLY block obvious out-of-scope (IT, medicine, finance, airline pilots)
- ONLY block infrastructure disruption (protests, strikes)
- PASS all climate/energy articles (no evidence requirement)
- Target: 30-50% pass rate on climate-relevant articles

Philosophy: Prefilter's job is noise reduction, not quality filtering.
Let the oracle (Gemini Flash) score articles and filter low-quality ones.
"""

import re
from typing import Dict
from filters.base_prefilter import BasePreFilter


class SustainabilityTechInnovationPreFilterV1_OptionD(BasePreFilter):
    """Pre-filter Option D: Minimal - Trust the oracle"""

    VERSION = "1.1-option-d"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_tech_innovation"
        self.version = "1.1-option-d"

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
        """Check if article mentions ANY climate/energy keyword"""

        # Broad climate/energy keywords
        climate_energy_keywords = [
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

            # Carbon & climate
            r'\b(carbon capture|ccs|ccus|direct air capture)\b',
            r'\b(climate tech|climate innovation|climate solution)\b',
            r'\b(decarbonization|net.?zero|carbon.?neutral)\b',
            r'\b(greenhouse gas|ghg emission|co2 reduction)\b',

            # Scale indicators
            r'\b\d+\s*(kw|mw|gw|kwh|mwh|gwh)\b',

            # Grid & efficiency
            r'\b(smart grid|grid.?scale|energy efficiency)\b',
            r'\b(demand response|virtual power plant)\b',

            # Sustainability concepts (climate-focused)
            r'\b(sustainable (energy|power|transport|fuel))\b',
            r'\b(circular economy).{0,100}\b(material|recycling|reuse)\b',
        ]

        return any(re.search(kw, text_lower) for kw in climate_energy_keywords)

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
