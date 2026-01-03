"""
Sustainability Technology Pre-Filter v1.0

This module defines a pre-filter for evaluating articles related to sustainability technology.
Fast keyword-based filtering before model inference.
"""

import re
from typing import Dict, List, Optional, Tuple
from filters.common.base_prefilter import BasePreFilter


class SustainabilityTechnologyPreFilterV1(BasePreFilter):
    """
    Pre-filter to evaluate articles on sustainability technology.
    """
    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_technology_v1"
        self.version = "1.0"

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for scoring.

        Returns:
            (should_score, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        text = self._get_combined_clean_text(article)

        # BLOCK: Not climate/sustainability related at all
        if not self._is_sustainability_related(text):
            return (False, "not_sustainability_topic")

        # PASS: Has sustainability evidence AND passed all other checks
        return (True, "passed")

    def _is_sustainability_related(self, text: str) -> bool:
        """Check if article is about climate/sustainability/clean energy (WIDEST NET)"""

        keywords = [
            # === ENGLISH ===
            # 1. Climate Core & Mitigation
            'climate', 'carbon', 'emission', 'greenhouse', 'warming',
            'net-zero', 'net zero', 'carbon neutral', 'decarboniz',
            'carbon capture', 'direct air capture', 'co2',
            'paris agreement', 'cop', 'unfccc', 'deforestation',

            # 2. Energy Transition & Materials (Technical)
            'renewable', 'solar', 'wind', 'geothermal', 'hydro', 'nuclear',
            'battery', 'energy storage', 'grid storage', 'hydrogen', 'electrif',
            'electric vehicle', ' ev ', 'bev', 'phev',
            'fossil fuel', 'coal', 'oil', 'gas',
            'critical material', 'lithium', 'copper', 'steel', 'plastic', 'mining',

            # 3. Resource Management & Circularity
            'sustainab', 'green energy', 'clean energy', 'circular economy',
            'recycl', 'waste reduction', 'water conservation', 'desalination',

            # 4. Land Use & Agriculture
            'regenerative', 'agrivoltaic', 'vertical farm', 'sustainable food',
            'biodiversity', 'reforestation',

            # 5. Policy, Finance & Systemic Context (The Wider Net)
            'esg', 'green bond', 'disclosure', 'sustainability report',
            'just transition', 'labor rights', 'indigenous', 'human rights',
            'infrastructure', 'innovation', 'regulations', 'mandate',
            'investments', 'subsidies', 'resilience', 'adaptation',
            'climate risk', 'stranded asset', 'supply chain',

            # === DUTCH (NL) ===
            'klimaat', 'koolstof', 'uitstoot', 'broeikasgas', 'opwarming',
            'duurzaam', 'hernieuwbaar', 'zonne-energie', 'zonnepane', 'windenergie',
            'elektrisch', 'elektrische auto', 'waterstof', 'batterij',
            'kernenergie', 'kerncentrale', 'circulair', 'recycl',
            'energie-transitie', 'energietransitie', 'groene energie',

            # === GERMAN (DE) ===
            'klima', 'kohlenstoff', 'treibhausgas', 'erwärmung', 'emissionen',
            'nachhaltig', 'erneuerbar', 'solarenergie', 'windkraft', 'wasserstoff',
            'elektroauto', 'elektrofahrzeug', 'elektromobil', 'batterie',
            'kernkraft', 'atomkraft', 'kreislaufwirtschaft', 'energiewende',
            'photovoltaik', 'wärmepumpe', 'grüne energie',

            # === FRENCH (FR) ===
            'climat', 'carbone', 'émission', 'gaz à effet de serre', 'réchauffement',
            'durable', 'renouvelable', 'solaire', 'éolien', 'hydrogène',
            'véhicule électrique', 'voiture électrique', 'batterie',
            'nucléaire', 'économie circulaire', 'transition énergétique',
            'photovoltaïque', 'énergie verte', 'décarbonation',

            # === SPANISH (ES) ===
            'clima', 'carbono', 'emisión', 'efecto invernadero', 'calentamiento',
            'sostenible', 'renovable', 'solar', 'eólica', 'hidrógeno',
            'vehículo eléctrico', 'coche eléctrico', 'batería',
            'nuclear', 'economía circular', 'transición energética',
            'fotovoltaica', 'energía verde', 'descarbonización',

            # === PORTUGUESE (PT) ===
            'clima', 'carbono', 'emissão', 'efeito estufa', 'aquecimento',
            'sustentável', 'renovável', 'solar', 'eólica', 'hidrogénio', 'hidrogênio',
            'veículo elétrico', 'carro elétrico', 'bateria',
            'nuclear', 'economia circular', 'transição energética',
            'fotovoltaica', 'energia verde', 'descarbonização',

            # === ITALIAN (IT) ===
            'clima', 'carbonio', 'emissione', 'effetto serra', 'riscaldamento',
            'sostenibile', 'rinnovabile', 'solare', 'eolico', 'idrogeno',
            'veicolo elettrico', 'auto elettrica', 'batteria',
            'nucleare', 'economia circolare', 'transizione energetica',
            'fotovoltaico', 'energia verde', 'decarbonizzazione',

            # === CHINESE (ZH) ===
            '气候', '碳', '排放', '温室气体', '变暖',
            '可持续', '可再生', '太阳能', '风能', '氢能',
            '电动汽车', '电动车', '电池', '新能源',
            '核能', '循环经济', '能源转型',
            '光伏', '绿色能源', '脱碳',
        ]

        # Very permissive - just needs ONE keyword mention
        return any(kw in text for kw in keywords)
