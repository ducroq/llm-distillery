"""
Sustainability Technology Pre-Filter v1.0
This module defines a pre-filter for evaluating articles related to sustainability technology.

"""

import re
from typing import Dict, List, Optional, Tuple
from filters.base_prefilter import BasePreFilter


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
        
        # # BLOCK: Obvious off-topic content
        # if self._is_obvious_off_topic(text):
        #     return (False, "obvious_off_topic")        

        # PASS: Has sustainability evidence AND passed all other checks
        return (True, "passed")

    def _is_sustainability_related(self, text: str) -> bool:
        """Check if article is about climate/sustainability/clean energy (WIDEST NET)"""

        keywords = [
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
        ]

        # Very permissive - just needs ONE keyword mention
        return any(kw in text for kw in keywords)    


    # def _is_obvious_off_topic(self, text: str) -> bool:
    #     """
    #     Block obvious off-topic content that accidentally matches sustainability keywords.

    #     Conservative blocking - only blocks clear cases where article is definitely NOT
    #     about sustainability technology despite keyword matches (e.g., 'oil' in 'turmoil').

    #     Examples blocked:
    #     - Sports news (soccer, basketball, etc.)
    #     - Celebrity gossip
    #     - Personal lifestyle (weddings, fashion)
    #     - Entertainment/TV/movies (unless about sustainability themes)

    #     Strategy: Require 2+ negative keyword matches to block (conservative threshold).
    #     Single match might be coincidental, but 2+ suggests clear off-topic content.

    #     Returns:
    #         True if article should be blocked (obvious off-topic)
    #         False if article might be relevant (let oracle decide)
    #     """

    #     # Sports keywords (very specific to avoid false positives)
    #     sports_keywords = [
    #         # Team sports
    #         'soccer', 'football match', 'basketball', 'baseball', 'hockey',
    #         'premier league', 'champions league', 'uefa', 'fifa',
    #         'nfl', 'nba', 'nhl', 'mlb',
    #         # Individual sports
    #         'tennis', 'golf tournament', 'boxing', 'wrestling',
    #         # Sports-specific terms
    #         'goal scorer', 'touchdown', 'home run', 'slam dunk',
    #         'penalty kick', 'hat trick', 'playoff', 'championship game',
    #     ]

    #     # Entertainment & Celebrity (very specific names/shows)
    #     entertainment_keywords = [
    #         # Celebrity names (only obvious ones that never relate to sustainability)
    #         'kardashian', 'baldwin', 'bieber', 'swift', 'beyonce',
    #         'real housewives', 'bachelor', 'bachelorette',
    #         # Entertainment-specific
    #         'box office', 'red carpet', 'grammy', 'oscar', 'emmy',
    #         'reality show', 'sitcom', 'season finale',
    #         'celebrity feud', 'celebrity split', 'celebrity wedding',
    #     ]

    #     # Personal lifestyle (non-sustainability)
    #     lifestyle_keywords = [
    #         'wedding dress', 'bridal', 'engagement ring',
    #         'makeup tutorial', 'beauty tips', 'hairstyle',
    #         'dating advice', 'relationship tips',
    #         'horoscope', 'astrology',
    #         'lottery', 'jackpot winner',
    #     ]

    #     # Combine all negative keywords
    #     negative_keywords = sports_keywords + entertainment_keywords + lifestyle_keywords

    #     # Count how many negative keywords appear
    #     negative_matches = sum(1 for kw in negative_keywords if kw in text)

    #     # Block if 2+ negative keywords (conservative threshold)
    #     # Single match might be coincidental, but 2+ suggests clear off-topic content
    #     return negative_matches >= 2