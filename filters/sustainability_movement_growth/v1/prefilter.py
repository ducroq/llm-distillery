"""Movement Growth Pre-Filter v1.0 - Passes growth stories, blocks static news"""
import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter

class MovementGrowthPreFilterV1(BasePreFilter):
    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_movement_growth"
        self.version = "1.0"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        text_lower = self._get_combined_text(article).lower()

        # BLOCK: Not climate/sustainability related
        if not self._is_sustainability_related(text_lower):
            return (False, "not_sustainability_topic")

        # Block: Static or declining participation
        if re.search(r'\b(membership (remains?|stable)|same number of|unchanged participation)\b', text_lower):
            if not self._has_growth_signals(text_lower):
                return (False, "static_membership")

        # Block: Pure activism without growth metrics
        if re.search(r'\b(protest|march|rally|demonstration)\b', text_lower):
            if not self._has_growth_signals(text_lower):
                return (False, "activism_no_growth_data")

        return (True, "passed")

    def _is_sustainability_related(self, text_lower: str) -> bool:
        """Check if article is about climate movement/activism/behavior change (PERMISSIVE)"""
        keywords = [
            'climate', 'sustainability', 'divestment', 'fossil fuel', 'coal', 'oil',
            'ev ', 'electric vehicle', 'plant-based', 'solar', 'renewable', 'carbon',
            'greta', 'thunberg', 'extinction rebellion', 'protest', 'activism',
            'movement', 'strike', 'march', 'rally', 'campaign', 'green',
        ]
        return any(kw in text_lower for kw in keywords)

    def _has_growth_signals(self, text_lower: str) -> bool:
        """Check for growth/acceleration indicators"""
        growth_patterns = [
            r'\b(doubled|tripled|quadrupled)\b',
            r'\bgrew \d+%',
            r'\bincreased (from|to) \d+',
            r'\bmillions? (more|joined|mobilized)',
            r'\b\$\d+[\.,]?\d*\s*(billion|trillion) divested',
            r'\bmarket share (grew|increased|rose) to \d+%',
            r'\byoy (growth|increase) \d+%',
            r'\bfaster than (last year|previous)',
        ]
        return any(re.search(p, text_lower) for p in growth_patterns)

    def _get_combined_text(self, article: Dict) -> str:
        return ' '.join([article.get('title',''), article.get('description',''), article.get('content','')[:2000]])
