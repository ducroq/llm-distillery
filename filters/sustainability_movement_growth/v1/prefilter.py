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
        """Check if article is about climate movement/activism/behavior change"""
        patterns = [
            r'\bclimate (strike|protest|march|rally|movement|activism)\b',
            r'\b(greta|thunberg|extinction rebellion|fridays for future)\b',
            r'\bdivestment\b.{0,50}\b(fossil|coal|oil)\b',
            r'\b(climate|sustainability|green) (movement|activism|campaign)\b',
            r'\bev (sales|adoption|market share)\b', r'\bplant-?based (adoption|sales|market)\b',
            r'\b(solar|renewable).{0,30}\badoption\b', r'\bcarbon (footprint|neutral|offset)\b',
            r'\bconsumer (behavior|choice).{0,30}\b(climate|sustainability)\b',
            r'\bclimate (action|policy).{0,30}\bvot(e|ing)\b', r'\byouth (climate|activism)\b',
        ]
        return any(re.search(pattern, text_lower) for pattern in patterns)

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
