"""Movement Growth Pre-Filter v1.0 - Passes growth stories, blocks static news"""
import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter

class MovementGrowthPreFilter(BasePreFilter):
    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_movement_growth"
        self.version = "1.0"

    def should_block(self, article: Dict) -> tuple[bool, Optional[str]]:
        text_lower = self._get_combined_text(article).lower()

        # Pass: Growth indicators
        growth_patterns = [
            r'\b(doubled|tripled|grew) \d+%',
            r'\bmillions? (marched|protested|signed)',
            r'\b\$\d+[\.,]?\d*\s*(billion|trillion) divested',
            r'\bmarket share (grew|increased|rose) to \d+%',
            r'\byoy growth \d+%',
        ]
        if any(re.search(p, text_lower) for p in growth_patterns):
            return (False, None)

        # Block: Static or declining
        if re.search(r'\b(same|stable|unchanged|declined|fell)\b', text_lower):
            return (True, "no_growth_signal")

        return (False, None)

    def _get_combined_text(self, article: Dict) -> str:
        return ' '.join([article.get('title',''), article.get('description',''), article.get('content','')[:2000]])
