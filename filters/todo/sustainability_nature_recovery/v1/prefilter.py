"""Nature Recovery Pre-Filter v1.0 - Passes recovery stories, blocks disaster news"""
import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter

class NatureRecoveryPreFilterV1(BasePreFilter):
    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_nature_recovery"
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

        # BLOCK: Not climate/sustainability/nature related
        if not self._is_sustainability_related(text_lower):
            return (False, "not_sustainability_topic")

        # Block: Pure disaster/decline news
        if re.search(r'\b(extinction|collapse|dying|destroyed)\b', text_lower):
            if not re.search(r'\b(recovery|restored|improving|rebounding)\b', text_lower):
                return (False, "disaster_no_recovery")

        return (True, "passed")

    def _is_sustainability_related(self, text_lower: str) -> bool:
        """Check if article is about nature/ecosystem/environmental issues (PERMISSIVE)"""
        keywords = [
            'ecosystem', 'biodiversity', 'habitat', 'deforestation', 'reforestation',
            'coral', 'reef', 'ocean', 'marine', 'wildlife', 'species', 'extinction',
            'pollution', 'air quality', 'water quality', 'environment', 'climate',
            'carbon', 'wetland', 'mangrove', 'conservation', 'restoration', 'recovery',
        ]
        return any(kw in text_lower for kw in keywords)

    def _get_combined_text(self, article: Dict) -> str:
        return ' '.join([article.get('title',''), article.get('description',''), article.get('content','')[:2000]])
