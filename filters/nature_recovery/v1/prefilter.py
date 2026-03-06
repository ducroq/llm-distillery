"""Nature Recovery Pre-Filter v1.0 - Passes recovery/nature stories, blocks non-environmental content"""
import re
from typing import Dict, Tuple

from filters.common.base_prefilter import BasePreFilter


class NatureRecoveryPreFilterV1(BasePreFilter):
    VERSION = "1.0"

    def __init__(self):
        super().__init__(use_commerce_prefilter=False)
        self.filter_name = "nature_recovery"
        self.version = "1.0"

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to oracle for scoring.

        Returns:
            (should_score, reason)
            - (True, "passed"): Send to oracle
            - (False, reason): Block from oracle
        """
        title = article.get('title', '')
        content = article.get('content', '') or article.get('text', '')
        text_lower = (title + ' ' + content[:self.MAX_PREFILTER_CONTENT]).lower()

        # BLOCK: Not nature/environment related at all
        if not self._is_nature_related(text_lower):
            return (False, "not_nature_topic")

        # BLOCK: Pure disaster/decline without any recovery framing
        doom_pattern = re.search(
            r'\b(extinction|collapse|dying|destroyed|devastating|catastroph|irreversible)\b',
            text_lower
        )
        if doom_pattern:
            recovery_pattern = re.search(
                r'\b(recover|restor|rebound|return|improv|increas|grow|thriv|heal|reintroduc|rewild)\b',
                text_lower
            )
            if not recovery_pattern:
                return (False, "disaster_no_recovery")

        return (True, "passed")

    def _is_nature_related(self, text_lower: str) -> bool:
        """Check if article is about nature/ecosystem/environmental issues (PERMISSIVE)."""
        keywords = [
            'ecosystem', 'biodiversity', 'habitat', 'deforestation', 'reforestation',
            'coral', 'reef', 'ocean', 'marine', 'wildlife', 'species', 'extinction',
            'pollution', 'air quality', 'water quality', 'environment', 'climate',
            'carbon', 'wetland', 'mangrove', 'conservation', 'restoration', 'recovery',
            'rewilding', 'endangered', 'protected area', 'national park', 'nature reserve',
            'emission', 'ozone', 'deforestation', 'afforestation', 'fish stock',
        ]
        return any(kw in text_lower for kw in keywords)
