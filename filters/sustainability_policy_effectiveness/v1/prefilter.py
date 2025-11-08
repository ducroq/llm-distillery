"""
Sustainability Policy Effectiveness Pre-Filter v1.0

Blocks policy announcements without outcomes, pure advocacy.
Passes policies with measurable before/after data.
"""

import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter


class PolicyEffectivenessPreFilter(BasePreFilter):
    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_policy_effectiveness"
        self.version = "1.0"

    def should_block(self, article: Dict) -> tuple[bool, Optional[str]]:
        text_lower = self._get_combined_text(article).lower()

        # Block: Future-only policy announcements
        if re.search(r'\b(plans? to introduce|will implement|proposes?)\b', text_lower):
            if not self._has_outcome_data(text_lower):
                return (True, "policy_announcement_no_outcomes")

        # Block: Pure advocacy
        if re.search(r'\b(should|must|ought to) (pass|implement|adopt)\b', text_lower):
            if not self._has_outcome_data(text_lower):
                return (True, "pure_advocacy")

        return (False, None)

    def _has_outcome_data(self, text_lower: str) -> bool:
        patterns = [
            r'\bemissions (fell|dropped|declined) \d+%\b',
            r'\b(renewable|ev|solar|wind) (deployment|sales?) (grew|increased|rose)\b',
            r'\bafter (policy|law|regulation)\b.{0,100}\b(fell|grew|increased)\b',
            r'\bfrom \d+% to \d+%\b',
            r'\b\d+ (countries|states|cities) (adopted|implementing)\b',
        ]
        return any(re.search(p, text_lower) for p in patterns)

    def _get_combined_text(self, article: Dict) -> str:
        return ' '.join([article.get('title',''), article.get('description',''), article.get('content','')[:2000]])
