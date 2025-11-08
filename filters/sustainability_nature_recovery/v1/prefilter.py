"""Nature Recovery Pre-Filter v1.0 - Passes recovery stories, blocks disaster news"""
import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter

class NatureRecoveryPreFilter(BasePreFilter):
    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_nature_recovery"
        self.version = "1.0"

    def should_block(self, article: Dict) -> tuple[bool, Optional[str]]:
        text_lower = self._get_combined_text(article).lower()

        # Block: Pure disaster/decline news
        if re.search(r'\b(extinction|collapse|dying|destroyed)\b', text_lower):
            if not re.search(r'\b(recovery|restored|improving|rebounding)\b', text_lower):
                return (True, "disaster_no_recovery")

        return (False, None)

    def _get_combined_text(self, article: Dict) -> str:
        return ' '.join([article.get('title',''), article.get('description',''), article.get('content','')[:2000]])
