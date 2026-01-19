"""Resilience Pre-Filter v1.0 - Passes recovery/response stories, blocks pure doom"""
import re
from typing import Dict, Optional
from filters.base_prefilter import BasePreFilter


class ResiliencePreFilterV1(BasePreFilter):
    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "resilience"
        self.version = "1.0"

        # Response/recovery language signals
        self.response_signals = [
            'recovery', 'recovered', 'recovering',
            'rebuild', 'rebuilt', 'rebuilding',
            'restore', 'restored', 'restoring', 'restoration',
            'reform', 'reformed', 'reforming',
            'lesson', 'lessons learned',
            'adapt', 'adapted', 'adapting', 'adaptation',
            'respond', 'responded', 'response',
            'address', 'addressed', 'addressing',
            'solve', 'solved', 'solving', 'solution',
            'fix', 'fixed', 'fixing',
            'improve', 'improved', 'improving', 'improvement',
            'bounce back', 'bounced back',
            'turnaround', 'turned around',
        ]

        # Temporal signals (after adversity)
        self.temporal_signals = [
            'after the', 'in the wake of', 'following the',
            'since the', 'years after', 'months after',
            'in response to', 'lessons from',
            'post-crisis', 'post-disaster',
        ]

        # Institutional learning signals
        self.learning_signals = [
            'new protocol', 'new policy', 'policy change',
            'commission', 'inquiry', 'investigation found',
            'review found', 'report recommends', 'audit',
            'best practice', 'case study', 'model for',
        ]

        # Pure doom signals (block if no response signal)
        self.doom_signals = [
            'threatens', 'threatens to', 'could cause', 'may cause',
            'warns of', 'warning of', 'fears of', 'fear of',
            'bracing for', 'preparing for the worst',
        ]

    def should_label(self, article: Dict) -> tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        text_lower = self._get_combined_text(article).lower()

        # Must have at least one response/recovery signal
        has_response = any(sig in text_lower for sig in self.response_signals)
        has_temporal = any(sig in text_lower for sig in self.temporal_signals)
        has_learning = any(sig in text_lower for sig in self.learning_signals)

        if not (has_response or has_temporal or has_learning):
            return (False, "no_response_signal")

        # Block pure doom (doom signals without response signals)
        has_doom = any(sig in text_lower for sig in self.doom_signals)
        if has_doom and not has_response:
            return (False, "pure_doom")

        # Block product launches / PR without adversity context
        if self._is_pure_pr(text_lower):
            return (False, "pr_no_adversity")

        return (True, "passed")

    def _is_pure_pr(self, text_lower: str) -> bool:
        """Check if article is pure PR/announcement without adversity baseline"""
        pr_signals = ['launches', 'announces', 'unveils', 'introduces', 'celebrates']
        adversity_signals = ['crisis', 'disaster', 'failure', 'collapse', 'outbreak',
                            'accident', 'incident', 'breach', 'scandal', 'problem']

        has_pr = any(sig in text_lower for sig in pr_signals)
        has_adversity = any(sig in text_lower for sig in adversity_signals)

        return has_pr and not has_adversity

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title, description, and content for analysis"""
        return ' '.join([
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', '')[:2000]
        ])
