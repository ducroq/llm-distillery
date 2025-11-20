"""AI-Augmented Practice Pre-Filter v1.0

Blocks: AI hype, model benchmarks, funding news, speculation
Passes: Empirical reports of workflow integration, real usage patterns
"""

import re
from typing import Dict
from filters.base_prefilter import BasePreFilter


class AIAugmentedPracticePreFilterV1(BasePreFilter):
    """Pre-filter for empirical AI-augmented cognitive work reports"""

    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "ai_augmented_practice"
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

        # BLOCK: Not AI/LLM related at all
        if not self._is_ai_related(text_lower):
            return (False, "not_ai_related")

        # BLOCK: Model benchmarks and technical specs
        if self._is_model_benchmark(text_lower):
            return (False, "model_benchmark")

        # BLOCK: Funding/business news
        if self._is_business_news(text_lower):
            return (False, "business_news")

        # BLOCK: Speculation/futurism without current practice
        if self._is_speculation(text_lower):
            return (False, "speculation_no_practice")

        # BLOCK: Generic AI overviews without specific usage
        if self._is_generic_overview(text_lower):
            return (False, "generic_overview")

        # PASS: Has practice/workflow evidence
        return (True, "passed")

    def _is_ai_related(self, text_lower: str) -> bool:
        """Check if article is about AI/LLMs/GenAI"""
        keywords = [
            'llm', 'large language model', 'gpt', 'chatgpt', 'claude', 'gemini',
            'copilot', 'cursor', 'generative ai', 'gen ai', 'ai assistant',
            'prompt', 'prompting', 'rag', 'retrieval augmented',
            'language model', 'transformer', 'openai', 'anthropic',
        ]
        return any(kw in text_lower for kw in keywords)

    def _is_model_benchmark(self, text_lower: str) -> bool:
        """Block model benchmarks, technical specs, capability demos"""
        benchmark_patterns = [
            r'\b(benchmark|mmlu|humaneval|gsm8k|hellaswag)\b',
            r'\bparameters?\b.{0,20}\b(billion|trillion)\b',
            r'\bmodel (release|announcement|unveiled)\b',
            r'\b(beats|outperforms|surpasses) (gpt|claude|gemini)\b',
            r'\bstate.of.the.art\b',
            r'\b(accuracy|f1|bleu) score\b',
        ]

        if any(re.search(p, text_lower) for p in benchmark_patterns):
            # Override if there's workflow discussion
            if self._has_workflow_evidence(text_lower):
                return False
            return True
        return False

    def _is_business_news(self, text_lower: str) -> bool:
        """Block funding, acquisitions, market news"""
        business_patterns = [
            r'\$\d+[.,]?\d*\s*(billion|million)\s*(funding|raised|valuation)\b',
            r'\b(series [a-e]|seed round|ipo)\b',
            r'\b(acquisition|acquires|acquired|merger)\b',
            r'\bmarket (cap|share|size)\b',
            r'\b(revenue|earnings|profit)\b',
        ]
        return any(re.search(p, text_lower) for p in business_patterns)

    def _is_speculation(self, text_lower: str) -> bool:
        """Block futurism/speculation without current practice"""
        speculation_patterns = [
            r'\bwill (transform|revolutionize|disrupt|change)\b',
            r'\b(future of|next generation of|coming years)\b',
            r'\b(could|might|may) (replace|eliminate|automate)\b',
            r'\bagi\b',
            r'\bsentient|conscious\b',
        ]

        if any(re.search(p, text_lower) for p in speculation_patterns):
            # Override if there's current practice evidence
            if self._has_workflow_evidence(text_lower):
                return False
            return True
        return False

    def _is_generic_overview(self, text_lower: str) -> bool:
        """Block generic AI explainers without specific usage"""
        generic_patterns = [
            r'\bwhat is (chatgpt|claude|llm|generative ai)\b',
            r'\bhow (chatgpt|llm|ai) works?\b',
            r'\bintroduction to (chatgpt|llm|ai)\b',
            r'\b(beginner|guide|tutorial|overview)\b',
        ]

        if any(re.search(p, text_lower) for p in generic_patterns):
            # Override if there's specific practice evidence
            if self._has_workflow_evidence(text_lower):
                return False
            return True
        return False

    def _has_workflow_evidence(self, text_lower: str) -> bool:
        """Check for evidence of actual workflow integration/practice"""
        workflow_patterns = [
            # Workflow integration
            r'\b(integrated|adopted|implemented) (into|in our|in my)\b',
            r'\bworkflow\b',
            r'\b(use|using|used) (chatgpt|claude|copilot|ai) (for|to)\b',
            r'\b(our team|we|i) (use|integrate|rely on)\b',

            # Practice reports
            r'\b(my experience|our experience) (with|using)\b',
            r'\b\d+ months? (of using|with|experience)\b',
            r'\bcase study\b',
            r'\bin practice\b',
            r'\bhow (we|i) (use|leverage)\b',

            # Empirical evidence
            r'\b(data shows?|study finds?|survey|measured)\b',
            r'\b(before|after) (adopting|using|implementing)\b',
            r'\b\d+% (faster|more|improvement|reduction)\b',
            r'\btime saved\b',

            # Specific tasks
            r'\b(code review|meeting notes|writing|research|analysis)\b.{0,30}\b(with|using) (ai|llm|chatgpt)\b',
            r'\b(prompt|prompting|prompts?) (for|to|that)\b',

            # Validation/trust patterns
            r'\b(verify|validation|check|review) (ai|llm|output)\b',
            r'\b(hallucination|accuracy|reliability|trust)\b',
            r'\bfailure (mode|case|pattern)\b',
        ]

        return any(re.search(p, text_lower) for p in workflow_patterns)

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title + description + content for analysis"""
        return ' '.join([
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', '')[:2000]
        ])
