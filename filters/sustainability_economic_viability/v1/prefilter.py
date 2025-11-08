"""
Sustainability Economic Viability Pre-Filter v1.0

Blocks pure advocacy/opinions without economic data.
Passes articles with cost analysis, profitability data, or economic impact.

Expected pass rate: ~50-60% (more economic news than deployment news)
"""

import re
from typing import Dict, List, Optional
from filters.base_prefilter import BasePreFilter


class EconomicViabilityPreFilterV1(BasePreFilter):
    """Pre-filter for economic analysis of climate solutions"""

    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self.filter_name = "sustainability_economic_viability"
        self.version = "1.0"

    def should_label(self, article: Dict) -> tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        text = self._get_combined_text(article)
        text_lower = text.lower()

        # BLOCK: Not climate/sustainability related at all
        if not self._is_sustainability_related(text_lower):
            return (False, "not_sustainability_topic")

        # BLOCK: Pure advocacy without economic data
        advocacy_patterns = [
            r'\b(moral imperative|ethical duty|must act now)\b(?!.{0,200}\b(cost|economic|investment)\b)',
            r'\bsave the planet\b(?!.{0,200}\b(economic|cost|investment|job)\b)',
            r'\bclimate crisis demands?\b(?!.{0,200}\b(economic|cost)\b)',
        ]

        for pattern in advocacy_patterns:
            if re.search(pattern, text_lower):
                # Check if there's economic data to override
                if not self._has_economic_data(text_lower):
                    return (False, "pure_advocacy_no_economics")

        # BLOCK: Opinion pieces without data
        if self._is_pure_opinion(text_lower):
            return (False, "opinion_without_data")

        # PASS: Has economic data
        return (True, "passed")

    def _is_sustainability_related(self, text_lower: str) -> bool:
        """Check if article is about climate/sustainability at all"""
        patterns = [
            r'\bclimate\b', r'\b(renewable|clean) energy\b', r'\b(solar|wind|geothermal|hydro)\b',
            r'\b(electric|ev) (vehicle|car)\b', r'\bcarbon (emissions?|capture|neutral)\b',
            r'\bnet-?zero\b', r'\bsustainab(le|ility)\b', r'\b(fossil fuel|coal|oil|gas).{0,30}\b(transition|phaseout)\b',
            r'\benergy transition\b', r'\bgreen (tech|energy|hydrogen)\b', r'\becosystem\b', r'\bbiodiversity\b',
        ]
        return any(re.search(pattern, text_lower) for pattern in patterns)

    def _has_economic_data(self, text_lower: str) -> bool:
        """Check if article has economic/cost data"""

        # Cost and pricing data
        cost_patterns = [
            r'\b(lcoe|levelized cost)\b',
            r'\$\d+[\.,]?\d*\s*(per|/)?\s*(mwh|kwh|kg|ton|unit)\b',
            r'\bcost (of|per|declined|fell|dropped)\b',
            r'\bprice (parity|competitive|cheaper|fell|dropped)\b',
            r'\b\d+%\s*(cheaper|more expensive|less|reduction)\b',
        ]

        for pattern in cost_patterns:
            if re.search(pattern, text_lower):
                return True

        # Investment and financial data
        financial_patterns = [
            r'\$\d+[\.,]?\d*\s*(billion|million|b|m)\s*(investment|invested|funding|raised)\b',
            r'\b(profitable|profitability|profit margin|operating margin)\b',
            r'\b(roi|return on investment|payback period|npv)\b',
            r'\b(revenue|earnings|sales) (of|grew|increased)\b',
            r'\bmarket (cap|capitalization|value)\b',
        ]

        for pattern in financial_patterns:
            if re.search(pattern, text_lower):
                return True

        # Job and employment data
        job_patterns = [
            r'\b\d{1,3}(,\d{3})*\s*(jobs?|workers?|positions?)\b',
            r'\b(job creation|employment|hiring)\b',
            r'\b(wage|salary|pay) (premium|increase|comparison)\b',
        ]

        for pattern in job_patterns:
            if re.search(pattern, text_lower):
                return True

        # Asset valuation
        asset_patterns = [
            r'\b(stranded assets?|writedown|write-down|impairment)\b',
            r'\b(retired early|closure|shut down|decommission)\b.{0,50}\b(plant|coal|gas|facility)\b',
            r'\b(divestment|divest)\b.{0,50}\b\$\d+',
        ]

        for pattern in asset_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _is_pure_opinion(self, text_lower: str) -> bool:
        """Detect opinion pieces without economic data"""

        opinion_signals = [
            r'\bi (believe|think|feel|argue) (that|we)\b',
            r'\bin my (view|opinion)\b',
            r'\bwe (should|must|need to)\b(?!.{0,100}\b(invest|fund|subsid)\b)',
            r'\bopinion:',
            r'\bcommentary:',
        ]

        data_signals = [
            r'\b(data|study|research|analysis|report) (shows?|finds?|indicates?)\b',
            r'\baccording to\b.{0,50}\b(iea|epa|report|study)\b',
            r'\b\d+%',
            r'\$\d+',
        ]

        has_opinion = any(re.search(pattern, text_lower) for pattern in opinion_signals)
        has_data = any(re.search(pattern, text_lower) for pattern in data_signals)

        # Block if opinion signals but NO data
        return has_opinion and not has_data

    def get_pass_indicators(self, article: Dict) -> List[str]:
        """Return list of indicators that led to passing (for debugging)"""

        text = self._get_combined_text(article)
        text_lower = text.lower()
        indicators = []

        # Cost indicators
        if re.search(r'\b(lcoe|levelized cost)\b', text_lower):
            indicators.append("has_lcoe")

        if re.search(r'\$\d+[\.,]?\d*\s*(per|/)?\s*(mwh|kwh|kg)\b', text_lower):
            indicators.append("has_unit_cost")

        if re.search(r'\bcost (declined|fell|dropped)\b', text_lower):
            indicators.append("has_cost_decline")

        if re.search(r'\bprice (parity|competitive|cheaper)\b', text_lower):
            indicators.append("has_price_comparison")

        # Financial indicators
        if re.search(r'\$\d+[\.,]?\d*\s*(billion|million)\s*investment\b', text_lower):
            indicators.append("has_investment_data")

        if re.search(r'\b(profitable|profitability|profit margin)\b', text_lower):
            indicators.append("has_profitability")

        if re.search(r'\b(roi|payback period)\b', text_lower):
            indicators.append("has_roi_data")

        # Job indicators
        if re.search(r'\b\d{1,3}(,\d{3})*\s*jobs?\b', text_lower):
            indicators.append("has_job_numbers")

        # Asset indicators
        if re.search(r'\b(stranded assets?|writedown)\b', text_lower):
            indicators.append("has_asset_valuation")

        return indicators

    def _get_combined_text(self, article: Dict) -> str:
        """Combine title + description + content for analysis"""
        parts = []

        if 'title' in article:
            parts.append(article['title'])

        if 'description' in article:
            parts.append(article['description'])

        if 'content' in article:
            parts.append(article['content'][:2000])

        return ' '.join(parts)
