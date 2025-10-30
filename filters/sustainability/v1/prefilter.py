"""
Sustainability Pre-Filter v1.0

Blocks obvious low-credibility content before LLM labeling:
- Greenwashing (unless verified/specific data/deployed tech)
- Vaporware (unless deployed units/contracts/operational data)
- Fossil fuel transition delay tactics (unless genuine renewables/retirement)

Purpose: Reduce LLM costs and improve training data quality.
"""

import re
from typing import Dict, Tuple


class SustainabilityPreFilterV1:
    """Fast rule-based pre-filter for sustainability content"""

    VERSION = "1.0"

    # A) Greenwashing Indicators
    GREENWASHING_PATTERNS = [
        # Corporate pledges
        r'\b(net-zero pledge|carbon neutral|sustainability report)\b',
        r'\b(carbon offset|esg rating|green bond)\b',
        r'\b(sustainable|sustainability)\s+(award|certification|ranking)\b',

        # Vague claims
        r'\b(committed to|pledges to|aims to)\s+(reduce emissions|carbon neutral)\b',
        r'\b(green|eco-friendly|environmentally friendly)\s+(without specifics)\b',
    ]

    GREENWASHING_EXCEPTIONS = [
        r'\b(third-party verification|independently verified|certified by)\b',
        r'\b(\d+\s*(tons?|tonnes?|mt)\s*(co2|carbon|ghg))\b',  # Specific emissions data
        r'\b(deployed|operational|installed capacity)\b',
        r'\b(regulatory compliance|government audit)\b',
        r'\b(science-based target|sbti)\b',
    ]

    # B) Vaporware Indicators
    VAPORWARE_PATTERNS = [
        r'\b(announced|unveils|reveals|introduces)\s+(new product|prototype)\b',
        r'\b(pilot project|demonstration|proof of concept)\b',
        r'\b(plans to|intends to|will develop)\b',
        r'\b(early-stage|concept|prototype)\b',
        r'\b(aims to launch|targets|expects to)\b',
    ]

    VAPORWARE_EXCEPTIONS = [
        r'\b(\d+\s*(units?|mw|gw|installations?)\s*(deployed|installed|operational))\b',
        r'\b(customer contract|signed agreement|purchase order)\b',
        r'\b(operational data|performance data|actual)\b',
        r'\b(peer-reviewed|published in|journal)\b',
        r'\b(commercial operation|in production)\b',
    ]

    # C) Fossil Fuel Transition Indicators
    FOSSIL_TRANSITION_PATTERNS = [
        r'\b(clean coal|carbon capture.*oil|ccs.*enhanced recovery)\b',
        r'\b(natural gas bridge|gas transition|lng expansion)\b',
        r'\b(blue hydrogen|fossil hydrogen)\b',  # without lifecycle mention
        r'\b(offset.*aviation|carbon credit.*oil)\b',
    ]

    FOSSIL_TRANSITION_EXCEPTIONS = [
        r'\b(green hydrogen|renewable hydrogen|electrolysis)\b',
        r'\b(direct air capture.*storage|dac.*sequestration)\b',
        r'\b(fossil asset retirement|stranded asset|phase out)\b',
        r'\b(lifecycle emission|full lifecycle|cradle to grave)\b',
    ]

    def __init__(self):
        """Initialize pre-filter with compiled regex patterns"""
        self.greenwashing_regex = [re.compile(p, re.IGNORECASE) for p in self.GREENWASHING_PATTERNS]
        self.greenwashing_exceptions_regex = [re.compile(p, re.IGNORECASE) for p in self.GREENWASHING_EXCEPTIONS]

        self.vaporware_regex = [re.compile(p, re.IGNORECASE) for p in self.VAPORWARE_PATTERNS]
        self.vaporware_exceptions_regex = [re.compile(p, re.IGNORECASE) for p in self.VAPORWARE_EXCEPTIONS]

        self.fossil_transition_regex = [re.compile(p, re.IGNORECASE) for p in self.FOSSIL_TRANSITION_PATTERNS]
        self.fossil_transition_exceptions_regex = [re.compile(p, re.IGNORECASE) for p in self.FOSSIL_TRANSITION_EXCEPTIONS]

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for labeling.

        Args:
            article: Dict with 'title', 'text', and optionally 'source'

        Returns:
            (should_label, reason)
            - (True, "passed"): Send to LLM
            - (False, "greenwashing"): Block - greenwashing without verification
            - (False, "vaporware"): Block - announcement without deployment
            - (False, "fossil_transition"): Block - fossil fuel delay tactics
        """
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        combined_text = f"{title} {text}".lower()

        # Check A) Greenwashing
        if self._has_greenwashing(combined_text):
            if not self._has_exception(combined_text, self.greenwashing_exceptions_regex):
                return False, "greenwashing"

        # Check B) Vaporware
        if self._has_vaporware(combined_text):
            if not self._has_exception(combined_text, self.vaporware_exceptions_regex):
                return False, "vaporware"

        # Check C) Fossil Transition
        if self._has_fossil_transition(combined_text):
            if not self._has_exception(combined_text, self.fossil_transition_exceptions_regex):
                return False, "fossil_transition"

        # Passed all filters
        return True, "passed"

    def _has_greenwashing(self, text: str) -> bool:
        """Check if text contains greenwashing indicators"""
        return any(pattern.search(text) for pattern in self.greenwashing_regex)

    def _has_vaporware(self, text: str) -> bool:
        """Check if text contains vaporware indicators"""
        return any(pattern.search(text) for pattern in self.vaporware_regex)

    def _has_fossil_transition(self, text: str) -> bool:
        """Check if text contains fossil transition indicators"""
        return any(pattern.search(text) for pattern in self.fossil_transition_regex)

    def _has_exception(self, text: str, exception_patterns: list) -> bool:
        """Check if text contains exception keywords"""
        return any(pattern.search(text) for pattern in exception_patterns)

    def get_statistics(self) -> Dict:
        """Return filter statistics"""
        return {
            'version': self.VERSION,
            'greenwashing_patterns': len(self.GREENWASHING_PATTERNS),
            'greenwashing_exceptions': len(self.GREENWASHING_EXCEPTIONS),
            'vaporware_patterns': len(self.VAPORWARE_PATTERNS),
            'vaporware_exceptions': len(self.VAPORWARE_EXCEPTIONS),
            'fossil_transition_patterns': len(self.FOSSIL_TRANSITION_PATTERNS),
            'fossil_transition_exceptions': len(self.FOSSIL_TRANSITION_EXCEPTIONS),
        }


def test_prefilter():
    """Test the prefilter with sample articles"""

    prefilter = SustainabilityPreFilterV1()

    test_cases = [
        # Should BLOCK - Greenwashing
        {
            'title': 'Oil Giant Announces Net-Zero Pledge by 2050',
            'text': 'The company committed to carbon neutrality through offsets and future technologies...',
            'expected': (False, 'greenwashing')
        },

        # Should PASS - Verified Data
        {
            'title': 'Solar Farm Reduces 50,000 Tons CO2 Annually',
            'text': 'Third-party verification confirms the 100 MW installation is operational...',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Vaporware
        {
            'title': 'Startup Unveils Revolutionary Battery Prototype',
            'text': 'The company plans to launch commercial production in 2026...',
            'expected': (False, 'vaporware')
        },

        # Should PASS - Deployed Technology
        {
            'title': 'Grid Battery System: 500 Units Deployed',
            'text': 'Operational data shows 2 GWh of storage capacity with signed customer contracts...',
            'expected': (True, 'passed')
        },

        # Should BLOCK - Fossil Transition
        {
            'title': 'Natural Gas as Bridge to Clean Energy',
            'text': 'Industry advocates argue for LNG expansion as climate solution...',
            'expected': (False, 'fossil_transition')
        },

        # Should PASS - Genuine Renewable
        {
            'title': 'Green Hydrogen Plant Begins Production',
            'text': 'Electrolysis facility powered by wind energy produces renewable hydrogen...',
            'expected': (True, 'passed')
        },
    ]

    print("Testing Sustainability Pre-Filter v1.0")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        result = prefilter.should_label(test)
        expected = test['expected']
        status = "[PASS]" if result == expected else "[FAIL]"

        print(f"\nTest {i}: {status}")
        print(f"Title: {test['title']}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 60)
    print("Pre-filter Statistics:")
    for key, value in prefilter.get_statistics().items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_prefilter()
