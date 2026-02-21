"""
Investment Risk v6 Prefilter

Re-exports the v5 prefilter unchanged. The prefilter logic is identical
between v5 and v6 - only the model and tier system changed.
"""

import importlib.util
from pathlib import Path

# Load v5 prefilter via importlib (hyphenated path prevents normal import)
_v5_prefilter_path = Path(__file__).parent.parent / "v5" / "prefilter.py"
_spec = importlib.util.spec_from_file_location("investment_risk_v5_prefilter", _v5_prefilter_path)
_v5_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v5_module)

# Re-export v5 prefilter class and functions
InvestmentRiskPreFilterV5 = _v5_module.InvestmentRiskPreFilterV5
InvestmentRiskPreFilter = InvestmentRiskPreFilterV5
prefilter = _v5_module.prefilter
get_stats = _v5_module.get_stats
