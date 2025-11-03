# Pre-filter Calibration Report

**Filter**: uplifting
**Version**: v1
**Sample Size**: 500 articles

---

## Summary

Pre-filter pass rate determines how many articles are sent to the LLM for labeling vs blocked locally.

### Pass Rate: 94.8%

- **Passed**: 474 articles → Will be labeled by oracle
- **Blocked**: 26 articles → Saved LLM cost

### Block Reason Distribution

| Reason | Count | Percentage |
|--------|-------|------------|
| corporate_finance | 15 | 57.7% |
| military_security | 11 | 42.3% |

---

## Pre-filter Statistics

- **version**: 1.0
- **corporate_finance_patterns**: 9
- **corporate_finance_exceptions**: 5
- **military_security_patterns**: 7
- **military_security_exceptions**: 5

---

## Interpretation

**Pass Rate Analysis**:
- **High pass rate (>70%)**: Pre-filter is conservative, may let low-value content through
- **Moderate pass rate (40-70%)**: Balanced filtering, good for initial training
- **Low pass rate (<40%)**: Aggressive filtering, may miss some relevant content

**Recommended Next Steps**:
1. ✅ Review sample blocked articles to check for false negatives
2. ⏳ Run oracle calibration: `calibrate_oracle.py`
3. ⏳ If needed, adjust pre-filter patterns and re-calibrate
4. ⏳ Generate ground truth: `batch_labeler.py --filter filters/uplifting/v1`

---

**Generated**: 
