# Uplifting Content Filter v1.0

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Focus**: MEANING not TONE - what is happening for human/planetary wellbeing, not emotional writing style.

---

## Filter Components

### 1. Pre-Filter (`prefilter.py`)
Fast rule-based filter that blocks obvious low-value content before LLM labeling.

**Blocks:**
- **Corporate Finance** (unless worker coop/public benefit/open source)
  - Stock prices, earnings, funding rounds, valuations, M&A, IPO
  - Triggers: 9 patterns, 5 exception patterns
  - Max score if passed: 2.0

- **Military/Security Buildups** (unless peace/demilitarization)
  - Military buildup, defense spending, weapons, NATO expansion
  - Triggers: 7 patterns, 5 exception patterns
  - Max score if passed: 4.0

**Expected Pass Rate**: TBD (calibration required)

**Test Results**: 5/5 test cases passing

---

### 2. LLM Prompt (`prompt.md`)
Comprehensive evaluation framework with 8 dimensions.

**Scoring Dimensions** (0-10 scale):
1. **Agency** (0.14 weight) - People taking effective action
2. **Progress** (0.19 weight) - Movement toward flourishing
3. **Collective Benefit** (0.38 weight) - GATEKEEPER dimension
4. **Connection** (0.10 weight) - Collaboration across groups
5. **Innovation** (0.08 weight) - Novel solutions that work
6. **Justice** (0.04 weight) - Wrongs being addressed
7. **Resilience** (0.02 weight) - Recovery and persistence
8. **Wonder** (0.05 weight) - Freely shared knowledge

**Gatekeeper Rules**:
- If `collective_benefit < 5` → max overall score = 3.0
- Exception: If `wonder ≥ 7` and `collective_benefit ≥ 3` → no cap

**Content Type Caps**:
- Corporate finance → max 2.0
- Military/security → max 4.0
- Business news (if collective_benefit < 6) → max 4.0

---

### 3. Configuration (`config.yaml`)
Weights, thresholds, and deployment parameters.

**Tier Thresholds**:
- Impact: ≥ 7.0
- Connection: ≥ 4.0
- Not uplifting: < 4.0

---

## Training Requirements

- **Target samples**: 2,500 labeled articles
- **Train/val split**: 90% / 10%
- **Quality threshold**: 0.7
- **Recommended model**: Qwen2.5-7B
- **Expected accuracy**: 90-95% vs oracle

---

## Deployment Specifications

- **Inference time**: 20-50ms per article
- **Cost per article**: $0.00 (runs locally)
- **Use cases**:
  - Positive news aggregation
  - Solutions journalism
  - Progress indicators

---

## Calibration Status

⚠️ **Pre-calibration required**:
1. **Oracle calibration**: Compare Flash vs Pro/Sonnet (100 samples)
2. **Pre-filter calibration**: Measure pass rate and score distribution (500 samples)

---

## Version History

### v1.0 (Current)
- Initial release
- Battle-tested prompt from NexusMind-Filter (5,000+ articles)
- Rule-based pre-filter implementation
- Comprehensive configuration
