# Sustainability Impact Filter v1.0

**Purpose**: Rate content for sustainability relevance, impact potential, and credibility based on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES.

**Focus**: Distinguishes announcements from deployments, detects greenwashing/vaporware/fossil fuel delay tactics.

---

## Filter Components

### 1. Pre-Filter (`prefilter.py`)
Fast rule-based filter that blocks low-credibility content before LLM labeling.

**Blocks:**
- **Greenwashing** (unless verified/specific data/deployed tech)
  - Corporate sustainability reports, net-zero pledges, carbon offsets
  - Triggers: 5 patterns, 5 exception patterns
  - Max credibility if flagged: 3, max overall: 4.0

- **Vaporware** (unless deployed units/contracts/operational data)
  - Product announcements, prototypes, pilots, early-stage concepts
  - Triggers: 5 patterns, 5 exception patterns
  - Max investment_readiness if flagged: 4, max overall: 5.0

- **Fossil Transition** (unless genuine renewables/retirement)
  - "Clean coal", natural gas bridge, fossil hydrogen without lifecycle accounting
  - Triggers: 4 patterns, 4 exception patterns
  - Max impact if flagged: 4, max overall: 4.0

**Expected Pass Rate**: TBD (calibration required)

**Test Results**: 6/6 test cases passing

---

### 2. LLM Prompt (`prompt.md`)
Comprehensive evaluation framework with 8 dimensions focused on credibility and deployment.

**Scoring Dimensions**:
1. **Climate Impact** (0.25 weight) - Demonstrable GHG reduction/sequestration
2. **Technical Credibility** (0.20 weight) - GATEKEEPER dimension
3. **Economic Viability** (0.15 weight) - Path to cost-competitiveness
4. **Deployment Readiness** (0.15 weight) - TRL stage (1-9)
5. **Systemic Impact** (0.10 weight) - Gigaton-scale potential
6. **Justice & Equity** (0.05 weight) - Equitable access
7. **Innovation Quality** (0.05 weight) - Genuine breakthrough
8. **Evidence Strength** (0.05 weight) - Quality of sources

**Gatekeeper Rules**:
- If `technical_credibility < 5` → max overall score = 4.0

**Content Type Caps**:
- Greenwashing risk → max 4.0
- Vaporware → max 5.0
- Fossil transition → max 4.0

---

### 3. Configuration (`config.yaml`)
Weights, thresholds, and deployment parameters.

**Tier Thresholds**:
- Breakthrough: ≥ 7.5
- Promising: ≥ 5.0
- Monitoring: ≥ 3.0
- Not relevant: < 3.0

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
  - Climate tech investment intelligence
  - Greenwashing detection
  - Progress tracking for climate solutions

---

## Calibration Status

⚠️ **Pre-calibration required**:
1. **Oracle calibration**: Compare Flash vs Pro/Sonnet (100 samples)
2. **Pre-filter calibration**: Measure pass rate and score distribution (500 samples)

---

## Version History

### v1.0 (Current)
- Initial release
- Deployment-focused evaluation framework
- Rule-based pre-filter for greenwashing/vaporware/fossil transition
- Comprehensive configuration
