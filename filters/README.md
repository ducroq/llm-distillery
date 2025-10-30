# Filter Packages

This directory contains versioned filter packages for LLM Distillery. Each filter is a complete, self-contained system for evaluating content on specific semantic dimensions.

---

## Filter Architecture

Each filter package contains:

```
filters/<filter-name>/v<version>/
├── prompt.md           # LLM evaluation prompt (STEP 1: pre-classification, STEP 2: scoring)
├── prefilter.py        # Fast rule-based filter (blocks obvious low-value content)
├── config.yaml         # Weights, thresholds, tier boundaries, deployment specs
└── README.md           # Filter documentation and calibration status
```

---

## Available Filters

### 1. Uplifting Content Filter (v1.0)
**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Pre-filter blocks**:
- Corporate finance (unless worker coop/public benefit)
- Military buildups (unless peace/demilitarization)

**Dimensions (8)**: agency, progress, collective_benefit (gatekeeper), connection, innovation, justice, resilience, wonder

**Status**: ✅ Implemented, ⏳ Calibration pending

[View details →](uplifting/v1/README.md)

---

### 2. Sustainability Impact Filter (v1.0)
**Purpose**: Rate content for sustainability relevance based on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES.

**Pre-filter blocks**:
- Greenwashing (unless verified/specific data)
- Vaporware (unless deployed units/contracts)
- Fossil fuel transition tactics (unless genuine renewables)

**Dimensions (8)**: climate_impact, technical_credibility (gatekeeper), economic_viability, deployment_readiness, systemic_impact, justice_equity, innovation_quality, evidence_strength

**Status**: ✅ Implemented, ⏳ Calibration pending

[View details →](sustainability/v1/README.md)

---

### 3. Investment Risk Filter (v1.0)
**Purpose**: Identify investment risk signals for defense-first portfolio management.

**Pre-filter blocks**:
- FOMO/speculation (hot stocks, meme stocks, crypto pumping)
- Stock picking (unless macro context)
- Affiliate marketing (broker links, promo codes)
- Clickbait (sensationalist headlines)

**Dimensions (8)**: macro_risk_severity, credit_market_stress, market_sentiment_extremes, valuation_risk, policy_regulatory_risk, systemic_risk, evidence_quality (gatekeeper), actionability

**Status**: ✅ Implemented, ⏳ Calibration pending

[View details →](investment-risk/v1/README.md)

---

### 4. SEECE Energy Tech Filter (v1.0)
**Purpose**: Evaluate clean energy and efficiency technologies.

**Status**: ⏳ Pending implementation

---

### 5. Future of Education Filter (v1.0)
**Purpose**: Assess educational innovations and AI in learning.

**Status**: ⏳ Pending implementation

---

## Filter Development Workflow

### Phase 1: Design & Implementation
1. Create filter directory: `filters/<name>/v1/`
2. Write prompt template: Extract pre-classification rules (STEP 1) and scoring dimensions (STEP 2)
3. Implement pre-filter: Convert STEP 1 rules to regex/keyword patterns
4. Create config: Define weights, thresholds, tier boundaries
5. Test pre-filter: Run built-in test cases

### Phase 2: Calibration
**Purpose**: Validate filter effectiveness before ground truth generation.

#### 2.1 Pre-filter Calibration (500 samples)
```bash
python -m ground_truth.calibrate_prefilter \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --sample-size 500 \
    --output reports/uplifting_v1_prefilter_calibration.md
```

**Measures**:
- Pass rate (% of articles sent to LLM)
- Block reason distribution
- Sample blocked articles for false negative review

**Goal**: 40-70% pass rate (balanced filtering)

---

#### 2.2 Oracle Calibration (100 samples)
```bash
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --sample-size 100 \
    --models gemini-flash,gemini-pro,claude-sonnet \
    --output reports/uplifting_v1_oracle_calibration.md
```

**Measures**:
- Agreement rates between models
- Score distributions
- Cost analysis
- Dimension-level correlation

**Goal**: Choose oracle (typically Flash for cost/speed, Pro/Sonnet if quality issues)

---

### Phase 3: Ground Truth Generation
```bash
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --target-labeled 2500 \
    --oracle gemini-flash \
    --output datasets/labeled/uplifting_v1/
```

**Process**:
1. Stream through master datasets
2. Apply pre-filter (fast, local, free)
3. Label passing articles with oracle (LLM API)
4. Save labeled articles (JSONL)
5. Stop when target reached (2,500 articles)

---

### Phase 4: Model Training
```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --dataset datasets/labeled/uplifting_v1/ \
    --output inference/deployed/uplifting_v1/
```

**Output**: Trained Qwen 2.5-7B model checkpoint

---

### Phase 5: Evaluation & Deployment
```bash
# Evaluate model vs oracle
python -m evaluation.evaluate_model \
    --filter filters/uplifting/v1 \
    --model inference/deployed/uplifting_v1/ \
    --test-set datasets/labeled/uplifting_v1/test.jsonl

# Deploy to production
cp -r filters/uplifting/v1/prefilter.py inference/deployed/uplifting_v1/
cp -r filters/uplifting/v1/config.yaml inference/deployed/uplifting_v1/
```

**Deployed filter**: `prefilter.py` + `model.bin` + `config.yaml`

---

## Design Principles

### 1. **Pre-filter + Prompt Coupling**
Pre-filter rules are extracted from prompt STEP 1. They evolve together within the same version.

**Example (Uplifting)**:
- Prompt: "If corporate finance AND NOT worker coop → FLAG corporate_finance (max_score=2)"
- Pre-filter: Blocks "corporate_finance" articles unless they match exception patterns

### 2. **Versioning**
Filters use semantic versioning: `v1.0`, `v1.1`, `v2.0`

**Version bump triggers**:
- Major (v1 → v2): Fundamental changes to scoring dimensions or pre-filter logic
- Minor (v1.0 → v1.1): Refinements to weights, thresholds, or patterns
- Patch (v1.0.0 → v1.0.1): Bug fixes, documentation updates

### 3. **Calibration-First**
Never generate ground truth without calibration:
1. Pre-filter calibration validates blocking logic
2. Oracle calibration chooses best LLM (Flash/Pro/Sonnet)
3. Both save costs and improve training data quality

### 4. **Deployment Package**
Each deployed filter includes:
- Pre-filter (Python module)
- Trained model (Qwen checkpoint)
- Config (YAML with thresholds)

**Inference pipeline**: Article → Pre-filter → [if passed] → Model → Score + Tier

---

## Performance Targets

| Metric | Target | Achieved (Uplifting v1) | Achieved (Sustainability v1) |
|--------|--------|-------------------------|------------------------------|
| Pre-filter pass rate | 40-70% | ⏳ Calibration pending | ⏳ Calibration pending |
| Oracle cost/article | < $0.01 | ⏳ Calibration pending | ⏳ Calibration pending |
| Model accuracy vs oracle | > 90% | ⏳ Training pending | ⏳ Training pending |
| Inference time | < 50ms | ⏳ Deployment pending | ⏳ Deployment pending |

---

## Next Steps

1. ✅ Implement uplifting and sustainability filters
2. ⏳ Run pre-filter calibration (500 samples each)
3. ⏳ Run oracle calibration (100 samples each)
4. ⏳ Generate ground truth (2,500 samples each)
5. ⏳ Train Qwen models
6. ⏳ Deploy and evaluate

---

**For detailed calibration workflow, see**: [docs/guides/ground-truth-generation.md](../docs/guides/ground-truth-generation.md)
