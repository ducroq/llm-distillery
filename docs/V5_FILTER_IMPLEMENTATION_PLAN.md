# V5 Filter Implementation Plan

**Date:** November 22, 2025
**Goal:** Create improved versions of all three filters with dimension reduction based on redundancy analysis
**Expected Impact:** 50-75% faster training, clearer interpretability, better generalization

---

## Executive Summary

Analysis of v4 models revealed severe dimension redundancy (62-87%) that was detectable BEFORE training. We're now creating v5 versions with optimized dimensional structure based on PCA/correlation analysis of oracle labels.

**Key Changes:**
- **investment-risk v4 → v5**: 8 dimensions → **2-3 dimensions** (75% reduction)
- **sustainability_tech_innovation v2 → v3**: 8 dimensions → **2-3 dimensions** (75% reduction)
- **uplifting v4 → v5**: 8 dimensions → **6 dimensions** (25% reduction)

**Benefits:**
- ⚡ 3-4x faster training
- 💰 Lower inference cost (fewer output dimensions)
- 📊 Clearer, more interpretable scores
- 🎯 Better generalization (less overfitting risk)
- 🔧 Easier to calibrate and monitor

---

## Dimensional Restructuring

### Investment-Risk v5

**Current (v4): 8 dimensions**
```yaml
dimensions:
  macro_risk_severity      # r=0.97 with systemic_risk
  credit_market_stress     # r=0.96 with systemic_risk
  market_sentiment_extremes # r=0.93 with actionability
  valuation_risk           # r=0.92 with actionability
  policy_regulatory_risk   # r=0.95 with macro_risk_severity
  systemic_risk            # Hub: correlated with ALL
  evidence_quality         # Independent (r<0.5)
  actionability            # r=0.95 with macro_risk_severity
```

**PCA Analysis:**
- PC1: 87.1% variance (single "risk" factor)
- Effective dimensions (95% variance): 3
- All risk dimensions r > 0.90 (almost identical!)

**Proposed (v5): 2-3 dimensions**

**Option A: Minimal (2 dimensions)**
```yaml
dimensions:
  overall_investment_risk:
    weight: 0.85
    description: "Composite risk score capturing macro, credit, sentiment, valuation, policy, and systemic risks"
    components: [macro, credit, sentiment, valuation, policy, systemic]

  evidence_quality:
    weight: 0.15
    description: "Quality and specificity of risk evidence"
```

**Option B: Granular (3 dimensions)**
```yaml
dimensions:
  systemic_risk:
    weight: 0.70
    description: "Overall economic/financial system risk"

  evidence_quality:
    weight: 0.15
    description: "Quality and specificity of risk evidence"

  actionability:
    weight: 0.15
    description: "How actionable are the signals for portfolio positioning"
```

**Recommendation:** Option A (2 dimensions)
- Clearest for users: "How risky?" + "How solid is the evidence?"
- 7 risk dimensions were saying the same thing anyway
- Can still provide risk_type metadata (macro/credit/sentiment) as categorical tags

---

### Sustainability Tech Innovation v3

**Current (v2): 8 dimensions**
```yaml
dimensions:
  deployment_maturity      # r=0.96 with technology_readiness
  technology_performance   # r=0.95 with technology_readiness
  cost_trajectory          # r=0.90 with deployment_maturity
  scale_of_deployment      # r=0.94 with market_penetration
  market_penetration       # r=0.94 with scale_of_deployment
  technology_readiness     # Hub: r > 0.90 with most dimensions
  supply_chain_maturity    # r=0.84 with technology_readiness
  proof_of_impact          # r=0.92 with technology_performance
```

**PCA Analysis:**
- PC1: 89.1% variance (single "maturity" factor)
- Effective dimensions (95% variance): 3
- ALL dimensions r > 0.85 (completely redundant!)

**Proposed (v3): 2-3 dimensions**

**Option A: Simple (1 dimension + metadata)**
```yaml
dimensions:
  technology_maturity:
    weight: 1.0
    description: "Overall deployment maturity from lab to widespread commercial adoption"

metadata:
  maturity_stage: [research, pilot, early_commercial, mature]
  technology_type: [energy, transport, materials, industrial, etc.]
  cost_status: [expensive, approaching_parity, cost_competitive]
```

**Option B: Tripartite (3 dimensions)**
```yaml
dimensions:
  deployment_scale:
    weight: 0.50
    description: "Current scale of commercial deployment and market penetration"
    merges: [deployment_maturity, scale_of_deployment, market_penetration]

  technology_performance:
    weight: 0.35
    description: "Technical maturity, readiness, and demonstrated impact"
    merges: [technology_performance, technology_readiness, proof_of_impact]

  economics:
    weight: 0.15
    description: "Cost trajectory and supply chain maturity"
    merges: [cost_trajectory, supply_chain_maturity]
```

**Recommendation:** Option B (3 dimensions)
- Captures three distinct aspects of "maturity"
- Deployment scale (Is it actually being used?)
- Technology performance (Does it work well?)
- Economics (Is it affordable?)
- Still 62.5% reduction, much clearer than 8 dimensions

---

### Uplifting v5

**Current (v4): 8 dimensions**
```yaml
dimensions:
  agency              # r=0.97 with progress
  progress            # r=0.93 with collective_benefit
  collective_benefit  # r=0.94 with agency
  connection          # r=0.80 with resilience
  innovation          # r=0.75 with progress
  justice             # r=0.58 with resilience
  resilience          # r=0.80 with connection
  wonder              # r=0.61 with connection
```

**PCA Analysis:**
- PC1: 75.0% variance (general positivity)
- Effective dimensions (95% variance): 5
- Clear cluster: agency/progress/benefit (r > 0.90)
- Other dimensions more independent

**Proposed (v5): 5-6 dimensions**

**Option A: Merged cluster (5 dimensions)**
```yaml
dimensions:
  concrete_progress:
    weight: 0.30
    description: "Tangible positive outcomes: agency, progress, collective benefit"
    merges: [agency, progress, collective_benefit]

  connection:
    weight: 0.15
    description: "Social bonds, relationships, community"

  innovation:
    weight: 0.15
    description: "Creativity, novelty, breakthrough thinking"

  justice:
    weight: 0.20
    description: "Fairness, equity, addressing wrongs"

  resilience:
    weight: 0.20
    description: "Strength, adaptation, overcoming adversity"
```

**Option B: Keep 6, merge carefully (6 dimensions)**
```yaml
dimensions:
  agency_progress:
    weight: 0.25
    description: "Individual and collective agency driving positive change"
    merges: [agency, progress]

  collective_benefit:
    weight: 0.15
    description: "Shared positive outcomes"

  connection:
    weight: 0.15
    description: "Social bonds and relationships"

  innovation:
    weight: 0.10
    description: "Creativity and breakthrough thinking"

  justice:
    weight: 0.20
    description: "Fairness and equity"

  resilience:
    weight: 0.15
    description: "Strength and adaptation"
```

**Recommendation:** Option A (5 dimensions)
- 37.5% reduction, still significant
- "concrete_progress" captures the tight cluster
- Other dimensions remain independent (they should!)
- Clearer for users: concrete outcomes vs abstract emotions

---

## Implementation Timeline

### Week 1: Setup and Design

**Days 1-2: Investment-Risk v5**
- [ ] Create `filters/investment-risk/v5/` directory
- [ ] Design config.yaml (2 dimensions)
- [ ] Draft prompt-compressed.md (emphasize dimension independence)
- [ ] Add contrastive examples (high risk + low evidence, low risk + high evidence)
- [ ] Phase 3: Oracle calibration on 100 articles
- [ ] **CRITICAL: Run dimension redundancy analysis**
- [ ] Validate redundancy < 30%

**Days 3-4: Sustainability Tech Innovation v3**
- [ ] Create `filters/sustainability_tech_innovation/v3/` directory
- [ ] Design config.yaml (3 dimensions)
- [ ] Draft prompt-compressed.md
- [ ] Add examples showing independence of deployment/performance/economics
- [ ] Phase 3: Oracle calibration on 100 articles
- [ ] **CRITICAL: Run dimension redundancy analysis**
- [ ] Validate redundancy < 30%

**Days 5-6: Uplifting v5**
- [ ] Create `filters/uplifting/v5/` directory
- [ ] Design config.yaml (5 dimensions)
- [ ] Draft prompt-compressed.md
- [ ] Add contrastive examples
- [ ] Phase 3: Oracle calibration on 100 articles
- [ ] **CRITICAL: Run dimension redundancy analysis**
- [ ] Validate redundancy < 30%

**Day 7: Review and Adjust**
- [ ] Review all dimension analyses
- [ ] Adjust any filters with redundancy > 30%
- [ ] Document design decisions

### Week 2: Training Data Generation

**All filters in parallel (using GPU):**
- [ ] Generate 5,000 training examples per filter
- [ ] 70/15/15 train/val/test split
- [ ] Validate data quality
- [ ] Check for oracle consistency

**Cost:** ~$15-20 total for all three filters

### Week 3: Model Training

**All filters in parallel (using GPU):**
- [ ] Train student models (Qwen 2.5-1.5B)
- [ ] Expected training time: 1.5-2 hours each (vs 6 hours for v4!)
- [ ] Monitor training curves
- [ ] Early stopping at convergence

**Expected speedup:** 3-4x faster than v4 due to fewer dimensions

### Week 4: Testing and Deployment

**Days 1-2: Benchmarking**
- [ ] Run test set benchmarks for all three models
- [ ] Compare MAE vs v4 models (expect similar or better)
- [ ] Run error analysis (check for bias, correlations)

**Days 3-4: Analysis and Documentation**
- [ ] Generate comprehensive reports
- [ ] Compare v5 vs v4 performance
- [ ] Document improvements
- [ ] Update deployment guide

**Days 5-7: Deployment Preparation**
- [ ] Prepare production inference code
- [ ] Update API to handle new dimensional structure
- [ ] Create migration guide for users
- [ ] Deploy to staging
- [ ] Final validation

---

## Success Criteria

### Dimension Independence (Phase 3 Validation)

**PASS:**
- Redundancy ratio < 30%
- PC1 variance < 70%
- No dimension pairs with r > 0.85
- Effective dimensions ≥ 70% of designed dimensions

**If FAIL:** Redesign dimensions and repeat calibration

### Model Performance (Phase 7 Testing)

**Target Metrics:**

| Filter | v4 Test MAE | v5 Target MAE | Status |
|--------|-------------|---------------|--------|
| investment-risk | 0.395 | ≤ 0.45 | Expect similar or better |
| sustainability_tech_innovation | 0.643 | ≤ 0.70 | Expect similar or better |
| uplifting | 0.968 | ≤ 1.05 | Expect similar or better |

**Why slightly higher MAE is acceptable:**
- Fewer dimensions = less overfitting
- Better interpretability worth small accuracy tradeoff
- Calibration curves can further improve accuracy

### Training Efficiency

**Target Improvements:**

| Metric | v4 | v5 Target | Improvement |
|--------|----|-----------| ------------|
| **Training time** | 6 hours | 1.5-2 hours | **3-4x faster** |
| **Model size** | 8D output | 2-5D output | **40-75% smaller** |
| **Inference time** | 100ms | 60-80ms | **20-40% faster** |
| **Interpretability** | 8 scores | 2-5 scores | **Clear winners** |

### Generalization

**Target:**
- Val→Test gap ≤ 10% (similar to or better than v4)
- No regression-to-mean worse than v4
- Error correlations should be LOWER (more independent dimensions)

---

## Risk Mitigation

### Risk 1: Dimensions still correlated despite redesign

**Mitigation:**
- Run redundancy analysis on 100-article pilot BEFORE full training data
- If redundancy > 30%, iterate on prompt design
- Budget 2-3 iteration cycles if needed

### Risk 2: Performance worse than v4

**Mitigation:**
- Accept up to 15% MAE increase for interpretability gains
- Use calibration curves to correct systematic bias
- If MAE > 15% worse, investigate whether dimension reduction was too aggressive

### Risk 3: Oracle prompt doesn't enforce independence

**Mitigation:**
- Use explicit instructions: "Rate dimensions independently"
- Provide 3-5 contrastive examples per filter (high X + low Y combinations)
- Consider separate oracle calls per dimension if needed (more expensive but independent)

### Risk 4: Users prefer granular dimensions

**Mitigation:**
- Keep dimension_breakdown in metadata
- Example: overall_risk=7.2, risk_components={macro: 7.5, credit: 7.0, sentiment: 7.3}
- Provide both "simplified" and "detailed" API responses

---

## Migration Guide for Users

### API Changes

**Before (v4):**
```json
{
  "filter": "investment-risk",
  "version": "4.0",
  "scores": {
    "macro_risk_severity": 7.2,
    "credit_market_stress": 7.0,
    "market_sentiment_extremes": 6.8,
    "valuation_risk": 7.3,
    "policy_regulatory_risk": 6.9,
    "systemic_risk": 7.5,
    "evidence_quality": 5.5,
    "actionability": 7.1
  },
  "overall_score": 6.91
}
```

**After (v5):**
```json
{
  "filter": "investment-risk",
  "version": "5.0",
  "scores": {
    "overall_investment_risk": 7.1,
    "evidence_quality": 5.5
  },
  "overall_score": 6.85,
  "risk_breakdown": {
    "dominant_risks": ["macro", "systemic", "credit"],
    "risk_intensity": "high"
  }
}
```

### Interpretation Guide

**investment-risk v5:**
- `overall_investment_risk` (0-10): Single comprehensive risk score
  - 0-2: Low risk environment
  - 3-5: Moderate risk
  - 6-8: Elevated risk
  - 9-10: Extreme risk
- `evidence_quality` (0-10): Quality of risk signals
  - Higher = more specific, actionable evidence
  - Lower = vague, speculative concerns

**sustainability_tech_innovation v3:**
- `deployment_scale` (0-10): Current commercial adoption
- `technology_performance` (0-10): Technical maturity and demonstrated impact
- `economics` (0-10): Cost competitiveness and supply chain readiness

**uplifting v5:**
- `concrete_progress` (0-10): Tangible positive outcomes
- `connection` (0-10): Social bonds and relationships
- `innovation` (0-10): Creativity and novelty
- `justice` (0-10): Fairness and equity
- `resilience` (0-10): Strength and adaptation

---

## Rollout Strategy

### Phase 1: Shadow Deployment (Week 4)
- Deploy v5 alongside v4
- Run both models on production traffic
- Log predictions from both
- Compare outputs, monitor for issues

### Phase 2: A/B Testing (Week 5)
- 10% of traffic to v5
- Monitor user feedback
- Track API response times
- Validate accuracy on real data

### Phase 3: Gradual Rollout (Week 6-7)
- 25% → 50% → 75% → 100%
- Monitor for issues at each stage
- Rollback plan if MAE degrades

### Phase 4: Deprecate v4 (Week 8)
- Announce v4 deprecation (1 month notice)
- Provide migration guide
- Keep v4 available for 1 month
- Archive v4 models

---

## Documentation Deliverables

### Per-Filter Documentation

**Each filter (`filters/{filter_name}/v5/`) should have:**

1. **config.yaml** - Dimension definitions, weights, tiers
2. **prompt-compressed.md** - Oracle prompt emphasizing independence
3. **dimension_analysis/** - PCA/correlation analysis results
4. **validation_report.md** - Oracle calibration results
5. **training_metadata.json** - Training configuration and results
6. **benchmarks/** - Test set results and error analysis
7. **README.md** - Filter-specific usage guide

### Project-Level Documentation

1. **V5_COMPARISON_REPORT.md** - Compare v5 vs v4 across all metrics
2. **DIMENSION_REDESIGN_RATIONALE.md** - Document why each dimension was merged/kept
3. **MIGRATION_GUIDE.md** - For users upgrading from v4 to v5
4. **Updated filter-development-guide.md** - Already done! ✅

---

## Next Steps

**Immediate (Today):**
1. ✅ Update filter-development-guide.md with dimension redundancy analysis
2. ✅ Document v4 redundancy findings
3. Create this implementation plan

**Day 1 (Tomorrow):**
1. Create investment-risk v5 structure
2. Design 2-dimension config
3. Draft new oracle prompt with independence emphasis
4. Run pilot calibration (100 articles)
5. **Check redundancy analysis**

**Day 2:**
1. If redundancy good, proceed to full training data
2. If not, iterate on prompt design
3. Start sustainability_tech_innovation v3

**Week 2-4:**
- Execute full plan as outlined above

---

## Appendix: Dimension Reduction Examples

### Example: How to Merge Dimensions

**Before:**
```yaml
macro_risk_severity:
  description: "Systemic economic/financial risk signals"

credit_market_stress:
  description: "Credit market disruption indicators"

systemic_risk:
  description: "Financial system vulnerability"
```

**After:**
```yaml
overall_investment_risk:
  description: |
    Comprehensive assessment of investment risk environment including:
    - Macroeconomic risks (recession, inflation, policy uncertainty)
    - Credit market stress (spreads, defaults, liquidity)
    - Systemic vulnerabilities (interconnected risks, contagion potential)

  scoring_rubric:
    0-2: "Low risk environment with stable conditions"
    3-5: "Moderate risks present but manageable"
    6-8: "Elevated risks requiring defensive positioning"
    9-10: "Extreme risks, crisis conditions"
```

### Example: Enforcing Independence in Prompt

**Bad (produces correlated dimensions):**
```markdown
Rate the article on these dimensions:
1. deployment_maturity (0-10)
2. technology_readiness (0-10)
3. scale_of_deployment (0-10)
```

**Good (enforces independence):**
```markdown
Rate these dimensions INDEPENDENTLY. An article can score high on one and low on another.

1. **deployment_scale** (0-10): Current commercial adoption
   - Focus: How widely deployed is this technology TODAY?
   - Example: Solar panels are 9/10 (widespread), fusion is 1/10 (not deployed)

2. **technology_performance** (0-10): Technical maturity
   - Focus: Does the technology work RELIABLY at scale?
   - Example: Batteries are 8/10 (mature), hydrogen storage is 4/10 (challenges remain)

3. **economics** (0-10): Cost competitiveness
   - Focus: Is it AFFORDABLE compared to alternatives?
   - Example: Wind is 9/10 (cost-competitive), carbon capture is 3/10 (expensive)

CONTRASTIVE EXAMPLES:
- High deployment + Low performance: Early EVs (deployed but reliability issues)
- High performance + Low deployment: Advanced nuclear (works great but few deployments)
- High performance + High deployment + Low economics: Renewables in 2010 (worked, deployed, but expensive)
```

---

**End of Implementation Plan**

This plan will be executed starting tomorrow. Expected completion: 4 weeks from now with all three v5 filters trained, tested, and ready for production deployment.
