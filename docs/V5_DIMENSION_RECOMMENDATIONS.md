# V5 Dimension Recommendations: What Should Actually Remain?

**Date:** November 22, 2025
**Based on:** PCA analysis and correlation matrix of v4 oracle labels

---

## Investment-Risk v5: Recommended Structure

### Current Reality (v4)

**Correlation matrix shows:**
- ALL 7 risk dimensions correlated r > 0.80 with each other
- Even `evidence_quality` has r=0.76 with systemic_risk (not as independent as we thought!)
- PC1 (single factor) explains 87.1% of variance
- All dimensions load similarly on PC1 (+0.31 to +0.37)

**What this means:**
The oracle is essentially rating articles on **ONE main axis**: "How risky is this overall?"

### Option A: Honest Minimal (1-2 dimensions) ✅ RECOMMENDED

Accept reality: investment risk is being rated as 1D. Design accordingly.

**Proposed structure:**

```yaml
filter:
  name: investment-risk
  version: "5.0"

scoring:
  dimensions:
    overall_risk:
      weight: 1.0
      description: |
        Composite investment risk capturing macro, credit, sentiment,
        valuation, policy, and systemic risks.

        Score holistically based on threat to capital preservation.

      scoring_rubric:
        0-2: Low risk - Stable environment, growth indicators positive
        3-4: Moderate - Some concerns but manageable
        5-6: Elevated - Multiple risks present, defensive positioning warranted
        7-8: High - Serious threats requiring risk reduction
        9-10: Crisis - Extreme conditions, capital preservation critical

      examples:
        - "2008 Lehman collapse: 9-10"
        - "2020 COVID crash: 8-9"
        - "2022 inflation shock: 6-7"
        - "2013 taper tantrum: 4-5"
        - "2017 'goldilocks' economy: 2-3"

  metadata_fields:
    risk_components:
      type: array
      description: "Which risk types are present"
      values: [macro, credit, sentiment, valuation, policy, systemic]

    evidence_quality:
      type: categorical
      description: "Quality of risk evidence"
      values: [weak, moderate, strong, very_strong]

    primary_risk_type:
      type: categorical
      values: [macro, credit, sentiment, valuation, policy, systemic, mixed]
```

**Why this works:**
- ✅ Honest about what oracle actually does (single risk assessment)
- ✅ Users get clear answer: "risk = 7.2" instead of confusing 8 scores
- ✅ Still captures nuance via metadata (which risks, how strong is evidence)
- ✅ 8x faster training (1D instead of 8D)
- ✅ Much faster inference
- ✅ Easier to calibrate and monitor

**Downside:**
- Less granular than v4 (but v4's granularity was illusory anyway!)

---

### Option B: Acknowledge Correlation (2 dimensions)

If you want to preserve SOME dimensionality:

```yaml
dimensions:
  financial_risk:
    weight: 0.90
    description: |
      Composite of macro, credit, sentiment, valuation, policy, systemic risks.
      Measures overall threat to capital preservation.

    scoring_rubric:
      0-2: Low - Stable conditions
      3-4: Moderate - Some concerns
      5-6: Elevated - Multiple risks
      7-8: High - Serious threats
      9-10: Crisis - Extreme conditions

  evidence_strength:
    weight: 0.10
    description: |
      Quality and specificity of risk signals.
      High = Specific data, clear mechanisms
      Low = Vague concerns, speculation

    scoring_rubric:
      0-2: Weak - Speculation, clickbait, vague fears
      3-4: Moderate - Some data but incomplete
      5-6: Good - Specific indicators, reasonable analysis
      7-8: Strong - Clear data, well-reasoned
      9-10: Very strong - Hard evidence, actionable signals
```

**Why this works:**
- ✅ Still honest (main dimension is "risk level")
- ✅ Evidence quality IS somewhat independent (r=0.76 vs 0.93+)
- ✅ Gives users two scores: "how risky?" + "how confident?"
- ✅ 4x faster training than v4

**Downside:**
- Evidence quality still correlates with risk (r=0.76)
- Might not be worth the extra dimension

---

### Option C: Force Independence (3 dimensions)

Design dimensions that CAN vary independently, with explicit oracle instructions:

```yaml
dimensions:
  systemic_threat:
    weight: 0.60
    description: |
      Overall threat to financial system stability and capital preservation.
      Combines macro, credit, and systemic collapse risks.

    independent_variation: |
      Can be HIGH even when market_psychology is LOW:
      - Example: 2015 energy crisis (high systemic threat to oil sector, low market panic)

      Can be LOW even when market_psychology is HIGH:
      - Example: 2013 taper tantrum (low systemic threat, high sentiment panic)

  market_psychology:
    weight: 0.30
    description: |
      Sentiment extremes (panic or euphoria) and valuation concerns.
      Measures behavioral risks vs fundamental risks.

    independent_variation: |
      Can be HIGH even when systemic_threat is LOW:
      - Example: 2018 Q4 selloff (panic despite strong economy)

      Can be LOW even when systemic_threat is HIGH:
      - Example: 2020 March rebound (systemic threat high but sentiment recovering)

  evidence_quality:
    weight: 0.10
    description: |
      Specificity and actionability of risk signals.

    independent_variation: |
      Can be LOW even when risks are HIGH:
      - Example: Vague recession fears without specific data

      Can be HIGH even when risks are LOW:
      - Example: Well-documented minor risk with clear metrics
```

**Why this MIGHT work:**
- ✅ Explicit instructions on how dimensions vary independently
- ✅ Contrastive examples for each dimension
- ✅ Groups correlated dimensions together (systemic_threat = macro+credit+systemic)

**Risk:**
- May STILL see high correlation if oracle takes cognitive shortcuts
- Needs testing with 100-article pilot

---

## Recommended Approach: Start Simple, Test, Iterate

### Phase 1: Test Option A (1 dimension)

**Week 1:**
1. Create investment-risk v5 with Option A (1D: overall_risk)
2. Generate 100 oracle labels
3. Check: Is there variance in scores? (0-10 scale used?)
4. Check: Do metadata fields work? (risk_components, evidence_quality)

**If successful:**
- Proceed to 5K training examples
- Train model (1.5 hours vs 6 hours!)
- Benchmark vs v4

**Expected result:**
- Similar MAE to v4 (maybe slightly higher but acceptable)
- Much clearer for users
- Much faster everything

### Phase 2: If Option A feels too simple, try Option B (2D)

**If users demand more granularity:**
1. Add evidence_strength as 2nd dimension
2. Test on 100 articles
3. Check correlation: if r < 0.70, proceed
4. If r > 0.80, revert to Option A

### Phase 3: Option C only if necessary

**Only try this if:**
- Users absolutely need dimension breakdown
- You're willing to invest time in oracle iteration
- You can get oracle to follow independence instructions

---

## For Sustainability Tech Innovation v3

### Current Reality (v2)

Even WORSE than investment-risk:
- PC1 = 89.1% variance (single "maturity" factor)
- ALL 8 dimensions r > 0.85 with each other
- Oracle is rating "how mature is this tech" as 1D

### Recommended: Option A (1-3 dimensions)

**Minimal (1D):**
```yaml
dimensions:
  technology_maturity:
    weight: 1.0
    description: "Overall deployment readiness: lab → pilot → commercial → widespread"

metadata:
  maturity_stage: [research, pilot, early_adoption, commercial, mature]
  cost_status: [expensive, improving, competitive, cheap]
  performance_status: [experimental, promising, reliable, proven]
```

**Moderate (3D):**
```yaml
dimensions:
  deployment_reality:
    weight: 0.50
    description: "ACTUAL current deployment scale and market adoption"

  technical_performance:
    weight: 0.35
    description: "Technology reliability and demonstrated results"

  economic_viability:
    weight: 0.15
    description: "Cost competitiveness and supply chain readiness"
```

**Why 3D might work here:**
These three aspects CAN vary independently:
- Deployed widely but unreliable (early EVs)
- Works great but not deployed (advanced nuclear)
- Deployed + reliable but expensive (renewables 2010)

Test with contrastive examples!

---

## For Uplifting v5

### Current Reality (v4)

**Best of the three!**
- PC1 = 75.0% variance (still dominant but less extreme)
- Clear cluster: agency/progress/collective_benefit (r > 0.90)
- Other dimensions more independent (r = 0.55-0.80)

### Recommended: Keep 5-6 dimensions

**Proposed structure:**

```yaml
dimensions:
  concrete_progress:
    weight: 0.30
    description: "Tangible positive outcomes: agency, progress, collective benefit"
    merges: [agency, progress, collective_benefit]

  connection:
    weight: 0.15
    description: "Social bonds, community, relationships"

  innovation:
    weight: 0.15
    description: "Creativity, novelty, breakthrough thinking"

  justice:
    weight: 0.20
    description: "Fairness, equity, addressing wrongs"

  resilience:
    weight: 0.20
    description: "Strength, adaptation, overcoming adversity"

  # Optional 6th:
  wonder:
    weight: 0.10
    description: "Awe, beauty, transcendence"
```

**Why this works:**
- ✅ Merges the tight cluster (agency/progress/benefit)
- ✅ Keeps genuinely independent dimensions (justice, resilience, innovation)
- ✅ Still captures emotional nuance
- ✅ 37.5% reduction (8→5) is meaningful

---

## Decision Framework

Use this flowchart to decide:

```
1. Run dimension redundancy analysis on 100 oracle labels
   ↓
2. Check PC1 variance:

   PC1 > 85%? → Use 1-2 dimensions (single-factor reality)
   PC1 = 70-85%? → Use 2-4 dimensions (moderate clustering)
   PC1 < 70%? → Keep most dimensions (genuine independence)

3. Check correlation pairs:

   All pairs r > 0.90? → Merge into 1 dimension
   Most pairs r > 0.80? → Merge into 2-3 dimensions
   Some pairs r > 0.70? → Merge tight clusters, keep independent ones

4. Design dimensions:

   If merging: Accept correlation, design 1-2 holistic dimensions
   If keeping: Add contrastive examples, enforce independence

5. Test with 100 articles, check redundancy analysis again

6. If redundancy still high:
   → Accept it and use fewer dimensions
   OR
   → Try separate oracle calls (expensive but independent)
```

---

## Summary Recommendations

| Filter | v4 Dims | PC1 | Recommended | Rationale |
|--------|---------|-----|-------------|-----------|
| **investment-risk** | 8 | 87% | **1-2** | Single risk factor dominates |
| **sustainability_tech_innovation** | 8 | 89% | **1-3** | Single maturity factor dominates |
| **uplifting** | 8 | 75% | **5-6** | Some independence exists, merge tight cluster only |

**Starting point for v5:**
- investment-risk: **1D** (overall_risk + metadata)
- sustainability_tech_innovation: **3D** (deployment/performance/economics)
- uplifting: **5D** (merge agency/progress/benefit cluster)

**Test each with 100 articles, analyze redundancy, adjust if needed**

The key is to be **honest about what the oracle actually does** rather than pretending we have 8 independent dimensions when we don't!
