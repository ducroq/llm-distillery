# Oracle Model Recommendation: Gemini Flash vs Pro

**Filter**: Sustainability Tech Deployment v1.0
**Date**: 2025-11-08
**Calibration Sample**: 55 articles (500 sampled, 11% passed prefilter)

---

## Executive Summary

**Recommendation: Use Gemini Flash for oracle labeling.**

**Key Finding**: Gemini Pro systematically assigns zeros (0.0 scores) to non-sustainability articles and extremely low scores (1.0-1.1) to borderline cases, resulting in 91% vaporware classification vs Flash's 53%. Manual analysis reveals that:

1. **Pro is overly conservative** - Treats ANY article not explicitly about climate tech deployment as completely irrelevant (score 0)
2. **Flash discriminates better** - Correctly identifies when articles discuss mature technologies (biosimilars, transmission infrastructure) even if not climate-focused
3. **Both models correctly identify non-sustainability content** - The disagreements are about how to score off-topic articles, not about sustainability tech

**Critical Issue**: The filter is designed for sustainability tech deployment, but is being fed many non-sustainability articles (pharmaceutical supply chains, physics papers, Dutch politics). Pro treats ALL off-topic articles as 0-score vaporware. Flash gives them credit for deployment maturity IF they discuss real deployed technology.

**Resolution**: Flash's approach is more appropriate because:
- Provides training signal about "deployed vs vaporware" distinction across domains
- Better differentiation between commercial products (biosimilars) vs pure theory (physics papers)
- More informative labels for downstream 7B model training

---

## Detailed Disagreement Analysis

### Case 1: Opinion: Onshoring could threaten the resilient supply chain for biosimilars and generics

**Article Type**: Opinion piece about pharmaceutical supply chains (NOT CLIMATE TECH)

**Flash Assessment**: 7.0 (commercial_proven)
- Deployment Maturity: 7 - "Biosimilars and generics are commercially available and widely used"
- Technology Readiness: 8 - "Well-established technologies with proven reliability"
- Market Penetration: 7 - "Generics have significant market share"

**Pro Assessment**: 0.0 (vaporware)
- ALL dimensions: 0
- No reasoning provided

**Manual Analysis**:

This article is about **pharmaceutical biosimilars and generic drugs**, NOT climate technology. However:

- **Flash is technically correct**: Biosimilars ARE deployed commercial technology with mature supply chains
- **Pro is technically correct**: This has NOTHING to do with climate/sustainability deployment

**Winner**: **Flash** - While off-topic, Flash correctly recognizes that biosimilars are mature commercial technology. Pro's blanket zero is unhelpful for training a model to distinguish deployment maturity across contexts.

---

### Case 2: Optimal transmission expansion modestly reduces decarbonization costs of U.S. electricity

**Article Type**: Academic research (arXiv preprint) on transmission grid optimization (CLIMATE RELEVANT)

**Flash Assessment**: 6.4 (early_commercial)
- Deployment Maturity: 7 - Transmission infrastructure is deployed
- Technology Readiness: 8 - Proven infrastructure
- Scale: 7 - Grid-scale deployment

**Pro Assessment**: 1.1 (vaporware)
- ALL dimensions: 1 (except Technology Readiness: 2)
- "There is no deployment, so the scale is zero"

**Manual Analysis**:

This article discusses **transmission grid expansion** - a REAL, DEPLOYED climate technology (electricity transmission lines).

- **Flash is correct**: Transmission infrastructure IS deployed, mature, and operating at scale
- **Pro is wrong**: Pro treats this as theoretical because it's an arXiv preprint modeling study, but the underlying technology (transmission lines) is very real

**Winner**: **Flash** - Pro completely missed that transmission grids are actual deployed infrastructure, not vaporware. Flash correctly recognized the real-world deployment context.

---

### Case 3: Byaidu/PDFMathTranslate

**Article Type**: Software tool for PDF translation (NOT CLIMATE TECH)

**Flash Assessment**: 6.25 (early_commercial)
- Deployment Maturity: 7 - "Released software with Docker/CLI support"
- Technology Readiness: 7 - "Functional software"

**Pro Assessment**: 1.0 (vaporware)
- ALL dimensions: 1
- "This is a theoretical model, so there is no market penetration"

**Manual Analysis**:

This is a **PDF translation tool** - completely off-topic for climate tech, but it IS deployed software (Docker images, CLI, GUI).

- **Flash is correct**: This is real, deployed software with users
- **Pro is wrong**: This is NOT a "theoretical model" - it's working software

**Winner**: **Flash** - Pro misidentified deployed software as theoretical. Flash correctly recognized deployment status even for off-topic content.

---

### Case 4: States sue to stop Trump cancellation of $7 billion solar grant program

**Article Type**: Policy news about solar energy funding (CLIMATE RELEVANT)

**Flash Assessment**: 6.15 (early_commercial)
- Deployment Maturity: 7 - Solar is deployed
- Technology Readiness: 7 - Proven technology
- Scale: 7 - Large-scale programs

**Pro Assessment**: 1.0 (vaporware)
- ALL dimensions: 1
- No reasoning provided

**Manual Analysis**:

This article discusses a **$7B solar grant program** - directly relevant to climate tech deployment policy.

- **Flash is correct**: Solar technology IS widely deployed and commercially proven
- **Pro is wrong**: Assigning 1.0 to solar deployment is absurd given solar's market maturity

**Winner**: **Flash** - Pro's assessment that solar is "vaporware" is completely disconnected from reality. Flash correctly recognizes solar as deployed technology.

---

### Case 5: Aangevallen Nederlands vrachtschip nog in brand op zee bij Jemen

**Article Type**: News about Dutch cargo ship attack (NOT CLIMATE TECH)

**Flash Assessment**: 3.9 (pilot_stage)
- Deployment Maturity: 7 - Ships are deployed
- Technology Readiness: 9 - Very mature technology
- But low scores on cost trajectory (1), market penetration (1), proof of impact (1)

**Pro Assessment**: 0.0 (vaporware)
- ALL dimensions: 0
- No reasoning provided

**Manual Analysis**:

This article is about a **cargo ship attack** - completely off-topic for climate tech.

- **Flash is semi-reasonable**: Recognizes that cargo ships are real deployed technology, but scores low on climate impact (correctly)
- **Pro's zero assignment**: Treats off-topic article as completely irrelevant

**Winner**: **Tie** - Both approaches are defensible. Flash gives credit for maritime technology deployment (even if off-topic). Pro correctly treats it as irrelevant to the filter.

---

## Pattern Analysis

### Gemini Pro's Systematic Problem

Pro exhibits a **binary classification failure**:

1. **If article mentions sustainability keywords** → Score 1.0-2.0 (still vaporware)
2. **If article doesn't mention sustainability** → Score 0.0 (all zeros across dimensions)

This is NOT useful for training a 7B model because:
- No nuance in deployment maturity assessment
- Cannot distinguish between theoretical physics (Case 2: arXiv abstract) and real deployed tech (Case 1: biosimilars, Case 4: solar)
- Provides minimal training signal for "deployed vs theoretical" classification

### Gemini Flash's Approach

Flash exhibits **contextual assessment**:

1. **Evaluates deployment status regardless of climate relevance**
2. **Distinguishes between mature commercial tech (biosimilars 7.0) vs software tools (6.25) vs theory (would be lower)**
3. **Provides rich training signal** for downstream model about what "deployed" means

### Why Flash is Better for Oracle Labeling

The goal of oracle labeling is to create **high-quality training data** for a 7B model. Flash provides:

1. **Better discrimination**: Can distinguish commercial_proven (7.0) from early_commercial (6.4) from pilot (3.9)
2. **More training signal**: Dimensional scores show nuanced assessment, not blanket zeros
3. **Transfer learning**: Even off-topic articles provide signal about "what is deployed tech" vs "what is vaporware"

Pro's approach provides almost no training signal - everything is either 0 or 1.

---

## Cost Comparison

| Model | Cost per 1K articles | Total Cost (5 filters × 2K samples) | Quality |
|-------|---------------------|-------------------------------------|---------|
| **Gemini Flash** | ~$2 | **~$20** | ✅ Good discrimination |
| **Gemini Pro** | ~$4 | **~$40** | ❌ Binary, no nuance |

Flash is both **cheaper AND higher quality** for this use case.

---

## Recommendation

### Use Gemini Flash for all 5 sustainability filter oracle labeling

**Rationale**:

1. **Better discrimination**: Flash distinguishes deployment maturity levels, Pro assigns blanket zeros
2. **Contextual assessment**: Flash evaluates real-world deployment even for off-topic articles
3. **Cost effective**: 50% cheaper than Pro ($20 vs $40 for all 5 filters)
4. **Training signal**: Flash provides richer labels for 7B model fine-tuning

**Caveat**: Flash may be TOO generous to off-topic articles. Consider:
- Strengthening prefilter to block more off-topic content
- Adding post-processing rule: "If article doesn't mention sustainability keywords, cap score at 3.0"
- Training 7B model to learn that off-topic articles should score low regardless of deployment maturity

---

## Next Steps

1. ✅ Use Gemini Flash as oracle model
2. 🔄 Run full oracle labeling on ~2K samples per filter
3. 🔍 Review Flash's labels for off-topic articles, add post-processing if needed
4. 🎯 Fine-tune Qwen2.5-7B on Flash labels
5. 📊 Evaluate 7B model performance vs Flash oracle

**Estimated Cost for Full Oracle Labeling**:
- 5 filters × 2,000 samples × $0.001/sample = **$10-20 total**

**Timeline**: ~3-5 hours of runtime (API rate limits)

---

## Appendix: Full Dimensional Analysis

### Why Pro Gives All Zeros to Off-Topic Articles

Looking at Pro's responses for Cases 1, 5:

```json
{
  "deployment_maturity": 0,
  "technology_performance": 0,
  "cost_trajectory": 0,
  "scale_of_deployment": 0,
  "market_penetration": 0,
  "technology_readiness": 0,
  "supply_chain_maturity": 0,
  "proof_of_impact": 0
}
```

This suggests Pro interprets the prompt as:
- **"If not sustainability tech, all dimensions = 0"**

Flash interprets the prompt as:
- **"Evaluate deployment dimensions, note if off-topic in assessment"**

For training a downstream model, Flash's interpretation provides more useful signal.
