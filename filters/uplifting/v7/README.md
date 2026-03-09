# Uplifting Filter v7

## What Changed from v6

### Problem: Assessment Dimensions Inflated Scores

In v6, `evidence_level` (weight 0.20) and `benefit_distribution` (weight 0.20) together comprised **40% of the weighted average** and scored high on virtually any well-written news article:

- `evidence_level` measured "how well-documented is the article" — any factual news article with sources scored 6-8
- `benefit_distribution` measured "how many people does this event affect" — any national/global news scored 5-7
- Result: these two dimensions contributed ~2.5-3.0 to every article's weighted average, pushing 71-80% of articles above the MEDIUM threshold (4.0)

### Fix 1: Reframe Assessment Dimensions

Both dimensions now measure properties **of the uplifting outcome specifically**, not of the article as journalism:

| Dimension | v6 (what it measured) | v7 (what it measures) |
|-----------|----------------------|----------------------|
| `evidence_level` | How well-documented is the article? | How well-documented is the **uplifting outcome**? |
| `benefit_distribution` | How many people does the news reach? | Who **receives the uplifting benefit**? |

**Key test case:** A well-sourced ECB interest rate article with Goldman Sachs analyst quotes:
- v6: evidence_level = 7-8 (multiple expert sources, verifiable data)
- v7: evidence_level = 0 (no uplifting outcome exists to verify)

### Fix 2: Rebalance Weights

| Dimension | v6 Weight | v7 Weight | Change |
|-----------|-----------|-----------|--------|
| human_wellbeing_impact | 0.25 | **0.30** | +0.05 |
| social_cohesion_impact | 0.15 | **0.20** | +0.05 |
| justice_rights_impact | 0.10 | **0.15** | +0.05 |
| evidence_level | 0.20 | **0.10** | -0.10 |
| benefit_distribution | 0.20 | **0.10** | -0.10 |
| change_durability | 0.10 | **0.15** | +0.05 |

Impact domains: 50% → **65%**. Assessment: 50% → **35%**. Evidence + distribution: 40% → **20%**.

### Fix 3: ADR-010 Prompt Structure

Adopted from belonging v1 (MAE 0.49 with 7.4K articles — best across all filters):

- **Step 1 scope check** before any scoring — forces scope decision before dimension scoring
- **Anti-hallucination rule** — evidence must be EXACT QUOTE from article
- **Noise detection checklist** — explicit list of out-of-scope patterns
- **Tighter critical filters** — per-dimension filters before scale tables

### Backwards Compatibility

- JSON field names unchanged: `evidence_level`, `benefit_distribution` (not renamed)
- Same 6 dimensions, same gatekeeper logic, same content-type caps
- Prefilter unchanged (reuse v6 prefilter)
- Config structure identical — drop-in replacement for training pipeline

## Oracle Scoring Results (2026-03-08/09)

6,590 articles scored with Gemini Flash in two rounds:
- **Round 1** (5,000 input): Stratified sample from NexusMind output — 500 v6-HIGH, 1,000 v6-MEDIUM, 3,500 random negatives (joined with enrichment cache). 4,975 scored.
- **Round 2** (2,000 input): Active learning enrichment — 2,000 additional v6-HIGH articles to boost HIGH/MEDIUM tiers. 1,615 scored (rate limit on day 1, completed day 2).

### Final Tier Distribution

| Tier | Count | % |
|------|-------|---|
| HIGH (>=7.0) | 100 | 1.5% |
| MEDIUM (4.0-6.9) | 2,082 | 31.6% |
| LOW (<4.0) | 4,408 | 66.9% |
| **Total** | **6,590** | |

### Comparison: HIGH Tier Across Filters

| Filter | Train Size | HIGH Count | HIGH % | MAE |
|--------|-----------|------------|--------|-----|
| belonging v1 | 5,894 | 52 | 0.9% | **0.49** |
| nature_recovery v1 | 2,623 | 13 | 0.5% | **0.54** |
| uplifting v6 | 8,396 | 7 | 0.1% | 0.67 |
| **uplifting v7** | **6,590** | **100** | **1.5%** | **?** |

100 HIGH articles is more than belonging v1 (52, MAE 0.49) and nature_recovery v1 (13, MAE 0.54) — both achieved excellent MAE. The prompt precision (ADR-010) matters more than HIGH count.

### v6 → v7 Score Shift

| v6 Tier | Count | → v7 HIGH | → v7 MEDIUM | → v7 LOW | Avg Shift |
|---------|-------|-----------|-------------|----------|-----------|
| HIGH | 495 | 18 (3.6%) | 328 (66.3%) | 149 (30.1%) | -3.15 |
| MEDIUM | 993 | 7 (0.7%) | 198 (19.9%) | 788 (79.4%) | -3.29 |
| NONE | 3,487 | 27 (0.8%) | 571 (16.4%) | 2,889 (82.9%) | +1.71 |

### Key Observations

1. **Assessment dimension fix confirmed**: evidence_level avg=2.18, benefit_distribution avg=1.90 — no longer inflating generic news (v6 would have scored these 5-7)
2. **v7 finds genuine uplift v6 missed**: 598 v6-NONE articles scored v7-MEDIUM+ (e.g., "Bolivia's Indigenous communities protect 1M acres", "Mobile clinic targets 30K", "Single-dose sleeping sickness treatment")
3. **Score compression mitigations**: isotonic calibration (ADR-008) corrects MSE regression-to-mean post-hoc; active learning (ADR-005) available if needed after initial training

## Training Plan

1. ~~Oracle score ~5,000 articles with v7 prompt~~ DONE (4,975 scored)
2. ~~Spot-check: verify assessment dimensions now score low on non-uplifting content~~ DONE (confirmed)
3. ~~Active learning enrichment: 2,000 more v6-HIGH articles~~ DONE (1,615 scored, total 6,590)
4. Prepare training splits (`training/prepare_data.py`)
5. Train Gemma-3-1B + LoRA on gpu-server
6. Fit isotonic calibration
7. Compare MAE against v6 (target: < 0.55)

## Cost

~7,000 articles scored × $0.001/article = **~$7 in Gemini Flash API costs** (6,590 succeeded).
