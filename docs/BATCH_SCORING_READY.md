# Batch Scoring Ready - Production Filters

**Date:** 2025-11-14
**Status:** ✅ Ready for production batch scoring and training

---

## Production-Ready Filters (3)

All filters have been calibrated and validated with the inline filters pattern.

### 1. Uplifting Filter v4

**Path:** `filters/uplifting/v4/`
**Domain:** Human and planetary wellbeing
**Calibration Results:**
- v3 → v4: 87.5% → 0% false positive rate (100% improvement)
- Sample: 43 articles across 3 calibration rounds
- **Production-ready:** ✅ YES

**Expected Performance:**
- Off-topic rejection: >95%
- Professional knowledge/productivity advice: Correctly rejected
- Doom-framed content: Correctly filtered

### 2. Investment-Risk Filter v2

**Path:** `filters/investment-risk/v2/`
**Domain:** Macro financial risk (capital preservation)
**Calibration Results:**
- v1 → v2: 50-75% → 25-37% false positive rate (~50% improvement)
- Calibration: 47 articles (seed 1000)
- Validation: 45 articles (seed 2000)
- **Production-ready:** ✅ YES

**Expected Performance:**
- Stock picking rejection: 67% better than v1
- NOISE filtering: 53% → 69%
- Trade-off accepted: 25-37% FP rate (oversensitive is acceptable for capital preservation)

**Known Edge Cases:**
- Company-specific macro analysis (e.g., Apple/China dependence)
- Acceptable: Better to have false warnings than miss real risks

### 3. Sustainability Tech Deployment Filter v2

**Path:** `filters/sustainability_tech_deployment/v2/`
**Domain:** Deployed climate/sustainability technology
**Calibration Results:**
- v1 → v2: 5.9% → 4.3% false positive rate (28% improvement)
- Calibration: 17 articles (seed 1000)
- Validation: 23 articles (seed 2000)
- **Production-ready:** ✅ YES

**Expected Performance:**
- Generic IT infrastructure rejection: 100% improvement (Kubernetes-type errors eliminated)
- Consumer appliances with environmental marketing: May have ~4% FP rate
- Baseline was already good (5.9%), so modest improvement expected

**Known Edge Cases:**
- Consumer electronics with "energy efficiency" claims (e.g., Dyson vacuums)
- Acceptable: 4.3% FP rate is low enough for production

---

## Cross-Filter Pattern Validation

**Inline Filters Pattern: Proven Across 3 Domains**

| Filter | Domain | v1 FP Rate | v2 FP Rate | Improvement | Generic IT Errors |
|--------|--------|------------|------------|-------------|-------------------|
| Uplifting | Wellbeing | 87.5% | 0% | 100% reduction | Eliminated ✅ |
| Investment-risk | Finance | 50-75% | 25-37% | ~50% reduction | Eliminated ✅ |
| Sustainability_tech | Climate | 5.9% | 4.3% | 28% reduction | Eliminated ✅ |

**Universal Finding:** All three filters had generic IT infrastructure false positives in v1 (Kubernetes, APIs, cloud tools) that were completely eliminated in v2 with inline filters.

---

## Batch Scoring Commands

### Dataset Info
- **Source:** `datasets/raw/historical_dataset_19690101_20251108.jsonl`
- **Total articles:** 47,967
- **Estimated labeling time:** ~4-6 hours per filter @ Gemini Flash rate limits
- **Estimated cost:** ~$8-12 per filter (47,967 articles × $0.001/1000 tokens)

### 1. Label with Uplifting Filter v4

```bash
python -m ground_truth.batch_scorer \
  --filter filters/uplifting/v4 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/uplifting_v4 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited
```

**Expected output:**
- `datasets/scored/uplifting_v4/uplifting/scored_batch_*.jsonl`
- Each article will have `uplifting_analysis` field with dimensional scores + signal_tier

### 2. Label with Investment-Risk Filter v2

```bash
python -m ground_truth.batch_scorer \
  --filter filters/investment-risk/v2 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/investment_risk_v2 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited
```

**Expected output:**
- `datasets/scored/investment_risk_v2/investment-risk/scored_batch_*.jsonl`
- Each article will have `investment-risk_analysis` field with:
  - Signal tier: RED, YELLOW, GREEN, BLUE, NOISE
  - Dimensional scores (macro_risk_severity, credit_market_stress, etc.)
  - Risk indicators (yield_curve_inversion, bank_stress_signals, etc.)

### 3. Label with Sustainability Tech Deployment Filter v2

```bash
python -m ground_truth.batch_scorer \
  --filter filters/sustainability_tech_deployment/v2 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/sustainability_tech_v2 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited
```

**Expected output:**
- `datasets/scored/sustainability_tech_v2/sustainability_tech_deployment/scored_batch_*.jsonl`
- Each article will have `sustainability_tech_deployment_analysis` field with:
  - overall_score (0-10)
  - Dimensional scores (deployment_maturity, technology_performance, etc.)
  - deployment_stage (mass_deployment, commercial_proven, etc.)

---

## Running All Filters in Parallel (Recommended)

To maximize throughput and minimize total processing time, run all 3 filters in parallel:

```bash
# Terminal 1: Uplifting
python -m ground_truth.batch_scorer \
  --filter filters/uplifting/v4 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/uplifting_v4 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited

# Terminal 2: Investment-Risk
python -m ground_truth.batch_scorer \
  --filter filters/investment-risk/v2 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/investment_risk_v2 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited

# Terminal 3: Sustainability Tech
python -m ground_truth.batch_scorer \
  --filter filters/sustainability_tech_deployment/v2 \
  --source datasets/raw/historical_dataset_19690101_20251108.jsonl \
  --output-dir datasets/scored/sustainability_tech_v2 \
  --llm gemini-flash \
  --batch-size 50 \
  --max-batches unlimited
```

**Note:** You may hit Gemini Flash API rate limits. The batch labeler will automatically retry with exponential backoff.

---

## Monitoring Progress

Each batch labeler creates:
- `distillation.log` - Full processing log
- `metrics.jsonl` - Per-article timing and success metrics
- `session_summary.json` - Final statistics

**Check progress:**
```bash
# View latest session summary
cat datasets/scored/uplifting_v4/uplifting/session_summary.json

# Count labeled articles
wc -l datasets/scored/uplifting_v4/uplifting/scored_batch_*.jsonl

# Check for errors
grep "FAILED" datasets/scored/uplifting_v4/uplifting/distillation.log
```

---

## Training Data Preparation

After batch scoring completes, prepare training data with:

```bash
python scripts/prepare_training_data.py \
  --filter uplifting \
  --labeled-dir datasets/scored/uplifting_v4/uplifting \
  --output-dir training/uplifting
```

This will create:
- `training/uplifting/train.jsonl` (80% of data)
- `training/uplifting/validation.jsonl` (10% of data)
- `training/uplifting/test.jsonl` (10% of data)

Repeat for each filter.

---

## Expected Timeline

**Batch Scoring (on different machine with good API limits):**
- Uplifting: 4-6 hours
- Investment-Risk: 4-6 hours
- Sustainability Tech: 4-6 hours
- **Total (parallel):** 4-6 hours
- **Total (sequential):** 12-18 hours

**Training Preparation:**
- Per filter: 10-20 minutes
- **Total:** 30-60 minutes

**Model Training:**
- Depends on your training infrastructure
- See `training/README.md` for guidance

---

## Quality Assurance

**Before training, validate a sample of labeled articles:**

```bash
# Sample 100 random articles from each filter
python -c "
import json, random
articles = [json.loads(line) for line in open('datasets/scored/uplifting_v4/uplifting/scored_batch_001.jsonl')]
sample = random.sample(articles, min(100, len(articles)))
for a in sample:
    print(f\"{a['title'][:60]}: {a['uplifting_analysis']['signal_tier']}\")
"
```

**Expected distribution for uplifting v4:**
- TIER_0/1/2: ~60-70% (off-topic, low uplifting)
- TIER_3/4: ~20-30% (moderately uplifting)
- TIER_5: ~5-10% (highly uplifting)

**Red flags:**
- All articles in same tier → Filter issue
- Generic IT scoring high → Inline filters not working
- Entertainment/gaming scoring high → Scope drift

---

## Calibration Reports

**Full calibration documentation:**
- Uplifting: `docs/decisions/2025-11-14-inline-filters-for-fast-models.md` (v1.0 section)
- Investment-Risk: `datasets/working/investment_risk_calibration_report.md`
- Sustainability Tech: `datasets/working/sustainability_tech_calibration_report.md`

**Pattern documentation:**
- Inline Filters ADR: `docs/decisions/2025-11-14-inline-filters-for-fast-models.md` (v1.2)
- Prompt Calibration Workflow: `docs/decisions/2025-11-13-prompt-calibration-before-batch-labeling.md`

---

## Next Steps

1. ✅ **Batch label all 47,967 articles** with 3 production filters (run on different machine)
2. ⏭️ **Prepare training data** using `prepare_training_data.py`
3. ⏭️ **Train regression models** (see `training/README.md`)
4. ⏭️ **Evaluate model performance** vs oracle
5. ⏭️ **Deploy to production** if model meets quality thresholds

---

## Summary

**Production-Ready Status:**
- ✅ 3 filters calibrated and validated
- ✅ Inline filters pattern proven across 3 domains
- ✅ False positive rates acceptable (0-37% depending on filter)
- ✅ Ready for batch scoring 47,967 articles
- ✅ Estimated cost: $24-36 total ($8-12 per filter)
- ✅ Estimated time: 4-6 hours (parallel) or 12-18 hours (sequential)

**You are ready to proceed with batch scoring on your training machine!**
