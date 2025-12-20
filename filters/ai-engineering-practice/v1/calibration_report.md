# AI Engineering Practice Filter - Calibration Report

**Date:** 2025-12-19
**Prefilter Version:** v1.1
**Oracle:** Gemini Flash

---

## v1.1 Recalibration Results

Following prefilter v1.1 updates (expanded to cover ME/EE/Embedded domains), recalibration was performed on 150 fresh articles.

### Content Type Distribution

| Content Type | Count | % |
|-------------|-------|-----|
| research_study | 87 | 58.0% |
| practitioner_account | 59 | 39.3% |
| not_relevant | 3 | 2.0% |
| thought_piece | 1 | 0.7% |

**Note:** Only 2% not_relevant indicates the prefilter v1.1 is working well.

### Dimension Statistics

| Dimension | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| workflow_detail | 5.87 | 1.23 | 0.0 | 7.0 |
| validation_coverage | 6.07 | 1.23 | 0.0 | 9.0 |
| methodological_rigor | 5.39 | 2.06 | 0.0 | 8.0 |
| practitioner_voice | 5.57 | 2.75 | 0.0 | 9.0 |
| educational_applicability | 6.71 | 1.03 | 0.0 | 8.0 |

### Dimension Correlations

|  | workflow | validation | rigor | voice | education |
|--|----------|------------|-------|-------|-----------
| workflow_detail | 1.00 | 0.41 | -0.34 | **0.83** | 0.70 |
| validation_coverage | 0.41 | 1.00 | 0.45 | 0.02 | 0.77 |
| methodological_rigor | -0.34 | 0.45 | 1.00 | **-0.74** | 0.29 |
| practitioner_voice | **0.83** | 0.02 | **-0.74** | 1.00 | 0.36 |
| educational_applicability | 0.70 | 0.77 | 0.29 | 0.36 | 1.00 |

**Key Correlations (Semantically Meaningful):**
- **methodological_rigor ↔ practitioner_voice: -0.74** (Academic papers lack practitioner voice)
- **workflow_detail ↔ practitioner_voice: +0.83** (Practitioners describe workflows)

### Tier Distribution

| Tier | Threshold | Count | % |
|------|-----------|-------|---|
| high | ≥7.0 | 0 | 0% |
| medium | ≥4.0 | 147 | 98% |
| low | <4.0 | 3 | 2% |

**Overall Score:** Mean=5.88, Std=0.94, Range=[0.00, 6.80]

---

## Comparison: v1.0 vs v1.1

| Metric | v1.0 (100 articles) | v1.1 (150 articles) | Status |
|--------|---------------------|---------------------|--------|
| research_study | 62% | 58% | ≈ Same |
| practitioner_account | 36% | 39% | ≈ Same |
| not_relevant | 1% | 2% | ≈ Same |
| Mean overall score | 5.95 | 5.88 | ≈ Same |
| rigor↔voice correlation | -0.83 | -0.74 | ✓ Similar |
| workflow↔voice correlation | +0.75 | +0.83 | ✓ Similar |

**Conclusion:** Prefilter v1.1 produces similar scoring distributions despite expanded domain coverage. The oracle prompt works correctly on the new article mix.

---

## Recommendations

1. **Proceed to Phase 5** - Filter is ready for training data generation
2. **Consider lowering tier thresholds** if more articles should be in "high" tier
3. **Monitor for domain drift** - Ensure ME/EE/Embedded articles are being scored appropriately

---

## Previous Calibration (v1.0 - Archived)

The v1.0 calibration from earlier today is archived in:
`filters/ai-engineering-practice/v1/calibration/v1.0_backup/`
