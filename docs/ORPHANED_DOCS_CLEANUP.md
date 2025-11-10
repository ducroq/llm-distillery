# Orphaned/Redundant Documentation Cleanup Analysis

**Date:** 2025-11-10
**Purpose:** Identify outdated, redundant, or orphaned markdown files for cleanup

---

## Files Recommended for Deletion

### Completed Plans/Status Docs
1. **`RESTRUCTURING_PLAN.md`** (root)
   - Status: COMPLETED (Nov 9, 2025)
   - Reason: Restructuring already done
   - Action: DELETE

2. **`docs/TRAINING_READY.md`**
   - Status: OUTDATED (shows 2,080 labels vs current 4,146)
   - Reason: Old status snapshot from Nov 9
   - Action: DELETE

3. **`docs/TRAINING_16GB_GPU.md`**
   - Status: OUTDATED (describes 4-bit quantization, old configs)
   - Reason: Superseded by `docs/guides/gpu-training-guide.md` and `qwen-finetuning-guide.md`
   - Action: DELETE

### Redundant Strategy Docs (Covered by ADRs)
4. **`docs/stratified_sampling_strategy.md`**
   - Reason: Covered by `docs/decisions/2025-11-09-class-imbalance-strategy.md`
   - Action: DELETE

5. **`docs/handling_imbalanced_datasets.md`**
   - Reason: Covered by `docs/decisions/2025-11-09-class-imbalance-strategy.md`
   - Action: DELETE

---

## Files to Check/Potentially Archive

### Potentially Redundant Root Docs
6. **`REPOSITORY_STRUCTURE.md`**
   - Reason: May duplicate `docs/ARCHITECTURE.md` content
   - Action: REVIEW - Compare with ARCHITECTURE.md, delete if redundant

7. **`docs/ORGANIZATION.md`**
   - Reason: May be outdated organizational doc
   - Action: REVIEW - Check if still relevant

8. **`docs/architecture/overview.md`**
   - Reason: May duplicate `docs/ARCHITECTURE.md`
   - Action: REVIEW - Compare and consolidate if needed

### Training Guides
9. **`training/GPU_TEST_GUIDE.md`**
   - Reason: May be superseded by `docs/guides/gpu-training-guide.md`
   - Action: REVIEW - Check if testing content is useful to keep

10. **`training/HUGGINGFACE_SETUP.md`**
    - Reason: Unclear if still relevant for current workflow
    - Action: REVIEW - Check if HuggingFace setup is documented elsewhere

### Workflow Docs
11. **`docs/workflows/large-ground-truth-workflow.md`**
    - Reason: May be superseded by `docs/guides/ground-truth-generation.md`
    - Action: REVIEW - Compare and consolidate if needed

---

## Legacy Filter Documentation (Keep but Note in OPEN_QUESTIONS)

These are documented in `OPEN_QUESTIONS.md` as future work:

- `docs/EDUCATION_QUICKSTART.md`
- `docs/EDUCATION_INTEGRATION_SUMMARY.md`
- `docs/PROMPT_COMPRESSION_SUMMARY.md`
- `docs/INVESTMENT_RISK_FILTER_SUMMARY.md`
- `docs/filters/education-integration.md`
- `docs/filters/investment-risk.md`
- `docs/filters/prompt-compression.md`

**Action:** KEEP (already tracked in OPEN_QUESTIONS as future filters)

---

## Old Reports (Consider Archiving)

### Tech Deployment Reports (Oct-Nov 2025)
- `reports/tech_deployment_label_validation.md` - Old validation
- `reports/tech_deployment_label_distribution_analysis.md` - Old analysis
- `reports/tech_deployment_corpus_analysis.md` - Old corpus analysis
- `reports/disagreement_analysis.md` - Old oracle disagreement analysis
- `reports/oracle_model_recommendation.md` - Superseded by final calibration
- `reports/session_summary_20251108.md` - Old session notes
- `reports/stratified_sampling_final_assessment.md` - Superseded by ADR

**Action:** ARCHIVE or DELETE (info captured in dataset README and ADRs)

### Uplifting Reports (Old Filter Work)
- `reports/uplifting_calibration_nov1_raw_200.md`
- `reports/uplifting_ground_truth_v1_final_report.md`
- `reports/uplifting_ground_truth_v1_top_articles.md`
- `reports/uplifting_prefilter_cal.md`

**Action:** KEEP (historical record of uplifting filter development)

### Current/Active Reports (KEEP)
- `reports/tech_deployment_oracle_calibration_final.md` - Final calibration
- `reports/ai_practice_oracle_calibration.md` - Recent calibration
- `reports/README.md` - Index file

---

## Technical Implementation Guides (Need Review)

12. **`docs/guides/distillation-logging-system.md`**
    - Status: UNKNOWN
    - Action: REVIEW - Check if logging system is documented or implemented

13. **`docs/guides/json-error-handling-improvements.md`**
    - Status: UNKNOWN
    - Action: REVIEW - Check if error handling improvements are documented elsewhere

14. **`docs/secrets_management.md`**
    - Status: UNKNOWN
    - Action: REVIEW - Check if covered by main docs or still needed

---

## Cleanup Actions Completed (2025-11-10)

### ✅ Deleted (5 files)
- `RESTRUCTURING_PLAN.md` - Completed Nov 9 restructuring
- `docs/TRAINING_READY.md` - Outdated status (2,080 labels vs 4,146)
- `docs/TRAINING_16GB_GPU.md` - Superseded by gpu-training-guide.md
- `docs/stratified_sampling_strategy.md` - Covered by ADR
- `docs/handling_imbalanced_datasets.md` - Covered by ADR

### ✅ Archived (7 files → archive/reports/tech_deployment_early_work/)
- `tech_deployment_label_validation.md`
- `tech_deployment_label_distribution_analysis.md`
- `tech_deployment_corpus_analysis.md`
- `disagreement_analysis.md`
- `oracle_model_recommendation.md`
- `session_summary_20251108.md`
- `stratified_sampling_final_assessment.md`

These reports are now in `archive/reports/tech_deployment_early_work/` for historical reference.

### Review Later (8 files)
- Check if these are redundant or still needed:
  - REPOSITORY_STRUCTURE.md
  - docs/ORGANIZATION.md
  - docs/architecture/overview.md
  - training/GPU_TEST_GUIDE.md
  - training/HUGGINGFACE_SETUP.md
  - docs/workflows/large-ground-truth-workflow.md
  - docs/guides/distillation-logging-system.md
  - docs/guides/json-error-handling-improvements.md
  - docs/secrets_management.md

---

## Summary

**Total Orphaned/Redundant:** ~20 files
**Immediate Deletions:** 5 files
**Archive Candidates:** 7 files
**Review Needed:** 8 files

**Next Steps:**
1. Delete the 5 confirmed outdated files
2. Archive the 7 old reports
3. Review the 8 questionable files to determine if they duplicate current docs
4. Update this analysis based on findings
