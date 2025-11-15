# ADR: Dimensional Scoring Terminology

**Date:** 2025-11-15
**Status:** ✅ Accepted
**Deciders:** Technical team

## Context

We produce ground truth training data for dimensional regression models (continuous scores 0-10 per dimension), not classification models (discrete labels). However, our codebase used "labeling" terminology throughout, which is misleading.

**The problem:**
- "Labeling" implies **classification** (discrete labels: positive/negative, class A/B/C)
- We actually do **dimensional regression** (continuous scores: agency=7.2, progress=8.1, etc.)
- This causes confusion about what the system does

## Decision

**Adopt "scoring" terminology** to accurately reflect dimensional regression:

### Terminology mapping:

| Old (Classification) | New (Regression) |
|---------------------|------------------|
| Labeling | Scoring |
| Labeled | Scored |
| Labeler | Scorer |
| Label | Score |
| Ground truth labels | Ground truth scores |
| `batch_scorer.py` | `batch_scorer.py` |
| `datasets/scored/` | `datasets/scored/` |
| `--target-scored` | `--target-scored` |
| `.labeled_ids.json` | `.scored_ids.json` |
| `scored_batch_001.jsonl` | `scored_batch_001.jsonl` |

## Rationale

**Why "scoring" is accurate:**
1. **Dimensional regression**: We predict 8 continuous dimensional scores (0-10 each)
2. **Oracle behavior**: LLM oracle assigns numerical scores, not categorical labels
3. **Training targets**: Models learn to predict continuous values, not discrete classes
4. **Post-processing**: Caps, gatekeeper rules, tiers are applied to scores at inference time

**Why "labeling" was misleading:**
- Implies binary/multiclass classification
- Doesn't convey the dimensional/regression nature
- Confuses new developers about system architecture

**Industry terminology:**
- Classification: "labeling", "labeled data", "class labels"
- Regression: "scoring", "target values", "ground truth scores"
- Our task: Dimensional regression → use regression terminology

## Consequences

### Positive:
- ✅ Clear distinction between classification and regression
- ✅ Accurate terminology for dimensional regression
- ✅ Easier onboarding (correct mental model)
- ✅ Consistent with ML literature

### Negative:
- ⚠️ Breaking change for existing code/docs/processes
- ⚠️ Migration required for old directories/files
- ⚠️ Must update all documentation

### Migration required:
1. ✅ Rename `ground_truth/batch_scorer.py` → `batch_scorer.py`
2. ✅ Update all internal variable/function names
3. ⏳ Update README documentation
4. ⏳ Migrate existing `datasets/scored/` → `datasets/scored/`
5. ⏳ Update training scripts to use new paths
6. ⏳ Update any external documentation/tutorials

## Implementation

**Phase 1 (Completed):**
- ✅ Renamed `batch_scorer.py` → `batch_scorer.py`
- ✅ Updated class: `GenericBatchScorer` → `GenericBatchScorer`
- ✅ Updated all internal terminology
- ✅ Updated CLI arguments: `--target-scored` → `--target-scored`
- ✅ Updated state files: `.labeled_ids.json` → `.scored_ids.json`
- ✅ Updated batch filenames: `scored_batch_` → `scored_batch_`

**Phase 2 (Completed - 2025-11-15):**
- ✅ Updated all documentation (39 markdown files)
- ✅ Updated all references to datasets/labeled/ → datasets/scored/
- ✅ Updated all terminology: labeling→scoring, labeled→scored, labeler→scorer
- ✅ Renamed BATCH_LABELING_READY.md → BATCH_SCORING_READY.md
- ✅ Updated training documentation and all agent templates

**Note:** Actual directory migration (datasets/labeled/ → datasets/scored/) should be done per-filter when needed.

## Example usage

**New (correct):**
```bash
python -m ground_truth.batch_scorer \
    --filter filters/uplifting/v4 \
    --source datasets/raw/historical_dataset.jsonl \
    --output-dir datasets/scored/uplifting_v4 \
    --target-scored 2500 \
    --llm gemini-flash
```

**Output:**
- Dimensional scores (0-10 per dimension)
- Saved to `datasets/scored/uplifting_v4/scored_batch_001.jsonl`
- State in `.scored_ids.json`

## Notes

- This change aligns with our architecture decision to remove post-classification from ground truth generation (see ADR 2025-11-13)
- "Scoring" accurately describes what the oracle does: assign continuous numerical scores to articles across multiple dimensions
- Post-processing (caps, gatekeeper, tiers) happens at inference time, not during ground truth generation
