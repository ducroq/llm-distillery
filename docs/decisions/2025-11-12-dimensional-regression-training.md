# Use Dimensional Regression (Not Tier Classification) for Training

**Date:** 2025-11-12
**Status:** Accepted

## Context

Initial QA agents focused on tier classification accuracy, but this misaligned with the actual training objective. Training uses dimensional scores (8 scores per article, 0-10 range) as targets, not tier labels.

Tier labels are derived metadata based on weighted sums of dimensions, but:
- Oracle may use holistic assessment for tier assignment
- Tier boundaries in config.yaml are guidelines, not strict rules
- Training goal is to predict dimensional scores accurately, not tiers
- Tier mismatches between oracle labels and config thresholds are expected

## Decision

Train models on multi-dimensional regression (8 scores per article, 0-10 range) rather than tier classification.

**Training targets:**
- Input: [title + content]
- Output: [dim1_score, dim2_score, ..., dim8_score]
- Loss: MSE(predicted_scores, ground_truth_scores)

**Tier labels:** Metadata only, not used in training or evaluation.

## Consequences

### Positive
- Model learns fine-grained distinctions (not just 3-4 tier buckets)
- More flexible: Can derive any tier scheme from dimensional scores
- Better alignment with oracle labeling (dimensions are primary, tiers secondary)
- QA focuses on dimensional score quality, not tier accuracy
- Richer output for downstream applications

### Negative
- Cannot directly evaluate "tier classification accuracy"
- Requires explaining why tier mismatches are acceptable
- More complex to interpret (8 scores vs 1 tier)

### Neutral
- Evaluation metrics: MAE and RMSE per dimension (not accuracy/F1)
- Tier labels remain useful for stratified splitting and human interpretation
- Can always compute tier from dimensional scores if needed

## Alternatives Considered

- **Multi-class classification (tier prediction):** Rejected because loses fine-grained information and doesn't match how oracle labels are structured (dimensions → tier, not tier directly)

- **Ordinal regression:** Rejected because ties training to specific tier boundaries, reducing flexibility. Also assumes equal spacing between tiers which may not be valid.

- **Both dimensional + tier (multi-task learning):** Rejected as unnecessarily complex. Tier can always be derived from dimensions, so predicting both is redundant.

## Implementation Notes

**Dataset QA:**
- Use dimensional regression criteria (see `docs/agents/templates/dimensional-regression-qa-agent.md`)
- Focus on: completeness, validity (0-10 range), range coverage, variance
- Tier mismatches are informational only, not failures

**Training format:**
- Input: `{"id": "...", "title": "...", "content": "..."}`
- Labels: `[7, 8, 6, 5, 7, 4, 6, 5]` as array (not JSON objects)
- Dimension order maintained from config.yaml

**Evaluation:**
- Primary metrics: MAE and RMSE per dimension
- Secondary: Overall MAE/RMSE across all dimensions
- Can compute "tier accuracy" from dimensional predictions for comparison

**Documentation:**
- README files for datasets clarify "Training Data Format" vs "Metadata"
- Tier labels explicitly marked as "metadata only"

## References

- `docs/agents/templates/dimensional-regression-qa-agent.md` - QA template
- `datasets/labeled/uplifting/README.md` - Training Data Format section
- `datasets/labeled/sustainability_tech_deployment/README.md` - Training Data Format section
- `scripts/prepare_training_data.py` - Generic preparation script
