# ADR: Model Output Format - Score Arrays vs Full Reasoning

**Date**: 2025-11-09
**Status**: Accepted
**Deciders**: User, Claude

## Context

When training a student model to replicate oracle (Gemini Flash) behavior, we must decide what the model should output. The oracle produces rich outputs including:
- 8 dimension scores (0-10)
- Textual reasoning explaining each score
- Overall score calculation
- Tier classification
- Metadata (timestamp, model version, etc.)

The question: Should the student model learn to produce all of this, or just the essential scores?

## Problem

**Rich Output (Reasoning + Scores + Tier)**:
- Oracle produces detailed explanations for its scoring decisions
- Reasoning helps humans understand and trust the model
- Tier classification provides actionable filtering categories

**Simplified Output (Score Arrays Only)**:
- Scores are the core signal needed for filtering
- Reasoning from 7B model may be lower quality than oracle
- Faster inference, less compute cost

## Options Considered

### Option 1: Full Oracle Output Replication
**Format**:
```json
{
  "dimensions": {
    "deployment_maturity": 8,
    "technology_performance": 7,
    ...
  },
  "reasoning": "This article describes a 300MW solar farm...",
  "overall_score": 7.8,
  "tier": "early_commercial"
}
```

**Pros**:
- ✅ Chain-of-thought learning may improve accuracy
- ✅ Interpretable outputs (can validate reasoning)
- ✅ Human-debuggable (understand model decisions)
- ✅ Multi-task learning (scores + reasoning = richer signal)

**Cons**:
- ❌ **Capacity mismatch**: 7B model can't reason like Gemini Flash
- ❌ Risk of plausible-sounding but incorrect reasoning
- ❌ Slower inference (generating text is expensive)
- ❌ More complex training objective
- ❌ Harder to evaluate (how to score reasoning quality?)

### Option 2: Simplified Score Arrays (SELECTED)
**Format**:
```json
{
  "id": "article-123",
  "title": "...",
  "content": "...",
  "labels": [8, 7, 6, 8, 5, 7, 6, 8],  # dimension scores
  "dimension_names": ["deployment_maturity", "technology_performance", ...]
}
```

**Pros**:
- ✅ **Simple, focused objective**: Learn dimension scores, nothing more
- ✅ **Consistency with uplifting filter**: Proven approach
- ✅ **Fast inference**: Just predict 8 numbers
- ✅ **Easy evaluation**: MAE, MSE, per-dimension accuracy
- ✅ **Avoids quality risk**: No hallucinated reasoning
- ✅ **Sufficient for filtering**: Scores → tier → keep/discard decision

**Cons**:
- ❌ Black box predictions (no explanation)
- ❌ Harder to debug disagreements with oracle
- ❌ May miss learning benefit from chain-of-thought

### Option 3: Hybrid (Scores + Optional Reasoning)
Train two model variants:
1. Scores-only (fast, production)
2. Scores + reasoning (debuggable, validation)

**Pros**:
- ✅ Best of both worlds

**Cons**:
- ❌ Double the training effort
- ❌ Complexity not justified without evidence reasoning helps

## Decision

**We will use simplified score arrays only, matching the uplifting filter format.**

### Rationale

1. **Proven Approach**: Uplifting filter already validated this format works
2. **Simplicity**: Easier training objective = faster iteration
3. **Efficiency**: Production inference only needs scores for filtering
4. **Avoid Quality Risk**: 7B models may produce plausible but incorrect reasoning
5. **Measurable**: Clear metrics (per-dimension MAE, tier accuracy)

### Implementation

**Training Data Format** (`prepare_training_data_tech_deployment.py`):
```python
{
    'id': 'article-123',
    'title': '...',
    'content': '...',  # truncated to ~800 words
    'url': '...',
    'labels': [8, 7, 6, 8, 5, 7, 6, 8],
    'dimension_names': [
        "deployment_maturity",
        "technology_performance",
        "cost_trajectory",
        "scale_of_deployment",
        "market_penetration",
        "technology_readiness",
        "supply_chain_maturity",
        "proof_of_impact"
    ]
}
```

**Model Training**:
- Objective: Predict 8-dimensional score vector
- Loss: Mean Absolute Error (MAE) or Mean Squared Error (MSE)
- Evaluation: Per-dimension MAE, tier classification accuracy

**Post-Processing**:
```python
# Calculate overall score using weights from config.yaml
weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07]
overall_score = sum(score * weight for score, weight in zip(predicted_scores, weights))

# Assign tier
if overall_score >= 8.0:
    tier = 'deployed'
elif overall_score >= 6.0:
    tier = 'early_commercial'
elif overall_score >= 4.0:
    tier = 'pilot'
else:
    tier = 'vaporware'
```

## Consequences

### Positive
- ✅ Fast training iterations
- ✅ Fast inference (100+ tok/sec on 7B model)
- ✅ Easy to evaluate and validate
- ✅ Low risk of reasoning quality issues
- ✅ Matches proven uplifting filter approach

### Negative
- ⚠️ No interpretability (black box scores)
  - **Mitigation**: Compare predictions to oracle on validation set, spot-check disagreements
- ⚠️ May lose learning benefit from chain-of-thought
  - **Mitigation**: Monitor training metrics; if performance poor, reconsider Option 3

### Neutral
- Reasoning can still be obtained from oracle for debugging specific cases
- User can always query oracle for explanation if student prediction is suspicious

## Future Work

**If score-only approach shows poor performance:**
1. Experiment with reasoning in Option 3 (hybrid approach)
2. Use reasoning during training, discard at inference
3. Try intermediate representations (embeddings of reasoning)

**Success metrics for this decision:**
- Per-dimension MAE < 1.5 on validation set
- Tier classification accuracy ≥ 70% per tier
- Deployed tier recall ≥ 60%

If these are not met, revisit the reasoning option.

## References

- Uplifting filter training data: `datasets/training/uplifting/*.jsonl`
- Training script: `scripts/prepare_training_data_tech_deployment.py`
- Filter config: `filters/sustainability_tech_deployment/v1/config.yaml`
- Class imbalance strategy: `docs/decisions/2025-11-09-class-imbalance-strategy.md`

## Related Decisions

- **Class Imbalance Handling**: Uses oversampling to balance training set
- **Content Truncation**: ~800 words to match oracle-student consistency
- **Oracle Selection**: Gemini Flash chosen over Pro for better discrimination
