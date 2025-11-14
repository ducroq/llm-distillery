# Remove Tier Classification from Oracle Labeling

**Date:** 2025-11-13
**Status:** Accepted

## Context

Oracle labeling currently produces:
1. **Dimensional scores** (8 scores per article, 0-10 range) - Used for training
2. **Tier classification** (e.g., impact/connection/not_uplifting) - NOT used for training

The oracle (Gemini Flash) was asked to:
- Score each dimension (primary task)
- Assign a tier based on those scores (secondary task)

**Problems with this approach:**
1. **Redundant**: Tier can be computed from dimensional scores using thresholds in config.yaml
2. **Oracle overhead**: Asking LLM to classify adds cognitive load and tokens
3. **Misleading QA**: Agents validate tier distributions, implying they matter for training (they don't)
4. **Confusion**: Developers think tier labels are important, but training uses dimensional scores only
5. **Prompt complexity**: Oracle prompt is longer and more complex than needed
6. **Error source**: Oracle can assign wrong tier even if dimensional scores are correct

**Current workflow:**
```
Oracle → [dimensional scores + tier label] → Training (uses scores only)
```

**Insight:** Post-filtering handles classification at inference time anyway. The inference pipeline maps dimensional scores → tiers using config thresholds, so oracle doesn't need to do this.

## Decision

**Remove tier classification from oracle labeling entirely.**

Oracle should produce:
- ✅ Dimensional scores (0-10 per dimension)
- ✅ Reasoning (explains scores)
- ❌ Tier classification (removed - will be computed post-hoc if needed)

**Tier assignment (if needed) moves to post-processing:**
```python
def assign_tier(dimensional_scores, tier_boundaries):
    """Compute tier from dimensional scores using config thresholds."""
    overall_score = compute_weighted_score(dimensional_scores)
    for tier_name, threshold in sorted(tier_boundaries.items(), reverse=True):
        if overall_score >= threshold:
            return tier_name
    return lowest_tier
```

**New workflow:**
```
Oracle → [dimensional scores only] → Training (dimensional regression)
                                   ↓
                    Post-filter (scores → tiers at inference time)
```

## Consequences

### Positive
- **Simpler oracle prompts**: Less cognitive load, fewer tokens, faster labeling
- **No tier assignment errors**: Oracle can't assign wrong tier
- **Clearer intent**: Dimensional scores are the only output that matters
- **Agent focus**: QA agents focus on dimensional score quality only
- **Flexibility**: Can change tier thresholds without re-labeling
- **Consistency**: Tier assignment uses exact same logic as inference pipeline

### Negative
- **Manual tier computation**: Need to compute tiers for stratified splitting
- **Migration needed**: Existing prompts reference tier assignment
- **Backward compatibility**: Old labeled data has tier fields (keep for compatibility)

### Neutral
- Stratified splitting can use overall_score ranges instead of tiers
- Tier labels in existing datasets remain as convenience metadata
- Post-filtering applies tier logic consistently across oracle and student

## Alternatives Considered

- **Keep tier assignment, but mark as optional:** Rejected because it doesn't simplify the prompt or remove confusion. Half-measure.

- **Have oracle assign overall score but not tier:** Rejected because overall score is also computable from dimensional scores. Keep it simple: dimensional scores only.

- **Stratify by overall score ranges instead of tiers:** Actually better! Avoids needing tier labels entirely for train/val/test splitting.

## Implementation Notes

### 1. Update Oracle Prompts (All Filters)

**Remove from prompt:**
```markdown
## Tier Classification
Based on the dimensional scores, assign one of the following tiers:
- impact: overall_score >= 7.0
- connection: overall_score >= 4.0
- not_uplifting: overall_score < 4.0
```

**Keep in prompt:**
```markdown
## Output Format
{
  "dimensions": {
    "agency": {"score": 7, "reasoning": "..."},
    "progress": {"score": 8, "reasoning": "..."},
    ...
  }
}
```

**Do NOT include:**
- `overall_score` field (computable from dimensions)
- `tier` field (computable from overall_score)

### 2. Update Agent Templates

**Oracle Calibration Agent:**
- Remove tier distribution from CRITICAL checks
- Remove tier distribution from QUALITY checks
- Remove entirely from report structure

**Dimensional Regression QA Agent:**
- Remove tier distribution validation
- Remove tier mismatch checking
- Focus purely on dimensional score quality

### 3. Post-Processing (If Tiers Needed)

**For stratified splitting:**
```python
# Compute overall score from dimensions
overall_scores = [compute_weighted_score(article['dimensions']) for article in labels]

# Stratify by score ranges instead of tiers
score_ranges = ['low' if s < 4 else 'mid' if s < 7 else 'high' for s in overall_scores]
train, val, test = stratified_split(labels, stratify_by=score_ranges)
```

**For analysis/visualization:**
```python
# Assign tiers using config thresholds (same as inference)
tier = assign_tier(dimensional_scores, tier_boundaries)
```

### 4. Backward Compatibility

**Existing datasets with tier labels:**
- Keep tier fields as-is (don't break anything)
- Mark in README as "convenience metadata, not used in training"
- Scripts should tolerate presence/absence of tier field

**New datasets:**
- Don't generate tier fields
- Compute tiers post-hoc if needed for analysis

### 5. Training Pipeline

**No changes needed!** Training already uses dimensional scores only.

## Migration Plan

### Phase 1: Update Agent Templates (Immediate)
- ✅ Remove tier validation from Oracle Calibration Agent
- ✅ Remove tier validation from Dimensional Regression QA Agent
- ✅ Update docs/agents/agent-operations.md

### Phase 2: Update Oracle Prompts (Completed 2025-11-13)
- ✅ Update uplifting filter prompt to clarify oracle output is dimensional scores only
- ✅ Update sustainability_tech_deployment filter prompt to clarify oracle output
- ✅ Clarify post-processing section in uplifting filter prompt
- ✅ Add oracle output notes to both filter READMEs
- ✅ Clarify tier classification sections in READMEs as post-processing only
- **Result**: Oracle prompts now clearly state they produce dimensional scores only; tier classification is post-processing

### Phase 3: Update Stratified Splitting (Future)
- Modify prepare_training_data.py to stratify by score ranges instead of tiers
- Test on existing datasets
- Verify train/val/test distributions are balanced

## References

- `docs/decisions/2025-11-12-dimensional-regression-training.md` - Training uses dimensional scores, not tiers
- `docs/agents/templates/oracle-calibration-agent.md` - Oracle calibration template (to be updated)
- `docs/agents/templates/dimensional-regression-qa-agent.md` - QA template (to be updated)
- `scripts/prepare_training_data.py` - Uses tier boundaries for stratification (to be updated)

## Discussion

**User insight:** "I made a mistake in having the llm determine the classification... Mapping the regression data onto classes is done in the post-filter anyway."

**Key realization:** If post-filtering handles dimensional scores → tier classification, then oracle doesn't need to do it. Keep oracle focused on what matters: dimensional scores.

**Principle:** Simplicity. Remove everything that isn't essential. Tier classification isn't essential for training, so remove it from oracle.

## Future Considerations

- Improve filter prompts (discuss separately)
- Stratified splitting by score ranges vs tiers (discuss separately)
- Can tier thresholds be learned rather than hardcoded? (research question)
