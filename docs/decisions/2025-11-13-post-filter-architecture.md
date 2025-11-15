# Post-Filter Architecture for Tier Classification

**Date:** 2025-11-13
**Status:** Accepted

## Context

Student models output dimensional scores (0-10 scale) for each article. We need a post-processing step to convert these scores into tier classifications for filtering and curation.

**Question:** How should dimensional scores be converted to tiers?

## Decision

**Use a generic post-filter module** (`scripts/postfilter.py`) that:
1. Loads filter configuration (dimensions, weights, tier thresholds)
2. Calculates weighted average of dimensional scores
3. Applies filter-specific rules (gatekeeper rules, content caps)
4. Assigns tier based on thresholds
5. Optionally flags top articles for reasoning generation

## Architecture

```
Student Model Output (dimensional scores)
    ↓
PostFilter.classify()
    ↓
├─ Step 1: Calculate weighted average
│    scores × weights → overall_score
│
├─ Step 2: Apply gatekeeper rules
│    if deployment_maturity < 5.0 → cap overall at 4.9
│    if proof_of_impact < 4.0 → cap overall at 3.9
│
├─ Step 3: Apply content type caps (uplifting filter)
│    if collective_benefit < 6 → cap business news at 4.0
│
├─ Step 4: Assign tier
│    overall_score >= 8.0 → mass_deployment
│    overall_score >= 6.5 → commercial_proven
│    overall_score >= 5.0 → early_commercial
│    ...
│
└─ Step 5: Flag for reasoning (optional)
     if overall_score >= 7.0 → needs_reasoning = True
    ↓
Classification Result
{
  "tier": "commercial_proven",
  "overall_score": 6.73,
  "dimensional_scores": {...},
  "needs_reasoning": false,
  "applied_rules": ["No gatekeeper rules triggered"],
  "tier_description": "Commercially viable, multiple deployments..."
}
```

## Implementation

**Module:** `scripts/postfilter.py`

**Usage:**
```python
from filters.{filter_name}.v1.postfilter import PostFilter

# Initialize with filter path
pf = PostFilter("filters/sustainability_tech_deployment/v1")

# Classify article from model scores
scores = {
    "deployment_maturity": 7.2,
    "technology_performance": 6.8,
    "cost_trajectory": 7.0,
    # ... all 8 dimensions
}

result = pf.classify(scores, flag_reasoning_threshold=7.0)
print(result["tier"])  # "commercial_proven"
print(result["needs_reasoning"])  # False
```

**Command-line usage:**
```bash
python scripts/postfilter.py \
    --filter filters/sustainability_tech_deployment/v1 \
    --scores '{"deployment_maturity": 7.2, "technology_performance": 6.8, ...}' \
    --flag-reasoning-threshold 7.0
```

## Key Design Decisions

### 1. Generic Module (Not Filter-Specific)

**Rationale:**
- Same logic works for all filters (uplifting, sustainability_tech_deployment, etc.)
- Filter-specific rules read from `config.yaml`
- No code changes needed when adding new filters

**Benefits:**
- ✅ Maintainability: One module to maintain
- ✅ Consistency: All filters use same tier assignment logic
- ✅ Flexibility: New filters just need config.yaml

### 2. Weighted Average Calculation

**Formula:**
```python
overall_score = Σ(dimension_score × dimension_weight)
```

**Example (sustainability_tech_deployment):**
```
deployment_maturity: 7.0 × 0.20 = 1.40
technology_performance: 6.5 × 0.15 = 0.98
cost_trajectory: 7.0 × 0.15 = 1.05
scale_of_deployment: 6.5 × 0.15 = 0.98
market_penetration: 7.0 × 0.15 = 1.05
technology_readiness: 6.5 × 0.10 = 0.65
supply_chain_maturity: 6.0 × 0.05 = 0.30
proof_of_impact: 6.5 × 0.05 = 0.33
                        Overall = 6.73
```

**Validation:** Weights must sum to 1.0 (checked at initialization)

### 3. Gatekeeper Rules (sustainability_tech_deployment)

**Purpose:** Hard constraints on critical dimensions

**Rules:**
1. If `deployment_maturity < 5.0` → cap overall score at 4.9
2. If `proof_of_impact < 4.0` → cap overall score at 3.9

**Rationale:**
- Lab/pilot tech can't support "tech works" narrative (deployment_maturity)
- Must have some verified impact data (proof_of_impact)

**Example:**
```python
# All dimensions = 7.0, but deployment_maturity = 4.0
scores = {..., "deployment_maturity": 4.0, ...}
# Without gatekeeper: overall = 6.8
# With gatekeeper: overall = 4.9 (capped)
```

### 4. Content Type Caps (uplifting filter)

**Purpose:** Reduce scores for certain content types

**Rules:**
1. Corporate finance content: max score 2.0
2. Military/security content: max score 4.0 (with exceptions)
3. Business news: max score 4.0 if `collective_benefit < 6`

**Rationale:**
- Corporate finance rarely uplifting (unless worker cooperative, etc.)
- Business news only uplifting if broad collective benefit

**Note:** Full implementation requires article text for keyword matching (triggers/exceptions). Current version implements condition-based caps only.

### 5. Tier Assignment

**Logic:** First matching tier (sorted by threshold descending)

**Example (sustainability_tech_deployment):**
```python
tiers = [
    ("mass_deployment", 8.0),
    ("commercial_proven", 6.5),
    ("early_commercial", 5.0),
    ("pilot_stage", 3.0),
    ("vaporware", 0.0)
]

# overall_score = 6.73 → commercial_proven
# (first tier where 6.73 >= threshold)
```

### 6. Reasoning Flagging

**Purpose:** Identify articles that need oracle-generated reasoning

**Logic:**
```python
if overall_score >= flag_reasoning_threshold:
    needs_reasoning = True
```

**Typical threshold:** 7.0 (top tier articles)

**Use cases:**
- Featured articles for publishing
- User-requested explanations
- Borderline cases requiring review

**Integration with ADR (2025-11-13-regression-only-student-models.md):**
- Stage 1 (bulk): Regression models score all articles
- Stage 2 (selective): Oracle generates reasoning for flagged articles (10-20/day)

## Testing

**Test suite:** `scripts/test_postfilter.py`

**Coverage:**
- Weighted average calculation
- Gatekeeper rule enforcement
- Content type caps
- Tier assignment logic
- Reasoning flagging

**Run tests:**
```bash
python scripts/test_postfilter.py
```

**Expected output:**
```
[PASS] Weighted average tests passed!
[PASS] All sustainability_tech_deployment tests passed!
[PASS] All uplifting tests passed!
[PASS] ALL TESTS PASSED!
```

## Configuration Format

**Filter config.yaml structure:**

```yaml
scoring:
  dimensions:
    dimension_name:
      weight: 0.20  # Must sum to 1.0 across all dimensions
      description: "..."
      scale: "..."

  tiers:
    tier_name:
      threshold: 8.0  # Minimum overall score for this tier
      description: "..."

  gatekeeper_rules:  # Optional, for sustainability filters
    dimension_name:
      threshold: 5.0
      max_overall_if_below: 4.9
      reasoning: "..."

  content_type_caps:  # Optional, for uplifting filter
    content_type_name:
      max_score: 4.0
      condition: "collective_benefit < 6"  # Optional
      triggers:  # Keywords (not yet implemented)
        - "..."
      exceptions:  # Keywords (not yet implemented)
        - "..."
```

## Alternatives Considered

### Alternative 1: Tier Assignment in Student Model

**Approach:** Train model to output tier directly (not dimensional scores)

**Pros:**
- Single-step inference
- No post-processing needed

**Cons:**
- ❌ Loses dimensional information (can't adjust tier thresholds without retraining)
- ❌ Can't change tier definitions without retraining
- ❌ Can't apply conditional rules (gatekeeper, content caps)
- ❌ Less transparent (can't see why article got tier X)

**Decision:** Rejected - Dimensional scores provide flexibility and transparency

### Alternative 2: Tier Classification in Oracle

**Approach:** Oracle outputs dimensional scores + tier

**Pros:**
- Oracle already knows tier thresholds

**Cons:**
- ❌ Violates separation of concerns (ADR: 2025-11-13-remove-tier-classification-from-oracle.md)
- ❌ Can't change tier thresholds without re-prompting oracle
- ❌ Tier logic duplicated in oracle prompts
- ❌ Student models still need post-filter anyway

**Decision:** Rejected - Tier classification is post-processing, not oracle responsibility

### Alternative 3: Filter-Specific Post-Filter Classes

**Approach:** Separate PostFilterSustainability, PostFilterUplifting classes

**Pros:**
- Can optimize each filter's logic

**Cons:**
- ❌ Code duplication (weighted average, tier assignment)
- ❌ Need new class for every filter
- ❌ Harder to maintain consistency

**Decision:** Rejected - Generic module with config-driven rules is simpler

## Consequences

### Positive

- ✅ **Flexibility**: Tier thresholds adjustable without retraining models
- ✅ **Transparency**: Can see dimensional scores + overall score + applied rules
- ✅ **Reusability**: Same module works for all filters
- ✅ **Testability**: Easy to test tier assignment logic
- ✅ **Separation of concerns**: Oracle scores, post-filter classifies
- ✅ **Conditional logic**: Gatekeeper rules, content caps, reasoning flags

### Negative

- ⚠️ **Extra step**: Requires post-processing after model inference
- ⚠️ **Complexity**: More components to maintain (model + post-filter)

### Neutral

- Post-filter latency negligible (<1ms per article)
- Config.yaml becomes source of truth for tier logic

## Future Enhancements

**Planned:**
1. **Keyword-based content caps** (uplifting filter)
   - Scan article text for triggers/exceptions
   - Apply caps based on content type detection

2. **Conditional gatekeeper rules**
   - Exception clauses (e.g., collective_benefit exception if wonder > 7)
   - Uplifting filter has this in config but not implemented yet

3. **Batch processing API**
   - Process multiple articles in parallel
   - Return batch classification results

4. **Confidence scores**
   - Return confidence interval for overall score
   - Based on model MAE and dimensional variance

**Not planned:**
- Tier prediction by student model (defeats purpose of dimensional regression)
- Separate post-filter per filter (generic module sufficient)

## Success Metrics

**Post-filter is successful if:**
- ✅ Works with all filters without code changes
- ✅ Tier assignment matches expected tiers in test cases
- ✅ Gatekeeper rules correctly cap scores
- ✅ Reasoning flagging identifies top 10-20 articles/day
- ✅ Latency < 1ms per article

**Quality checks:**
- All tests pass (`scripts/test_postfilter.py`)
- Tier distributions match expected patterns (most articles in mid tiers)
- Flagged articles are indeed high-quality (human review)

## References

- Implementation: `scripts/postfilter.py`
- Tests: `scripts/test_postfilter.py`
- Related ADR: `docs/decisions/2025-11-13-remove-tier-classification-from-oracle.md`
- Related ADR: `docs/decisions/2025-11-13-regression-only-student-models.md`
- Filter configs: `filters/*/v1/config.yaml`

## Version History

### v1.0 (2025-11-13)
- Initial implementation
- Generic module supporting all filters
- Weighted average, gatekeeper rules, content caps
- Reasoning flagging for selective oracle usage
- Test suite validates sustainability_tech_deployment and uplifting filters
