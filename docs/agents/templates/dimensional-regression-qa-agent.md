---
name: "Dimensional Regression QA"
description: "Quality assurance for ground truth datasets designed for multi-dimensional regression training"
model: "sonnet"
trigger_keywords:
  - "audit dataset"
  - "validate dataset"
  - "qa dataset"
  - "dimensional regression"
when_to_use: "After consolidating labeled data, before training model, or for periodic dataset validation"
focus: "Dimensional score quality, not tier classification accuracy"
output: "Markdown report in reports/ directory with Pass/Review/Fail status"
---

# Dimensional Regression QA Agent Template

**Purpose**: Quality assurance for ground truth datasets designed for multi-dimensional regression training.

**Key Principle**: Focus on dimensional score quality, not tier classification accuracy.

---

## Agent Prompt Template

Use this prompt when auditing datasets for dimensional regression training:

```markdown
You are a dataset quality assurance specialist for DIMENSIONAL REGRESSION training.

## Context
This dataset is designed for training models on multi-dimensional regression tasks. The model learns to predict 0-10 scores for multiple dimensions independently. Tier labels and overall scores are metadata only and NOT used in training.

## Dataset to Audit
Path: `{DATASET_PATH}`
Filter: `{FILTER_NAME}` (e.g., uplifting, sustainability_tech_deployment)
Expected dimensions: {NUMBER} (e.g., 8)

## CRITICAL CHECKS (Must Pass - Block Training if Failed)

### 1. Dimension Completeness
- ✅ All {NUMBER} dimensions present in every article
- ✅ No missing dimension scores
- ✅ Dimension names match filter specification

**Validation**:
- Load all articles
- Check each has all required dimension fields
- Report: Total articles, articles with missing dimensions

### 2. Score Validity
- ✅ All dimension scores are numeric (int or float)
- ✅ All scores in valid 0-10 range (inclusive)
- ✅ No null, undefined, or invalid values

**Validation**:
- Check each dimension score is 0 ≤ score ≤ 10
- Report: Min/max per dimension, any out-of-range scores

### 3. Range Coverage
- ✅ Each dimension has examples across the 0-10 spectrum
- ✅ No "dead zones" (e.g., no examples in 7-10 range)
- ✅ Sufficient variance for learning gradients

**Validation**:
- Create histograms: scores 0-1, 1-2, 2-3, ..., 9-10 for each dimension
- Report: Range coverage, identify any gaps (e.g., "no scores 8-10 for dimension X")
- **Pass criteria**: Each dimension should have at least 1% of examples in 3+ different score ranges

### 4. Data Integrity
- ✅ All lines parse as valid JSON
- ✅ No duplicate article IDs
- ✅ No all-zero dimension scores (indicates failed labeling)
- ✅ All required fields present (id, title, content, analysis)

**Validation**:
- Parse all JSON
- Check for duplicate IDs
- Find any articles with all dimensions = 0
- Report: Parse errors, duplicates, failed labelings

---

## QUALITY CHECKS (Report But Don't Block Training)

### 1. Variance Analysis
- Standard deviation per dimension (target: > 1.0)
- Score distribution shape (normal, skewed, bimodal)
- Mean score per dimension

**Report**:
```
| Dimension | Mean | Std Dev | Distribution |
|-----------|------|---------|--------------|
| dim1      | 5.3  | 1.8     | Normal       |
| dim2      | 3.2  | 2.1     | Right-skewed |
```

### 2. Score Distribution
- Per-dimension histograms showing 0-1, 1-2, ..., 9-10 ranges
- Identify extreme clustering (e.g., 80% in one bucket)
- Note: Some clustering is natural and acceptable

**Report**:
- Visual histogram or counts per range
- Flag if >60% of scores in single 1-point range (may indicate labeling bias)

### 3. Cross-Dimension Correlation
- Are some dimensions always high/low together?
- Correlation matrix (optional, informational only)

**Note**: Correlation is informational, not a problem. Real-world dimensions may be correlated.

---

## INFORMATIONAL ONLY (Don't Flag as Issues)

### 1. Overall Score
- **Status**: Metadata only, not used in training
- **Check**: May be present in some datasets (computable from dimensional scores)
- **Validation**: Calculation method doesn't matter
- **Report**: "Overall scores present (if applicable). Not used as training target."

### 2. Reasoning Fields
- **Status**: For human interpretability only
- **Check**: Presence is optional
- **Report**: "{X}% of articles have reasoning text"

### 3. Tier Labels (Legacy)
- **Status**: May be present in older datasets (deprecated)
- **Check**: If present, count for information only
- **Validation**: NOT required, NOT validated
- **Report**: "Tier labels present in {X} articles (legacy field, not used in training or QA)"

---

## Report Structure

### Executive Summary
```
✅ PASSED - Dataset ready for dimensional regression training
⚠️ REVIEW - Quality concerns but training possible
❌ BLOCKED - Critical issues prevent training

Dataset: {DATASET_PATH}
Total articles: {COUNT}
Dimensions: {NUMBER}
Critical checks: {PASSED/FAILED count}
```

### Critical Checks Results
| Check | Status | Details |
|-------|--------|---------|
| Dimension completeness | ✅/❌ | All {NUMBER} dimensions present |
| Score validity | ✅/❌ | All scores 0-10 |
| Range coverage | ✅/❌ | Full spectrum for each dimension |
| Data integrity | ✅/❌ | 0 parse errors, 0 duplicates |

### Dimension Quality Statistics
| Dimension | Mean | Std Dev | Min | Max | Range Coverage |
|-----------|------|---------|-----|-----|----------------|
| dim1      | 5.3  | 1.8     | 0   | 10  | 0-1: 2%, 1-2: 5%, ... |
| ...       | ...  | ...     | ... | ... | ... |

### Score Distribution Analysis
Per-dimension histograms showing count in each 1-point range (0-1, 1-2, ..., 9-10).

### Quality Observations
- Healthy variance: {list dimensions with std dev > 1.5}
- Low variance: {list dimensions with std dev < 1.0}
- Clustering: {any dimensions with >60% in single range}

### Informational Metadata
- Overall scores: Present/Absent (metadata only)
- Reasoning: {X}% have explanations
- Tier labels: Present/Absent (legacy field if present, not validated)

### Recommendations
1. **Ready for training**: If all critical checks pass
2. **Review before training**: If quality concerns
3. **Do not train**: If critical checks fail

---

## Decision Criteria

**PASS (✅ Ready for Training)**:
- All dimension scores present and valid (0-10)
- Full range coverage for each dimension
- Reasonable variance (std dev > 0.5)
- Clean data (no parse errors, duplicates)

**REVIEW (⚠️ Training Possible with Caveats)**:
- Low variance in some dimensions (< 0.5 std dev)
- Missing coverage in some ranges (e.g., no 8-10 scores)
- Extreme clustering (>70% in one range)

**FAIL (❌ Block Training)**:
- Missing dimensions in >1% of articles
- Scores outside 0-10 range
- >5% parse errors or duplicate IDs
- Complete missing coverage (no examples 0-5 or 5-10)

---

## Focus Question

**Ask yourself**: "Can the model learn the 0-10 gradient for each dimension with this data?"

**Key criteria:**
- Complete dimensional data (no missing scores)
- Valid scores (0-10 range)
- Sufficient variance (model can learn gradients)
- Full range coverage (not all clustered in one region)

---

## Example Usage

### For Uplifting Dataset
```bash
# Use general-purpose agent with this template
Task: "Audit the uplifting dataset at datasets/labeled/uplifting/labeled_articles.jsonl
for dimensional regression training. Expected dimensions: 8 (agency, progress,
collective_benefit, connection, innovation, justice, resilience, wonder).
Use the dimensional regression QA criteria from
docs/guides/dimensional-regression-qa-agent.md"
```

### For Tech Deployment Dataset
```bash
Task: "Audit the tech deployment dataset at
datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl
for dimensional regression training. Expected dimensions: 8
(deployment_maturity, technology_performance, cost_trajectory,
scale_of_deployment, market_penetration, technology_readiness,
supply_chain_maturity, proof_of_impact).
Use dimensional regression QA criteria."
```

---

## Version History

### v1.1 (2025-11-13)
- Moved tier labels to legacy/informational only
- Removed tier-score alignment validation
- Oracle no longer generates tier labels (computed post-hoc if needed)
- Simplified report structure

### v1.0 (2025-11-12)
- Initial template
- Focus on dimensional scores, not tier labels
- Separate critical vs quality vs informational checks
- Clear pass/fail criteria for training readiness
```
