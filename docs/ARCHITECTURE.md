# LLM Distillery - Architecture

**Last Updated**: 2025-11-17
**Status**: Harmonized architecture across all filters

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Knowledge Distillation Pipeline](#knowledge-distillation-pipeline)
3. [Filter Package Architecture](#filter-package-architecture)
4. [Oracle Output Discipline](#oracle-output-discipline)
5. [Harmonized Prompt Structure](#harmonized-prompt-structure)
6. [Prefilter Philosophy](#prefilter-philosophy)
7. [Inline Filters](#inline-filters)
8. [Training Pipeline](#training-pipeline)
9. [Deployment Architecture](#deployment-architecture)

---

## Core Principles

### 1. Oracle Output Discipline

**Rule**: Oracles output **dimensional scores ONLY** (0-10), never tier/stage classifications.

**Why**: Enables changing tier thresholds without re-labeling training data. Separates concerns: oracle scores dimensions, postfilter classifies tiers.

**Benefits**:
- ✅ **Flexible thresholds**: Adjust tier boundaries without retraining
- ✅ **Clean distillation**: Student models learn dimensional scoring, not classification logic
- ✅ **Separation of concerns**: Oracle scores, postfilter classifies
- ✅ **Maintainability**: Update tier logic in postfilter without touching oracle prompt

**Example**:

```json
// ✅ CORRECT - Oracle outputs dimensions only
{
  "deployment_maturity": {"score": 7, "reasoning": "Commercial deployment with multiple customers"},
  "technology_performance": {"score": 6, "reasoning": "Demonstrated efficiency gains"},
  "cost_trajectory": {"score": 5, "reasoning": "Approaching grid parity"},
  // ... 5 more dimensions ...
  "primary_technology": "solar",  // Simple metadata OK
  "confidence": "HIGH"
}

// ❌ WRONG - Oracle outputs classification
{
  "deployment_maturity": {"score": 7, "reasoning": "..."},
  "technology_performance": {"score": 6, "reasoning": "..."},
  "deployment_stage": "commercial_proven",  // ← Classification, should be in postfilter
  "tier": "validated",                      // ← Classification, should be in postfilter
  ...
}
```

### 2. Three-Stage Processing Pipeline

```
┌─────────────┐
│   Article   │ Raw input (title + text)
└──────┬──────┘
       │
       v
┌─────────────────┐
│   Prefilter     │ Fast rule-based (blocks obvious noise)
└──────┬──────────┘ Python regex/keyword matching
       │            Target: <10% false negatives
       │
       v
┌─────────────────┐
│  Oracle (LLM)   │ Scores 8 dimensions (0-10) + reasoning
└──────┬──────────┘ Output: dimensional_scores ONLY
       │            Models: Gemini Flash (cheap), Claude Sonnet (accurate)
       │
       v
┌─────────────────┐
│   Postfilter    │ Computes tier/stage from dimensional scores
└──────┬──────────┘ Applies gatekeeper rules
       │            Python logic, configurable thresholds
       v
┌─────────────────┐
│  Final Output   │ Tier + dimensional scores + metadata
└─────────────────┘
```

**Key Insight**: Each stage has a distinct purpose:
- **Prefilter**: Noise reduction (not quality filtering!)
- **Oracle**: Dimensional scoring (not classification!)
- **Postfilter**: Tier classification (flexible thresholds)

### 3. Knowledge Distillation

**Teacher-Student Model**:
- **Teacher (Oracle)**: Large foundation model (Gemini Flash, Claude Sonnet)
  - Expensive per inference ($0.00015-0.003 per article)
  - Slow (2-5 seconds per article)
  - High quality dimensional scoring

- **Student**: Small local model (Qwen2.5-7B)
  - Zero cost per inference (local)
  - Fast (20-50ms per article)
  - Learns to replicate oracle judgments
  - Target: 92-96% accuracy

**Workflow**:
1. Oracle scores 5K+ articles → training dataset
2. Student model fine-tuned on oracle scores
3. Student deployed for production inference
4. 100x cost reduction, 50x speed improvement

---

## Knowledge Distillation Pipeline

### Phase 1: Ground Truth Generation

```bash
# 1. Calibrate prefilter (500 articles)
python -m ground_truth.calibrate_prefilter \
  --filter filters/uplifting/v4 \
  --sample-size 500

# 2. Calibrate oracle (100 articles, compare models)
python -m ground_truth.calibrate_oracle \
  --filter filters/uplifting/v4 \
  --models gemini-flash,gemini-pro,claude-sonnet

# 3. Score training data (5K+ articles)
python -m ground_truth.batch_scorer \
  --filter filters/uplifting/v4 \
  --target-scored 5000 \
  --llm gemini-flash
```

**Cost**: ~$0.75 per filter @ $0.00015/article (Gemini Flash)

### Phase 2: Training

```bash
# Fine-tune Qwen2.5-7B on oracle scores
python -m training.knowledge_distillation \
  --filter uplifting \
  --version v4 \
  --scored-data datasets/scored/uplifting_v4 \
  --base-model Qwen/Qwen2.5-7B \
  --epochs 3
```

**Requirements**: 16GB+ GPU (RTX 4090, A100), ~2-4 hours

### Phase 3: Deployment

```bash
# Run inference with trained model
python -m inference.predict \
  --filter inference/deployed/uplifting_v4/ \
  --input articles.jsonl \
  --output predictions.jsonl
```

**Performance**: 20-50ms per article, $0 cost

---

## Filter Package Architecture

Each filter is a self-contained package:

```
filters/<filter-name>/v<version>/
├── prompt-compressed.md    # Oracle prompt (ALWAYS USED for scoring)
├── prompt-extended.md      # Extended version with examples (optional)
├── prefilter.py           # Fast rule-based filter (noise reduction)
├── postfilter.py          # Tier classification from dimensional scores
├── config.yaml            # Weights, thresholds, tier boundaries, deployment specs
├── README.md              # Filter documentation
└── validation_report.md   # Calibration and validation results
```

### config.yaml Structure

```yaml
filter:
  name: "uplifting"
  version: "4.0"
  updated: "2025-11-17"

dimensions:
  - name: "agency"
    weight: 1.0
  - name: "progress"
    weight: 1.0
  # ... 6 more dimensions

tiers:
  impact:
    min_score: 7.0
    description: "High impact uplifting content"
  connection:
    min_score: 4.0
    description: "Moderate uplifting value"
  not_uplifting:
    max_score: 3.9
    description: "Not uplifting"

gatekeeper_rules:
  - dimension: "collective_benefit"
    threshold: 5.0
    action: "IF collective_benefit < 5.0 THEN max_overall = 3.0"
    exceptions:
      - "wonder >= 7"
```

---

## Oracle Output Discipline

### What Oracles Should Output

**Dimensional scores** (0-10 per dimension):
- Score value (integer 0-10)
- Per-dimension reasoning (brief justification)

**Simple metadata** (NOT classification):
- primary_technology, asset_classes_affected, geographic_scope
- confidence (HIGH/MEDIUM/LOW)
- time_horizon (IMMEDIATE/SHORT_TERM/LONG_TERM)

**Overall reasoning**: Why these scores, what's the overall picture

### What Oracles Should NOT Output

**❌ Tier/stage classifications**:
- deployment_stage (commercial_proven, pilot, research, concept)
- signal_tier (RED_FLAG, YELLOW_WARNING, GREEN_OPPORTUNITY)
- tier (impact, connection, not_uplifting)

**Why**: These are computed by postfilter from dimensional scores. If oracle outputs them:
- Can't change thresholds without re-labeling training data
- Student model learns classification logic (not dimensional scoring)
- Tight coupling between oracle and tier definitions

### Postfilter Classification

Postfilter computes tiers from dimensional scores using configurable logic:

```python
# Example: investment-risk postfilter
def classify_tier(scores: dict) -> str:
    macro = scores['macro_risk_severity']
    credit = scores['credit_market_stress']
    systemic = scores['systemic_risk']
    evidence = scores['evidence_quality']
    actionability = scores['actionability']

    # RED FLAG: High risk + strong evidence + actionable
    if (macro >= 7 or credit >= 7 or systemic >= 8) \
       and evidence >= 5 and actionability >= 5:
        return 'RED_FLAG'

    # YELLOW WARNING: Moderate risk
    elif (macro >= 5 or credit >= 5 or systemic >= 6) \
         and evidence >= 5 and actionability >= 4:
        return 'YELLOW_WARNING'

    # GREEN OPPORTUNITY: Low valuation, extremes
    elif scores['valuation_risk'] <= 3 \
         and scores['market_sentiment_extremes'] >= 7:
        return 'GREEN_OPPORTUNITY'

    # BLUE EDUCATIONAL: Framework improvement
    elif evidence >= 6 and actionability < 4:
        return 'BLUE_EDUCATIONAL'

    # NOISE: Everything else
    else:
        return 'NOISE'
```

**Benefits**:
- Change thresholds (e.g., RED_FLAG requires macro >= 8 instead of 7)
- No need to re-label training data
- Just update postfilter logic and redeploy

---

## Harmonized Prompt Structure

All filter prompts follow consistent structure:

### 1. Header
```markdown
# Filter Name v{version}

**Version**: {version}
**Updated**: {date}
**Purpose**: {one-sentence purpose}
**Philosophy**: {guiding principle in quotes}

**Oracle Output**: This oracle outputs DIMENSIONAL SCORES ONLY (0-10 per dimension).
Tier/stage classification is computed by post-processing, NOT by this oracle.
```

### 2. Tier/Stage Definitions (Reference Only)
```markdown
## Tier/Stage Definitions (Reference - NOT for oracle to output)

These definitions guide dimensional scoring but are NOT output by the oracle:

### Tier 1: High Impact
- Description...
- Dimensional characteristics: dimension_a >= 7, dimension_b >= 6

### Tier 2: Moderate
...
```

**Note**: These help oracle understand what different score levels mean, but oracle does NOT output tier classifications.

### 3. Prompt Template Marker
```markdown
## PROMPT TEMPLATE

The following is the actual prompt sent to the oracle (everything above is context).
```

### 4. Scope and Rules
```markdown
**SCOPE**: What articles are in scope vs out of scope

**CRITICAL FILTERS**:
- Block criterion 1
- Block criterion 2
- Block criterion 3

**GATEKEEPER RULES**:
- IF condition THEN action
```

### 5. Article Placement
```markdown
---

**ARTICLE:**

**Title**: {title}

**Text**: {text}

---
```

**Critical**: ARTICLE must come AFTER scope/gatekeepers, BEFORE dimensions. This ensures oracle reads rules before evaluating content.

### 6. Dimensions with Inline Filters
```markdown
## Dimensions

### 1. dimension_name (0-10)

**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Filter criterion 1
- Filter criterion 2
- Filter criterion 3

**If NONE of above filters match, score normally:**
- **0-2**: Description
- **3-4**: Description
- **5-6**: Description
- **7-8**: Description
- **9-10**: Description
```

**Every dimension MUST have inline filters** (fast model compatibility).

### 7. Output Format
```markdown
## Output Format (JSON)

{
  "dimension_1": {"score": <0-10>, "reasoning": "..."},
  "dimension_2": {"score": <0-10>, "reasoning": "..."},
  ...
  "metadata_field": "value",
  "confidence": "HIGH|MEDIUM|LOW"
}

// NO tier/stage classification fields!
```

### 8. Examples (Optional)
```markdown
## Examples

### Example 1: High Score Article
[Example with dimensional scores and reasoning]

### Example 2: Low Score Article
[Example with dimensional scores and reasoning]
```

### 9. Changelog
```markdown
## CHANGELOG

### v{current} ({date})
- Change description

### v{previous} ({date})
- Previous changes
```

---

## Prefilter Philosophy

### Purpose
**Noise reduction**, NOT quality filtering.

### Key Principle
- **False negatives** (blocking good articles): **CRITICAL FAILURE** - lost forever
- **False positives** (passing bad articles): Acceptable - oracle catches them

### Target
- <10% false negative rate
- 50-80% pass rate (depends on data quality)

### Design Rules

**DO**:
- Block obvious out-of-scope content (e.g., "pilot" in aviation context for tech filter)
- Use fast keyword/regex matching
- Be conservative - when uncertain, pass the article
- Measure false negative rate on validation set

**DON'T**:
- Try to filter quality (oracle's job!)
- Block based on nuanced criteria
- Over-optimize pass rate (prefer false positives)
- Use complex NLP (defeats purpose of fast prefilter)

### Example: sustainability_tech_innovation v1

**Option A (Too Aggressive)**: 16% pass rate, 62.7% false negatives → ❌ FAILED

**Option D (Minimal Filtering)**: 68% pass rate, 23.9% false negatives → ✅ DEPLOYED

```python
# Option D prefilter - Minimal filtering
BLOCK_KEYWORDS = [
    'hospital', 'medical device', 'clinical trial',  # Medicine
    'cloud computing', 'data center', 'cybersecurity',  # IT
    'banking', 'cryptocurrency', 'fintech',  # Finance
    'airline pilot', 'flight training',  # Aviation pilots
]

REQUIRE_KEYWORDS = [
    'climate', 'energy', 'renewable', 'solar', 'wind',
    'carbon', 'emissions', 'sustainability', 'green'
]
```

---

## Screening Filters (Training Data Enrichment)

### Purpose

Screening filters solve the **regression-to-mean problem** in training data collection for needle-in-haystack filters.

**Problem:** When 94% of random articles score low (< 4.0), models learn to predict the mean (~2.0) because that minimizes overall error. This makes them useless for finding rare high-quality content.

**Solution:** Screen articles BEFORE oracle scoring to enrich the training distribution with signal-bearing content.

### Workflow

```
WITHOUT Screening (Regression-to-Mean):
Raw articles ─────────────────────→ Oracle ─────→ Training
                                    (94% zeros)
                                    Model learns: "predict 2.0 for everything"

WITH Screening (Enriched Distribution):
Raw articles ──→ Screening Filter ──→ Oracle ──→ Training
                 (reject 60-80%,      (~50% zeros,
                  enriches signal)     richer gradient)
                                    Model learns: "distinguish 3 from 7"
```

### When to Use

- **Use screening:** When random corpus is >80% low-scoring (weighted avg < 4.0)
- **Skip screening:** When filter scope matches corpus well (e.g., tech filter on tech news)

### Screening vs Prefilter

| Aspect | Screening Filter | Prefilter |
|--------|------------------|-----------|
| **Purpose** | Training data enrichment | Inference noise reduction |
| **When used** | Before oracle scoring | Before model inference |
| **Aggressiveness** | Aggressive (reject 60-85%) | Conservative (pass 50-80%) |
| **False negatives** | Acceptable (10-20%) | Critical failure (< 10%) |
| **False positives** | Critical failure | Acceptable (oracle catches) |
| **Location** | `screening_filter.py` | `prefilter.py` |

**Key insight:** A good prefilter is NOT a good screening filter. They optimize for opposite goals.

### Target Distribution After Screening

| Score Range | Random Corpus | After Screening |
|-------------|---------------|-----------------|
| Low (0-3) | ~85% | ~50-60% |
| Medium (4-6) | ~12% | ~30-35% |
| High (7-10) | ~3% | ~10-15% |

### Implementation

See:
- [ADR-003: Screening Filter for Training Data Enrichment](adr/003-screening-filter-for-training-data.md)
- [Screening Filter Template](templates/screening-filter-template.md)
- [Filter Development Guide - Phase 5](agents/filter-development-guide.md#phase-5-training-data-collection)

---

## Inline Filters

### Why Required
Fast models (Gemini Flash) may skip top-level scope sections. Inline filters ensure every dimension has filtering logic directly embedded.

### Format
```markdown
**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Filter criterion 1 (specific, actionable)
- Filter criterion 2
- Filter criterion 3
```

### Example: investment-risk macro_risk_severity

```markdown
### 1. macro_risk_severity (0-10)

**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Stock picking (analysis of individual stocks without macro context)
- FOMO/speculation (hot stocks, meme stocks, get-rich-quick)
- Personal finance advice (budgeting, savings accounts, credit cards)
- Cryptocurrency pumping (price predictions, to-the-moon rhetoric)

**If NONE of above filters match, score normally:**
- **0-2**: No macro risk signals, routine market updates
- **3-4**: Minor concerns, limited systemic implications
- **5-6**: Moderate risk signals, potential portfolio impact
- **7-8**: Significant macro risks, clear portfolio defense needed
- **9-10**: Severe systemic threats, immediate action required
```

### Guidelines
- 3-5 filter criteria per dimension
- Specific, actionable (not vague)
- Directly relevant to dimension being scored
- If ANY criterion matches, score 0-2 (low)

---

## Training Pipeline

### Data Preparation
```
Oracle Scores (5K articles)
    ↓
[Validation Split] → 10% held-out test set
    ↓
[Training Split] → 90% for fine-tuning
    ↓
[Data Augmentation] → Optional: paraphrasing, back-translation
    ↓
Training Dataset (JSONL)
```

### Model Architecture

**Base Model**: Qwen2.5-7B-Instruct
- 7 billion parameters
- Instruction-tuned
- Good reasoning capabilities

**Fine-tuning Approach**: Multi-dimensional regression
- Input: Article text (title + body)
- Output: 8 dimensional scores (0-10) + reasoning
- Loss: MSE on dimensional scores + optional reasoning quality

**Training Hyperparameters**:
```yaml
epochs: 3
batch_size: 8
learning_rate: 5e-5
warmup_steps: 100
gradient_accumulation: 4
```

### Validation

**Metrics**:
- Per-dimension MAE (Mean Absolute Error)
- Overall accuracy (within ±1 of oracle score)
- Tier classification accuracy (after postfilter)
- Reasoning quality (optional)

**Target Performance**:
- 92-96% accuracy on tier classification
- MAE < 1.0 per dimension
- Reasoning coherent and relevant

---

## Deployment Architecture

### Production Pipeline

```
Input Articles (JSONL)
    ↓
┌──────────────┐
│  Prefilter   │ Fast rule-based (Python)
└──────┬───────┘ Blocks obvious noise
       │
       v (50-80% pass)
┌──────────────┐
│ Student Model│ Qwen2.5-7B fine-tuned (local GPU)
└──────┬───────┘ Outputs dimensional scores
       │
       v
┌──────────────┐
│  Postfilter  │ Computes tiers from scores (Python)
└──────┬───────┘ Applies gatekeeper rules
       │
       v
Output (Tier + Scores + Metadata)
```

### Deployment Options

**Option 1: Local GPU** (recommended for batch processing)
- Hardware: RTX 4090, A100, or similar
- Throughput: 20-50ms per article
- Cost: $0 per article (one-time hardware)
- Best for: Large daily batches (1000+ articles)

**Option 2: Cloud GPU** (recommended for API service)
- Service: Modal, Replicate, or custom K8s
- Throughput: 50-100ms per article (includes network)
- Cost: ~$0.0001-0.0005 per article
- Best for: On-demand API, variable load

**Option 3: Hybrid** (recommended for production)
- Prefilter: Serverless (AWS Lambda, Cloudflare Workers)
- Model: GPU instance (Modal, Replicate)
- Postfilter: Serverless
- Benefits: Auto-scaling, cost-effective

### Performance Benchmarks

| Component | Latency | Throughput | Cost/Article |
|-----------|---------|------------|--------------|
| Prefilter | <5ms | 200/sec | $0 |
| Student Model (GPU) | 20-50ms | 20-50/sec | $0 (local) |
| Postfilter | <10ms | 100/sec | $0 |
| **Total Pipeline** | **30-65ms** | **15-45/sec** | **$0** |

vs. Oracle (Gemini Flash):
- Latency: 2-5 seconds
- Cost: $0.00015/article
- **100x faster, infinite cost reduction**

---

## Key Documents

- **This file**: Architecture principles and design patterns
- **[README.md](README.md)**: Project overview and quick start
- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)**: Current system status
- **[filters/README.md](filters/README.md)**: Filter development guide
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes
- **[docs/agents/filter-harmonizer.md](docs/agents/filter-harmonizer.md)**: Consistency checking
- **[docs/agents/filter-development-guide.md](docs/agents/filter-development-guide.md)**: Lifecycle guidance

---

**Last Updated**: 2025-11-17 (Harmonization milestone)
