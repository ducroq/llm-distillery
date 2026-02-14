# Filter Packages

This directory contains versioned filter packages for LLM Distillery. Each filter is a complete, self-contained system for evaluating content on specific semantic dimensions.

**Organization**:
- `filters/` - Active filters (production or in-development)
- `filters/todo/` - Planned filters (design/planning phase)

**See also**: [SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md) for comprehensive filter status and training progress.

---

## Harmonized Architecture

All filters follow consistent structure to enable flexible deployment and retraining:

### Core Principles

1. **Oracle Output Discipline**
   - Oracle outputs **dimensional scores ONLY** (0-10 per dimension + reasoning)
   - Oracle does NOT output tier/stage classifications
   - Postfilter classifies tiers based on dimensional scores

2. **Benefits**
   - ✅ **Flexible thresholds**: Change tier boundaries without re-labeling training data
   - ✅ **Clean distillation**: Student models learn dimensional scoring, not classification
   - ✅ **Separation of concerns**: Oracle scores, postfilter classifies

3. **Harmonized Prompt Structure**
   - Header (Purpose, Version, Philosophy, Oracle Output)
   - Tier/Stage Definitions (reference only)
   - ## PROMPT TEMPLATE
   - Scope/Rules (what's in/out of scope)
   - ARTICLE: {title}\n{text}
   - Dimensions with inline `❌ CRITICAL FILTERS`
   - Output Format (dimensional scores + metadata)
   - Examples (optional)
   - CHANGELOG

4. **Inline Filters**
   - Every dimension MUST have inline filters for fast model compatibility
   - Format: `**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**`

---

## Filter Package Structure

Each filter package contains:

```
filters/<filter-name>/v<version>/
├── prompt-compressed.md    # Oracle prompt (ALWAYS USED for scoring)
├── prompt-extended.md      # Extended version with examples (optional, documentation)
├── prefilter.py           # Fast rule-based filter (noise reduction)
├── postfilter.py          # Tier classification from dimensional scores
├── config.yaml            # Weights, thresholds, tier boundaries, deployment specs
├── README.md              # Filter documentation
└── validation_report.md   # Calibration and validation results
```

---

## Active Filters

### Production Ready

#### 1. uplifting v5 ✅ Deployed

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Status**: Deployed on HuggingFace Hub (private)

**Metrics**: Val MAE 0.68, 10K training articles

**Dimensions (6)**: human_wellbeing_impact, social_cohesion_impact, justice_rights_impact, evidence_level (gatekeeper), benefit_distribution, change_durability

**Location**: [filters/uplifting/v5/](uplifting/v5/)

---

#### 2. investment-risk v5 ✅ Production Ready

**Purpose**: Identify investment risk signals for defense-first portfolio management.

**Status**: Production ready

**Metrics**: Test MAE 0.484 (excellent), 10K training articles

**Location**: [filters/investment-risk/v5/](investment-risk/v5/)

---

#### 3. sustainability_technology v1 ✅ Deployed

**Purpose**: Rate sustainable technology that WORKS - deployed tech, working pilots, validated research.

**Status**: Deployed on HuggingFace Hub

**Metrics**: Test MAE 0.690

**Location**: [filters/sustainability_technology/v1/](sustainability_technology/v1/)

---

#### 4. sustainability_technology v2 ✅ Complete

**Purpose**: Updated sustainability technology filter with improved prefilter.

**Status**: Complete (prefilter + model)

**Metrics**: Val MAE 0.71, 7,990 training samples, Prefilter FP Block 88.2%, TP Pass 89.0%

**Location**: [filters/sustainability_technology/v2/](sustainability_technology/v2/)

---

### In Development

#### 5. cultural-discovery v3 ✅ Production Ready

**Purpose**: Discoveries about art, culture, history + cross-cultural connections.

**Status**: Production ready, deployed on HuggingFace Hub

**Target**: ovr.news (Wisdom tab), Busara

**Metrics**: Val MAE 0.77, 7,827 training articles (merged random + screened datasets)

**Key achievement**: 39% improvement on medium-tier, 23% on high-tier vs v1 (screen+merge strategy per ADR-003)

**Versions**: v1 (baseline, MAE 0.82) → v2 (screening experiment) → v3 (merged, production)

**Dimensions (5)**: discovery_novelty, heritage_significance, cross_cultural_connection, human_resonance, evidence_quality (gatekeeper)

**Location**: [filters/cultural-discovery/v3/](cultural-discovery/v3/) (v1, v2 also available)

---

#### 6. belonging v1 📋 Needs Assessment

**Status**: Needs assessment and development

**Location**: [filters/belonging/v1/](belonging/v1/)

---

#### 7. ai-engineering-practice v2 🚫 Blocked

**Status**: Blocked - needs FluxusSource hardware engineering sources

**Location**: [filters/ai-engineering-practice/v2/](ai-engineering-practice/v2/)

---

#### 8. nature_recovery v1 📋 Early Development

**Status**: Concept and README complete, 8 dimensions defined

**Next**: Harmonized prompt, prefilter

**Location**: [filters/nature_recovery/v1/](nature_recovery/v1/)

---

#### 9. signs_of_wisdom v1 📋 Early Development

**Status**: Concept and README complete

**Next**: Harmonized prompt, prefilter

**Challenge**: Wisdom is rare in news

**Location**: [filters/signs_of_wisdom/v1/](signs_of_wisdom/v1/)

---

### Cross-Cutting Components

#### Commerce Prefilter SLM 🔄 Needs Rework

**Purpose**: ML classifier for commerce/promotional content detection.

**Status**: v1 complete but needs redo - concerns about multilingual embeddings and context size

**Location**: [filters/common/commerce_prefilter/](common/commerce_prefilter/)

---

## Planned Filters (filters/todo/)

**Future filters** (design/planning phase):

1. **future-of-education** - Educational innovation and transformation
2. **seece** - Social, economic, and environmental corporate excellence
3. **sustainability_economic_viability** - Economic aspects of sustainability
4. **sustainability_policy_effectiveness** - Policy impact and effectiveness

**Note**: Move to active filters when development begins.

---

## Filter Development

### Using the Filter Development Guide Agent

For comprehensive lifecycle guidance (9 phases: planning → deployment):

```
Use the filter-development-guide agent to:
1. Start a new filter from scratch
2. Review existing filter for production readiness
3. Debug validation issues
```

**Documentation**: [docs/agents/filter-development-guide.md](../docs/agents/filter-development-guide.md)

### Using the Filter Harmonizer Agent

For consistency checking and validation:

```
Use the filter-harmonizer agent to:
1. Validate filter structure and architecture
2. Check oracle output format (dimensional scores only)
3. Generate harmonization reports
```

**Documentation**: [docs/agents/filter-harmonizer.md](../docs/agents/filter-harmonizer.md)

---

## Development Workflow

### 1. Planning Phase
- Define purpose, philosophy, target use case
- Design dimensions (6-8 recommended)
- Define tiers/stages (classification scheme)
- Identify gatekeeper rules

### 2. Architecture Phase
- Create harmonized prompt (scope → gatekeepers → article → dimensions)
- Design prefilter (minimal, avoid false negatives)
- Design postfilter (tier classification from scores)
- Create config.yaml

### 3. Validation Phase
- Score 50-100 validation articles
- Analyze score distribution
- Verify gatekeeper enforcement
- Calibrate thresholds

### 4. Training Data Phase
- Score 5K+ articles with oracle
- Validate training data quality
- Split into train/val sets (90/10)

### 5. Training Phase
- Fine-tune Qwen2.5-1.5B on oracle scores (LoRA)
- Validate model performance
- Target MAE < 0.80

### 6. Deployment Phase
- Production testing
- Deploy model
- Monitor performance

---

## Key Documents

- **[SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md)** - Comprehensive system status and filter progress
- **[DOCUMENTATION_IMPROVEMENTS.md](../DOCUMENTATION_IMPROVEMENTS.md)** - Documentation audit and improvement plan
- **[docs/agents/filter-development-guide.md](../docs/agents/filter-development-guide.md)** - Full lifecycle guidance
- **[docs/agents/filter-harmonizer.md](../docs/agents/filter-harmonizer.md)** - Consistency checking
- **[docs/agents/FILTER_CHECKLIST.md](../docs/agents/FILTER_CHECKLIST.md)** - Development checklist

---

## Quick Reference

### Oracle Output Discipline

**✅ CORRECT** - Oracle outputs dimensions only:
```json
{
  "dimension_name": {"score": 7, "reasoning": "..."},
  "another_dimension": {"score": 6, "reasoning": "..."},
  "metadata_field": "value",  // Simple metadata OK
  "confidence": "HIGH"
}
```

**❌ WRONG** - Oracle outputs classification:
```json
{
  "dimension_name": {"score": 7, "reasoning": "..."},
  "signal_tier": "RED_FLAG",        // ← Classification, should be in postfilter
  "deployment_stage": "commercial", // ← Classification, should be in postfilter
  ...
}
```

### Prefilter Philosophy

- **False negatives** (blocking good articles): **CRITICAL FAILURE** - lost forever
- **False positives** (passing bad articles): Acceptable - oracle catches them
- Target: <10% false negative rate
- Philosophy: Noise reduction, not quality filtering

### Inline Filters

Every dimension MUST have inline filters:

```markdown
**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Filter criterion 1
- Filter criterion 2
- Filter criterion 3

**If NONE of above filters match, score normally:**
- 0-2: Description | 3-4: Description | 5-6: Description | 7-8: Description | 9-10: Description
```

---

**Last Updated**: 2026-02-14
