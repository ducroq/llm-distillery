# Filter Packages

This directory contains versioned filter packages for LLM Distillery. Each filter is a complete, self-contained system for evaluating content on specific semantic dimensions.

**Organization**:
- `filters/` - Active filters (production or in-development)
- `filters/todo/` - Planned filters (design/planning phase)

**See also**: [SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md) for comprehensive filter status and training progress.

---

## Harmonized Architecture (November 2025)

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

### 1. uplifting v4 ✅ Production-Ready

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Philosophy**: "MEANING not TONE"

**Status**: Harmonized, validated (16 samples), ready for training data generation

**Dimensions (8)**:
- agency, progress, collective_benefit (gatekeeper), connection, innovation, justice, resilience, wonder

**Pre-filter blocks**:
- Corporate finance (unless worker coop/public benefit)
- Military buildups (unless peace/demilitarization)
- Business news without collective benefit

**Oracle Output**: Flat dimensional scores (0-10) + overall reasoning

**Tier Classification** (postfilter):
- impact, connection, not_uplifting (based on dimensional scores)

**Location**: [filters/uplifting/v4/](uplifting/v4/)

---

### 2. investment-risk v3 ✅ Ready for Training

**Purpose**: Identify investment risk signals for defense-first portfolio management focused on capital preservation.

**Philosophy**: "You can't predict crashes, but you can prepare for them."

**Status**: Harmonized, ready for training data generation (queued after sustainability_tech_innovation)

**Dimensions (8)**:
- macro_risk_severity, credit_market_stress, market_sentiment_extremes, valuation_risk, policy_regulatory_risk, systemic_risk, evidence_quality (gatekeeper), actionability

**Pre-filter blocks**:
- FOMO/speculation (hot stocks, meme stocks, crypto pumping)
- Stock picking (unless macro context)
- Affiliate marketing (broker links, promo codes)
- Clickbait (sensationalist headlines)
- Academic research papers (unless actionable)

**Oracle Output**: Dimensional scores (0-10) + reasoning + metadata (risk indicators, asset classes, time horizon)

**Tier Classification** (postfilter):
- 🔴 RED FLAG, 🟡 YELLOW WARNING, 🟢 GREEN OPPORTUNITY, 🔵 BLUE EDUCATIONAL, ⚫ NOISE

**Changes from v2**:
- v2 → v3: Removed signal_tier from oracle output (moved to postfilter)
- Clean fork to ensure training data has no classification artifacts

**Location**: [filters/investment-risk/v3/](investment-risk/v3/)

---

### 3. sustainability_tech_innovation v1 🔄 Scoring Training Data

**Purpose**: Rate sustainable technology that WORKS - deployed tech, working pilots, validated research (not theory or vaporware).

**Philosophy**: "Pilots and research need real results, not just theory."

**Status**: Harmonized, validation complete (31/50 articles), scoring 5K training articles (in progress)

**Dimensions (8)**:
- deployment_maturity (gatekeeper), technology_performance, cost_trajectory, scale_of_deployment, market_penetration, technology_readiness, supply_chain_maturity, proof_of_impact (gatekeeper)

**Pre-filter** (Option D: Minimal Filtering):
- Block obvious out-of-scope (IT infrastructure, medicine, finance, airline pilots)
- Block infrastructure disruption (protests, strikes)
- Require climate/energy mention
- Pass rate: 68% (vs 16% for v1.0)
- False negative improvement: 62% reduction (84 → 32 blocked articles)

**Oracle Output**: Dimensional scores (0-10) + per-dimension reasoning + metadata (primary_technology, confidence)

**Tier Classification** (postfilter):
- breakthrough (8.0+), validated (6.0+), promising (4.0+), early_stage (2.0+), vaporware (<2.0)

**Gatekeeper Enforcement**:
- IF deployment_maturity < 3.0 OR proof_of_impact < 3.0 → SET all scores = 1.0
- Status: ✅ Working perfectly (0% violations vs 85.7% in v1.0)

**Key Improvements** (v1.0 → v1.1):
- Fixed gatekeeper enforcement (85.7% FP → 0% FP)
- Optimized prefilter (16% → 68% pass rate)
- Harmonized architecture (dimensional scores only)

**Location**: [filters/sustainability_tech_innovation/v1/](sustainability_tech_innovation/v1/)

---

### 4. sustainability_tech_deployment v3 🔄 Scoring in Progress

**Purpose**: Track deployment at scale (GW-level renewable energy, mass adoption)

**Status**: Scoring training data (background)

**Focus**: Deployment metrics, scaling evidence, infrastructure buildout

**Location**: [filters/sustainability_tech_deployment/v3/](sustainability_tech_deployment/v3/)

---

## Planned Filters (filters/todo/)

**Future sustainability pillar filters** (design/planning phase):

1. **ai_augmented_practice** - AI augmentation for professional practice
2. **future-of-education** - Educational innovation and transformation
3. **seece** - Social, economic, and environmental corporate excellence
4. **sustainability_economic_viability** - Economic aspects of sustainability (cost, profitability, jobs)
5. **sustainability_movement_growth** - Growth of sustainability movement (social momentum, behavior change)
6. **sustainability_nature_recovery** - Nature restoration and recovery (ecosystem health, pollution reduction)
7. **sustainability_policy_effectiveness** - Policy impact and effectiveness (outcomes, replicability, durability)

**Note**: These filters are in early planning stages. Move to active filters (filters/) when development begins.

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
- Fine-tune Qwen2.5-7B on oracle scores
- Validate model performance
- Target: 92-96% accuracy

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

**Last Updated**: 2025-11-17 (Harmonization milestone)
