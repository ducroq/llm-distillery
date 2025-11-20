# Documentation Improvement Plan

**Created**: 2025-11-17
**Status**: Comprehensive audit completed

---

## Executive Summary

Several key documents are outdated and don't reflect:
1. **Harmonization milestone** (Nov 2025) - Oracle output discipline
2. **Filter reorganization** - Active vs todo/ directory structure
3. **New agents** - filter-harmonizer, filter-development-guide
4. **Current filter versions** - investment-risk v3, uplifting v4, sustainability_tech_innovation v1
5. **Architecture improvements** - Dimensional scores only, postfilter tier classification

---

## Priority 1: Critical Updates (High Impact)

### 1. README.md (Project Root) - **OUTDATED**

**Location**: `C:\local_dev\llm-distillery\README.md`

**Issues**:
- ❌ Status shows "October 2025" (now November 2025)
- ❌ References old filter versions (v1 for everything)
- ❌ No mention of harmonization milestone
- ❌ No mention of filter-harmonizer or filter-development-guide agents
- ❌ No mention of investment-risk v3
- ❌ Doesn't reflect oracle output discipline architecture

**Recommended Updates**:

```markdown
## Current Status (November 2025)

### ✅ Completed (NEW)
- **Harmonization Milestone**: All filters follow consistent architecture
  - Oracle outputs dimensional scores ONLY
  - Tier classification in postfilters (flexible thresholds)
  - Consistent structure across uplifting v4, investment-risk v3, sustainability_tech_innovation v1
- **Filter Development Guide Agent**: End-to-end lifecycle guidance (9 phases)
- **Filter Harmonizer Agent**: Automated consistency checking
- **Filter Reorganization**: Active filters (filters/) vs planned (filters/todo/)
- **investment-risk v3**: Harmonized fork with clean architecture
- **sustainability_tech_innovation v1**: Tech that works filter (pilots/research/deployment)
- **Dataset Profiling**: 402K articles from master_dataset (Oct-Nov 2025)

### ✅ Production Filters
- **uplifting v4**: Harmonized, validated (16 samples)
- **investment-risk v3**: Harmonized, ready to score
- **sustainability_tech_innovation v1**: Harmonized, scoring 5K articles
- **sustainability_tech_deployment v3**: Scoring in progress

### 🚧 In Progress
- Training data generation (5K articles per filter)
- Knowledge distillation with Qwen2.5-7B
```

**Priority**: 🔴 **CRITICAL** (first impression document)
**Effort**: Medium (2-3 hours)
**Impact**: High (affects all users)

---

### 2. filters/README.md - **VERY OUTDATED**

**Location**: `C:\local_dev\llm-distillery\filters\README.md`

**Issues**:
- ❌ Lists old versions (uplifting v1, investment-risk v1, sustainability v1)
- ❌ No mention of filters/todo/ organization
- ❌ No mention of harmonization
- ❌ Incorrect filter names (sustainability vs sustainability_tech_innovation)
- ❌ No mention of oracle output discipline
- ❌ No reference to SYSTEM_OVERVIEW.md

**Recommended Updates**:

```markdown
# Filter Packages

This directory contains versioned filter packages for LLM Distillery. Each filter is a complete, self-contained system for evaluating content on specific semantic dimensions.

**Organization**:
- `filters/` - Active filters (production or in-development)
- `filters/todo/` - Planned filters (design phase)

See [SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md) for comprehensive filter status.

---

## Harmonized Architecture (November 2025)

All filters follow consistent structure:
- **Oracle outputs dimensional scores ONLY** (0-10 per dimension)
- **Postfilter classifies tiers** based on dimensional scores
- **Enables flexible thresholds** without retraining
- **Consistent prompt structure** (scope → gatekeepers → article → dimensions → output)

---

## Active Filters

### 1. uplifting v4 ✅ Production
**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.
**Philosophy**: MEANING not TONE
**Status**: Harmonized, validated
**Output**: 8 dimensional scores (agency, progress, collective_benefit, connection, innovation, justice, resilience, wonder)
[View details →](uplifting/v4/README.md)

### 2. investment-risk v3 ✅ Ready to Score
**Purpose**: Capital preservation for defense-first portfolio management
**Philosophy**: "You can't predict crashes, but you can prepare for them."
**Status**: Harmonized, ready for training data generation
**Output**: 8 dimensional scores (macro_risk, credit_stress, sentiment, valuation, policy, systemic, evidence, actionability)
[View details →](investment-risk/v3/README.md)

### 3. sustainability_tech_innovation v1 🔄 In Progress
**Purpose**: Rate sustainable tech that WORKS (deployed, pilots, validated research)
**Philosophy**: "Pilots and research need real results, not just theory."
**Status**: Harmonized, scoring 5K training articles
**Output**: 8 dimensional scores (deployment, performance, cost, scale, market, readiness, supply_chain, proof)
[View details →](sustainability_tech_innovation/v1/README.md)

### 4. sustainability_tech_deployment v3 🔄 In Progress
**Purpose**: Track deployment at scale (GW-level renewable energy)
**Status**: Scoring training data
[View details →](sustainability_tech_deployment/v3/README.md)

---

## Planned Filters (filters/todo/)

- **ai_augmented_practice** - AI augmentation for professional practice
- **future-of-education** - Educational innovation and transformation
- **seece** - Social, economic, and environmental corporate excellence
- **sustainability_economic_viability** - Economic aspects of sustainability
- **sustainability_movement_growth** - Growth of sustainability movement
- **sustainability_nature_recovery** - Nature restoration and recovery
- **sustainability_policy_effectiveness** - Policy impact and effectiveness

---

## Filter Development

Use the **filter-development-guide agent** for comprehensive lifecycle guidance:

```
Use the filter-development-guide agent to:
1. Start a new filter from scratch
2. Review existing filter for production readiness
3. Debug validation issues
```

See [docs/agents/filter-development-guide.md](../docs/agents/filter-development-guide.md) for details.

---

## Quick Reference

**Key Documents**:
- [SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md) - Comprehensive system status
- [docs/agents/filter-harmonizer.md](../docs/agents/filter-harmonizer.md) - Consistency checking
- [docs/agents/filter-development-guide.md](../docs/agents/filter-development-guide.md) - Lifecycle guidance
```

**Priority**: 🔴 **CRITICAL** (filter developers reference this)
**Effort**: Medium (2-3 hours)
**Impact**: High (affects filter development)

---

## Priority 2: Important Updates (Medium Impact)

### 3. SYSTEM_OVERVIEW.md - **UPDATED** ✅

**Status**: ✅ Just updated with filter organization section

**Recent Changes**:
- ✅ Added "Filter Organization" section with active/todo structure
- ✅ Lists all planned filters in todo/
- ✅ Documents organizational principles

**Remaining Updates Needed**:
- Update filter status if scoring completes
- Add production readiness percentages from filter-development-guide report

**Priority**: 🟡 **MEDIUM** (keep current as work progresses)
**Effort**: Low (ongoing, 30 min updates as needed)
**Impact**: High (central reference document)

---

### 4. docs/agents/README.md - **CHECK STATUS**

**Location**: `C:\local_dev\llm-distillery\docs\agents\README.md`

**Needs**:
- Comprehensive list of all agents
- filter-harmonizer (created Nov 17)
- filter-development-guide (created Nov 17)
- Cross-references between agents

**Priority**: 🟡 **MEDIUM** (agent discovery)
**Effort**: Low (1-2 hours)
**Impact**: Medium (helps users find the right agent)

---

## Priority 3: Optional Enhancements (Nice to Have)

### 5. CHANGELOG.md - **MISSING**

**Recommended**: Create project-level changelog

**Content**:
```markdown
# Changelog

All notable changes to LLM Distillery will be documented in this file.

## [November 2025] - Harmonization Milestone

### Added
- **Harmonized Architecture**: All filters follow oracle output discipline
  - Oracle outputs dimensional scores ONLY
  - Tier classification in postfilters
  - Flexible thresholds without retraining
- **filter-development-guide agent**: End-to-end lifecycle guidance (9 phases)
- **filter-harmonizer agent**: Automated consistency checking
- **investment-risk v3**: Clean fork with harmonized architecture
- **sustainability_tech_innovation v1**: Tech that works filter
- **Filter Organization**: Active (filters/) vs planned (filters/todo/)
- **Dataset Profiling**: 402K article master dataset

### Changed
- **uplifting v4**: Harmonization clarifications (content_type is metadata)
- **investment-risk v2→v3**: Removed signal_tier from oracle output
- **Directory Structure**: Reorganized filters into active/todo

### Fixed
- **sustainability_tech_innovation v1**: Gatekeeper enforcement (85.7% FP → 0%)
- **Prefilter optimization**: Option D (68% pass rate, 62% fewer false negatives)

## [October 2025] - Initial Framework

### Added
- Initial filter architecture
- Uplifting v1, sustainability v1, investment-risk v1
- Oracle calibration framework
- Batch labeling system
```

**Priority**: 🟢 **LOW** (nice to have)
**Effort**: Low (1 hour initial, 15 min ongoing)
**Impact**: Low (mostly historical reference)

---

### 6. ARCHITECTURE.md - **MISSING**

**Recommended**: Document core architectural principles

**Content**:
```markdown
# Architecture

## Core Principles

### 1. Oracle Output Discipline

**Rule**: Oracles output dimensional scores ONLY, never tier/stage classifications.

**Why**: Enables changing tier thresholds without re-labeling training data. Separates concerns: oracle scores, postfilter classifies.

**Example**:
```json
// ✅ CORRECT - Oracle outputs dimensions only
{
  "deployment_maturity": {"score": 7, "reasoning": "..."},
  "technology_performance": {"score": 6, "reasoning": "..."},
  ...
}

// ❌ WRONG - Oracle outputs classification
{
  "deployment_maturity": {"score": 7, "reasoning": "..."},
  "deployment_stage": "commercial_proven",  // ← Classification, should be in postfilter
  ...
}
```

### 2. Harmonized Prompt Structure

All filters follow consistent order:
1. Header (Purpose, Version, Philosophy, Oracle Output)
2. Tier/Stage Definitions (reference only)
3. ## PROMPT TEMPLATE
4. Scope/Rules (what's in/out of scope)
5. ARTICLE: {title}\n{text}
6. Dimensions with inline filters
7. Output Format (dimensional scores + metadata)
8. Examples (optional)
9. CHANGELOG

### 3. Inline Filters

Fast models (Gemini Flash) may skip top-level scope sections. Therefore, every dimension MUST have inline filters:

```markdown
**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Filter criterion 1
- Filter criterion 2
```

### 4. Prefilter Philosophy

- **False negatives** (blocking good articles): CRITICAL failure
- **False positives** (passing bad articles): Acceptable, oracle catches them
- Target: <10% false negative rate
```

**Priority**: 🟢 **LOW** (architectural reference)
**Effort**: Medium (3-4 hours)
**Impact**: Medium (helps new contributors understand principles)

---

## Priority 4: Cleanup (Technical Debt)

### 7. Remove Outdated Reports

**Action**: Archive old validation reports that don't reflect current harmonization

**Files to Check**:
- `filters/sustainability_tech_innovation/v1/validation_report.md` (shows v1.0, now v1.1)
- Any reports referencing pre-harmonization architecture

**Priority**: 🟢 **LOW** (cleanup)
**Effort**: Low (1 hour)
**Impact**: Low (reduces confusion)

---

## Implementation Plan

### Week 1 (High Priority) - ✅ COMPLETED
1. ✅ Update SYSTEM_OVERVIEW.md with filter organization (DONE - 2025-11-17)
2. ✅ Update README.md (project root) (DONE - 2025-11-17)
3. ✅ Update filters/README.md (DONE - 2025-11-17)

### Week 2 (Medium Priority) - ✅ COMPLETED
4. ✅ Update docs/agents/README.md (DONE - 2025-11-17)
5. 🟡 Keep SYSTEM_OVERVIEW.md current as work progresses - **ongoing**

### Week 3 (Optional) - ✅ COMPLETED
6. ✅ Create CHANGELOG.md (DONE - 2025-11-17)
7. ✅ Create ARCHITECTURE.md (DONE - 2025-11-17)
8. 🟢 Archive outdated reports - **1 hour** (optional cleanup)

**Total Effort Completed**: ~10 hours (2025-11-17)
**Remaining**: Ongoing updates + optional cleanup

---

## Success Metrics

Documentation is successful if:
- ✅ New users can understand current project status (README.md)
- ✅ Filter developers know current versions and status (filters/README.md)
- ✅ Everyone understands harmonization principles (SYSTEM_OVERVIEW.md, ARCHITECTURE.md)
- ✅ Agents are discoverable and usable (docs/agents/README.md)
- ✅ Historical context is preserved (CHANGELOG.md)

---

## Quick Wins

**If you only have 1 hour**:
- Update README.md "Current Status" section

**If you have 3-4 hours**:
- Update README.md (2-3 hours)
- Update filters/README.md (1-2 hours)

**If you have a full day**:
- All Priority 1 + Priority 2 items (8-10 hours)

---

**End of Documentation Improvement Plan**
