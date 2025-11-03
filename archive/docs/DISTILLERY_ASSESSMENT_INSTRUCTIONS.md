# llm-distillery Assessment & Reconciliation Instructions

**Context**: This document provides instructions for Claude Code to assess the llm-distillery repository, reconcile it with the production prompts in content-aggregator, and clarify its role in the new architecture.

**Run from**: `C:\local_dev\llm-distillery`

**Date**: 2025-10-30

---

## 🎯 Mission Brief

You are running from the llm-distillery repository. Your task is to:

1. **Assess** what currently exists in llm-distillery (prompts, experiments, docs)
2. **Compare** with production prompts in content-aggregator
3. **Identify** what's outdated, what's experimental, what should be synced
4. **Clarify** the role of llm-distillery in the new architecture
5. **Recommend** what to keep, archive, update, or delete
6. **Sync** with NexusMind-Filter if needed (after migration)

---

## 📂 Repository Context

### Current Architecture

```
content-aggregator       → Has current production prompts (5 filters)
    ↓
NexusMind-Filter        → Will receive production prompts
    ↓
content-applications    → Apps use labeled data

llm-distillery          → ??? (YOUR CURRENT LOCATION - Need to assess role)
```

### Questions to Answer

1. **What is llm-distillery's purpose?**
   - R&D workspace for new prompts?
   - Historical archive?
   - Experimental prompt testing?
   - Obsolete and should be archived?

2. **What prompts exist here vs. content-aggregator?**
   - Are they older versions?
   - Are they experimental variants?
   - Are they completely different?

3. **Should llm-distillery stay in the workflow?**
   - As R&D workspace (develop → test → promote to NexusMind)
   - As archive only
   - Merged into NexusMind-Filter
   - Deprecated entirely

---

## 📋 Phase 1: Discovery

### Step 1.1: Inventory Current Contents

**Task**: Catalog everything in llm-distillery:

```bash
# Structure
ls -la
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d

# Prompts
ls prompts/ 2>/dev/null || echo "No prompts directory"
ls prompts/*.md 2>/dev/null | wc -l
ls prompts/*.md 2>/dev/null

# Other content
ls experiments/ 2>/dev/null || echo "No experiments directory"
ls calibration/ 2>/dev/null || echo "No calibration directory"
ls docs/ 2>/dev/null || echo "No docs directory"

# Check for data
ls data/ 2>/dev/null || echo "No data directory"
```

**Document**:
```markdown
## llm-distillery Inventory

### Directory Structure
- prompts/: [exists? contains what?]
- experiments/: [exists? contains what?]
- calibration/: [exists? contains what?]
- docs/: [exists? contains what?]
- data/: [exists? contains what?]

### Prompts Found
1. [prompt1.md] - [date modified] - [description]
2. [prompt2.md] - [date modified] - [description]
...

### Other Files
- [file1] - [purpose]
- [file2] - [purpose]
```

### Step 1.2: Check Git History

**Task**: Understand the history of llm-distillery:

```bash
# When was it last actively used?
git log --oneline -20

# What was the last significant change?
git log --oneline --since="2025-01-01"

# How many commits in last 6 months?
git log --oneline --since="6 months ago" | wc -l

# Who's been working on it?
git log --since="6 months ago" --pretty=format:"%an" | sort | uniq -c | sort -rn
```

**Document**:
```markdown
## Activity History

- Last commit: [date]
- Commits in last 6 months: [count]
- Active contributors: [names]
- Last significant change: [description]
```

### Step 1.3: Compare with content-aggregator Prompts

**Task**: Compare prompts in llm-distillery vs. content-aggregator:

```bash
# List prompts in content-aggregator
ls ../content-aggregator/prompts/*.md

# Compare specific files (if they exist in both places)
# Example for uplifting.md:
if [ -f "prompts/uplifting.md" ] && [ -f "../content-aggregator/prompts/uplifting.md" ]; then
    echo "=== Comparing uplifting.md ==="
    wc -l prompts/uplifting.md ../content-aggregator/prompts/uplifting.md
    diff -u prompts/uplifting.md ../content-aggregator/prompts/uplifting.md | head -50
fi

# Check dates
ls -l prompts/*.md 2>/dev/null
ls -l ../content-aggregator/prompts/*.md
```

**Document**:
```markdown
## Prompt Comparison

| Prompt | llm-distillery | content-aggregator | Status |
|--------|---------------|-------------------|--------|
| uplifting.md | [exists? date?] | [exists? date?] | [Same/Different/Outdated/Missing] |
| sustainability.md | [exists? date?] | [exists? date?] | [Same/Different/Outdated/Missing] |
| education.md | [exists? date?] | [exists? date?] | [Same/Different/Outdated/Missing] |
| seece.md | [exists? date?] | [exists? date?] | [Same/Different/Outdated/Missing] |
| investment-risk.md | [exists? date?] | [exists? date?] | [Same/Different/Outdated/Missing] |

### Differences Found
- [Prompt X]: llm-distillery version is [X lines], content-aggregator is [Y lines]
- [Prompt Y]: Different structure/content
- [Prompt Z]: Only exists in one location
```

### Step 1.4: Read Existing Documentation

**Task**: Check if llm-distillery has documentation explaining its purpose:

```bash
# Check README
cat README.md 2>/dev/null || echo "No README.md"

# Check for CLAUDE.md or similar
cat CLAUDE.md 2>/dev/null || echo "No CLAUDE.md"

# Check docs directory
ls docs/ 2>/dev/null
cat docs/*.md 2>/dev/null | head -100
```

**Document**:
```markdown
## Stated Purpose (from existing docs)

From README.md:
[excerpt or "No README found"]

From other docs:
[key points about purpose]
```

---

## 🔍 Phase 2: Analysis

### Step 2.1: Determine Current State

Based on discovery, classify llm-distillery:

**Choose one**:

**A. Active R&D Workspace**
- Prompts here are experimental/in-development
- Gets regular commits
- Has test/calibration infrastructure
- Purpose: Develop prompts before promoting to production

**B. Historical Archive**
- Contains older versions of prompts
- No recent development
- Prompts have been superseded by content-aggregator versions
- Purpose: Historical reference only

**C. Abandoned/Stale**
- No recent activity
- Outdated prompts
- No clear purpose
- Recommendation: Archive or delete

**D. Mixed State**
- Some active work, some old content
- Needs cleanup and clarification
- Could become active R&D if organized

### Step 2.2: Identify Version Conflicts

**Task**: For each prompt that exists in both places, determine which is authoritative:

```bash
# Compare line counts
echo "uplifting.md:"
wc -l prompts/uplifting.md ../content-aggregator/prompts/uplifting.md 2>/dev/null

# Compare modification dates
stat -c "%y %n" prompts/uplifting.md ../content-aggregator/prompts/uplifting.md 2>/dev/null

# Check content differences
diff prompts/uplifting.md ../content-aggregator/prompts/uplifting.md > /tmp/uplifting-diff.txt 2>/dev/null
wc -l /tmp/uplifting-diff.txt
```

**Document**:
```markdown
## Version Conflicts

### Prompts in Both Locations

**uplifting.md**:
- llm-distillery: [X] lines, modified [date]
- content-aggregator: [Y] lines, modified [date]
- Difference: [Major/Minor/Identical]
- Recommendation: [Use content-aggregator version / Merge changes / Keep separate]

[Repeat for each prompt]

### Prompts Only in llm-distillery
- [prompt1.md] - [What is this? Experimental? Old?]
- [prompt2.md] - [What is this? Experimental? Old?]

### Prompts Only in content-aggregator
- [prompt1.md] - [Should be copied here for future R&D?]
- [prompt2.md] - [Should be copied here for future R&D?]
```

### Step 2.3: Assess Infrastructure

**Task**: Check if llm-distillery has infrastructure for prompt development:

```bash
# Testing scripts
ls scripts/ 2>/dev/null
ls *.py 2>/dev/null

# Sample data for testing
ls data/ 2>/dev/null
ls data/samples/ 2>/dev/null

# Calibration/evaluation
ls calibration/ 2>/dev/null
ls experiments/ 2>/dev/null

# Requirements
cat requirements.txt 2>/dev/null
```

**Document**:
```markdown
## Infrastructure Assessment

### Testing Infrastructure
- Test scripts: [exists? functional?]
- Sample data: [exists? up-to-date?]
- Evaluation tools: [exists? functional?]

### Quality
- Can test prompts locally? [Yes/No]
- Can A/B test variants? [Yes/No]
- Can measure quality? [Yes/No]

### Usefulness
- Infrastructure is: [Ready to use / Needs updates / Missing / Obsolete]
```

---

## 💡 Phase 3: Recommendations

### Step 3.1: Recommend Role for llm-distillery

Based on your analysis, recommend one of these paths:

**Option A: Active R&D Workspace** (Recommended if infrastructure exists)

**Purpose**: Experimental prompt development before production

**Workflow**:
```
1. Create experimental prompt in llm-distillery/prompts/experimental/
2. Test with sample data using llm-distillery/scripts/
3. A/B test variants
4. Iterate based on results
5. When stable → Copy to content-aggregator/prompts/
6. content-aggregator promotes to NexusMind-Filter
7. NexusMind-Filter uses in production
```

**What llm-distillery needs**:
- [ ] Sync latest production prompts from content-aggregator (as baseline)
- [ ] Update test infrastructure
- [ ] Add sample data for testing
- [ ] Document prompt development workflow
- [ ] Create prompts/experimental/ directory

---

**Option B: Historical Archive** (If no active development)

**Purpose**: Reference only, not active development

**Actions**:
- [ ] Add README clarifying it's archived
- [ ] Keep as-is for historical reference
- [ ] Don't sync with production prompts
- [ ] Consider GitHub archive/read-only

---

**Option C: Merge into NexusMind-Filter** (If overlapping purpose)

**Purpose**: Consolidate prompt development in one place

**Actions**:
- [ ] Move experimental prompts to NexusMind-Filter/prompts/experimental/
- [ ] Move test infrastructure to NexusMind-Filter
- [ ] Archive or delete llm-distillery repo
- [ ] Update documentation

---

**Option D: Clean Up and Reactivate** (If mixed state)

**Purpose**: Clean up old content, reactivate as R&D workspace

**Actions**:
- [ ] Archive old/outdated prompts to archive/ directory
- [ ] Sync latest production prompts from content-aggregator
- [ ] Update infrastructure
- [ ] Document clear R&D workflow
- [ ] Create directory structure (experimental/, stable/, archive/)

### Step 3.2: Sync Plan (If Keeping Active)

If llm-distillery stays active, create sync plan:

```markdown
## Sync Plan: llm-distillery ↔ content-aggregator

### One-time Initial Sync
1. Copy all production prompts from content-aggregator to llm-distillery/prompts/production/
2. Move current llm-distillery prompts to llm-distillery/prompts/archive/
3. Create llm-distillery/prompts/experimental/ for new work

### Ongoing Workflow
1. New filter development happens in llm-distillery/prompts/experimental/
2. When stable → Copy to content-aggregator/prompts/
3. content-aggregator → Copies to NexusMind-Filter/prompts/
4. NexusMind-Filter uses in production

### Sync Frequency
- llm-distillery → content-aggregator: When filter is production-ready
- content-aggregator → llm-distillery: Never (distillery is upstream)
- NexusMind-Filter ← content-aggregator: After testing/validation
```

### Step 3.3: Directory Structure Recommendation

**Proposed structure for llm-distillery** (if keeping active):

```
llm-distillery/
├── prompts/
│   ├── production/          # Latest production prompts (read-only reference)
│   │   ├── education.md
│   │   ├── sustainability.md
│   │   ├── seece.md
│   │   ├── uplifting.md
│   │   └── investment-risk.md
│   ├── experimental/        # Work in progress
│   │   ├── new-filter-v1.md
│   │   └── education-v2-test.md
│   └── archive/            # Old/abandoned experiments
│       └── old-attempts/
│
├── scripts/
│   ├── test_prompt.py      # Test single prompt
│   ├── compare_variants.py # A/B test
│   └── evaluate_quality.py # Measure performance
│
├── data/
│   └── samples/            # Sample articles for testing
│       ├── education_samples.json
│       └── investment_samples.json
│
├── experiments/
│   └── 2025-10-30_education-v2/
│       ├── results.md
│       └── comparison.csv
│
├── docs/
│   ├── prompt-development-workflow.md
│   └── quality-criteria.md
│
└── README.md              # Clear explanation of purpose
```

---

## ✅ Phase 4: Present Findings

### Step 4.1: Create Assessment Report

Generate report for user:

```markdown
# llm-distillery Assessment Report

**Date**: 2025-10-30
**Current Status**: [Active/Stale/Mixed]

## Current State

### Inventory
- Prompts: [count] files
- Last activity: [date]
- Infrastructure: [Exists/Missing/Outdated]

### Comparison with content-aggregator
| Prompt | Status | Recommendation |
|--------|--------|----------------|
| [prompt] | [Same/Different/Outdated] | [Action] |

## Identified Issues
1. [Issue 1]
2. [Issue 2]
...

## Recommended Role

**Recommendation**: [Option A/B/C/D from above]

**Rationale**: [Why this option]

**Actions Required**:
1. [Action 1]
2. [Action 2]
...

## Integration with Migration

How llm-distillery fits into new architecture:

```
content-aggregator (Collection) ← [No connection]
         ↓
NexusMind-Filter (Production Filtering) ← Receives stable prompts from llm-distillery
         ↓
content-applications (Apps)

llm-distillery (R&D) → Develops prompts → Promotes to NexusMind-Filter when stable
```

## Next Steps

1. [Step 1]
2. [Step 2]
...

## Questions for User

1. Should llm-distillery remain active for R&D?
2. If yes, do you want to sync production prompts here?
3. If no, should it be archived or deleted?
4. Are there experimental prompts here worth keeping?
```

---

## 🚀 Phase 5: Execution (After User Approval)

### Step 5.1: Create Backup

```bash
# Create git tag
git tag pre-distillery-cleanup-$(date +%Y%m%d)
git tag -l | tail -5
echo "✅ Backup tag created"
```

### Step 5.2: Execute Chosen Option

#### If Option A (Active R&D):

```bash
# Create directory structure
mkdir -p prompts/production
mkdir -p prompts/experimental
mkdir -p prompts/archive
mkdir -p data/samples
mkdir -p experiments

# Move current prompts to archive (if outdated)
mv prompts/*.md prompts/archive/ 2>/dev/null || true

# Sync production prompts from content-aggregator
cp ../content-aggregator/prompts/*.md prompts/production/

# Copy compressed versions too
mkdir -p prompts/production/compressed
cp ../content-aggregator/prompts/compressed/*.md prompts/production/compressed/

echo "✅ Directory structure created and prompts synced"
```

#### If Option B (Archive):

```bash
# Create clear README
cat > README.md << 'EOF'
# llm-distillery (ARCHIVED)

**Status**: This repository is archived and no longer actively maintained.

**Purpose**: Historical reference for early prompt development work.

**For active development**: See NexusMind-Filter repository.

**Last Active**: [Date]
EOF

echo "✅ Archive README created"
```

#### If Option C (Merge into NexusMind):

```bash
# Copy experimental content to NexusMind-Filter
cp -r prompts/experimental ../NexusMind-Filter/prompts/experimental/ 2>/dev/null || true
cp -r scripts/* ../NexusMind-Filter/scripts/ 2>/dev/null || true

# Create deprecation notice
cat > README.md << 'EOF'
# llm-distillery (DEPRECATED)

**Status**: This repository has been merged into NexusMind-Filter.

**See**: C:\local_dev\NexusMind-Filter for prompt development.

**Date Deprecated**: 2025-10-30
EOF

echo "✅ Content merged to NexusMind-Filter, deprecation notice added"
```

#### If Option D (Clean and Reactivate):

```bash
# Create structure
mkdir -p prompts/production
mkdir -p prompts/experimental
mkdir -p prompts/archive

# Archive old content
mv prompts/*.md prompts/archive/ 2>/dev/null || true

# Sync production prompts
cp ../content-aggregator/prompts/*.md prompts/production/

# Update infrastructure
# [Update test scripts, add sample data, etc.]

echo "✅ Cleaned up and reactivated"
```

### Step 5.3: Create Documentation

**Create or update README.md** based on chosen option.

**If staying active**, create workflow documentation:

```bash
cat > docs/prompt-development-workflow.md << 'EOF'
# Prompt Development Workflow

## Purpose
llm-distillery is the R&D workspace for developing new semantic filters before production deployment.

## Workflow

### 1. Create Experimental Prompt
```
prompts/experimental/new-filter-name.md
```

### 2. Test Locally
```bash
python scripts/test_prompt.py --prompt experimental/new-filter-name.md --data data/samples/
```

### 3. Iterate
- Refine dimensions
- Add examples
- Adjust rubrics

### 4. A/B Test
```bash
python scripts/compare_variants.py --variant-a v1.md --variant-b v2.md
```

### 5. Promote to Production
When stable and tested:
1. Copy to content-aggregator/prompts/
2. content-aggregator team validates
3. Promotes to NexusMind-Filter/prompts/
4. NexusMind-Filter uses in production

## Quality Criteria
- [ ] All dimensions defined (0-10 scales)
- [ ] Pre-filters included
- [ ] 2+ validation examples
- [ ] Tested on 20+ sample articles
- [ ] Scores align with expectations

## Directory Structure
- prompts/production/ - Reference copies of production prompts
- prompts/experimental/ - Active development
- prompts/archive/ - Old experiments
- data/samples/ - Test data
- experiments/ - A/B test results
EOF
```

### Step 5.4: Commit Changes

```bash
git add .
git commit -m "Reorganize llm-distillery based on assessment: [chosen option]"
echo "✅ Changes committed"
```

---

## 🧪 Phase 6: Verification

### Step 6.1: Verify Structure

```bash
# Check directory structure
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d

# Verify production prompts (if synced)
ls prompts/production/*.md 2>/dev/null | wc -l
# Should be 5 (if synced)

# Check documentation
cat README.md | head -20
```

### Step 6.2: Test Infrastructure (If Active)

```bash
# Test scripts work
python scripts/test_prompt.py --help 2>/dev/null || echo "Test scripts need updating"

# Check sample data exists
ls data/samples/ 2>/dev/null || echo "Need sample data"
```

---

## 📊 Phase 7: Final Report

```markdown
# llm-distillery Reconciliation Report

**Date**: 2025-10-30
**Decision**: [Chosen option]

## What Was Done

### Actions Taken
1. [Action 1]
2. [Action 2]
...

### File Changes
- Moved: [count] files
- Added: [count] files
- Deleted: [count] files

### Structure
[New directory structure]

## Role in New Architecture

llm-distillery's role: [Clear statement]

Integration:
- Upstream: [What feeds into it]
- Downstream: [What it feeds into]

## Next Steps

1. [Recommendation 1]
2. [Recommendation 2]
...

## Rollback

If needed:
```bash
git checkout pre-distillery-cleanup-[date]
```
```

---

## 🎯 Success Criteria

Assessment is successful if:

- ✅ **Clear role** defined for llm-distillery
- ✅ **No version conflicts** with content-aggregator/NexusMind-Filter
- ✅ **Documented** purpose and workflow (if active)
- ✅ **Clean structure** (no ambiguous or duplicate files)
- ✅ **Integration clear** with new architecture

---

## 🤖 Execution Instructions for Claude Code

When running this assessment:

1. **Start with thorough discovery** - inventory everything
2. **Compare carefully** with content-aggregator
3. **Present findings** before making changes
4. **Get user approval** on which option to pursue
5. **Execute chosen option** with proper backups
6. **Document clearly** what llm-distillery's role is

**Use TodoWrite tool** to track progress.

**Be thorough** - understanding what's here determines if this repo stays active or gets archived.

---

## 📞 Questions for User (Ask Before Deciding)

1. **Intent**: Was llm-distillery meant to be the R&D workspace, or was that content-aggregator?

2. **Current need**: Do you want a separate R&D workspace for prompt development?

3. **Workflow preference**:
   - Develop prompts in llm-distillery, promote to production?
   - Develop prompts directly in content-aggregator?
   - Develop prompts in NexusMind-Filter itself?

4. **Content value**: Are there experimental prompts here worth preserving?

5. **Future use**: Will you actively use llm-distillery going forward?

---

**END OF INSTRUCTIONS**

---

**Last Updated**: 2025-10-30
**Status**: Ready for Claude Code execution
**Estimated Duration**: 1-2 hours
