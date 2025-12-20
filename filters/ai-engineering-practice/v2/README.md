# AI Engineering Practice Filter v2

**Status:** BLOCKED ON DATA
**Previous Version:** v1 (FAILED - oracle hallucination)
**Blocked Since:** 2025-12-20
**Unblock Dependency:** FluxusSource hardware engineering sources (see below)

## Purpose

Surface articles providing evidence/insights on AI-augmented engineering practice for research on digital engineer competencies.

**Key Distinction:** We want articles about **engineers describing their experience using AI tools** - NOT articles that merely mention AI tools.

## What's Different from v1

### Prefilter (Preserved)
- Using v1.1 prefilter (ALLOWLIST approach)
- Requires AI tool mention + engineering context
- Expanded for ME/EE/Embedded domains

### Oracle Prompt (Rewritten)

v1 failed because the oracle:
1. **Hallucinated evidence** - invented quotes/facts not in articles
2. **Misinterpreted purpose** - scored "AI tools" instead of "using AI in engineering"
3. **Skipped top-level rules** - scope section at top was ignored during dimensional scoring

v2 fixes (based on `docs/decisions/2025-11-14-inline-filters-for-fast-models.md`):

| Fix | Technique | Reference |
|-----|-----------|-----------|
| Hallucination | Require EXACT quotes from article, explicit "If no evidence, score 0-2" | v1 FAILED.md |
| Misinterpretation | Inline filters with each dimension (not just top-level scope) | Inline Filters ADR |
| Scope skipping | CRITICAL FILTERS block before each scoring scale | Inline Filters ADR |

## Target Content Examples

**GOOD (high score):**
- "I'm a senior engineer at Spotify. Here's my daily Copilot workflow..."
- "Our team switched to Cursor 3 months ago. Here's what we learned..."
- "Study of 500 developers using GitHub Copilot found..."

**BAD (low score):**
- "New AI coding tool announced" (tool announcement, not usage experience)
- "AI pricing algorithm changes grocery costs" (AI in other domain)
- "How to build an LLM-powered app" (building with AI, not using AI for engineering)
- "Privacy firewall for LLM prompts" (tool FOR AI, not using AI in engineering)

## Critical Scope Filters (Inline with Each Dimension)

These MUST be checked before scoring each dimension:

1. **Tool Announcements** - Product launches, feature announcements without usage experience
2. **AI in Other Domains** - Medical AI, business AI, policy discussions
3. **Building WITH AI** - Tutorials on creating AI apps (not using AI for engineering work)
4. **Tool FOR AI** - Privacy tools, monitoring tools, etc. that help with AI usage but aren't about engineering WITH AI
5. **Marketing/PR** - Vendor promotions, hype without substance

## Files

| File | Status | Description |
|------|--------|-------------|
| prefilter.py | Ready | v1.1 prefilter (preserved) |
| config.yaml | Ready | Dimension weights, tier thresholds (6.0/3.5) |
| prompt-compressed.md | Ready | Oracle prompt with inline filters + anti-hallucination |
| calibration_sample.jsonl | Ready | 25 articles (software + hardware + negatives + edge cases) |
| README.md | Ready | This file |

## Calibration Results (2025-12-20)

**Sample:** 25 articles (10 software, 10 hardware/ME, 5 negative/edge)
**Scored:** 15 articles (10 blocked by prefilter - expected for negatives)
**Accuracy:** ~60% tier match

### What Worked
- **No hallucination** - Major fix from v1. Oracle uses exact quotes or "No evidence in article"
- **Negatives correctly rejected** - Urban transformation, marketing fluff scored LOW
- **Edge cases identified** - Privacy firewall correctly classified as "tool FOR AI" (not relevant)
- **Content type classification** - marketing_fluff, not_relevant labels applied correctly

### Tier Threshold Adjustment
Original thresholds (7.0/4.0) were too aggressive for practitioner accounts:
- Practitioner accounts score 3-4 on methodological_rigor (anecdotal by nature)
- This created a structural ceiling ~6.5 for even excellent practitioner content
- **Solution:** Lowered thresholds to 6.0/3.5
- Preserves score differentiation while correctly tiering valuable practitioner content

### Remaining Gap
4 HIGH-expected articles scored 5.4-5.9 (MEDIUM). These have weaker validation_coverage or workflow_detail - arguably correct classification.

## Development Plan

### Phase 1: Prompt Development - COMPLETE
1. [x] Document v1 failure (FAILED.md, counter_examples.jsonl)
2. [x] Review relevant documentation (inline filters ADR, calibration ADR)
3. [x] Write new oracle prompt with:
   - Inline filters pattern (CRITICAL FILTERS before each dimension)
   - Anti-hallucination measures (require exact quotes)
   - Clear scope distinction (experience vs mention)
   - Negative examples for each failure mode

### Phase 2: Calibration - COMPLETE
4. [x] Create stratified calibration sample (25 articles):
   - 10 software practitioner accounts
   - 10 hardware/ME practitioner accounts (FPGA, embedded, CAD, structural, civil)
   - 5 negatives + edge cases
5. [x] Run calibration:
   - Off-topic rejection: YES (negatives scored LOW)
   - No hallucinated evidence: YES (exact quotes used)
   - Tier accuracy: ~60%
6. [x] Adjust tier thresholds (7.0 -> 6.0, 4.0 -> 3.5)

### Phase 3: Training Data Generation - BLOCKED ON DATA
7. [ ] BLOCKED: Insufficient HIGH-scoring articles in current FluxusSource data
   - Current data is software-heavy, mostly news/announcements
   - Practitioner accounts with workflow detail are rare
   - Need hardware engineering sources added to FluxusSource

### Phase 4: Training - BLOCKED
8. [ ] Generate 5K+ training samples
9. [ ] Train Qwen2.5-1.5B model

## Data Dependency

**Issue filed:** `FluxusSource/docs/issues/add_hardware_engineering_sources.md`

This filter needs practitioner content that current FluxusSource feeds don't provide:
- Embedded/firmware blogs (beningo.com, embedded.com)
- FPGA practitioner content (controlpaths.com, adiuvoengineering.com)
- Mechanical/CAD blogs (mechanical-engineering.com, engineersrule.com)
- 3D printing forums (forum.prusa3d.com)
- Structural/civil engineering (structuremag.org, aecmag.com)

**Verified articles:** 22 practitioner articles confirmed across these sources (see FluxusSource issue)

**Unblock criteria:** FluxusSource adds 10+ hardware engineering feeds, collects 2+ weeks of data

## Learnings from Documentation

### Key ADR: Inline Filters for Fast Models
**Problem:** Top-level scope rules get skipped by fast models (Gemini Flash)
**Solution:** Put CRITICAL FILTERS inline with each dimension, before the scoring scale
**Result:** 87.5% → 0% false positive rate in uplifting filter

### Key ADR: Prompt Calibration Before Batch Scoring
**Problem:** Bad prompts waste batch scoring budget ($8 for 8K articles)
**Solution:** Mandatory calibration on 10-20 articles with stratified sample
**Criteria:** <10% off-topic false positives, <20% on-topic false negatives

## Anti-Hallucination Measures

1. **Require quotes:** "Cite the EXACT text from the article"
2. **Absence acknowledgment:** "If no evidence found, explicitly state 'No evidence in article'"
3. **Grounding check:** Evidence field must contain verbatim text or "Not found"
4. **Warning in prompt:** "DO NOT invent evidence. If the article doesn't discuss X, say so."
