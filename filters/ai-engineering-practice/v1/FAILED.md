# AI Engineering Practice Filter v1 - FAILED

**Status:** FAILED
**Date:** 2025-12-20
**Reason:** Oracle hallucination and misinterpretation

## What Worked

- **Prefilter v1.1** - Successfully expanded to cover ME/EE/Embedded domains
- **ALLOWLIST approach** - Correctly required AI + engineering context
- **Pass rate** - Reasonable 3.2% on FluxusSource, 1.7% on master dataset

## What Failed

### 1. Oracle Hallucination
The oracle (Gemini Flash) invented evidence that did not exist in articles:
- Article about Instacart grocery pricing → Oracle claimed "uses GitHub Copilot to generate code"
- Article about stock market circuit breakers → Oracle claimed "uses ChatGPT for coding workflows"

The oracle fabricated detailed evidence about AI tool usage that was completely absent from the actual content.

### 2. Misinterpretation of Filter Purpose
The oracle scored articles about:
- **Tools FOR AI** (e.g., privacy firewall for LLM usage) - instead of engineers USING AI
- **AI announcements** (e.g., opencode-ai tool) - instead of practitioner experiences
- **AI in other domains** (medical, policy) - instead of engineering practice

### 3. Poor Tier Discrimination
- 99% of articles fell in "medium" tier
- Only 8 articles reached "high" tier
- Of those 8: **0 were actually relevant** to the filter's purpose

## Validation Results (8 HIGH tier articles)

| # | Title | Issue |
|---|-------|-------|
| 1 | Privacy Firewall | Wrong: tool FOR AI, not using AI |
| 2 | Instacart pricing | Hallucination: no engineering content |
| 3-5 | Circuit Breakers (duplicates) | Hallucination: stock market article |
| 6 | sst/opencode | Wrong: tool announcement |
| 7 | Urban Digital Transformation | Not relevant: policy study |
| 8 | AI detects cancer | Not relevant: medical AI |

## Root Cause Analysis

1. **Prompt ambiguity** - The oracle prompt didn't clearly distinguish between:
   - "Articles about AI tools" (NOT wanted)
   - "Articles about engineers using AI tools in their work" (WANTED)

2. **Lack of grounding** - The oracle was not forced to cite specific text from the article

3. **Hallucination tendency** - Gemini Flash appears prone to generating plausible-sounding but fabricated evidence

## Recommendations for v2

1. **Clarify prompt** - Explicitly state we want "engineers describing their experience using AI tools" not "articles mentioning AI tools"
2. **Require quotes** - Force oracle to cite exact text from article as evidence
3. **Add negative examples** - Show examples of what NOT to score highly
4. **Consider different oracle** - Test Claude Sonnet or Gemini Pro for less hallucination

## Assets to Preserve

- `prefilter.py` (v1.1) - Keep for v2
- `counter_examples.jsonl` - Use for prompt improvement
- Calibration learnings

## Files

- `FAILED.md` - This document
- `prefilter.py` - Prefilter v1.1 (reuse in v2)
- `prompt-compressed.md` - Failed oracle prompt
- `counter_examples.jsonl` - Misclassified articles for training
