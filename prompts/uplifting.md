# Uplifting Content Semantic Filter - Ground Truth Generation Prompt

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Version**: 1.0 (Migrated from NexusMind-Filter)
**Target LLM**: Claude 3.5 Sonnet / Gemini 1.5 Pro
**Use Case**: Generate ground truth labels for fine-tuning local models

---

## PROMPT TEMPLATE

```
Analyze this article for uplifting semantic content based on what is happening in the story, not the emotional tone of the writing.

ARTICLE:
Title: {title}
Text: {text}

STEP 1: Pre-classification

A) CORPORATE FINANCE FILTER
Is this primarily about: stock prices, earnings, funding rounds, valuations, M&A, IPO, or corporate financial performance?
- If YES and article does NOT mention: worker cooperative, public benefit corporation, open source, affordable access, community ownership
  → FLAG as "corporate_finance" (max_score = 2)

B) BUSINESS NEWS FILTER
Is this about: product launches, business expansion, company announcements?
- If YES → NOTE that collective_benefit must be ≥6 to score above 4

C) MILITARY/SECURITY FILTER
Is this primarily about: military buildup, defense spending, weapons systems, NATO expansion, security measures, armed forces deployment, military readiness, border defense?
- If YES and NOT about: demilitarization, peace processes, conflict resolution, post-conflict reconciliation, veterans organizing for peace
  → FLAG as "military_security" (max_score = 4)
- EXCEPTION: Peace negotiations, disarmament treaties, reconciliation programs CAN score 7+

D) SPECIAL CASE - Documentation of Harm
If article describes BOTH harm (war, disaster, crisis) AND documentation/reporting of that harm:
- Score the ACT of documentation (agency/justice/resilience of journalists/witnesses)
- Do NOT score the harm itself
- Example: "Journalists risk lives documenting war" → score journalist courage, not the war

STEP 2: Evaluate dimensions (score 0-10 for each)

1. **Agency**: Are people/communities taking effective action toward HUMAN WELLBEING (health, safety, equity, dignity, livelihoods) or PLANETARY HEALTH (climate, ecosystems, biodiversity, pollution reduction)?
   - NOT corporate profit goals, individual wealth, OR military power projection

2. **Progress**: Is there movement toward human flourishing or planetary health?
   - NOT organizational success, market metrics, OR military/security capabilities
   - Completion not required - attempts, learning, and iterative improvement count

3. **Collective Benefit**: Does this help many people or broad ecosystems - NOT just individuals, shareholders, elites, or national security apparatuses?
   - Military protection = limited benefit (max 4), genuine peace = high benefit
   - Does it address genuine needs? Is knowledge/access shared openly?
   - (GATEKEEPER: if <5, max overall = 3 unless wonder exemption applies)

4. **Connection**: Is there collaboration, solidarity, or mutual aid across groups? Cross-community cooperation? Strengthening of social bonds? NOT isolation or exclusive gatekeeping.

5. **Innovation**: Are novel solutions addressing REAL HUMAN/PLANETARY NEEDS and actually working (proven/deployed)?
   - NOT just hype, vaporware, disruption for profit, proprietary solutions limiting access, OR weapons innovation

6. **Justice in Motion**: Are wrongs being addressed, accountability pursued, or truth documented for the record - NOT just problems identified? Policy changes from advocacy? Rights expanded? Evidence preserved?

7. **Resilience**: Is there recovery after setbacks, adaptation to challenges, persistence through obstacles, or cultural continuity under pressure? Learning from failure?

8. **Wonder & Enrichment**: Does this expand human understanding through freely shared knowledge, create accessible beauty/culture for communities, or preserve traditional knowledge? NOT corporate innovation theater, celebrity gossip, luxury products, paywalled content, or exclusive experiences. Examples: open-access scientific discovery, community cultural expression, public athletic achievement, traditional knowledge documentation.

STEP 3: Provide scores
DO NOT calculate overall_uplift yourself - just provide dimension scores. The system will calculate the weighted score.

Respond with ONLY valid JSON in this exact format:
{{
  "content_type": "solutions_story|corporate_finance|business_news|military_security|peace_process|environmental|community_building",
  "agency": <score 0-10>,
  "progress": <score 0-10>,
  "collective_benefit": <score 0-10>,
  "connection": <score 0-10>,
  "innovation": <score 0-10>,
  "justice": <score 0-10>,
  "resilience": <score 0-10>,
  "wonder": <score 0-10>,
  "reasoning": "<2-3 sentence explanation focusing on what concrete actions/progress are happening>",
  "key_markers": ["<semantic_marker1>", "<semantic_marker2>"]
}}

CRITICAL REMINDERS:
- Focus on WHAT IS HAPPENING for human/planetary wellbeing
- Military preparation ≠ progress (it's threat response, not flourishing)
- Defense buildups ≠ genuine security (only response to threats, capped at 4)
- Peace processes/demilitarization = genuine progress (can score 7+)
- Agency/progress must be toward genuine needs, not profit or military power
- Corporate success ≠ human flourishing

DO NOT include any text outside the JSON object.
```

---

## SCORING WEIGHTS (for downstream processing)

### Overall Uplift Score (0-10)
Used for tier classification:

```python
weights = {
    'agency': 0.14,
    'progress': 0.19,
    'collective_benefit': 0.38,  # Highest weight - gatekeeper dimension
    'connection': 0.10,
    'innovation': 0.08,
    'justice': 0.04,
    'resilience': 0.02,
    'wonder': 0.05
}

overall_uplift_score = sum(dimensions[k] * weights[k] for k in dimensions)

# Apply content-type caps
if content_type == "corporate_finance":
    overall_uplift_score = min(overall_uplift_score, 2.0)
elif content_type == "military_security":
    overall_uplift_score = min(overall_uplift_score, 4.0)
elif content_type == "business_news" and collective_benefit < 6:
    overall_uplift_score = min(overall_uplift_score, 4.0)

# Apply gatekeeper rule
if collective_benefit < 5:
    # Wonder exemption
    if wonder >= 7 and collective_benefit >= 3:
        pass  # No cap
    else:
        overall_uplift_score = min(overall_uplift_score, 3.0)
```

### Tier Classification
```python
if overall_uplift_score >= 7.0:
    tier = "impact"
elif overall_uplift_score >= 4.0:
    tier = "connection"
else:
    tier = "not_uplifting"
```

---

## EXPECTED TIER DISTRIBUTION

Based on 5,000 labeled articles in NexusMind-Filter:

- **Impact (score >= 7.0)**: 5-15%
- **Connection (score 4.0-6.9)**: 20-40%
- **Not uplifting (score < 4.0)**: 50-70%

If your distribution significantly differs, review sample articles for calibration.

---

## PRE-FILTER RECOMMENDATION

To reduce labeling costs by ~50% with minimal false negatives:

**Only analyze articles where:**
- VADER sentiment score >= 5.0 OR
- Joy emotion score >= 0.25

This filter is implemented in `batch_labeler.py` as `uplifting_pre_filter()`.
