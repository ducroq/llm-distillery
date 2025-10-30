# Uplifting Content Filter

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Version**: 1.0-compressed (from NexusMind-Filter, battle-tested on 5,000+ articles)
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models

**Focus**: MEANING not TONE - what is happening for human/planetary wellbeing, not emotional writing style.

---

## PROMPT TEMPLATE

```
Analyze this article for uplifting semantic content based on what is HAPPENING in the story, not the emotional tone.

ARTICLE:
Title: {title}
Text: {text}

STEP 1: Pre-classification

A) CORPORATE FINANCE: Stock prices, earnings, funding rounds, valuations, M&A, IPO, financial performance?
   - If YES and NOT worker cooperative/public benefit/open source/affordable access/community ownership → FLAG "corporate_finance" (max_score = 2)

B) BUSINESS NEWS: Product launches, business expansion, company announcements?
   - If YES → NOTE: collective_benefit must be ≥6 to score above 4

C) MILITARY/SECURITY: Military buildup, defense spending, weapons, NATO expansion, security measures, armed forces deployment, border defense?
   - If YES and NOT demilitarization/peace processes/conflict resolution/reconciliation → FLAG "military_security" (max_score = 4)
   - Exception: Peace negotiations, disarmament treaties, reconciliation CAN score 7+

D) DOCUMENTATION OF HARM: Article describes BOTH harm AND documentation?
   - Score the ACT of documentation (journalist courage), NOT the harm itself

STEP 2: Score Dimensions (0-10)

1. **Agency**: People/communities taking effective action toward HUMAN WELLBEING (health, safety, equity, dignity, livelihoods) or PLANETARY HEALTH (climate, ecosystems, biodiversity, pollution reduction)?
   - NOT corporate profit, individual wealth, OR military power projection
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative

2. **Progress**: Movement toward human flourishing or planetary health?
   - NOT organizational success, market metrics, OR military capabilities
   - Attempts/learning count - completion not required
   - 0-2: None | 3-4: Minor | 5-6: Moderate | 7-8: Significant | 9-10: Major breakthrough

3. **Collective Benefit** (GATEKEEPER: if <5, max overall = 3 unless wonder ≥7):
   - Helps many people or broad ecosystems - NOT just individuals, shareholders, elites, security apparatuses
   - Military protection = limited benefit (max 4), genuine peace = high benefit
   - Does it address genuine needs? Knowledge/access shared openly?
   - 0-2: Elite only | 3-4: Limited group | 5-6: Moderate community | 7-8: Broad community | 9-10: Universal benefit

4. **Connection**: Collaboration, solidarity, mutual aid across groups? Cross-community cooperation?
   - NOT isolation or exclusive gatekeeping
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative solidarity

5. **Innovation**: Novel solutions addressing REAL HUMAN/PLANETARY NEEDS and actually working (proven/deployed)?
   - NOT hype, vaporware, disruption for profit, proprietary limitations, OR weapons innovation
   - 0-2: None/hype | 3-4: Minor | 5-6: Moderate | 7-8: Significant | 9-10: Breakthrough

6. **Justice in Motion**: Wrongs being addressed, accountability pursued, truth documented - NOT just problems identified?
   - Policy changes from advocacy? Rights expanded? Evidence preserved?
   - 0-2: None | 3-4: Initial action | 5-6: Moderate progress | 7-8: Significant action | 9-10: Major justice achieved

7. **Resilience**: Recovery after setbacks, adaptation to challenges, persistence through obstacles, cultural continuity under pressure?
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Extraordinary resilience

8. **Wonder & Enrichment**: Expands human understanding through freely shared knowledge, accessible beauty/culture for communities, traditional knowledge preservation?
   - NOT corporate innovation theater, celebrity gossip, luxury products, paywalled content, exclusive experiences
   - Examples: open-access scientific discovery, community cultural expression, public athletic achievement
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Significant | 9-10: Profound wonder

STEP 3: Output JSON

{{
  "content_type": "solutions_story|corporate_finance|business_news|military_security|peace_process|environmental|community_building",
  "agency": <0-10>,
  "progress": <0-10>,
  "collective_benefit": <0-10>,
  "connection": <0-10>,
  "innovation": <0-10>,
  "justice": <0-10>,
  "resilience": <0-10>,
  "wonder": <0-10>,
  "reasoning": "<2-3 sentences on what actions/progress are happening for human/planetary wellbeing>",
  "key_markers": ["<marker1>", "<marker2>"]
}}

CRITICAL REMINDERS:
- Focus on WHAT IS HAPPENING for human/planetary wellbeing
- Military preparation ≠ progress (threat response, not flourishing)
- Defense buildups ≠ genuine security (capped at 4)
- Peace processes/demilitarization = genuine progress (can score 7+)
- Agency/progress toward genuine needs, not profit or military power
- Corporate success ≠ human flourishing

VALIDATION EXAMPLES:

HIGH SCORE (8.7/10) - Neutral Tone:
Article: "Farmers in six villages restored 200 hectares of degraded land using indigenous agroforestry methods. Yields increased 250% while water retention improved. Technique shared freely with neighboring communities."
Scores: Agency=9, Progress=9, Collective=10, Connection=8, Innovation=7, Justice=6, Resilience=8, Wonder=5
Reasoning: "Farmers acting to restore ecosystems and livelihoods with measurable improvement. Knowledge shared freely across communities, benefits both people and land."

LOW SCORE (1.1/10) - Enthusiastic Tone:
Article: "Tech unicorn announces exciting Series C funding! CEO thrilled about disrupting the market with innovative AI-powered productivity tools!"
Scores: Agency=2, Progress=0, Collective=1, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Content Type: corporate_finance (capped at 2)
Reasoning: "Corporate funding without addressing real human needs. Proprietary venture-funded product. No wellbeing improvement despite positive tone."

MILITARY DEFENSE (Capped at 4.0):
Article: "Finland deepens defense strategy after joining NATO, with increased military spending and border fortifications."
Calculated: 4.8 → Capped: 4.0
Reasoning: "Military buildups represent preparation for violence, not progress toward flourishing. Harm prevention, not peace creation."

PEACE PROCESS (Not Capped - 9.1/10):
Article: "After 15 years of armed conflict, former combatants meet for historic reconciliation summit. Truth commission begins documenting war crimes."
Scores: Agency=9, Progress=9, Collective=9, Connection=10, Justice=9, Resilience=8
Content Type: peace_process (exception to military filter)
Reasoning: "Genuine progress toward flourishing through reconciliation and truth-seeking, not military preparation."

DO NOT include any text outside the JSON object.
```

---

## SCORING FORMULA (Applied post-labeling)

```python
weights = {
    'agency': 0.14,
    'progress': 0.19,
    'collective_benefit': 0.38,  # Highest - gatekeeper
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

**Tier Classification**:
- overall_uplift_score >= 7.0: "impact"
- overall_uplift_score >= 4.0: "connection"
- overall_uplift_score < 4.0: "not_uplifting"
