# Uplifting Content Filter

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.

**Version**: 1.0-compressed (from NexusMind-Filter, battle-tested on 5,000+ articles)
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models

**Focus**: MEANING not TONE - what is happening for human/planetary wellbeing, not emotional writing style.

**Philosophy**: "Focus on what is HAPPENING for human/planetary wellbeing, not tone."

**Oracle Output**: Dimensional scores only (0-10 per dimension). Tier classification is applied post-processing, not by the oracle.

---

## PROMPT TEMPLATE

```
Analyze this article for uplifting semantic content based on what is HAPPENING in the story, not the emotional tone.

**SCOPE: Human/Planetary Wellbeing Progress ONLY**

**IN SCOPE (score normally):**
- Health improvements (medical access, treatments, prevention)
- Safety & security (from harm, not military buildup)
- Equity & justice (rights expanded, wrongs addressed)
- Livelihoods & economic dignity (fair work, shared prosperity)
- Planetary health (ecosystems restored, pollution reduced, climate solutions)
- Community flourishing (cooperation, mutual aid, cultural continuity)
- Knowledge sharing (open access, education, truth documentation)
- Peace processes (reconciliation, demilitarization, conflict resolution)

**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Corporate optimization (efficiency, productivity, market share without societal benefit)
- Technical achievement alone (faster APIs, better code, new features without wellbeing impact)
- Professional knowledge sharing (developer tutorials, business advice, technical skills) - unless addressing urgent human needs (medical knowledge, disaster response, survival skills)
- Business success (funding, profits, growth without broad benefit)
- Individual wealth (billionaire philanthropy announcements, luxury products)
- Military buildup (weapons, defenses, security theater - capped at 4 unless peace process)
- Productivity advice (life hacks, optimization tips, self-help without systemic change)
- Speculation without outcomes ("could lead to", "promises to", "aims to" - score 2-3 max)

**When in doubt:** If article doesn't show DOCUMENTED PROGRESS toward human/planetary wellbeing → OUT OF SCOPE or low score (2-3)

---

## Doom-Framing vs Solutions-Framing

**CRITICAL:** Score the MAIN CONTENT, not silver linings in doom stories.

**Decision rule:**
- If **>50% describes harm/problem/crisis**, treat as doom-framed (max score 3-4)
  - Exception: Journalism documenting harm can score 5-6 for courage/accountability
- If **>50% describes progress/solutions/recovery**, score normally (5-10)

**Examples:**
- ❌ "SNAP benefits slashed 50% (but court restored 10%)" → Doom-framed, max 3-4
  - Main content: Harm (50% cut), silver lining (10% restored) doesn't make it uplifting
- ✅ "Community builds solidarity fund after benefits cut" → Solutions-framing, score 5-8
  - Main content: Community response and mutual aid
- ❌ "Climate disaster destroys homes (but neighbors help)" → Doom-framed, max 3-4
  - Main content: Disaster, response is reactive not transformative
- ✅ "Community builds flood-resistant housing after lessons learned" → Solutions, score 5-8

---

## Outcome Requirement

**Score DOCUMENTED outcomes, not speculation or potential.**

**Documented outcomes (score 5-10):**
- "Restored 200 hectares of land"
- "Increased access to healthcare for 50,000 people"
- "Court ruling expanded voting rights"
- "Workers formed cooperative and doubled wages"

**Speculation/Potential (score 2-3 max):**
- "Could democratize access"
- "Promises to transform industry"
- "May lead to better outcomes"
- "Aims to improve wellbeing"

**Rule:** If article uses "could/might/may/promises/aims to" without showing results → score 2-3

---

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

**IMPORTANT:** Check CRITICAL FILTERS for each dimension BEFORE scoring. If article matches any filter → score 0-2.

1. **Agency**: People/communities taking effective action toward HUMAN WELLBEING or PLANETARY HEALTH?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Professional knowledge sharing (developer tutorials, coding courses, business advice, technical skills) - UNLESS urgent human needs (medical knowledge, disaster response)
   - Productivity advice (budgeting apps, life hacks, optimization tips, self-help)
   - Speculation without outcomes ("could lead to", "may enable", "promises to", "aims to" without results shown)
   - Corporate optimization (efficiency, productivity for profit, market share)
   - Business success (funding rounds, company growth, IPO, revenue without broad benefit)
   - Individual wealth (billionaire philanthropy announcements, luxury products)
   - Military buildup (weapons, defenses, security theater - not peace processes)
   - Doom-framed content (>50% describes harm, even with silver lining mentioned)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative

2. **Progress**: Movement toward human flourishing or planetary health?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Speculation without documented outcomes ("could", "might", "may", "promises", "aims to")
   - Technical achievement alone (faster APIs, better code, new features) without wellbeing impact
   - Corporate/organizational success without societal benefit
   - Market metrics (stock price, revenue, growth) without community impact
   - Military capabilities or defense spending (not peace processes)
   - Professional development (learning to code, business skills) without addressing urgent needs
   - Doom-framed with silver lining (main content is harm/crisis)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Minor | 5-6: Moderate | 7-8: Significant | 9-10: Major breakthrough

3. **Collective Benefit** (GATEKEEPER: if <5, max overall = 3 unless wonder ≥7):

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-4 MAX:**
   - Elite only (billionaires, executives, shareholders, investors)
   - Professional/technical audience only (developers, business leaders, specialists)
   - Individual optimization (personal productivity, self-help, budgeting apps)
   - Corporate employees only (productivity tools, workplace optimization)
   - Proprietary/paywalled access (not open to community)
   - Speculation about future benefit (not documented outcomes)
   - Military/security apparatus only (defense contractors, armed forces)

   **If NONE of above filters match, assess collective benefit:**
   - 0-2: Elite only | 3-4: Limited group | 5-6: Moderate community | 7-8: Broad community | 9-10: Universal benefit

4. **Connection**: Collaboration, solidarity, mutual aid across groups?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Individual action only (no collaboration mentioned)
   - Professional networking or business relationships (not solidarity)
   - Corporate partnerships for profit
   - Exclusive communities or gatekeeping
   - Technical collaboration without social purpose (coding projects, business deals)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative solidarity

5. **Innovation**: Novel solutions addressing REAL HUMAN/PLANETARY NEEDS and actually working?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Hype, vaporware, or speculation ("could revolutionize", "promises to transform")
   - Technical innovation alone (new API, coding framework, software tool) without wellbeing impact
   - Disruption for profit (not social benefit)
   - Proprietary limitations (paywalled, closed-source, corporate-controlled)
   - Weapons or military technology
   - Business model innovation (not addressing genuine needs)
   - No evidence of deployment/working (just concepts or prototypes)

   **If NONE of above filters match, score normally:**
   - 0-2: None/hype | 3-4: Minor | 5-6: Moderate | 7-8: Significant | 9-10: Breakthrough

6. **Justice in Motion**: Wrongs being addressed, accountability pursued, truth documented?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Only problem identification (no action toward justice)
   - Corporate accountability theater (no real consequences)
   - Speculation about future justice ("could lead to reform")
   - Business/technical issues (not genuine harm to people or planet)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Initial action | 5-6: Moderate progress | 7-8: Significant action | 9-10: Major justice achieved

7. **Resilience**: Recovery after setbacks, adaptation to challenges, persistence through obstacles?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Business/corporate resilience (company recovery, market bouncing back)
   - Individual productivity/optimization (personal resilience advice)
   - Technical system resilience (API uptime, code reliability)
   - Doom-framed content (focus on crisis, not recovery)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Extraordinary resilience

8. **Wonder & Enrichment**: Expands human understanding through freely shared knowledge, accessible beauty/culture?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Corporate innovation theater (product launches, tech demos without substance)
   - Celebrity gossip or luxury products
   - Paywalled content (not openly accessible)
   - Exclusive experiences (not available to community)
   - Professional/technical content only (developer tutorials, business knowledge) without broader significance
   - Speculation about future discovery (not documented findings)

   **If NONE of above filters match, score normally:**
   - Examples: open-access scientific discovery, community cultural expression, public athletic achievement
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Significant | 9-10: Profound wonder

STEP 3: Output JSON

**NOTE:** content_type is descriptive metadata (what kind of story?), NOT tier classification. Oracle classifies story type, postfilter computes tier (impact/connection/not_uplifting) from dimensional scores.

**ORACLE OUTPUTS DIMENSIONAL SCORES ONLY. Tier classification applied by postfilter.**

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
- Score the MAIN CONTENT, not silver linings in doom stories (>50% harm = doom-framed, max 3-4)
- Score DOCUMENTED outcomes, not speculation ("could/might/may" = 2-3 max)
- Corporate/technical optimization ≠ human wellbeing (score 0-2 unless broad societal benefit)
- Military preparation ≠ progress (threat response, not flourishing - capped at 4)
- Peace processes/demilitarization = genuine progress (can score 7+)
- Agency/progress toward genuine needs, not profit or military power

VALIDATION EXAMPLES:

HIGH SCORE (8.7/10) - Neutral Tone:
Article: "Farmers in six villages restored 200 hectares of degraded land using indigenous agroforestry methods. Yields increased 250% while water retention improved. Technique shared freely with neighboring communities."
Scores: Agency=9, Progress=9, Collective=10, Connection=8, Innovation=7, Justice=6, Resilience=8, Wonder=5
Reasoning: "Farmers acting to restore ecosystems and livelihoods with measurable improvement. Knowledge shared freely across communities, benefits both people and land."

LOW SCORE (1.1/10) - Corporate Finance:
Article: "Tech unicorn announces exciting Series C funding! CEO thrilled about disrupting the market with innovative AI-powered productivity tools!"
Scores: Agency=2, Progress=0, Collective=1, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Content Type: corporate_finance (capped at 2)
Reasoning: "Corporate funding without addressing real human needs. Proprietary venture-funded product. No wellbeing improvement despite positive tone."

LOW SCORE (1.5/10) - Professional Knowledge Sharing:
Article: "Senior developer shares API gateway design patterns using Rust and Go for building high-performance microservices."
Scores: Agency=2, Progress=2, Collective=2, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Reasoning: "Professional knowledge sharing (developer tutorial) without addressing urgent human needs. Technical content for specialist audience, not wellbeing progress."

LOW SCORE (1.8/10) - Productivity Advice:
Article: "5 best budgeting apps to replace Mint and optimize your personal finances. Expert recommendations for tracking expenses."
Scores: Agency=2, Progress=2, Collective=2, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Reasoning: "Personal productivity advice for individual optimization. No systemic change or community benefit. Self-help, not wellbeing."

LOW SCORE (2.3/10) - Speculation Without Outcomes:
Article: "New AI approach promises to revolutionize healthcare by enabling faster diagnosis. Could lead to breakthrough treatments."
Scores: Agency=2, Progress=2, Collective=3, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Reasoning: "Speculation about future impact ('promises to', 'could lead to') without documented outcomes. No evidence of deployment or results shown."

LOW SCORE (3.4/10) - Doom-Framed with Silver Lining:
Article: "SNAP benefits slashed 50% as federal shutdown continues, affecting 40 million Americans. Federal judge rules partial restoration for 10%."
Scores: Agency=4, Progress=3, Collective=4, Connection=0, Innovation=0, Justice=5, Resilience=0, Wonder=0
Reasoning: "Main content is harm (benefit cuts). Silver lining (10% restored) doesn't make it uplifting. Doom-framed story with minor positive element."

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

## POST-PROCESSING REFERENCE (NOT part of oracle output)

The oracle produces dimensional scores only. Post-filtering logic computes overall scores and tier classification:

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

# Tier classification (post-processing only, at inference time)
if overall_uplift_score >= 7.0:
    tier = "impact"
elif overall_uplift_score >= 4.0:
    tier = "connection"
else:
    tier = "not_uplifting"
```

**Note**: This logic is applied by the post-filter at inference time, not by the oracle during labeling. The oracle's job is to produce accurate dimensional scores; tier assignment is computed from those scores.
