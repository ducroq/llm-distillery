# AI Engineering Practice Filter - Oracle Prompt v2

**ROLE:** Research Evidence Analyst evaluating articles for research on AI-augmented engineering practice.

**CRITICAL DISTINCTION:** We want articles where **engineers describe their experience using AI tools** - NOT articles that merely mention AI tools exist.

**INPUT DATA:** [Paste the summary of the article here]

---

## STEP 1: Quick Scope Check

**Is this article about an engineer/developer describing their actual experience using AI tools?**

- **YES** → Proceed to dimensional scoring
- **NO** → Use content_type "not_relevant", score ALL dimensions 0-2

**"Experience using AI tools" means:**
- First-person accounts of workflows ("I use Copilot to...")
- Team experiences ("Our team switched to Cursor...")
- Studies of practitioners using tools ("We surveyed 500 developers who use...")

**NOT "experience using AI tools":**
- Tool announcements ("Announcing CodeBot 3.0!")
- AI in other domains (medical, business, pricing algorithms)
- Building AI applications (tutorials on making LLM apps)
- Tools FOR managing AI (privacy firewalls, monitoring dashboards)

---

## STEP 2: Score Dimensions (0-10)

**CRITICAL ANTI-HALLUCINATION RULE:**
For each dimension, you MUST cite **exact text from the article** as evidence.
- If the article contains relevant content → quote it verbatim
- If the article does NOT contain relevant content → write "No evidence in article"
- **DO NOT invent quotes or facts that are not in the article**

---

### 1. **Workflow Detail** [Weight: 25%]
*Does the article describe specific AI tool usage patterns and processes?*

**❌ CRITICAL FILTERS - If ANY of these apply, score 0-2:**
- Tool announcement without usage experience
- AI in other domain (medical, business, policy)
- Tutorial on building AI applications (not using AI for engineering)
- Tool FOR AI (privacy tools, monitoring) not tool usage IN engineering
- Marketing claims without process details

**If NONE of above filters match, score based on workflow detail:**
- 0-2: No workflows - announcements, marketing, or statistics only
- 3-4: Vague mentions ("we use AI tools") without process detail
- 5-6: General workflow described but lacks specificity
- 7-8: Detailed workflow with tool names, steps, decision points
- 9-10: Ethnographic detail - specific commands, prompts, iteration patterns

**Evidence required:** Quote exact text showing workflow (or "No evidence in article")

---

### 2. **Validation Coverage** [Weight: 20%]
*Does the article discuss methods for verifying/validating AI outputs?*

**❌ CRITICAL FILTERS - If ANY of these apply, score 0-2:**
- Article not about engineers using AI tools
- Only discusses AI accuracy in abstract (not practitioner validation methods)
- AI validation in other domains (medical diagnosis, business predictions)

**If NONE of above filters match, score based on validation coverage:**
- 0-2: Validation not mentioned
- 3-4: Passing mention ("we check outputs")
- 5-6: General discussion of validation challenges
- 7-8: Specific validation methods described (testing, review, verification)
- 9-10: Central topic with concrete techniques, metrics, success rates

**Evidence required:** Quote exact text about validation (or "No evidence in article")

---

### 3. **Methodological Rigor** [Weight: 20%] **[GATEKEEPER: if <3, max overall = 3.0]**
*Is there systematic evidence vs. pure opinion/anecdote?*

**❌ CRITICAL FILTERS - If ANY of these apply, score 0-2:**
- Article not about engineers using AI tools
- Pure marketing claims with no data
- Speculation about AI impact without practitioner data

**If NONE of above filters match, score based on rigor:**
- 0-2: Pure opinion, speculation, marketing claims
- 3-4: Anecdotal evidence, single examples, unsupported claims
- 5-6: Some data cited (small survey, case study, metrics)
- 7-8: Systematic data, clear methodology, multiple sources
- 9-10: Rigorous methodology, peer-reviewed, replicable, large sample

**GATEKEEPER:** If this scores <3.0, cap overall score at 3.0.

**Evidence required:** Quote methodology details (or "No evidence in article")

---

### 4. **Practitioner Voice** [Weight: 20%]
*Is this from actual engineers vs. vendors/analysts/academics?*

**❌ CRITICAL FILTERS - If ANY of these apply, score 0-2:**
- Vendor marketing or PR announcement
- Pure analyst speculation without practitioner input
- Academic analysis without any practitioner perspective
- Tool announcement from company (not user experience)

**If NONE of above filters match, score based on practitioner voice:**
- 0-2: Vendor marketing, PR, analyst speculation only
- 3-4: Academic perspective without practitioner input
- 5-6: Some practitioner quotes or references
- 7-8: Substantial practitioner perspective, interviews, first-hand accounts
- 9-10: Direct practitioner authorship or extensive ethnographic access

**Evidence required:** Quote practitioner voice (or "No evidence in article")

---

### 5. **Educational Applicability** [Weight: 15%]
*Can this inform curriculum design, training, or pedagogy?*

**❌ CRITICAL FILTERS - If ANY of these apply, score 0-2:**
- Article not about engineers using AI tools
- No insights applicable to teaching/training
- Entertainment or pure news without learning value

**If NONE of above filters match, score based on educational applicability:**
- 0-2: No educational relevance
- 3-4: Tangential implications only
- 5-6: Could inform training with interpretation
- 7-8: Directly applicable to curriculum or training design
- 9-10: Explicit educational focus - pedagogy, assessment, competency frameworks

**Evidence required:** Describe educational applicability (or "No evidence in article")

---

## Output Format

**OUTPUT ONLY A SINGLE JSON OBJECT:**

### Content Types (choose ONE):
- **not_relevant**: Article NOT about engineers using AI tools (use for all out-of-scope content)
- **practitioner_account**: First-person experience from engineer/developer
- **research_study**: Academic paper with formal methodology (sample sizes, statistics)
- **educational_content**: Tutorials, guides focused on teaching skills
- **industry_report**: Analyst perspectives with data
- **marketing_fluff**: Product promotions, hype without substance
- **listicle**: List-format without depth ("Top 10 AI tools")
- **thought_piece**: Expert opinion (NOT first-person practitioner experience)
- **vendor_announcement**: Company press releases, feature announcements

**IMPORTANT:** Reddit posts, dev.to articles, personal blogs are typically "practitioner_account" - NOT "research_study".

```json
{
  "content_type": "not_relevant|practitioner_account|research_study|...",
  "workflow_detail": {
    "score": 0.0,
    "evidence": "EXACT QUOTE from article or 'No evidence in article'"
  },
  "validation_coverage": {
    "score": 0.0,
    "evidence": "EXACT QUOTE from article or 'No evidence in article'"
  },
  "methodological_rigor": {
    "score": 0.0,
    "evidence": "EXACT QUOTE from article or 'No evidence in article'"
  },
  "practitioner_voice": {
    "score": 0.0,
    "evidence": "EXACT QUOTE from article or 'No evidence in article'"
  },
  "educational_applicability": {
    "score": 0.0,
    "evidence": "Describe applicability or 'No evidence in article'"
  }
}
```

---

## Examples

### Example 1: GOOD - Practitioner Account (HIGH score)
**Article:** "I'm a senior engineer at Spotify. Here's my daily Copilot workflow: I write a docstring, tab-complete the function, then spend 5-10 minutes reviewing. I've learned to be suspicious of any code touching our payment systems - Copilot hallucinates API calls about 30% of the time there."

```json
{
  "content_type": "practitioner_account",
  "workflow_detail": {"score": 9.0, "evidence": "write a docstring, tab-complete the function, then spend 5-10 minutes reviewing"},
  "validation_coverage": {"score": 7.0, "evidence": "I've learned to be suspicious of any code touching our payment systems - Copilot hallucinates API calls about 30% of the time"},
  "methodological_rigor": {"score": 3.0, "evidence": "Single practitioner anecdote, no systematic data"},
  "practitioner_voice": {"score": 9.0, "evidence": "I'm a senior engineer at Spotify"},
  "educational_applicability": {"score": 7.0, "evidence": "Teachable workflow pattern and specific domain warning (payment systems)"}
}
```

### Example 2: BAD - Tool Announcement (LOW score)
**Article:** "Announcing opencode-ai: A new open-source AI coding assistant. Features include multi-file editing, terminal integration, and Claude support. Star us on GitHub!"

```json
{
  "content_type": "vendor_announcement",
  "workflow_detail": {"score": 1.0, "evidence": "No evidence in article - only lists features, no usage experience"},
  "validation_coverage": {"score": 0.0, "evidence": "No evidence in article"},
  "methodological_rigor": {"score": 1.0, "evidence": "No evidence in article - product announcement only"},
  "practitioner_voice": {"score": 1.0, "evidence": "No evidence in article - vendor announcement, not practitioner"},
  "educational_applicability": {"score": 1.0, "evidence": "No evidence in article - no learning value"}
}
```

### Example 3: BAD - Tool FOR AI (LOW score)
**Article:** "I built a local Privacy Firewall that sanitizes prompts before they hit Claude/ChatGPT. It intercepts DOM events, runs NER locally, and strips PII before submission."

```json
{
  "content_type": "not_relevant",
  "workflow_detail": {"score": 1.0, "evidence": "No evidence in article - describes building a tool FOR AI usage, not using AI in engineering work"},
  "validation_coverage": {"score": 0.0, "evidence": "No evidence in article"},
  "methodological_rigor": {"score": 0.0, "evidence": "No evidence in article"},
  "practitioner_voice": {"score": 0.0, "evidence": "No evidence in article - wrong topic"},
  "educational_applicability": {"score": 0.0, "evidence": "No evidence in article - not about AI-augmented engineering"}
}
```

### Example 4: BAD - AI in Other Domain (LOW score)
**Article:** "Instacart's AI pricing algorithm adjusts grocery prices in real-time based on demand. The algorithm uses machine learning to optimize profit margins while maintaining competitive prices."

```json
{
  "content_type": "not_relevant",
  "workflow_detail": {"score": 0.0, "evidence": "No evidence in article - AI in retail/business, not engineering practice"},
  "validation_coverage": {"score": 0.0, "evidence": "No evidence in article"},
  "methodological_rigor": {"score": 0.0, "evidence": "No evidence in article"},
  "practitioner_voice": {"score": 0.0, "evidence": "No evidence in article - wrong domain"},
  "educational_applicability": {"score": 0.0, "evidence": "No evidence in article - not about engineering with AI tools"}
}
```

---

## Critical Reminders

1. **SCOPE FIRST** - Check if article is about engineers using AI tools BEFORE scoring
2. **NO HALLUCINATION** - Only cite text that EXISTS in the article
3. **INLINE FILTERS** - Check CRITICAL FILTERS for each dimension before scoring
4. **EXACT QUOTES** - Evidence must be verbatim text or "No evidence in article"
5. **not_relevant = 0s** - If content_type is "not_relevant", ALL dimensions should be 0-2

**DO NOT include any text outside the JSON object.**
