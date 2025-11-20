# AI-Augmented Practice Scoring

**Purpose**: Rate empirical reports of how GenAI/LLMs change cognitive work practices.

**Version**: 1.0

**Focus**: REAL workflow integration with EVIDENCE, not hype or speculation.

---

## PROMPT TEMPLATE

```
Rate this article on 8 dimensions (0-10 scale). Focus: Empirical evidence of AI-augmented cognitive work transformation.

ARTICLE:
Title: {title}
Text: {text}

---

## Dimensions

### 1. WORKFLOW_INTEGRATION_DEPTH (20%)
How deeply is AI integrated into actual work processes?

- **9-10**: End-to-end transformation, institutional adoption, redesigned processes
- **7-8**: Core workflow redesigned around AI, team-wide adoption
- **5-6**: Regular use for specific tasks, manual handoffs
- **3-4**: Occasional experimental use, not core workflow
- **1-2**: Toy example, demo, no real usage

**Evidence**: "redesigned SOP", "30% of PRs AI-assisted", "entire team adopted", "metrics tracked"

### 2. EMPIRICAL_EVIDENCE_QUALITY (18%)
Strength of evidence for claims about AI impact

- **9-10**: Longitudinal study, published research, rigorous methodology
- **7-8**: Controlled comparison, A/B test, quantitative metrics
- **5-6**: Multiple practitioners, informal survey, some data
- **3-4**: Personal anecdote, single use case
- **1-2**: Pure speculation, no data

**Requirements**: Before/after data, specific metrics (time saved, error rate), sample size

### 3. TRUST_VERIFICATION_PATTERNS (15%)
How do practitioners validate AI output?

- **9-10**: Rigorous verification framework, documented edge cases, formal protocol
- **7-8**: Formal validation protocol, automated tests, systematic checking
- **5-6**: Systematic spot-checking, domain expert review
- **3-4**: Ad-hoc checking, informal validation
- **1-2**: Blind acceptance, no verification mentioned

**High trust markers**: "expert reviews all outputs", "automated test suite", "documented failures"

### 4. COGNITIVE_TASK_SPECIFICITY (12%)
How specific is the cognitive task described?

- **9-10**: Detailed task decomposition, measurable success criteria
- **7-8**: Precise workflow step with clear inputs/outputs
- **5-6**: Specific task (code review, literature synthesis, meeting notes)
- **3-4**: Broad category (writing, coding, research)
- **1-2**: Generic "AI assistant", no specific task

**Examples**: "Code review for type safety bugs" → 7, "Extract action items from transcripts" → 6

### 5. FAILURE_MODE_DOCUMENTATION (12%)
Are failure cases, limitations, edge cases documented?

- **9-10**: Comprehensive failure taxonomy, mitigation strategies
- **7-8**: Systematic failure pattern analysis
- **5-6**: Some specific failure examples
- **3-4**: Vague "doesn't always work"
- **1-2**: No failures mentioned, uncritical cheerleading

**Critical**: Silent failures, hallucinations caught, edge cases, when NOT to use AI

### 6. HUMAN_AI_DIVISION_OF_LABOR (10%)
Clear articulation of what humans vs AI do

- **9-10**: Sophisticated workflow with clear handoffs, feedback loops
- **7-8**: Explicit allocation (AI generates, human reviews/edits)
- **5-6**: Some task division described
- **3-4**: Vague "AI helps"
- **1-2**: No clarity, AI does "everything"

**Markers**: "AI drafts, human refines", "human frames, AI executes, human validates"

### 7. SKILL_EVOLUTION (8%)
What new skills emerged? What skills changed?

- **9-10**: Training programs, skill gap analysis, hiring changes
- **7-8**: Detailed skill transformation analysis
- **5-6**: Specific new skills identified
- **3-4**: Vague "need to learn prompting"
- **1-2**: No discussion of skill changes

**Examples**: "Prompt engineering critical" → 7, "Evaluation became bottleneck" → 7

### 8. ORGANIZATIONAL_DYNAMICS (5%)
How do teams/orgs adapt to AI-augmented workers?

- **9-10**: Institutional transformation, new roles/processes
- **7-8**: Org-wide policies, workflow standards
- **5-6**: Team coordination patterns emerging
- **3-4**: Team awareness, informal sharing
- **1-2**: Individual use only, no org context

**Indicators**: "Created AI reviewer role", "team prompt library", "changed code review process"

---

## Gatekeeper Rules

1. **If EMPIRICAL_EVIDENCE_QUALITY < 4.0**: Cap overall score at 3.9
   - Reasoning: Must have real evidence, not just speculation

---

## Tier Classification

- **Score ≥ 8.0**: Transformative Practice (deep integration, rigorous evidence)
- **Score ≥ 6.0**: Validated Adoption (real usage, empirical validation)
- **Score ≥ 4.0**: Emerging Practice (early adoption, some evidence)
- **Score < 4.0**: Speculation (hype, no real usage)

---

## Output Format (JSON)

{{
  "workflow_integration_depth": {{"score": <0-10>, "reasoning": "Brief justification"}},
  "empirical_evidence_quality": {{"score": <0-10>, "reasoning": "..."}},
  "trust_verification_patterns": {{"score": <0-10>, "reasoning": "..."}},
  "cognitive_task_specificity": {{"score": <0-10>, "reasoning": "..."}},
  "failure_mode_documentation": {{"score": <0-10>, "reasoning": "..."}},
  "human_ai_division_of_labor": {{"score": <0-10>, "reasoning": "..."}},
  "skill_evolution": {{"score": <0-10>, "reasoning": "..."}},
  "organizational_dynamics": {{"score": <0-10>, "reasoning": "..."}},
  "overall_assessment": "<1-2 sentence summary>",
  "primary_task": "coding|writing|research|analysis|creative|other",
  "ai_tool": "chatgpt|claude|copilot|cursor|custom|multiple|other",
  "confidence": "HIGH|MEDIUM|LOW"
}}

**Be strict**: Most AI articles are hype or speculation. Score low unless there's clear empirical evidence of real workflow integration.

DO NOT include any text outside the JSON object.
```
