# AI-Augmented Practice Filter v1.0

## Purpose

Track **empirical evidence** of how generative AI (LLMs) transforms cognitive work practices.

**Focus**: Real workflow integration, not hype, benchmarks, or speculation.

## What This Filter Captures

✅ **Workflow transformation reports** - "How we integrated Claude into legal review: 6-month study"
✅ **Empirical evidence** - Before/after metrics, A/B tests, longitudinal studies
✅ **Trust patterns** - How practitioners verify AI output
✅ **Failure modes** - What breaks, hallucinations caught, edge cases
✅ **Skill evolution** - New skills needed (prompt engineering, AI output evaluation)
✅ **Human-AI task allocation** - What each does best

❌ **Model benchmarks** - "GPT-5 beats Claude on MMLU"
❌ **Funding news** - "OpenAI raises $10B"
❌ **Speculation** - "AI will revolutionize everything"
❌ **Generic overviews** - "What is ChatGPT?"

## 8 Scoring Dimensions

| Dimension | Weight | Focus |
|-----------|--------|-------|
| **Workflow Integration Depth** | 20% | Toy demo → Core workflow → Institutional transformation |
| **Empirical Evidence Quality** | 18% | Anecdote → Survey → A/B test → Published research |
| **Trust/Verification Patterns** | 15% | Blind acceptance → Systematic validation → Formal protocols |
| **Cognitive Task Specificity** | 12% | Generic assistant → Specific task → Precise workflow step |
| **Failure Mode Documentation** | 12% | No failures → Examples → Systematic taxonomy |
| **Human-AI Division of Labor** | 10% | Unclear → Task division → Sophisticated handoffs |
| **Skill Evolution** | 8% | No discussion → New skills identified → Training programs |
| **Organizational Dynamics** | 5% | Individual use → Team patterns → Institutional change |

## Tier System

### Transformative Practice (8.0-10.0)
Deep workflow integration with rigorous empirical evidence.

**Example**: "18-month study: AI code review reduced bugs 40%, but introduced new verification bottleneck. Team created hybrid validation protocol."

### Validated Adoption (6.0-7.9)
Real usage with measurable outcomes and validation patterns.

**Example**: "Our legal team uses Claude for contract drafting. Lawyer review found 12% error rate, created checking protocol."

### Emerging Practice (4.0-5.9)
Early adoption with some empirical evidence.

**Example**: "Survey of 50 developers: 80% use Copilot occasionally, mixed results on productivity."

### Speculation (<4.0)
Hype, speculation, or no real usage evidence.

**Example**: "AI will transform knowledge work" (no current practice data)

## Gatekeeper Rule

- **Empirical Evidence Quality < 4.0** → Cap score at 3.9
- Reasoning: Must have real evidence, not just opinions/speculation

## Target Use Cases

1. **Personal Learning Feed** - Track how peers are actually using AI
2. **Workflow Adoption** - Identify proven patterns to replicate
3. **Failure Mode Learning** - Learn from others' mistakes
4. **Newsletter** - "AI-Augmented Practice Weekly"

## Expected Pass Rate

**~5-10% of AI/LLM articles** will pass prefilter.

Most AI news is:
- Model releases/benchmarks (40%)
- Business/funding news (20%)
- Speculation/futurism (20%)
- Generic overviews (10%)
- **Empirical practice reports (10%)** ← What we want

## Example Scores

### High Score (8.7): Legal Contract Review Case Study

```
Title: "How we integrated AI into legal contract review: 18-month study"

Scores:
- Workflow Integration: 9 (core workflow redesigned, institutional adoption)
- Empirical Evidence: 9 (18-month study, quantitative metrics, control group)
- Trust Patterns: 8 (formal validation protocol, documented failure modes)
- Task Specificity: 8 (precise task: contract clause review)
- Failure Documentation: 9 (systematic failure analysis, 12% error rate)
- Division of Labor: 8 (AI drafts, lawyer validates, hybrid workflow)
- Skill Evolution: 7 (new role: AI output reviewer)
- Org Dynamics: 7 (firm-wide policy, training program)

Overall: 8.7 → Transformative Practice
```

### Medium Score (5.3): Developer Survey

```
Title: "Developer survey: How we use GitHub Copilot"

Scores:
- Workflow Integration: 5 (regular use, not core workflow yet)
- Empirical Evidence: 6 (survey of 200 devs, some metrics)
- Trust Patterns: 5 (informal spot-checking)
- Task Specificity: 6 (specific tasks: boilerplate, tests)
- Failure Documentation: 4 (some examples, not systematic)
- Division of Labor: 6 (AI suggests, dev reviews)
- Skill Evolution: 4 (vague "learning prompting")
- Org Dynamics: 3 (individual adoption, no team policy)

Overall: 5.3 → Emerging Practice
```

### Low Score (2.1): Generic Hype

```
Title: "ChatGPT will revolutionize knowledge work"

Scores:
- Workflow Integration: 2 (no real usage described)
- Empirical Evidence: 1 (pure speculation)
- Trust Patterns: 0 (not discussed)
- Task Specificity: 2 (vague "productivity")
- Failure Documentation: 0 (no failures mentioned)
- Division of Labor: 2 (unclear)
- Skill Evolution: 0 (not discussed)
- Org Dynamics: 0 (not discussed)

Overall: 2.1 → Speculation (GATEKEEPER capped at 3.9 anyway)
```

## Training Data Strategy

- **Target**: 2,000 oracle-labeled samples
- **Expected cost**: ~$20 (2K × $0.01 per label)
- **Model**: Qwen2.5-7B fine-tuned
- **Expected accuracy**: 88-92%

## Deployment

Local 7B model for fast, free filtering of your AI news feed.

**Personal workflow**:
1. Content aggregator pulls AI news daily
2. Prefilter blocks 90-95% (hype/benchmarks/funding)
3. Trained 7B model scores remaining 5-10%
4. High-scoring articles (≥6.0) go to your reading list
5. Learn from peers' real experiences, avoid hype
