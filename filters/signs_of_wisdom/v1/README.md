# Signs of Wisdom Filter

**Version**: 1.0
**Status**: Active development (created 2025-12-28)
**Philosophy**: "Wisdom is a structural evaluation of a decision, not an emotional response to an event"
**Purpose**: Identify decisions and actions that demonstrate systemic wisdom - long-term thinking, humility, complexity acknowledgement

## Core Insight

"Uplifting" measures **what happened** (outcomes).
"Signs of Wisdom" measures **how decisions were made** (process quality).

A wise decision might not yet have uplifting outcomes. An uplifting outcome might have come from luck, not wisdom.

---

## Key Dimensions (6)

| # | Dimension | Weight | What it measures |
|---|-----------|--------|------------------|
| 1 | **Long-termism** | 25% | Decisions where payoff is >10 years out; sacrificing short-term for long-term |
| 2 | **Complexity Acknowledgement** | 20% | Avoiding "silver bullet" rhetoric; recognizing trade-offs and uncertainty |
| 3 | **Humility & Course Correction** | 20% | Admitting previous path was wrong; changing direction based on evidence |
| 4 | **Systems Thinking** | 15% | Seeing interconnections; avoiding siloed solutions; considering second-order effects |
| 5 | **Intergenerational Consideration** | 10% | Explicit consideration of future generations; stewardship mindset |
| 6 | **Evidence Quality** | 10% | Gatekeeper - decision based on evidence, not ideology or popularity |

---

## Scoring Philosophy

**This filter is RARE**. Most news is not about wise decisions - it's about events, outcomes, conflicts.

Expected distribution:
- 80% of articles: Score 0-3 (not about decision-making, or poor decision-making)
- 15% of articles: Score 3-5 (some wisdom signals, but mixed)
- 5% of articles: Score 5+ (genuine signs of wisdom)

---

## In Scope vs Out of Scope

**IN SCOPE (score normally):**
- Policy decisions with explicit long-term framing
- Leaders/institutions admitting mistakes and changing course
- Decisions that sacrifice short-term gains for long-term benefit
- Indigenous/traditional governance being adopted
- Systemic reforms that acknowledge complexity
- Intergenerational contracts (climate, debt, infrastructure)

**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Outcomes without decision-making context (just "good thing happened")
- Speculation about wise decisions ("this could lead to...")
- Rhetoric without action ("we must think long-term" but no policy change)
- Individual wisdom (personal life choices, self-help)
- Corporate strategy (unless genuine public benefit and long-term sacrifice)

---

## Examples

### High Score (8.0+)
- **New Zealand's Wellbeing Budget**: GDP deprioritized for wellbeing metrics. Long-term, systems thinking, explicit intergenerational framing.
- **Costa Rica reversing deforestation**: 30-year project, multiple governments, sacrificed short-term logging revenue.
- **Montreal Protocol**: Nations agreed to sacrifice economic convenience for ozone layer. Course-corrected when evidence showed harm.

### Medium Score (5.0-7.0)
- **City banning cars from downtown**: Long-term thinking, but limited scope. Some complexity acknowledgement.
- **Company choosing B-Corp status**: Sacrifices some profit for purpose, but still corporate framing.

### Low Score (0-3)
- **Solar farm built**: Good outcome, but no decision-making wisdom visible in article.
- **Politician promises long-term thinking**: Rhetoric without evidence of action.
- **Tech CEO's 10-year vision**: Corporate strategy, not public wisdom.

---

## Relationship to Other Filters

| Filter | Overlap | Distinction |
|--------|---------|-------------|
| `uplifting` | Both care about positive outcomes | Wisdom = process, Uplifting = outcome |
| `sustainability_technology` | Tech decisions could show wisdom | Wisdom focuses on decision quality, not tech |
| `nature_recovery` | Recovery could result from wise policy | Wisdom looks at the policy decision itself |

---

## Tier Classification (Postfilter)

| Tier | Score Range | Meaning |
|------|-------------|---------|
| **Profound** | 8.0+ | Landmark wisdom - historic decisions, paradigm shifts |
| **Notable** | 5.0-7.9 | Clear wisdom signals - worth highlighting |
| **Emerging** | 3.0-4.9 | Some wisdom, mixed signals |
| **Absent** | 0-2.9 | No wisdom signals, or poor decision-making |

---

## Next Steps

- [ ] Develop harmonized prompt (follow uplifting v5 structure)
- [ ] Create prefilter.py (will be aggressive - most news is not about decisions)
- [ ] Create postfilter.py
- [ ] Find 50+ validation articles (this will be hard - wisdom is rare in news)
- [ ] Score training data

---

## Challenges

1. **Rarity**: Wise decisions are underreported. May need to actively source from policy/governance outlets.
2. **Subjectivity**: "Wisdom" feels subjective. Anchor to observable markers (long-term framing, course correction, etc.)
3. **Attribution**: Hard to know if decision was wise until years later. Focus on process quality, not outcome.

---

*Created: 2025-12-28*
