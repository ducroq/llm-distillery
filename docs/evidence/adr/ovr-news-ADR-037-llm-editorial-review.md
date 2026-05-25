# ADR-037: LLM editorial review — multi-pass quality gate

**Date**: 2026-04-16
**Status**: Accepted — Phase 1 (infrastructure) shipped 2026-04-17; Phases 2-5 pending
**Deciders**: Jeroen Veen, Claude
**Tags**: architecture, editorial, pipeline, llm

---

## Context

**What is the issue we're facing?**

The current pipeline builds the newspaper in a single pass: NexusMind scores articles by topic relevance, ovr.news summarizes the top-N, the Chief Editor applies structural rules (dedup, diversity, corroboration), and the result is published. No human editor reviews the output.

This works for most articles, but produces systematic errors that rule-based systems cannot catch:

1. **Non-constructive content passes scoring.** The nature_recovery filter gave a 9.17 weighted average to "Colombia to Euthanize Escobar's Cocaine Hippos" — a story about killing animals with no recovery outcome. The scorer rewards topic relevance (invasive species, ecosystem management) but cannot judge editorial fit (NexusMind#171).

2. **Articles land in the wrong lens.** A study about decolonial research methods gets classified as Foresight when it could be Belonging or Culture. The NexusMind filter that scores highest isn't always the lens where the article serves readers best.

3. **Images don't match content.** An article about Syrian community voices gets a stock laptop photo because the curated image fallback matches on the keyword "research" (ovr.news#200).

4. **No editorial judgment on the final product.** A real newspaper has an editor who reads the page proof before print. We don't. The pipeline is a one-pass assembly line where each stage trusts the previous stage's output.

**The insight:** We cannot expect a newspaper to be built first time right. Real editorial processes are iterative — draft, review, revise. The pipeline should reflect this.

**Forces:**

- Ollama (gemma3:27b) is already in the pipeline for summarization — no new infrastructure
- After summarization, each article has a clean English summary, selection rationale, scores, and image — rich context for editorial judgment
- The article pool post-summarization is small (~30-50 articles per cycle) — LLM cost is trivial
- Editorial decisions must be explainable and auditable — "the editor rejected this because..." not a black box
- ADR-029 rejected LLM-based selection because the rules were structural metadata filters. This proposal is different: it addresses judgment calls that metadata filters genuinely cannot make

---

## Decision

**Add an LLM editorial review pass after summarization, before DB write and R2 upload.**

The LLM Editor runs during `scheduled_summarize.sh`, after each article is summarized but before the article is committed to the database. It uses Ollama (gemma3:27b) on gpu-server to make three editorial judgments per article:

### Pass 1: Editorial gate

> "Given this summary, does this article belong on ovr.news? Does it show documented, constructive progress — not a plan, a pledge, doom with a silver lining, or controversy?"

- **Input:** English summary, selection rationale, lens name, lens description, scores
- **Output:** `publish` | `reject` with reason
- **Prompt grounding:** The seven lens descriptions from BRAND.md, the anti-patterns (no hype, no doom, no pledges-without-outcomes), and 3-5 examples of good/bad fits per lens

### Pass 2: Lens assignment

> "If this article should be published, which lens is the best fit? The NexusMind scorer assigned it to Recovery, but given the summary, would it serve readers better under Solutions, Culture, or another lens?"

- **Input:** English summary, current lens, all seven lens descriptions
- **Output:** confirmed lens or reassigned lens with reason
- **Key rule:** The LLM may only reassign to a lens where the article was also scored by NexusMind (articles pass through all 7 filters — the second-highest scoring lens is often a better editorial fit)

### Pass 3: Image check

> "Does this image make sense for this article? A photo of a laptop keyboard for an article about Syrian communities is wrong. A photo of a solar panel for an article about solar energy is right."

- **Input:** English summary, image URL, image attribution/description, image source type
- **Output:** `keep` | `flag` with reason
- **Action on flag:** Fall back to next image in the fallback chain, or mark as no-image (better than a misleading image)

### Pipeline position

```
NexusMind scores → summarize.ts (Ollama) → LLM Editor (Ollama) → DB write → R2 → deploy
                                                   │                              │
                                            Pass 1: gate                   Chief Editor
                                            Pass 2: lens + dedup           (build-time rules)
                                            Pass 3: image                  Source diversity
```

### Division of labor with the Chief Editor (ADR-029)

The LLM Editor and the rule-based Chief Editor have distinct responsibilities. Some responsibilities that were previously in the Chief Editor move to the LLM Editor where judgment outperforms rules:

| Responsibility          | Before                           | After                 | Rationale                                                                                                                                                     |
| ----------------------- | -------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Story dedup**         | Chief Editor (Jaccard, disabled) | LLM Editor (pass 2)   | Jaccard fails on paraphrased titles; LLM reads summaries                                                                                                      |
| **Corroboration boost** | Chief Editor (rule)              | Drop — keep data only | NexusMind already computes corroboration data; the LLM Editor's editorial gate naturally promotes well-evidenced stories; a separate boost rule double-counts |
| **Source diversity**    | Chief Editor (rule)              | Chief Editor (rule)   | Collection-level balancing ("reserve slots for scientific sources") — inherently a pool decision, not per-article                                             |
| **Editorial gate**      | None                             | LLM Editor (pass 1)   | New capability — no rule can judge "is this constructive?"                                                                                                    |
| **Lens assignment**     | None                             | LLM Editor (pass 2)   | New capability — scorer's highest-scoring filter isn't always the best editorial fit                                                                          |
| **Image check**         | None                             | LLM Editor (pass 3)   | New capability — no rule can judge image-content relevance                                                                                                    |

**Principle:** The Chief Editor handles **collection-level balance** (deterministic, fast, operates on the full pool at build time). The LLM Editor handles **per-article judgment** (requires reading comprehension, runs during summarization). No responsibility lives in both.

### Implementation approach

- **Single prompt, structured output.** All three passes can be combined into one LLM call per article with a structured JSON response:
  ```json
  {
    "publish": true,
    "publish_reason": "Documents measurable ecosystem recovery with specific data",
    "lens": "recovery",
    "lens_reason": "Best fit — article centers on species returning",
    "image_ok": false,
    "image_reason": "Laptop photo unrelated to Syrian community research"
  }
  ```
- **Audit trail.** Every editorial decision is logged with the full prompt and response. The `/ops` dashboard can display rejection rates, lens reassignment frequency, and image flags per cycle.
- **Override config.** A JSON config file controls thresholds and toggles:
  ```json
  {
    "enabled": true,
    "passes": {
      "editorial_gate": { "enabled": true },
      "lens_assignment": { "enabled": true },
      "image_check": { "enabled": true }
    },
    "model": "gemma3:27b",
    "max_articles_per_cycle": 100
  }
  ```
- **Soft start.** Initially run in "audit mode" — log decisions but don't act on them. Compare LLM editor decisions against human judgment for a week before activating.

### Prompt design principles

The prompts are the editorial policy. They deserve the same care as code:

1. **Ground in BRAND.md.** The prompt includes the three pillars (Grounded, Alive, Clear), all seven lens descriptions, and the anti-patterns. The LLM editor enforces the brand constitution.
2. **Show, don't tell.** Include 3-5 concrete examples per lens of articles that should pass and articles that should be rejected, with reasons. Few-shot examples are more reliable than abstract instructions.
3. **Explain every decision.** The prompt requires a reason field for every output. If the LLM can't articulate why, the decision is suspect.
4. **No creativity.** The LLM is a reviewer, not a writer. It doesn't edit summaries, suggest rewrites, or generate content. It makes binary/categorical judgments with explanations.
5. **Conservative bias.** When uncertain, publish. False rejections (hiding good content) are worse than false passes (showing mediocre content). The prompt says: "When in doubt, publish."

---

## Consequences

### Positive

- Catches non-constructive content that metadata filters miss (hippo culling, doom-with-silver-lining)
- Enables lens reassignment — articles reach the lens where they serve readers best
- Catches image-content mismatches before publication
- Auditable: every decision has a reason, logged and displayable
- Uses existing infrastructure (Ollama on gpu-server, already warm for summarization)
- Embodies the insight that editorial quality requires iteration, not perfection-in-one-pass

### Negative

- Adds ~2-5 minutes to each summarization cycle (30-50 LLM calls at ~2-4s each)
- Non-deterministic: same article may get different decisions on re-run (mitigated by caching decisions per article hash)
- Prompt engineering becomes a maintenance surface — changes to editorial policy require prompt updates
- Introduces LLM dependency for a core editorial function (mitigated by fallback: if Ollama is unreachable, skip the review and publish as before)

### Neutral

- ADR-029's rejection of LLM editorial selection was about using an LLM for structural metadata filters. This is a different use case: editorial judgment that requires reading comprehension. ADR-029 remains valid for its scope; this ADR extends the Chief Editor with a capability that rules cannot provide.
- The rule-based Chief Editor (ADR-029) is refocused on collection-level balancing (source diversity, future geographic balance). Story dedup and corroboration boost move to the LLM Editor or are dropped. Clean separation: judgment per-article, balance per-collection.

---

## Alternatives Considered

### Alternative 1: Improve NexusMind scorers

**Description:** Add training data for edge cases (hippo culling, doom-with-pledge) so the fine-tuned classifiers catch them.

**Pros:** No new infrastructure, deterministic, fast.

**Cons:** Whack-a-mole — each new failure mode requires new training data and retraining. Cannot handle lens reassignment or image checking. Scorers are optimized for topic relevance, not editorial judgment.

**Why not chosen alone:** Necessary but not sufficient. Better scorers reduce the editor's workload but cannot replace editorial judgment. Both should happen.

### Alternative 2: Human editorial review

**Description:** A human reviews the pipeline output before publication.

**Pros:** Gold standard for editorial judgment. Catches everything.

**Cons:** Doesn't scale to 4-6 cycles/day. Single point of failure. The project aims for autonomous operation.

**Why not chosen:** ovr.news is designed to run autonomously. A human review step contradicts the architecture. However, the LLM editor's audit trail makes occasional human spot-checks easy — review the rejection log, not every article.

### Alternative 3: Run at build time instead of summarization time

**Description:** Run the LLM review during the Astro build (like the current Chief Editor).

**Pros:** Cleaner separation — summarization doesn't need GPU for review.

**Cons:** At build time, Ollama may not be running (builds happen on Cloudflare). Would require a separate API call to gpu-server during build, adding a remote dependency to the static site build.

**Why not chosen:** Running during summarization means Ollama is already warm and local. The editorial decision is cached in the DB, so the build just reads it.

---

## Implementation Plan

1. **Design prompts** — Write and test the editorial gate, lens assignment, and image check prompts against the hippo, DNA study, and Syrian research examples. Iterate until decisions match human judgment.
2. **Add audit-mode infrastructure** — Log LLM editorial decisions alongside articles without acting on them. Add to `/ops` dashboard.
3. **Run audit mode for 1 week** — Compare LLM decisions against manual review. Measure false rejection rate and lens reassignment accuracy.
4. **Activate gate** — Enable the editorial gate (pass 1). Monitor rejection rate and reasons.
5. **Activate lens assignment** — Enable pass 2. Monitor reassignment frequency.
6. **Activate image check** — Enable pass 3. Monitor flag rate.

---

## Related Decisions

- [ADR-029: Chief Editor](./ADR-029-chief-editor-layer.md) — Rule-based editorial layer; this ADR adds LLM judgment alongside it
- [ADR-035: Pipeline quality agent](./ADR-035-pipeline-quality-agent.md) — QA checks pipeline health; this ADR checks editorial quality
- [NexusMind#171](https://github.com/ducroq/NexusMind/issues/171) — Scorer passes non-constructive content (motivating example)
- [ovr.news#200](https://github.com/ducroq/ovr.news/issues/200) — Image-content mismatch (motivating example)

---

## Open Questions

1. **Should rejected articles be retried with a different lens?** If the gate rejects an article from nature_recovery, should it be re-evaluated for solutions or foresight before final rejection?
2. **How to handle disagreement?** If the LLM editor rejects an article that scored tier:high in NexusMind, should there be a confidence threshold or a second opinion?
3. **Prompt versioning.** How to track which prompt version produced which decisions, for auditability?
4. **Can pass 2 (lens assignment) use the article's scores across all 7 filters** to constrain reassignment to lenses where the article actually scored above a threshold?

---

**Last Updated**: 2026-04-16
**Version**: 1.0 (Proposed)
