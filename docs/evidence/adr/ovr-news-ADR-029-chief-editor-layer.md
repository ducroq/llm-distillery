# ADR-029: Chief Editor — editorial selection layer

**Date**: 2026-04-02
**Status**: Accepted
**Deciders**: Jeroen Veen, Claude
**Tags**: architecture, editorial, pipeline

---

## Context

**What is the issue we're facing?**

ovr.news takes the top-N articles per lens by NexusMind score and summarizes them. This produces a feed with systematic blind spots:

- Scientific sources (arXiv, PubMed) score ~7.5 avg but need 8+ to appear — peer-reviewed evidence is effectively invisible (NexusMind#108)
- Anglo/English-language news dominates due to RSS feed composition
- Multiple articles on the same breakthrough all score 8+ and all appear
- Engaging writing outcompetes dry-but-rigorous scientific papers on human resonance dimensions

A real editorial desk applies judgment about what the reader needs to see, not just what scores highest.

**Forces:**

- NexusMind scores must stay honest (used by Aegis, dashboard, scoring integrity)
- Editorial rules change often (thresholds, quotas, new rules) — must be easy to tweak
- The build pipeline is TypeScript/Node/Astro — adding a second runtime has a maintenance cost
- The dataset is small (~50 articles per lens) — no need for heavy data tooling
- Rules must be transparent and auditable — editorial policy, not magic

---

## Decision

**Add a Chief Editor module at build time, implemented in TypeScript with config-driven rules.**

The Chief Editor is a composable pipeline of independent editorial rules that runs during the Astro build, after all articles are summarized and stored in SQLite. It filters and balances the article pool before display.

**Key aspects:**

- **Build-time, not summarize-time.** Summarization budget is not a constraint. Running at build time means the Chief Editor operates on the full pool of enriched, summarized articles in SQLite with no coupling to the summarization script.
- **TypeScript, config-driven.** Rules are TS pure functions parameterized by a JSON config file. Thresholds and quotas change in config without touching code. New rule logic is a new module — no existing rules touched.
- **Modular rule architecture.** Each rule is an independent module with a shared interface: takes articles in, returns articles out, with an audit trail. Rules compose in sequence via a thin orchestrator.
- **No LLM.** The five editorial rules are structural metadata filters (source type, geography, recency, topic overlap). Deterministic logic is the right tool. Topic clustering (the one case where an LLM could help) is already partially covered by Jaccard dedup, and can be added as a single rule later if needed.

**Module structure:**

```
src/lib/data/editor/
  chief-editor.ts        — orchestrator: runs rules in sequence
  editor-config.json     — thresholds, quotas, rule toggles
  types.ts               — EditorRule interface, EditorDecision type
  rules/
    source-diversity.ts   — ensure scientific sources get slots
    story-dedup.ts        — cluster similar stories, pick best
    geographic-balance.ts — soft cap per region
    recency-balance.ts    — mix breaking + slow evidence
    topic-diversity.ts    — avoid sub-topic flooding
```

**Pipeline position:**

```
summarize.ts (unchanged)         pipeline.ts (build-time)
score → enrich → produce    →   fetch from DB → Chief Editor → dedup → display
```

**Initial editorial rules:**

1. **Source diversity quota** — reserve slots per lens for scientific/peer-reviewed sources above a minimum score
2. **Story deduplication** — cluster similar stories, keep best-scored, note corroboration (subsumes existing `dedup.ts`)
3. **Geographic balance** — soft cap on articles from any single country/region per lens
4. **Recency vs depth** — balance breaking news with slower evidence-rich pieces
5. **Topic diversity** — limit articles on the same sub-topic even when all score high

---

## Consequences

### Positive

- Clean separation of concerns: scoring (NexusMind), production (summarize.ts), selection (Chief Editor), display (Astro)
- Editorial policy is readable in one config file — non-developers can understand the choices
- Each rule is independently testable, toggleable, and replaceable
- No new runtime or infrastructure — stays within the existing Node/TS stack
- Existing `dedup.ts` logic migrates into the rule framework rather than being duplicated

### Negative

- New rule logic still requires a code change and deploy (config only covers parameterization)
- Geographic balance requires country/region data that is not currently in the schema — needs extraction or upstream enrichment
- More build-time processing, though negligible on a dataset of ~250 articles total

### Neutral

- Summarization budget is spent on articles that may not be selected for display — acceptable given budget is not a constraint
- The orchestrator imposes a sequential rule ordering that may affect results depending on order — document and test

---

## Alternatives Considered

### Alternative 1: Python editorial layer

**Description:** Implement the Chief Editor in Python, using pandas for data manipulation, with potential for scikit-learn embeddings for topic clustering.

**Pros:**

- More natural for data filtering and analysis
- Easier path to embedding-based topic clustering

**Cons:**

- Introduces a second runtime into the Astro build pipeline
- Team maintains two languages for one pipeline
- Overkill for filtering ~50 articles per lens

**Why rejected:** The dataset is small and the operations are simple filters. Adding Python to a Node build pipeline has a real maintenance cost with no proportional benefit.

### Alternative 2: Pure SQL views

**Description:** Express all editorial rules as SQL queries and views over the SQLite database.

**Pros:**

- Zero runtime dependency — SQL is already there
- Naturally expressive for quotas and caps (window functions)

**Cons:**

- Complex rule logic becomes hard to read and debug in SQL
- No composable audit trail per rule
- Difficult to unit test individual rules

**Why rejected:** Simple quotas work well in SQL, but editorial rules will evolve and need the testability and composability that a proper module structure provides.

### Alternative 3: LLM-based editorial selection

**Description:** Use an LLM to make editorial judgments about which articles to feature, given the full pool and editorial guidelines.

**Pros:**

- Can handle nuanced judgment calls (thematic overlap, editorial voice fit)
- Flexible — change the prompt instead of code

**Cons:**

- Non-deterministic: same input can produce different selections
- Adds latency and cost to every build
- Harder to audit and explain editorial decisions
- The actual rules are structural metadata filters, not language tasks

**Why rejected:** Four of five rules are deterministic filters on metadata. An LLM adds unpredictability and cost for decisions that don't require language understanding. If topic clustering later needs semantic similarity, it can be added as one rule using embeddings, not a full LLM call.

---

## Implementation

1. Define `EditorRule` interface and `EditorDecision` audit type in `src/lib/data/editor/types.ts`
2. Create orchestrator in `src/lib/data/editor/chief-editor.ts` that composes rules
3. Create `editor-config.json` with initial thresholds and rule toggles
4. Implement rules incrementally (source diversity and topic diversity first, geographic balance last — needs schema work)
5. Integrate into `pipeline.ts` between DB fetch and existing dedup/display logic
6. Migrate existing `dedup.ts` into `rules/story-dedup.ts`
7. Add tests per rule with fixture data

---

## Related Decisions

- [ADR-028: Title dedup](./ADR-028-title-dedup.md) — Story dedup rule subsumes and extends this
- [ADR-025: Brand identity](./ADR-025-brand-identity.md) — Editorial rules serve the "Grounded" pillar

---

## References

- [Issue #161](https://github.com/ducroq/ovr.news/issues/161) — Chief Editor: editorial selection layer between scoring and display
- [NexusMind#108](https://github.com/ducroq/NexusMind/issues/108) — Scientific sources scoring below cutoffs

---

## Notes

- Geographic balance (rule 3) depends on country/region data. This may come from NexusMind enrichment (TLD-based or metadata extraction) or a lightweight heuristic on source URL. Defer until data is available.
- Rule ordering matters: dedup should run before quotas (so duplicates don't consume slots). Document the canonical order in the orchestrator.
- The existing `dedup.ts` at build-time and the new story-dedup rule overlap. Migration should be explicit — remove the old call site when the rule is active.

---

**Last Updated**: 2026-04-02
**Version**: 1.0
