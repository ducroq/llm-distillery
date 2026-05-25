# ADR-038: Lens taxonomy refactor — 5 lenses, scorers decoupled

**Date**: 2026-04-19
**Status**: Accepted
**Deciders**: Jeroen Veen, Claude
**Tags**: architecture, editorial, brand, lenses

---

## Context

ovr.news has six active lenses in code (Thriving, Belonging, Recovery, Solutions, Culture, Foresight), one ghost lens (Resilience — listed in BRAND.md but absent from code for weeks), and six NexusMind scorers mapped one-to-one to lenses via `FILTER_TO_TAB`.

Three observations accumulated over 2026-04-19's Phase 2 audit work challenge this structure.

**Observation 1: Foresight is weak as a reader-facing lens.**

Foresight has the smallest eligible article pool (130 articles vs Cultural Discovery's 586) and the highest editorial-gate reject rate (70% in the backfill sample of 100). Of 70 rejects, 69 were policy announcements, pledges, conflicts, or awards — _not_ foresight-shaped content. Of 30 publishes, most could credibly fit Solutions (DOCUMENTED_OUTCOME 17, HERITAGE_ALIVE 7, DOCUMENTED_INSTANCE 6). Long-horizon decisions with delivered outcomes are naturally a facet of Solutions; genuine timescale-is-the-story content is rare in news.

**Observation 2: Culture is doing two jobs that should separate.**

The broadened Culture lens (ADR-037 Phase 2a) admits both living cultural practice (Māori language enrollment, Sardinian shepherds) and historical/archaeological discovery (Lapis Lazuli, Egyptian ostraca, Ancient DNA). Sampling 61 HERITAGE_ALIVE passes in the backfill shows ~90% are discovery-shaped (archaeology, rediscovery, historical context), ~10% are community-heritage. These are distinct reader experiences. Community heritage belongs with Belonging (intergenerational practices, community bonds). Archaeological and historical discovery deserves its own shape — which the user has explicitly signalled wanting ("I love historical thingies").

**Observation 3: Resilience has been a ghost, and the bias-correction framework is a design tool — not a product frame.**

BRAND.md's table pairs each lens with a cognitive bias (Negativity → Thriving, Catastrophizing → Resilience, etc.). Pretty intellectual structure, but the lenses have to serve _reader experience_, not just bias-correction symmetry. When a bias has no supply of scorable constructive content (Resilience), the lens can't ship. Keeping it listed in BRAND.md as aspirational creates ambient debt. Some biases naturally fold (Declinism + Presentism can both be corrected by Discovery; Learned helplessness + Short-termism can both be corrected by Solutions with a long-horizon facet).

**Forces:**

- The scorer layer and the lens layer serve different purposes. Scorers are _input signals_ (what NexusMind can identify). Lenses are _presentation and editorial output_ (how readers browse). They do not have to map one-to-one.
- Phase 3 lens-fit (ADR-037) is designed to reassign articles between lenses based on LLM judgment. If scorers feed default lenses and lens-fit adjusts edge cases, the lens structure can evolve independently of the scorer set.
- Refactoring the lens set without touching NexusMind is cheap: ~8-10 local files, ~half a day of work.
- A Thriving-specific scorer was attempted previously in llm-distillery and initially failed. This experience suggests scorer work is domain-specific research that should not be coupled to every lens-structure revision.

---

## Decision

**Reduce to five lenses and decouple them from scorers.**

### The five lenses

| Lens          | Finds                                                                                                             | Bias(es) corrected                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| **Thriving**  | People thriving, health improving, lives getting materially better                                                | Negativity bias                     |
| **Belonging** | Community bonds, rootedness, intergenerational ties and practices                                                 | Atomization                         |
| **Recovery**  | Ecosystems healing, species returning, measurable ecological repair                                               | Availability heuristic              |
| **Solutions** | Technology, policy, and initiatives that demonstrably work — including long-horizon plans with delivered outcomes | Learned helplessness, Short-termism |
| **Discovery** | Archaeology, rediscovered knowledge, historical context surfacing, ancient cultures understood in new ways        | Declinism, Presentism               |

### What's dropped

- **Foresight** as a lens. Long-horizon decisions become a facet of Solutions (surfaced editorially, not as a tab).
- **Culture** as a lens. Content splits: community-heritage → Belonging; archaeology/history/rediscovery → Discovery.
- **Resilience** from BRAND.md. The aspirational listing created ambient debt; we remove it rather than keep it ghost. It can be proposed as a future ADR if a scorer can produce durable resilience-shaped content.

### Scorer-lens mapping (default before lens-fit)

NexusMind scorers stay as-is. All six continue producing signal. Each maps to a default lens (the one its content most commonly fits), and Phase 3 lens-fit reassigns edge cases per article.

| Scorer (NexusMind)          | Default lens                                                                |
| --------------------------- | --------------------------------------------------------------------------- |
| `uplifting`                 | thriving                                                                    |
| `belonging`                 | belonging                                                                   |
| `nature_recovery`           | recovery                                                                    |
| `sustainability_technology` | solutions                                                                   |
| `foresight`                 | **solutions** (was: foresight; data shows most content is solutions-shaped) |
| `cultural_discovery`        | **discovery** (was: culture; data shows ~90% content is discovery-shaped)   |

### Decoupling principle

> Scorers identify editorial angles NexusMind can find signal on. Lenses are the reader-facing presentation. The two evolve independently. Scorer work (new scorers, retraining, calibration in llm-distillery) is scheduled per scorer; lens work (taxonomy, branding, routes) is scheduled per lens. Phase 3 lens-fit bridges the two by reassigning articles when the scorer-chosen default doesn't match the best editorial fit.

---

## Consequences

### Positive

- Lens taxonomy is organised around reader experience, not scorer availability.
- Drops a lens (Foresight) that the audit data already shows is weak.
- Splits Culture into two clearer shapes (community-heritage → Belonging; discovery → Discovery).
- Closes the Resilience ghost.
- Scorer work becomes independent of lens-structure decisions. A future Thriving retraining, or a new Discovery scorer, can ship without re-opening lens taxonomy.

### Negative

- Users who previously navigated to `/foresight` get a redirect or 404 (needs handling).
- The bias-correction framework loses some symmetry (two biases each for Solutions and Discovery; others 1:1). Narrative needs a small rewrite in BRAND.md.
- Feed cards currently tagged with `foresight` or `culture` need re-tagging.
- Cached articles in production `editorial_decisions` may have `lens='culture'` or `lens='foresight'` in their rows. Stale but non-blocking; new decisions write the new lens names.

### Neutral

- NexusMind untouched. No upstream work.
- Phase 3 lens-fit design unchanged; it operates on the new lens set as soon as the code is updated.

---

## Alternatives Considered

### Alternative 1: Keep 6 lenses, swap Foresight for Discovery

Keep Culture, drop Foresight, add Discovery.

**Pros:** Even cheaper — one lens rename, no Culture split.
**Cons:** Leaves the Culture community-vs-discovery duality unresolved. The broadened rubric (ADR-037 Phase 2a) acknowledged it was a stretch. Better to finish the separation now.

**Why not chosen:** Taking half the fix is worse than taking the whole fix when the extra cost is marginal.

### Alternative 2: Build Resilience properly, keep 7 lenses

Commit to the full BRAND.md framework. Build a Resilience scorer in llm-distillery, add the lens, ship.

**Pros:** Full bias-correction symmetry. BRAND.md as written.
**Cons:** Blocks on scorer research (unknown timeline, historically hard for this class of content). Keeps Foresight live despite its weak data. Doesn't solve Culture's split personality.

**Why not chosen:** Resilience can be added later as its own ADR when a scorer exists. Keeping it as aspirational debt doesn't help ship.

### Alternative 3: Do nothing, let Phase 3 lens-fit move articles at runtime

Keep the 6 lenses. Phase 3 reassigns articles. Nature will sort itself out.

**Pros:** Zero refactor.
**Cons:** Phase 3 reassigns _to an existing lens_. If Foresight stays in the lens set, lens-fit can't reassign OUT of it — it would just confirm or move within the existing set. Doesn't fix the weak lens. Doesn't fix the Culture split. Doesn't close the Resilience ghost.

**Why not chosen:** Lens-fit is a runtime adjustment, not a taxonomy decision. Structural fixes need structural changes.

---

## Implementation Plan

1. Update `src/lib/data/filters.ts` — `TabName`, `FILTER_TO_TAB`, `FILTERS` constants
2. Update `src/lib/data/editor/prompts.ts` — `LENS_RUBRIC` (drop foresight, drop culture, add discovery), update lens-specific examples
3. Update `src/config.ts` — `tabs` list
4. Rename route `[lang]/foresight.astro` → `[lang]/discovery.astro` (plus the legacy path under `/artikel`)
5. Update home page + navigation components to show the new tab set
6. Update `docs/BRAND.md` — lens table, remove Resilience row, adjust bias-correction narrative
7. Update `CLAUDE.md` — lens list, brand description
8. Update feed cards JSON seed — re-tag any `foresight` or `culture` entries
9. Update tests — fixtures referencing old TabName values
10. Update hypothesis log — close "Foresight merge/keep" (resolved by this ADR), HERITAGE_ALIVE edges (subsumed by Discovery)
11. Add a redirect from `/foresight` → `/discovery` or home, per deployment preference

---

## Related Decisions

- [ADR-025: Brand identity — Signs of life](./ADR-025-brand-identity-signs-of-life.md) — establishes the brand framework this ADR refines
- [ADR-037: LLM editorial review](./ADR-037-llm-editorial-review.md) — Phase 3 lens-fit runs on the new lens set
- NexusMind: future ADR if a dedicated scorer per lens is pursued (e.g., Discovery scorer, Thriving retry)

---

**Last Updated**: 2026-04-19
**Version**: 1.0
