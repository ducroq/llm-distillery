# The Deep Roots of Cultural Discovery

This document traces the scientific, philosophical, and editorial foundations underlying the `cultural_discovery` filter. The filter detects **new findings, insights, and connections about art, culture, and history that expand human understanding** — the past opening up, not the past being recapitulated.

---

## Why Ground This Filter?

The cultural_discovery filter risks being confused with:

- **Historical-harm reckoning** (slavery apologies, UN atrocity recognitions) — institutional response to known wrongs, not new discovery
- **Memorial / commemoration content** (memorial unveilings, anniversary ceremonies) — honoring known loss, not surfacing new insight
- **Perpetrator biography** (war criminals, weapons designers) — harm-figure framing, not victim agency
- **Decline / loss narratives** (vanishing languages, depopulation) — loss trajectory, not preservation-with-outcomes
- **Launch announcements** (festival openings, exhibition rollouts) — future-tense activity without delivered outcomes
- **Tourism listicles** (Top 10 must-see attractions) — marketing without insight
- **Celebrity art** (auction prices, market speculation) — commerce without culture
- **Cultural appropriation debates** (polarizing accusations) — division, not connection

Grounding the filter in the trajectory-of-knowledge framing (Pinker / Rosling declinism correction, archaeological / restitution / cross-cultural finding evidence) clarifies what we're actually looking for: **the past opening up rather than fading away.**

---

## Editorial Foundation: ADR-038 + Pinker/Rosling Declinism Correction

ovr.news organizes around a **bias-correction lens taxonomy**: each reader-facing lens corrects a specific cognitive bias the news industry over-amplifies. The Cultural Discovery lens (formerly "Culture", refactored 2026-04-19 per `ovr.news/docs/decisions/ADR-038-lens-taxonomy-refactor.md`) corrects two biases simultaneously:

1. **Declinism** (the bias that things are constantly getting worse). Steven Pinker (*The Better Angels of Our Nature*, 2011; *Enlightenment Now*, 2018) and Hans Rosling (*Factfulness*, 2018) documented how mainstream news systematically over-reports decline and under-reports the discoveries, recoveries, and reconciliations that constitute the actual long-term trajectory of human knowledge.

2. **Presentism** (the bias that contemporary debates and conflicts are the most important things ever). Discovery counters this by surfacing archaeological finds, restored artworks, rediscovered manuscripts, repatriated artifacts, language-revival programs — content where *the past actively opens up* in ways that recontextualize the present.

The brand metaphor (`ovr.news/docs/BRAND.md`, "Discovery is nerve activity — the past opening up") and the methodology page (`ovr.news/src/i18n/translations.ts:468`) both name this directly: *"the past opening up rather than fading away."*

---

## The Trajectory Principle (v5 Innovation)

Heritage vocabulary (*slavery, colonial, ancient, indigenous, heritage, memorial*) is **necessary but not sufficient**. The same set of words appears across very different article trajectories:

| Trajectory | Article shape | In scope? |
|---|---|---|
| **Discovery** | New finding → expanded understanding → connection built | ✅ Yes |
| **Reckoning** | Known wrong → political/institutional response → ongoing debate | ❌ No (Flag F penalty −2.5) |
| **Commemoration** | Past event → ritual remembrance → no new content | ❌ No (Flag G penalty −2.5) |
| **Loss** | Former state → decline → diminished present | ❌ No (Flag I penalty −2.0) |
| **Harm-figure** | Perpetrator's life → legacy of damage | ❌ No (Flag H penalty −3.5) |
| **Announcement** | New initiative → launched → outcomes TBD | ❌ No (Flag K penalty −2.5) |

The v5 oracle prompt's Section 3 Pre-Classification step applies these distinctions via flag definitions with carve-outs. The carve-outs matter — repatriation tied to commemoration (e.g., Modigliani returned), victim names surfaced for the first time by community research (e.g., Antwerp 7-Congolese memorial), explainer pieces about commemorative practice mechanics (e.g., Stolperstein origins) — all carve out into legitimate Discovery content even when the surface vocabulary triggers a flag.

---

## Scientific Foundation: What Counts as Discovery

### Archaeology and the Material Record

The filter draws on the standards of contemporary academic archaeology and material culture studies:

- **Primary evidence required** (peer-reviewed publication, lab analysis, documented provenance) — distinguishes archaeological *findings* from speculation
- **Physical objects returned in restitution** — Modigliani-to-Jewish-heirs, repatriated indigenous remains, looted artifacts returned to origin communities — concrete outcomes, not just intentions
- **First-of-kind regional practices** — Ghanaian university memorializing colonial-era genocide victims, first-ever Stolpersteine in a regional capital — *the act of opening* is the discovery

### Linguistic Recovery

- **Documented language-family connections** — when new evidence rewrites family trees (e.g., Indo-Anatolian linkage refinements)
- **Language revival with measurable outcomes** — Cornish, Hawaiian, Welsh classroom enrollment growth, intergenerational transmission rates

### Cross-Cultural Connection Findings

- **Trade-route evidence** that connects previously-isolated civilizations (e.g., pre-Columbian Mesoamerican-Andean trade, Maritime Silk Road material evidence)
- **DNA-based connection findings** that bridge cultural narratives (e.g., Viking-Indigenous American contact)
- **Cultural-exchange historiography** that reveals previously unrecognized influence

### Heritage Restoration with Delivered Outcomes

- **Restoration revealing hidden layers** (Uffizi frescoes, palimpsest manuscripts)
- **Conservation projects with published findings** (not just funding announcements)

---

## Philosophical Foundation: Why "Discovery" Specifically

The choice of "Discovery" as the lens name (rather than "Heritage", "Culture", or "Memory") is deliberate per ADR-038's data audit:

- ~90% of the prior "Culture" lens content was discovery-shaped (archaeology, rediscovery, historical context surfacing)
- Community-heritage content (intergenerational practice, place attachment) was split out to the Belonging lens
- "Discovery" maps cleanly to the bias-correction framing: declinism is countered by surfacing *active openings* in the historical record

This is a forward-leaning lens. It's not about preserving the past for its own sake (that's Belonging's brief) or honoring loss (which falls outside ovr.news's constructive frame entirely). It's about the act of *seeing more*.

---

## What This Filter Explicitly Does NOT Detect

- **Reckoning / accountability journalism** — important, but covered by mainstream news at high volume; not what ovr.news adds value on
- **Personal obituaries** — handled by the universal upstream obit detector (`filters/common/obit_signal.py`; trained version per llm-distillery#51)
- **Routine cultural events** — festivals, gallery openings without delivered findings
- **Cultural-war partisan content** — divisive framing fails the connection-building criterion

These exclusions are operationalized via:
- v5 oracle prompt's CATEGORY A (max_score caps A-E for low-substance content)
- v5 oracle prompt's CATEGORY B (penalty F-K for high-substance wrong-trajectory content)
- The upstream obit detector and prefilter domain blocks (VC/startup, defense, code-hosting)

---

## References

- ADR-038 (ovr.news): Lens taxonomy refactor — Discovery lens definition
- ADR-015 (llm-distillery): Lenses as perspectives, not partitions
- ADR-017 (llm-distillery): Inter-oracle MAE as distillation floor
- ovr.news `docs/BRAND.md` — "Signs of life" brand framework
- ovr.news `src/lib/data/editor/prompts.ts:87` — Discovery lens rubric used by Chief Editor LLM gate
- Pinker, S. (2011). *The Better Angels of Our Nature*; (2018). *Enlightenment Now*.
- Rosling, H. (2018). *Factfulness: Ten Reasons We're Wrong About the World*.
- Pope Leo XIV apology (May 2026) — example F-flag, validation row 13
- Antwerp 7-Congolese memorial (April 2026) — example G carve-out, validation row 20
