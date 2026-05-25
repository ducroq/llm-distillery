# Hypothesis-Log Excerpts — R&D Loop Evidence

**Source:** `ovr.news/docs/hypothesis-log.md` (private repo), as of 2026-05-22.
**Scope of this file:** Two specific entries cited in the NLnet NGI Zero Commons Fund R&D-led restart claim registry (R-12 and R-13). The full hypothesis log contains 17 open + 8 resolved entries as of 2026-05-22; many entries cover business strategy, funding posture, or editorial-policy deliberation that is out of scope for this evidence companion. Only the two R&D-process entries are reproduced here.

**Why these two entries:** They illustrate the structured R&D-loop format (provisional position → method → counter-hypothesis → revisit trigger → dated updates) that the claim registry treats as evidence of R&D-primary work (K-3 in NLnet's call). Both demonstrate hypotheses revised under measurement, not just enumerated as plans.

---

## Format used across the hypothesis log

Every open entry follows the same structure:

> **Position (provisional):** what we currently believe, with the evidence we have
>
> **Method:** how we will know if the position is wrong
>
> **Alternative / counter-hypothesis:** the next-most-likely interpretation if the position fails
>
> **Revisit trigger:** the concrete event that re-opens the entry (count threshold, calendar date, external dependency landing)
>
> **Review by:** date for forced revisit
>
> **Domain / Status**

Dated updates accumulate in-place as evidence arrives. Resolution moves the entry to the `## Resolved` section with a one-line outcome.

This format is the artifact cited as R-8 (continuous R&D operations) in the claim registry.

---

## R-12: Editorial-gate trajectory-framing collision (Discovery lens)

*Cited in the NLnet application §6 Challenge 2 (hard-negatives gap in trajectory-framing classification).*

### [2026-04-19] HERITAGE_ALIVE edges — historical-harm-adjacent content

**Position (provisional):** v5 prompt tightened HERITAGE_ALIVE to exclude "narratives primarily focused on historical harm, exploitation, or conflict" (British Empire case flipped correctly). The 600-article backfill sampled 61 HERITAGE_ALIVE passes — only 1 borderline (Nazi Party archive engagement) slipped through. Retained as open because the DISCOVERY lens (post ADR-038) inherits the same edge-case question: when does "the past opening up" cross into "historical harm"?

**Revisit trigger:** 5+ historical-harm-adjacent articles surface in audit with manual calls; check if the gate's decisions align with your editorial instinct.

**Review by:** End of Phase 2 audit week.

**Domain:** editorial-gate (ADR-037), discovery lens (ADR-038)
**Status:** open

**2026-05-11 update — trigger fired; evidence collected; resolution gated on ovr.news#210 + llm-distillery#62.** User-flagged Discovery-lens degradation; audit-mode inspection found ≥10 historical-harm-adjacent articles surfaced at high score (8–9.4) in the last 3 days, spanning 20th-century-atrocity commemorations, colonial-history reckoning speeches, and biographies of weapons designers — content where heavy heritage/conservation vocabulary collides with decline- or harm-shaped narrative framing. Editorial-gate in audit mode correctly flagged 75/198 (38%) of cultural_discovery articles with reasons like "Primary subject is historical harm and lack of accountability" — gate's decisions align with editorial instinct. Trigger demonstrably fired well past the 5-article threshold. Two follow-up issues filed: **llm-distillery#62** (cultural_discovery v5 scorer retrain with historical-harm-reckoning + commemoration hard-negatives) and **ovr.news#210** (per-scorer kill-mode flip + audit-trail tooling). Hypothesis remains open as a tracking record; resolution should follow #210 landing in kill mode for cultural_discovery.

---

## R-13: Cold-load vs eviction fault-family discrimination

*Cited in the NLnet claim registry as systems-R&D depth evidence — distinct fault families peeled off as they surfaced.*

### [2026-04-23] Serialization override prevents Ollama-eviction summarize failures

**Position (provisional):** Today's 16:04 cycle's summarize was mid-run when the 19:00 cycle's nexusmind.service's ExecStartPre triggered `scorer-start.sh`, which via `Conflicts=ollama.service` killed Ollama. Four ovr.news editorial-gate filters (sustainability_technology, cultural_discovery, belonging, foresight) halted with "5 consecutive fetch failures". Commit NexusMind `416254d` adds an `ExecStartPre=` wait to nexusmind.service.d/override.conf that polls `systemctl is-active --quiet ovrnews-summarize.service` every 10s, up to 60 min, before proceeding. Expect: zero Ollama-eviction summarize failures in subsequent cycles where nexusmind + summarize would have overlapped.

**2026-05-04 update (NexusMind#194):** Today's investigation surfaced the symmetric failure direction — Ollama→scorer eviction killing NexusMind filters. Three filter watchdog kills 2026-05-02→05-03 caused by ovr.news's `ensure-ollama.sh` running `systemctl start ollama` while NexusMind was mid-pipeline. Mirror guard added in ovr.news commit `def733d`: `ensure-ollama.sh` now waits up to 60 min for `nexusmind.service` to be inactive before evicting the scorer.

Both directions of the symmetric guard are now in place. **Position scope widens accordingly:** zero Ollama-eviction summarize failures AND zero scorer-eviction NexusMind watchdog kills in cycles where the two would have overlapped.

**2026-05-05 update (cold-load failure mode, not eviction):** Run #551 at 06:04:58Z halted with the same `5 consecutive fetch failures` log line — but it was NOT eviction. Ollama was started fresh by `ensure-ollama.sh` 7s before the run; `/api/tags` returned 200 immediately (HTTP listener binds before any model loads) but the gemma3:27b cold-load took ~2:16, longer than the gate's 5-attempt halt budget. The eviction-prevention hypothesis is _independently still being tested_ — today's failure didn't exercise it. The shared log line conflates two distinct failure families. Fix: `ensure-ollama.sh` now does a 1-token `/api/generate` warmup (forces VRAM load) before returning ready. New _method addition_: when counting recurrences of `5 consecutive fetch failures`, separate cold-load (Ollama just started, model not yet warm) from eviction (Ollama was running, then stopped via `Conflicts=`). The `nexusmind.service` state at the time of the halt is the discriminator — eviction means scorer was active in the same window; cold-load means no overlap.

**Method addition:** count occurrences of the new log line `"NexusMind pipeline active — waiting up to 60 min before evicting scorer"` in ovrnews-summarize journals. Tells you how often the new guard fires (and indirectly: how often the `:02`/`:30` stagger fails to hold).

**Alternative addition:** ovrnews-summarize needlessly waits because `nexusmind.service` is in spurious `active` (post-run shutdown straggler, reload). Symmetric to the original alternative. Signal: summarize cadence slippage observable in pipeline-runs SQLite or in the NexusMind-side latency dashboard.

**Alternative:** Summarize occasionally stays in `activating` state for reasons unrelated to active work (post-run flush, dashboard deploy), causing nexusmind to needlessly wait its full 60-min cap, delaying the pipeline chain.

**Method:** Check service-chain journals over 2–3 weeks for (a) any recurrence of "editorial-gate: Ollama unreachable — 5 consecutive fetch failures" in ovrnews-summarize logs, and (b) any nexusmind runs where the new ExecStartPre wait took >5 min and wasn't justified by genuine in-flight summarize work.

**Revisit trigger:** 2026-06-06, or sooner if a post-mortem flags either symptom.

**2026-05-09 update:** Review-by date arrived 2 days ago without a formal journal review. Anecdotal status since 2026-04-23: zero Ollama-eviction summarize failures observed; the 2026-05-05 cold-load case was separated out and fixed via `ensure-ollama.sh` warmup; the 2026-05-04 mirror guard added the symmetric direction (NM#194 / commit `def733d`); the 2026-05-08 evening NM#206 entry exposed a different timeout-watchdog issue but not eviction. Distinct fault families have been peeled off as they surfaced; no recurrence of the original eviction symptom in the residual.

**Domain:** pipeline orchestration (systemd), gpu-server VRAM contention
**Status:** open

---

*End of excerpts. Full hypothesis log is internal to ovr.news; reviewers can request access via the contact in the NLnet application if needed.*
