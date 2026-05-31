# Cultural Discovery Filter v5 — Development Status

**Last Updated:** 2026-05-31
**Status:** Phase 3 Oracle Calibration — multi-oracle in progress (Qwen3 batching, Phi4 queued)
**Driver:** llm-distillery#62 (discovery-lens leakage in cd v4 production)
**Target deployment:** ovr.news Discovery tab (ADR-038)

---

## Completed

- [x] **Filter design (Phase 1)** — 2026-05-29
  - [x] Inherited v4 dimensions: discovery_novelty (25%), heritage_significance (20%), cross_cultural_connection (25%), human_resonance (15%), evidence_quality (15%) — sum to 1.0
  - [x] No `score_scale_factor` per ADR-014 (normalization replaces it)
  - [x] No `tiers` block per ADR-016 (post-processing only; oracle outputs scores)
  - [x] `evidence_quality` gatekeeper (threshold 3.0, cap 3.0) retained from v4
- [x] **Prompt scaffold v5 (Phase 2)** — 2026-05-29 (commit 137ec45)
  - [x] 5 new pre-classification flags: F (historical_harm_reckoning), G (commemoration_memorial), H (perpetrator_biography), I (decline_loss), K (launch_announcement) — J intentionally omitted (delegated to upstream obit detector per llm-distillery#51)
  - [x] Hard-negative cohort (49 articles, `datasets/raw/cd_v5_hard_negatives_candidates.jsonl`) collected from production v4-leaked articles
  - [x] Initial hard-cap mechanism: `max_score = N` clamps all dims to N when flag fires
  - [x] 7 contrastive validation examples added (rows 13-19 of Section 2 table)
- [x] **Phase 3 oracle calibration — iterative tightening (2026-05-30 / 31)**
  - [x] **v5 prompt iteration 1** (hard cap, 2026-05-29): 49 hard-negs + 473 active-learning lane scored under Gemini Flash 2.5. Output: `datasets/scored/cd_v5_hard_negatives/` + `datasets/scored/active_learning_cd_v5_rescored/`.
  - [x] **Critical lens-design audit** (2026-05-30): found ~4 v5-scored articles (Antwerp 7-Congolese memorial, Muslim WWII memorial, Stolperstein explainer, Ghanaian genocide monument) hitting G cap despite arguably falling in carve-out intent. Same parsing-too-narrowly bug F had pre-Modigliani broadening.
  - [x] **G carve-out broadening (2026-05-30)**: mirror F's pattern. Explicitly includes (1) victim names from local/community/journalistic/descendant-led research, (2) explainer pieces about commemorative practice mechanics, (3) first-of-kind regional commemorative practices. Added validation example #20.
  - [x] **Soft-penalty conversion (2026-05-30, prompt iteration 2)**: F/G/H/I/K converted from `max_score = N` hard caps to `penalty = N` (subtract from each dim, floor 0). Preserves gradient — heritage > novelty stays meaningful for slavery content. Per ADR-015 (orthogonal lens design causes MAE cliffs; thriving v1 parked at MAE 0.94 for this failure mode). Penalty values: F=2.5, G=2.5, H=3.5 (strictest), I=2.0 (lightest), K=2.5. Re-scored same 522 → `datasets/scored/cd_v5_softpenalty_rescored/`.
  - [x] **DeepSeek V4 Flash validation (2026-05-30)**: scored same 522 with soft-penalty prompt. Gemini fires F-K on 60% of articles; DeepSeek on 26% (same prompt, 2.3× difference).
  - [x] **OUT-OF-SCOPE wording restructure (2026-05-31)**: split into CATEGORY A (inherently low-substance, max_score caps A-E) vs CATEGORY B (high-substance wrong-trajectory, penalty F-K) with explicit "may fit other lenses" framing. Fixed Gemini's spurious A-E reclassification of political_conflict / tourism_fluff articles as cultural_discovery.
  - [x] **F-K trigger tightening (2026-05-31, prompt iteration 3)** per hand-judge agent finding (DeepSeek 12, Gemini 6, Unclear 4 out of 22):
    - K narrowed: explicit anti-triggers (award ceremonies, individual speeches, opinion essays, book/film/theater reviews, retrospective coverage, obituaries are NOT K)
    - I tightened: explicit "ends on rebound" carve-out + paleontological/archaeological extinction exclusion + "READ THE FINAL PARAGRAPHS" test
    - G explicit obit exclusion: "G does NOT fire on obituaries of recently-deceased public figures"
  - [x] **522-article re-scoring (Gemini v3 + DeepSeek v3)** with tightened prompt: outputs `datasets/scored/cd_v5_softpenalty_rescored_v3/` and `cd_v5_softpenalty_deepseek_v3/`
  - [x] **Two-pool spot-check on DeepSeek v3** (Opus agent): Pool A false-neg 13/15 CORRECT, Pool B false-pos 11/15 CORRECT (4 wrong, above 20% threshold but mostly K-boundary cases)
  - [x] **Haiku cross-validation of Opus** on handjudge_30: 77% exact agreement → agent layer validated
  - [x] **Broader 2-oracle disagreement truth set** (Opus agent, n=30 stratified): DeepSeek 21 / Gemini 5 / Unclear 2 / Both wrong 2. Per-oracle accuracy on disagreement set: **DeepSeek 80.8% vs Gemini 19.2%**. Confirms v3 tightening did NOT close Gemini's K/I/G over-firing — Gemini's reading style appears to be a stable disposition.

## In Progress (2026-05-31 afternoon)

- [ ] **Multi-oracle batch calibration (Phase 3 extension)** — ADR-020 draft methodology
  - [x] Gemini Flash 2.5 v3 — 522/522 scored
  - [x] DeepSeek V4 Flash v3 — 522/522 scored
  - [ ] Qwen3:14b on gpu-server (Ollama) — running, ~30/522 at last check, ETA ~14:00
  - [ ] Phi4:14b on gpu-server — queued after Qwen3, ETA ~17:00
- [ ] **4-oracle consensus computation** (`scripts/multi_oracle_consensus.py`, written, awaits all 4 oracle outputs)
- [ ] **Agent judges on hard cases** (Opus + Haiku via Agent tool — script extract_hard_cases.py ready)
- [ ] **Formal calibration_report.md** — draft skeleton landing now, full sections fill as data lands

## Pending

- [ ] **Phase 4: Oracle pick + production retrain decision** — leading candidate is DeepSeek V4 Flash on evidence so far:
  - Per-oracle accuracy on agent-judged truth set: 80.8% (DS) vs 19.2% (Gem)
  - Dimension redundancy: DS 20% pairs above |r|=0.7, Gem 60% (DS more orthogonal — better student signal)
  - Cost: ~$2 (DS) vs ~$15 (Gem Batch) for 8K-article retrain
  - Conservative-oracle principle (user-confirmed): DS's lower F-K firing rate is the GOAL behavior, not a bug
- [ ] **Phase 4: Batch labeling** — re-score 8K v4 records under chosen oracle + v5 prompt
- [ ] **Phase 5: Training** — Gemma-3-1B + LoRA on gpu-server (~3 epochs, batch 8, lr 2e-5, head+tail 256+256). Re-merge with 522 v5-prompt cohort.
- [ ] **Phase 6a: Calibration** — isotonic regression on val set → `calibration.json`
- [ ] **Phase 6b: Normalization** — fit on production CDF → `normalization.json`
- [ ] **Phase 7: Hub upload** — `jeergrvgreg/cultural-discovery-filter-v5` (private)
- [ ] **Phase 8: NexusMind deployment** — copy filter package, restart scorer service
- [ ] **Phase 9: ovr.news integration** — verify Discovery tab picks up v5 scores via FILTER_TO_TAB

## Key Decisions This Cycle

| Decision | Source | Rationale |
|---|---|---|
| Soft penalty (subtract+floor) over hard cap (clamp) for F-K | ADR-015 (cliffs hurt MAE; thriving v1 parked for this) | Preserves dim-to-dim gradient (heritage > novelty stays meaningful) |
| Penalty values F=2.5 G=2.5 H=3.5 I=2.0 K=2.5 | Calibrated to push borderline content (honest weighted ~5-6) just below 4.5 display threshold | Empirical; revisit after training-set MAE |
| Multi-oracle calibration (4 batch + 2 agents) | data-analyzer review of ADR-020 draft | Inter-oracle consensus is best ground-truth proxy when no human labels exist |
| Conservative-oracle preference for needle-filter penalty flags | User explicit (2026-05-31) — saved as memory | False positives (over-fire → article disappears) worse than false negatives (under-fire → known leakage stays) |
| Prefilter inherits from v4 unchanged | Behavior delta would mask training-data-shift effect | F-K classification is LLM-prompt level, not regex-detectable reliably |

## Files in this Filter Package

| File | Purpose | Status |
|---|---|---|
| `config.yaml` | Dimensions, weights, gatekeeper, version | ✅ |
| `prompt-compressed.md` | Oracle prompt (v3 with all tightenings) | ✅ |
| `prefilter.py` | Rule-based blocker (thin v4 subclass) | ✅ |
| `STATUS.md` | This file — phase-by-phase tracker | ✅ |
| `DEEP_ROOTS.md` | Philosophical/scientific grounding | ✅ |
| `README.md` | Short summary + links | ✅ |
| `calibration_report.md` | Phase 3 formal artifact | 🟡 draft (sections fill as data lands) |
| `dimension_analysis/` | PCA + correlations + plots | 🟡 generation in progress |
| `model/` | LoRA adapter | ⏳ Phase 5 |
| `probe/` | e5 embedding probe (hybrid) | ⏳ Phase 6c |
| `calibration.json` | Isotonic calibration | ⏳ Phase 6a |
| `normalization.json` | Production CDF normalization | ⏳ Phase 6b |
| `training_history.json`, `training_metadata.json` | Training records | ⏳ Phase 5 |
| `base_scorer.py`, `inference.py`, `inference_hub.py`, `inference_hybrid.py` | Scorer Python | ⏳ Phase 5/6 (templated from v4) |

## Open Questions

1. **Final oracle pick** — DeepSeek strongly favored on evidence so far. Need Qwen3 + Phi4 to confirm (3rd/4th conservative voice → unanimous case for DS) or contradict (3rd/4th align with Gemini → reconsider).
2. **8K corpus re-score under v5 prompt** — required for clean training data, but ~$2 (DS) or ~$15 (Gem). Cheap relative to caliber of insight already accumulated.
3. **ADR-020 final shape** — provisional methodology vs ratified standard. Per reviewer convergence: mark PROVISIONAL until validated on 2nd filter.

---

## Post-Ship Cleanup TODO

Deferred during cd v5 retrain push (2026-05-31) to stay on critical path. Address after v5 ships to NexusMind / ovr.news:

- [ ] **Patch `CLAUDE.md` cost figure**: oracle line currently says `$0.001/article` — actual is `$0.003-0.004/article` (Gemini Flash 2.5 with v5-class 8K-token prompt). Stale figure misled multiple cost estimates today.
- [ ] **Patch `docs/agents/filter-development-guide.md` Phase 1-2 checklist** to require the belonging v1 standard (STATUS.md + DEEP_ROOTS.md + README.md). Convention is now locked in (see memory: `filter-doc-standard.md`).
- [ ] **Revise `docs/adr/draft-020-extended-oracle-calibration.md`** per 4-reviewer convergent feedback: (1) cut 5-oracle → 3-oracle as default, (2) split soft-penalty mechanism into ADR-020a (extension of ADR-015) and multi-oracle into ADR-020b, (3) replace "Expected outcome: DS wins" prediction with actual results from this retrain, (4) explicitly address Alternative 5 ("iterate prompt one more round per ADR-010"), (5) add precedence rule for conservative-oracle vs consensus-alignment when they diverge, (6) mark Status: PROVISIONAL until validated on 2nd filter.
- [ ] **Code refactor**: extract `extract_dim_score`, `smart_compress`, `build_prompt`, `pearson`, `spearman`, `wavg` into `ground_truth/oracle_utils.py`. 9 scripts have copy-pasted variants today. ~1hr work. Defer to next filter cycle (refactoring-guide reviewer recommendation).
- [ ] **OracleClient ABC refactor of `ground_truth/batch_scorer.py`**: currently hardcoded if/elif over `claude/gemini/gemini-pro/gemini-flash/gpt4`. No DeepSeek/Ollama support → caused fork pressure. 2-4hr work. Defer to first task of next filter cycle (or whenever a new oracle provider is needed).
- [ ] **Apply remaining v5 prompt tightenings** per oracle-calibration agent's final review:
   - K anti-trigger: add "festivals currently running with delivered programming are NOT K — future-tense only"
   - G carve-out tighten: "at least three previously-unidentified victims as primary subject with the identification work — not the ceremony — driving the narrative"
   - I anti-trigger: "metaphorical loss in feature headlines does NOT fire I — requires documented demographic/linguistic/community decline as primary subject"
   - K/I anti-trigger: "report verbs (announces, launches, unveils) describing already-delivered content do NOT fire K/I"
   - Add multi-flag interaction worked example (max_score + penalty stacking)
   - Add validation row demonstrating non-zero novelty under F-K penalty
- [ ] **Fix the v3→v4 prefilter regression** (`check_content_length()` no longer called before pattern matching) — documented in cd v4 prefilter docstring; defer to v6 prefilter cleanup.
- [ ] **(Optional) Compile final 4-oracle calibration_report.md sections** when Qwen3 + Phi4 batches complete + agent hard-case judges adjudicate. Worked example for ADR-020.
