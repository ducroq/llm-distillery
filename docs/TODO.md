# LLM Distillery - TODO

## Commerce Prefilter SLM - NEEDS REWORK

ML classifier for commerce/promotional content detection. Cross-cutting prefilter for all filters.

**Status:** v1 complete but needs redo - concerns about multilingual embeddings and context size.

- [x] **v1 Training data collection** - 2,847 examples (commerce + journalism)
- [x] **v1 Model training** - DistilBERT, MiniLM, XLM-RoBERTa compared
- [x] **v1 Backtesting** - 56,336 articles, threshold optimization
- [ ] **Redo with proper multilingual embeddings** - Current approach may not handle Dutch/multilingual well
- [ ] **Redo with proper context size** - May need longer context

See `filters/common/commerce_prefilter/docs/` for full documentation.

---

## Filters

### Production Ready
- [x] **uplifting v6** - Deployed on HuggingFace Hub (private)
  - Val MAE: 0.673 (was 0.688 in v5), 12% faster inference
  - Gemma-3-1B base model (was Qwen2.5-1.5B)
  - 10,495 training articles with data sculpting: active learning (495 MEDIUM enrichment) + label correction (57 crime articles capped)
  - v5 crime news issue fixed via manual label correction in training data
- [x] **uplifting v5** - Superseded by v6
  - Val MAE: 0.68, 10,000 training articles
- [x] **sustainability_technology v1** - Deployed on HuggingFace Hub
  - Test MAE: 0.690
- [x] **sustainability_technology v3** - Deployed on HuggingFace Hub (private)
  - Val MAE: 0.734 (calibrated test: 0.724), Gemma-3-1B
  - 10,608 training articles (v2 10,039 + 569 active learning enrichment)
  - All 3 inference paths: local, Hub, hybrid (probe MAE 0.91)
- [x] **sustainability_technology v2** - Superseded by v3
  - Val MAE: 0.71, 7,990 training samples
- [x] **investment-risk v6** - Deployed on HuggingFace Hub (private)
  - Val MAE: 0.497 (calibrated: 0.465), Gemma-3-1B
  - 10,448 training articles (v5 10,198 + 250 active learning enrichment)
  - Tier simplification: RED/YELLOW/GREEN/BLUE/NOISE -> high/medium_high/medium/low
  - All 3 inference paths: local, Hub, hybrid (probe MAE 0.557)
- [x] **investment-risk v5** - Superseded by v6
  - Test MAE: 0.484 (excellent)
  - 10,000 training articles
- [x] **cultural-discovery v4** - Deployed on HuggingFace Hub (private)
  - Calibrated test MAE: 0.74 (v3 was 0.77), Gemma-3-1B
  - 8,029 training articles (v3 7,827 + 202 active learning enrichment)
  - All 3 inference paths verified (local, Hub, hybrid)
  - Target: ovr.news (Wisdom tab), Busara
- [x] **cultural-discovery v3** - Superseded by v4

### In Active Development (priority: ovr.news tabs)
- [x] **belonging v1** - Deployed, val MAE 0.49 (calibrated), 7,370 articles. Next: ovr.news tab
- [x] **nature_recovery v2** - Deployed to Hub + gpu-server + sadalsuud (Hub upload actually completed 2026-04-19 after #44; prior commit claimed it without uploading)
  - Val MAE 0.53 (calibrated), probe MAE 0.49, 3,517 articles
  - v1 had zero discrimination (#41); v2 uses sample weighting (scale=2)
  - Recall@20: 0.70 (v1: 0.55), NDCG@10: 0.86 (v1: 0.71), false negatives: 17% (v1: 41%)
  - Hub: `jeergrvgreg/nature-recovery-filter-v2` (private)
  - Remaining: normalization (needs production CDF), ovr.news Recovery tab frontend
- [x] **uplifting v7** - ADR-010 prompt rewrite, deployed with hybrid inference (2026-04-06)
  - v7 prompt: scope check, anti-hallucination, reframed assessment dimensions
  - Hybrid inference: probe MAE 1.10, threshold 1.00, 0.5% FN, 1.07x speedup
  - Evolved into thriving v1: renamed, social_cohesion_impact removed, 3-run averaging planned
- [ ] ~~**thriving v1**~~ - PARKED indefinitely. Uplifting v7 (MAE 0.67) stays as Thriving tab.
  - Root cause: orthogonal lens design created bimodal distribution (ADR-015)
  - A fixed thriving v2 would converge back to uplifting v7. Not worth retraining.
  - Assets preserved in `memory/thriving-v1-scoring.md` if ever revisited
- [x] **foresight v1** - Deployed on HuggingFace Hub (private) — was signs_of_wisdom
  - Val MAE 0.75, 3,480 training articles, 6 dimensions
  - Hybrid inference: probe trained, threshold 2.25 (default, calibrate on production data)
  - Remaining: ovr.news Foresight tab frontend integration

### Active Learning In Progress
- [ ] **cultural-discovery v5** - Training data ready (8,502 articles = v4 8,029 + 473 enrichment)
  - Oracle scored 473 production MEDIUM+ articles with Gemini Flash
  - Smooth distribution (bell curve centered at WA 4.8), no bimodality
  - Next: train on gpu-server, calibrate, retrain probe, deploy
- [x] **nature_recovery v2** - Trained, calibrated, deployed (2026-04-16)
  - Sample weighting (scale=2) + active learning enrichment (237 articles)
  - Remaining: normalization (needs production CDF), hybrid threshold recalibration

### Other Filters
- [ ] ~~**future-of-education**~~ - DROPPED: education stories land naturally in Breakthroughs (research)
- [ ] **ai-engineering-practice v2** - Ready for oracle scoring (not ovr.news, separate product)
  - FluxusSource hardware sources active (1,193 articles)
  - Prompt calibration complete (~60% tier accuracy)
- [ ] **seece** - Corporate excellence (not ovr.news)
- [ ] **sustainability_economic_viability** - Sustainability sub-dimension (not ovr.news)
- [ ] **sustainability_policy_effectiveness** - Sustainability sub-dimension (not ovr.news)

## Training Pipeline

- [x] **Data preparation pipeline** - Stratified splits working
- [x] **Training script** - Gemma-3-1B + LoRA working (was Qwen2.5-1.5B)
- [x] **Context length experiments** - 1024/2048/head+tail tested
  - 1024tok: MAE 0.652, 2048tok: MAE 0.627
  - head+tail (256+256): MAE ~0.69 (deployed to production)
  - See `docs/IDEAS.md` for full results
- [x] **Stage 2 model comparison** - Gemma-3-1B adopted as default Stage 2. Wins on both uplifting (MAE 0.652 vs 0.660) and cultural-discovery (MAE 0.743 vs 0.755). 8% faster, fewer params. Qwen-0.5B rejected (MAE 0.760)
- [x] **Gemma-3-1B training support** - `training/train.py` updated with `load_base_model_for_seq_cls()` for both initial and resume paths
- [x] **Stage 2 model selection** - Gemma-3-1B adopted as default (was Qwen2.5-1.5B). Larger models deferred.
- [ ] **Training monitoring improvements** - Better logging, early stopping

## Score Calibration (ADR-008)

Post-hoc isotonic regression to correct MSE score compression at inference time.

- [x] **Shared calibration library** - `filters/common/score_calibration.py` (fit, apply, save, load)
- [x] **CLI fitting tool** - `scripts/calibration/fit_calibration.py` (works for any filter)
- [x] **Uplifting v6 calibration** - Fitted on 1,049 val articles, val MAE 0.673 -> 0.653 (+3.1%)
- [x] **Cultural-discovery v4 calibration** - Fitted on 803 val articles, test MAE 0.77 -> 0.74 (+4.4%)
- [x] **Base scorer integration** - `_load_calibration()` + `apply_calibration()` in `_process_raw_scores()`
- [x] **sustainability_technology v3 calibration** - Fitted on 1,061 val articles, test MAE 0.725 -> 0.724
- [x] **investment-risk v6 calibration** - Fitted on 1,045 val articles, val MAE 0.497 -> 0.465 (+6.5%)
- [x] **belonging v1 calibration** - Fitted on 738 val articles, val MAE 0.534 -> 0.489 (+8.3%)
- [x] **nature_recovery v1 calibration** - Fitted on 328 val articles, val MAE 0.540 -> 0.507 (+6.2%)
- [x] **nature_recovery v2 calibration** - Fitted on 352 val articles, val MAE 0.632 -> 0.533 (+15.7%)

## Hybrid Inference Pipeline (ADR-006)

Two-stage pipeline: fast embedding probe (Stage 1) + fine-tuned model (Stage 2).

- [x] **Shared infrastructure** - `filters/common/embedding_stage.py`, `hybrid_scorer.py`
- [x] **Uplifting v5 integration** - `inference_hybrid.py` + MLP probe
- [x] **Calibration script** - `evaluation/calibrate_hybrid_threshold.py`
- [x] **Threshold calibration** - Calibrated on 24K production articles. Probe retrained (v2): MAE 0.49, bias +0.007. Threshold 3.5 → 1.7% FN rate on MEDIUM+
- [x] **Speed benchmark** - RTX 4080: e5-small 1.3ms + Qwen 37.9ms. Threshold 4.5 → 2.09x on skewed data, ~2.5-3x in production
- [x] **Stage 2 model evaluation** - Gemma-3-1B adopted as default Stage 2 model. Confirmed on two filters: uplifting v5 (MAE 0.652 vs 0.660, tier 86.6% vs 85.4%) and cultural-discovery v3 (MAE 0.743 vs 0.755, tier 94.6% vs 94.5%). 8% faster inference, 38% faster training
- [x] **Generalize to other filters** - Phase A complete: inference_hybrid.py + probe dirs + calibration fix for sustainability_technology v2, investment-risk v5, cultural-discovery v3
- [x] **Train probes + calibrate thresholds** - Phase B complete: e5-small MLP probes trained and calibrated for all 3 filters
  - sustainability_technology v2: probe MAE 0.707, threshold 1.25, 1.2% FN, 1.25x speedup
  - investment-risk v5: probe MAE 0.497, threshold 1.50, 0.8% FN, 1.07x speedup
  - cultural-discovery v3: probe MAE 0.609, threshold 1.25, 0.0% FN, 1.52x speedup
- [x] **Cultural-discovery v4 probe** - Retrained for Gemma-3-1B, MAE 0.87, threshold 1.25, 3% FN, 1.51x speedup
- [x] **Sustainability_technology v3 probe** - Trained for Gemma-3-1B, MAE 0.91, threshold 1.25 (to be calibrated)
- [x] **Investment-risk v6 probe** - Trained for Gemma-3-1B, MAE 0.557, threshold 1.50
- [x] **Belonging v1 probe** - Trained for Gemma-3-1B, MAE 0.54
- [x] **Nature_recovery v1 probe** - Trained for Gemma-3-1B, MAE 0.50
- [x] **Nature_recovery v2 probe** - Retrained for v2 model, MAE 0.49 (early stop epoch 24)
- [x] **Foresight v1 probe** - Trained for Gemma-3-1B, threshold 2.25
- [x] **Foresight v1 calibration** - Fitted, calibration.json committed with filter package
- [x] **Uplifting v7 probe** - Trained for Gemma-3-1B, MAE 1.10, threshold 1.00 (#34)
- [x] **Harmonize all filters** (2026-04-06) - All 7 production filters now have hybrid inference with calibrated thresholds and `--compare` CLI. Fixed investment-risk import path bug (hyphen vs underscore). Deployed to sadalsuud + gpu-server.

## Code Quality (Feb 2026)

- [x] **FilterBaseScorer extraction** (#10) - Shared base class in `filters/common/filter_base_scorer.py`, all 4 production filters migrated
- [x] **load_lora extraction** (#11) - Shared `load_lora_model()` in `filters/common/model_loading.py`
- [x] **Code quality sweep** (#12-#19) - Resolved 8 issues: removed dead code, cleaned stale comments, fixed inconsistencies (-314 lines)

## Energy-Efficient Inference (#24)

- [x] **PyTorch dynamic quantization experiment** - 2026-03-07
  - Tested FP32/FP16/INT8 on uplifting v6, CPU-only
  - INT8: 2.6x faster, 3.3x smaller, but MAE +0.63 (unusable)
  - FP16: NaN on CPU (no native fp16 ALUs)
  - **Verdict:** Naive quantization rejected
  - See `docs/experiments/quantization-benchmark-2026-03-07.md`
- [ ] **ONNX Runtime INT8** - Calibrated quantization with representative data
- [ ] **Smaller base model retraining** - SmolLM-360M or similar sub-1B models
- [ ] **llama.cpp / GGUF** - Purpose-built CPU inference engine

## Deployment

- [ ] **Inference server** - Unified prefilter + model + postfilter pipeline
- [ ] **Batch processing** - High-volume article scoring
- [ ] **Production monitoring** - Latency, accuracy drift detection

## Infrastructure

- [x] **Prefilter evaluation framework** - Complete for sustainability_technology
- [ ] **Generalize prefilter evaluation** - Apply to all filters
- [ ] **Dataset QA pipeline** - Automated quality checks
- [ ] **Cost tracking** - Monitor API usage for oracle scoring
- [x] **Hub scorers: add torch_dtype parameter** - All 6 `inference_hub.py` files now accept optional `torch_dtype` param and pass it to `from_pretrained()`. Use `torch_dtype=torch.float16` on hardware without bfloat16 support.
- [x] **Deploy all filters to NexusMind** (#7) - All 6 filters deployed to gpu-server + sadalsuud + HuggingFace Hub
- [x] **Auto-compute score_scale_factor** (#22/#26) - Calibration script writes `score_scale_factor` to config.yaml; backfilled to all 6 filters
- [x] **Harmonize filters: llm-distillery as single source of truth** - Fixed drift between llm-distillery and NexusMind
  - base_prefilter.py: threading.Lock() for commerce detector (was bool flag)
  - investment-risk v5: merged source-based + content-pattern approaches, removed academic source blocking
  - Deployed all production prefilters to NexusMind (sadalsuud + gpu-server)
  - Verified 0 diff between all three locations
- [x] **Manifest-aware deploy script (#50)** - 2026-04-28. `.nexusmind-owns` at repo root + `--dry-run` + `--force-skip-owned-drift` in both `.sh` and `.ps1`. Lists `filter_base_scorer.py` and `hybrid_scorer.py` (NexusMind-owned). Deploy now exits non-zero on drift between distillery and NexusMind copies.
- [ ] **Harmonize prefilter structure across all 7 production filters (#52)** - Filed 2026-04-28. Survey shows 5 different override mechanisms, 3 with class/version drift between class name and dir, mixed flat-list vs dict containers. ~12-16h work; per-filter migration in priority order.
  - [x] **ADR-018** (2026-04-28) - Declarative shape decision documented; backwards-compatible BasePreFilter extension chosen
  - [x] **BasePreFilter extension** (2026-04-28) - EXCLUSION_PATTERNS / OVERRIDE_KEYWORDS / POSITIVE_PATTERNS / POSITIVE_THRESHOLD class attrs + default apply_filter() pipeline + _is_excluded / _has_override / _filter_specific_final_check helpers. All 7 production prefilters import + run unchanged (verified)
  - [x] **sustainability_technology v3 migrated** (2026-04-28) - 6/6 self-tests pass; behavior preserved
  - [x] **belonging v1 migrated** (2026-04-29) - 19/19 self-tests pass; behavior preserved. Data shape (EXCLUSION_PATTERNS dict, base-compiled patterns) harmonized; apply_filter stays custom because per-category positive-count thresholds + URL-based domain exclusions + obituary floor rule don't fit the base pipeline (ADR-018 explicitly permits this).
  - [x] **cultural-discovery v4 migrated** (2026-04-29) - 10/10 self-tests pass; behavior preserved. Data shape harmonized: EXCLUSION_PATTERNS dict + parallel EXCEPTION_PATTERNS_PER_CATEGORY dict (per-category exceptions don't fit base's single OVERRIDE_KEYWORDS slot). CULTURAL_DISCOVERY_BOOST_PATTERNS renamed to POSITIVE_PATTERNS so base compiles them. classify_content_type() preserved. Surfaced regression vs v3: v4's apply_filter doesn't call check_content_length (preserved as-is in this commit; tracked separately under Prefilter Quality below).
  - [x] **uplifting v7 migrated** (2026-04-29) - 12/12 self-tests pass; behavior preserved. Same EXCLUSION_PATTERNS + EXCEPTION_PATTERNS_PER_CATEGORY pattern as CD v4 for the 3 pattern-with-exception categories (corporate_finance, military_security, crime_violence); 4th category (pure_speculation) is count-based (speculation_count >= 3 AND outcome_count == 0) and stays as separate class attrs with an inline check after the dict iteration. classify_content_type preserved. ThrivingPreFilterV1 (which subclasses UpliftingPreFilterV7) verified working. Surfaced bug: Dutch `munitie` and similar multilingual patterns lack `\b` boundaries — fire on English substrings like "co-MMUNITIE-s" (preserved as-is; tracked under Prefilter Quality).
  - [x] **investment-risk v6 migrated + class drift fix** (2026-04-29) - 11/11 self-tests pass; behavior preserved. v6 now has its own InvestmentRiskPreFilterV6 class (was a re-export of V5). Backward-compat aliases (InvestmentRiskPreFilterV5 = V6, InvestmentRiskPreFilter = V6) + legacy prefilter()/get_stats() functions kept so existing imports don't break. base_scorer.py updated to reference V6 directly. Data-shape harmonization only — apply_filter stays custom because the source-based flow + matched-pattern reason strings + title-only clickbait don't fit the base pipeline.
  - [ ] **nature_recovery v2 migration** - inline list in method form, no override (simplest); class-name drift fix (V1→V2) at same time
  - [ ] **foresight v1 migration** - count-then-pattern check (POSITIVE_THRESHOLD = 3)
  - [ ] **Class-name drift cleanup batch** - sustech V2→V3, nature_recovery V1→V2 still pending. (investment-risk v6 own class — DONE 2026-04-29 as part of its #52 migration.) Deferred until remaining migrations done to avoid cross-repo coordination noise (NexusMind tests/unit/test_prefilter.py imports the V2 name).

## Prefilter Quality (Apr 2026)

- [x] **belonging v1 obituary leak (#45)** - 2026-04-28. 5 bypass classes patched (dies-with-verb, procession, vigil, RIP/rest in peace, killed-in-year), `dies at \d` → `\d+` bug fix, override floor on obit branch. Plus `(?-i:\bRIP\b)` follow-up after the case-insensitive false positive on "rip current".
- [x] **sustainability_technology v3 clickbait leak (#46)** - 2026-04-28. CLICKBAIT category added with 6 patterns (you-won't-believe, without-knowing, this-common, you're-probably, X-things-you-didn't, shocking-fact). Pattern 5 bounded `.{0,120}` after review caught cross-sentence FP risk.
- [ ] **cultural-discovery v4 missing content_length check** - Surfaced during #52 migration (2026-04-29). v4's `apply_filter` skips the `check_content_length` call that v3 had — short articles bypass the 300-char minimum and go straight to pattern matching. Likely unintentional regression when v4 was created. Low priority (oracle handles short articles fine; just slightly wasteful), but worth a one-line fix at the next CD version bump.
- [ ] **uplifting v7 multilingual `\b` boundary leak** - Surfaced during #52 migration (2026-04-29). Several Dutch/German/French patterns in v7's military_security / corporate_finance / crime_violence lists are written without `\b` word boundaries (e.g. `(wapensysteem|...|munitie|...)` for Dutch military). Result: Dutch `munitie` matches inside the English word "communities" (co-MMUNITIE-s), turning any English article mentioning communities into a military FP. Same bug shape as the RIP/rip-current case (#45). Fix: add `\b` boundaries to the multilingual alternations. Audit all 3 multilingual exclusion lists at next uplifting version bump (or do a focused regex-audit pass across all filters — this is likely not v7-only). Tracked alongside ADR-018 #52 work.
- [ ] **Universal obituary detector (#51)** - Filed 2026-04-28, design simplified 2026-04-28. Trained SLM at `filters/common/obituary_detector/v1/` (mirrors `commerce_prefilter` shape). **Universal block with tunable threshold** — accept ~1-3% recall cost on cultural-discovery / investment-risk / breakthroughs to clean ~14% noise on belonging + uplifting. Per-filter consumption deferred unless measurement proves it necessary. Extends ADR-004 (no supersede). ~2-3 weeks calendar, ~1.5 weeks engineer time (labeling is bottleneck).

## Cross-Filter Normalization (ADR-014)

- [x] **uplifting v6 normalization** - Fitted on production CDF
- [x] **belonging v1 normalization** - Fitted on production CDF
- [x] **cultural-discovery v4 normalization** - Fitted on production CDF
- [x] **sustainability_technology v3 normalization** - Fitted on production CDF
- [x] **uplifting v7 normalization** - Fitted on 73,986 production articles (2026-04-06)
- [x] **foresight v1 normalization** - Fitted on 623 articles (thin LUT, improves as data accumulates)
- [x] **nature_recovery v1 normalization** - Refitted on 76,500 articles (still clamped — extreme needle filter, #32)
- [x] **nature_recovery v2 normalization** - Fitted on 1,397 v2 production articles (filter_version=2.0, weighted_average >= 1.5), deployed to sadalsuud + gpu-server (2026-04-28). Patched `fit_normalization.py` with `--filter-version` to exclude v1 leftovers (19,948 articles correctly skipped). Curve: raw range 1.50–7.08, p95=4.49.
  - [ ] **Follow-up (~2026-04-30)**: sample fresh `filtered_*.jsonl` on sadalsuud and confirm new v2 articles show `normalization_method != null` and `raw_weighted_average != null`. Proves the deployed curve is being applied at runtime. Without this verification we'd be repeating the #44 pattern (claiming deploy without proof).

## Documentation

- [ ] **Update filters/README.md** - Current status is outdated (Nov 2025)
- [ ] **Training guide** - Step-by-step for new filters
- [ ] **Deployment guide** - Production setup instructions

---

*Last updated: 2026-04-29*

## #52 belonging v1 migration notes (2026-04-29)

Belonging is the second prefilter migrated to ADR-018 declarative shape.
Diverged from sustech v3's "fully declarative" template in two ways:

1. **Data shape only.** Exclusion patterns moved into `EXCLUSION_PATTERNS`
   dict (compiled once by base `__init__`); per-category counts dropped from
   `get_statistics()` and rebuilt from the dict. Iteration order preserved.
2. **Custom apply_filter retained.** Belonging uses per-category
   positive-signal thresholds (3/3/3/2/3/2/special), not BasePreFilter's
   binary `OVERRIDE_KEYWORDS` bypass. Plus URL-based domain exclusions and
   the obit `pos>=1`-floor-when-exception-present rule. None of that fits
   the standard `apply_filter()` pipeline; ADR-018 explicitly allows
   "custom form" for this. The harmonization is at the *data* layer; the
   *control* layer stays specialized.

`POSITIVE_PATTERNS` class attr was kept (shadows `BasePreFilter.POSITIVE_PATTERNS`)
so base compiles it into `_compiled_positives`. `POSITIVE_THRESHOLD` stays at
0, so base's `_has_override` never reads it — belonging consumes the
compiled list directly via `count_pattern_matches`. Documented at the class
attr.

Pattern preservation verified by counts (9/7/9/9/7/6/11/6 exclusion
categories; 10 exceptions; 12 positives; 9 multilingual positives — all
identical to baseline) and 19/19 self-test pass.

No downstream consumers reference the renamed private attrs (verified via
grep across the repo); only the public class symbol + `apply_filter()`
contract are used by `base_scorer.py` and `verify_belonging_v1.py`.

## #52 cultural-discovery v4 migration notes (2026-04-29)

CD v4 is the third migrated prefilter. Same partial-declarative shape as
belonging — exclusion data harmonized, custom `apply_filter` retained.
But the divergence from base differs:

1. **Per-category exception lists.** Each exclusion category
   (appropriation_debate, political_conflict, tourism_fluff, celebrity_art)
   has its own escape-hatch list — celebrity_art has philanthropy /
   repatriation exceptions, political_conflict has reconciliation / peace
   exceptions, etc. BasePreFilter's single `OVERRIDE_KEYWORDS` slot is
   global; CD's exceptions are category-scoped. Modeled with a parallel
   `EXCEPTION_PATTERNS_PER_CATEGORY` dict keyed by exclusion-category name,
   compiled in `__init__` into `_compiled_exceptions_per_category`.

2. **classify_content_type method preserved.** Distinct from apply_filter
   — used (currently only by self-tests, but kept for API stability) to
   tag articles as `cultural_discovery` (>=2 positive boost matches) or
   one of the four exclusion categories or `general`. Rewritten on the
   new dict-based structure.

3. **CULTURAL_DISCOVERY_BOOST_PATTERNS → POSITIVE_PATTERNS.** Same trick
   as belonging: rename so base's `__init__` compiles them into
   `_compiled_positives`. POSITIVE_THRESHOLD stays at 0, so base's
   `_has_override` never reads them — only `classify_content_type` does.

4. **Surfaced bug: missing content-length check.** v3's `apply_filter`
   called `check_content_length` first; v4's does not. Looks like an
   unintentional regression when v4 was created. **Preserved as-is in
   this migration commit** (scope: zero behavior change). Tracked above
   under "Prefilter Quality" as a separate one-line fix at next CD bump.

Behavior preservation verified by 10/10 self-test pass plus identical
pattern counts (11/14, 17/12, 15/14, 15/14 across the four categories;
12 positives; 8/4/6 domain counts).

No downstream consumers (verified via grep): only `base_scorer.py`
references `CulturalDiscoveryPreFilterV4` as a class symbol +
`apply_filter()` call. Older CD versions (v1/v2/v3) keep their old
attr names internally — no cross-version import.

Next: uplifting v7 (flat-list-per-category, pattern-pair override — no count).

## #52 uplifting v7 migration notes (2026-04-29)

Uplifting v7 is the fourth migrated prefilter. Same shape as CD v4 for 3 of
4 categories, with one extra wrinkle: a count-based block.

1. **Three pattern-with-exception categories.** corporate_finance,
   military_security, crime_violence — all use the
   `EXCLUSION_PATTERNS` + `EXCEPTION_PATTERNS_PER_CATEGORY` pair, identical
   to CD v4's structure.

2. **One count-based block (pure_speculation).** Doesn't fit the
   pattern-with-exception shape. Outcome-evidence patterns are a parallel
   *count* check, not a per-pattern exception. Kept as separate
   `SPECULATION_PATTERNS` / `OUTCOME_EVIDENCE_PATTERNS` class attrs;
   inline check after the exclusion-dict iteration:
   `speculation_count >= 3 AND outcome_count == 0`.

3. **classify_content_type preserved.** Has a custom first-check ordering:
   "peace_process" wins when both military_security pattern AND its
   exception fire (e.g. military buildup article that's actually a peace
   accord). Standard category iteration follows. Speculation classification
   uses a looser threshold (>=2 / <=1) than apply_filter (>=3 / 0).

4. **Subclass ThrivingPreFilterV1 verified.** `filters/thriving/v1/prefilter.py`
   inherits from UpliftingPreFilterV7 with only a VERSION override. Public
   API preserved, so the subclass still works post-migration (verified with
   a smoke test exercising all 4 categories).

5. **Surfaced bug: multilingual `\b` boundary leak.** Dutch `munitie`
   (without `\b`) matches inside English "communities". Pre-existing v7
   FP — preserved here, tracked separately under Prefilter Quality.
   Same bug shape as the RIP/rip-current case (#45). Audit all 3
   multilingual exclusion lists at next uplifting version bump.

Behavior preservation verified by 12/12 self-test pass plus identical
pattern counts (21/11, 19/18, 37/25 across the three pattern-with-exception
categories; 7 speculation; 6 outcome-evidence; 8/4/6 domain counts).

No additional downstream consumers (verified via grep): only
`base_scorer.py` references `UpliftingPreFilterV7` directly, plus
`thriving/v1/prefilter.py` via inheritance — neither reaches into private
attrs.

Next: investment-risk v6 (re-exports v5; needs own class — class-name drift
fix is part of the migration).

## #52 investment-risk v6 migration notes (2026-04-29)

Investment-risk is the fifth migrated prefilter and the most structurally
divergent so far. Two things landed in this commit:

1. **Drift fix** — v6 was a thin re-export of v5 (importlib trick because
   the hyphen in `investment-risk` blocks normal imports). v6 now has its
   own `InvestmentRiskPreFilterV6` class. Backward-compat aliases
   (`InvestmentRiskPreFilterV5 = V6`, `InvestmentRiskPreFilter = V6`) plus
   legacy `prefilter()` / `get_stats()` functions preserved so existing
   imports keep working — including v6/base_scorer.py's import via
   importlib (now updated to call `InvestmentRiskPreFilterV6` directly).

2. **Migration to declarative shape** — but only data-shape harmonization;
   apply_filter stays custom for three reasons:
     - **Source-based filtering** runs against `source` / `source_type` /
       `id` fields, not URL or text. Has its own early-return flow:
       allowed-source -> pass, investment-keyword -> pass, blocked-source
       -> block, all before content patterns.
     - **Reasons include matched-pattern info** —
       `allowed_source:reuters`, `investment_keyword:recession`,
       `blocked_source:github`. The base pipeline's `excluded_<category>`
       shape would lose this signal.
     - **Clickbait operates on title only**, not combined text. Stays as
       a separate class attr with its own check below the EXCLUSION_PATTERNS
       iteration.

Three text-pattern categories did get the dict treatment:
fomo_speculation (8 patterns, no exceptions), stock_picking (6 patterns,
12 macro-context exceptions), affiliate_conflict (4 patterns, no
exceptions). The macro_context list is the only per-category exception
this filter has — modeled as `EXCEPTION_PATTERNS_PER_CATEGORY['stock_picking']`.

`(True, "default_allow")` and `(True, "passed")` are intentionally
distinct — investment-risk reports the *reason* an article passed, not
just the fact that it did. Default-allow means "no source/keyword/pattern
fired, falling through to the philosophy: when in doubt, score it."

Behavior preservation verified by 11/11 self-test pass plus identical
pattern counts (19 blocked sources, 25 allowed, 30 keywords; 8/0, 6/12,
4/0 across pattern-with-optional-exception categories; 5 clickbait).

Next: nature_recovery v2 (inline list in method form — simplest of the
remaining; class-name drift fix V1→V2 deferred to the cleanup batch).

