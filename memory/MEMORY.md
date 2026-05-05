# LLM Distillery — Memory Index

Loaded every session. Topic files loaded on demand via triggers below.

## Topic Files

| File | When to load | Key insight |
|------|-------------|-------------|
| `gemma3-model.md` | Model loading, PEFT, or Hub upload issues | Auto mapping fix, OLD vs NEW key format, torch float16 |
| `gpu-server.md` | Training on GPU or deploying to gpu-server | HF_HUB_OFFLINE, scp not rsync, venv path, PYTHONPATH |
| `filter-status.md` | Checking filter versions, MAE, or hybrid probe stats | Per-filter deployment status and in-dev blockers |
| `gotcha-log.md` | Stuck on infra, tooling, or something weird | Problem → Root cause → Fix archive |
| `thriving-v1-scoring.md` | Understanding thriving v1 attempts | PAUSED — bimodal distribution, MAE 0.94, findings and open questions |
| `uplifting-v7-training.md` | Understanding thriving v1 history | v7 prompt evolution → thriving v1 rename (ADR-012) |

## Universal Gotchas

- **Gemma-3 Auto mapping**: `AutoModelForSequenceClassification` doesn't support `gemma3_text`. Always use `load_base_model_for_seq_cls()`. See `gemma3-model.md`.
- **PEFT adapter format**: Keep OLD format for Hub. Never run `resave_adapter.py`. See `gemma3-model.md`.
- **PYTHONPATH**: Always set `PYTHONPATH=.` when running scripts that import `filters.*`.
- **scp not rsync**: rsync fails with dup() errors on gpu-server.
- **Windows safetensors**: Can't write to memory-mapped file. Save to temp, then `os.replace()`.
- **Training data dir naming**: Hyphens preserved: `cultural-discovery_v3`. Check actual dir names before scripting.
- **Config format variation**: `tiers:` (uplifting, cultural-discovery, investment-risk v6) vs `tier_thresholds:` (sustainability_tech v5).
- **Hyphenated filter imports**: Use `importlib.import_module()` — Python can't import hyphens.
- **Git Bash path mangling**: Set `MSYS_NO_PATHCONV=1` before any command that passes Unix paths as arguments.

## Key File Paths

| Path | Purpose |
|------|---------|
| `filters/common/model_loading.py` | Gemma-3 model loading + LoRA utilities |
| `filters/common/filter_base_scorer.py` | Base class for all filter scorers |
| `filters/common/score_calibration.py` | Isotonic calibration fit/apply |
| `filters/common/embedding_stage.py` | e5-small probe for hybrid inference |
| `filters/common/hybrid_scorer.py` | Two-stage hybrid inference orchestrator |
| `filters/common/base_prefilter.py` | Base prefilter with commerce detection |
| `ground_truth/batch_scorer.py` | Oracle scoring pipeline |
| `training/train.py` | Model training pipeline |
| `training/prepare_data.py` | Training data preparation |
| `scripts/calibration/fit_calibration.py` | Isotonic calibration fitting |
| `scripts/deployment/upload_to_huggingface.py` | Hub upload + verification |
| `ground_truth/__init__.py` | `analysis_field_name()` — shared convention for scored JSONL keys |
| `scripts/screening/embedding_screener.py` | ADR-011: embedding similarity screener for needle filters |
| `scripts/oracle/average_oracle_runs.py` | Multi-run oracle score averaging |
| `filters/common/score_normalization.py` | Cross-filter percentile normalization (ADR-014) |
| `scripts/normalization/fit_normalization.py` | Fit normalization CDF from production data |
| `docs/adr/README.md` | ADR index (001-019) |
| `filters/common/obit_signal.py` | Hoisted regex obit probe — used by belonging v1 prefilter and NexusMind#199 cross-lens leak measurement |

## Cross-Project: NexusMind

- NexusMind's `src/filters/filter_loader.py` auto-discovers from `filters/` directory
- NexusMind's output includes `nexus_mind_attributes` — downstream apps (ovr.news, dashboard, Aegis) depend on field names. Don't rename fields.
- gpu-server runs NexusMind scorer at `~/NexusMind/filters/`. Deploy new filters with `scp`, restart service.
- NexusMind pipeline runs every 4h on sadalsuud. Filter changes should be deployed to both gpu-server and sadalsuud.

## Deployment Targets

- **gpu-server**: `~/NexusMind/filters/` — uses `investment_risk` (underscore, not hyphen)
- **sadalsuud**: `~/local_dev/NexusMind/filters/` — Hub inference (no local model/ needed), venv at `~/local_dev/NexusMind/venv/`
- Deploy config changes to both servers after calibration or config updates

## Experiments

- **Quantization (#24)**: Naive PyTorch INT8 rejected — 2.6x faster but MAE +0.63. FP16 produces NaN on CPU. Next: ONNX Runtime or smaller base models. See `docs/experiments/quantization-benchmark-2026-03-07.md`.
- Benchmark script: `scripts/experiments/quantization_benchmark.py` (reusable for any filter)

## Recently Promoted

<!-- Gotchas promoted to topic files or the project file.
     Format: "if [situation], then [what to do] — promoted from gotcha-log YYYY-MM-DD"
     Retire entries once they appear in their destination. -->

- if [landing a non-trivial migration or refactor], then [fire code-reviewer + refactoring-guide + security-auditor in parallel before considering it shipped — they have non-overlapping blind spots] — promoted from gotcha-log 2026-04-29
- if [a regex correctness bug is found], then [audit siblings in the same file/author-style — same-shape bugs cluster] — promoted from gotcha-log 2026-04-29 (recurrence of #45 RIP issue → today's multilingual sweep)

## Active Decisions

<!-- One-liners about recent architectural choices, pointing to ADRs.
     If a decision lives here for more than one session without a formal ADR, create one. -->

- English-only lens/tab names — ADR-013 (2026-03-28)
- Lens-aligned filter naming at version bumps — ADR-012
- Cross-filter percentile normalization, supersedes score_scale_factor — ADR-014 (2026-03-30)
- Thriving v1 paused, bimodal distribution problem — uplifting v6 stays (2026-03-30)
- Declarative prefilter shape via BasePreFilter extension — ADR-018 (2026-04-28). Per-filter migration COMPLETE 2026-04-29 (#52, all 7 production filters); review-battery follow-ups also landed (RIP guard repair, POSITIVE_PATTERNS shadow rename, CD v4 truncation, uplifting v7 multilingual `\b` boundary sweep, investment-risk cleanups, CD v4 colonial tightening, `_check_domain_exclusions` hoist, `_pre_exclusion_check` hook). Class-name drift cleanup (sustech V2→V3, NR V1→V2) and per-category exception extension to `_is_excluded` (potential ADR-019) deferred — see `docs/TODO.md` "Post-#52 Review-Battery Followups".
- Cross-repo cleavage rule, post-2026-05-04 manifest-as-anti-pattern incident — production-runtime concerns live in NexusMind wrappers (composition over inheritance), shared math lives in `filters/common/`, `.nexusmind-owns` manifest is the escape hatch (empty by default; entries require tracked issue + deadline). See `memory/gotcha-log.md` "Manifest as Anti-Pattern" entry + closure note for the full lesson and the cross-repo coordination shape that worked.
- Per-category exclusion overrides via Template Method — ADR-019 (2026-05-05). `BasePreFilter` extended with `CATEGORY_OVERRIDES: Dict[str, CategoryOverrideCfg]` (TypedDict-typed) + `_compound_override_applies()` hook. Subclasses override the narrow hook; base owns the fallback chain (compound → dict → global `_has_override`). Unblocks belonging/foresight/sustech/cultural-discovery from custom `apply_filter()`. Per-filter migrations under #52; belonging v1 is the recommended first migration target. <!-- verify: grep -q 'CATEGORY_OVERRIDES' filters/common/base_prefilter.py && grep -q '_compound_override_applies' filters/common/base_prefilter.py && echo PASS || echo FAIL -->
- Solutions broadening v4 DRAFT scaffolded — `filters/sustainability_technology/v4/` (2026-05-05). Forks signed off: C (broaden ST v3 in place), combine ST v3 + foresight v1 corpora, foresight retired when v4 supersedes ST v3. 7 dims, weight=1.00, calibration batch spec inline (300 articles, ~$0.30). Awaiting prompt drafting before any oracle spend. <!-- verify: test -f filters/sustainability_technology/v4/config.yaml && echo PASS || echo FAIL -->

## Last Session Recap (2026-05-05, second session)

Picked up the 6-item backlog from this morning's "Next Session Pickup". Closed 3 issues, deferred 1 with an upstream measurement issue, drafted 2 design artifacts, ran a multi-agent review battery, applied all major findings, committed and pushed.

- **#58** closed — investment-risk inference.py importlib pattern + existence guard (commit `d466b18`).
- **#59** closed — `LEGACY_DATASETS` skip-list in test_data_pipeline (commit `a2f3359`).
- **#47** closed — Option B chosen (NO_HUB sentinel + `verify_filter_package.py` coexistence guard + removed v7 inference_hub.py). Reason: gpu-server file-copy deploy skipped training_metadata/history JSON files needed for the model card; reconstruction risks fabricating MAE numbers. Commit `4f43ed6`.
- **ADR-019 shipped** — `BasePreFilter` extended with `CATEGORY_OVERRIDES` TypedDict + `_compound_override_applies()` Template Method hook. Commit `a1d73b5`. ADR `019-per-category-exclusion-overrides.md`.
- **#43 v4 DRAFT scaffolded** — `filters/sustainability_technology/v4/{config.yaml,README.md}` with 7 dims, weight=1.00, calibration batch spec. Commit `17e9079`. Refs #43.
- **Sustech regex correctness** — escaped dots in `state.of.the.art` patterns (v2 + v3). Commit `bf2e0ad`.
- **Obit regex hoisted** — `filters/common/obit_signal.py` extracted from belonging v1's `obituary_funeral` patterns. belonging refactored to import. NexusMind#199 will import the same module so there's one canonical pattern source. (separate post-push commit)
- **#51 deferred** — measurement-as-labeling-pipeline rationale: NexusMind#199 ships regex P(obit) signal in production scoring output, 2-week observation window, then escalate or close based on per-lens leak rate. NexusMind#199 already filed with +2-week reminder comment.

Multi-agent review battery (code-reviewer + refactoring-guide + security-auditor in parallel) caught: (a) `Dict[str, Any]` typo-vulnerability on `CATEGORY_OVERRIDES` → TypedDict fix, (b) Template Method super()-forgetting risk on `_category_override_applies` → `_compound_override_applies` Template Method split with base owning the fallback chain, (c) NO_HUB + inference_hub.py coexistence ambiguity → explicit FAIL guard. All applied before commit.

Open & non-urgent: NexusMind#199 implementation (cross-repo, when ready), v4 prompt drafting, ADR-019 per-filter migrations under #52, confirm `score_scale_factor` is inert before any v4 corpus re-score.

## Next Session Pickup (set 2026-05-05 EOD, second session)

Pick by appetite:

1. **NexusMind#199** — implement the regex P(obit) probe in production scoring output (NexusMind side; this side is ready — `filters/common/obit_signal.py` exposes `loose_obit_signal(article) -> int` and `has_obit_signal(article) -> bool`). +2-week reminder comment is on the issue; calendar that after deploy.
2. **v4 prompt drafting** — `filters/sustainability_technology/v4/prompt-compressed.md` and `prompt-full.md`. Encode the 7 dimensions' scales + critical filters from `config.yaml`. Run the 300-article calibration batch (~$0.30) once drafted. Decision criteria pre-defined in config.yaml.
3. **ADR-019 first migration: belonging v1** — drop custom `apply_filter()`, move per-category positive thresholds → `CATEGORY_OVERRIDES`, move obit compound rule → `_compound_override_applies` hook. 20 self-tests are the regression gate. Validates the Template Method shape on the most-tested filter.
4. **`score_scale_factor` inertness check** — grep current scoring code for `score_scale_factor` reads. Confirm key is parsed but not consumed (per ADR-014 supersession). If consumed, fix before v4 re-score.
5. Class-name drift cleanup, CD v4 + NR v2 missing `check_content_length` — still queued, still non-urgent.

## Next Up (from ROADMAP "Now")

- **foresight v1** — PARKED (#43, 2026-04-16). Captures governance solutions, not foresight; merging into broadened Solutions lens at sustainability_technology v4. <!-- verify: grep -qE "\*\*foresight\*\*.*PARKED" CLAUDE.md && echo PASS || echo FAIL -->
- **nature_recovery v2** — DEPLOYED to Hub 2026-04-19 after #44 fix (v2 package referenced v1 imports + repo_id before). <!-- verify: PYTHONPATH=. python scripts/deployment/verify_filter_package.py --filter filters/nature_recovery/v2 --check-hub > /dev/null && echo PASS || echo FAIL -->
- **nature_recovery v1 normalization** — FIXED (#32 closed 2026-04-09). Refit covers full score range (354 articles, x: 0.10–10.0). gpu-server scorer verified producing differentiated scores.
- **nature_recovery v2 normalization** — FITTED 2026-04-29 on 1,397 v2 production articles (filter_version=2.0, weighted_average≥1.5; raw range 1.50–7.08, p95=4.49). Patched `fit_normalization.py` with `--filter-version` to exclude v1 leftovers (#52 follow-up). Deployed to sadalsuud + gpu-server. <!-- verify: test -f filters/nature_recovery/v2/normalization.json && echo PASS || echo FAIL -->
- **prefilter shape harmonization** (#52) — COMPLETE 2026-04-29. All 7 production filters migrated to declarative shape; review-battery follow-ups also landed (8 commits). Remaining work: class-name drift cleanup batch (sustech V2→V3, NR V1→V2 — gated on NexusMind cross-repo coordination) and potential `_is_excluded` extension for per-category exceptions (deferred to ADR-019 design). See `docs/TODO.md` "Post-#52 Review-Battery Followups" for the full punch-list status. <!-- verify: grep -q "ADR-018" filters/common/base_prefilter.py && grep -q "DOMAIN_EXCLUSIONS" filters/common/base_prefilter.py && echo PASS || echo FAIL -->
- **gpu-server filter discovery dedup + canonical alignment** (2026-04-30, NexusMind 2d3c666 + manual weight migration) — #53 STRUCTURALLY FIXED, with bonus canonical alignment. (1) `FilterLoader.discover_filters()` collapses hyphen/underscore variant collisions to one registered entry (winner = most complete artifacts), alias map covers both API name forms. (2) After the structural fix shipped, weights were moved gpu-server-side from `investment_risk/v6/model/` to `investment-risk/v6/model/` so the registered winner is now `investment-risk` — matches llm-distillery's source-of-truth convention. Discovery flipped: `using 'investment-risk' (most complete artifacts), ignoring ['investment_risk']`; alias map: `{'investment_risk': 'investment-risk'}`. Both API name forms still resolve to the same scorer (verified — same wa returned for both). The earlier band-aid symlink loop (which caused two outages in 24h) is fully retired. <!-- verify: ssh gpu-server "curl -fs http://localhost:8000/health > /dev/null && curl -fs http://localhost:8000/models/status | python3 -c 'import sys,json; d=json.load(sys.stdin); names=set(d[\"available_filters\"]); assert len(names)==7 and \"investment-risk\" in names and \"investment_risk\" not in names, names; print(\"OK\")'" && echo PASS || echo FAIL -->
- **raw_weighted_average** — Now passed through gpu-server API → sadalsuud pipeline → filtered output (#36 closed 2026-04-09). Normalization fitting script prefers it to avoid double-normalization.
- **thriving v1** — PAUSED. Candidate for two-stage scoring fix. See `memory/thriving-v1-scoring.md`.
- **#24** — ONNX Runtime INT8 or smaller base model retraining
