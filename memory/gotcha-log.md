# Gotcha Log

Problems encountered and resolved. Format: Problem → Root cause → Fix.

---

## PEFT Adapter Resave Breaks Hub Loading (Feb 2026)

**Problem**: After running `resave_adapter.py`, `PeftModel.from_pretrained()` fails to load the adapter from HuggingFace Hub.

**Root cause**: `resave_adapter.py` converts keys from OLD format (`.lora_A.weight`, `score.weight`) to NEW format (`.lora_A.default.weight`, `score.modules_to_save.default.weight`). Hub loading via `PeftModel.from_pretrained()` expects OLD format and doesn't remap.

**Fix**: Never run `resave_adapter.py` before Hub upload. Keep adapters in OLD format. Local `inference.py` remaps at load time. Documented in ADR-007.

---

## Gemma-3 Auto Mapping Not Supporting gemma3_text (Feb 2026)

**Problem**: `AutoModelForSequenceClassification.from_pretrained("google/gemma-3-1b-pt")` fails because `gemma3_text` model type isn't in the Auto mapping (only `gemma3` for multimodal is mapped).

**Root cause**: `google/gemma-3-1b-pt` uses `Gemma3TextConfig` with `model_type: gemma3_text`, but transformers 4.55.3 doesn't register it in `AutoModelForSequenceClassification`.

**Fix**: Created `load_base_model_for_seq_cls()` in `filters/common/model_loading.py`. Falls back to building a custom `Gemma3TextForSequenceClassification` using `Gemma3TextModel` + `nn.Linear` head when Auto fails.

---

## Windows Safetensors Memory-Mapped Write Conflict (Feb 2026)

**Problem**: Saving a safetensors file on Windows fails if the same file is currently loaded (e.g., modifying adapter weights in place).

**Root cause**: Safetensors uses memory-mapped I/O. Windows locks memory-mapped files, preventing overwrite.

**Fix**: Save to a temp file first, then `os.replace()` to atomically swap.

---

## rsync dup() Errors on gpu-server (Feb 2026)

**Problem**: `rsync` fails with `dup()` errors when transferring files to gpu-server.

**Root cause**: Unknown — likely related to LXC container filesystem or Tailscale network layer.

**Fix**: Use `scp` instead of `rsync` for all file transfers to gpu-server.

---

## Training Data Dir Naming Mismatch (Feb 2026)

**Problem**: Training data directories don't follow a single naming convention, causing confusion when scripting.

**Root cause**: Organic growth. Some dirs use filter version from when data was scored (e.g., `sustainability_technology_v3`) vs the filter version being trained. Hyphenated filter names (investment-risk, cultural-discovery) keep hyphens in dir names.

**Fix**: Convention: `datasets/training/{filter_name}_{version}/` where `{filter_name}` preserves the filter's canonical name (including hyphens). Check actual dir names before scripting.

---

## Hyphenated Filter Names Break Python Imports (Feb 2026)

**Problem**: `import filters.investment-risk.v6.inference` fails — Python interprets hyphen as minus.

**Root cause**: Python identifiers can't contain hyphens.

**Fix**: Use `importlib.import_module("filters.investment-risk.v6.inference")` for hyphenated filter names.

---

## Pipeline is I/O-Bound, Not Compute-Bound (Mar 2026)

**Problem**: Instinct says "optimize model inference" (#24), but production logs show GPU scoring is only 12% of pipeline time.

**Root cause**: The NexusMind pipeline spends most time on pre-enrichment (HTTP-fetching full article text from source URLs) — 55% of wall time on big runs. GPU scoring does ~2K articles × 5 filters in under 4 minutes (~22ms/article). Story dedup (GPU embeddings) adds another 8%.

**Data** (2026-03-08, 1,949 articles × 5 filters):
- Pre-enrichment: ~16 min (55%)
- GPU scoring: ~3.6 min (12%)
- Story dedup: ~2.3 min (8%)
- Aegis export: ~3.3 min (11%)
- Cleanup/sync: ~4 min (14%)

**Implication**: On GPU, scoring is fast and not the bottleneck — pre-enrichment is. But GPU access is borrowed. Without it, scoring becomes the bottleneck: ~900ms/article on CPU × 1,949 articles × 5 filters ≈ 2.4 hours per run (vs 3.6 min on GPU). That's why #24 matters — it's not about optimizing today's pipeline, it's about surviving without the GPU.

---

## score_scale_factor Is Linear, Cross-Filter Normalization Is Not (Mar 2026)

**Problem**: Filters produce structurally different score distributions. Uplifting passes 62.8% of articles as MEDIUM+, nature_recovery passes 0.3%. The HOME tab uses `max(weighted_average)` across filters, so uplifting dominates. Articles open in the wrong tab (uplifting instead of recovery).

**Root cause**: `score_scale_factor` (e.g., 1.53 for nature_recovery) applies a linear stretch to compensate for calibration range compression. But the distributions are non-linear — most nature_recovery articles cluster near 0, and linear stretching doesn't help them. Meanwhile, calibration itself is fitted on enriched val sets (ADR-003/005), not production data, so the calibration ceiling reflects what the oracle saw in enriched data, not what's possible.

**Fix**: Replace `score_scale_factor` with percentile normalization (ADR-014). Non-linear monotonic mapping fitted from production MEDIUM+ data. Same pattern as isotonic calibration (ADR-008) but applied on the weighted average across filters, not per-dimension within a filter. Set `score_scale_factor` to 1.0 for all filters after deploying normalization.

---

## SCP Creates Nested Directories When Target Exists (Mar 2026)

**Problem**: `scp -r filters/thriving/v1/ gpu-server:~/llm-distillery/filters/thriving/v1/` creates `v1/v1/` nesting. Hit twice: filter directory and model directory.

**Root cause**: When the target directory already exists, `scp -r source/ target/` copies `source` INTO `target` rather than merging contents.

**Fix**: Either create parent only (`mkdir -p .../thriving`) and let scp create `v1/`, or scp individual files. Always verify with `ls` after copying.

---

## Git Bash Mangles Unix Paths in Arguments (Mar 2026)

**Problem**: `--remote-dir /home/jeroen/...` becomes `C:/Program Files/Git/home/jeroen/...` when passed through Python on Windows Git Bash.

**Root cause**: Git Bash's POSIX-to-Windows path conversion applies to command arguments that look like Unix paths.

**Fix**: Set `MSYS_NO_PATHCONV=1` before the command: `MSYS_NO_PATHCONV=1 PYTHONPATH=. python ...`
