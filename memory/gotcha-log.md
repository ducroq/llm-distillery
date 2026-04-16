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

## SCP Creates Nested Directories When Target Exists (Mar 2026, recurred Apr 2026)

**Problem**: `scp -r source/dir/ dest/dir/` creates `dir/dir/` nesting. Hit three times: filter directory, model directory, and nature_recovery v2 model copy from gpu-server.

**Root cause**: When the target directory already exists, `scp -r source/ target/` copies `source` INTO `target` rather than merging contents.

**Fix**: Always scp to the PARENT directory: `scp -r source/dir/ dest/` (not `dest/dir/`). RUNBOOK.md updated 2026-04-15 with correct patterns. Promoted to feedback memory.

---

## Git Bash Mangles Unix Paths in Arguments (Mar 2026, recurred Apr 2026)

**Problem**: `--remote-dir /home/jeroen/...` becomes `C:/Program Files/Git/home/jeroen/...` when passed through Python on Windows Git Bash.

**Root cause**: Git Bash's POSIX-to-Windows path conversion applies to command arguments that look like Unix paths.

**Fix**: Set `MSYS_NO_PATHCONV=1` before the command: `MSYS_NO_PATHCONV=1 PYTHONPATH=. python ...`

---

## Systemd Service Context Differs From Interactive SSH (Apr 2026)

**Problem**: Filter works when tested interactively on gpu-server (`ssh gpu-server "python3 ..."`) but fails when the NexusMind scorer systemd service restarts.

**Root cause**: The systemd service runs with a different environment than an interactive SSH session. Key differences: working directory, PYTHONPATH, HF_HUB_OFFLINE, PATH, and available GPU memory (other services may claim VRAM). Interactive testing bypasses these constraints, so "it works when I run it" doesn't guarantee it works in production.

**Fix**: Always test through the actual execution context after deploying changes: `sudo systemctl restart nexusmind-scorer && journalctl -u nexusmind-scorer -f`. Check the service's EnvironmentFile and WorkingDirectory in the unit file, not just interactive shell behavior.

---

## MAE Is Misleading for Needle-in-Haystack Filters (Apr 2026)

**Problem**: nature_recovery v1 had val MAE 0.54 — looks great. But in production, 98.6% of articles scored below 1.0. The model had zero discrimination. v2 has "worse" MAE (0.63) but dramatically better ranking (Recall@20: 0.70 vs 0.55).

**Root cause**: MAE treats all errors equally. When 95% of articles are noise with oracle WA ~0, predicting zero for everything gives low MAE. The model is "accurate" on noise but useless on the articles that matter.

**Fix**: For needle filters, use ranking metrics: Recall@k, NDCG@k, false negative rate on MEDIUM+. Documented in filter development guide (Issue 4). Overall MAE is still fine for balanced filters (uplifting, belonging, etc.).

---

## [RESOLVED] train.py --output-dir Creates Nested model/model/ (Apr 2026)

**Problem**: `--output-dir filters/foresight/v1/model` saves adapter to `model/model/`. Then `--resume-from filters/foresight/v1/model/model` looks for `model/model/model/`.

**Root cause**: `train.py` appends `/model` to the output dir for the adapter save path. Both `--output-dir` and `--resume-from` do this, so the nesting doubles each time.

**Fix**: train.py now strips trailing `model` from both `--output-dir` and `--resume-from` before appending. Either path form works now.
