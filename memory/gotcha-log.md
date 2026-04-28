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

## Memory Claimed "Shipped" But Feature Only Existed in Running Process (Apr 2026)

**Problem**: Agent memory can state a feature is "shipped and working" based on a point-in-time test during a session. If the feature lives only in a running process (not persisted to the deployed codebase), it disappears on restart. Future sessions that trust the memory never re-verify.

**Root cause**: Memory records a session observation as deployed state. There's no mechanism to distinguish "I tested this once" from "this is persistently deployed."

**Fix** (v1.9.0 self-verifying memory): Never write "shipped"/"deployed"/"live" in memory based on a session observation alone. Qualify: *"responded correctly during session — verify persistence after restart."* Include a verification command in an HTML comment so future sessions can check before trusting: `<!-- verify: curl https://endpoint | grep expected -->`. The `/curate` skill now scans for unverified state claims and runs verify commands automatically.

---

## Commit Claimed "Deploy to Hub" But Upload Never Ran (#44, 2026-04-19)

**Problem**: Commit `399d739` "Deploy nature_recovery v2 with sample weighting (#41)" states in its body *"Deployed to HuggingFace Hub, gpu-server, sadalsuud."* The Hub upload was never actually executed. For three days production ran v2 config + v2 calibration × v1 weights (pulled from `jeergrvgreg/nature-recovery-filter-v1` by an `inference_hub.py` that had been scaffolded as a copy of v1). Caused NexusMind#155 / #161 scoring anomalies.

**Root cause**: Two failures compounded.
  1. *Scaffold-by-copy without translation*: all three v2 inference files (`inference.py`, `inference_hub.py`, `inference_hybrid.py`) were copies of their v1 equivalents with `v1` imports and `v1` repo_id left intact.
  2. *No gate between commit-message intent and actual upload*: the agent wrote "Deployed to Hub" based on intent, not verification. The upload script's post-upload `PeftModel.from_pretrained()` verification never ran because the script wasn't invoked.

**Fix** (2026-04-19):
  - `scripts/deployment/verify_filter_package.py` — 8 checks (imports match dir version, `repo_id` matches dir version, config/FILTER_VERSION consistency, Hub repo exists, Hub `last_modified` ≥ local `adapter_model.safetensors` mtime).
  - `scripts/deploy_to_nexusmind.{sh,ps1}` Step 0 runs `verify_filter_package.py --check-hub`; deploy aborts on failure.
  - `.githooks/commit-msg` refuses any commit whose message contains *deploy/shipped/uploaded* if the staged diff touches filters and verification fails.
  - See follow-up issues #47, #48, #49.

[PROMOTED to feedback memory: `feedback-claim-requires-verify.md`]

---

## [RESOLVED] deploy_to_nexusmind.sh Regressed BFloat16 Fix Owned by NexusMind (2026-04-19)

**Problem**: `deploy_to_nexusmind.sh` copies `filters/common/` from llm-distillery to the NexusMind checkout. llm-distillery's `filter_base_scorer.py` lacked a BFloat16 → float32 cast (`outputs.logits.float().cpu().numpy()`) that NexusMind had added in `68e3d5d` (2026-04-16). Running the deploy script for nature_recovery v2 today silently overwrote the fixed NexusMind copy with the broken llm-distillery copy. Production `/filter/nature_recovery/score` started returning 500s with `TypeError: Got unsupported ScalarType BFloat16`.

**Root cause**: `filters/common/filter_base_scorer.py` exists in both repos, but NexusMind had been evolved without back-porting fixes to llm-distillery. The deploy script blindly copies the entire `filters/common/` tree with no "NexusMind-owns" carve-out. NexusMind's own gotcha-log actually notes this pattern ("filter_base_scorer.py can't be synced from distillery"), but the rule was docs-only — no script enforcement.

**Fix** (immediate): Today I ported the `.float()` cast to llm-distillery (`b98fc6f`) so `filters/common/` is consistent both sides, and restored it on NexusMind (`2d9a11f`). Production verified via smoke test (nature_recovery wa=4.31, belonging wa=6.48).

**Fix** (durable — shipped 2026-04-28, issue #50): Added `.nexusmind-owns` manifest at repo root and patched both `deploy_to_nexusmind.sh` and `.ps1` to skip listed files (currently `filter_base_scorer.py` + `hybrid_scorer.py`) and warn on drift. Initial run after the patch caught real comment-level drift on `filter_base_scorer.py` that would have been silently overwritten. CLAUDE.md Hard Constraints now references the manifest.

---

## rsync dup() Errors from Windows Git Bash (Recurred 2026-04-19, NexusMind)

**Problem**: `NexusMind/scripts/deploy_filters.sh` fails with `rsync: dup() in/out/err failed` / `connection unexpectedly closed (0 bytes received so far)` when run from Windows Git Bash targeting gpu-server — even though gpu-server is reachable via plain SSH.

**Root cause**: Windows Git Bash / MSYS runtime doesn't cleanly hand rsync's fd management to the Tailscale SSH subprocess. Specific to the workstation runtime, not gpu-server. This is an old gotcha (Feb 2026, originally fixed by switching to scp) that recurred when NexusMind switched the deploy script back to rsync (Apr 2026, to preserve model/ directories via `--exclude`).

**Fix**: Run `deploy_filters.sh` from a Linux host (sadalsuud) instead of Windows. `llm-distillery/scripts/remote_deploy.sh` wraps the SSH hop — single command from the workstation, Linux→Linux rsync inside. Structurally unreachable on Windows now.

---

## [RESOLVED] train.py --output-dir Creates Nested model/model/ (Apr 2026)

**Problem**: `--output-dir filters/foresight/v1/model` saves adapter to `model/model/`. Then `--resume-from filters/foresight/v1/model/model` looks for `model/model/model/`.

**Root cause**: `train.py` appends `/model` to the output dir for the adapter save path. Both `--output-dir` and `--resume-from` do this, so the nesting doubles each time.

**Fix**: train.py now strips trailing `model` from both `--output-dir` and `--resume-from` before appending. Either path form works now.
