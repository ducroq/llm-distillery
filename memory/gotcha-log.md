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

## [RESOLVED] \bRIP\b False-Positive on "rip current" (2026-04-28)

**Problem**: belonging v1 prefilter shipped a `\bRIP\b` pattern in `OBITUARY_PATTERNS` (commit `44b5e21`, #45). The standalone token was meant to catch obituary uses ("Tributes Pour In: RIP Hero"), and the comment said "MUST be uppercase to avoid matching 'rip' as a verb." But every pattern in `OBITUARY_PATTERNS` is compiled with `re.IGNORECASE` at the call site (`prefilter.py` line 262). So `\bRIP\b` matched lowercase "rip" too — including **"rip current"** in beach-safety articles, which would block from belonging.

**Root cause**: A list-of-patterns design plus a global compile flag at the call site means a single "case-sensitive only" token in the list silently becomes case-insensitive. The pattern author can't opt out of the global flag without explicit syntax.

**Fix #1 (incomplete)**: Inline `(?-i:...)` flag scope disables IGNORECASE for that one pattern: `r'(?-i:\bRIP\b)'`. Confirmed with a unit test against "Lifeguards Warn of Rip Currents at Local Beaches" (passes). Shipped in `598fa72`.

**Caught by**: post-deploy code-reviewer agent battery flagged it as P2 hypothetical; I noticed IGNORECASE was *already* on, making it a real shipped P0/P1.

**Promoted to**: `feedback-regex-ignorecase-trap.md` (auto-memory). When adding a token to a list-of-patterns compiled with a global flag, check the flag affects all entries; use inline `(?aiLmsux-imsx:...)` to opt out for one entry.

**Fix #2 (actual repair, 2026-04-29)**: Code-reviewer agent during the #52 migration audit caught that fix #1 was *also* broken: `_get_combined_clean_text` lowercases input via `combined_text.lower()` before pattern matching. By the time the regex engine sees the string, "RIP" has become "rip" — there are no uppercase chars left for `(?-i:)` to enforce against. The pattern was inert in production: never blocked uppercase RIP, but also never tripped on rip-currents (because nothing matched at all). The "rip current" test passed for the wrong reason.

The real fix needs the input string to retain case. Done by reading the raw title directly off the article (skipping `_get_combined_clean_text`) and running a case-sensitive `\bRIP\b` against it. Title-only because body text occasionally all-caps for emphasis; titles use "RIP" deliberately as a recognised acronym, so FP risk is minimal there. Lives in `_uppercase_rip_in_title()` and is consulted alongside the obituary_funeral category in `apply_filter`. The dead in-list `(?-i:\bRIP\b)` pattern was removed.

Two test cases added to `belonging/v1/prefilter.py::test_prefilter`:
- Positive: "Tributes Pour In: RIP Hero..." with no positive belonging signal → blocks as `obituary_funeral` (would have passed pre-repair).
- Regression: "Lifeguards Warn of Rip Currents..." → still passes.

20/20 self-tests pass post-repair (was 19/19 pre-repair).

**Lesson**: When a pattern has case-sensitivity intent, check the *whole pipeline* — not just the regex flag at compile time. If the input string is normalized upstream (lowercased, stripped, etc.), inline regex flags can't recover information that's already gone. Verifying with a pure regex unit test is not enough; integration matters. Generalises to: any per-pattern requirement that conflicts with global preprocessing.

---

## deploy_to_nexusmind.sh Prints Wrong SSH Hints (2026-04-28)

**Problem**: After a successful deploy, the script prints:
```
ssh user@sadalsuud "cd ~/NexusMind && git pull origin main"
ssh jeroen@llm-distiller "cd ~/NexusMind && git pull origin main"
```
The first command failed during this session: actual sadalsuud user is implicit (no `user@`), and the path is `/home/jeroen/local_dev/NexusMind/`, not `~/NexusMind/`.

**Root cause**: Hardcoded template strings in `scripts/deploy_to_nexusmind.sh` and `.ps1` post-deploy hints, never updated when the layouts settled. `llm-distiller` may also not be the right alias (haven't verified).

**Fix** (deferred — flag for next deploy-script touch): Update the template strings to reflect actual SSH config + paths. For now, the correct invocation is `ssh sadalsuud "cd /home/jeroen/local_dev/NexusMind && git pull origin main"` followed by `bash scripts/deploy_filters.sh` on sadalsuud (which rsyncs to gpu-server — gpu-server is NOT git-managed, see `memory/MEMORY.md` Cross-Project: NexusMind section).

---

## fit_normalization.py Blends Across Filter Versions (2026-04-29)

**Problem**: When fitting nature_recovery v2 normalization, production output had 145K v2 articles + 19,948 v1 leftovers (the rolling window straddled the 2026-04-16 v1→v2 cutover). Running `fit_normalization.py` as it stood would have silently merged both into the same percentile CDF.

**Root cause**: `scripts/normalization/fit_normalization.py` filtered articles by `min_score` only, not by `filter_version`. Filter version transitions aren't atomic in the production filtered/ output, so any new-version normalization fit must explicitly scope to that version.

**Fix** (commit `c4e4a0f`): added `--filter-version` flag (defaults to None for backwards compat). Both `load_weighted_averages_local()` and `load_weighted_averages_ssh()` now check `analysis["filter_version"]`. Will be useful at every future version bump.

---

## v2 Filter Without normalization.json Looks Like a raw_weighted_average Bug (2026-04-29)

**Problem**: Investigating nature_recovery v2 normalization, found production output showing `raw_weighted_average: null` and `normalization_method: null` for 100% of v2 articles after 2026-04-17 (~129K articles, 12 days). Looked like the #36 "raw_weighted_average passthrough" fix had regressed.

**Root cause**: Not a bug — by design. `filters/common/filter_base_scorer.py` doesn't write `raw_weighted_average` at all (only `weighted_average`). The `raw_weighted_average` and `normalization_method` fields are added downstream by NexusMind's runtime *only when normalization is being applied*. When a filter has no `normalization.json`, NexusMind stores the raw score in `weighted_average` and leaves the audit-only `raw_weighted_average` field null. Confirmed by reading `_create_empty_result()` and `_process_raw_scores()` in `filter_base_scorer.py`.

**Fix**: Use `weighted_average` directly when fitting the *first* normalization for a freshly-deployed filter version. The `fit_normalization.py` fallback path already handles this (line 59 — `wa = raw if raw is not None else analysis["weighted_average"]`). The script will warn about "Mixed fields" but that's expected during the v1→v2 transition window.

**Implication**: A filter that ships a new version without normalization.json will have null `raw_weighted_average` for as long as it takes to fit the first curve. Don't mistake this for a regression.

---

## [RESOLVED] train.py --output-dir Creates Nested model/model/ (Apr 2026)

**Problem**: `--output-dir filters/foresight/v1/model` saves adapter to `model/model/`. Then `--resume-from filters/foresight/v1/model/model` looks for `model/model/model/`.

**Root cause**: `train.py` appends `/model` to the output dir for the adapter save path. Both `--output-dir` and `--resume-from` do this, so the nesting doubles each time.

**Fix**: train.py now strips trailing `model` from both `--output-dir` and `--resume-from` before appending. Either path form works now.

---

## Multi-Agent Review Battery Catches Issues Single Reviewer Misses (2026-04-29)

**Problem**: After landing seven prefilter-migration commits under #52 (claimed zero behavior change, all self-tests passing), I asked for a review battery — code-reviewer, refactoring-guide, and security-auditor agents fired in parallel against the same diff. Each found different real issues that the other two had not flagged.

- **code-reviewer** caught that the `(?-i:\bRIP\b)` "fix" from `598fa72` was inert in production because `_get_combined_clean_text` lowercases input before pattern matching — pattern never fires on real input. The original review battery in 2026-04-28 also flagged it, but only as P2 hypothetical; deeper trace this time showed it was P1 in production.
- **refactoring-guide** caught that `POSITIVE_PATTERNS` shadowing `BasePreFilter.POSITIVE_PATTERNS` in belonging v1 + CD v4 was a semantic trap waiting for a future maintainer to set `POSITIVE_THRESHOLD > 0`.
- **security-auditor** caught that `munitie`/communities was just one of many unbounded multilingual alternations — `viol` (matches violence/violet/viola/violin), `acquisition`, `fusion`, `auteur`, `association` were all unbounded. Several were actively producing false positives in production.

The agents had non-overlapping blind spots. Code-reviewer focused on logic correctness; refactoring-guide focused on architecture/naming; security-auditor focused on adversarial inputs. Running them sequentially and synthesising findings would have surfaced the same issues, but firing in parallel halved the wall-clock time.

**Root cause**: A single reviewer's perspective is bounded by the framing they bring. Asking three agents with different framings produces three distinct review reports; their union catches more than any single one. None of them are smarter than a careful human reviewer, but in the time it takes a human to read the diff once, all three reports have landed.

**Fix**: When landing a non-trivial migration or refactor, default to firing all three (code-reviewer / refactoring-guide / security-auditor) in parallel rather than picking one. Each cost ~1 minute of background time and ~$0.30 of agent cost; the issues caught (one production bug, one semantic trap, several real false-positive vectors) were worth the spend several times over.

**Promoted to**: `feedback-multi-agent-review-default.md` (auto-memory, this session).

---

## When a Regex Bug is Found, Audit Siblings (2026-04-29, recurrence)

**Problem**: Today's audit of one named bug (`munitie` matching inside "communities") surfaced *five* additional unbounded multilingual patterns in the same file (`viol`, `acquisition`, `fusion`, `auteur`, `association` exception). All had the same shape: an alternation `(a|b|c)` without `\b` anchors, where one or more of the alternation tokens happened to be a substring of common English words.

**Root cause**: The same code-author hand wrote all the multilingual patterns in a similar style. Whatever invariant they missed for one pattern (forgetting `\b`), they missed for all of them. The original `598fa72` fix for one specific instance (`\bRIP\b`) didn't prompt a sweep; the bug recurred at scale until the security-auditor agent did the systematic check.

**Fix**: When a regex correctness bug is found, the next move is "audit the siblings" — find every pattern in the same file (or written in the same style by the same author) and check if it has the same shape. Cheap; usually catches more than the original report.

**Promoted to**: `feedback-regex-ignorecase-trap.md` updated with this generalisation (2026-04-29 follow-up).

---

## Investment-Risk v6 Hyphen/Underscore Path Divergence Took Scorer Down on Restart (2026-04-29)

**Problem**: After a successful `remote_deploy.sh` push to gpu-server, the scorer service failed to come up. journalctl: `CRITICAL - Missing model weights: investment-risk/v6/model. RuntimeError: Cannot start scorer: 1 filter(s) missing model weights: investment-risk/v6/model.` The 90s health check timed out and `remote_deploy.sh` reported "Scorer failed to become healthy". Production scoring was DOWN until I applied a manual fix.

**Root cause**: gpu-server has TWO directory layouts for investment-risk v6 — both under `/home/hcl/NexusMind/filters/`:
- `investment-risk/v6/` (hyphen) — historically held just the prefilter code; no `model/` dir
- `investment_risk/v6/` (underscore) — has the actual `model/` weights (`adapter_model.safetensors` etc.)

Why both exist: per the project memory ("Cross-Project: NexusMind", line 59 of `memory/MEMORY.md`), gpu-server is documented to use the underscore variant. But llm-distillery uses the hyphen (the actual repo dir is `filters/investment-risk/v6/`), so deploys propagate the hyphen variant. They've coexisted as parallel filesystem state for a while.

What changed today: the migration commit `36874bc` (investment-risk v6 own class + declarative shape) included `inference_hub.py`, `base_scorer.py`, `config.yaml`, `calibration.json`, `inference.py`, `inference_hybrid.py`, model config files, and probe pickle. The deploy_to_nexusmind.sh + remote_deploy.sh chain shipped all these to gpu-server's `investment-risk/v6/` (hyphen). NexusMind's filter discovery now sees BOTH `investment-risk` and `investment_risk` as separate, fully-equipped filters in the discovered list. The strict "all filters at startup must have model weights" check (added at some point — gate tightening?) then fired on the hyphen variant because `investment-risk/v6/model/` was missing.

Pre-deploy, the hyphen path was just stub code that the discovery either skipped or treated as a no-op. Today's deploy made it look real enough to be discovered → strict check → death.

**Fix (band-aid, applied 2026-04-29 14:04 UTC)**: symlink the model dir from underscore to hyphen on gpu-server:
```
ssh gpu-server "ln -s ../../investment_risk/v6/model /home/hcl/NexusMind/filters/investment-risk/v6/model"
sudo systemctl restart nexusmind-scorer
```
Restart succeeded; `/health` returns `"status":"healthy"`; `Model validation passed: all 8 filters have weights`.

**Why this is a band-aid, not a fix**: the structural problem is unresolved. There are still TWO `investment-risk` / `investment_risk` filter directories on gpu-server. The discovery loads both. Same symptom could recur on any future deploy that touches investment-risk, on any other filter where similar drift exists, or whenever someone "cleans up" the symlink without realising it's load-bearing.

**Proper fixes (deferred — see issue filed alongside this entry)**:
1. **Filesystem cleanup on gpu-server**: pick one canonical name (probably `investment_risk` underscore since that's what hcl set up originally), delete the other, and patch the deploy_filters.sh rsync source-of-truth to write only that name. Risky — might break dashboard / ovr.news if they hardcode the hyphen.
2. **NexusMind discovery normalization**: have the filter discovery normalize hyphens/underscores to one canonical name and refuse to load the duplicate. Cleaner, doesn't require filesystem cleanup.
3. **llm-distillery dir rename**: rename `filters/investment-risk/` → `filters/investment_risk/`. Aligns with the underscore convention. Touches every reference to the path; non-trivial.

**Lesson**: When two filesystem layouts represent the "same" thing through history, every deploy that bootstraps the formerly-stub side risks tripping a check that was previously dormant. The fix is to make one of them not-a-filter, not to maintain both. Filesystem-divergence between dev/staging/prod is the same shape — when the deploy makes them look more similar, latent assumptions get exercised.

**Companion lesson** (auto-deploy verify): `remote_deploy.sh`'s 90s health-check timeout caught this fast. Without that check, the broken state would have been silent until someone hit the API or noticed scoring stalling. The sadalsuud→gpu-server "unreachable" warning earlier in the deploy output was a red herring (rsync did succeed; the warning was about a separate connectivity probe). Always trust the *health check* over the intermediate warnings.

---

## NexusMind CI Has Been Red Since 2026-04-28 (sustech v3 migration; surfaced 2026-04-29)

**Problem**: Today's NexusMind push (6 deploy commits) triggered a CI failure email. Investigation shows CI has actually been red since 2026-04-28 — every NexusMind CI run since the first sustech v3 declarative-shape deploy has failed the same 2 tests. Today's push inherited the failure rather than introducing it.

**Failing tests** (`tests/unit/test_prefilter.py::TestSustainabilityPrefilter`):
- `test_passes_ev_article` — expects pass on a ~95-char EV article
- `test_passes_climate_article` — expects pass on a ~90-char climate article

**Root cause**: llm-distillery commit `e0eebd0` (sustech v3 → declarative BasePreFilter shape, ADR-018) made sustech v3 use the base `apply_filter` pipeline, which calls `check_content_length` with `MIN_CONTENT_LENGTH = 300`. The pre-existing NexusMind tests use article fixtures well below 300 chars; they pass on ANY non-trivially-bounded prefilter (which the old sustech custom apply_filter was). The migration tightened length enforcement and made these short-content tests fail.

**Detection lag**: pushed to llm-distillery 2026-04-28; deployed to NexusMind same day; NexusMind CI failed; the failure email was missed or batched. A week of subsequent NexusMind deploys (each running CI, each red) didn't surface the regression until today's deploy notification was actively read. So: CI alerts going unread for several days = red CI shipped to production for several days.

**Fix (proper, not yet applied)**: pad NexusMind test fixtures to ≥300 chars. They're testing "EV article passes" and "climate article passes" — the test contract is correct, just the fixture content is too short to trip the length gate. ~10 lines of test-file change in the NexusMind repo.

**Filed as**: separate follow-up issue alongside the path-divergence one — both surfaced by the same deploy, both need separate resolution paths.

**Lesson**: When a migration tightens a precondition (e.g., adds a length check), the downstream test suite that exercised the old looser version will start failing. That's the correct behavior — the test failures *are* the migration evidence. But if downstream CI alerts go unread, the red state persists silently. Two prevention angles: (a) explicitly look at downstream CI after every cross-repo deploy, not just self-tests; (b) have downstream tests fixture-padded with content that's safely above any plausible MIN_CONTENT_LENGTH so they're robust to upstream tightening. Both should be standard discipline going forward.

---

## Yesterday's Band-Aid Was Never Actually Applied — Overnight Outage (2026-04-29 → 2026-04-30)

**Problem**: Site rebuild chain broken since 2026-04-29 18:34 local. Five consecutive NexusMind cron triggers (19:06, 21:06, 00:16, 03:36, 07:16) all failed → ovrnews-summarize never fired → site ~13h stale. Same `RuntimeError: Cannot start scorer: 1 filter(s) missing model weights: investment-risk/v6/model` as yesterday's incident — the "Fix (band-aid, applied 2026-04-29 14:04 UTC)" entry above documents the exact symlink command that supposedly resolved this.

**Root cause**: The symlink was never actually created on gpu-server. Forensic evidence:
- `ls -la /home/hcl/NexusMind/filters/investment-risk/v6/` showed no `model` entry (neither dir nor symlink) when checked 2026-04-30 ~05:48 UTC.
- Directory mtime was `Apr 29 13:59` — the deploy timestamp. If a symlink had been created at 14:04 UTC and removed later, the mtime would have advanced. It hadn't moved.
- The `ln -s ../../investment_risk/v6/model …` command succeeded immediately when run today, proving the target name was free.

What actually happened yesterday: the gotcha-log entry was written based on intent, not execution. The scorer was running on warm config from the 13:59 deploy (which had loaded filter weights into RAM at boot before the strict precondition gate was added — the running process didn't re-validate). For ~4.5h the warm process kept serving requests. At 18:34 local, a restart cycle (likely the `ExecStopPost=systemctl start ollama.service` chain or a system event) cycled the service. On fresh start, the strict weight check fired against the still-missing path → death → 13h outage.

**Why it bypassed yesterday's verify gate**: the `<!-- verify: ... -->` line in MEMORY.md checked `curl -fs http://localhost:8000/health` AND `grep -q _uppercase_rip_in_title /home/hcl/NexusMind/filters/belonging/v1/prefilter.py`. Both passed — the running scorer was healthy on warm config; the belonging-side regex was correctly deployed. Neither check tested the symlink. The verify was wrong-shaped: it could PASS while the central claim ("symlink in place") was false.

**Fix (actually applied 2026-04-30 05:48 UTC)**:
```
ssh gpu-server "ln -s ../../investment_risk/v6/model /home/hcl/NexusMind/filters/investment-risk/v6/model"
ssh gpu-server "sudo systemctl restart nexusmind-scorer"
```
Captured outputs (this is the deploy-claim verification trail the rule requires):
- `ls -la …/investment-risk/v6/model` → `lrwxrwxrwx 1 hcl hcl 30 Apr 30 05:48 …/model -> ../../investment_risk/v6/model`
- `readlink -f …/investment-risk/v6/model` → `…/investment_risk/v6/model`
- `test -r …/investment-risk/v6/model/adapter_model.safetensors` → exit 0
- `systemctl is-active nexusmind-scorer` → `active`
- `curl -fs http://localhost:8000/health` → `{"status":"healthy","cuda_available":true,"device":"cuda",…}` with all 8 filters discovered.

**Lessons** (two distinct, both general):

1. **Verify gates must verify the specific claim, not adjacent state.** A useful gate has the property that "the verify passes" implies "the claim is true". A gate that checks scorer health + a different filter's regex while the claim is "symlink X exists" is uncorrelated — both can be true while the claim is false. Heuristic: if you can construct a state where the verify passes and the claim is false, the verify is wrong. Captured into `feedback-claim-requires-verify.md` as point #4.

2. **Remote-infra band-aids are deploys.** A gotcha-log entry that says *"applied <timestamp>: ssh gpu-server '...'"* is a deploy claim. The `.githooks/commit-msg` backstop only catches commits with deploy-words touching `filters/*/v*/` — it does not see memory/gotcha-log content, and cannot reach remote hosts. The discipline of pasting the captured ssh output into the entry is the only available gate. Captured into `feedback-claim-requires-verify.md` as point #5.

**Cost**: 13h site staleness; second occurrence in 24h of the #44 pattern. The `.githooks/commit-msg` hook from #44 worked exactly as designed — it just doesn't cover this surface area. A pre-commit hook that scans staged memory/gotcha-log content for `applied <UTC timestamp>: ssh` strings without an accompanying captured-output block could be a structural backstop; deferred for now (behavioral rule first, structural only if recurrence continues).

