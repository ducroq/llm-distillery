# Implement Checklist

Definition-of-done for implementation work. Use before claiming a feature or fix is complete.

## Code Completeness

- [ ] All acceptance criteria from the design are implemented
- [ ] No TODO/FIXME/HACK comments added without a linked issue or explicit deferral
- [ ] No commented-out code left behind
- [ ] No untracked files that should be committed (check `git status`)

## Architecture Compliance

- [ ] Implementation follows established patterns (base_scorer, inference, inference_hub, config.yaml)
- [ ] No new dependencies added without justification
- [ ] File placement matches the project's directory conventions (`filters/{name}/v{N}/`)
- [ ] Hard constraints in CLAUDE.md respected (oracle outputs scores only, OLD PEFT keys, etc.)

## Filter Deployment (if deploying a filter)

- [ ] `calibration.json` fitted and committed
- [ ] `normalization.json` fitted from production data (or documented as pending)
- [ ] Hub upload tested — `PeftModel.from_pretrained()` loads correctly
- [ ] `config.yaml` has correct dimensions, weights, and thresholds
- [ ] Production filters table in CLAUDE.md updated

## Operational

- [ ] No secrets, credentials, or tokens in code or config files
- [ ] Error handling exists at system boundaries
- [ ] RUNBOOK updated if deployment steps changed
- [ ] Git diff matches what you'd describe in a PR — no unrelated changes
