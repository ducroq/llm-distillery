# GPU Server (gpu-server)

Proxmox LXC container on HCL edge server, accessed via Tailscale.

## Access

```bash
ssh gpu-server   # configured in ~/.ssh/config
```

## Environment

- **venv**: `~/gpu-server/nexusmind-scorer/venv/bin/python` — torch 2.10, sentence-transformers, scikit-learn
- **Working dir**: `~/llm-distillery/` — scripts, training data, embeddings (SCP'd, not git cloned)
- **NexusMind filters**: `~/NexusMind/filters/` — deployed filter packages
- **PYTHONPATH**: Must set `PYTHONPATH=.` or `PYTHONPATH=/home/hcl/NexusMind` for imports
- **HF_HUB_OFFLINE=1**: Can't resolve huggingface.co. Base model must be pre-cached.
- **Model cache**: `~/.cache/huggingface/hub/` — contains `google/gemma-3-1b-pt`

## File Transfers

Use **scp**, not rsync. rsync fails with dup() errors on this server when invoked from Windows Git Bash. (Linux→Linux rsync from sadalsuud works — that's why `deploy_filters.sh` runs on sadalsuud, not on the workstation.)

```bash
# Copy training data for training runs (workstation → gpu-server)
scp -r datasets/training/{name}_v{N}/ gpu-server:~/llm-distillery/datasets/training/

# Copy training output back (gpu-server → workstation)
scp -r gpu-server:~/llm-distillery/datasets/training/{name}_v{N}/model/ filters/{name}/v{N}/
```

### Deploy path (DO NOT direct-scp filters to NexusMind)

As of 2026-05-23 the canonical deploy is `scripts/deploy_to_nexusmind.{sh,ps1}` from the workstation. The script (a) verifies the filter package via `verify_filter_package.py`, (b) refuses if the NexusMind target is dirty (escape: `--force-dirty`), and (c) commits via explicit-staged `git add $FILTER_PATH filters/common/` instead of blanket `git add -A`. From there sadalsuud pulls and runs `bash scripts/deploy_filters.sh`, which rsyncs to gpu-server and restarts the scorer service. Don't bypass with direct `scp -r filters/.../v.../ gpu-server:~/NexusMind/...` — that skips the verify gate (#44) and the origin-contamination guard (#71 / 2026-05-22 incident, see gotcha-log).

## NexusMind Scorer Service

```bash
# Restart after deploying new filters
ssh gpu-server "sudo systemctl restart nexusmind-scorer"

# Check status
ssh gpu-server "sudo systemctl status nexusmind-scorer"

# Logs
ssh gpu-server "journalctl -u nexusmind-scorer -f"
```

Canonical scorer source is `deploy/gpu-server/main.py` in NexusMind repo (not llm-distillery).

## Tailscale DNS Limitation

DNS resolution to external hosts (huggingface.co) may fail. This is why `HF_HUB_OFFLINE=1` is required and models must be pre-cached.
