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

Use **scp**, not rsync. rsync fails with dup() errors on this server.

```bash
# Deploy filters to NexusMind
scp -r filters/common/ gpu-server:~/NexusMind/filters/common/
scp -r filters/{name}/v{N}/ gpu-server:~/NexusMind/filters/{name}/v{N}/

# Copy training data for training runs
scp -r datasets/training/{name}_v{N}/ gpu-server:~/llm-distillery/datasets/training/
```

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
