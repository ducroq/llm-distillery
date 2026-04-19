#!/bin/bash
# Run the NexusMind deploy on sadalsuud (Linux) rather than the workstation.
#
# Why: NexusMind/scripts/deploy_filters.sh uses rsync to push filters to gpu-server.
# rsync from Windows Git Bash intermittently fails with `dup() in/out/err failed`
# (documented in NexusMind/memory/gotcha-log.md, 2026-02). Running the script on
# sadalsuud (Linux host) sidesteps the issue entirely — Linux->Linux rsync is
# reliable. This wrapper does the SSH hop in one command.
#
# Prerequisites:
#   - Your NexusMind commit is already pushed to origin (otherwise sadalsuud pulls
#     nothing new).
#   - `ssh sadalsuud` works (alias in ~/.ssh/config).
#
# Usage:
#   ./scripts/remote_deploy.sh
set -euo pipefail

SADALSUUD_NEXUSMIND="/home/jeroen/local_dev/NexusMind"

echo "=== Remote deploy via sadalsuud ==="
echo ""

# Sanity: confirm the host is reachable before committing to the work.
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes sadalsuud "true" 2>/dev/null; then
    echo "ERROR: sadalsuud unreachable over SSH. Check Tailscale, SSH config, and network."
    exit 1
fi

# Pull latest on sadalsuud and run deploy_filters.sh there.
# deploy_filters.sh itself verifies sadalsuud's HEAD matches origin before rsyncing
# and does a /health round-trip hash check after restarting the scorer.
ssh sadalsuud "cd \"$SADALSUUD_NEXUSMIND\" && git pull && bash scripts/deploy_filters.sh"

echo ""
echo "=== Done ==="
