# Remote Server Sync Guide

This guide explains how to develop locally with Claude Code and run batch jobs on a remote server.

## Problem

- You want to develop code locally (with Claude Code on Windows)
- You want to run expensive batch jobs on a remote server (Linux)
- Data is too large to commit to git (datasets can be 100s of MB)
- You need to sync code and data between machines

## Solution: SSH Sync Tool

The `sync.py` script uses SCP (SSH copy) to efficiently sync data between machines while keeping your git repo clean.

---

## Setup (One-Time)

### 1. Configure SSH Access

Ensure you can SSH into your server:

```bash
ssh your-username@your-server.example.com
```

If you need key-based auth:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key to server
ssh-copy-id your-username@your-server.example.com
```

### 2. Create Sync Configuration

```bash
# Copy example config
cp sync_config.example.json sync_config.json

# Edit with your server details
notepad sync_config.json  # Windows
nano sync_config.json     # Linux/Mac
```

**Example config:**

```json
{
  "remote": {
    "host": "llm-distiller.example.com",
    "user": "jeroen",
    "port": 22,
    "remote_path": "/home/jeroen/llm-distillery"
  },
  "sync": {
    "data_dirs": [
      "datasets/",
      "reports/"
    ],
    "exclude_patterns": [
      "*.pyc",
      "__pycache__/",
      ".git/",
      "venv/",
      ".env",
      "*.log"
    ]
  },
  "ssh": {
    "key_path": null,
    "comment": "Optional: path to SSH private key"
  }
}
```

**Security Note:** `sync_config.json` is in `.gitignore` - it will NOT be committed to git.

### 3. Verify SSH/SCP is installed

**Windows:**
- OpenSSH is included in Windows 10/11 by default
- Test it: `ssh -V` and `scp` should work in CMD/PowerShell
- If not installed: Settings → Apps → Optional Features → Add "OpenSSH Client"

**Linux/Mac:**
- SSH/SCP is pre-installed
- If not: `sudo apt install openssh-client` (Ubuntu) or `brew install openssh` (Mac)

### 4. Clone Repo on Server

```bash
# SSH into server
ssh your-username@your-server.example.com

# Clone repo
cd ~
git clone https://github.com/ducroq/llm-distillery.git
cd llm-distillery

# Set up environment (if needed)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create necessary directories
mkdir -p datasets/raw
mkdir -p reports
```

---

## Common Workflows

### Starting a Development Session

```bash
# Get latest code and data
python sync.py pull-code  # Pull code from git
python sync.py pull       # Pull data from server

# Now develop with Claude Code...
```

### Deploying Code Changes

```bash
# Push code changes
python sync.py push-code  # git push

# SSH to server and pull
ssh your-username@your-server.example.com
cd llm-distillery
git pull
```

### Running Batch Job on Server

```bash
# 1. Push latest code
python sync.py push-code

# 2. SSH to server
ssh your-username@your-server.example.com

# 3. Navigate and run
cd llm-distillery
source venv/bin/activate

# 4. Run batch labeler
python -m ground_truth.batch_labeler \
  --filter filters/uplifting/v1 \
  --source "datasets/raw/master_dataset_2025*.jsonl" \
  --output-dir datasets/uplifting_training_1500 \
  --llm gemini-flash \
  --target-count 1500 \
  --random-sample \
  --seed 42

# 5. (Optional) Run in tmux for long jobs
tmux new -s labeling
# ... run command ...
# Ctrl+B, then D to detach
# Exit SSH - job keeps running
```

### Retrieving Results

```bash
# Pull data from server to local
python sync.py pull

# Now analyze locally with Claude Code
python -m ground_truth.analyze_coverage \
  --labeled-file datasets/uplifting_training_1500/uplifting/labeled_articles.jsonl
```

### Full Sync

```bash
# Do everything: pull code, push code, pull data
python sync.py full-sync
```

---

## Sync Commands Reference

### Data Sync

```bash
# Pull data FROM server TO local
python sync.py pull

# Push data FROM local TO server
python sync.py push

# Check what would be synced (dry run)
python sync.py status
python sync.py pull --dry-run
python sync.py push --dry-run
```

### Code Sync

```bash
# Pull code from git
python sync.py pull-code

# Push code to git
python sync.py push-code
```

### Combined

```bash
# Full sync: pull code, push code, pull data
python sync.py full-sync
```

---

## What Gets Synced?

**Synced directories** (configured in `sync_config.json`):
- ✅ `datasets/` - Raw data, labeled data, training data
- ✅ `reports/` - Calibration reports, analysis results

**NOT synced** (stays in git):
- ✅ Python code (`ground_truth/`, `filters/`, etc.)
- ✅ Configuration files (`.gitignore`, `README.md`)
- ✅ Documentation (`docs/`)

**Explicitly excluded** (in both git and rsync):
- ❌ Virtual environments (`venv/`, `env/`)
- ❌ Compiled Python (`__pycache__/`, `*.pyc`)
- ❌ Secrets (`.env`, API keys)
- ❌ Log files (`*.log`)

---

## Troubleshooting

### "scp: command not found"

**Windows:**
```bash
# OpenSSH should be pre-installed on Windows 10/11
# If not, install it:
# Settings → Apps → Optional Features → Add "OpenSSH Client"

# Test if it works:
ssh -V
scp
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt install openssh-client

# Mac
brew install openssh
```

### "Permission denied (publickey)"

Your SSH keys aren't set up:

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to server
ssh-copy-id your-username@your-server.example.com

# Or manually: copy ~/.ssh/id_ed25519.pub contents to server's ~/.ssh/authorized_keys
```

### Sync appears to be stuck

**SCP is silent during transfers** - you won't see progress output, but it IS working. Large datasets take time (100s of MB can take several minutes).

Use `--dry-run` first to check what will be transferred:

```bash
python sync.py pull --dry-run
```

The sync will show:
```
Running: scp -r -p ...
(This may take a while for large datasets...)
```

Then it will be silent while transferring. Wait for "Successfully synced" message.

### "Config file not found"

```bash
# Create config from example
cp sync_config.example.json sync_config.json

# Edit with your server details
notepad sync_config.json
```

---

## Architecture: Why This Approach?

### Why NOT put data in git?

❌ **Git is not designed for large binary files:**
- Slow clones (download entire history)
- Bloated repo size (100s of MB → GBs over time)
- Merge conflicts on binary files
- GitHub has 100 MB file limit

✅ **SCP (SSH copy) is designed for file transfer:**
- Simple and reliable
- Works anywhere SSH works
- Built-in compression
- Cross-platform (Windows, Linux, Mac)

### Alternative: Git LFS

Git Large File Storage (LFS) is another option:

```bash
# Install git-lfs
git lfs install

# Track large files
git lfs track "datasets/**/*.jsonl"
```

**Pros:**
- Integrated with git
- Version control for data

**Cons:**
- Requires GitHub LFS quota (paid)
- Slower than SCP for frequent changes
- More complex setup

**Verdict:** SCP is simpler and free for our use case.

---

## Best Practices

### 1. Develop Locally, Run Remotely

```
┌─────────────────┐         ┌──────────────────┐
│  Local (Win)    │         │  Server (Linux)  │
│  - Claude Code  │──git──→│  - batch_labeler │
│  - Development  │←─rsync──│  - Heavy compute │
│  - Analysis     │         │  - Data storage  │
└─────────────────┘         └──────────────────┘
```

### 2. Commit Often, Sync Less

- **Code changes**: Commit and push frequently
- **Data sync**: Only when needed (start/end of work session)

### 3. Use Descriptive Commit Messages

```bash
# Good
git commit -m "Add random sampling to batch_labeler"

# Bad
git commit -m "fix"
```

### 4. Check Status Before Sync

```bash
# See what would be synced
python sync.py status

# Then decide: pull or push?
```

### 5. Use tmux for Long Jobs

```bash
# On server
tmux new -s labeling

# Run long job
python -m ground_truth.batch_labeler ...

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t labeling
```

---

## Example: Complete Workflow

### Scenario: Run calibration on server, analyze locally

```bash
# 1. LOCAL: Push latest code
python sync.py push-code

# 2. SSH to server
ssh jeroen@llm-distiller.example.com
cd llm-distillery
git pull

# 3. Run calibration (on server)
python -m ground_truth.calibrate_oracle \
  --filter filters/uplifting/v1 \
  --source "datasets/raw/*.jsonl" \
  --models gemini-flash,gemini-pro \
  --sample-size 100 \
  --seed 42

# Report saved to: reports/uplifting_calibration.md

# 4. Exit SSH
exit

# 5. LOCAL: Pull results
python sync.py pull

# 6. LOCAL: Analyze with Claude Code
# reports/uplifting_calibration.md now available locally
```

---

## Security Considerations

### What NOT to commit:

- ❌ `sync_config.json` (contains server hostnames/usernames)
- ❌ SSH private keys (`.pem`, `*.key`)
- ❌ API keys (`.env`, `secrets.ini`)
- ❌ Labeled data (may contain sensitive content)

### What's safe to commit:

- ✅ `sync_config.example.json` (template without real details)
- ✅ All Python code
- ✅ Filter configurations
- ✅ Documentation

### SSH Key Security:

```bash
# Set correct permissions on private key
chmod 600 ~/.ssh/id_ed25519

# Use passphrase-protected keys
ssh-keygen -t ed25519 -C "your-email@example.com"
# (enter passphrase when prompted)
```

---

## FAQ

### Q: Can I sync TO server?

**A:** Yes! `python sync.py push` syncs data from local to server. Useful for:
- Uploading new raw datasets
- Sharing analysis results with server

### Q: What if I forget to pull before editing?

**A:** Git will warn about conflicts. Commit or stash your changes:

```bash
# Option 1: Commit first
git add .
git commit -m "WIP: local changes"
git pull

# Option 2: Stash
git stash
git pull
git stash pop
```

### Q: Can multiple people use the same server?

**A:** Yes! Each person should:
1. Have their own user account on server
2. Clone repo to their own home directory
3. Use separate `output_dir` for batch jobs

### Q: Sync is stuck. How do I cancel?

**A:** Press `Ctrl+C`. rsync will stop cleanly (no corruption).

---

## Summary

**Development Workflow:**

1. 📥 `python sync.py pull-code` - Get latest code
2. 🔨 Develop locally with Claude Code
3. 📤 `python sync.py push-code` - Push code changes
4. 🖥️  SSH to server, `git pull`, run batch jobs
5. 📥 `python sync.py pull` - Pull results
6. 📊 Analyze locally with Claude Code
7. Repeat!

**Key Commands:**

```bash
python sync.py pull         # Get data from server
python sync.py push         # Send data to server
python sync.py pull-code    # Get code from git
python sync.py push-code    # Send code to git
python sync.py status       # Check sync status
python sync.py full-sync    # Do everything
```

---

For questions or issues, see [GitHub Issues](https://github.com/ducroq/llm-distillery/issues).
