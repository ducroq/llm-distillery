# Local ↔ GPU Sync Workflow

## Overview

This project uses a **hybrid approach**:
- **Git**: Version control (local machine only)
- **FreeFileSync**: File synchronization between local and GPU machine

## The Setup

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   LOCAL (CPU) Machine   │         │   GPU Machine           │
│                         │         │                         │
│  ✓ Git repository       │         │  ✗ No git               │
│  ✓ Version control      │◄────────┤  ✓ Working files only   │
│  ✓ Commits & pushes     │  Sync   │  ✓ Training jobs        │
│  ✓ Code editing         │────────►│  ✓ Model generation     │
│  ✓ Dataset creation     │ FreeFile│                         │
│                         │  Sync   │                         │
└─────────────────────────┘         └─────────────────────────┘
         │                                      │
         │                                      │
         └──────────┐                ┌──────────┘
                    │                │
                    ▼                ▼
            ┌───────────────────────────┐
            │   GitHub (Remote)         │
            │   github.com/ducroq/...   │
            │                           │
            │   Pushed from LOCAL only  │
            └───────────────────────────┘
```

## What Gets Synced Where

### FreeFileSync (Local ↔ GPU)
✅ Python code (`.py` files)
✅ Configuration files
✅ Datasets (large `.jsonl`, `.parquet` files)
✅ Trained models (`.pt`, `.safetensors` files)
✅ Results and reports
❌ `.git` folder (EXCLUDED)
❌ `venv` folders (EXCLUDED)
❌ `__pycache__` (EXCLUDED)

### Git (Local → GitHub)
✅ Python code (`.py` files)
✅ Configuration files
✅ Documentation (`.md` files)
✅ Small metadata files
❌ Large datasets (in `.gitignore`)
❌ Model files (in `.gitignore`)
❌ Virtual environments (in `.gitignore`)

## Workflow

### Daily Development

1. **Edit code locally**
   ```bash
   # On LOCAL machine
   vim scripts/train.py
   ```

2. **Commit to git (local only)**
   ```bash
   # On LOCAL machine
   git add scripts/train.py
   git commit -m "Update training script"
   git push origin main
   ```

3. **Sync to GPU**
   - Open FreeFileSync
   - Click "Compare"
   - Review changes
   - Click "Synchronize"
   - Your code is now on GPU!

4. **Run training on GPU**
   ```bash
   # SSH to GPU machine
   cd /path/to/llm-distillery
   python scripts/train.py
   ```

5. **Sync results back**
   - FreeFileSync automatically brings models back to local
   - Review results locally

### Creating Datasets

**On LOCAL machine:**
```bash
python scripts/create_dataset.py
# Creates datasets/my_dataset.jsonl
```

**FreeFileSync** → Automatically syncs to GPU

### Training Models

**On GPU machine:**
```bash
python training/train_model.py
# Creates models/my_model.pt
```

**FreeFileSync** → Automatically syncs back to local

## Important Rules

### ⚠️ DO:
- ✅ Use git on LOCAL machine only
- ✅ Commit and push from LOCAL only
- ✅ Let FreeFileSync handle file synchronization
- ✅ Run FreeFileSync in "Compare" mode first before syncing
- ✅ Keep separate venvs on each machine

### ⚠️ DON'T:
- ❌ Never use git on GPU machine
- ❌ Never push to GitHub from GPU
- ❌ Never manually copy `.git` folder
- ❌ Never sync `venv` folders
- ❌ Never edit the same file on both machines simultaneously

## Setup Checklist

### On LOCAL Machine:
- [ ] Git repository configured ✓
- [ ] FreeFileSync installed
- [ ] Exclusion rules configured (see `FREEFILESYNC_EXCLUSIONS.txt`)
- [ ] Virtual environment: `venv/` (excluded from sync)

### On GPU Machine:
- [ ] Working directory created
- [ ] FreeFileSync configured (if running on GPU) OR network share accessible
- [ ] Virtual environment: `venv_gpu/` or similar (excluded from sync)
- [ ] **No `.git` folder** (will be excluded by FreeFileSync)

### FreeFileSync Configuration:
- [ ] Exclusions added (see `FREEFILESYNC_EXCLUSIONS.txt`)
- [ ] Comparison method: "File time and size"
- [ ] Conflict resolution: "Newer file wins" or "Show popup"
- [ ] Test sync in preview mode first!

## Troubleshooting

### "Files are different but git says up to date"
- This is expected! Git only tracks code, FreeFileSync handles data/models
- Git = version control
- FreeFileSync = file synchronization

### "Virtual environment is corrupted"
- Make sure `venv/` is in FreeFileSync exclusions
- Each machine should have its own separate venv
- Reinstall venv if needed: `python -m venv venv`

### "Git is acting weird"
- Check if `.git` folder was synced by mistake
- If corrupted: `git fsck` or re-clone from GitHub
- Update FreeFileSync exclusions to prevent this

### "Conflicts in FreeFileSync"
- If you edited same file on both machines, FreeFileSync will flag it
- Choose "newer file" or manually merge
- Better: only edit code on LOCAL, let GPU just run it

## See Also

- `FREEFILESYNC_EXCLUSIONS.txt` - Copy-paste ready exclusion rules
- `.gitignore` - What git ignores
- `README.md` - Project overview
