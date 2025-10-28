# Tmux Usage Guide for Server Execution

This guide explains how to run long-running jobs from this repository on a server using tmux, which allows processes to continue running even after you disconnect from the server.

## Starting a Job in Tmux

### 1. Create a new tmux session

```bash
# Start a new tmux session named "distillery"
tmux new -s distillery
```

### 2. Run your command

Inside the tmux session:

```bash
# Navigate to the project directory
cd /path/to/llm-distillery

# Activate your virtual environment
source venv/bin/activate

# Run your command (example: batch labeling)
python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source datasets/raw/master_dataset.jsonl \
    --llm gemini \
    --pre-filter uplifting \
    --batch-size 50 \
    --output-dir datasets
```

  <!-- 1. Uplifting:
  python -m ground_truth.batch_labeler \
      --prompt prompts/uplifting.md \
      --source datasets/raw/master_dataset.jsonl \
      --llm gemini \
      --pre-filter uplifting \
      --batch-size 50 \
      --output-dir datasets

  2. Sustainability:
  python -m ground_truth.batch_labeler \
      --prompt prompts/sustainability.md \
      --source datasets/raw/master_dataset.jsonl \
      --llm gemini \
      --pre-filter sustainability \
      --batch-size 50 \
      --output-dir datasets

  3. SEECE:
  python -m ground_truth.batch_labeler \
      --prompt prompts/seece-energy-tech.md \
      --source datasets/raw/master_dataset.jsonl \
      --llm gemini \
      --pre-filter seece \
      --batch-size 50 \
      --output-dir datasets -->

## Detaching from Tmux

To leave the session running in the background:

**Press:** `Ctrl+B`, then press `D`

You'll return to your normal terminal, and the job will keep running in the background.

## Reconnecting to Your Session

### List all tmux sessions

```bash
tmux ls
```

### Reattach to your session

```bash
tmux attach -t distillery
```

## Other Useful Tmux Commands

### Kill a session

When you're completely done with the session:

```bash
tmux kill-session -t distillery
```

### Create a new window inside tmux

If you need another terminal while inside tmux:

**Press:** `Ctrl+B`, then press `C`

### Switch between windows

**Press:** `Ctrl+B`, then press `0` (or `1`, `2`, etc.)

## Quick Reference

| Action | Command / Shortcut |
|--------|-------------------|
| Create new session | `tmux new -s <name>` |
| Detach from session | `Ctrl+B`, then `D` |
| List sessions | `tmux ls` |
| Attach to session | `tmux attach -t <name>` |
| Kill session | `tmux kill-session -t <name>` |
| New window | `Ctrl+B`, then `C` |
| Switch windows | `Ctrl+B`, then `<number>` |
| List windows | `Ctrl+B`, then `W` |

## Notes

- Tmux sessions persist even if you lose your SSH connection
- Long-running jobs (e.g., ~30 hours) can safely run in detached tmux sessions
- Always remember to kill sessions when completely done to free up server resources
