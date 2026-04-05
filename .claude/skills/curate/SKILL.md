---
name: curate
description: End-of-session curation — freshness check, review gotcha log, promote patterns, update memory index
disable-model-invocation: true
---

End-of-session curation for the agent-ready-projects framework.

Review the session's work and update the layered memory system:

## Step 0 — Freshness check

Check for context rot from *previous* sessions. This catches what the session-focused steps below miss.

1. **Dead references**: Read the memory index and project file. For every file path mentioned, verify it still exists. List any broken paths.
2. **Stale memory**: Check modification dates of files in `memory/`. Flag any that haven't been modified in 30+ days — they may be outdated. (Use `git log -1 --format=%ci -- <file>` for each.)
3. **Lingering gotchas**: Read the gotcha log. Flag any unresolved entries older than 14 days — they're either fixed (mark `[RESOLVED]`) or stuck (surface to the user).
4. **Ground truth drift**: If the project file has a "Ground Truth Designations" table, verify each listed file exists and has been modified more recently than the artifacts that defer to it. Flag any where a downstream artifact is newer than its source of truth.

Report findings before proceeding. Don't fix anything in this step — just surface what's stale so the engineer can decide.

## Step 1 — Gotcha log review

Read `memory/gotcha-log.md`. For each existing entry:
- If the root cause was fixed during this session, mark it `[RESOLVED]`
- If the same issue came up again, note the recurrence

Then check: did anything go wrong or surprise you during this session? For each one, append a new entry:

```
### [Short description] (YYYY-MM-DD)
**Problem**: What went wrong or was confusing.
**Root cause**: Why it happened.
**Fix**: What solved it.
```

## Step 2 — Pattern detection and promotion

Scan the gotcha log for entries that have recurred 2-3 times. For each:
- Propose promoting it as an "if [situation], then [what to do]" pattern
- Suggest where it belongs: the memory index (if broadly relevant) or a topic file (if subsystem-specific)
- If approved, add it to the destination and update the Promoted table in the gotcha log

## Step 3 — Memory index update

Read `memory/MEMORY.md`. Update:
- **Active Decisions** — add any architectural choices made, with ADR pointers if created
- **Key File Paths** — add any important files discovered during work
- **Next Up** — reflect what shipped or changed this session
- Remove or correct anything that is now stale

## Step 4 — Doc sync check

Check whether key docs reflect the current repo state. Code changes during a session can leave docs stale — this step catches drift that inline updates missed.

1. **CLAUDE.md Architecture section**: Compare listed files/directories against actual repo contents. Flag new files not listed, or listed files that no longer exist.
2. **CLAUDE.md Key Commands / Getting Started**: Verify commands still match actual CLI flags and defaults. Flag any mismatches (e.g., a renamed flag, a changed default).
3. **RUNBOOK** (`docs/RUNBOOK.md`): Check that operational details (environment setup, deployment steps, common problems) match reality. Flag anything that looks stale.
4. **Production Filters table**: Check if any filters changed version, MAE, or status during this session. Update the table.
5. **TODO / ROADMAP**: Check if any open items were resolved during this session. Mark them.

Fix what you can. Flag anything that needs engineer input.

## Step 5 — Verify references

Skip if Step 0 already ran a full freshness check. Otherwise, spot-check that paths mentioned in the memory index and project file still exist. Flag any broken references.

## Step 6 — Report

Summarize what you changed:
- **Freshness**: Dead references, stale memory files, lingering gotchas, ground truth drift (from Step 0)
- **Gotchas**: New entries added, entries resolved or promoted
- **Memory index**: Updates made
- **Doc sync**: CLAUDE.md, RUNBOOK, production filters table updates made or flagged (from Step 4)
- **Action needed**: Anything flagged that requires engineer decision
