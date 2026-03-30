---
name: curate
description: End-of-session curation — review gotcha log, promote patterns, update memory index
disable-model-invocation: true
---

End-of-session curation for the agent-ready-projects framework.

Review the session's work and update the layered memory system:

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

## Step 4 — Verify references

Spot-check that paths mentioned in the memory index and project file still exist. Flag any broken references.

## Step 5 — Report

Summarize what you changed:
- New gotcha entries added
- Entries resolved or promoted
- Memory index updates
- Anything flagged as stale or broken
