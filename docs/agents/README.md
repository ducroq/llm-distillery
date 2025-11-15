# Agent Documentation

**Purpose:** This directory contains documentation for AI assistants working on this project.

**Portability:** This entire `docs/agents/` directory is designed to be portable - you can copy it to any project and AI assistants will understand how to work effectively.

---

## Core Documents

### AI_AUGMENTED_WORKFLOW.md
**For:** AI assistants (Claude Code, etc.)
**Purpose:** Defines philosophy, principles, and operational protocols
**When to read:** At the start of every session

**Contains:**
- Core philosophy (effortless docs, single source of truth, progressive disclosure)
- Session start protocol (what to read, how to orient the user)
- Proactive behaviors (when to create ADRs, update docs)
- Session end protocol (final updates, commit messages)
- Rules of engagement (always/never/when uncertain)

### agent-operations.md
**For:** AI assistants
**Purpose:** Detailed operational guide for agent workflows
**When to read:** When creating agents, ADRs, or doing complex tasks

**Contains:**
- Agent workflow (when to use agents, types, invocation)
- Dimensional regression QA agent (project-specific example)
- Creating new agent templates (YAML frontmatter, structure)
- Architecture Decision Records (ADR creation protocol)
- Automated documentation updates (what/when to update)
- Progressive context loading (how to efficiently load context)

---

## templates/

Reusable templates for agent tasks:

### ADR-TEMPLATE.md
Template for Architecture Decision Records. Copy this when documenting significant technical decisions.

### dimensional-regression-qa-agent.md
**Purpose:** Dataset quality assurance for dimensional regression training
**When:** After consolidating labeled data, before training
**Output:** QA report with Pass/Review/Fail status
**Example:** Validates 8 dimensional scores are present, valid (0-10), with good range coverage

### oracle-calibration-agent.md
**Purpose:** Validate oracle performance before large-scale batch labeling
**When:** Before labeling thousands of articles, after prompt changes, periodic quality checks
**Output:** Calibration report with Ready/Review/Block recommendation
**Oracle:** Uses Gemini Pro for calibration (accurate), Gemini Flash for production (cheap)
**Cost:** ~$0.20 for 200-article calibration sample

**Format:**
```yaml
---
name: "Agent Name"
description: "What this agent does"
model: "sonnet"  # or "haiku" for quick tasks
trigger_keywords:
  - "keyword 1"
when_to_use: "When to invoke"
focus: "Primary focus"
output: "What it produces"
---
```

---

## How to Use This Directory

### For AI Assistants

**At session start:**
1. Read `AI_AUGMENTED_WORKFLOW.md` - Get the operating system
2. Read project's `docs/SESSION_STATE.md` - Understand current status
3. Orient the user

**During work:**
- Follow proactive behaviors (offer ADRs, update docs)
- Use progressive context loading
- Reference `agent-operations.md` for detailed workflows

**At session end:**
- Update project's `docs/SESSION_STATE.md`
- Suggest commit message
- Summarize progress

### For Humans (Copying to New Project)

**To use in a new project:**
1. Copy entire `docs/agents/` directory to new project
2. Update `AI_AUGMENTED_WORKFLOW.md` "Project-Specific Context" section:
   - Key concepts
   - Directory structure
   - Important files
   - Common tasks
3. Create project's `docs/SESSION_STATE.md`
4. AI assistants now know how to work!

**What to customize:**
- Project-specific context in `AI_AUGMENTED_WORKFLOW.md`
- Agent templates in `templates/` (add project-specific agents)
- Keep core philosophy and protocols unchanged

---

## Version History

### v1.0 (2025-11-12)
- Initial agent documentation structure
- Portable design for reuse across projects
- Core philosophy: effortless docs, single source of truth, progressive disclosure
- Session protocols for AI assistants
- Agent templates with YAML frontmatter
