# AI-Augmented Development Workflow

**Purpose:** This document defines how AI assistants (Claude Code and others) should work on this project. It establishes the philosophy, principles, and operational protocols for AI-augmented solo development.

**For AI Assistants:** Read this document at the start of each session to understand how to behave.

**For Humans:** This documents the "contract" between you and your AI assistant.

---

## Core Philosophy

### 1. Documentation Should Be Effortless

**Principle:** Documentation is maintained by AI, not humans.

**Why:** Documentation decay happens when it requires manual effort. If the AI assistant automatically maintains docs as a side effect of development work, documentation stays fresh without cognitive overhead.

**In practice:**
- Human: "Create a generic training script"
- AI: Creates script, offers to update docs
- Human: "Yes"
- AI: Updates training/README.md, docs/SESSION_STATE.md, creates ADR

### 2. Single Source of Truth

**Principle:** Configuration drives behavior, not hardcoded values.

**Why:** When information exists in multiple places, they drift out of sync. By making config files the authoritative source, scripts automatically stay correct when config changes.

**In practice:**
- Filter dimensions defined in `config.yaml`
- Generic scripts read config, not hardcoded lists
- New filters work with zero code changes

### 3. Progressive Disclosure

**Principle:** Load context as needed, not everything at once.

**Why:** AI context windows are limited. Loading entire codebases wastes tokens and time. Progressive disclosure (broad → specific) gets answers faster with relevant context only.

**In practice:**
- Start: Read docs/SESSION_STATE.md (2k tokens)
- Navigate: Find relevant component (1k tokens)
- Detail: Read specific code section (5k tokens)
- Total: 8k tokens vs 50k+ for full codebase dump

### 4. Sandbox for Experiments

**Principle:** Git-ignored directories for friction-free experimentation.

**Why:** Fear of polluting git history stifles experimentation. Sandbox directories give freedom to try ideas, fail fast, and document learnings without committing garbage.

**In practice:**
- Risky idea → Work in `sandbox/2025-11-12_new_approach/`
- Success → Clean up, move to proper location, create ADR
- Failure → Document in `sandbox/failed/` with learnings

### 5. Architecture Decision Records

**Principle:** Significant decisions are documented with context, not just the outcome.

**Why:** Future developers (including yourself in 6 months) need to understand WHY decisions were made, not just WHAT was decided. ADRs capture the thinking.

**In practice:**
- Significant decision made in conversation
- AI offers: "Should I create an ADR for this?"
- ADR documents context, decision, consequences, alternatives

### 6. Agent-Assisted Quality Assurance

**Principle:** Complex validation tasks are delegated to specialized agents.

**Why:** Multi-step validation across many files is tedious and error-prone for humans. Agents can systematically validate datasets, generate reports, and catch issues.

**In practice:**
- After labeling 8,162 articles
- Human: "Audit the dataset using dimensional regression QA"
- Agent validates, generates report, flags issues
- Human reviews report, proceeds to training

---

## Session Start Protocol

### When Starting a New Session

**1. Read Core Context (Required)**
```
Read: docs/SESSION_STATE.md
Purpose: Understand current status, recent accomplishments, next steps
```

**2. Check Recent Changes (If applicable)**
```
Run: git status
Check: Any uncommitted work?
Run: git log -5 --oneline
Check: Recent commits since last session
```

**3. Scan Recent Decisions (If applicable)**
```
Read: docs/decisions/ (sort by date, read latest 2-3)
Purpose: Understand recent architectural decisions
```

**4. Orient the User**
```
Provide concise summary:
- Where we left off (from docs/SESSION_STATE.md)
- Current status of key components
- Suggested next steps (from "Next Steps" section)
```

**Example:**
```
"Welcome back! Based on docs/SESSION_STATE.md:

Current Status:
- Ground truth datasets validated (7,715 uplifting, 8,162 tech deployment)
- Training data prepared for uplifting filter
- Generic preparation script created and tested

Next Steps:
1. Prepare tech deployment training data
2. Train models on both filters
3. Evaluate on test sets

What would you like to work on?"
```

---

## During Development

### Proactive Behaviors

**1. Recognize Significant Decisions**

When the conversation involves:
- Choosing between architectural approaches
- Making trade-offs with long-term impact
- Establishing patterns or conventions
- Changing previous decisions

→ Offer to create an ADR:
```
"This seems like a significant architectural decision. Should I create an ADR documenting:
- Context: [brief context]
- Decision: [what we decided]
- Alternatives considered: [what we rejected and why]?"
```

**2. Suggest Documentation Updates**

After completing significant work:
- Created/modified scripts
- Changed interfaces
- Resolved issues
- Completed features

→ Offer to update docs:
```
"I've completed [task]. Should I update the documentation?
- docs/SESSION_STATE.md (add to accomplishments)
- training/README.md (update usage examples)
- Create ADR if applicable"
```

**3. Use Progressive Context Loading**

When answering questions:
1. Start broad (docs/SESSION_STATE.md, README files)
2. Navigate to relevant area (directory structure, component docs)
3. Load specific context (relevant code sections only)
4. Synthesize answer with file:line references

**Don't:**
- Read entire files unless needed
- Load entire codebase for simple questions
- Dump large code blocks in responses

**4. Move Experiments to Sandbox**

When user wants to try something risky:
```
"This sounds like a good experiment for sandbox/. Want me to create
sandbox/2025-11-12_[experiment_name]/ for this?"
```

**5. Flag Potential Issues**

When you notice:
- Documentation getting stale
- Hardcoded values that should be in config
- Code duplication
- Missing tests or validation

→ Proactively mention it:
```
"I notice [issue]. Should we address this now or add it to the backlog?"
```

---

## Session End Protocol

### Before User Leaves

**1. Offer Final Documentation Update**
```
"Before you go, should I update docs/SESSION_STATE.md with today's progress?
- Add accomplishments
- Update current status
- Refresh next steps"
```

**2. Suggest Commit Message (If work completed)**
```
"Ready to commit? Suggested message:
'Add generic training data preparation script

- Created scripts/prepare_training_data.py (works for any filter)
- Removed filter-specific scripts
- Updated training/README.md with new usage
- Created ADRs for dimensional regression and generic script approach'
"
```

**3. Summarize Open Items**
```
"Summary of where we're at:
✅ Completed: [list completed items]
🚧 In Progress: [list partial work]
📋 Next: [list next steps]

See docs/SESSION_STATE.md for details."
```

---

## Rules of Engagement

### Always

- ✅ Use progressive context loading (aim for 10-20k tokens)
- ✅ Offer to document significant decisions as ADRs
- ✅ Update docs/SESSION_STATE.md at session end
- ✅ Move experiments to sandbox/ (git-ignored)
- ✅ Reference specific file:line locations when helpful
- ✅ Verify commands before running destructive operations
- ✅ Explain trade-offs when multiple approaches exist

### Never

- ❌ Dump entire files unless specifically asked
- ❌ Create documentation "just because" (pragmatic over dogmatic)
- ❌ Commit to git without user approval
- ❌ Make breaking changes without discussion
- ❌ Hardcode values that should be in config
- ❌ Duplicate code instead of making generic
- ❌ Leave documentation stale after significant changes

### When Uncertain

- 🤔 Ask clarifying questions instead of guessing
- 🤔 Offer options with trade-offs instead of dictating
- 🤔 Suggest searching docs before reading entire codebase
- 🤔 Propose experiments in sandbox/ for risky ideas

---

## Project-Specific Context

### Key Concepts

**1. Filter Packages**
- Location: `filters/{filter_name}/v1/`
- Contents: `config.yaml`, `prefilter.py`, `prompt-compressed.md`, `README.md`
- Principle: `config.yaml` is single source of truth for dimensions, weights, tiers

**2. Multi-Dimensional Regression**
- Training objective: Predict 8 dimensional scores (0-10 range)
- NOT tier classification (tiers are metadata only)
- See: `docs/decisions/2025-11-12-dimensional-regression-training.md`

**3. Ground Truth Datasets**
- Location: `datasets/scored/{filter_name}/labeled_articles.jsonl`
- Oracle: Google Gemini Flash (cost-effective batch scoring)
- Validation: Dimensional regression QA criteria

**4. Training Pipeline**
- Preparation: `scripts/prepare_training_data.py` (generic, reads config)
- Training: `training/train.py` (planned)
- Evaluation: Per-dimension MAE/RMSE

### Directory Structure

```
llm-distillery/
├── docs/
│   ├── agents/                    # For AI assistants (portable!)
│   │   ├── AI_AUGMENTED_WORKFLOW.md  # This file
│   │   ├── agent-operations.md    # Agent operations guide
│   │   ├── templates/
│   │   │   ├── dimensional-regression-qa-agent.md
│   │   │   └── ADR-TEMPLATE.md
│   │   └── README.md
│   ├── guides/                    # For humans
│   │   ├── getting-started.md
│   │   └── ...
│   └── decisions/                 # Architecture Decision Records
│       ├── README.md
│       └── 2025-11-12-*.md
├── filters/
│   └── {filter_name}/v1/
│       ├── config.yaml            # Single source of truth
│       ├── prefilter.py
│       ├── prompt-compressed.md
│       └── README.md
├── datasets/
│   ├── labeled/{filter_name}/     # Ground truth
│   │   ├── labeled_articles.jsonl
│   │   └── README.md
│   └── training/{filter_name}/    # Prepared splits
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── scripts/
│   └── prepare_training_data.py   # Generic (config-driven)
├── sandbox/                       # Git-ignored experiments
│   └── README.md
├── reports/                       # Agent outputs
└── docs/SESSION_STATE.md               # Current status (START HERE)
```

### Important Files

**Always read first:**
- `docs/SESSION_STATE.md` - Current status, accomplishments, next steps

**Reference as needed:**
- `docs/agents/agent-operations.md` - Agent operational guide
- `docs/agents/templates/` - Agent templates (QA, ADR)
- `docs/decisions/` - Recent ADRs
- Filter READMEs - Filter-specific documentation
- Dataset READMEs - Data quality info

### Common Tasks

**Task: Prepare training data**
```bash
python scripts/prepare_training_data.py \
    --filter filters/{filter_name}/v1 \
    --input datasets/scored/{filter_name}/labeled_articles.jsonl \
    --output-dir datasets/training/{filter_name}
```

**Task: Validate dataset**
```
Use Task tool with subagent_type: "general-purpose"
Prompt: "Audit the {filter_name} dataset at datasets/scored/{filter_name}/labeled_articles.jsonl
for dimensional regression training. Expected dimensions: 8.
Use dimensional regression QA criteria from docs/agents/templates/dimensional-regression-qa-agent.md"
```

**Task: Create ADR**
```
1. Copy docs/agents/templates/ADR-TEMPLATE.md
2. Fill in: Context, Decision, Consequences, Alternatives
3. Save as: docs/decisions/YYYY-MM-DD-title.md
4. Update docs/decisions/README.md with active ADR
5. Update docs/SESSION_STATE.md to reference ADR
```

**Task: Create agent template**
```
1. Start with YAML frontmatter:
   ---
   name: "Agent Name"
   description: "What this agent does"
   model: "sonnet"  # or "haiku" for quick tasks
   trigger_keywords:
     - "keyword 1"
     - "keyword 2"
   when_to_use: "When to invoke"
   focus: "Primary focus"
   output: "What it produces"
   ---

2. Write agent prompt template with:
   - Critical checks (must pass)
   - Quality checks (report but don't block)
   - Informational only (don't flag)
   - Decision criteria (Pass/Review/Fail)

3. Save as: docs/agents/templates/{agent-name}.md
4. Document in docs/agents/agent-operations.md
```

---

## Success Metrics

You're doing this right when:

- ✅ User never manually updates documentation
- ✅ Context loads are fast and focused (<30 seconds)
- ✅ Significant decisions are captured as ADRs
- ✅ Experiments happen freely in sandbox/
- ✅ docs/SESSION_STATE.md accurately reflects reality
- ✅ New filters require zero code changes (config-driven)
- ✅ User can resume after weeks away with clear context

---

## Anti-Patterns to Avoid

- ❌ **Documentation debt:** Completing work without updating docs
- ❌ **Hardcoding:** Duplicating config values in code
- ❌ **Context overload:** Reading entire codebase for simple questions
- ❌ **Git pollution:** Committing experiments instead of using sandbox
- ❌ **Undocumented decisions:** Making significant choices without ADRs
- ❌ **Stale state:** docs/SESSION_STATE.md not reflecting current reality
- ❌ **Assuming context:** Not reading docs/SESSION_STATE.md at session start

---

## Version History

### v1.0 (2025-11-12)
- Initial AI-augmented workflow guide
- Core philosophy: effortless docs, single source of truth, progressive disclosure
- Session protocols: start, during, end
- Rules of engagement and anti-patterns
- Project-specific context for llm-distillery

---

## See Also

- `docs/agents/agent-operations.md` - Detailed agent operations guide (agents, ADRs, automated docs)
- `docs/agents/templates/dimensional-regression-qa-agent.md` - Dataset QA template
- `docs/agents/templates/ADR-TEMPLATE.md` - Architecture decision record template
- `docs/decisions/` - Architecture Decision Records
- `docs/SESSION_STATE.md` - Current project status (read this first!)
- `sandbox/README.md` - Experimentation guidelines
- `C:\local_dev\AI_AUGMENTED_SOLO_DEV_FRAMEWORK.md` - Original framework inspiration
