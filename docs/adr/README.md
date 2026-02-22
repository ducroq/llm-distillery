# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting important technical decisions made in the LLM Distillery project.

## What are ADRs?

ADRs are short documents that capture important architectural decisions along with their context and consequences. They help teams understand:
- Why decisions were made
- What alternatives were considered
- What trade-offs were accepted

## ADR Index

- [ADR-001: Moderate Dimension Correlations Are Acceptable](001-moderate-correlation-acceptable.md) - Clarifies when dimension correlations reflect real domain relationships vs problematic redundancy
- [ADR-002: Modern Prompt Format Support](002-modern-prompt-format.md) - Introduces flexible prompt format without wrapper sections, supports JSON examples and custom structures
- [ADR-003: Screening Filter for Training Data](003-screening-filter-for-training-data.md) - Enriches training data with signal-bearing content before oracle scoring
- [ADR-004: Commerce Prefilter as Universal Noise Filter](004-universal-noise-prefilter.md) - Only commerce is universal noise; filter-specific noise handled by model (PROPOSED)
- [ADR-005: Active Learning for Filter Improvement](005-active-learning-for-filter-improvement.md) - Use model predictions to guide training data collection; includes needle hunting strategy
- [ADR-006: Hybrid Inference Pipeline](006-hybrid-inference-pipeline.md) - Two-stage pipeline: fast embedding probe (Stage 1) + fine-tuned model (Stage 2)
- [ADR-007: Adapter Format and Deployment](007-adapter-format-and-deployment.md) - PEFT adapter key format conventions for local and Hub loading
- [ADR-008: Isotonic Score Calibration](008-isotonic-score-calibration.md) - Post-hoc per-dimension isotonic regression to correct MSE score compression

## Format

Each ADR includes:
- **Date** - When the decision was made
- **Status** - Proposed, Accepted, Deprecated, Superseded
- **Decision** - The decision in 1-2 sentences
- **Context** - Background and problem being solved
- **Rationale** - Why this decision was made
- **Consequences** - Positive and negative outcomes
- **References** - Related documents, code, data
