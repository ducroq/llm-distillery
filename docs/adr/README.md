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

## Format

Each ADR includes:
- **Date** - When the decision was made
- **Status** - Proposed, Accepted, Deprecated, Superseded
- **Decision** - The decision in 1-2 sentences
- **Context** - Background and problem being solved
- **Rationale** - Why this decision was made
- **Consequences** - Positive and negative outcomes
- **References** - Related documents, code, data
