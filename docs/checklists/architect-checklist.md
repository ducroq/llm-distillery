# Architect Checklist

Definition-of-done for the design phase. Use before moving to implementation.

## Context & Requirements

- [ ] Problem statement is written down (not just discussed)
- [ ] Constraints identified — what can't change, what's non-negotiable
- [ ] Existing patterns reviewed — does the codebase already solve something similar?
- [ ] Scope is bounded — "what we're NOT doing" is explicit

## Design Decisions

- [ ] ADR created for any non-obvious choice between alternatives (see `docs/adr/`)
- [ ] Trade-offs documented — what you're giving up and why that's acceptable
- [ ] Key interfaces defined — how components talk to each other
- [ ] Oracle prompt design reviewed if new filter (ADR-010: use belonging v1 as template)

## Filter-Specific (if designing a new filter)

- [ ] Dimensions defined with scoring rubric (0-10 scale, ADR-001)
- [ ] Needle-in-haystack or broad distribution? Determines screening strategy (ADR-011)
- [ ] Training data source identified — which corpus, estimated volume needed
- [ ] Deployment target clear — ovr.news tab, standalone product, or both

## Handoff

- [ ] Implementation can start from this design without a follow-up conversation
- [ ] Test strategy is clear — MAE target, validation set size, evaluation approach
- [ ] No open questions marked "TBD" — resolve or explicitly defer with a reason
