---
name: test-verify-memory
description: Test the self-verifying memory protocol against fixture files
disable-model-invocation: false
---

Test the self-verifying memory protocol (curate Step 0, sub-step 5) against fixture files with known expected outcomes.

## Setup

Fixtures live at `.claude/skills/test-verify-memory/test-fixtures/memory/` in this repo. Copy them to a temporary location before running:

```
cp -r .claude/skills/test-verify-memory/test-fixtures/memory/ /tmp/test-verify-memory/
```

If the fixtures are missing, fetch them from the [agent-ready-projects](https://github.com/ducroq/agent-ready-projects) repository under `templates/test-fixtures/memory/`.

## Test protocol

For each `.md` file in the fixture directory, run the curate verification logic from Step 0 sub-step 5:

1. Read the file
2. Detect whether it contains a state claim (trigger words: "shipped," "deployed," "live," "running," "working in production")
3. If it's a state claim, check for a `<!-- verify: ... -->` comment
4. If a verify command exists, run it and record the result
5. Classify the outcome

## Expected results

| Fixture file | Expected claim type | Expected outcome |
|---|---|---|
| `verified-pass.md` | State ("deployed") | **PASS** — verify command runs, outputs PASS |
| `verified-fail.md` | State ("shipped," "running") | **FAIL** — verify command runs, outputs FAIL |
| `verified-error.md` | State ("deployed") | **ERROR** — verify command exits non-zero with no PASS/FAIL output |
| `verified-manual.md` | State ("deployed") | **MANUAL CHECK NEEDED** — has `<!-- verify: manual — ... -->` |
| `unverified-state.md` | State ("deployed," "running") | **UNVERIFIED** — state claim without verify comment |
| `unverified-live.md` | State ("live") | **UNVERIFIED** — exercises the "live" trigger word |
| `unverified-working-in-production.md` | State ("working in production") | **UNVERIFIED** — exercises the multi-word trigger phrase |
| `decision-no-verify.md` | Decision ("chose") | **SKIP** — not a state claim, no verification needed |
| `observation-no-verify.md` | Observation ("during session," "tested") | **SKIP** — not a state claim, no verification needed |
| `pattern-no-verify.md` | Pattern ("always," "when X") | **SKIP** — not a state claim, no verification needed |

## Execution

Process each fixture and compare actual outcome against expected:

```
PASS  verified-pass.md       — expected: PASS, got: ___
PASS  verified-fail.md       — expected: FAIL, got: ___
PASS  verified-error.md      — expected: ERROR, got: ___
PASS  verified-manual.md     — expected: MANUAL CHECK NEEDED, got: ___
PASS  unverified-state.md    — expected: UNVERIFIED, got: ___
PASS  unverified-live.md     — expected: UNVERIFIED, got: ___
PASS  unverified-working-in-production.md — expected: UNVERIFIED, got: ___
PASS  decision-no-verify.md  — expected: SKIP, got: ___
PASS  observation-no-verify.md — expected: SKIP, got: ___
PASS  pattern-no-verify.md   — expected: SKIP, got: ___
```

Replace `PASS` with `FAIL` if the actual outcome doesn't match expected.

## Report

Summarize:
- **Total fixtures**: 10
- **Passed**: N/10
- **Failed**: N/10 (list each with expected vs actual)

If all 10 pass, the curate verification protocol is working correctly for these cases.

If any fail, diagnose:
- **False positive** (flagged a non-state claim as state): the trigger-word detection is too broad
- **False negative** (missed a state claim): the trigger-word detection is too narrow
- **Wrong outcome** (detected the claim but misclassified the verify status): the verify-command parsing needs attention

## Cleanup

```
rm -rf /tmp/test-verify-memory/
```
