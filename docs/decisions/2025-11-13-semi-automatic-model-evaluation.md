# Semi-Automatic Model Evaluation (Post-Training Checklist)

**Date:** 2025-11-13
**Status:** Accepted

## Context

After creating the Model Evaluation Agent template (`docs/agents/templates/model-evaluation-agent.md`), we need to decide how to integrate model evaluation into the training workflow.

**The question:** Should model evaluation be:
1. **Fully automatic** - Built into training script, runs immediately after training
2. **Semi-automatic** - Manual checklist after training, agent does evaluation work
3. **Manual** - User runs all evaluation steps manually

## Decision

**Use semi-automatic approach (Option 2): Post-training checklist with agent-assisted evaluation.**

Training script remains focused on training. After training completes, users follow a documented checklist to:
1. Copy training artifacts from remote
2. Invoke Model Evaluation Agent
3. Review production readiness report
4. Make deployment decision

## Rationale

### Why Not Fully Automatic (Option 1)?

**Cons of automatic evaluation in training script:**
- ❌ **Mixing concerns** - Training script does training, evaluation is separate concern
- ❌ **No review opportunity** - User doesn't see raw results before agent analysis
- ❌ **Inflexible** - Can't skip evaluation for experimental runs
- ❌ **Coupling** - Training script depends on evaluation script and agent template
- ❌ **Silent failures** - If agent fails, training "succeeds" but evaluation missing

**Example problematic flow:**
```python
# train.py (NOT recommended)
train_model()  # Takes 2 hours
save_model()
run_evaluation()  # What if this fails? User comes back to confusion
generate_report()  # What if agent template changes?
```

### Why Semi-Automatic (Option 2)? ✅

**Pros of checklist approach:**
- ✅ **Separation of concerns** - Training does training, evaluation does evaluation
- ✅ **User control** - Can review metrics before invoking agent
- ✅ **Flexible workflow** - Skip evaluation for experiments, run for production candidates
- ✅ **Clear handoff** - Training completes → User reviews → Decides to evaluate
- ✅ **Easy debugging** - Each step is explicit, failures are obvious
- ✅ **Iterative friendly** - Can re-run evaluation without re-training

**Example clear flow:**
```bash
# 1. Training completes (remote GPU)
python -m training.train ...
# Output: "Training complete! Best val MAE: 0.997"

# 2. User reviews results
scp remote:filters/uplifting/v1/training_*.json local/filters/uplifting/v1/

# 3. User decides to evaluate for production
# (skips this step if just experimenting)

# 4. User invokes agent
Task: "Evaluate trained uplifting model using Model Evaluation Agent..."

# 5. Agent generates report
# Output: filters/uplifting/v1/model_evaluation.md
# Decision: DEPLOY / REVIEW / FAIL

# 6. User reads report and decides next action
```

### Why Not Fully Manual (Option 3)?

Manual evaluation (no agent) would require:
- Reading training history JSON manually
- Running test evaluation script manually
- Calculating metrics manually
- Writing report manually
- Making deployment decision manually

**Too error-prone and time-consuming.** Agent automation is valuable here.

## Implementation

### 1. Training Script (No Changes)

Training script remains focused:
```python
# training/train.py stays as-is
# Trains model, saves best checkpoint, prints summary
# Does NOT run evaluation
```

### 2. Post-Training Checklist (Add to README)

Add to `training/README.md`:

```markdown
## After Training Completes

### Step 1: Copy Training Results
```bash
# From local machine
scp user@remote:/path/to/llm-distillery/filters/{filter_name}/v1/training_*.json \
    filters/{filter_name}/v1/
```

### Step 2: Review Training Metrics
```bash
# Check final validation MAE
cat filters/{filter_name}/v1/training_metadata.json | grep best_val_mae

# Review training progression
cat filters/{filter_name}/v1/training_history.json
```

**Quick check:** If val MAE > 1.5, consider retraining before full evaluation.

### Step 3: Run Model Evaluation Agent
```
Task: "Evaluate the trained {filter_name} model using the Model Evaluation Agent
criteria from docs/agents/templates/model-evaluation-agent.md.

Model location: filters/{filter_name}/v1
Test data: datasets/training/{filter_name}/test.jsonl

Run test evaluation and generate production readiness report."
```

**Agent will:**
- Run test set evaluation
- Analyze all metrics against criteria
- Generate report: `filters/{filter_name}/v1/model_evaluation.md`
- Recommend: DEPLOY / REVIEW / FAIL

### Step 4: Review Production Readiness Report
```bash
cat filters/{filter_name}/v1/model_evaluation.md
```

**Decision matrix:**
- ✅ **DEPLOY** → Proceed to deployment (copy model to production)
- ⚠️ **REVIEW** → Discuss trade-offs with team
- ❌ **FAIL** → Retrain with adjusted hyperparameters or more data

### Step 5: Optional - Copy Model to Production
```bash
# Only if evaluation report says DEPLOY
scp -r filters/{filter_name}/v1/model/ production-server:/models/
```
```

### 3. Agent Template (Already Complete)

`docs/agents/templates/model-evaluation-agent.md` is reusable for all filters:
- Generic template with `{filter_name}` placeholder
- User just substitutes actual filter name in invocation
- Agent does all evaluation work automatically

## Consequences

### Positive

- ✅ **Clear separation** - Training and evaluation are distinct, well-defined steps
- ✅ **User awareness** - Users see training results before evaluation
- ✅ **Flexible** - Can skip evaluation for experimental runs
- ✅ **Debuggable** - Each step explicit, easy to retry if failures
- ✅ **Maintainable** - Training script stays simple, evaluation logic in agent template
- ✅ **Reusable** - Same checklist works for all filters (uplifting, tech_deployment, etc.)

### Negative

- ⚠️ **Manual step** - User must remember to run evaluation (not automatic)
- ⚠️ **Documentation dependency** - Users must read README checklist
- ⚠️ **Possible skip** - User might forget evaluation step for production models

### Mitigation

**Add reminder at end of training:**
```python
# training/train.py - at very end
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best validation MAE: {best_val_mae:.4f}")
print(f"\nNext steps:")
print(f"1. Review training results in {output_dir}/")
print(f"2. Run Model Evaluation Agent (see training/README.md)")
print(f"3. Review report: {output_dir}/model_evaluation.md")
print("="*60)
```

## Alternatives Considered

### Alternative 1: Fully Automatic (Rejected)
**Approach:** Training script automatically runs evaluation after training

**Pros:**
- Fully automated workflow
- User can't forget evaluation step

**Cons:**
- Tight coupling (training depends on evaluation)
- No user review between steps
- Hard to skip for experiments
- Silent failures possible

**Decision:** Rejected - Coupling and inflexibility outweigh automation benefits

### Alternative 2: Shell Script Wrapper (Rejected)
**Approach:** `train_and_evaluate.sh` wraps both steps

**Pros:**
- One command runs both
- Still separated internally

**Cons:**
- Extra script to maintain
- Less flexible than checklist
- Doesn't work well with remote training (can't run evaluation on GPU machine)

**Decision:** Rejected - Checklist in README is simpler and more flexible

### Alternative 3: CI/CD Pipeline (Future)
**Approach:** GitHub Actions or similar runs evaluation automatically on push

**Pros:**
- Fully automated for production workflows
- Version controlled
- Can run on schedule

**Cons:**
- Overkill for current scale
- Requires CI/CD infrastructure
- Less interactive

**Decision:** Defer to future - When training becomes regular/scheduled, consider CI/CD automation

## Success Metrics

**This decision is successful if:**
- ✅ Users consistently run Model Evaluation Agent after training
- ✅ No production models deployed without evaluation report
- ✅ Evaluation step takes < 30 minutes (including agent invocation)
- ✅ Users find checklist clear and easy to follow

**Red flags:**
- ❌ Users frequently skip evaluation step
- ❌ Confusion about when to run evaluation
- ❌ Evaluation report not found in filter directories

**If red flags appear:** Consider more automation (Option 1 or CI/CD)

## Implementation Checklist

- [x] Model Evaluation Agent template created
- [x] Agent saves reports to filter directory (portability)
- [x] Add post-training checklist to `training/README.md`
- [x] Add "Next steps" reminder to end of `training/train.py`
- [ ] Test workflow with uplifting filter training

## References

- Model Evaluation Agent template: `docs/agents/templates/model-evaluation-agent.md`
- Training documentation: `training/README.md`
- Related decision: `docs/decisions/2025-11-13-regression-only-student-models.md`

## Discussion

**Key insight:** Training and evaluation serve different purposes and audiences.

**Training:**
- **Purpose:** Optimize model weights
- **Audience:** Data scientist / ML engineer
- **Success:** Low loss, convergence
- **Speed:** As fast as possible (GPU time is expensive)

**Evaluation:**
- **Purpose:** Production readiness decision
- **Audience:** Team / Product owner
- **Success:** Meets business requirements (MAE < 1.0, no overfitting)
- **Speed:** Thorough analysis over speed

**Principle:** Keep tools focused on their primary purpose. Don't merge training and evaluation just for convenience.

**User workflow matters:** Users training models need to review results before committing to production. Forced automation removes this critical review step.
