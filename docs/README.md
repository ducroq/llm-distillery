# LLM Distillery Documentation

Complete guides for generating ground truth datasets and training specialized semantic filters.

## 📚 Documentation Index

### Getting Started
- **[Quick Start Guide](guides/getting-started.md)** - First steps with LLM Distillery
- **[Migration Guide](guides/migration-from-nexusmind.md)** - Understanding the migration from NexusMind-Filter
- **[Migration Complete](guides/migration-complete.md)** - Current status and what's ready

### Data Preparation
- **[Data Preparation Guide](guides/data-preparation.md)** - Merge historical databases and create master datasets
- **[Calibration Guide](guides/calibration.md)** - Compare Claude vs Gemini to select best LLM oracle

### Ground Truth Generation
- **[Best Practices](ground_truth_best_practices.md)** - Tips for high-quality labeling *(planned)*
- **[Cost Optimization](cost_optimization.md)** - Minimize API costs *(planned)*
- **[Quality Assurance](quality_assurance.md)** - Validation and calibration *(planned)*

### Semantic Filters
- **[Creating New Filters](creating_new_filters.md)** - How to add your own semantic dimensions
- **[Prompt Engineering Guide](prompt_engineering.md)** - Writing effective evaluation prompts

### Training & Fine-Tuning
- **[Training Pipeline](training_pipeline.md)** - Coming soon
- **[Model Selection](model_selection.md)** - Choosing the right base model
- **[Hyperparameter Tuning](hyperparameter_tuning.md)** - Coming soon

### Deployment
- **[Inference Guide](inference_guide.md)** - Coming soon
- **[Integration with NexusMind-Filter](nexusmind_integration.md)** - Coming soon

### Reference
- **[Repository Organization](ORGANIZATION.md)** - Directory structure and file conventions
- **[API Keys & Secrets](secrets_management.md)** - Managing credentials securely
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[FAQ](faq.md)** - Frequently asked questions

---

## 🎯 Quick Links by Task

### I want to...

**Generate ground truth for a new filter**
1. Read: [Creating New Filters](creating_new_filters.md)
2. Read: [Prompt Engineering Guide](prompt_engineering.md)
3. Follow: [Best Practices](ground_truth_best_practices.md)
4. Run: `python -m ground_truth.batch_labeler --prompt prompts/your_filter.md`

**Optimize my labeling costs**
1. Read: [Cost Optimization](cost_optimization.md)
2. Implement pre-filters
3. Use Gemini Tier 1 instead of Claude

**Validate my ground truth quality**
1. Read: [Quality Assurance](quality_assurance.md)
2. Check tier distributions
3. Review edge cases

**Train a local model**
1. Read: [Training Pipeline](training_pipeline.md) (coming soon)
2. Prepare datasets with train/val/test splits
3. Run training with your config

---

## 📖 Reading Order for New Users

1. **[Quick Start Guide](../GETTING_STARTED.md)** - Understand the overall workflow
2. **[Secrets Management](secrets_management.md)** - Set up API keys
3. **[Ground Truth Best Practices](ground_truth_best_practices.md)** - Learn best practices
4. **[Cost Optimization](cost_optimization.md)** - Save money while generating data
5. **[Creating New Filters](creating_new_filters.md)** - Build your first custom filter

---

## 🎓 Learning Path

### Beginner
- Set up environment and API keys
- Run test labeling (3 articles)
- Understand prompt structure
- Review validation examples

### Intermediate
- Create custom semantic filter
- Generate 1,000 article ground truth
- Validate quality metrics
- Optimize costs with pre-filters

### Advanced
- Generate 50K+ article datasets
- Fine-tune local models
- Deploy to production
- Contribute new filters

---

## 💡 Best Practices Summary

### Ground Truth Generation
✅ Start small (test with 10-100 articles)
✅ Use stratified sampling
✅ Implement pre-filters (50% cost savings)
✅ Monitor quality early and often
✅ Document your prompt rationale

### Cost Management
✅ Use Gemini Tier 1 for bulk labeling ($0.00018/article)
✅ Use Claude for quality validation
✅ Implement resume capability
✅ Batch process to avoid rate limits

### Quality Assurance
✅ Check tier distributions match expectations
✅ Review edge cases manually
✅ Validate against known examples
✅ Monitor for drift over time

---

## 🆘 Getting Help

**Something not working?**
→ Check [Troubleshooting](troubleshooting.md)

**Have a question?**
→ Check [FAQ](faq.md)

**Found a bug?**
→ Report at https://github.com/yourusername/llm-distillery/issues

**Want to contribute?**
→ See [Contributing Guide](contributing.md)
