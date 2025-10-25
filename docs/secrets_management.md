# API Keys & Secrets Management

How to securely manage API keys for Claude, Gemini, and GPT-4 in LLM Distillery.

---

## 🔐 Quick Setup

### Method 1: Environment Variables (Recommended)

**Create `.env` file** in project root:

```bash
# Navigate to project root
cd C:\local_dev\llm-distillery

# Copy example file
cp .env.example .env

# Edit .env with your API keys
# (Use your text editor - DO NOT commit this file!)
```

**`.env` contents**:
```bash
# LLM API Keys (choose one or more)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
GOOGLE_API_KEY=AIza-your-gemini-key-here
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your-wandb-key-here
WANDB_PROJECT=llm-distillery

# Optional: Content Aggregator data path
CONTENT_AGGREGATOR_DATA_PATH=../content-aggregator/data/collected
```

**How it works**:
- The `batch_labeler.py` automatically loads `.env` using `python-dotenv`
- Keys are read with `os.getenv('ANTHROPIC_API_KEY')`
- `.env` is in `.gitignore` - never committed to git

### Method 2: System Environment Variables

**Windows (PowerShell)**:
```powershell
# Temporary (current session only)
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Permanent (user-level)
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-your-key-here', 'User')
```

**Linux/Mac (Bash)**:
```bash
# Temporary (current session)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Method 3: Command-Line Argument

```bash
python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source articles.jsonl \
    --api-key "sk-ant-your-key-here"
```

⚠️ **Not recommended**: API key visible in command history

---

## 🔑 Getting API Keys

### Anthropic Claude
1. Visit: https://console.anthropic.com/
2. Sign up / Log in
3. Navigate to: API Keys section
4. Create new key
5. **Copy immediately** (shown only once)
6. **Format**: `sk-ant-api03-...`

**Pricing** (as of 2025):
- Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
- Typical ground truth labeling: ~$0.009/article

### Google Gemini
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Create API key
4. **Enable Cloud Billing** for Tier 1 (150 RPM instead of 2 RPM)
5. **Format**: `AIza...`

**Pricing** (as of 2025):
- Gemini 1.5 Pro: $0.075/MTok input, $0.30/MTok output
- **With Cloud Billing (Tier 1)**: ~$0.00018/article
- **Free tier**: Only 2 RPM (not viable for batch labeling)

**⚠️ IMPORTANT**: Enable Cloud Billing for Tier 1:
1. Visit https://console.cloud.google.com/billing
2. Link billing account to your API project
3. Rate limit automatically upgrades to 150 RPM
4. Pay only for usage (very cheap!)

### OpenAI GPT-4
1. Visit: https://platform.openai.com/api-keys
2. Create account / Log in
3. Create new secret key
4. **Format**: `sk-...`

**Pricing** (as of 2025):
- GPT-4 Turbo: $10/MTok input, $30/MTok output
- Typical ground truth labeling: ~$0.012/article

---

## 🔒 Security Best Practices

### DO ✅
- Use `.env` file (in `.gitignore`)
- Rotate keys periodically
- Use separate keys for dev/prod
- Set spending limits in provider dashboards
- Revoke unused keys immediately

### DON'T ❌
- Commit `.env` to git
- Share keys in chat/email
- Use production keys in examples
- Hardcode keys in scripts
- Use same key across multiple projects

### File Permissions (Linux/Mac)

```bash
# Make .env readable only by you
chmod 600 .env

# Verify permissions
ls -la .env
# Should show: -rw------- (600)
```

---

## 📊 Cost Monitoring

### Check Your Usage

**Anthropic Console**:
- https://console.anthropic.com/settings/usage
- View: API calls, tokens, costs

**Google Cloud Console**:
- https://console.cloud.google.com/billing
- View: Gemini API usage and costs

**OpenAI Dashboard**:
- https://platform.openai.com/usage
- View: Token usage and costs

### Set Spending Limits

**Anthropic**:
- Console → Settings → Billing → Set monthly limit

**Google Cloud**:
- Billing → Budgets & alerts → Create budget

**OpenAI**:
- Settings → Billing → Usage limits

**Recommended limits** for testing:
- Start with $10-20/month
- Increase as needed for production

---

## 🧪 Testing Your Setup

### Verify API Keys Work

```bash
cd C:\local_dev\llm-distillery

# Test with 1 article
python test_batch_labeler.py

python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source datasets/test/test_articles.jsonl \
    --llm claude \
    --batch-size 1 \
    --max-batches 1
```

**Expected**:
- No "API key not found" errors
- Article gets labeled
- Output saved to `datasets/uplifting/labeled_batch_001.jsonl`
- Cost: ~$0.01

### Common Errors

**Error**: `ValueError: Claude API key not found`
**Solution**: Check `.env` file exists and has `ANTHROPIC_API_KEY=...`

**Error**: `AuthenticationError: Invalid API key`
**Solution**: Verify key is correct, not expired

**Error**: `RateLimitError: Too many requests`
**Solution**:
- For Claude: Script already has 1.5s delay (should not happen)
- For Gemini: Enable Cloud Billing for Tier 1 (150 RPM)

---

## 🔄 Key Rotation

**When to rotate**:
- Every 90 days (best practice)
- When team member leaves
- If key might be compromised
- Before open-sourcing code

**How to rotate**:
1. Generate new key in provider console
2. Update `.env` with new key
3. Test with small batch
4. Revoke old key in console

---

## 🏢 Team Collaboration

### Sharing Project (Without Sharing Keys)

**`.gitignore` includes**:
```
.env
.env.local
.env.*.local
```

**Team workflow**:
1. Each team member gets own API keys
2. Each creates own `.env` file (not committed)
3. Use `.env.example` as template
4. Document required keys in README

### CI/CD Secrets

**GitHub Actions**:
```yaml
# .github/workflows/test.yml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Add secrets: Repo Settings → Secrets → Actions → New repository secret

**GitLab CI**:
Settings → CI/CD → Variables → Add variable (Protected, Masked)

---

## 📝 Example `.env` File

```bash
# =============================================================================
# LLM Distillery - API Keys & Configuration
# =============================================================================
#
# IMPORTANT: This file contains secrets - DO NOT commit to git!
# Copy from .env.example and add your actual keys below.
#

# -----------------------------------------------------------------------------
# LLM API Keys (at least one required)
# -----------------------------------------------------------------------------

# Anthropic Claude (recommended for high quality)
# Get from: https://console.anthropic.com/
# Cost: ~$0.009/article for ground truth generation
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Google Gemini (recommended for low cost)
# Get from: https://aistudio.google.com/app/apikey
# IMPORTANT: Enable Cloud Billing for Tier 1 (150 RPM)
# Cost: ~$0.00018/article (50x cheaper than Claude)
GOOGLE_API_KEY=AIza-your-gemini-key-here

# OpenAI GPT-4 (optional)
# Get from: https://platform.openai.com/api-keys
# Cost: ~$0.012/article
# OPENAI_API_KEY=sk-your-openai-key-here

# -----------------------------------------------------------------------------
# Optional: Experiment Tracking
# -----------------------------------------------------------------------------

# Weights & Biases (for tracking training experiments)
# Get from: https://wandb.ai/authorize
# WANDB_API_KEY=your-wandb-key-here
# WANDB_PROJECT=llm-distillery

# -----------------------------------------------------------------------------
# Optional: Data Source Configuration
# -----------------------------------------------------------------------------

# Path to Content Aggregator data
CONTENT_AGGREGATOR_DATA_PATH=../content-aggregator/data/collected

# -----------------------------------------------------------------------------
# Optional: Model Training Configuration
# -----------------------------------------------------------------------------

# GPU configuration
# CUDA_VISIBLE_DEVICES=0
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Inference settings
# INFERENCE_BATCH_SIZE=32
# INFERENCE_MAX_LENGTH=512
```

---

## ✅ Security Checklist

Before running in production:

- [ ] `.env` file exists and has valid keys
- [ ] `.env` is in `.gitignore` (verify: `git status` should not show it)
- [ ] Spending limits set in provider dashboards
- [ ] Keys are unique per environment (dev/staging/prod)
- [ ] File permissions set to 600 (Linux/Mac)
- [ ] Team members have their own keys (not shared)
- [ ] Old/unused keys have been revoked

---

## 📞 Support

**API key issues?**
- Anthropic: https://support.anthropic.com
- Google Cloud: https://cloud.google.com/support
- OpenAI: https://help.openai.com

**LLM Distillery issues?**
- Check: [Troubleshooting](troubleshooting.md)
- Report: https://github.com/yourusername/llm-distillery/issues
