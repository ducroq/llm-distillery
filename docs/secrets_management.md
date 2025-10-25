# API Keys & Secrets Management

How to securely manage API keys for Claude, Gemini, and GPT-4 in LLM Distillery.

---

## 🔐 Quick Setup

### Method 1: secrets.ini File (Recommended for Local Development)

**Create `secrets.ini` file** in `config/credentials/`:

```bash
# Navigate to project root
cd C:\local_dev\llm-distillery

# Copy example file
cp config/credentials/secrets.ini.example config/credentials/secrets.ini

# Edit secrets.ini with your API keys
# (Use your text editor - DO NOT commit this file!)
```

**`secrets.ini` contents**:
```ini
[api_keys]
# LLM API Keys for ground truth generation
# Get from:
# - Anthropic Claude: https://console.anthropic.com/
# - Google Gemini: https://aistudio.google.com/app/apikey
# - OpenAI GPT-4: https://platform.openai.com/api-keys

anthropic_api_key = sk-ant-api03-your-key-here
gemini_api_key = AIza-your-gemini-key-here
openai_api_key = sk-your-openai-key-here

# Optional: For experiment tracking
wandb_api_key = your-wandb-key-here
```

**How it works**:
- The `SecretsManager` automatically loads `config/credentials/secrets.ini`
- Keys are accessed with `secrets.get_anthropic_key()`, `secrets.get_gemini_key()`, etc.
- `secrets.ini` is in `.gitignore` - never committed to git

### Method 2: Environment Variables (Recommended for CI/CD)

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

### Priority Order

The `SecretsManager` checks for API keys in this order:

1. **Environment variables** (highest priority)
   - `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY`
   - `GEMINI_API_KEY`, `GOOGLE_API_KEY`
   - `OPENAI_API_KEY`
   - `WANDB_API_KEY`

2. **secrets.ini file** (fallback)
   - `config/credentials/secrets.ini`

This allows:
- Local development: Use `secrets.ini`
- CI/CD: Use environment variables
- Override `secrets.ini` with environment variables if needed

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
- Use `secrets.ini` file (in `.gitignore`)
- Rotate keys periodically
- Use separate keys for dev/prod
- Set spending limits in provider dashboards
- Revoke unused keys immediately

### DON'T ❌
- Commit `secrets.ini` to git
- Share keys in chat/email
- Use production keys in examples
- Hardcode keys in scripts
- Use same key across multiple projects

### File Permissions (Linux/Mac)

```bash
# Make secrets.ini readable only by you
chmod 600 config/credentials/secrets.ini

# Verify permissions
ls -la config/credentials/secrets.ini
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
**Solution**: Check `config/credentials/secrets.ini` exists and has `anthropic_api_key = ...` under `[api_keys]` section, or set `ANTHROPIC_API_KEY` environment variable

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
config/credentials/secrets.ini
secrets.ini
```

**Team workflow**:
1. Each team member gets own API keys
2. Each creates own `secrets.ini` file (not committed)
3. Use `secrets.ini.example` as template
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

## 📝 Example secrets.ini File

See `config/credentials/secrets.ini.example` for the complete template.

**Minimal example**:
```ini
[api_keys]
# At least one LLM API key required
anthropic_api_key = sk-ant-api03-your-actual-key-here
gemini_api_key = AIza-your-actual-key-here

# Optional: For experiment tracking
wandb_api_key = your-wandb-key-here
```

**Location**: `config/credentials/secrets.ini`

**Security**:
- This file is in `.gitignore` - it will NEVER be committed
- Each team member creates their own copy from `secrets.ini.example`
- Set file permissions to 600 (Linux/Mac): `chmod 600 config/credentials/secrets.ini`

---

## ✅ Security Checklist

Before running in production:

- [ ] `secrets.ini` file exists and has valid keys
- [ ] `secrets.ini` is in `.gitignore` (verify: `git status` should not show it)
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
