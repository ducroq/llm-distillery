# Hugging Face Model Upload Guide

Quick guide to upload your trained models to Hugging Face Hub.

## One-Time Setup (on GPU machine)

### 1. Create Hugging Face Account

Go to https://huggingface.co/join and create a free account.

### 2. Get Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "llm-distillery-upload"
4. Type: **Write** access
5. Copy the token (starts with `hf_...`)

### 3. Install Hugging Face Hub

```bash
# On your GPU machine
pip install huggingface_hub
```

### 4. Login

```bash
hf auth login
# Paste your token when prompted
# Press Y when asked "Add token as git credential?"
```

**Note**: The old command `huggingface-cli login` still works but is deprecated.

**Optional: Set git credential helper** (recommended to avoid re-authentication):
```bash
git config --global credential.helper store
```

### 5. Verify Your Username

Check your Hugging Face username (you'll need this for upload):

```bash
hf whoami
```

This shows your username (e.g., `jeergrvgreg`). Use this in the repo name when uploading.

## Uploading a Model

After training completes, upload with one command:

```bash
# Use YOUR Hugging Face username (check with 'hf whoami')
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name YOUR_USERNAME/uplifting-filter-v1 \
    --private
```

**Important:** Replace `YOUR_USERNAME` with your actual Hugging Face username from `hf whoami`.

**Arguments:**
- `--filter`: Path to filter directory containing trained model
- `--repo-name`: `username/model-name` format (use YOUR username, not someone else's)
- `--private`: Keeps model private (remove flag to make public)

**Example:**
```bash
# If your username is 'jeergrvgreg'
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name jeergrvgreg/uplifting-filter-v1 \
    --private
```

### Example: Full Training + Upload Workflow

```bash
# 1. Train the model (saves to filter directory automatically)
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 10 \
    --batch-size 4

# 2. Upload to Hugging Face (private)
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name YOUR_USERNAME/uplifting-filter-v1 \
    --private
```

## What Gets Uploaded

The script uploads:
- ✓ Model weights (`pytorch_model.bin` or `model.safetensors`)
- ✓ Model config (`config.json`)
- ✓ Tokenizer files
- ✓ Training metadata (`training_metadata.json`, `training_history.json`)
- ✓ Auto-generated model card (README.md) with:
  - Model description
  - Training details
  - Performance metrics
  - Usage examples
  - Limitations

**Total size**: ~1GB for 0.5B model

## Using Your Model

Once uploaded, you can use it anywhere:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from Hugging Face
model_name = "your-username/uplifting-filter-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Score an article
article = {
    "title": "Example Article",
    "content": "Article content here..."
}

text = f"{article['title']}\n\n{article['content']}"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].numpy()

print(f"Scores: {scores}")
```

## Privacy Settings

### Default: Private

Models are **private by default**:
- Only you can see and use it
- Perfect for testing
- Free (no limits)

### Making It Public

To share your model:

1. Go to https://huggingface.co/your-username/your-model
2. Click "Settings"
3. Scroll to "Change repository visibility"
4. Choose public

Or upload without `--private` flag.

### Choose a License

If making public, add a license:
- **MIT**: Anyone can use (most permissive)
- **Apache 2.0**: Anyone can use, must credit you
- **CC BY-NC**: Non-commercial use only
- **Custom**: Your own terms

Edit the model card on Hugging Face to change license.

## Downloading from Hugging Face

To use your model on another machine:

```bash
# The model auto-downloads when you use it
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('your-username/your-model')
"
```

Or download manually:

```bash
huggingface-cli download your-username/your-model
```

## Troubleshooting

### "Error: huggingface_hub not installed"

```bash
pip install huggingface_hub
```

### "Error: Hugging Face token required"

You need to authenticate first:
```bash
hf auth login
# Paste your token when prompted
```

The script automatically uses tokens saved by `hf auth login`.

Alternative methods:
```bash
# Set environment variable
export HF_TOKEN="hf_your_token_here"

# Or pass token directly
python -m training.upload_to_huggingface --token "hf_your_token_here" ...
```

### "403 Forbidden: You don't have the rights to create a model under the namespace"

**Problem:** You're using the wrong username in the repo name.

**Solution:** Check your username and use it:
```bash
hf whoami  # Shows your username
python -m training.upload_to_huggingface \
    --repo-name YOUR_USERNAME/model-name \  # Use YOUR username
    ...
```

### "Error: Repository already exists"

The script handles this automatically with `exist_ok=True`. It will update the existing repo.

### Upload is slow

The 0.5B model is ~1GB. On slow connections:
- Use `--private` (fewer metadata uploads)
- Upload only once, update rarely
- Consider compressing with safetensors (already used)

## Best Practices

1. **Start private**: Test your model works before making public
2. **Use semantic versioning**: `filter-name-v1`, `filter-name-v2`, etc.
3. **Document changes**: Update model card when retraining
4. **Test before sharing**: Run inference locally first
5. **Add limitations**: Be honest about model weaknesses in card

## Cost

**Hugging Face Hub is FREE for:**
- ✓ Unlimited private repositories
- ✓ Unlimited public repositories
- ✓ Unlimited downloads
- ✓ Basic inference API

**Pro tier ($9/month) adds:**
- Faster inference API
- More compute for Spaces
- Early access to features
- (Not needed for basic model hosting)

## Next Steps

After uploading:

1. Test loading model from Hugging Face
2. Update model card with usage examples
3. Share with team (if private, add collaborators)
4. Deploy inference API (see `inference/` directory)
5. Monitor usage and feedback
