# Qwen 2.5 Fine-tuning Guide for Multi-Agent Classification

**Version:** 1.0  
**Last Updated:** October 29, 2025  
**Hardware:** NVIDIA RTX 4080 16GB  
**Use Case:** Multi-agent semantic classification for RSS feed screening

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Decision](#architecture-decision)
3. [Hardware Requirements](#hardware-requirements)
4. [Environment Setup](#environment-setup)
5. [Data Preparation](#data-preparation)
6. [Fine-tuning Process](#fine-tuning-process)
7. [Testing & Validation](#testing--validation)
8. [Deployment Strategy](#deployment-strategy)
9. [Troubleshooting](#troubleshooting)
10. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

### Project Goals

- Screen 5,000 RSS articles daily across multiple semantic dimensions
- Run multiple classification agents in parallel
- Maintain high quality on multilingual content (English, Dutch, French, German, etc.)
- Keep costs low with local inference

### Why This Approach?

**Separate Fine-tuned Models (Selected):**
- ✅ Each agent specialized for its specific task
- ✅ Higher accuracy per dimension
- ✅ Independent updates and debugging
- ✅ No task interference
- ✅ Full FP16 quality with sequential loading

**Alternative (Multi-task):**
- Single model handling all tasks
- More efficient VRAM usage
- Better for highly correlated tasks

**Our choice:** Separate models because semantic dimensions are orthogonal (uplifting ≠ sustainability)

---

## Architecture Decision

### Sequential vs Parallel Processing

**✅ SELECTED: Sequential Processing**

```
Process Flow:
1. Load Agent 1 (Uplifting) → Score 5,000 articles → Unload
2. Load Agent 2 (Sustainability) → Score 5,000 articles → Unload
3. Combine results → Save to database
```

**Performance:**
- Time per article: ~4-5 seconds per agent
- Total time: ~14 hours for 5,000 articles (overnight batch)
- VRAM usage: 7-8GB (plenty of headroom)
- Quality: Full FP16 precision

**Why Sequential?**
- ✅ Higher quality models (FP16 vs INT8)
- ✅ Simpler code and debugging
- ✅ VRAM headroom for experimentation
- ✅ Fast enough for batch processing
- ✅ Can upgrade to Qwen 14B per agent if needed

---

## Hardware Requirements

### Your Hardware (Confirmed)

```
GPU:        RTX 4080 16GB
CPU:        AMD Ryzen 7 7700
RAM:        50GB DDR5
Storage:    CT2000P3PSSD8 (2TB NVMe)
```

### Minimum Requirements

**For Fine-tuning:**
- GPU: 16GB VRAM minimum (24GB recommended)
- RAM: 32GB system RAM
- Storage: 100GB free space for models + data

**For Inference:**
- GPU: 8GB VRAM per model (sequential)
- RAM: 16GB system RAM
- Storage: 50GB for models

### VRAM Usage Breakdown

| Model | Precision | VRAM | Speed (tok/s) | Quality |
|-------|-----------|------|---------------|---------|
| Qwen 2.5 7B | FP16 | 14GB | 150-200 | 100% |
| Qwen 2.5 7B | INT8 | 7GB | 180-220 | 95-98% |
| Qwen 2.5 7B | INT4 | 4GB | 200-250 | 85-90% |
| Qwen 2.5 14B | FP16 | 28GB | 100-150 | 105% |
| Qwen 2.5 14B | INT8 | 14GB | 120-180 | 100-103% |

**Recommendation:** FP16 for quality, sequential loading for 7B models

---

## Environment Setup

### Step 1: Verify CUDA in LXC Container

```bash
# Access your LXC container
lxc exec your-container-name -- bash

# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA GeForce RTX 4080   Off  | 00000000:01:00.0  On |                  N/A |
```

**If nvidia-smi doesn't work:**
```bash
# On LXC host, configure GPU passthrough:
lxc config device add your-container gpu gpu gputype=physical
lxc restart your-container
```

### Step 2: Install System Dependencies

```bash
# Update system
apt update && apt upgrade -y

# Install Python 3.10+ and essentials
apt install -y python3.10 python3.10-venv python3-pip git wget curl

# Install build tools
apt install -y build-essential

# Verify Python version
python3 --version  # Should be 3.10 or higher
```

### Step 3: Create Python Environment

```bash
# Create project directory
mkdir -p /root/qwen-finetuning
cd /root/qwen-finetuning

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 4: Install ML Dependencies

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (optimized for Qwen fine-tuning)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional ML libraries
pip install datasets transformers trl peft accelerate bitsandbytes

# Verify CUDA installation
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA available: True
# Device: NVIDIA GeForce RTX 4080
```

### Step 5: Download Base Model

```bash
# Create model directory
mkdir -p models

# Download Qwen 2.5 7B Instruct
python3 << 'EOF'
from huggingface_hub import snapshot_download

print("Downloading Qwen 2.5 7B Instruct model...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/qwen2.5-7b-instruct",
    local_dir_use_symlinks=False
)
print("✅ Model downloaded successfully!")
EOF
```

**Download time:** 10-20 minutes (14GB)

**Quick test option:**
```bash
# Start with smaller 3B model for faster testing
model_id = "Qwen/Qwen2.5-3B-Instruct"  # Only 6GB
```

---

## Data Preparation

### Ground Truth Generation

You're using Gemini Flash to generate ground truth labels. Here's the recommended format:

**Input format (from Gemini Flash):**
```json
{
  "id": "article_123",
  "title": "Community builds water pipeline in rural Kenya",
  "text": "Full article text here...",
  "source": "rss_feed_name",
  "published_date": "2025-10-29",
  "uplifting_scores": {
    "content_type": "solutions_story",
    "agency": 9,
    "progress": 9,
    "collective_benefit": 10,
    "connection": 8,
    "innovation": 7,
    "justice": 6,
    "resilience": 8,
    "wonder": 5,
    "reasoning": "Community took effective action...",
    "key_markers": ["community_action", "water_access"]
  },
  "sustainability_scores": {
    "environmental_impact": 8,
    "resource_efficiency": 9,
    "circular_economy": 5,
    "climate_action": 6,
    "biodiversity": 4,
    "reasoning": "Sustainable water infrastructure...",
    "key_markers": ["water_conservation", "infrastructure"]
  }
}
```

### Data Conversion Script

```python
# create_training_data.py
import json
from pathlib import Path
from typing import Literal

def convert_gemini_to_training_format(
    input_file: str,
    output_file: str, 
    agent_type: Literal["uplifting", "sustainability"],
    system_prompt_file: str
):
    """
    Convert Gemini-labeled data to Unsloth training format
    
    Args:
        input_file: Path to JSONL file with Gemini labels
        output_file: Path for output training data
        agent_type: Which agent to prepare data for
        system_prompt_file: Path to markdown file with system prompt
    """
    
    # Load system prompt from markdown file
    with open(system_prompt_file, 'r') as f:
        system_prompt = f.read()
    
    # Determine which scores to use
    scores_key = f"{agent_type}_scores"
    
    training_data = []
    skipped = 0
    
    print(f"Converting {agent_type} data from {input_file}...")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                
                # Skip if scores are missing
                if scores_key not in item:
                    skipped += 1
                    continue
                
                # Create training example in chat format
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user", 
                            "content": f"ARTICLE:\nTitle: {item.get('title', 'No title')}\nText: {item['text']}"
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(item[scores_key], indent=2)
                        }
                    ]
                }
                training_data.append(example)
                
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Invalid JSON on line {line_num}")
                skipped += 1
                continue
            except KeyError as e:
                print(f"⚠️  Warning: Missing key {e} on line {line_num}")
                skipped += 1
                continue
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Created {len(training_data)} training examples")
    print(f"⚠️  Skipped {skipped} invalid entries")
    print(f"📁 Saved to {output_file}")
    
    return len(training_data)

def split_train_val(
    input_file: str,
    train_file: str,
    val_file: str,
    val_split: float = 0.1
):
    """Split data into train and validation sets"""
    
    import random
    
    # Load all examples
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    # Shuffle
    random.seed(42)
    random.shuffle(examples)
    
    # Split
    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Save splits
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"📊 Train: {len(train_examples)} examples")
    print(f"📊 Val: {len(val_examples)} examples")

# Usage
if __name__ == "__main__":
    # Convert uplifting data
    convert_gemini_to_training_format(
        input_file="gemini_labels.jsonl",
        output_file="uplifting_full.jsonl",
        agent_type="uplifting",
        system_prompt_file="uplifting.md"
    )
    
    # Split into train/val
    split_train_val(
        input_file="uplifting_full.jsonl",
        train_file="uplifting_train.jsonl",
        val_file="uplifting_val.jsonl",
        val_split=0.1
    )
    
    # Repeat for sustainability
    convert_gemini_to_training_format(
        input_file="gemini_labels.jsonl",
        output_file="sustainability_full.jsonl",
        agent_type="sustainability",
        system_prompt_file="sustainability.md"
    )
    
    split_train_val(
        input_file="sustainability_full.jsonl",
        train_file="sustainability_train.jsonl",
        val_file="sustainability_val.jsonl",
        val_split=0.1
    )
```

**Run the conversion:**
```bash
python3 create_training_data.py
```

### Data Quality Checks

```python
# check_data_quality.py
import json
from collections import Counter

def analyze_training_data(file_path: str):
    """Analyze training data quality"""
    
    print(f"Analyzing {file_path}...\n")
    
    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    # Basic stats
    print(f"📊 Total examples: {len(examples)}")
    
    # Check token lengths
    lengths = []
    for ex in examples:
        user_msg = ex['messages'][1]['content']
        assistant_msg = ex['messages'][2]['content']
        total_len = len(user_msg) + len(assistant_msg)
        lengths.append(total_len)
    
    print(f"📏 Average example length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"📏 Min length: {min(lengths)} chars")
    print(f"📏 Max length: {max(lengths)} chars")
    
    # Check score distributions (for uplifting)
    if 'uplifting' in file_path:
        scores = []
        for ex in examples:
            try:
                output = json.loads(ex['messages'][2]['content'])
                if 'agency' in output:
                    scores.append(output['agency'])
            except:
                pass
        
        if scores:
            print(f"\n📈 Agency score distribution:")
            for score, count in sorted(Counter(scores).items()):
                print(f"   Score {score}: {count} examples")
    
    print()

if __name__ == "__main__":
    analyze_training_data("uplifting_train.jsonl")
    analyze_training_data("sustainability_train.jsonl")
```

---

## Fine-tuning Process

### Fine-tuning Script

```python
# finetune_qwen.py
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import json

def finetune_agent(
    agent_name: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    base_model_path: str = "./models/qwen2.5-7b-instruct",
    max_seq_length: int = 2048,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    lora_rank: int = 16,
):
    """
    Fine-tune Qwen 2.5 7B for your classification agent
    
    Args:
        agent_name: Name of the agent (for logging)
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file
        output_dir: Where to save the fine-tuned model
        base_model_path: Path to base Qwen model
        max_seq_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for optimization
        lora_rank: LoRA rank (higher = more parameters, slower)
    """
    
    print(f"\n{'='*70}")
    print(f"Starting fine-tuning for {agent_name.upper()} agent")
    print(f"{'='*70}\n")
    
    # 1. Load base model with Unsloth optimization
    print("📥 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_path,
        max_seq_length = max_seq_length,
        dtype = None,  # Auto-detect (FP16 for RTX 4080)
        load_in_4bit = False,  # Use FP16 for quality
    )
    print("✅ Base model loaded\n")
    
    # 2. Configure LoRA for efficient fine-tuning
    print("🔧 Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,  # LoRA rank
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = lora_rank,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )
    print("✅ LoRA configured\n")
    
    # 3. Load training and validation data
    print(f"📚 Loading training data from {train_file}...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    print(f"✅ Loaded {len(train_dataset)} training examples\n")
    
    print(f"📚 Loading validation data from {val_file}...")
    val_dataset = load_dataset("json", data_files=val_file, split="train")
    print(f"✅ Loaded {len(val_dataset)} validation examples\n")
    
    # 4. Configure training arguments
    print("⚙️  Configuring training parameters...")
    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = 4,  # Effective batch size = batch_size * 4
        warmup_steps = 50,
        num_train_epochs = num_epochs,
        learning_rate = learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 2,  # Keep only 2 best checkpoints
        load_best_model_at_end = True,
        optim = "adamw_8bit",  # Memory-efficient optimizer
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 42,
        report_to = "none",  # Disable wandb/tensorboard
    )
    print("✅ Training configured\n")
    
    # 5. Create trainer
    print("👨‍🏫 Creating trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "messages",
        max_seq_length = max_seq_length,
        args = training_args,
    )
    print("✅ Trainer created\n")
    
    # 6. Train!
    print(f"🚀 Starting training ({num_epochs} epochs)...")
    print(f"{'='*70}\n")
    
    trainer.train()
    
    print(f"\n{'='*70}")
    print("✅ Training complete!")
    print(f"{'='*70}\n")
    
    # 7. Save final model
    print(f"💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        "agent_name": agent_name,
        "base_model": base_model_path,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Model saved to: {output_dir}")
    print(f"✅ Training info saved to: {output_dir}/training_info.json\n")
    
    return trainer

# Main execution
if __name__ == "__main__":
    import sys
    
    # Fine-tune uplifting agent
    print("\n" + "="*70)
    print("PHASE 1: UPLIFTING AGENT")
    print("="*70)
    
    try:
        trainer_1 = finetune_agent(
            agent_name="uplifting",
            train_file="uplifting_train.jsonl",
            val_file="uplifting_val.jsonl",
            output_dir="./models/qwen-uplifting-7b",
            num_epochs=3,
        )
        print("✅ Uplifting agent training successful!\n")
    except Exception as e:
        print(f"❌ Error training uplifting agent: {e}")
        sys.exit(1)
    
    # Fine-tune sustainability agent
    print("\n" + "="*70)
    print("PHASE 2: SUSTAINABILITY AGENT")
    print("="*70)
    
    try:
        trainer_2 = finetune_agent(
            agent_name="sustainability",
            train_file="sustainability_train.jsonl",
            val_file="sustainability_val.jsonl",
            output_dir="./models/qwen-sustainability-7b",
            num_epochs=3,
        )
        print("✅ Sustainability agent training successful!\n")
    except Exception as e:
        print(f"❌ Error training sustainability agent: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*70)
```

### Run Fine-tuning

```bash
# Activate environment
cd /root/qwen-finetuning
source venv/bin/activate

# Monitor GPU in another terminal
watch -n 1 nvidia-smi

# Start fine-tuning (runs both agents sequentially)
python3 finetune_qwen.py 2>&1 | tee training.log

# Expected output:
# ==================================================================
# Starting fine-tuning for UPLIFTING agent
# ==================================================================
# 
# 📥 Loading base model...
# ✅ Base model loaded
# 
# 🔧 Configuring LoRA adapters...
# ✅ LoRA configured
# 
# 📚 Loading training data from uplifting_train.jsonl...
# ✅ Loaded 900 training examples
# ...
```

### Training Time Estimates

**On RTX 4080 16GB:**

| Training Set Size | Epochs | Batch Size | Time per Agent | Total Time |
|-------------------|--------|------------|----------------|------------|
| 500 examples | 3 | 2 | 2-3 hours | 4-6 hours |
| 1,000 examples | 3 | 2 | 3-4 hours | 6-8 hours |
| 2,000 examples | 3 | 2 | 6-8 hours | 12-16 hours |
| 5,000 examples | 3 | 2 | 15-20 hours | 30-40 hours |

**Recommendation:** Start with 1,000-2,000 examples per agent for good quality/time balance.

---

## Testing & Validation

### Test Script

```python
# test_finetuned_model.py
from unsloth import FastLanguageModel
import json
import time

def load_model(model_path: str):
    """Load a fine-tuned model"""
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode
    print("✅ Model loaded\n")
    return model, tokenizer

def test_single_article(
    model,
    tokenizer,
    article_title: str,
    article_text: str,
    system_prompt: str
):
    """Test model on a single article"""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"ARTICLE:\nTitle: {article_title}\nText: {article_text}"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate
    start_time = time.time()
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    elapsed = time.time() - start_time
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = result.split("assistant")[-1].strip()
    
    return assistant_response, elapsed

def run_validation_set(
    model,
    tokenizer,
    val_file: str,
    system_prompt: str,
    num_examples: int = 20
):
    """Test model on validation set"""
    
    print(f"Testing on {num_examples} examples from {val_file}...\n")
    
    with open(val_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    examples = examples[:num_examples]
    
    results = []
    total_time = 0
    
    for i, example in enumerate(examples, 1):
        user_msg = example['messages'][1]['content']
        expected_output = example['messages'][2]['content']
        
        # Extract title and text from user message
        lines = user_msg.split('\n')
        title = lines[1].replace("Title: ", "")
        text = '\n'.join(lines[3:])
        
        print(f"Example {i}/{len(examples)}: {title[:50]}...")
        
        # Run inference
        prediction, elapsed = test_single_article(
            model, tokenizer, title, text, system_prompt
        )
        
        total_time += elapsed
        
        # Try to parse as JSON
        try:
            pred_json = json.loads(prediction)
            exp_json = json.loads(expected_output)
            
            results.append({
                "title": title,
                "expected": exp_json,
                "predicted": pred_json,
                "time": elapsed
            })
            
            print(f"✅ Parsed successfully ({elapsed:.2f}s)\n")
            
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse JSON ({elapsed:.2f}s)\n")
            results.append({
                "title": title,
                "expected": expected_output,
                "predicted": prediction,
                "time": elapsed,
                "error": "json_parse_failed"
            })
    
    avg_time = total_time / len(examples)
    print(f"\n{'='*70}")
    print(f"Average inference time: {avg_time:.2f}s per article")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'='*70}\n")
    
    return results

# Example usage
if __name__ == "__main__":
    
    # Test uplifting agent
    print("="*70)
    print("TESTING UPLIFTING AGENT")
    print("="*70 + "\n")
    
    model, tokenizer = load_model("./models/qwen-uplifting-7b")
    
    with open("uplifting.md", 'r') as f:
        system_prompt = f.read()
    
    # Test a single example
    test_article = """
    Community members in rural Kenya built a 15km water pipeline 
    after government funding was denied. The project now serves 
    5,000 people with clean water. Local women led the effort, 
    organizing work teams and teaching maintenance skills.
    """
    
    print("Testing single article...")
    result, elapsed = test_single_article(
        model, tokenizer,
        "Community builds water pipeline",
        test_article,
        system_prompt
    )
    print(f"\nResult ({elapsed:.2f}s):")
    print(result)
    print()
    
    # Test on validation set
    results = run_validation_set(
        model, tokenizer,
        "uplifting_val.jsonl",
        system_prompt,
        num_examples=20
    )
    
    # Save results
    with open("uplifting_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to uplifting_validation_results.json")
```

### Compare with Gemini Baseline

```python
# compare_with_baseline.py
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compare_scores(predictions_file: str, baseline_file: str):
    """Compare fine-tuned model with Gemini baseline"""
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Extract score dimensions
    dimensions = ['agency', 'progress', 'collective_benefit', 'connection',
                  'innovation', 'justice', 'resilience', 'wonder']
    
    for dim in dimensions:
        pred_scores = []
        true_scores = []
        
        for result in predictions:
            if 'error' not in result:
                pred_scores.append(result['predicted'].get(dim, 0))
                true_scores.append(result['expected'].get(dim, 0))
        
        if pred_scores:
            mae = mean_absolute_error(true_scores, pred_scores)
            rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
            
            print(f"{dim:20s} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    
    # Calculate overall accuracy
    exact_matches = sum(1 for r in predictions if r.get('predicted') == r.get('expected'))
    accuracy = exact_matches / len(predictions) * 100
    
    print(f"\nExact match accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    compare_scores("uplifting_validation_results.json", "uplifting_val.jsonl")
```

---

## Deployment Strategy

### Sequential Processing Architecture

```python
# sequential_processor.py
from unsloth import FastLanguageModel
import torch
import json
from typing import List, Dict
from dataclasses import dataclass
import time

@dataclass
class ProcessingResult:
    article_id: str
    title: str
    uplifting_scores: Dict
    sustainability_scores: Dict
    processing_time: float

class SequentialMultiAgentProcessor:
    """
    Process articles sequentially through multiple agents
    Each agent is loaded, processes all articles, then unloaded
    """
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.models_config = {
            "uplifting": {
                "path": "./models/qwen-uplifting-7b",
                "prompt_file": "uplifting.md",
                "precision": "fp16",
                "batch_size": 8,
            },
            "sustainability": {
                "path": "./models/qwen-sustainability-7b",
                "prompt_file": "sustainability.md",
                "precision": "fp16",
                "batch_size": 8,
            }
        }
    
    def load_model(self, agent_name: str):
        """Load a specific agent model"""
        config = self.models_config[agent_name]
        
        print(f"\n{'='*70}")
        print(f"Loading {agent_name.upper()} agent...")
        print(f"{'='*70}\n")
        
        self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["path"],
            max_seq_length=2048,
            dtype=None,  # FP16
            load_in_4bit=False,
        )
        
        FastLanguageModel.for_inference(self.current_model)
        
        # Load system prompt
        with open(config["prompt_file"], 'r') as f:
            self.current_system_prompt = f.read()
        
        print(f"✅ {agent_name} agent loaded\n")
    
    def unload_model(self):
        """Free VRAM by unloading current model"""
        del self.current_model
        del self.current_tokenizer
        torch.cuda.empty_cache()
        print("✅ Model unloaded, VRAM freed\n")
    
    def score_batch(self, articles: List[Dict], batch_size: int = 8) -> List[Dict]:
        """Score a batch of articles with current model"""
        
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            # Prepare prompts
            prompts = []
            for article in batch:
                messages = [
                    {"role": "system", "content": self.current_system_prompt},
                    {"role": "user", "content": f"ARTICLE:\nTitle: {article['title']}\nText: {article['text']}"}
                ]
                prompt = self.current_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
            
            # Tokenize
            inputs = self.current_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to("cuda")
            
            # Generate
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )
            
            # Decode
            for j, output in enumerate(outputs):
                result_text = self.current_tokenizer.decode(output, skip_special_tokens=True)
                assistant_response = result_text.split("assistant")[-1].strip()
                
                try:
                    scores = json.loads(assistant_response)
                    results.append({
                        "article_id": batch[j]["id"],
                        "scores": scores
                    })
                except json.JSONDecodeError:
                    print(f"⚠️  Failed to parse JSON for article {batch[j]['id']}")
                    results.append({
                        "article_id": batch[j]["id"],
                        "scores": None,
                        "error": "json_parse_failed"
                    })
            
            print(f"Processed batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}")
        
        return results
    
    def process_daily_batch(self, articles: List[Dict]) -> List[ProcessingResult]:
        """
        Process all articles through all agents sequentially
        
        Args:
            articles: List of dicts with keys: id, title, text
            
        Returns:
            List of ProcessingResult objects with scores from all agents
        """
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Starting sequential multi-agent processing")
        print(f"Total articles: {len(articles)}")
        print(f"{'='*70}\n")
        
        # Initialize results storage
        all_results = {article["id"]: {"id": article["id"], "title": article["title"]} 
                       for article in articles}
        
        # Process through each agent
        for agent_name in self.models_config.keys():
            agent_start = time.time()
            
            # Load agent
            self.load_model(agent_name)
            
            # Score all articles
            print(f"Processing {len(articles)} articles through {agent_name} agent...")
            scores = self.score_batch(articles, batch_size=self.models_config[agent_name]["batch_size"])
            
            # Store results
            for score_result in scores:
                article_id = score_result["article_id"]
                all_results[article_id][f"{agent_name}_scores"] = score_result["scores"]
            
            agent_elapsed = time.time() - agent_start
            print(f"✅ {agent_name} agent complete in {agent_elapsed/60:.1f} minutes\n")
            
            # Unload agent
            self.unload_model()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"All agents complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per article: {total_time/len(articles):.2f} seconds")
        print(f"{'='*70}\n")
        
        return list(all_results.values())

# Usage example
if __name__ == "__main__":
    
    # Load articles from RSS feeds
    articles = [
        {
            "id": "article_1",
            "title": "Community builds water pipeline",
            "text": "Full article text here..."
        },
        # ... more articles
    ]
    
    # Process
    processor = SequentialMultiAgentProcessor()
    results = processor.process_daily_batch(articles)
    
    # Save results
    with open("daily_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to daily_results.json")
```

### Scheduled Batch Processing

```bash
# setup_cron.sh
#!/bin/bash

# Add to crontab for daily processing at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * cd /root/qwen-finetuning && /root/qwen-finetuning/venv/bin/python3 daily_processing.py >> /var/log/article-processing.log 2>&1") | crontab -

echo "✅ Cron job installed: Daily processing at 2 AM"
```

```python
# daily_processing.py
from sequential_processor import SequentialMultiAgentProcessor
import json
from datetime import datetime

def fetch_daily_articles():
    """Fetch new articles from RSS feeds"""
    # Your RSS fetching logic here
    # Return list of articles
    pass

def save_results_to_database(results):
    """Save processed results to your database"""
    # Your database logic here
    pass

def main():
    print(f"\n{'='*70}")
    print(f"Daily Processing Job Started: {datetime.now()}")
    print(f"{'='*70}\n")
    
    # Fetch articles
    print("Fetching articles from RSS feeds...")
    articles = fetch_daily_articles()
    print(f"✅ Fetched {len(articles)} articles\n")
    
    # Process
    processor = SequentialMultiAgentProcessor()
    results = processor.process_daily_batch(articles)
    
    # Save
    print("Saving results to database...")
    save_results_to_database(results)
    print("✅ Results saved\n")
    
    print(f"{'='*70}")
    print(f"Daily Processing Job Complete: {datetime.now()}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```python
# In finetune_qwen.py, reduce batch size:
per_device_train_batch_size = 1  # Instead of 2
gradient_accumulation_steps = 8  # Instead of 4

# Or enable gradient checkpointing:
use_gradient_checkpointing = True

# Or use 4-bit quantization during training:
load_in_4bit = True
```

#### Issue 2: nvidia-smi Not Found in Container

**Symptoms:**
```bash
nvidia-smi: command not found
```

**Solution:**
```bash
# On LXC host machine:
lxc config device add your-container gpu gpu gputype=physical
lxc restart your-container

# Verify in container:
lxc exec your-container -- nvidia-smi
```

#### Issue 3: Slow Training Speed

**Symptoms:**
Training taking much longer than expected

**Solutions:**
```python
# 1. Increase batch size (if VRAM allows):
per_device_train_batch_size = 4

# 2. Use mixed precision:
fp16 = True  # or bf16 = True

# 3. Reduce sequence length:
max_seq_length = 1024  # Instead of 2048

# 4. Use faster optimizer:
optim = "adamw_8bit"  # Already recommended
```

#### Issue 4: Model Not Learning

**Symptoms:**
Validation loss not decreasing, poor predictions

**Solutions:**
```python
# 1. Increase learning rate:
learning_rate = 5e-5  # Instead of 2e-5

# 2. Train longer:
num_epochs = 5  # Instead of 3

# 3. Increase LoRA rank:
lora_rank = 32  # Instead of 16

# 4. Check data quality:
# - Ensure consistent formatting
# - Verify labels are correct
# - Check for duplicate examples
```

#### Issue 5: JSON Parsing Failures

**Symptoms:**
Model output is not valid JSON

**Solutions:**
```python
# 1. Add explicit JSON formatting instruction in system prompt:
"You MUST respond with ONLY valid JSON. Do not include any markdown formatting."

# 2. Use constrained decoding (if available):
# Force model to only generate valid JSON

# 3. Post-process outputs:
def extract_json(text):
    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "")
    # Try to parse
    return json.loads(text)
```

#### Issue 6: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**Solution:**
```bash
# Reinstall packages
pip install --upgrade --force-reinstall "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade transformers datasets trl
```

### Performance Optimization Checklist

- [ ] Using batch_size > 1 (if VRAM allows)
- [ ] Using gradient_accumulation_steps for larger effective batch
- [ ] Using FP16/BF16 mixed precision
- [ ] Using efficient optimizer (adamw_8bit)
- [ ] Using gradient checkpointing
- [ ] Using Unsloth optimizations
- [ ] Monitoring GPU utilization with nvidia-smi
- [ ] Using appropriate max_seq_length (not too large)

---

## Performance Benchmarks

### Expected Performance (RTX 4080 16GB)

#### Fine-tuning

| Dataset Size | Batch Size | Epochs | Time per Epoch | Total Time |
|--------------|------------|--------|----------------|------------|
| 500 examples | 2 | 3 | 40-50 min | 2-2.5 hours |
| 1,000 examples | 2 | 3 | 1-1.5 hours | 3-4.5 hours |
| 2,000 examples | 2 | 3 | 2-3 hours | 6-9 hours |
| 5,000 examples | 2 | 3 | 5-7 hours | 15-21 hours |

#### Inference (Sequential Processing)

| Articles | Agents | Processing Mode | Time per Article | Total Time |
|----------|--------|----------------|------------------|------------|
| 5,000 | 2 | Sequential FP16 | ~10s | ~14 hours |
| 5,000 | 2 | Sequential INT8 | ~8s | ~11 hours |
| 5,000 | 2 | Parallel INT8 | ~5s | ~7 hours |

**Recommended:** Sequential FP16 for best quality

#### VRAM Usage

| Configuration | VRAM Usage | Notes |
|---------------|------------|-------|
| 1× Qwen 7B FP16 | 14 GB | Training or inference |
| 1× Qwen 7B INT8 | 7 GB | Inference only |
| 2× Qwen 7B INT8 parallel | 14 GB | Tight fit on 16GB |
| 2× Qwen 7B FP16 sequential | 8 GB peak | Recommended approach |

### Quality Benchmarks

**Expected accuracy on validation set:**

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| MAE (per dimension) | <1.5 | <1.0 | <0.5 |
| RMSE (per dimension) | <2.0 | <1.5 | <1.0 |
| JSON parse success | >90% | >95% | >98% |
| Exact match | >60% | >75% | >85% |

---

## Cost Analysis

### Hardware Costs

**Option 1: Your Current Setup (RTX 4080)**
- **Capital cost:** ~€1,200 (one-time)
- **Power consumption:** ~320W × 24h × €0.30/kWh = €2.30/day
- **Monthly operating cost:** ~€70
- **Break-even vs API:** 6-12 months

**Option 2: Hetzner GPU Server**
- **Monthly rental:** €120-180
- **No capital investment**
- **Instant scalability**

**Option 3: API (Gemini Flash)**
- **Monthly cost:** €10-30 for 1 agent
- **Scales linearly:** €100-300 for 10 agents
- **No infrastructure management**

### Total Cost of Ownership (1 year)

| Option | Setup | Monthly | 1-Year Total | 2-Year Total |
|--------|-------|---------|--------------|--------------|
| Your RTX 4080 | €1,200 | €70 | €2,040 | €2,880 |
| Hetzner GPU | €0 | €150 | €1,800 | €3,600 |
| Gemini API (5 agents) | €0 | €50 | €600 | €1,200 |
| Gemini API (20 agents) | €0 | €200 | €2,400 | €4,800 |

**Recommendation:** Your RTX 4080 is cost-effective for 2+ agents, especially long-term.

---

## Next Steps

### Immediate Actions

1. **✅ Set up environment** (1-2 hours)
   - Install dependencies
   - Verify CUDA
   - Download base model

2. **✅ Prepare training data** (2-4 hours)
   - Convert Gemini labels to training format
   - Split train/validation sets
   - Run quality checks

3. **✅ Fine-tune first agent** (3-8 hours)
   - Start with uplifting agent
   - Monitor training
   - Validate results

4. **✅ Fine-tune second agent** (3-8 hours)
   - Sustainability agent
   - Compare with baseline

5. **✅ Test both agents** (1-2 hours)
   - Run validation tests
   - Measure accuracy and speed
   - Compare with Gemini

6. **✅ Deploy to production** (4-8 hours)
   - Set up sequential processor
   - Configure cron job
   - Test end-to-end pipeline

### Future Enhancements

- **Add more agents** as semantic dimensions evolve
- **Fine-tune with more data** to improve accuracy
- **Upgrade to Qwen 14B** for higher quality (if needed)
- **Implement active learning** to continuously improve
- **Add monitoring and alerting** for production
- **Build web dashboard** for results visualization

---

## Resources

### Documentation
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

### Community
- [Unsloth Discord](https://discord.gg/unsloth)
- [HuggingFace Forums](https://discuss.huggingface.co/)

### Benchmarks
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2409.12186)
- [LLM Leaderboards](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

## Appendix: Quick Reference Commands

```bash
# Environment setup
python3 -m venv venv
source venv/bin/activate
pip install torch transformers datasets trl unsloth

# Check CUDA
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Download model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='./models/qwen2.5-7b-instruct')"

# Prepare data
python3 create_training_data.py

# Fine-tune
python3 finetune_qwen.py

# Test
python3 test_finetuned_model.py

# Deploy
python3 sequential_processor.py
```

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review training logs in `training.log`
3. Monitor GPU usage with `nvidia-smi`
4. Verify data format with `check_data_quality.py`

Good luck with your fine-tuning! 🚀

---

**Document Version:** 1.0  
**Created:** October 29, 2025  
**Author:** Claude (Anthropic)
