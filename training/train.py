"""
Training script for LLM Distillery using Qwen 2.5.

Fine-tunes Qwen 2.5 models for multi-dimensional regression on filter-specific datasets.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FilterDataset(Dataset):
    """PyTorch Dataset for filter training data."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 512,
        prompt: str = None,
    ):
        """
        Args:
            data_path: Path to JSONL file with training examples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            prompt: Optional filter prompt to prepend to each example (for instruction tuning)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.examples = []

        # Load examples
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

        # Get dimension count from first example
        if self.examples:
            self.num_dimensions = len(self.examples[0]["labels"])
            self.dimension_names = self.examples[0]["dimension_names"]
        else:
            raise ValueError(f"No examples found in {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Combine title and content
        article_text = f"{example['title']}\n\n{example['content']}"

        # Optionally prepend prompt (instruction tuning mode)
        if self.prompt:
            text = f"{self.prompt}\n\n{article_text}"
        else:
            text = article_text

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert labels to tensor
        labels = torch.tensor(example["labels"], dtype=torch.float32)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


class QwenFilterModel(torch.nn.Module):
    """
    Qwen 2.5 model adapted for multi-dimensional regression.

    Uses the base model with a custom regression head that outputs
    multiple continuous scores (one per dimension).
    """

    def __init__(self, model_name: str, num_dimensions: int, use_gradient_checkpointing: bool = True, use_fp16: bool = False, use_quantization: bool = False):
        super().__init__()

        # Load base Qwen model (we'll use the sequence classification variant)
        # Note: AutoModelForSequenceClassification expects num_labels, but we'll
        # use it for regression by setting num_labels = num_dimensions
        load_kwargs = {
            "num_labels": num_dimensions,
            "problem_type": "regression",
        }

        # Optional: Configure 8-bit quantization for large models (7B+) on 16GB GPU
        # Reduces memory: 15GB -> 4GB, but adds complexity
        # Default: False for 1.5B model (fits in memory without quantization)
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        
        # Only use FP16 if explicitly requested (can cause NaN issues)
        if use_fp16:
            load_kwargs["torch_dtype"] = torch.float16

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        # Apply LoRA for memory-efficient training
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Apply LoRA
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"  LoRA applied: {trainable_params:,} / {all_params:,} parameters ({100 * trainable_params / all_params:.2f}% trainable)")

        self.num_dimensions = num_dimensions
        self.use_fp16 = use_fp16

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, dimension_names: List[str]) -> Dict:
    """
    Compute evaluation metrics per dimension.

    Args:
        predictions: Model predictions (batch_size, num_dimensions)
        labels: Ground truth labels (batch_size, num_dimensions)
        dimension_names: Names of dimensions

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Overall metrics
    mae = torch.mean(torch.abs(predictions - labels)).item()
    rmse = torch.sqrt(torch.mean((predictions - labels) ** 2)).item()

    metrics["mae"] = mae
    metrics["rmse"] = rmse

    # Per-dimension metrics
    for i, dim_name in enumerate(dimension_names):
        dim_predictions = predictions[:, i]
        dim_labels = labels[:, i]

        dim_mae = torch.mean(torch.abs(dim_predictions - dim_labels)).item()
        dim_rmse = torch.sqrt(torch.mean((dim_predictions - dim_labels) ** 2)).item()

        metrics[f"{dim_name}_mae"] = dim_mae
        metrics[f"{dim_name}_rmse"] = dim_rmse

    return metrics


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    dimension_names: List[str],
):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        predictions = outputs.logits

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[WARNING] NaN loss detected at step {len(all_predictions)}, skipping batch")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion with FP16
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_labels.append(labels.detach().cpu())

        progress.set_postfix({"loss": loss.item()})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels, dimension_names)
    metrics["loss"] = avg_loss

    return metrics


def evaluate(model, dataloader, device, dimension_names: List[str]):
    """Evaluate model on validation/test set."""
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            predictions = outputs.logits

            # Track metrics
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels, dimension_names)
    metrics["loss"] = avg_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train filter model using Qwen 2.5")
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter directory (e.g., filters/uplifting/v1)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to prepared dataset directory (with train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for trained model (default: saves to filter directory)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Qwen model name (default: Qwen/Qwen2.5-1.5B, use 1.5B for 16GB GPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g., filters/uplifting/v1)",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include filter prompt in training (instruction tuning mode). Prepends prompt-compressed.md to each article.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Load filter config
    print(f"Loading filter config from {args.filter}")
    config_path = args.filter / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filter_name = config["filter"]["name"]
    dimension_names = list(config["scoring"]["dimensions"].keys())
    num_dimensions = len(dimension_names)

    print(f"Filter: {filter_name}")
    print(f"Dimensions ({num_dimensions}): {dimension_names}")

    # Set output directory (default: save to filter directory)
    if args.output_dir is None:
        args.output_dir = args.filter
        print(f"Output directory: {args.output_dir} (filter directory)")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if not set (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")

    # Optionally load prompt for instruction tuning
    prompt = None
    if args.include_prompt:
        prompt_path = args.filter / "prompt-compressed.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\nInstruction tuning mode enabled")
        print(f"  Loaded prompt from: {prompt_path}")
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Warning: Longer sequences may require --max-length adjustment")

    # Load datasets
    print(f"\nLoading datasets from {args.data_dir}")
    train_dataset = FilterDataset(
        args.data_dir / "train.jsonl",
        tokenizer,
        max_length=args.max_length,
        prompt=prompt,
    )
    val_dataset = FilterDataset(
        args.data_dir / "val.jsonl",
        tokenizer,
        max_length=args.max_length,
        prompt=prompt,
    )

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load or initialize model
    start_epoch = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")

        # Load model from checkpoint
        checkpoint_model_path = args.resume_from / "model"
        if not checkpoint_model_path.exists():
            raise ValueError(f"Checkpoint model not found at {checkpoint_model_path}")

        # Load metadata to get base model name
        metadata_path = args.resume_from / "training_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"training_metadata.json not found at {args.resume_from}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            base_model_name = metadata["model_name"]

        print(f"  Base model: {base_model_name}")

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_dimensions,
            problem_type="regression"
        )

        # Load PEFT adapter (already trained LoRA weights)
        from peft import PeftModel
        model_with_adapter = PeftModel.from_pretrained(base_model, checkpoint_model_path)

        # Wrap in simple container to match interface
        class ResumedModel(torch.nn.Module):
            def __init__(self, peft_model):
                super().__init__()
                self.base_model = peft_model
                self.num_dimensions = num_dimensions
                self.use_fp16 = False

            def forward(self, input_ids, attention_mask, labels=None):
                return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            def parameters(self):
                return self.base_model.parameters()

        model = ResumedModel(model_with_adapter)
        print(f"  Loaded PEFT model from checkpoint (no double LoRA)")

        # Load training history to determine start epoch
        history_path = args.resume_from / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                training_history = json.load(f)
            start_epoch = training_history[-1]["epoch"]
            best_val_mae = training_history[-1]["val"]["mae"]
            print(f"  Resuming from epoch {start_epoch} (best val MAE: {best_val_mae:.4f})")
        else:
            training_history = []
            print(f"  Warning: No training history found, starting fresh")
    else:
        print(f"\nInitializing model: {args.model_name}")
        # Use FP32 by default for stability (FP16 causes NaN issues)
        model = QwenFilterModel(args.model_name, num_dimensions, use_gradient_checkpointing=True, use_fp16=False)
        training_history = []
        best_val_mae = float("inf")

    # Set pad_token_id in model config to match tokenizer
    if model.base_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"  Set model pad_token_id to {tokenizer.pad_token_id}")

    model.to(device)

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} ({num_trainable:,} trainable)")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_epochs = start_epoch + args.epochs
    print(f"\nStarting training from epoch {start_epoch + 1} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            dimension_names,
        )

        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_dataloader, device, dimension_names)

        print(f"\nValidation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")

        # Save metrics
        epoch_history = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        training_history.append(epoch_history)

        # Save best model
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            print(f"\n✓ New best validation MAE: {best_val_mae:.4f}")

            # Save model
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / "model"
            model.base_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            print(f"  Model saved to: {model_path}")

    # Save training history
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation MAE: {best_val_mae:.4f}")

    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    print(f"Training history saved to: {history_path}")

    # Save training metadata
    metadata = {
        "filter_name": filter_name,
        "filter_version": config["filter"]["version"],
        "dimension_names": dimension_names,
        "num_dimensions": num_dimensions,
        "model_name": args.model_name,
        "num_parameters": num_params,
        "num_trainable_parameters": num_trainable,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "best_val_mae": best_val_mae,
        "include_prompt": args.include_prompt,
        "training_mode": "instruction_tuning" if args.include_prompt else "knowledge_distillation",
    }

    metadata_path = args.output_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training metadata saved to: {metadata_path}")

    # Print next steps reminder
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Review training results in {args.output_dir}/")
    print(f"2. Run Model Evaluation Agent (see training/README.md)")
    print(f"3. Review report: {args.output_dir}/model_evaluation.md")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
