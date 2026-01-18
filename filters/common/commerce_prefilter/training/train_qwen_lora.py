"""
Commerce Prefilter - Qwen LoRA Training

Trains Qwen 0.5B with LoRA for binary classification.

Usage:
    python train_qwen_lora.py --output models/qwen-lora
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_data(path: Path) -> list:
    """Load JSONL data."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_input(article: dict) -> str:
    """Format article as [TITLE] ... [CONTENT] ..."""
    title = article.get('title', '')
    content = article.get('content', '')
    return f"[TITLE] {title} [CONTENT] {content}"


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)).numpy() > 0.5).astype(int).flatten()

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA for commerce detection")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--data-dir", type=str, default="splits", help="Data directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)  # Smaller for larger model
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # Higher for LoRA
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    args = parser.parse_args()

    print("=" * 60)
    print(f"Training: {args.model} with LoRA")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    train_data = load_data(Path(args.data_dir) / "train.jsonl")
    val_data = load_data(Path(args.data_dir) / "val.jsonl")

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load tokenizer and model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,  # Binary classification with sigmoid
        problem_type="multi_label_classification",  # Use BCEWithLogitsLoss
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Ensure pad token id is set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    print(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Qwen attention modules
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_texts = [format_input(ex) for ex in train_data]
    train_labels = [[float(ex['label'])] for ex in train_data]
    val_texts = [format_input(ex) for ex in val_data]
    val_labels = [[float(ex['label'])] for ex in val_data]

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels,
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels,
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.output}/logs",
        logging_steps=50,
        bf16=torch.cuda.is_available(),  # Use bf16 instead of fp16 for LoRA
        gradient_accumulation_steps=2,  # Effective batch size = 16
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nTraining...")
    trainer.train()

    # Save LoRA adapter and tokenizer
    print(f"\nSaving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save metrics
    metrics_path = Path(args.output) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save config for loading later
    config = {
        'base_model': args.model,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'max_length': args.max_length,
    }
    config_path = Path(args.output) / "lora_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
