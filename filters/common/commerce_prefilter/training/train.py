"""
Commerce Prefilter - Model Training

Fine-tunes DistilBERT for binary classification of commerce vs journalism content.

Usage:
    python -m filters.common.commerce_prefilter.training.train \
        --data-dir datasets/commerce_prefilter/splits/ \
        --output filters/common/commerce_prefilter/v1/model/

Requirements:
    pip install transformers datasets scikit-learn
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def prepare_text(example: Dict) -> str:
    """Prepare input text from article fields."""
    title = example.get('title', '')
    content = example.get('content', '')

    # Truncate content for efficiency (model handles final truncation)
    content = content[:2000] if content else ''

    return f"{title}\n\n{content}".strip()


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    # Convert logits to predictions using sigmoid + threshold
    probs = torch.sigmoid(torch.tensor(predictions)).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train commerce prefilter model"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("datasets/commerce_prefilter/splits"),
        help="Directory with train.jsonl, val.jsonl, test.jsonl"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("filters/common/commerce_prefilter/v1/model"),
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/Multilingual-MiniLM-L12-H384",
        help="Base model to fine-tune. Options: "
             "distilbert-base-multilingual-cased, "
             "microsoft/Multilingual-MiniLM-L12-H384, "
             "xlm-roberta-base, "
             "Qwen/Qwen2.5-0.5B"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Model Training")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")

    # Load data
    print("\nLoading data...")
    train_examples = load_jsonl(args.data_dir / "train.jsonl")
    val_examples = load_jsonl(args.data_dir / "val.jsonl")
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Create datasets
    train_data = {
        'text': [prepare_text(ex) for ex in train_examples],
        'label': [float(ex.get('label', 0)) for ex in train_examples],
    }
    val_data = {
        'text': [prepare_text(ex) for ex in val_examples],
        'label': [float(ex.get('label', 0)) for ex in val_examples],
    }

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Detect if decoder model (Qwen)
    is_decoder_model = "qwen" in args.base_model.lower()

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=is_decoder_model,
    )

    # Handle padding token for decoder models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # DataCollator handles padding
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Load model
    print(f"\nLoading model: {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,  # Single output for binary classification with sigmoid
        problem_type="regression",  # Use BCEWithLogitsLoss internally
        trust_remote_code=is_decoder_model,
    )

    # Handle padding token ID for decoder models
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(args.output / "logs"),
        logging_steps=100,
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Final evaluation on validation set")
    print("=" * 60)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save model
    print(f"\nSaving model to {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    # Save training config
    config = {
        'base_model': args.base_model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'final_metrics': {k: float(v) for k, v in eval_results.items()},
    }
    with open(args.output / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print(f"Best F1: {eval_results.get('eval_f1', 0):.4f}")

    # Evaluate on test set if available
    test_path = args.data_dir / "test.jsonl"
    if test_path.exists():
        print("\n" + "=" * 60)
        print("Evaluating on test set")
        print("=" * 60)

        test_examples = load_jsonl(test_path)
        test_data = {
            'text': [prepare_text(ex) for ex in test_examples],
            'label': [float(ex.get('label', 0)) for ex in test_examples],
        }
        test_dataset = Dataset.from_dict(test_data)
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

        test_results = trainer.evaluate(test_dataset)
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

        # Update config with test results
        config['test_examples'] = len(test_examples)
        config['test_metrics'] = {k: float(v) for k, v in test_results.items()}
        with open(args.output / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
