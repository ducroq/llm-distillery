"""
Commerce Prefilter - Encoder Model Training

Trains encoder models (DistilBERT, MiniLM, XLM-RoBERTa) for binary classification.

Usage:
    python train_encoder.py --model distilbert-base-multilingual-cased --output models/distilbert
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
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
    predictions = logits.argmax(axis=-1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train encoder model for commerce detection")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--data-dir", type=str, default="splits", help="Data directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    args = parser.parse_args()

    print("=" * 60)
    print(f"Training: {args.model}")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,  # Binary classification with softmax
    )

    # Prepare datasets
    def preprocess(examples):
        texts = [format_input(ex) for ex in examples]
        labels = [ex['label'] for ex in examples]

        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
        )
        encodings['labels'] = [[float(l)] for l in labels]
        return encodings

    # Convert to HuggingFace Dataset
    train_texts = [format_input(ex) for ex in train_data]
    train_labels = [int(ex['label']) for ex in train_data]
    val_texts = [format_input(ex) for ex in val_data]
    val_labels = [int(ex['label']) for ex in val_data]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length)

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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.output}/logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
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

    # Save best model
    print(f"\nSaving model to {args.output}")
    trainer.save_model(args.output)
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

    print("\nDone!")


if __name__ == "__main__":
    main()
