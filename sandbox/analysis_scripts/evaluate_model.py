#!/usr/bin/env python3
"""
Quick model evaluation script for test set.

Evaluates a trained filter model on test data and reports MAE/RMSE metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

        if self.examples:
            self.num_dimensions = len(self.examples[0]['labels'])
            self.dimension_names = self.examples[0]['dimension_names']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = f"{example['title']}\n\n{example['content']}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = torch.tensor(example['labels'], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def evaluate(model, dataloader, device, dimension_names):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)

    # Calculate metrics
    errors = predictions - labels
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    # Per-dimension metrics
    per_dim_mae = np.abs(errors).mean(axis=0)
    per_dim_rmse = np.sqrt((errors ** 2).mean(axis=0))

    return {
        'overall_mae': float(mae),
        'overall_rmse': float(rmse),
        'per_dimension_mae': {name: float(val) for name, val in zip(dimension_names, per_dim_mae)},
        'per_dimension_rmse': {name: float(val) for name, val in zip(dimension_names, per_dim_rmse)},
        'num_examples': len(labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model-dir', required=True, help='Path to trained model directory')
    parser.add_argument('--test-data', required=True, help='Path to test.jsonl')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load metadata to get base model info
    metadata_path = model_dir.parent / 'training_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            base_model_name = metadata.get('model_name', 'Qwen/Qwen2.5-7B')
            num_dimensions = metadata.get('num_dimensions', 8)
    else:
        print("Warning: training_metadata.json not found, using defaults")
        base_model_name = 'Qwen/Qwen2.5-7B'
        num_dimensions = 8

    print(f"Loading model from {model_dir}...")
    print(f"Base model: {base_model_name}")
    print(f"Device: {args.device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with proper padding token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_dimensions,
        problem_type="regression"
    )

    # Set model's pad_token_id to match tokenizer
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.to(args.device)

    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    dataset = SimpleDataset(args.test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test examples: {len(dataset)}")
    print(f"Dimensions: {dataset.dimension_names}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate(model, dataloader, args.device, dataset.dimension_names)

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {results['overall_mae']:.4f}")
    print(f"  RMSE: {results['overall_rmse']:.4f}")

    print(f"\nPer-Dimension MAE:")
    for dim, mae in results['per_dimension_mae'].items():
        print(f"  {dim:30s}: {mae:.4f}")

    print(f"\nPer-Dimension RMSE:")
    for dim, rmse in results['per_dimension_rmse'].items():
        print(f"  {dim:30s}: {rmse:.4f}")

    print(f"\nExamples evaluated: {results['num_examples']}")
    print("=" * 60)

    # Save results
    output_file = model_dir.parent / 'test_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
