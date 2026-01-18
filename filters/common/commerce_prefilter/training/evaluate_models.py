"""
Commerce Prefilter - Model Evaluation

Evaluates all trained models on the test set.

Usage:
    python evaluate_models.py --models-dir models --test-data splits/test.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def evaluate_encoder_model(model_path: Path, test_data: list, device: str, max_length: int = 512):
    """Evaluate an encoder model."""
    print(f"\nLoading {model_path.name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    texts = [format_input(ex) for ex in test_data]
    labels = [ex['label'] for ex in test_data]

    predictions = []
    scores = []
    inference_times = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            start = time.time()
            outputs = model(**inputs)
            inference_time = (time.time() - start) * 1000  # ms

            logits = outputs.logits
            # For 2-label classification, use softmax and take probability of class 1
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, 1].cpu().item()  # Probability of positive class
            pred = 1 if score > 0.5 else 0

            predictions.append(pred)
            scores.append(score)
            inference_times.append(inference_time)

    return {
        'predictions': predictions,
        'scores': scores,
        'labels': labels,
        'inference_times': inference_times,
    }


def evaluate_qwen_lora(model_path: Path, test_data: list, device: str, max_length: int = 512):
    """Evaluate Qwen model with LoRA adapter."""
    print(f"\nLoading {model_path.name} (LoRA)...")

    # Load config
    config_path = model_path / "lora_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    base_model_name = config['base_model']

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        problem_type="multi_label_classification",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    texts = [format_input(ex) for ex in test_data]
    labels = [ex['label'] for ex in test_data]

    predictions = []
    scores = []
    inference_times = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            start = time.time()
            outputs = model(**inputs)
            inference_time = (time.time() - start) * 1000  # ms

            logits = outputs.logits
            score = torch.sigmoid(logits).cpu().float().item()
            pred = 1 if score > 0.5 else 0

            predictions.append(pred)
            scores.append(score)
            inference_times.append(inference_time)

    return {
        'predictions': predictions,
        'scores': scores,
        'labels': labels,
        'inference_times': inference_times,
    }


def compute_metrics(results: dict) -> dict:
    """Compute all metrics from results."""
    predictions = results['predictions']
    labels = results['labels']
    inference_times = results['inference_times']

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'avg_inference_ms': sum(inference_times) / len(inference_times),
        'p50_inference_ms': sorted(inference_times)[len(inference_times) // 2],
        'p95_inference_ms': sorted(inference_times)[int(len(inference_times) * 0.95)],
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--test-data", type=str, default="splits/test.jsonl", help="Test data")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--max-length", type=int, default=512)

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - Model Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test data
    print(f"\nLoading test data from {args.test_data}")
    test_data = load_data(Path(args.test_data))
    print(f"Test samples: {len(test_data)}")

    # Find models
    models_dir = Path(args.models_dir)
    model_paths = [p for p in models_dir.iterdir() if p.is_dir()]
    print(f"\nFound models: {[p.name for p in model_paths]}")

    # Evaluate each model
    all_results = {}

    for model_path in model_paths:
        model_name = model_path.name

        try:
            # Check if it's a LoRA model
            if (model_path / "lora_config.json").exists():
                results = evaluate_qwen_lora(model_path, test_data, device, args.max_length)
            else:
                results = evaluate_encoder_model(model_path, test_data, device, args.max_length)

            metrics = compute_metrics(results)
            all_results[model_name] = metrics

            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Avg inference: {metrics['avg_inference_ms']:.1f}ms")
            print(f"  P95 inference: {metrics['p95_inference_ms']:.1f}ms")

        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Inf(ms)':>10}")
    print("-" * 60)

    for model_name, metrics in sorted(all_results.items(), key=lambda x: x[1].get('f1', 0), reverse=True):
        if 'error' in metrics:
            print(f"{model_name:<20} ERROR: {metrics['error'][:30]}")
        else:
            print(f"{model_name:<20} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} {metrics['avg_inference_ms']:>10.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
