"""
Commerce Prefilter - CPU Inference Benchmark

Benchmarks inference speed on CPU for the selected model.

Usage:
    python benchmark_inference.py --model-dir ../v1/models/distilbert --test-data splits/test.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU inference speed")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--test-data", type=str, required=True, help="Test data JSONL")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to benchmark")

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter - CPU Inference Benchmark")
    print("=" * 60)
    print(f"Device: CPU")
    print(f"Model: {args.model_dir}")

    # Load model
    print("\nLoading model...")
    model_path = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_data = load_data(Path(args.test_data))
    texts = [format_input(ex) for ex in test_data]

    # Limit samples
    texts = texts[:args.n_samples]
    print(f"Benchmarking on {len(texts)} samples")

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    with torch.no_grad():
        for i in range(args.warmup):
            inputs = tokenizer(
                texts[i % len(texts)],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            _ = model(**inputs)

    # Benchmark
    print("Benchmarking...")
    inference_times = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

            start = time.perf_counter()
            outputs = model(**inputs)
            inference_time = (time.perf_counter() - start) * 1000  # ms

            inference_times.append(inference_time)

    # Calculate statistics
    inference_times.sort()
    avg_time = sum(inference_times) / len(inference_times)
    p50_time = inference_times[len(inference_times) // 2]
    p95_time = inference_times[int(len(inference_times) * 0.95)]
    p99_time = inference_times[int(len(inference_times) * 0.99)]
    min_time = inference_times[0]
    max_time = inference_times[-1]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples: {len(texts)}")
    print(f"Average: {avg_time:.1f}ms")
    print(f"P50: {p50_time:.1f}ms")
    print(f"P95: {p95_time:.1f}ms")
    print(f"P99: {p99_time:.1f}ms")
    print(f"Min: {min_time:.1f}ms")
    print(f"Max: {max_time:.1f}ms")

    # Check against target
    target_ms = 50
    min_ms = 100
    if avg_time < target_ms:
        print(f"\nPASS: Average inference ({avg_time:.1f}ms) < target ({target_ms}ms)")
    elif avg_time < min_ms:
        print(f"\nACCEPTABLE: Average inference ({avg_time:.1f}ms) < minimum ({min_ms}ms)")
    else:
        print(f"\nFAIL: Average inference ({avg_time:.1f}ms) > minimum ({min_ms}ms)")

    # Save results
    results = {
        'model': str(args.model_dir),
        'n_samples': len(texts),
        'avg_ms': avg_time,
        'p50_ms': p50_time,
        'p95_ms': p95_time,
        'p99_ms': p99_time,
        'min_ms': min_time,
        'max_ms': max_time,
    }

    output_path = model_path / "cpu_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
