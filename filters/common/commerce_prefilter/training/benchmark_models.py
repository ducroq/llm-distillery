"""
Commerce Prefilter - Model Benchmarking

Compares inference speed and (optionally) accuracy across model candidates.

Usage:
    # Speed benchmark only (no training data needed)
    python -m filters.common.commerce_prefilter.training.benchmark_models --speed-only

    # Full benchmark with accuracy (requires trained models)
    python -m filters.common.commerce_prefilter.training.benchmark_models \
        --test-data datasets/commerce_prefilter/splits/test.jsonl
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Model candidates to benchmark
MODEL_CANDIDATES = {
    "distilbert-multilingual": {
        "name": "distilbert-base-multilingual-cased",
        "type": "encoder",
        "params": "135M",
    },
    "multilingual-minilm": {
        "name": "microsoft/Multilingual-MiniLM-L12-H384",
        "type": "encoder",
        "params": "118M",
    },
    "xlm-roberta": {
        "name": "xlm-roberta-base",
        "type": "encoder",
        "params": "270M",
    },
    "qwen-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B",
        "type": "decoder",
        "params": "500M",
    },
}

# Sample texts for speed benchmarking (English + multilingual)
BENCHMARK_TEXTS = [
    # English - Commerce
    "Green Deals: Save $500 on Jackery Solar Generator. Today's deals are headlined by an exclusive discount. Originally $1,999, now just $1,499!",
    # English - Journalism
    "Researchers at MIT have developed a breakthrough perovskite-silicon tandem solar cell that achieves 30% efficiency, surpassing traditional silicon panels.",
    # German - Commerce
    "Black Friday Angebote: Sparen Sie 40% auf Solaranlagen. Nutzen Sie den Rabattcode SOLAR40 für zusätzliche 10% Rabatt.",
    # German - Journalism
    "Die Bundesregierung hat neue Vorschriften zur Reduzierung der CO2-Emissionen um 50% bis 2030 angekündigt.",
    # Dutch - Commerce
    "Cyber Monday deals: Bespaar €300 op elektrische fietsen. Gebruik kortingscode EBIKE300 voor extra korting.",
    # Dutch - Journalism
    "Nederlandse onderzoekers hebben een doorbraak bereikt in waterstoftechnologie voor duurzame energie.",
    # French - Journalism
    "L'Agence de Protection de l'Environnement a annoncé de nouvelles réglementations pour les énergies renouvelables.",
    # Chinese - Journalism
    "全球风电装机容量已达到1太瓦里程碑，中国和美国在新装机方面领先。",
    # Long text
    "This is a longer article about sustainability technology. " * 50,
]


def get_model_size_mb(model) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)


def benchmark_encoder_model(
    model_name: str,
    texts: List[str],
    device: str,
    num_runs: int = 3,
    warmup_runs: int = 2,
) -> Dict:
    """Benchmark an encoder model (BERT-style)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    results = {
        "model": model_name,
        "type": "encoder",
        "device": device,
        "status": "success",
    }

    try:
        print(f"  Loading {model_name}...")
        load_start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )
        model = model.to(device)
        model.eval()

        load_time = time.perf_counter() - load_start
        results["load_time_s"] = round(load_time, 2)
        results["model_size_mb"] = round(get_model_size_mb(model), 1)

        # Warmup
        print(f"  Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            for text in texts[:3]:
                inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)

        # Benchmark
        print(f"  Benchmarking ({num_runs} runs)...")
        times_per_text = []

        for _ in range(num_runs):
            for text in texts:
                inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**inputs)
                    _ = torch.sigmoid(outputs.logits)  # Include sigmoid
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                times_per_text.append(elapsed * 1000)  # Convert to ms

        results["inference_times_ms"] = {
            "min": round(min(times_per_text), 2),
            "max": round(max(times_per_text), 2),
            "mean": round(sum(times_per_text) / len(times_per_text), 2),
            "median": round(sorted(times_per_text)[len(times_per_text) // 2], 2),
        }

        # Cleanup
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    return results


def benchmark_decoder_model(
    model_name: str,
    texts: List[str],
    device: str,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> Dict:
    """Benchmark a decoder model (Qwen-style) for classification."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    results = {
        "model": model_name,
        "type": "decoder",
        "device": device,
        "status": "success",
    }

    try:
        print(f"  Loading {model_name}...")
        load_start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model = model.to(device)
        model.eval()

        load_time = time.perf_counter() - load_start
        results["load_time_s"] = round(load_time, 2)
        results["model_size_mb"] = round(get_model_size_mb(model), 1)

        # Warmup
        print(f"  Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            for text in texts[:2]:
                inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)

        # Benchmark
        print(f"  Benchmarking ({num_runs} runs)...")
        times_per_text = []

        for _ in range(num_runs):
            for text in texts:
                inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**inputs)
                    _ = torch.sigmoid(outputs.logits)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                times_per_text.append(elapsed * 1000)

        results["inference_times_ms"] = {
            "min": round(min(times_per_text), 2),
            "max": round(max(times_per_text), 2),
            "mean": round(sum(times_per_text) / len(times_per_text), 2),
            "median": round(sorted(times_per_text)[len(times_per_text) // 2], 2),
        }

        # Cleanup
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    return results


def benchmark_accuracy(
    model_path: Path,
    test_data_path: Path,
    model_type: str = "encoder",
) -> Dict:
    """Benchmark accuracy on test data (requires trained model)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    results = {"status": "success"}

    try:
        # Load test data
        test_examples = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_examples.append(json.loads(line))

        # Load model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.eval()

        # Predict
        predictions = []
        labels = []

        for ex in test_examples:
            text = f"{ex.get('title', '')}\n\n{ex.get('content', '')[:2000]}"
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                prob = torch.sigmoid(outputs.logits).item()

            predictions.append(1 if prob >= 0.5 else 0)
            labels.append(int(ex.get('label', 0)))

        # Calculate metrics
        results["accuracy"] = round(accuracy_score(labels, predictions), 4)
        results["f1"] = round(f1_score(labels, predictions), 4)
        results["precision"] = round(precision_score(labels, predictions), 4)
        results["recall"] = round(recall_score(labels, predictions), 4)
        results["test_size"] = len(test_examples)

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    return results


def print_results_table(results: List[Dict]):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Model':<45} {'Size':>8} {'Load':>8} {'Inference (ms)':>20}")
    print(f"{'':45} {'(MB)':>8} {'(s)':>8} {'mean':>8} {'min':>6} {'max':>6}")
    print("-" * 80)

    # Sort by mean inference time
    sorted_results = sorted(
        [r for r in results if r.get("status") == "success"],
        key=lambda x: x.get("inference_times_ms", {}).get("mean", 9999)
    )

    for r in sorted_results:
        model = r["model"].split("/")[-1][:44]
        size = r.get("model_size_mb", "?")
        load = r.get("load_time_s", "?")
        times = r.get("inference_times_ms", {})

        print(f"{model:<45} {size:>8} {load:>8} {times.get('mean', '?'):>8} {times.get('min', '?'):>6} {times.get('max', '?'):>6}")

    # Failed models
    failed = [r for r in results if r.get("status") != "success"]
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  {r['model']}: {r.get('error', 'Unknown error')}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark commerce prefilter model candidates"
    )
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="Only run speed benchmarks (no accuracy)"
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Test data for accuracy benchmark"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to benchmark on"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of benchmark runs per text"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CANDIDATES.keys()) + ["all"],
        default=["all"],
        help="Models to benchmark"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("Commerce Prefilter - Model Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Benchmark texts: {len(BENCHMARK_TEXTS)}")
    print(f"Runs per text: {args.num_runs}")

    # Select models
    if "all" in args.models:
        models_to_test = MODEL_CANDIDATES
    else:
        models_to_test = {k: v for k, v in MODEL_CANDIDATES.items() if k in args.models}

    print(f"Models to test: {list(models_to_test.keys())}")

    # Run benchmarks
    results = []

    for key, config in models_to_test.items():
        print(f"\n--- {key} ---")
        model_name = config["name"]
        model_type = config["type"]

        if model_type == "encoder":
            result = benchmark_encoder_model(
                model_name,
                BENCHMARK_TEXTS,
                device,
                num_runs=args.num_runs,
            )
        else:  # decoder
            result = benchmark_decoder_model(
                model_name,
                BENCHMARK_TEXTS,
                device,
                num_runs=args.num_runs,
            )

        result["key"] = key
        result["params"] = config["params"]
        results.append(result)

        # Print intermediate result
        if result["status"] == "success":
            times = result["inference_times_ms"]
            print(f"  Mean: {times['mean']:.1f}ms, Min: {times['min']:.1f}ms, Max: {times['max']:.1f}ms")
        else:
            print(f"  ERROR: {result.get('error', 'Unknown')}")

    # Print summary table
    print_results_table(results)

    # Recommendation
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        fastest = min(successful, key=lambda x: x["inference_times_ms"]["mean"])
        print(f"\nFastest model: {fastest['model']}")
        print(f"  Mean inference: {fastest['inference_times_ms']['mean']:.1f}ms")

        # Check if meets <50ms target
        if fastest["inference_times_ms"]["mean"] < 50:
            print(f"  ✓ Meets <50ms target on {device}")
        else:
            print(f"  ✗ Does not meet <50ms target on {device}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                "config": {
                    "device": device,
                    "num_runs": args.num_runs,
                    "num_texts": len(BENCHMARK_TEXTS),
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
