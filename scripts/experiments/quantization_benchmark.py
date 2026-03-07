"""
Quick benchmark: PyTorch dynamic quantization on Gemma-3-1B + LoRA.

Compares fp32/bf16 vs INT8 dynamic quantization:
- Model size on disk
- Inference speed (CPU only)
- MAE on validation set

Usage (on gpu-server):
    PYTHONPATH=. python scripts/experiments/quantization_benchmark.py \
        --filter filters/uplifting/v6 \
        --val-data datasets/training/uplifting_v6/val.jsonl
"""

import argparse
import json
import time
import tempfile
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

from filters.common.model_loading import load_base_model_for_seq_cls


def load_merged_model(filter_dir: Path):
    """Load base model with LoRA merged in."""
    from peft import PeftModel

    model_dir = filter_dir / "model"
    config_path = model_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]

    # Load config.yaml for num_dimensions
    import yaml
    with open(filter_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    dims = (config.get("dimensions")
            or config.get("scoring", {}).get("dimensions", {})
            or config.get("filter", {}).get("dimensions", {}))
    num_dims = len(dims)

    print(f"Loading base model: {base_model_name} ({num_dims} dims)...")
    base_model = load_base_model_for_seq_cls(
        base_model_name, num_labels=num_dims, torch_dtype=torch.float32
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(model_dir))

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer, config


def load_val_data(val_path: Path, config: dict, max_articles: int = 200):
    """Load validation articles with oracle scores."""
    dims = (config.get("dimensions")
            or config.get("scoring", {}).get("dimensions", {})
            or config.get("filter", {}).get("dimensions", {}))
    dimensions = list(dims.keys())

    articles = []
    with open(val_path) as f:
        for line in f:
            if len(articles) >= max_articles:
                break
            article = json.loads(line)

            # Training data format: labels + dimension_names
            if "labels" in article and "dimension_names" in article:
                oracle_scores = article["labels"]
                if len(oracle_scores) != len(dimensions):
                    continue
            else:
                continue

            title = article.get("title", "")
            content = article.get("content", article.get("text", ""))
            text = f"{title}\n\n{content}" if title else content

            articles.append({
                "text": text,
                "oracle_scores": oracle_scores,
            })

    print(f"Loaded {len(articles)} validation articles")
    return articles, dimensions


def prepare_input(text: str, tokenizer, max_length: int = 512):
    """Tokenize with head+tail truncation (256+256)."""
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    half = max_length // 2

    if len(tokens) > max_length - 2:  # room for special tokens
        head = tokens[:half]
        tail = tokens[-half:]
        tokens = head + tail

    # Re-encode with special tokens
    encoding = tokenizer(
        tokenizer.decode(tokens),
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return encoding


def run_inference(model, tokenizer, articles, device="cpu", batch_label=""):
    """Run inference and return predictions + timing."""
    model.eval()
    model.to(device)

    predictions = []
    times = []

    with torch.no_grad():
        for i, article in enumerate(articles):
            inputs = prepare_input(article["text"], tokenizer)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start = time.perf_counter()
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            elapsed = time.perf_counter() - start

            predictions.append(logits)
            times.append(elapsed)

            if (i + 1) % 50 == 0:
                avg_ms = np.mean(times[-50:]) * 1000
                print(f"  [{batch_label}] {i+1}/{len(articles)} — avg {avg_ms:.0f}ms/article")

    return np.array(predictions), np.array(times)


def compute_metrics(predictions, articles, dimensions):
    """Compute MAE per dimension and weighted average."""
    oracle = np.array([a["oracle_scores"] for a in articles])

    # Clip predictions to 0-10
    predictions = np.clip(predictions, 0, 10)

    per_dim_mae = {}
    for i, dim in enumerate(dimensions):
        mae = np.mean(np.abs(predictions[:, i] - oracle[:, i]))
        per_dim_mae[dim] = mae

    overall_mae = np.mean(list(per_dim_mae.values()))
    return per_dim_mae, overall_mae


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1e6


def quantize_dynamic(model):
    """Apply PyTorch dynamic quantization (INT8 weights)."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # quantize all Linear layers
        dtype=torch.qint8,
    )
    return quantized


def quantize_dynamic_mlp_only(model):
    """Quantize only MLP (feed-forward) layers, leaving attention intact."""
    # Collect only MLP linear layers
    mlp_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".mlp." in name:
            mlp_modules.add(name)

    print(f"  Quantizing {len(mlp_modules)} MLP layers (leaving attention intact)")

    # PyTorch quantize_dynamic doesn't support per-name filtering,
    # so we manually quantize specific submodules
    import torch.ao.nn.quantized.dynamic as nnqd
    for name in mlp_modules:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old_linear = getattr(parent, parts[-1])
        new_linear = nnqd.Linear(
            old_linear.in_features, old_linear.out_features,
            bias=old_linear.bias is not None, dtype=torch.qint8,
        )
        new_linear.set_weight_bias(
            torch.quantize_per_tensor(old_linear.weight,
                                       scale=old_linear.weight.abs().max() / 127,
                                       zero_point=0, dtype=torch.qint8),
            old_linear.bias
        )
        setattr(parent, parts[-1], new_linear)

    return model


def main():
    parser = argparse.ArgumentParser(description="Quantization benchmark")
    parser.add_argument("--filter", required=True, help="Filter directory")
    parser.add_argument("--val-data", required=True, help="Validation JSONL")
    parser.add_argument("--max-articles", type=int, default=200, help="Max articles to score")
    parser.add_argument("--skip-fp32", action="store_true", help="Skip fp32 baseline (just run quantized)")
    args = parser.parse_args()

    filter_dir = Path(args.filter)
    val_path = Path(args.val_data)

    # Load model
    model, tokenizer, config = load_merged_model(filter_dir)

    # Load validation data
    articles, dimensions = load_val_data(val_path, config, args.max_articles)
    if not articles:
        print("ERROR: No valid articles found. Check val data path and analysis field names.")
        return

    results = {}

    # --- FP32 baseline (CPU) ---
    if not args.skip_fp32:
        print(f"\n{'='*60}")
        print("FP32 baseline (CPU)")
        print(f"{'='*60}")
        fp32_size = get_model_size_mb(model)
        print(f"Model size: {fp32_size:.0f} MB")

        preds_fp32, times_fp32 = run_inference(
            model, tokenizer, articles, device="cpu", batch_label="fp32"
        )
        per_dim_mae, overall_mae = compute_metrics(preds_fp32, articles, dimensions)

        results["fp32"] = {
            "model_size_mb": fp32_size,
            "overall_mae": overall_mae,
            "per_dim_mae": per_dim_mae,
            "avg_ms": np.mean(times_fp32) * 1000,
            "median_ms": np.median(times_fp32) * 1000,
            "p95_ms": np.percentile(times_fp32, 95) * 1000,
            "articles_per_sec": 1.0 / np.mean(times_fp32),
        }
        print(f"\nFP32 results:")
        print(f"  MAE: {overall_mae:.4f}")
        print(f"  Avg latency: {results['fp32']['avg_ms']:.0f}ms")
        print(f"  Throughput: {results['fp32']['articles_per_sec']:.2f} articles/sec")

    # --- Float16 (CPU) ---
    print(f"\n{'='*60}")
    print("Float16 (CPU)")
    print(f"{'='*60}")

    model_fp16 = model.half()
    fp16_size = get_model_size_mb(model_fp16)
    print(f"Model size: {fp16_size:.0f} MB")

    preds_fp16, times_fp16 = run_inference(
        model_fp16, tokenizer, articles, device="cpu", batch_label="fp16"
    )
    per_dim_mae_fp16, overall_mae_fp16 = compute_metrics(preds_fp16, articles, dimensions)

    results["fp16"] = {
        "model_size_mb": fp16_size,
        "overall_mae": overall_mae_fp16,
        "per_dim_mae": per_dim_mae_fp16,
        "avg_ms": np.mean(times_fp16) * 1000,
        "median_ms": np.median(times_fp16) * 1000,
        "p95_ms": np.percentile(times_fp16, 95) * 1000,
        "articles_per_sec": 1.0 / np.mean(times_fp16),
    }
    print(f"\nFP16 results:")
    print(f"  MAE: {overall_mae_fp16:.4f}")
    print(f"  Avg latency: {results['fp16']['avg_ms']:.0f}ms")
    print(f"  Throughput: {results['fp16']['articles_per_sec']:.2f} articles/sec")

    # Back to fp32 for INT8 test
    model = model.float()

    # --- INT8 dynamic quantization (CPU) ---
    print(f"\n{'='*60}")
    print("INT8 dynamic quantization (CPU)")
    print(f"{'='*60}")

    model_int8 = quantize_dynamic(model)
    int8_size = get_model_size_mb(model_int8)
    print(f"Model size: {int8_size:.0f} MB")

    preds_int8, times_int8 = run_inference(
        model_int8, tokenizer, articles, device="cpu", batch_label="int8"
    )
    per_dim_mae_int8, overall_mae_int8 = compute_metrics(preds_int8, articles, dimensions)

    results["int8"] = {
        "model_size_mb": int8_size,
        "overall_mae": overall_mae_int8,
        "per_dim_mae": per_dim_mae_int8,
        "avg_ms": np.mean(times_int8) * 1000,
        "median_ms": np.median(times_int8) * 1000,
        "p95_ms": np.percentile(times_int8, 95) * 1000,
        "articles_per_sec": 1.0 / np.mean(times_int8),
    }
    print(f"\nINT8 results:")
    print(f"  MAE: {overall_mae_int8:.4f}")
    print(f"  Avg latency: {results['int8']['avg_ms']:.0f}ms")
    print(f"  Throughput: {results['int8']['articles_per_sec']:.2f} articles/sec")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    variants = [k for k in ["fp32", "fp16", "int8"] if k in results]
    header = f"{'Metric':<25}" + "".join(f" {v:>12}" for v in variants)
    print(header)
    print("-" * len(header))

    for metric, fmt in [("model_size_mb", ".0f"), ("overall_mae", ".4f"),
                         ("avg_ms", ".0f"), ("articles_per_sec", ".2f")]:
        label = {"model_size_mb": "Model size (MB)", "overall_mae": "MAE",
                 "avg_ms": "Avg latency (ms)", "articles_per_sec": "Articles/sec"}[metric]
        row = f"{label:<25}"
        for v in variants:
            val = results[v][metric]
            row += f" {val:>12{fmt}}"
        print(row)

    # Save results
    output_path = filter_dir / "quantization_benchmark.json"
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            sk: (sv if not isinstance(sv, dict) else {dk: float(dv) for dk, dv in sv.items()})
            for sk, sv in v.items()
        }
        for sk in ["overall_mae", "avg_ms", "median_ms", "p95_ms", "articles_per_sec", "model_size_mb"]:
            if sk in serializable[k]:
                serializable[k][sk] = float(serializable[k][sk])

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
