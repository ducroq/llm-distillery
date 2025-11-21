"""
Benchmark trained models on test sets.

Loads trained LoRA models and evaluates them on test data,
comparing student model predictions against oracle ground truth.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, AutoPeftModelForSequenceClassification, PeftConfig, get_peft_model


class FilterDataset(Dataset):
    """PyTorch Dataset for filter test data."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 512,
        prompt: str = None,
    ):
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

        # Optionally prepend prompt
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
            "article_id": example.get("article_id", f"article_{idx}"),
        }


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, dimension_names: List[str]) -> Dict:
    """Compute evaluation metrics."""
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


def evaluate_model(model, dataloader, device, dimension_names: List[str]):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_labels = []
    all_article_ids = []

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
            )

            predictions = outputs.logits

            # Track results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_article_ids.extend(batch["article_id"])

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels, dimension_names)

    # Store detailed predictions for analysis
    detailed_results = {
        "predictions": all_predictions.numpy().tolist(),
        "labels": all_labels.numpy().tolist(),
        "article_ids": all_article_ids,
        "dimension_names": dimension_names,
    }

    return metrics, detailed_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained model on test set")
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter directory with trained model (e.g., filters/investment-risk/v4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset directory (with test.jsonl)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to trained model directory (default: {filter}/model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark results (default: {filter}/benchmarks)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )

    args = parser.parse_args()

    # Set model and output directories
    if args.model_dir is None:
        args.model_dir = args.filter / "model"
    if args.output_dir is None:
        args.output_dir = args.filter / "benchmarks"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Test Set Benchmarking")
    print("=" * 80)
    print(f"Filter: {args.filter}")
    print(f"Model: {args.model_dir}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Load filter config
    config_path = args.filter / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filter_name = config["filter"]["name"]
    filter_version = config["filter"]["version"]
    dimension_names = list(config["scoring"]["dimensions"].keys())
    num_dimensions = len(dimension_names)

    print(f"Filter: {filter_name} v{filter_version}")
    print(f"Dimensions ({num_dimensions}): {dimension_names}")
    print()

    # Load training metadata to get base model info
    metadata_path = args.filter / "training_metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: Training metadata not found: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    base_model_name = metadata["model_name"]
    include_prompt = metadata.get("include_prompt", False)

    print(f"Base model: {base_model_name}")
    print(f"Include prompt: {include_prompt}")
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompt if needed
    prompt = None
    if include_prompt:
        prompt_path = args.filter / "prompt-compressed.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print(f"Loaded prompt: {len(prompt)} characters")

    # Load test dataset
    test_path = args.data_dir / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: Test set not found: {test_path}")
        sys.exit(1)

    print(f"\nLoading test set: {test_path}")
    test_dataset = FilterDataset(
        test_path,
        tokenizer,
        max_length=512,
        prompt=prompt,
    )
    print(f"Test examples: {len(test_dataset)}")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load model with LoRA adapters
    print(f"\nLoading model with LoRA adapters from: {args.model_dir}")
    model_path = args.model_dir.resolve()

    # Workaround for PEFT's HF Hub validation: Load config first, then manually load weights
    print("Loading PEFT config...")
    peft_config = PeftConfig.from_pretrained(str(model_path))

    print(f"Loading base model: {peft_config.base_model_name_or_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=num_dimensions,
        problem_type="regression",
    )

    # Set pad_token_id on model config to match tokenizer
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model pad_token_id to: {tokenizer.pad_token_id}")

    # Load adapter weights including modules_to_save (score layer)
    print("Loading LoRA adapter weights...")
    from safetensors.torch import load_file

    adapter_weights_path = model_path / "adapter_model.safetensors"
    if not adapter_weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_weights_path}")

    # Load the full state dict
    adapter_state_dict = load_file(str(adapter_weights_path))
    print(f"Loaded {len(adapter_state_dict)} weight tensors from adapter file")

    # Debug: Print score layer keys in saved weights
    print("\nDEBUG: Score layer keys in saved weights:")
    score_keys = [k for k in adapter_state_dict.keys() if 'score' in k.lower()]
    for k in score_keys:
        print(f"  {k}: shape {adapter_state_dict[k].shape}")
    if not score_keys:
        print("  (No score keys found)")

    # Fix key names for PEFT version compatibility
    # Old format: lora_A.weight -> New format: lora_A.default.weight
    # Score layer: base_model.model.score.weight -> base_model.model.score.modules_to_save.default.weight
    print("\nRemapping keys for PEFT compatibility...")
    remapped_state_dict = {}
    lora_keys_remapped = 0
    score_keys_remapped = 0
    for key, value in adapter_state_dict.items():
        # Fix LoRA adapter keys
        if ".lora_A.weight" in key or ".lora_B.weight" in key:
            # Insert .default before .weight
            new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped_state_dict[new_key] = value
            lora_keys_remapped += 1
        # Fix score layer (modules_to_save)
        elif key == "base_model.model.score.weight":
            new_key = "base_model.model.score.modules_to_save.default.weight"
            remapped_state_dict[new_key] = value
            score_keys_remapped += 1
        elif key == "base_model.model.score.bias":
            new_key = "base_model.model.score.modules_to_save.default.bias"
            remapped_state_dict[new_key] = value
            score_keys_remapped += 1
        else:
            remapped_state_dict[key] = value

    print(f"  Remapped {lora_keys_remapped} LoRA weight keys")
    print(f"  Remapped {score_keys_remapped} score layer keys")

    # Apply PEFT to base model
    print("\nCreating PEFT model structure...")
    model = get_peft_model(base_model, peft_config)

    # Debug: Print score layer keys in model structure
    print("\nDEBUG: Score layer keys in model structure:")
    model_score_keys = [k for k, v in model.named_parameters() if 'score' in k.lower()]
    for k in model_score_keys:
        print(f"  {k}")
    if not model_score_keys:
        print("  (No score keys found)")

    print("\nDEBUG: All module names in model:")
    all_modules = [name for name, _ in model.named_modules() if 'score' in name.lower()]
    for name in all_modules:
        print(f"  {name}")

    # Load adapter weights with detailed logging
    print("\nLoading remapped weights into model...")
    incompatible = model.load_state_dict(remapped_state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"  Warning: {len(incompatible.missing_keys)} missing keys")
        # Show first few
        for key in list(incompatible.missing_keys)[:5]:
            print(f"    - {key}")
        if len(incompatible.missing_keys) > 5:
            print(f"    ... and {len(incompatible.missing_keys) - 5} more")

    if incompatible.unexpected_keys:
        print(f"  Warning: {len(incompatible.unexpected_keys)} unexpected keys")
        # Show first few
        for key in list(incompatible.unexpected_keys)[:5]:
            print(f"    - {key}")
        if len(incompatible.unexpected_keys) > 5:
            print(f"    ... and {len(incompatible.unexpected_keys) - 5} more")

    print("Weights loaded successfully")

    model = model.to(device)

    print(f"Model loaded successfully")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    print()
    metrics, detailed_results = evaluate_model(model, test_dataloader, device, dimension_names)

    # Print results
    print("=" * 80)
    print("Test Set Results")
    print("=" * 80)
    print()
    print(f"Overall Metrics:")
    print(f"  Test MAE:  {metrics['mae']:.4f}")
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print()
    print(f"Per-Dimension MAE:")

    for dim_name in dimension_names:
        mae = metrics[f"{dim_name}_mae"]
        rmse = metrics[f"{dim_name}_rmse"]
        print(f"  {dim_name:30s}: MAE={mae:.4f}  RMSE={rmse:.4f}")

    print()

    # Save results
    results_file = args.output_dir / "test_set_results.json"
    print(f"Saving results to: {results_file}")

    results = {
        "filter_name": filter_name,
        "filter_version": filter_version,
        "base_model": base_model_name,
        "test_examples": len(test_dataset),
        "dimension_names": dimension_names,
        "metrics": metrics,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save detailed predictions
    predictions_file = args.output_dir / "test_set_predictions.json"
    print(f"Saving detailed predictions to: {predictions_file}")

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    print()
    print("=" * 80)
    print("Benchmarking Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
