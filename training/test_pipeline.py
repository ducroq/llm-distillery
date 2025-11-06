"""
Quick test of training pipeline with small subset and tiny model.

Tests the full pipeline end-to-end without requiring GPU or large model downloads.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import our training modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import FilterDataset, QwenFilterModel, compute_metrics, train_epoch


def create_test_dataset(output_dir: Path, num_samples: int = 10):
    """Create a tiny test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dimension_names = [
        "agency",
        "progress",
        "collective_benefit",
        "connection",
        "innovation",
        "justice",
        "resilience",
        "wonder",
    ]

    # Create dummy training examples
    examples = []
    for i in range(num_samples):
        example = {
            "id": f"test_article_{i}",
            "title": f"Test Article {i}",
            "content": f"This is test content for article {i}. " * 20,  # Make it longer
            "labels": [float(5 + (i % 3)) for _ in dimension_names],  # Scores 5-7
            "dimension_names": dimension_names,
        }
        examples.append(example)

    # Save train and val splits
    for split_name in ["train", "val"]:
        split_file = output_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    print(f"[OK] Created test dataset with {num_samples} examples in {output_dir}")
    return dimension_names


def test_dataset_loading(data_dir: Path, tokenizer):
    """Test that FilterDataset loads correctly."""
    print("\n[1/4] Testing dataset loading...")

    dataset = FilterDataset(
        data_dir / "train.jsonl",
        tokenizer,
        max_length=128,
    )

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Dimensions: {dataset.num_dimensions}")
    print(f"  Dimension names: {dataset.dimension_names}")

    # Check a sample
    sample = dataset[0]
    print(f"  Sample keys: {sample.keys()}")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Labels: {sample['labels']}")

    assert len(dataset) == 10, "Expected 10 examples"
    assert dataset.num_dimensions == 8, "Expected 8 dimensions"
    assert sample["input_ids"].shape[0] == 128, "Expected max_length=128"
    assert sample["labels"].shape[0] == 8, "Expected 8 labels"

    print("  [OK] Dataset loading works!")
    return dataset


def test_model_loading(model_name: str, num_dimensions: int):
    """Test that model loads correctly."""
    print("\n[2/4] Testing model loading...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_dimensions,
        problem_type="regression",
    )

    print(f"  Model type: {type(model).__name__}")
    print(f"  Config: {model.config.problem_type}, {model.config.num_labels} outputs")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    print("  [OK] Model loading works!")
    return model


def test_forward_pass(model, dataset, device):
    """Test forward pass and loss computation."""
    print("\n[3/4] Testing forward pass...")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    # Move to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"  Batch input shape: {input_ids.shape}")
    print(f"  Batch labels shape: {labels.shape}")

    # Forward pass
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Predictions shape: {outputs.logits.shape}")
    print(f"  Sample predictions: {outputs.logits[0].cpu().numpy()}")

    assert outputs.logits.shape == labels.shape, "Output shape mismatch"
    assert not torch.isnan(outputs.loss), "Loss is NaN"

    print("  [OK] Forward pass works!")
    return outputs


def test_metrics(predictions, labels, dimension_names):
    """Test metrics computation."""
    print("\n[4/4] Testing metrics computation...")

    metrics = compute_metrics(predictions, labels, dimension_names)

    print(f"  Overall MAE: {metrics['mae']:.4f}")
    print(f"  Overall RMSE: {metrics['rmse']:.4f}")
    print(f"  Sample dimension metrics:")
    for dim in dimension_names[:3]:
        print(f"    {dim}_mae: {metrics[f'{dim}_mae']:.4f}")

    assert "mae" in metrics, "Missing MAE"
    assert "rmse" in metrics, "Missing RMSE"
    assert all(f"{dim}_mae" in metrics for dim in dimension_names), "Missing per-dim metrics"

    print("  [OK] Metrics computation works!")


def test_training_step(model, dataset, device, dimension_names):
    """Test one training step."""
    print("\n[Bonus] Testing training step...")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Get a batch
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Training step
    model.train()
    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"  Training loss: {loss.item():.4f}")
    print(f"  Gradients computed: {any(p.grad is not None for p in model.parameters())}")

    print("  [OK] Training step works!")


def main():
    print("="*60)
    print("LLM Distillery Training Pipeline Test")
    print("="*60)

    # Setup
    test_dir = Path("training/test_output")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use a tiny model for testing (distilbert-base-uncased is ~66M params)
    model_name = "distilbert-base-uncased"
    device = torch.device("cpu")  # Use CPU for testing

    print(f"\nTest configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Test directory: {test_dir}")

    try:
        # Create test data
        dimension_names = create_test_dataset(test_dir, num_samples=10)

        # Load tokenizer
        print(f"\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  [OK] Tokenizer loaded")

        # Run tests
        dataset = test_dataset_loading(test_dir, tokenizer)
        model = test_model_loading(model_name, num_dimensions=8)

        # Test forward pass
        outputs = test_forward_pass(model, dataset, device)

        # Test metrics
        test_metrics(outputs.logits, outputs.logits, dimension_names)  # Use logits as both pred and labels for testing

        # Test training step
        test_training_step(model, dataset, device, dimension_names)

        print("\n" + "="*60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*60)
        print("\nThe training pipeline is working correctly!")
        print("You can now train on real data with:")
        print("  python -m training.train --filter filters/uplifting/v1 ...")

    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n[OK] Cleaned up test directory")


if __name__ == "__main__":
    main()
