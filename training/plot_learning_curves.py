"""
Plot learning curves from training history.

Creates visualizations of:
- Overall MAE and RMSE over epochs
- Per-dimension MAE over epochs
- Training vs validation comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_path: Path) -> List[Dict]:
    """Load training history from JSON file."""
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_overall_metrics(history: List[Dict], output_dir: Path):
    """Plot overall MAE and RMSE over epochs."""
    epochs = [h["epoch"] for h in history]

    train_mae = [h["train"]["mae"] for h in history]
    val_mae = [h["val"]["mae"] for h in history]

    train_rmse = [h["train"]["rmse"] for h in history]
    val_rmse = [h["val"]["rmse"] for h in history]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot MAE
    ax1.plot(epochs, train_mae, 'o-', label='Train MAE', linewidth=2)
    ax1.plot(epochs, val_mae, 's-', label='Val MAE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error', fontsize=12)
    ax1.set_title('MAE over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot RMSE
    ax2.plot(epochs, train_rmse, 'o-', label='Train RMSE', linewidth=2)
    ax2.plot(epochs, val_rmse, 's-', label='Val RMSE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Root Mean Squared Error', fontsize=12)
    ax2.set_title('RMSE over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "overall_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_per_dimension_mae(history: List[Dict], dimension_names: List[str], output_dir: Path):
    """Plot per-dimension MAE over epochs."""
    epochs = [h["epoch"] for h in history]

    # Extract per-dimension MAE for train and val
    train_dim_mae = {dim: [] for dim in dimension_names}
    val_dim_mae = {dim: [] for dim in dimension_names}

    for h in history:
        for dim in dimension_names:
            train_dim_mae[dim].append(h["train"][f"{dim}_mae"])
            val_dim_mae[dim].append(h["val"][f"{dim}_mae"])

    # Create figure with subplots (2 rows, 4 cols for 8 dimensions)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, dim in enumerate(dimension_names):
        ax = axes[i]
        ax.plot(epochs, train_dim_mae[dim], 'o-', label='Train', linewidth=2)
        ax.plot(epochs, val_dim_mae[dim], 's-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.set_title(f'{dim.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "per_dimension_mae.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_loss_curves(history: List[Dict], output_dir: Path):
    """Plot training and validation loss over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(history: List[Dict], dimension_names: List[str], output_dir: Path):
    """Create a summary table of final metrics."""
    final = history[-1]

    summary = []
    summary.append("=" * 80)
    summary.append(f"Training Summary - Epoch {final['epoch']}")
    summary.append("=" * 80)
    summary.append("")

    summary.append("Overall Metrics:")
    summary.append(f"  Train MAE:  {final['train']['mae']:.4f}")
    summary.append(f"  Val MAE:    {final['val']['mae']:.4f}")
    summary.append(f"  Train RMSE: {final['train']['rmse']:.4f}")
    summary.append(f"  Val RMSE:   {final['val']['rmse']:.4f}")
    summary.append(f"  Train Loss: {final['train']['loss']:.4f}")
    summary.append(f"  Val Loss:   {final['val']['loss']:.4f}")
    summary.append("")

    summary.append("Per-Dimension Validation MAE:")
    for dim in dimension_names:
        train_mae = final['train'][f"{dim}_mae"]
        val_mae = final['val'][f"{dim}_mae"]
        gap = val_mae - train_mae
        summary.append(f"  {dim:20s}: Train={train_mae:.4f}  Val={val_mae:.4f}  Gap={gap:+.4f}")

    summary_text = "\n".join(summary)

    # Print to console
    print("\n" + summary_text)

    # Save to file
    output_path = output_dir / "training_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from training history")
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as history file)",
    )

    args = parser.parse_args()

    # Load history
    print(f"Loading training history from {args.history}")
    history = load_training_history(args.history)

    if not history:
        print("Error: Training history is empty")
        return

    # Get dimension names from first entry
    dimension_names = [
        key.replace("_mae", "")
        for key in history[0]["train"].keys()
        if key.endswith("_mae") and key not in ["mae"]
    ]

    print(f"Found {len(history)} epochs")
    print(f"Dimensions: {dimension_names}")

    # Set output directory
    output_dir = args.output_dir or args.history.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}")

    # Create plots
    plot_overall_metrics(history, output_dir)
    plot_per_dimension_mae(history, dimension_names, output_dir)
    plot_loss_curves(history, output_dir)
    create_summary_table(history, dimension_names, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
