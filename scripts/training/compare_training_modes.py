"""
Compare knowledge distillation vs instruction tuning results.

Analyzes and compares two training modes:
- Knowledge Distillation: Model learns from oracle's scores directly
- Instruction Tuning: Model learns from oracle's reasoning and generates scores

Creates:
- Comparative metrics analysis
- Per-dimension performance comparison
- Visualizations showing differences
- Markdown report with recommendations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_training_data(history_path: Path, metadata_path: Path) -> Tuple[List[Dict], Dict]:
    """Load training history and metadata."""
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return history, metadata


def compare_overall_metrics(dist_history: List[Dict], inst_history: List[Dict]) -> Dict:
    """Compare overall metrics between two training modes."""
    dist_final = dist_history[-1]
    inst_final = inst_history[-1]

    comparison = {
        'distillation': {
            'train_mae': dist_final['train']['mae'],
            'val_mae': dist_final['val']['mae'],
            'train_rmse': dist_final['train']['rmse'],
            'val_rmse': dist_final['val']['rmse'],
            'train_loss': dist_final['train']['loss'],
            'val_loss': dist_final['val']['loss'],
            'gap': dist_final['val']['mae'] - dist_final['train']['mae']
        },
        'instruction': {
            'train_mae': inst_final['train']['mae'],
            'val_mae': inst_final['val']['mae'],
            'train_rmse': inst_final['train']['rmse'],
            'val_rmse': inst_final['val']['rmse'],
            'train_loss': inst_final['train']['loss'],
            'val_loss': inst_final['val']['loss'],
            'gap': inst_final['val']['mae'] - inst_final['train']['mae']
        }
    }

    # Calculate improvements
    comparison['improvement'] = {
        'val_mae': ((inst_final['val']['mae'] - dist_final['val']['mae']) / inst_final['val']['mae']) * 100,
        'val_rmse': ((inst_final['val']['rmse'] - dist_final['val']['rmse']) / inst_final['val']['rmse']) * 100,
        'val_loss': ((inst_final['val']['loss'] - dist_final['val']['loss']) / inst_final['val']['loss']) * 100,
    }

    return comparison


def compare_per_dimension(dist_history: List[Dict], inst_history: List[Dict],
                         dimension_names: List[str]) -> Dict:
    """Compare per-dimension performance."""
    dist_final = dist_history[-1]
    inst_final = inst_history[-1]

    dimension_comparison = {}

    for dim in dimension_names:
        dist_train_mae = dist_final['train'][f'{dim}_mae']
        dist_val_mae = dist_final['val'][f'{dim}_mae']
        inst_train_mae = inst_final['train'][f'{dim}_mae']
        inst_val_mae = inst_final['val'][f'{dim}_mae']

        improvement = ((inst_val_mae - dist_val_mae) / inst_val_mae) * 100

        dimension_comparison[dim] = {
            'distillation': {
                'train_mae': dist_train_mae,
                'val_mae': dist_val_mae,
                'gap': dist_val_mae - dist_train_mae
            },
            'instruction': {
                'train_mae': inst_train_mae,
                'val_mae': inst_val_mae,
                'gap': inst_val_mae - inst_train_mae
            },
            'improvement_pct': improvement,
            'winner': 'distillation' if dist_val_mae < inst_val_mae else 'instruction'
        }

    return dimension_comparison


def plot_comparison(dist_history: List[Dict], inst_history: List[Dict],
                   dimension_names: List[str], output_dir: Path):
    """Create comparison visualizations."""

    # 1. Overall MAE comparison over epochs
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    dist_epochs = [h['epoch'] for h in dist_history]
    inst_epochs = [h['epoch'] for h in inst_history]

    dist_train_mae = [h['train']['mae'] for h in dist_history]
    dist_val_mae = [h['val']['mae'] for h in dist_history]
    inst_train_mae = [h['train']['mae'] for h in inst_history]
    inst_val_mae = [h['val']['mae'] for h in inst_history]

    ax.plot(dist_epochs, dist_train_mae, 'o-', label='Distillation Train',
            linewidth=2, color='#1f77b4')
    ax.plot(dist_epochs, dist_val_mae, 's-', label='Distillation Val',
            linewidth=2, color='#ff7f0e')
    ax.plot(inst_epochs, inst_train_mae, '^-', label='Instruction Train',
            linewidth=2, color='#2ca02c')
    ax.plot(inst_epochs, inst_val_mae, 'd-', label='Instruction Val',
            linewidth=2, color='#d62728')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Knowledge Distillation vs Instruction Tuning - MAE Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mode_comparison_mae.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 2. Per-dimension comparison (final epoch)
    dist_final = dist_history[-1]
    inst_final = inst_history[-1]

    dist_dim_mae = [dist_final['val'][f'{dim}_mae'] for dim in dimension_names]
    inst_dim_mae = [inst_final['val'][f'{dim}_mae'] for dim in dimension_names]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    x = np.arange(len(dimension_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, dist_dim_mae, width, label='Distillation', color='#1f77b4')
    bars2 = ax.bar(x + width/2, inst_dim_mae, width, label='Instruction', color='#2ca02c')

    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Validation MAE', fontsize=12)
    ax.set_title('Per-Dimension Performance Comparison (Final Epoch)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dim.replace('_', '\n') for dim in dimension_names],
                       rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "mode_comparison_per_dimension.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 3. Improvement percentage chart
    improvements = [(inst_final['val'][f'{dim}_mae'] - dist_final['val'][f'{dim}_mae']) /
                    inst_final['val'][f'{dim}_mae'] * 100
                    for dim in dimension_names]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    bars = ax.barh(dimension_names, improvements, color=colors)

    ax.set_xlabel('Improvement % (Positive = Distillation Better)', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    ax.set_title('Knowledge Distillation Improvement Over Instruction Tuning',
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{imp:+.1f}%',
               ha='left' if width > 0 else 'right',
               va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "mode_comparison_improvement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_markdown_report(overall_comparison: Dict, dimension_comparison: Dict,
                            dist_metadata: Dict, inst_metadata: Dict,
                            output_path: Path):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# Training Mode Comparison Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")

    # Determine winner
    dist_better = overall_comparison['improvement']['val_mae'] > 0
    winner = "Knowledge Distillation" if dist_better else "Instruction Tuning"
    improvement = abs(overall_comparison['improvement']['val_mae'])

    report.append(f"**Winner:** {winner}")
    report.append(f"**Improvement:** {improvement:.1f}% better validation MAE")
    report.append("")

    # Quick stats
    dist_mae = overall_comparison['distillation']['val_mae']
    inst_mae = overall_comparison['instruction']['val_mae']

    report.append("| Metric | Distillation | Instruction | Better |")
    report.append("|--------|--------------|-------------|--------|")
    report.append(f"| Val MAE | {dist_mae:.4f} | {inst_mae:.4f} | "
                 f"{'✅ Distillation' if dist_mae < inst_mae else '✅ Instruction'} |")
    report.append(f"| Val RMSE | {overall_comparison['distillation']['val_rmse']:.4f} | "
                 f"{overall_comparison['instruction']['val_rmse']:.4f} | "
                 f"{'✅ Distillation' if overall_comparison['distillation']['val_rmse'] < overall_comparison['instruction']['val_rmse'] else '✅ Instruction'} |")
    report.append(f"| Train/Val Gap | {overall_comparison['distillation']['gap']:+.4f} | "
                 f"{overall_comparison['instruction']['gap']:+.4f} | "
                 f"{'✅ Distillation' if abs(overall_comparison['distillation']['gap']) < abs(overall_comparison['instruction']['gap']) else '✅ Instruction'} |")
    report.append("")

    report.append("## Training Configuration Differences")
    report.append("")
    report.append("| Setting | Distillation | Instruction |")
    report.append("|---------|--------------|-------------|")
    report.append(f"| Training Mode | {dist_metadata['training_mode']} | {inst_metadata['training_mode']} |")
    report.append(f"| Include Prompt | {dist_metadata['include_prompt']} | {inst_metadata['include_prompt']} |")
    report.append(f"| Max Length | {dist_metadata['max_length']} tokens | {inst_metadata['max_length']} tokens |")
    report.append(f"| Best Val MAE | {dist_metadata['best_val_mae']:.4f} | {inst_metadata['best_val_mae']:.4f} |")
    report.append("")

    report.append("## Overall Metrics Comparison")
    report.append("")
    report.append("### Final Epoch Results")
    report.append("")
    report.append("#### Knowledge Distillation")
    report.append("")
    report.append(f"- **Train MAE:** {overall_comparison['distillation']['train_mae']:.4f}")
    report.append(f"- **Val MAE:** {overall_comparison['distillation']['val_mae']:.4f}")
    report.append(f"- **Train RMSE:** {overall_comparison['distillation']['train_rmse']:.4f}")
    report.append(f"- **Val RMSE:** {overall_comparison['distillation']['val_rmse']:.4f}")
    report.append(f"- **Train/Val Gap:** {overall_comparison['distillation']['gap']:+.4f}")
    report.append("")

    report.append("#### Instruction Tuning")
    report.append("")
    report.append(f"- **Train MAE:** {overall_comparison['instruction']['train_mae']:.4f}")
    report.append(f"- **Val MAE:** {overall_comparison['instruction']['val_mae']:.4f}")
    report.append(f"- **Train RMSE:** {overall_comparison['instruction']['train_rmse']:.4f}")
    report.append(f"- **Val RMSE:** {overall_comparison['instruction']['val_rmse']:.4f}")
    report.append(f"- **Train/Val Gap:** {overall_comparison['instruction']['gap']:+.4f}")
    report.append("")

    report.append("## Per-Dimension Analysis")
    report.append("")
    report.append("| Dimension | Distillation Val MAE | Instruction Val MAE | Improvement % | Winner |")
    report.append("|-----------|---------------------|---------------------|---------------|--------|")

    for dim, comp in dimension_comparison.items():
        dim_display = dim.replace('_', ' ').title()
        dist_val = comp['distillation']['val_mae']
        inst_val = comp['instruction']['val_mae']
        improvement = comp['improvement_pct']
        winner_icon = '✅' if comp['winner'] == 'distillation' else '⚠️'

        report.append(f"| {dim_display} | {dist_val:.4f} | {inst_val:.4f} | "
                     f"{improvement:+.1f}% | {winner_icon} {comp['winner'].title()} |")

    report.append("")

    report.append("## Key Findings")
    report.append("")

    # Count dimension wins
    dist_wins = sum(1 for comp in dimension_comparison.values() if comp['winner'] == 'distillation')
    inst_wins = len(dimension_comparison) - dist_wins

    report.append(f"1. **Dimension Performance:** Distillation won {dist_wins}/{len(dimension_comparison)} dimensions")
    report.append(f"2. **Overall Accuracy:** Distillation MAE is {improvement:.1f}% better than instruction tuning")

    # Analyze overfitting
    dist_gap = abs(overall_comparison['distillation']['gap'])
    inst_gap = abs(overall_comparison['instruction']['gap'])

    if overall_comparison['distillation']['gap'] > 0.1:
        report.append(f"3. **Overfitting:** Distillation shows slight overfitting (gap: {overall_comparison['distillation']['gap']:+.4f})")
    elif overall_comparison['instruction']['gap'] < -0.1:
        report.append(f"3. **Underfitting:** Instruction tuning shows underfitting (gap: {overall_comparison['instruction']['gap']:+.4f})")
    else:
        report.append("3. **Generalization:** Both models show good generalization")

    report.append(f"4. **Best Absolute Performance:** {winner} achieved the lowest validation MAE ({min(dist_mae, inst_mae):.4f})")
    report.append("")

    report.append("## Recommendations")
    report.append("")

    if dist_better:
        report.append("### ✅ Use Knowledge Distillation for Production")
        report.append("")
        report.append("**Reasons:**")
        report.append(f"- {improvement:.1f}% better validation accuracy")
        report.append("- More efficient training (no prompt overhead)")
        report.append("- Lower inference cost (512 vs 1024 token limit)")
        report.append("- Direct score learning is more effective for this task")
        report.append("")
        report.append("**When to consider Instruction Tuning:**")
        report.append("- If interpretability of reasoning is critical")
        report.append("- If you need the model to explain its scores")
        report.append("- For multi-task learning scenarios")
    else:
        report.append("### ✅ Use Instruction Tuning for Production")
        report.append("")
        report.append("**Reasons:**")
        report.append(f"- {improvement:.1f}% better validation accuracy")
        report.append("- Better generalization (negative train/val gap)")
        report.append("- More robust to distribution shift")
        report.append("- Learns reasoning patterns, not just scores")
        report.append("")
        report.append("**Trade-offs:**")
        report.append("- Higher inference cost (1024 vs 512 tokens)")
        report.append("- Slower training due to prompt overhead")

    report.append("")
    report.append("## Visualizations")
    report.append("")
    report.append("See generated plots:")
    report.append("- `mode_comparison_mae.png` - Overall MAE over epochs")
    report.append("- `mode_comparison_per_dimension.png` - Per-dimension performance")
    report.append("- `mode_comparison_improvement.png` - Improvement breakdown")
    report.append("")

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nMarkdown report saved to: {output_path}")


def print_console_summary(overall_comparison: Dict, dimension_comparison: Dict):
    """Print summary to console."""

    print("\n" + "=" * 80)
    print("TRAINING MODE COMPARISON SUMMARY")
    print("=" * 80)
    print()

    print("OVERALL METRICS (Final Epoch):")
    print("-" * 80)
    print(f"{'Metric':<20} {'Distillation':>15} {'Instruction':>15} {'Improvement':>15}")
    print("-" * 80)

    dist = overall_comparison['distillation']
    inst = overall_comparison['instruction']

    print(f"{'Val MAE':<20} {dist['val_mae']:>15.4f} {inst['val_mae']:>15.4f} "
          f"{overall_comparison['improvement']['val_mae']:>14.1f}%")
    print(f"{'Val RMSE':<20} {dist['val_rmse']:>15.4f} {inst['val_rmse']:>15.4f} "
          f"{overall_comparison['improvement']['val_rmse']:>14.1f}%")
    print(f"{'Train/Val Gap':<20} {dist['gap']:>15.4f} {inst['gap']:>15.4f}")

    print()
    print("PER-DIMENSION VALIDATION MAE:")
    print("-" * 80)
    print(f"{'Dimension':<25} {'Distillation':>15} {'Instruction':>15} {'Winner':>15}")
    print("-" * 80)

    for dim, comp in dimension_comparison.items():
        dim_display = dim.replace('_', ' ').title()
        winner = '[' + comp['winner'].title() + ']'
        print(f"{dim_display:<25} {comp['distillation']['val_mae']:>15.4f} "
              f"{comp['instruction']['val_mae']:>15.4f} {winner:>20}")

    print()

    # Determine overall winner
    dist_better = overall_comparison['improvement']['val_mae'] > 0
    winner = "KNOWLEDGE DISTILLATION" if dist_better else "INSTRUCTION TUNING"
    improvement = abs(overall_comparison['improvement']['val_mae'])

    print("=" * 80)
    print(f"WINNER: {winner} ({improvement:.1f}% better validation MAE)")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare knowledge distillation vs instruction tuning"
    )
    parser.add_argument(
        "--distillation-history",
        type=Path,
        required=True,
        help="Path to distillation training_history.json"
    )
    parser.add_argument(
        "--distillation-metadata",
        type=Path,
        required=True,
        help="Path to distillation training_metadata.json"
    )
    parser.add_argument(
        "--instruction-history",
        type=Path,
        required=True,
        help="Path to instruction training_history.json"
    )
    parser.add_argument(
        "--instruction-metadata",
        type=Path,
        required=True,
        help="Path to instruction training_metadata.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports and plots (default: distillation filter's training_reports/)"
    )

    args = parser.parse_args()

    # Load data
    print("Loading training data...")
    dist_history, dist_metadata = load_training_data(
        args.distillation_history, args.distillation_metadata
    )
    inst_history, inst_metadata = load_training_data(
        args.instruction_history, args.instruction_metadata
    )

    # Get dimension names
    dimension_names = dist_metadata['dimension_names']

    print(f"Comparing {len(dist_history)} epochs of training data...")
    print(f"Dimensions: {len(dimension_names)}")

    # Compare metrics
    overall_comparison = compare_overall_metrics(dist_history, inst_history)
    dimension_comparison = compare_per_dimension(dist_history, inst_history, dimension_names)

    # Set default output directory to distillation filter's training_reports
    if args.output_dir is None:
        dist_parent = args.distillation_history.parent
        if "filters" in args.distillation_history.parts:
            args.output_dir = dist_parent / "training_reports"
        else:
            args.output_dir = Path("reports") / "training_mode_comparison"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    plot_comparison(dist_history, inst_history, dimension_names, args.output_dir)

    # Generate markdown report
    print("\nGenerating markdown report...")
    report_path = args.output_dir / "comparison_report.md"
    generate_markdown_report(
        overall_comparison, dimension_comparison,
        dist_metadata, inst_metadata,
        report_path
    )

    # Print console summary
    print_console_summary(overall_comparison, dimension_comparison)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
