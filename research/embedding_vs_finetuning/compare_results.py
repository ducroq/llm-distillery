"""
Generate comparison reports and visualizations for embedding experiments.

This script loads evaluation results and generates markdown reports,
comparison tables, and visualizations.

Usage:
    python research/embedding_vs_finetuning/compare_results.py \
        --dataset uplifting_v5 \
        --output-dir research/embedding_vs_finetuning/results
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_results(results_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate markdown summary table."""
    baseline_mae = results.get('baseline_mae', None)

    lines = [
        "## Summary Results",
        "",
        f"**Dataset:** {results.get('dataset', 'unknown')}",
        f"**Baseline (Fine-tuned Qwen2.5-1.5B) MAE:** {baseline_mae:.4f}" if baseline_mae else "",
        "",
        "| Embedding Model | Probe | MAE | RMSE | Spearman | vs Baseline |",
        "|-----------------|-------|-----|------|----------|-------------|"
    ]

    # Collect all results for sorting
    all_rows = []

    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            continue

        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type not in model_results:
                continue

            probe_results = model_results[probe_type]
            if 'error' in probe_results:
                continue

            mae = probe_results.get('overall_mae', None)
            rmse = probe_results.get('overall_rmse', None)
            spearman = probe_results.get('overall_spearman', None)

            if mae is None:
                continue

            if baseline_mae:
                delta = mae - baseline_mae
                delta_str = f"{delta:+.4f}"
                delta_pct = (delta / baseline_mae) * 100
                delta_str += f" ({delta_pct:+.1f}%)"
            else:
                delta_str = "N/A"
                delta = float('inf')

            all_rows.append({
                'model': model_name,
                'probe': probe_type,
                'mae': mae,
                'rmse': rmse,
                'spearman': spearman,
                'delta': delta,
                'delta_str': delta_str
            })

    # Sort by MAE
    all_rows.sort(key=lambda x: x['mae'])

    for row in all_rows:
        # Highlight best result
        marker = " **" if row == all_rows[0] else ""
        marker_end = "**" if row == all_rows[0] else ""

        lines.append(
            f"| {marker}{row['model']}{marker_end} | {row['probe']} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['spearman']:.4f} | {row['delta_str']} |"
        )

    return "\n".join(lines)


def generate_per_dimension_table(results: Dict[str, Any], model_name: str, probe_type: str) -> str:
    """Generate per-dimension breakdown table."""
    model_results = results.get('models', {}).get(model_name, {})
    probe_results = model_results.get(probe_type, {})

    if 'error' in probe_results or 'per_dimension' not in probe_results:
        return f"No per-dimension data available for {model_name} / {probe_type}"

    per_dim = probe_results['per_dimension']

    lines = [
        f"### Per-Dimension Results: {model_name} ({probe_type})",
        "",
        "| Dimension | MAE | RMSE | Spearman |",
        "|-----------|-----|------|----------|"
    ]

    for dim_name, metrics in per_dim.items():
        lines.append(
            f"| {dim_name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['spearman']:.4f} |"
        )

    return "\n".join(lines)


def generate_best_model_analysis(results: Dict[str, Any]) -> str:
    """Analyze and report on the best performing models."""
    baseline_mae = results.get('baseline_mae', None)

    # Find best results for each category
    best_linear = None
    best_mlp = None
    best_overall = None

    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            continue

        # Check Ridge (linear)
        if 'ridge' in model_results and 'error' not in model_results['ridge']:
            mae = model_results['ridge'].get('overall_mae')
            if mae and (best_linear is None or mae < best_linear['mae']):
                best_linear = {'model': model_name, 'probe': 'ridge', 'mae': mae}

        # Check MLP
        if 'mlp' in model_results and 'error' not in model_results['mlp']:
            mae = model_results['mlp'].get('overall_mae')
            if mae and (best_mlp is None or mae < best_mlp['mae']):
                best_mlp = {'model': model_name, 'probe': 'mlp', 'mae': mae}

        # Check all probes for overall best
        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type in model_results and 'error' not in model_results[probe_type]:
                mae = model_results[probe_type].get('overall_mae')
                if mae and (best_overall is None or mae < best_overall['mae']):
                    best_overall = {'model': model_name, 'probe': probe_type, 'mae': mae}

    lines = [
        "## Best Model Analysis",
        ""
    ]

    if best_linear:
        lines.append(f"**Best Linear Probe:** {best_linear['model']} (Ridge) - MAE: {best_linear['mae']:.4f}")
    if best_mlp:
        lines.append(f"**Best MLP Probe:** {best_mlp['model']} (MLP) - MAE: {best_mlp['mae']:.4f}")
    if best_overall:
        lines.append(f"**Best Overall:** {best_overall['model']} ({best_overall['probe']}) - MAE: {best_overall['mae']:.4f}")

    lines.append("")

    if baseline_mae and best_overall:
        delta = best_overall['mae'] - baseline_mae
        delta_pct = (delta / baseline_mae) * 100

        if delta <= 0:
            lines.append(f"**Result:** Embedding approach **matches or beats** fine-tuned baseline!")
            lines.append(f"- Improvement: {-delta:.4f} ({-delta_pct:.1f}%)")
        elif delta_pct <= 5:
            lines.append(f"**Result:** Embedding approach is **within 5%** of fine-tuned baseline.")
            lines.append(f"- Gap: {delta:.4f} ({delta_pct:.1f}%)")
            lines.append("- Consider for production if inference speed is critical.")
        elif delta_pct <= 10:
            lines.append(f"**Result:** Embedding approach is **within 10%** of fine-tuned baseline.")
            lines.append(f"- Gap: {delta:.4f} ({delta_pct:.1f}%)")
            lines.append("- May be acceptable for some use cases.")
        else:
            lines.append(f"**Result:** Fine-tuning provides **significant advantage** over embedding approach.")
            lines.append(f"- Gap: {delta:.4f} ({delta_pct:.1f}%)")
            lines.append("- Recommend continuing with fine-tuning approach.")

    # Check if MLP beats linear significantly
    if best_linear and best_mlp:
        mlp_improvement = best_linear['mae'] - best_mlp['mae']
        if mlp_improvement > 0.05:
            lines.append("")
            lines.append(f"**Note:** MLP beats linear by {mlp_improvement:.4f}, suggesting non-linear combinations help.")
        elif mlp_improvement < -0.02:
            lines.append("")
            lines.append("**Note:** Linear probe outperforms MLP - non-linear complexity not needed.")

    return "\n".join(lines)


def generate_recommendations(results: Dict[str, Any]) -> str:
    """Generate actionable recommendations based on results."""
    baseline_mae = results.get('baseline_mae', None)

    # Find best result
    best_mae = float('inf')
    best_config = None

    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            continue

        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type in model_results and 'error' not in model_results[probe_type]:
                mae = model_results[probe_type].get('overall_mae')
                if mae and mae < best_mae:
                    best_mae = mae
                    best_config = {'model': model_name, 'probe': probe_type, 'mae': mae}

    lines = [
        "## Recommendations",
        ""
    ]

    if baseline_mae and best_config:
        gap_pct = ((best_mae - baseline_mae) / baseline_mae) * 100

        if gap_pct <= 0:
            lines.extend([
                "### Use Embedding Approach",
                "",
                "The embedding approach matches or beats fine-tuning. Consider:",
                f"1. Deploy {best_config['model']} + {best_config['probe']} for production",
                "2. Benefits: Faster embedding (once), simpler training, smaller models",
                "3. Test on additional datasets to confirm generalization"
            ])
        elif gap_pct <= 10:
            lines.extend([
                "### Consider Embedding Approach for Specific Use Cases",
                "",
                f"The embedding approach (MAE: {best_mae:.4f}) is within {gap_pct:.1f}% of fine-tuning.",
                "",
                "**Use embeddings when:**",
                "- Inference speed is critical",
                "- Training data is limited (probe needs less data than fine-tuning)",
                "- Need to deploy many filters (embeddings can be shared)",
                "",
                "**Stick with fine-tuning when:**",
                "- Maximum accuracy is required",
                "- Filter domain is specialized"
            ])
        else:
            lines.extend([
                "### Continue with Fine-Tuning",
                "",
                f"Fine-tuning provides a {gap_pct:.1f}% advantage over embedding approaches.",
                "",
                "**Potential improvements to try:**",
                "1. Try larger embedding models (e.g., e5-mistral-7b-instruct)",
                "2. Experiment with task-specific embedding fine-tuning",
                "3. Ensemble multiple embedding models",
                "4. Add domain-specific pre-training"
            ])

    # Add efficiency note
    lines.extend([
        "",
        "### Efficiency Considerations",
        "",
        "| Approach | Embedding Time | Probe Training | Inference |",
        "|----------|---------------|----------------|-----------|",
        "| Fine-tuned Qwen | N/A | 2-3 hours | 20-50ms |",
        "| Embedding + Probe | ~1 min/1000 articles | ~1 min | <1ms |"
    ])

    return "\n".join(lines)


def generate_full_report(results: Dict[str, Any]) -> str:
    """Generate complete markdown report."""
    lines = [
        "# Embedding vs Fine-Tuning Experiment Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Overview",
        "",
        "This experiment compares frozen embeddings + linear/MLP probes against",
        "fine-tuned Qwen2.5-1.5B for semantic dimension scoring.",
        "",
        generate_summary_table(results),
        "",
        generate_best_model_analysis(results),
        "",
    ]

    # Add per-dimension analysis for top models
    best_results = []
    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            continue
        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type in model_results and 'error' not in model_results[probe_type]:
                mae = model_results[probe_type].get('overall_mae')
                if mae:
                    best_results.append((mae, model_name, probe_type))

    best_results.sort()

    if best_results:
        lines.append("## Per-Dimension Analysis (Top 3 Models)")
        lines.append("")

        for _, model_name, probe_type in best_results[:3]:
            lines.append(generate_per_dimension_table(results, model_name, probe_type))
            lines.append("")

    lines.append(generate_recommendations(results))

    return "\n".join(lines)


def generate_visualization(results: Dict[str, Any], output_path: Path):
    """Generate visualization of results (bar chart)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    baseline_mae = results.get('baseline_mae', None)

    # Collect data
    data = []
    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            continue

        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type in model_results and 'error' not in model_results[probe_type]:
                mae = model_results[probe_type].get('overall_mae')
                if mae:
                    # Shorten model name for display
                    short_name = model_name.split('/')[-1]
                    if len(short_name) > 20:
                        short_name = short_name[:17] + "..."
                    data.append({
                        'name': f"{short_name}\n({probe_type})",
                        'mae': mae,
                        'probe': probe_type
                    })

    if not data:
        logger.warning("No data for visualization")
        return

    # Sort by MAE
    data.sort(key=lambda x: x['mae'])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [d['name'] for d in data]
    maes = [d['mae'] for d in data]

    # Color by probe type
    colors = {'ridge': '#2196F3', 'mlp': '#4CAF50', 'lightgbm': '#FF9800'}
    bar_colors = [colors[d['probe']] for d in data]

    bars = ax.bar(range(len(data)), maes, color=bar_colors)

    # Add baseline line
    if baseline_mae:
        ax.axhline(y=baseline_mae, color='red', linestyle='--', linewidth=2,
                   label=f'Fine-tuned baseline ({baseline_mae:.3f})')
        ax.legend()

    ax.set_xlabel('Model + Probe')
    ax.set_ylabel('MAE')
    ax.set_title('Embedding Model Comparison (Lower is Better)')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison reports for embedding experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--results-dir', type=str, default='research/embedding_vs_finetuning/results',
                       help='Directory with evaluation results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for reports (default: same as results-dir)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    # Load evaluation results
    results_path = results_dir / f"{args.dataset}_evaluation_results.json"
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run evaluate.py first to generate evaluation results.")
        return

    results = load_evaluation_results(results_path)

    # Generate markdown report
    report = generate_full_report(results)
    report_path = output_dir / f"{args.dataset}_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Generated report: {report_path}")

    # Generate visualization
    viz_path = output_dir / f"{args.dataset}_comparison_chart.png"
    generate_visualization(results, viz_path)

    # Print summary to console
    print("\n" + "=" * 80)
    print(report)


if __name__ == '__main__':
    main()
