"""Dimension redundancy analysis for a filter version — generates the conventional
dimension_analysis/ artifacts: dimension_analysis.json + 4 PNGs (correlation heatmap,
PCA variance, PCA loadings, dim clustering).

Per the prescribed workflow doc Phase 3 step "Dimension redundancy analysis" + the
artifact convention from uplifting v4 and investment-risk v4.

Usage:
    PYTHONPATH=. python scripts/analyze_dim_redundancy.py \
        --filter filters/cultural_discovery/v5 \
        --oracle-source datasets/scored/cd_v5_softpenalty_rescored_v3/cultural_discovery/scored_batch_*.jsonl \
        --oracle-name gemini_v3

For multi-oracle filters: run once per oracle, outputs go to filter_dir/dimension_analysis/{oracle_name}/
"""

import argparse
import json
import glob
import math
import sys
from pathlib import Path

# Defer heavy imports until we know we need them (matplotlib/sklearn may not be on all venvs)


def extract(val):
    if val is None:
        return None
    if isinstance(val, dict):
        s = val.get("score")
        return float(s) if s is not None else None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def load_oracle_scores(source_pattern, analysis_field, dim_names):
    files = sorted(glob.glob(source_pattern))
    if not files:
        # Maybe it's a single file not a glob
        if Path(source_pattern).exists():
            files = [source_pattern]
        else:
            raise SystemExit(f"No files matched: {source_pattern}")
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # Try Gemini-style nested first
                af = r.get(analysis_field, {})
                if af and all(extract(af.get(d)) is not None for d in dim_names):
                    rows.append([extract(af[d]) for d in dim_names])
                    continue
                # Try DeepSeek validate_deepseek_oracle.py shape: {deepseek: {dim: score}}
                ds = r.get("deepseek", {})
                if ds and all(extract(ds.get(d)) is not None for d in dim_names):
                    rows.append([extract(ds[d]) for d in dim_names])
                    continue
                # Try Ollama shape: {dims: {dim: score}}
                dims = r.get("dims", {})
                if dims and all(extract(dims.get(d)) is not None for d in dim_names):
                    rows.append([extract(dims[d]) for d in dim_names])
                    continue
    if not rows:
        raise SystemExit(f"No records with complete dim scores in {source_pattern}")
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def correlation_matrix(rows, dim_names):
    n_dims = len(dim_names)
    mat = [[None] * n_dims for _ in range(n_dims)]
    for i in range(n_dims):
        for j in range(n_dims):
            xs = [r[i] for r in rows]
            ys = [r[j] for r in rows]
            mat[i][j] = pearson(xs, ys) if i != j else 1.0
    return mat


def naive_pca(rows, dim_names):
    """Compute PC variance explained + loadings using SVD on standardized data.

    Standardize: subtract mean, divide by std (so all dims weighted equally).
    Then SVD: cov = U S V^T, variance ratio = S^2 / sum(S^2).
    Loadings: V matrix (each row a PC, each column a dim).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    X = np.array(rows, dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    variance_explained = (S ** 2) / (S ** 2).sum()
    loadings = Vt  # rows = PCs, columns = dims
    return {
        "variance_explained": variance_explained.tolist(),
        "loadings": loadings.tolist(),
        "n": int(X.shape[0]),
        "dim_names": dim_names,
    }


def plot_correlation_heatmap(corr_mat, dim_names, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    arr = np.array(corr_mat, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(dim_names)))
    ax.set_yticks(range(len(dim_names)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dim_names], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([d.replace("_", "\n") for d in dim_names], fontsize=8)
    for i in range(len(dim_names)):
        for j in range(len(dim_names)):
            text_color = "white" if abs(arr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pca_variance(pca_data, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    var = pca_data["variance_explained"]
    cumvar = [sum(var[:i + 1]) for i in range(len(var))]
    fig, ax = plt.subplots(figsize=(7, 5))
    pcs = [f"PC{i+1}" for i in range(len(var))]
    ax.bar(pcs, [100 * v for v in var], label="Individual %")
    ax.plot(pcs, [100 * v for v in cumvar], "o-", color="red", label="Cumulative %")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    for i, (v, cv) in enumerate(zip(var, cumvar)):
        ax.text(i, 100 * v + 2, f"{100 * v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pca_loadings(pca_data, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    loadings = np.array(pca_data["loadings"])
    dim_names = pca_data["dim_names"]
    n_pcs = min(3, loadings.shape[0])
    n_dims = loadings.shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_dims)
    width = 0.25
    for i in range(n_pcs):
        ax.bar(x + (i - n_pcs / 2) * width + width / 2, loadings[i], width, label=f"PC{i+1}")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dim_names], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Loading")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_dimension_clustering(corr_mat, dim_names, output_path, title):
    """Cluster dims by 1 - |r| distance, produce a simple dendrogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        import numpy as np
    except ImportError:
        return False
    arr = np.array(corr_mat, dtype=float)
    dist = 1 - np.abs(arr)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=dim_names, ax=ax, leaf_rotation=30, leaf_font_size=9)
    ax.set_ylabel("Distance (1 - |r|)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", required=True, help="Path to filter dir (e.g., filters/cultural_discovery/v5)")
    parser.add_argument("--oracle-source", required=True, help="Glob pattern for oracle scored files")
    parser.add_argument("--oracle-name", required=True, help="Short name for oracle (e.g., gemini_v3)")
    args = parser.parse_args()

    filter_dir = Path(args.filter)
    config_path = filter_dir / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    # Parse config to get dim names + filter name
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim_names = list(cfg["scoring"]["dimensions"].keys())
    filter_name = cfg["filter"]["name"]
    version = cfg["filter"]["version"]
    analysis_field = f"{filter_name}_analysis"

    print(f"Filter: {filter_name} v{version}")
    print(f"Dims: {dim_names}")
    print(f"Oracle: {args.oracle_name}")
    print(f"Source: {args.oracle_source}")

    rows = load_oracle_scores(args.oracle_source, analysis_field, dim_names)
    print(f"Loaded {len(rows)} records with complete dim scores")

    # Output dir
    out_dir = filter_dir / "dimension_analysis" / args.oracle_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Correlation matrix
    corr_mat = correlation_matrix(rows, dim_names)
    pairs_above_07 = []
    for i, d1 in enumerate(dim_names):
        for j in range(i + 1, len(dim_names)):
            d2 = dim_names[j]
            r = corr_mat[i][j]
            if r is not None and abs(r) > 0.7:
                pairs_above_07.append((d1, d2, round(r, 3)))
    redundancy_ratio = len(pairs_above_07) / max(len(dim_names) * (len(dim_names) - 1) / 2, 1)

    # PCA
    pca_data = naive_pca(rows, dim_names)

    analysis_json = {
        "filter_name": filter_name,
        "filter_version": version,
        "oracle": args.oracle_name,
        "n_articles": len(rows),
        "dim_names": dim_names,
        "correlation_matrix": [[round(x, 3) if x is not None else None for x in row] for row in corr_mat],
        "pairs_above_abs_r_0.7": pairs_above_07,
        "redundancy_ratio_pct": round(100 * redundancy_ratio, 1),
        "pca": {
            "variance_explained_pct": [round(100 * v, 1) for v in pca_data["variance_explained"]] if pca_data else None,
            "loadings": [[round(x, 3) for x in row] for row in pca_data["loadings"]] if pca_data else None,
        } if pca_data else None,
    }

    json_path = out_dir / "dimension_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_json, f, indent=2)
    print(f"Wrote {json_path}")

    # Plots
    title_suffix = f"({filter_name} v{version}, oracle={args.oracle_name}, n={len(rows)})"
    plot_results = {}
    plot_results["correlation_heatmap"] = plot_correlation_heatmap(
        corr_mat, dim_names, out_dir / f"{args.oracle_name}_oracle_correlations.png",
        f"Per-dim Pearson correlation\n{title_suffix}"
    )
    if pca_data:
        plot_results["pca_variance"] = plot_pca_variance(
            pca_data, out_dir / f"{args.oracle_name}_pca_variance.png",
            f"PCA variance explained\n{title_suffix}"
        )
        plot_results["pca_loadings"] = plot_pca_loadings(
            pca_data, out_dir / f"{args.oracle_name}_pca_loadings.png",
            f"PCA loadings (top 3 PCs)\n{title_suffix}"
        )
    plot_results["dim_clustering"] = plot_dimension_clustering(
        corr_mat, dim_names, out_dir / f"{args.oracle_name}_dimension_clustering.png",
        f"Dimension clustering by 1-|r|\n{title_suffix}"
    )

    print(f"Plots written:")
    for name, ok in plot_results.items():
        print(f"  {name}: {'✅' if ok else '❌ (missing matplotlib/scipy)'}")

    print(f"\nSummary:")
    print(f"  Redundancy ratio: {analysis_json['redundancy_ratio_pct']}%")
    print(f"  Pairs above |r|=0.7: {len(pairs_above_07)}")
    for d1, d2, r in pairs_above_07:
        print(f"    {d1} ↔ {d2}: {r}")
    if pca_data:
        print(f"  PC1 explains: {analysis_json['pca']['variance_explained_pct'][0]}%")


if __name__ == "__main__":
    main()
