#!/usr/bin/env python3
"""Generate comparison charts for v1 vs v2 evaluation."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Data from comparison (v2.1 results)
data = {
    'v1_pass': 271,
    'v1_block': 0,
    'v2_pass': 141,
    'v2_block': 130,
    'both_pass': 141,
    'both_block': 0,
    'v1_pass_v2_block': 130,
    'v1_block_v2_pass': 0
}

output_dir = Path(__file__).parent

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'v1': '#e74c3c',  # Red
    'v2': '#27ae60',  # Green
    'improvement': '#3498db',  # Blue
    'regression': '#e67e22',  # Orange
    'remaining_fp': '#95a5a6'  # Gray
}

# Chart 1: Bar comparison
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['v1', 'v2']
pass_counts = [data['v1_pass'], data['v2_pass']]
block_counts = [data['v1_block'], data['v2_block']]

x = range(len(categories))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], pass_counts, width, label='Pass (False Positives)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], block_counts, width, label='Block (Correct)', color='#27ae60', alpha=0.8)

ax.set_ylabel('Number of Articles', fontsize=12)
ax.set_title('Prefilter Performance on 271 Known False Positives', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Filter v1', 'Filter v2'], fontsize=12)
ax.legend(fontsize=10)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, 300)
plt.tight_layout()
plt.savefig(output_dir / 'chart_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: chart_comparison.png")
plt.close()

# Chart 2: Pie chart showing v2 results breakdown
fig, ax = plt.subplots(figsize=(8, 8))

sizes = [data['both_pass'], data['v1_pass_v2_block']]
labels = ['Still False Positives\n(184 articles, 67.9%)', 'Now Blocked by v2\n(87 articles, 32.1%)']
colors_pie = ['#e74c3c', '#27ae60']
explode = (0, 0.05)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                   autopct='', startangle=90, textprops={'fontsize': 11})

ax.set_title('v2 Improvement on Known False Positives', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'chart_improvement.png', dpi=150, bbox_inches='tight')
print(f"Saved: chart_improvement.png")
plt.close()

# Chart 3: False Positive Rate comparison
fig, ax = plt.subplots(figsize=(8, 5))

versions = ['v1', 'v2.1']
fp_rates = [100.0, 52.0]
bar_colors = ['#e74c3c', '#27ae60']

bars = ax.bar(versions, fp_rates, color=bar_colors, width=0.5, alpha=0.8)

ax.set_ylabel('False Positive Rate (%)', fontsize=12)
ax.set_title('False Positive Rate: v1 vs v2', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)

# Add value labels and improvement arrow
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
ax.annotate('', xy=(1, 60), xytext=(0, 95),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
ax.text(0.5, 80, '-48%', ha='center', fontsize=12, fontweight='bold', color='#3498db')

plt.tight_layout()
plt.savefig(output_dir / 'chart_fp_rate.png', dpi=150, bbox_inches='tight')
print(f"Saved: chart_fp_rate.png")
plt.close()

print("\nAll charts generated!")
