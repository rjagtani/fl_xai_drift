#!/usr/bin/env python
"""
Runtime breakdown plot for FedHeart (Sudden) and Credit (Sudden).

Shows per-round FL training time, RDS detection overhead, and one-time
SAGE diagnosis cost — averaged over 5 seeds with error bars.

Usage:
  python plot_runtime_panel.py
  python plot_runtime_panel.py --results-dir results/drift_types --out-dir results/drift_types/summary
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENTS = ['fed_heart_sudden', 'diabetes_gradual', 'hyperplane_gradual', 'credit_sudden']
DISPLAY_NAMES = {
    'fed_heart_sudden': 'FedHeart\n(K=4, F=13)',
    'diabetes_gradual': 'Diabetes\n(K=6, F=8)',
    'hyperplane_gradual': 'Hyperplane\n(K=10, F=5)',
    'credit_sudden': 'Credit\n(K=5, F=20)',
}


def load_timings(results_dir: Path):
    """Load timings across all seeds for selected experiments."""
    data = defaultdict(lambda: defaultdict(list))
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^(.+)_seed(\d+)$', d.name)
        if not m:
            continue
        exp = m.group(1)
        if exp not in EXPERIMENTS:
            continue
        tf = d / 'timings.csv'
        if not tf.exists():
            continue
        with open(tf) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[exp][row['metric']].append(float(row['value']))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results/drift_types'))
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (args.results_dir / 'summary')
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_timings(args.results_dir)

    # --- Grouped bar chart: datasets on x-axis, components as groups ---
    components = [
        ('fl_train_total_s', 'FL Training (total)'),
        ('fl_train_per_round_avg_s', 'FL Training (per round)'),
        ('detection_rds_loss_s', 'RDS Detection (total)'),
        ('sage_trigger_client_mean_s', 'SAGE Diagnosis (per client)'),
    ]
    # Softer, more pleasant palette
    colors = ['#7eb8da', '#4a90d9', '#5cba7d', '#e07b54']

    n_datasets = len(EXPERIMENTS)
    n_components = len(components)
    bar_width = 0.18
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.2))

    for i, (key, label) in enumerate(components):
        means = []
        for exp in EXPERIMENTS:
            d = data.get(exp, {})
            vals = d.get(key, [0])
            means.append(np.mean(vals))
        offset = (i - (n_components - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, width=bar_width, color=colors[i],
                      edgecolor='white', linewidth=0.6, label=label, zorder=3)
        # Value labels (larger so they read clearly)
        for bar, mean_val in zip(bars, means):
            if mean_val >= 100:
                txt = f'{mean_val:.0f}'
            elif mean_val >= 1:
                txt = f'{mean_val:.1f}'
            else:
                txt = f'{mean_val:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                    txt, ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=colors[i])

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[e] for e in EXPERIMENTS], fontsize=9)
    ax.set_ylabel('Time (seconds, log scale)', fontsize=10)
    ax.legend(fontsize=11, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18),
              frameon=True, fancybox=True, shadow=False)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#aaaaaa')

    # ylim with headroom
    all_means = []
    for exp in EXPERIMENTS:
        d = data.get(exp, {})
        for key, _ in components:
            vals = d.get(key, [0])
            all_means.append(np.mean(vals))
    ax.set_ylim(0.008, max(all_means) * 4)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'runtime_panel.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_dir / "runtime_panel.png"} and runtime_panel.pdf')


if __name__ == '__main__':
    main()
