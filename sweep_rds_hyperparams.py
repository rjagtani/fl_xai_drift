"""
Sweep RDS hyperparameters on existing runs (no retraining).
Varies: FWER (1%, 5%, 10%), window size (1, 3, 5), confirm_consecutive (1, 3, 5).
Outputs: F1 and Delay per (dataset, param combo), then plots for 3–4 datasets.

Usage:
  python sweep_rds_hyperparams.py --results-dir results/drift_types --out-dir results/drift_types/summary
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.triggers.drift_detectors import RDSDetector, bonferroni_k

DATASETS = ['fed_heart_sudden', 'diabetes_gradual', 'hyperplane_gradual', 'credit_sudden']
DISPLAY_NAMES = {
    'fed_heart_sudden': 'FedHeart (Sudden)',
    'diabetes_gradual': 'Diabetes (Gradual)',
    'hyperplane_gradual': 'Hyperplane (Gradual)',
    'credit_sudden': 'Credit (Sudden)',
}

FWER_VALUES = [0.01, 0.03, 0.05, 0.10]
WINDOW_VALUES = [1, 3, 5, 7]
CONSECUTIVE_VALUES = [1, 2, 3, 5]


def discover_runs(results_dir: Path):
    """Return dict: exp_name -> list of (seed, run_dir)."""
    runs = defaultdict(list)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^(.+)_seed(\d+)$', d.name)
        if not m:
            continue
        exp, seed = m.group(1), int(m.group(2))
        if exp not in DATASETS:
            continue
        if not (d / 'loss_matrix.npy').exists() or not (d / 'metadata.json').exists():
            continue
        runs[exp].append((seed, d))
    for exp in runs:
        runs[exp].sort(key=lambda x: x[0])
    return runs


def load_run(run_dir: Path):
    """Load loss_matrix, aggregated loss, t0, and config from a run."""
    loss_matrix = np.load(run_dir / 'loss_matrix.npy')
    if loss_matrix.ndim == 1:
        loss_matrix = loss_matrix.reshape(-1, 1)
    aggregated = np.mean(loss_matrix, axis=1)
    with open(run_dir / 'metadata.json') as f:
        meta = json.load(f)
    cfg = meta.get('exp_cfg', {})
    t0 = meta.get('t0')
    warmup = cfg.get('warmup_rounds', 40)
    cal_start = cfg.get('calibration_start', 41)
    cal_end = cfg.get('calibration_end', 80)
    n_rounds = loss_matrix.shape[0]
    return {
        'loss_matrix': loss_matrix,
        'aggregated': aggregated,
        't0': t0,
        'warmup': warmup,
        'cal_start': cal_start,
        'cal_end': cal_end,
        'n_rounds': n_rounds,
    }


def run_rds_with_params(data, fwer_p: float, window_size: int, confirm_consecutive: int):
    """Run RDS detection with given hyperparams. Returns (triggered, delay).
    Delay is 1-based: (trigger_round+1) - t0, so >=0 means correct detection."""
    n_rounds = data['n_rounds']
    cal_end = data['cal_end']
    t0 = data['t0']
    k = bonferroni_k(n_rounds, cal_end, fwer_p)
    detector = RDSDetector(
        warmup_rounds=data['warmup'],
        calibration_start=data['cal_start'],
        calibration_end=cal_end,
        window_size=window_size,
        alpha=k,
        min_instances=5,
        confirm_consecutive=confirm_consecutive,
        require_loss_increase=False,
        use_fixed_threshold=True,
    )
    result = detector.detect(
        data['loss_matrix'],
        data['aggregated'],
        t0=t0,
        client_weights=None,
    )
    if result.triggered and result.trigger_round is not None and t0 is not None:
        delay = (result.trigger_round + 1) - t0
    else:
        delay = None
    return result.triggered, delay


def compute_f1_delay(seed_results):
    """seed_results = list of (triggered, delay) per seed. Return (f1, mean_delay)."""
    tp = sum(1 for t, d in seed_results if t and d is not None and d >= 0)
    fp = sum(1 for t, d in seed_results if t and (d is None or d < 0))
    fn = sum(1 for t, d in seed_results if not t)
    n = len(seed_results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    delays = [d for t, d in seed_results if t and d is not None and d >= 0]
    mean_delay = float(np.mean(delays)) if delays else np.nan
    return f1, mean_delay


def run_sweep(runs_by_exp):
    """Sweep all param combos; return nested dicts for plotting."""
    default_fwer = 0.05
    default_window = 3
    default_consec = 3

    results_fwer = defaultdict(dict)
    results_window = defaultdict(dict)
    results_consec = defaultdict(dict)

    for exp in DATASETS:
        if exp not in runs_by_exp:
            continue
        for fwer_p in FWER_VALUES:
            f1_list, delay_list = [], []
            for _seed, run_dir in runs_by_exp[exp]:
                data = load_run(run_dir)
                triggered, delay = run_rds_with_params(data, fwer_p, default_window, default_consec)
                f1_list.append((triggered, delay))
            f1, mean_delay = compute_f1_delay(f1_list)
            results_fwer[exp][fwer_p] = (f1, mean_delay)

        for w in WINDOW_VALUES:
            f1_list, delay_list = [], []
            for _seed, run_dir in runs_by_exp[exp]:
                data = load_run(run_dir)
                triggered, delay = run_rds_with_params(data, default_fwer, w, default_consec)
                f1_list.append((triggered, delay))
            f1, mean_delay = compute_f1_delay(f1_list)
            results_window[exp][w] = (f1, mean_delay)

        for c in CONSECUTIVE_VALUES:
            f1_list = []
            for _seed, run_dir in runs_by_exp[exp]:
                data = load_run(run_dir)
                triggered, delay = run_rds_with_params(data, default_fwer, default_window, c)
                f1_list.append((triggered, delay))
            f1, mean_delay = compute_f1_delay(f1_list)
            results_consec[exp][c] = (f1, mean_delay)

    return results_fwer, results_window, results_consec


def _draw_sweep_axes(ax_f1, ax_delay, results_fwer_or_window_or_consec, x_vals, x_label, x_ticks, x_ticklabels,
                     colors, markers):
    """Draw F1 and Delay on the two axes; return handles for legend (from ax_f1)."""
    handles = []
    for idx, exp in enumerate(DATASETS):
        if exp not in results_fwer_or_window_or_consec:
            continue
        res = results_fwer_or_window_or_consec[exp]
        x = sorted(res.keys())
        f1_vals = [res[p][0] for p in x]
        delay_vals = [res[p][1] for p in x]
        delay_vals = [d if not np.isnan(d) else np.nan for d in delay_vals]
        h, = ax_f1.plot(x, f1_vals, color=colors[idx], marker=markers[idx], label=DISPLAY_NAMES.get(exp, exp), linewidth=1.5, markersize=5)
        handles.append(h)
        ax_delay.plot(x, delay_vals, color=colors[idx], marker=markers[idx], linewidth=1.5, markersize=5)
    ax_f1.set_xlabel(x_label, fontsize=9)
    ax_f1.set_ylabel('F1 score', fontsize=9)
    ax_f1.set_xticks(x_ticks)
    ax_f1.set_xticklabels(x_ticklabels, fontsize=8)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.set_ylim(-0.05, 1.05)
    ax_delay.set_xlabel(x_label, fontsize=9)
    ax_delay.set_ylabel('Delay (rounds)', fontsize=9)
    ax_delay.set_xticks(x_ticks)
    ax_delay.set_xticklabels(x_ticklabels, fontsize=8)
    ax_delay.grid(True, alpha=0.3)
    return handles


def plot_sweep(results_fwer, results_window, results_consec, out_dir: Path):
    """Create 3 figures: one per hyperparam, compact, single legend above.
    Also create one combined panel for appendix (3 rows x 2 cols: FWER, window, consecutive x F1, Delay)."""
    colors = ['#4a90d9', '#5cba7d', '#e07b54', '#9b59b6']
    markers = ['o', 's', '^', 'D']

    fig, (ax_f1, ax_delay) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    handles = _draw_sweep_axes(
        ax_f1, ax_delay, results_fwer, FWER_VALUES, 'FWER',
        FWER_VALUES, ['1%', '3%', '5%', '10%'], colors, markers)
    fig.legend(handles=handles, labels=[DISPLAY_NAMES.get(e, e) for e in DATASETS if e in results_fwer],
               loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'rds_sweep_fwer.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax_f1, ax_delay) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    handles = _draw_sweep_axes(
        ax_f1, ax_delay, results_window, WINDOW_VALUES, 'Window size (rounds)',
        WINDOW_VALUES, [str(v) for v in WINDOW_VALUES], colors, markers)
    fig.legend(handles=handles, labels=[DISPLAY_NAMES.get(e, e) for e in DATASETS if e in results_window],
               loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'rds_sweep_window.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax_f1, ax_delay) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    handles = _draw_sweep_axes(
        ax_f1, ax_delay, results_consec, CONSECUTIVE_VALUES, 'Consecutive triggers',
        CONSECUTIVE_VALUES, [str(v) for v in CONSECUTIVE_VALUES], colors, markers)
    fig.legend(handles=handles, labels=[DISPLAY_NAMES.get(e, e) for e in DATASETS if e in results_consec],
               loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'rds_sweep_consecutive.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 7.2))
    h = _draw_sweep_axes(
        axes[0, 0], axes[0, 1], results_fwer, FWER_VALUES, 'FWER',
        FWER_VALUES, ['1%', '3%', '5%', '10%'], colors, markers)
    axes[0, 0].set_title('Varying FWER', fontsize=10)
    _draw_sweep_axes(
        axes[1, 0], axes[1, 1], results_window, WINDOW_VALUES, 'Window size (rounds)',
        WINDOW_VALUES, [str(v) for v in WINDOW_VALUES], colors, markers)
    axes[1, 0].set_title('Varying window size', fontsize=10)
    _draw_sweep_axes(
        axes[2, 0], axes[2, 1], results_consec, CONSECUTIVE_VALUES, 'Consecutive triggers',
        CONSECUTIVE_VALUES, [str(v) for v in CONSECUTIVE_VALUES], colors, markers)
    axes[2, 0].set_title('Varying consecutive triggers', fontsize=10)
    fig.legend(handles=h, labels=[DISPLAY_NAMES.get(e, e) for e in DATASETS if e in results_fwer],
               loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'rds_sweep_panel.{ext}', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results/drift_types'))
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (args.results_dir / 'summary')
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_by_exp = discover_runs(args.results_dir)
    for exp in DATASETS:
        n = len(runs_by_exp.get(exp, []))
        print(f"  {exp}: {n} runs")
    if not any(runs_by_exp.get(e) for e in DATASETS):
        print("No runs found. Exiting.")
        return

    print("Running RDS sweep (FWER × window × consecutive)...")
    results_fwer, results_window, results_consec = run_sweep(runs_by_exp)

    print("Writing plots...")
    plot_sweep(results_fwer, results_window, results_consec, out_dir)

    lines = ['dataset,param_type,param_value,f1,delay']
    for exp in DATASETS:
        for p, (f1, d) in results_fwer.get(exp, {}).items():
            lines.append(f"{exp},fwer,{p},{f1:.3f},{d if not np.isnan(d) else ''}")
        for w, (f1, d) in results_window.get(exp, {}).items():
            lines.append(f"{exp},window,{w},{f1:.3f},{d if not np.isnan(d) else ''}")
        for c, (f1, d) in results_consec.get(exp, {}).items():
            lines.append(f"{exp},consecutive,{c},{f1:.3f},{d if not np.isnan(d) else ''}")
    (out_dir / 'rds_sweep_results.csv').write_text('\n'.join(lines) + '\n')
    print(f"Saved {out_dir / 'rds_sweep_results.csv'}")
    print(f"Saved rds_sweep_fwer.png, rds_sweep_window.png, rds_sweep_consecutive.png, rds_sweep_panel.png (+ .pdf)")


if __name__ == '__main__':
    main()
