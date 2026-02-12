"""
Ad-hoc script: 4x2 Loss + RDS panel — FedHeart & Diabetes on top row, Hyperplane & Wine below (more horizontal space per plot).

Reads from results/drift_types/, uses first seed per experiment.

Usage:
  python plot_loss_panel_adhoc.py
  python plot_loss_panel_adhoc.py --results-dir results/drift_types --out-dir results/drift_types/summary
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

EXPERIMENTS = ['fed_heart_sudden', 'diabetes_gradual', 'hyperplane_gradual', 'wine_sudden']
DISPLAY_NAMES = {
    'fed_heart_sudden': 'FedHeart (Sudden)',
    'diabetes_gradual': 'Diabetes (Gradual)',
    'hyperplane_gradual': 'Hyperplane (Gradual)',
    'wine_sudden': 'Wine (Sudden)',
}


def load_run_data(run_dir: Path):
    if not (run_dir / 'loss_history.npy').exists() or not (run_dir / 'metadata.json').exists():
        return None
    loss = np.load(run_dir / 'loss_history.npy')
    with open(run_dir / 'metadata.json') as f:
        meta = json.load(f)
    cfg = meta.get('exp_cfg', {})
    warmup = cfg.get('warmup_rounds', 40)
    cal_start = cfg.get('calibration_start', 41)
    cal_end = cfg.get('calibration_end', 80)
    t0, t1 = meta.get('t0'), meta.get('t1')
    trigger_data = {}
    if (run_dir / 'trigger_results.json').exists():
        with open(run_dir / 'trigger_results.json') as f:
            trigger_data = json.load(f)
    rds_trigger = trigger_data.get('detection_t0', {}).get('rds', {}).get('round') if trigger_data.get('detection_t0', {}).get('rds', {}).get('triggered') else None
    rds_t1 = None
    if t1 is not None and trigger_data.get('detection_t1', {}).get('rds', {}).get('triggered'):
        rds_t1 = trigger_data['detection_t1']['rds'].get('round')
    rds_scores = None
    rds_thresh = None
    if (run_dir / 'rds_scores.npy').exists():
        r = np.load(run_dir / 'rds_scores.npy')
        rds_scores = np.nanmean(r, axis=1) if r.ndim > 1 else r
    if (run_dir / 'rds_thresholds.npy').exists():
        rds_thresh = np.load(run_dir / 'rds_thresholds.npy', allow_pickle=True)
    drift_type = meta.get('drift_type', 'sudden')
    return {
        'loss': loss, 'n_rounds': len(loss),
        'warmup': warmup, 'cal_start': cal_start, 'cal_end': cal_end,
        't0': t0, 't1': t1, 'rds_trigger_round': rds_trigger, 'rds_t1_round': rds_t1,
        'rds_scores': rds_scores, 'rds_thresholds': rds_thresh,
        'drift_type': drift_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results/drift_types'), help='Parent of *_seed* run dirs')
    ap.add_argument('--out-dir', type=Path, default=None, help='Output dir (default: results-dir/summary)')
    args = ap.parse_args()
    out_dir = args.out_dir or (args.results_dir / 'summary')
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {}
    for exp in EXPERIMENTS:
        for d in sorted(args.results_dir.iterdir()):
            if d.is_dir() and d.name.startswith(exp + '_seed'):
                run_dirs[exp] = d
                break

    fig = plt.figure(figsize=(12, 10))
    outer = GridSpec(2, 1, figure=fig, hspace=0.22)
    inner_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0)
    inner_bot = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0)
    axes = np.empty((4, 2), dtype=object)
    for c in range(2):
        axes[0, c] = fig.add_subplot(inner_top[0, c])
        axes[1, c] = fig.add_subplot(inner_top[1, c], sharex=axes[0, c])
        axes[2, c] = fig.add_subplot(inner_bot[0, c])
        axes[3, c] = fig.add_subplot(inner_bot[1, c], sharex=axes[2, c])
    for row in range(4):
        for col in range(2):
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#aaaaaa')
    color_loss = '#2171b5'
    color_t0 = '#cb181d'
    color_trigger = '#238b45'
    color_threshold = '#c0392b'
    gray_warmup = '#969696'
    amber_cal = '#fec44f'
    alpha_warmup, alpha_cal = 0.35, 0.25

    for idx, exp in enumerate(EXPERIMENTS):
        run_dir = run_dirs.get(exp)
        col = idx % 2
        row_block = 0 if idx < 2 else 2
        ax_loss = axes[row_block, col]
        ax_rds = axes[row_block + 1, col]
        if run_dir is None:
            ax_loss.text(0.5, 0.5, f'No data: {exp}', ha='center', va='center', transform=ax_loss.transAxes)
            ax_loss.set_title(DISPLAY_NAMES.get(exp, exp))
            ax_rds.set_visible(False)
            continue
        data = load_run_data(run_dir)
        if data is None:
            ax_loss.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_loss.transAxes)
            ax_loss.set_title(DISPLAY_NAMES.get(exp, exp))
            ax_rds.set_visible(False)
            continue

        n_rounds = data['n_rounds']
        warmup = data['warmup']
        cal_start, cal_end = data['cal_start'], data['cal_end']
        t0, t1 = data['t0'], data['t1']
        rds_trigger_round = data['rds_trigger_round']
        rds_t1_round = data['rds_t1_round']
        rds_scores = data['rds_scores']
        rds_thresholds = data['rds_thresholds']
        drift_type = data.get('drift_type', 'sudden')
        rounds = np.arange(1, n_rounds + 1)

        ax_loss.plot(rounds, data['loss'], color=color_loss, linewidth=1.8, zorder=3)
        ax_loss.axvspan(1, warmup + 0.5, alpha=alpha_warmup, color=gray_warmup, zorder=0)
        ax_loss.axvspan(cal_start - 0.5, cal_end + 0.5, alpha=alpha_cal, color=amber_cal, zorder=0)
        if t0 is not None:
            ax_loss.axvline(x=t0, color=color_t0, linestyle='--', linewidth=1.8, zorder=2)
        if rds_trigger_round is not None:
            ax_loss.axvline(x=rds_trigger_round, color=color_trigger, linestyle=':', linewidth=2, zorder=2)
        if t1 is not None:
            ax_loss.axvline(x=t1, color='#88419d', linestyle='-.', linewidth=1.5, zorder=2)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title(DISPLAY_NAMES.get(exp, exp), fontsize=11)
        ax_loss.grid(True, alpha=0.35)
        ax_loss.set_xlim(1, n_rounds)
        ax_loss.tick_params(axis='x', labelbottom=False)

        if rds_scores is not None and len(rds_scores) > 0:
            rds_rounds = np.arange(warmup + 1, warmup + 1 + len(rds_scores), dtype=float)[:len(rds_scores)]
            is_recurring = drift_type == 'recurring'
            if not is_recurring and rds_trigger_round is not None:
                mask = rds_rounds <= rds_trigger_round
                rds_rounds = rds_rounds[mask]
                rds_scores = rds_scores[:len(rds_rounds)]
                if rds_thresholds is not None and len(rds_thresholds) >= len(rds_rounds):
                    rds_thresholds = rds_thresholds[:len(rds_rounds)]
            ax_rds.plot(rds_rounds, rds_scores[:len(rds_rounds)], color='#6a51a3', linewidth=1.8, zorder=3)
            ax_rds.axvspan(1, warmup + 0.5, alpha=alpha_warmup, color=gray_warmup, zorder=0)
            ax_rds.axvspan(cal_start - 0.5, cal_end + 0.5, alpha=alpha_cal, color=amber_cal, zorder=0)
            if rds_thresholds is not None and len(rds_thresholds) == len(rds_scores):
                th_rounds, th_vals = [], []
                for i, t in enumerate(rds_thresholds):
                    try:
                        v = float(t)
                        if np.isfinite(v):
                            th_vals.append(v)
                            th_rounds.append(rds_rounds[i] if i < len(rds_rounds) else warmup + 1 + i)
                    except (TypeError, ValueError):
                        pass
                if th_vals:
                    ax_rds.plot(th_rounds, th_vals, color=color_threshold, linestyle='-', linewidth=2.5, zorder=4)
                    ylo, yhi = ax_rds.get_ylim()
                    th_min, th_max = min(th_vals), max(th_vals)
                    ax_rds.set_ylim(min(ylo, th_min) - 0.02 * (yhi - ylo or 1),
                                    max(yhi, th_max) + 0.02 * (yhi - ylo or 1))
            if t0 is not None:
                ax_rds.axvline(x=t0, color=color_t0, linestyle='--', linewidth=1.8, zorder=2)
            if rds_trigger_round is not None:
                ax_rds.axvline(x=rds_trigger_round, color=color_trigger, linestyle=':', linewidth=2, zorder=2)
            if t1 is not None and rds_t1_round is not None:
                ax_rds.axvline(x=rds_t1_round, color='#88419d', linestyle='-.', linewidth=1.5, zorder=2)
            ax_rds.set_ylabel('RDS score', fontsize=10)
            ax_rds.grid(True, alpha=0.35)
            ax_rds.set_xlim(1, n_rounds)
        else:
            ax_rds.text(0.5, 0.5, 'No RDS data', ha='center', va='center', transform=ax_rds.transAxes, fontsize=10)
            ax_rds.set_ylabel('RDS score', fontsize=10)
            ax_rds.set_xlim(1, n_rounds)
        ax_rds.set_xlabel('Round', fontsize=10)

    fig.legend(
        handles=[
            Patch(facecolor=gray_warmup, alpha=alpha_warmup, edgecolor='none'),
            Patch(facecolor=amber_cal, alpha=alpha_cal, edgecolor='none'),
            Line2D([0], [0], color=color_t0, linestyle='--', linewidth=2),
            Line2D([0], [0], color=color_trigger, linestyle=':', linewidth=2),
            Line2D([0], [0], color=color_threshold, linestyle='-', linewidth=2.5),
        ],
        labels=['Warmup', 'Calibration', 'Drift onset', 'RDS trigger', 'Threshold'],
        loc='lower center', ncol=5, fontsize=9, frameon=True,
    )
    fig.subplots_adjust(bottom=0.09)
    for ext in ['png', 'pdf']:
        fig.savefig(out_dir / f'loss_panel.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_dir / "loss_panel.png"} and loss_panel.pdf')


if __name__ == '__main__':
    main()
