"""
Aggregate drift_types experiment results and generate summary tables and panels.

Scans results/drift_types/ for run directories (*_seed*), aggregates detection
and diagnosis metrics across seeds, and writes:
  - summary/detection_table.csv, detection_table.tex
  - summary/diagnosis_table.csv, diagnosis_table.tex
  - summary/loss_panel.png, loss_panel.pdf
  - summary/fi_rds_panel.png, fi_rds_panel.pdf

Usage:
  python aggregate_drift_types_summary.py
  python aggregate_drift_types_summary.py --results-dir results/drift_types
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# Display names for experiment keys (dataset + drift type)
EXP_DISPLAY = {
    'fed_heart_sudden': 'FedHeart (Sudden)',
    'agrawal_recurring': 'Agrawal (Recurring)',
    'hyperplane_gradual': 'Hyperplane (Gradual)',
    'hyperplane_gradual_low': 'Hyperplane (Gradual-Low)',
    'hyperplane_gradual_vlow': 'Hyperplane (Gradual-VLow)',
    'agrawal_sudden': 'Agrawal (Sudden)',
    'wine_sudden': 'Wine (Sudden)',
    'elec2_recurring': 'ELEC2 (Recurring)',
    'diabetes_gradual': 'Diabetes (Gradual)',
    'credit_sudden': 'Credit (Sudden)',
    'adult_sudden': 'Adult (Sudden)',
    'adult_gradual': 'Adult (Gradual)',
}

# Canonical order for tables (matches existing summary style)
EXP_ORDER = [
    'agrawal_sudden', 'wine_sudden', 'fed_heart_sudden', 'adult_gradual', 'adult_sudden',
    'hyperplane_gradual', 'elec2_recurring', 'diabetes_gradual', 'credit_sudden',
    'agrawal_recurring',
]

DETECTION_METHODS = ['rds', 'cusum', 'page_hinkley', 'adwin', 'kswin']
DETECTION_METHOD_LABELS = {'rds': 'RDS', 'cusum': 'CUSUM', 'page_hinkley': 'PH', 'adwin': 'ADWIN', 'kswin': 'KSWIN'}


def exp_display_name(exp_name: str) -> str:
    return EXP_DISPLAY.get(exp_name, exp_name.replace('_', ' ').title())


def discover_runs(results_dir: Path):
    """Discover all run directories (pattern: *_seed<N>) and group by experiment name."""
    runs_by_exp = defaultdict(list)
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r'^(.+)_seed(\d+)$', d.name)
        if not m:
            continue
        exp_name, seed = m.group(1), int(m.group(2))
        trigger_file = d / 'trigger_results.json'
        if not trigger_file.exists():
            continue
        runs_by_exp[exp_name].append((seed, d))
    for exp_name in runs_by_exp:
        runs_by_exp[exp_name].sort(key=lambda x: x[0])
    return runs_by_exp


def load_detection_results(run_dir: Path):
    """Load detection results. Returns (res_t0, t0, res_t1 or None, t1 or None)."""
    with open(run_dir / 'trigger_results.json') as f:
        data = json.load(f)
    t0 = data['t0']
    det0 = data.get('detection_t0', {})
    out0 = {}
    for method in DETECTION_METHODS:
        if method not in det0:
            out0[method] = {'triggered': False, 'round': None, 'delay': None}
            continue
        r = det0[method]
        rnd = r.get('round')
        delay = r.get('delay')
        if delay is None and rnd is not None and t0 is not None:
            delay = rnd - t0
        out0[method] = {'triggered': r.get('triggered', False), 'round': rnd, 'delay': delay}
    out1 = None
    t1 = None
    if 'detection_t1' in data:
        t1 = data.get('t1')
        det1 = data['detection_t1']
        out1 = {}
        for method in DETECTION_METHODS:
            if method not in det1:
                out1[method] = {'triggered': False, 'round': None, 'delay': None}
                continue
            r = det1[method]
            rnd = r.get('round')
            delay = r.get('delay')
            if delay is None and rnd is not None and t1 is not None:
                delay = rnd - t1
            out1[method] = {'triggered': r.get('triggered', False), 'round': rnd, 'delay': delay}
    return out0, t0, out1, t1


# Recurring experiments: two change points (t0 and t1); TP only if both detected with delay >= 0.
RECURRING_EXPERIMENTS = {'elec2_recurring', 'agrawal_recurring'}


def aggregate_detection(runs_by_exp):
    """Aggregate detection metrics: Precision, Recall, F1, Delay per (experiment, method).
    For recurring: TP = both t0 and t1 detected with delay >= 0; early trigger = FP; else FN.
    """
    rows = {}
    for exp_name in EXP_ORDER:
        if exp_name not in runs_by_exp:
            continue
        runs = runs_by_exp[exp_name]
        n = len(runs)
        is_recurring = exp_name in RECURRING_EXPERIMENTS
        row = {}
        for method in DETECTION_METHODS:
            if is_recurring:
                tp = fp = 0
                delays_both = []  # (delay_t0 + delay_t1) / 2 per TP run
                for _seed, run_dir in runs:
                    res0, t0, res1, t1 = load_detection_results(run_dir)
                    if res1 is None or t1 is None:
                        # No t1 results (e.g. old run): treat as single-instance
                        r = res0[method]
                        if r['triggered']:
                            d = r['delay']
                            if d is not None and d >= 0:
                                tp += 1
                                delays_both.append(d)
                            else:
                                fp += 1
                        continue
                    r0, r1 = res0[method], res1[method]
                    correct_t0 = r0['triggered'] and r0['delay'] is not None and r0['delay'] >= 0
                    correct_t1 = r1['triggered'] and r1['delay'] is not None and r1['delay'] >= 0
                    early_t0 = r0['triggered'] and r0['delay'] is not None and r0['delay'] < 0
                    early_t1 = r1['triggered'] and r1['delay'] is not None and r1['delay'] < 0
                    if correct_t0 and correct_t1:
                        tp += 1
                        delays_both.append((r0['delay'] + r1['delay']) / 2.0)
                    elif early_t0 or early_t1:
                        fp += 1
                    # else: fn (missed one or both, no early trigger)
                fn = n - tp - fp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                delay_mean = float(np.mean(delays_both)) if delays_both else None
                delay_std = float(np.std(delays_both, ddof=1)) if len(delays_both) > 1 else 0.0
                row[method] = {'prec': prec, 'rec': rec, 'f1': f1, 'delay': delay_mean, 'delay_std': delay_std if delay_mean is not None else None, 'n_seeds': n}
            else:
                triggered_correct = 0
                triggered_any = 0
                delays = []
                for _seed, run_dir in runs:
                    res0, t0, _res1, _t1 = load_detection_results(run_dir)
                    r = res0[method]
                    if r['triggered']:
                        triggered_any += 1
                        d = r['delay']
                        if d is not None and d >= 0:
                            triggered_correct += 1
                            delays.append(d)
                prec = triggered_correct / triggered_any if triggered_any > 0 else 0.0
                rec = triggered_correct / n if n > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                delay_mean = float(np.mean(delays)) if delays else None
                delay_std = float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0
                row[method] = {'prec': prec, 'rec': rec, 'f1': f1, 'delay': delay_mean, 'delay_std': delay_std if delay_mean is not None else None, 'n_seeds': n}
        rows[exp_name] = row
    return rows


def load_diagnosis_metrics(run_dir: Path):
    path = run_dir / 'diagnosis' / 'diagnosis_metrics.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_diagnosis(runs_by_exp):
    """Aggregate diagnosis metrics: H@1, H@2, MRR per (experiment, method)."""
    # Methods we might have: dist_sage, dist_sage_w3, dist_sage_w5, dist_pfi, dist_shap, delta_*, ...
    method_order = [
        'dist_sage', 'dist_sage_w3', 'dist_sage_w5', 'dist_pfi', 'dist_shap',
        'delta_sage', 'delta_pfi', 'delta_shap',
        'meanph_sage', 'meanph_pfi', 'meanph_shap',
    ]
    rows = {}
    for exp_name in EXP_ORDER:
        if exp_name not in runs_by_exp:
            continue
        runs = runs_by_exp[exp_name]
        collected = defaultdict(list)  # method -> list of {h1, h2, mrr}
        for _seed, run_dir in runs:
            data = load_diagnosis_metrics(run_dir)
            if not data:
                continue
            for method, vals in data.items():
                if not isinstance(vals, dict):
                    continue
                h1 = vals.get('hits_at_1')
                h2 = vals.get('hits_at_2')
                mrr = vals.get('mrr')
                if h1 is not None or h2 is not None or mrr is not None:
                    collected[method].append({
                        'h1': float(h1) if h1 is not None else 0.0,
                        'h2': float(h2) if h2 is not None else 0.0,
                        'mrr': float(mrr) if mrr is not None else 0.0,
                    })
        row = {}
        for method in method_order:
            if method not in collected or not collected[method]:
                continue
            L = collected[method]
            row[method] = {
                'h1': np.mean([x['h1'] for x in L]),
                'h2': np.mean([x['h2'] for x in L]),
                'mrr': np.mean([x['mrr'] for x in L]),
            }
        if row:
            rows[exp_name] = row
    return rows


EXCLUDE_EXPERIMENTS = {'elec2_recurring', 'agrawal_recurring'}

def write_detection_csv(agg_det, summary_dir: Path):
    """Compact CSV: one row per (experiment, metric), methods as columns. Excludes recurring."""
    header = ['Experiment', 'Metric']
    for m in DETECTION_METHODS:
        header.append(DETECTION_METHOD_LABELS[m])
    lines = [','.join(header)]
    for exp_name in EXP_ORDER:
        if exp_name not in agg_det or exp_name in EXCLUDE_EXPERIMENTS:
            continue
        display = exp_display_name(exp_name)
        for metric_key, metric_label in [('prec', 'Prec'), ('rec', 'Rec'), ('f1', 'F1'), ('delay', 'Delay')]:
            row = [display if metric_key == 'prec' else '', metric_label]
            for method in DETECTION_METHODS:
                r = agg_det[exp_name][method]
                if metric_key == 'delay':
                    if r['delay'] is not None:
                        std = r.get('delay_std')
                        if std is not None and std > 0:
                            row.append(f"{r['delay']:.1f}±{std:.1f}")
                        else:
                            row.append(f"{r['delay']:.1f}")
                    else:
                        row.append('--')
                else:
                    row.append(f"{r[metric_key]:.2f}")
            lines.append(','.join(row))
    # Summary row (macro-average across included experiments)
    included = [e for e in EXP_ORDER if e in agg_det and e not in EXCLUDE_EXPERIMENTS]
    if included:
        for metric_key, metric_label in [('prec', 'Prec'), ('rec', 'Rec'), ('f1', 'F1'), ('delay', 'Delay')]:
            row = ['Avg.' if metric_key == 'prec' else '', metric_label]
            for method in DETECTION_METHODS:
                vals = [agg_det[e][method][metric_key] for e in included if agg_det[e][method][metric_key] is not None]
                if vals:
                    row.append(f"{np.mean(vals):.2f}")
                else:
                    row.append('--')
            lines.append(','.join(row))
    (summary_dir / 'detection_table.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _bold_best(values, method_keys, higher_is_better=True):
    """Return dict {method: formatted_str} with the best value(s) in bold.
    values: dict method -> float or None.  None entries get '--'."""
    valid = {m: v for m, v in values.items() if v is not None and m in method_keys}
    if not valid:
        return {m: '--' for m in method_keys}
    best_val = max(valid.values()) if higher_is_better else min(valid.values())
    result = {}
    for m in method_keys:
        v = values.get(m)
        if v is None:
            result[m] = '--'
        elif abs(v - best_val) < 1e-9:
            result[m] = r'\textbf{' + f'{v:.2f}' + '}'
        else:
            result[m] = f'{v:.2f}'
    return result


def _bold_best_delay(values, std_values, method_keys):
    """Return dict {method: formatted_str} with the lowest delay in bold, including ±std."""
    valid = {m: v for m, v in values.items() if v is not None and m in method_keys}
    if not valid:
        return {m: '--' for m in method_keys}
    best_val = min(valid.values())
    result = {}
    for m in method_keys:
        v = values.get(m)
        s = std_values.get(m)
        if v is None:
            result[m] = '--'
        else:
            if s is not None and s > 0:
                txt = f'{v:.1f}{{\scriptstyle\\pm{s:.1f}}}'
            else:
                txt = f'{v:.1f}'
            if abs(v - best_val) < 1e-9:
                result[m] = r'\textbf{' + txt + '}'
            else:
                result[m] = txt
    return result


def write_detection_tex(agg_det, summary_dir: Path):
    """Compact LaTeX table: 4 metric rows per dataset, methods as sub-columns, bold best, summary row.
    Excludes recurring experiments."""
    methods_tex = ['RDS', 'CUSUM', 'PH', 'ADWIN', 'KSWIN']
    n_seeds_note = ''
    # Find n_seeds from first experiment
    for exp_name in EXP_ORDER:
        if exp_name in agg_det and exp_name not in EXCLUDE_EXPERIMENTS:
            first_method = DETECTION_METHODS[0]
            n = agg_det[exp_name][first_method].get('n_seeds', '?')
            n_seeds_note = f' ($n={n}$ seeds)'
            break
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Drift detection performance across datasets' + n_seeds_note + r'. Best per metric in \textbf{bold}.}',
        r'\label{tab:detection}',
        r'\small',
        r'\begin{tabular}{ll' + 'c' * len(DETECTION_METHODS) + '}',
        r'\toprule',
        'Dataset & Metric & ' + ' & '.join(methods_tex) + r' \\',
        r'\midrule',
    ]
    included = [e for e in EXP_ORDER if e in agg_det and e not in EXCLUDE_EXPERIMENTS]
    for i, exp_name in enumerate(included):
        display = exp_display_name(exp_name)
        # Precision
        prec_vals = {m: agg_det[exp_name][m]['prec'] for m in DETECTION_METHODS}
        bold_prec = _bold_best(prec_vals, DETECTION_METHODS, higher_is_better=True)
        lines.append(r'\multirow{4}{*}{' + display + '} & Prec. & ' + ' & '.join(bold_prec[m] for m in DETECTION_METHODS) + r' \\')
        # Recall
        rec_vals = {m: agg_det[exp_name][m]['rec'] for m in DETECTION_METHODS}
        bold_rec = _bold_best(rec_vals, DETECTION_METHODS, higher_is_better=True)
        lines.append(' & Rec. & ' + ' & '.join(bold_rec[m] for m in DETECTION_METHODS) + r' \\')
        # F1
        f1_vals = {m: agg_det[exp_name][m]['f1'] for m in DETECTION_METHODS}
        bold_f1 = _bold_best(f1_vals, DETECTION_METHODS, higher_is_better=True)
        lines.append(' & F1 & ' + ' & '.join(bold_f1[m] for m in DETECTION_METHODS) + r' \\')
        # Delay (lower is better)
        delay_vals = {m: agg_det[exp_name][m]['delay'] for m in DETECTION_METHODS}
        delay_std_vals = {m: agg_det[exp_name][m].get('delay_std') for m in DETECTION_METHODS}
        bold_delay = _bold_best_delay(delay_vals, delay_std_vals, DETECTION_METHODS)
        lines.append(' & Delay & ' + ' & '.join(bold_delay[m] for m in DETECTION_METHODS) + r' \\')
        if i < len(included) - 1:
            lines.append(r'\midrule')
    # Summary row
    lines.append(r'\midrule')
    for metric_key, metric_label, higher in [('prec', 'Prec.', True), ('rec', 'Rec.', True), ('f1', 'F1', True), ('delay', 'Delay', False)]:
        avg_vals = {}
        avg_std = {}
        for m in DETECTION_METHODS:
            vals = [agg_det[e][m][metric_key] for e in included if agg_det[e][m][metric_key] is not None]
            avg_vals[m] = float(np.mean(vals)) if vals else None
            avg_std[m] = None  # no std for averages-of-averages
        if metric_key == 'delay':
            bold = _bold_best_delay(avg_vals, avg_std, DETECTION_METHODS)
        else:
            bold = _bold_best(avg_vals, DETECTION_METHODS, higher_is_better=higher)
        prefix = r'\multirow{4}{*}{\textit{Avg.}}' if metric_key == 'prec' else ''
        lines.append(prefix + ' & ' + metric_label + ' & ' + ' & '.join(bold[m] for m in DETECTION_METHODS) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    (summary_dir / 'detection_table.tex').write_text('\n'.join(lines), encoding='utf-8')


def _diagnosis_method_label(m: str) -> str:
    if m == 'dist_sage_w3':
        return 'Dist-SAGE (w3)'
    if m == 'dist_sage_w5':
        return 'Dist-SAGE (w5)'
    parts = m.split('_')
    if len(parts) >= 2:
        return f'{parts[0].title()}-{parts[1].upper()}'
    return m.replace('_', '-').title()


def write_diagnosis_csv(agg_diag, summary_dir: Path):
    # Columns: Experiment, then for each method: H@1, H@2, MRR
    method_order_csv = [
        'dist_sage', 'dist_sage_w3', 'dist_sage_w5', 'dist_pfi', 'dist_shap',
        'delta_sage', 'delta_pfi', 'delta_shap', 'meanph_sage', 'meanph_pfi', 'meanph_shap',
    ]
    method_cols = [(m, _diagnosis_method_label(m)) for m in method_order_csv]
    header = ['Experiment']
    for _m, label in method_cols:
        header.extend([f'{label}_H@1', f'{label}_H@2', f'{label}_MRR'])
    lines = [','.join(header)]
    for exp_name in EXP_ORDER:
        if exp_name not in agg_diag:
            continue
        row = [exp_display_name(exp_name)]
        for method, label in method_cols:
            if method not in agg_diag[exp_name]:
                row.extend(['', '', ''])
                continue
            r = agg_diag[exp_name][method]
            row.extend([f"{r['h1']:.3f}", f"{r['h2']:.3f}", f"{r['mrr']:.3f}"])
        lines.append(','.join(row))
    (summary_dir / 'diagnosis_table.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_diagnosis_tex(agg_diag, summary_dir: Path):
    # Build list of methods that appear in data
    methods_found = []
    for exp_name in list(agg_diag.keys())[:1] if agg_diag else []:
        methods_found = list(agg_diag[exp_name].keys())
        break
    if not methods_found:
        return
    method_labels = [_diagnosis_method_label(m) for m in methods_found]
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Drift diagnosis performance (feature ranking quality) across datasets.}',
        r'\label{tab:diagnosis}',
        r'\small',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{ll' + 'c' * len(methods_found) + '}',
        r'\toprule',
        'Dataset & Metric & ' + ' & '.join(method_labels) + r' \\',
        r'\midrule',
    ]
    for exp_name in EXP_ORDER:
        if exp_name not in agg_diag:
            continue
        display = exp_display_name(exp_name)
        lines.append(r'\multirow{3}{*}{' + display + r'} & Hits@1 & ')
        row1 = []
        for m in methods_found:
            row1.append(f"{agg_diag[exp_name].get(m, {}).get('h1', 0):.2f}")
        lines[-1] += ' & '.join(row1) + r' \\'
        lines.append(' & Hits@2 & ')
        row2 = [f"{agg_diag[exp_name].get(m, {}).get('h2', 0):.2f}" for m in methods_found]
        lines[-1] += ' & '.join(row2) + r' \\'
        lines.append(' & MRR & ')
        row3 = [f"{agg_diag[exp_name].get(m, {}).get('mrr', 0):.2f}" for m in methods_found]
        lines[-1] += ' & '.join(row3) + r' \\'
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')
    (summary_dir / 'diagnosis_table.tex').write_text('\n'.join(lines), encoding='utf-8')


# Three experiments for the 2x3 loss/RDS panel (Sudden, Gradual, Recurring)
LOSS_PANEL_EXPERIMENTS = ['fed_heart_sudden', 'diabetes_gradual', 'elec2_recurring']


def _load_run_plot_data(run_dir: Path):
    """Load loss, RDS, thresholds, metadata, trigger_results for one run. Return dict or None."""
    loss_path = run_dir / 'loss_history.npy'
    meta_path = run_dir / 'metadata.json'
    trigger_path = run_dir / 'trigger_results.json'
    if not loss_path.exists() or not meta_path.exists():
        return None
    loss = np.load(loss_path)
    with open(meta_path) as f:
        meta = json.load(f)
    exp_cfg = meta.get('exp_cfg', {})
    # Warmup/calibration: from exp_cfg with run_drift_types defaults
    warmup = exp_cfg.get('warmup_rounds', 40)
    cal_start = exp_cfg.get('calibration_start', 41)
    cal_end = exp_cfg.get('calibration_end', 80)
    t0 = meta.get('t0')
    t1 = meta.get('t1')
    trigger_data = {}
    if trigger_path.exists():
        with open(trigger_path) as f:
            trigger_data = json.load(f)
    rds_trigger_round = None
    if trigger_data.get('detection_t0', {}).get('rds', {}).get('triggered'):
        rds_trigger_round = trigger_data['detection_t0']['rds'].get('round')
    rds_t1_round = None
    if t1 is not None and trigger_data.get('detection_t1', {}).get('rds', {}).get('triggered'):
        rds_t1_round = trigger_data['detection_t1']['rds'].get('round')

    rds_scores = None
    rds_thresholds = None
    rds_path = run_dir / 'rds_scores.npy'
    thresh_path = run_dir / 'rds_thresholds.npy'
    if rds_path.exists():
        rds = np.load(rds_path)
        rds_scores = np.nanmean(rds, axis=1) if rds.ndim > 1 else rds
    if thresh_path.exists():
        th = np.load(thresh_path, allow_pickle=True)
        rds_thresholds = th  # 1d, same length as rds_scores; may contain None/nan

    n_rounds = len(loss)
    return {
        'loss': loss,
        'n_rounds': n_rounds,
        'warmup': warmup,
        'cal_start': cal_start,
        'cal_end': cal_end,
        't0': t0,
        't1': t1,
        'rds_trigger_round': rds_trigger_round,
        'rds_t1_round': rds_t1_round,
        'rds_scores': rds_scores,
        'rds_thresholds': rds_thresholds,
    }


def plot_loss_panel(runs_by_exp, summary_dir: Path):
    """2x3 grid: top row = Loss, bottom row = RDS. Columns = FedHeart (Sudden), Diabetes (Gradual), ELEC2 (Recurring).
    Publication-style colors; shared legend for Warmup, Calibration, Drift onset, RDS Trigger, Threshold.
    """
    if not PLOTTING_AVAILABLE:
        return
    exps = [e for e in LOSS_PANEL_EXPERIMENTS if e in runs_by_exp and runs_by_exp[e]]
    if len(exps) != 3:
        # Fallback: if any of the three missing, try original multi-panel or skip
        exps = [e for e in EXP_ORDER if e in runs_by_exp and runs_by_exp[e]]
        if not exps:
            return
        n = len(exps)
        ncol = 2
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for idx, exp_name in enumerate(exps):
            ax = axes[idx]
            _seed, run_dir = runs_by_exp[exp_name][0]
            loss_path = run_dir / 'loss_history.npy'
            meta_path = run_dir / 'metadata.json'
            if not loss_path.exists():
                ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(exp_display_name(exp_name))
                continue
            loss = np.load(loss_path)
            t0 = None
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                t0 = meta.get('t0')
            rounds = np.arange(1, len(loss) + 1)
            ax.plot(rounds, loss, 'b-', linewidth=1.5, label='Global Loss')
            if t0 is not None:
                ax.axvline(x=t0, color='r', linestyle='--', linewidth=2, label=f't0={t0}')
            ax.set_xlabel('Round')
            ax.set_ylabel('Loss')
            ax.set_title(exp_display_name(exp_name))
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        for j in range(len(exps), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(summary_dir / f'loss_panel.{ext}', dpi=300, bbox_inches='tight')
        plt.close()
        return

    # 2x3 panel for FedHeart, Diabetes, ELEC2
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex='col')
    # Publication-style colors (colorblind-friendly, distinct)
    color_loss = '#2171b5'
    color_t0 = '#cb181d'
    color_trigger = '#238b45'
    color_threshold = '#e6550d'
    gray_warmup = '#969696'
    amber_cal = '#fec44f'
    alpha_warmup = 0.35
    alpha_cal = 0.25

    legend_handles = []
    legend_labels = []

    for col, exp_name in enumerate(exps):
        _seed, run_dir = runs_by_exp[exp_name][0]
        data = _load_run_plot_data(run_dir)
        if data is None:
            axes[0, col].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, col].transAxes)
            axes[0, col].set_title(exp_display_name(exp_name))
            axes[1, col].set_visible(False)
            continue

        loss = data['loss']
        n_rounds = data['n_rounds']
        warmup = data['warmup']
        cal_start = data['cal_start']
        cal_end = data['cal_end']
        t0 = data['t0']
        t1 = data['t1']
        rds_trigger_round = data['rds_trigger_round']
        rds_t1_round = data['rds_t1_round']
        rds_scores = data['rds_scores']
        rds_thresholds = data['rds_thresholds']

        rounds = np.arange(1, n_rounds + 1)
        ax_loss = axes[0, col]
        ax_rds = axes[1, col]

        # ---- Top: Loss ----
        ax_loss.plot(rounds, loss, color=color_loss, linewidth=1.8, label='Loss', zorder=3)
        ax_loss.axvspan(1, warmup + 0.5, alpha=alpha_warmup, color=gray_warmup, zorder=0)
        ax_loss.axvspan(cal_start - 0.5, cal_end + 0.5, alpha=alpha_cal, color=amber_cal, zorder=0)
        if t0 is not None:
            ax_loss.axvline(x=t0, color=color_t0, linestyle='--', linewidth=1.8, label='Drift onset', zorder=2)
        if rds_trigger_round is not None:
            ax_loss.axvline(x=rds_trigger_round, color=color_trigger, linestyle=':', linewidth=2, label='RDS trigger', zorder=2)
        if t1 is not None:
            ax_loss.axvline(x=t1, color='#88419d', linestyle='-.', linewidth=1.5, label='Reversion (t1)', zorder=2)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title(exp_display_name(exp_name), fontsize=11)
        ax_loss.grid(True, alpha=0.35, linestyle='-')
        ax_loss.set_xlim(1, n_rounds)

        # ---- Bottom: RDS ----
        if rds_scores is not None and len(rds_scores) > 0:
            # RDS rounds: from warmup+1 (1-indexed)
            rds_rounds = np.arange(warmup + 1, warmup + 1 + len(rds_scores), dtype=float)
            if len(rds_rounds) > len(rds_scores):
                rds_rounds = rds_rounds[:len(rds_scores)]
            elif len(rds_scores) > len(rds_rounds):
                rds_scores = rds_scores[:len(rds_rounds)]
            ax_rds.plot(rds_rounds, rds_scores, color='#6a51a3', linewidth=1.8, label='RDS score', zorder=3)
            ax_rds.axvspan(1, warmup + 0.5, alpha=alpha_warmup, color=gray_warmup, zorder=0)
            ax_rds.axvspan(cal_start - 0.5, cal_end + 0.5, alpha=alpha_cal, color=amber_cal, zorder=0)
            if rds_thresholds is not None and len(rds_thresholds) == len(rds_scores):
                th_vals = []
                th_rounds = []
                for i, t in enumerate(rds_thresholds):
                    if t is not None and (isinstance(t, (int, float)) and np.isfinite(t)):
                        th_vals.append(float(t))
                        th_rounds.append(rds_rounds[i] if i < len(rds_rounds) else warmup + 1 + i)
                if th_vals:
                    ax_rds.plot(th_rounds, th_vals, color=color_threshold, linestyle='--',
                                linewidth=1.8, label='Threshold', zorder=2)
            if t0 is not None:
                ax_rds.axvline(x=t0, color=color_t0, linestyle='--', linewidth=1.8, zorder=2)
            if rds_trigger_round is not None:
                ax_rds.axvline(x=rds_trigger_round, color=color_trigger, linestyle=':', linewidth=2, zorder=2)
            if t1 is not None and rds_t1_round is not None:
                ax_rds.axvline(x=rds_t1_round, color='#88419d', linestyle='-.', linewidth=1.5, zorder=2)
            ax_rds.set_ylabel('RDS score', fontsize=10)
            ax_rds.grid(True, alpha=0.35, linestyle='-')
            ax_rds.set_xlim(1, n_rounds)
        else:
            ax_rds.text(0.5, 0.5, 'No RDS data', ha='center', va='center', transform=ax_rds.transAxes, fontsize=10)
            ax_rds.set_ylabel('RDS score', fontsize=10)
            ax_rds.set_xlim(1, n_rounds)

        ax_rds.set_xlabel('Round', fontsize=10)

    # Shared legend (one set of entries)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=gray_warmup, alpha=alpha_warmup, edgecolor='none', label='Warmup'),
        Patch(facecolor=amber_cal, alpha=alpha_cal, edgecolor='none', label='Calibration'),
        Line2D([0], [0], color=color_t0, linestyle='--', linewidth=2, label='Drift onset'),
        Line2D([0], [0], color=color_trigger, linestyle=':', linewidth=2, label='RDS trigger'),
        Line2D([0], [0], color=color_threshold, linestyle='--', linewidth=2, label='Threshold'),
        Line2D([0], [0], color='#88419d', linestyle='-.', linewidth=1.5, label='Reversion (t1)'),
    ]
    legend_labels = ['Warmup', 'Calibration', 'Drift onset', 'RDS trigger', 'Threshold', 'Reversion (t1)']
    fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', ncol=3, fontsize=9,
               frameon=True, fancybox=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ['png', 'pdf']:
        fig.savefig(summary_dir / f'loss_panel.{ext}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_fi_rds_panel(runs_by_exp, summary_dir: Path):
    if not PLOTTING_AVAILABLE:
        return
    exps = [e for e in EXP_ORDER if e in runs_by_exp and runs_by_exp[e]]
    if not exps:
        return
    n = len(exps)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, exp_name in enumerate(exps):
        ax = axes[idx]
        _seed, run_dir = runs_by_exp[exp_name][0]
        rds_path = run_dir / 'rds_scores.npy'
        meta_path = run_dir / 'metadata.json'
        if not rds_path.exists():
            ax.text(0.5, 0.5, 'No RDS data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp_display_name(exp_name))
            continue
        rds = np.load(rds_path)
        t0 = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            t0 = meta.get('t0')
        # rds can be 1d (scores per round) or 2d
        if rds.ndim == 1:
            y = rds
        else:
            y = np.nanmean(rds, axis=1) if rds.shape[1] > 1 else rds[:, 0]
        rounds = np.arange(1, len(y) + 1)
        ax.plot(rounds, y, 'purple', linewidth=1.5, label='RDS Score')
        if t0 is not None:
            ax.axvline(x=t0, color='r', linestyle='--', linewidth=2, label=f't0={t0}')
        ax.set_xlabel('Round')
        ax.set_ylabel('RDS Score')
        ax.set_title(exp_display_name(exp_name))
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(len(exps), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(summary_dir / f'fi_rds_panel.{ext}', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Aggregate drift_types results and generate summary.')
    parser.add_argument('--results-dir', type=str, default='results/drift_types',
                        help='Path to results/drift_types directory')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}")
        return
    summary_dir = results_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs_by_exp = discover_runs(results_dir)
    if not runs_by_exp:
        print("No run directories (*_seed<N>) found.")
        return
    print(f"Found {sum(len(v) for v in runs_by_exp.values())} runs across {len(runs_by_exp)} experiments.")

    # Detection tables
    agg_det = aggregate_detection(runs_by_exp)
    write_detection_csv(agg_det, summary_dir)
    write_detection_tex(agg_det, summary_dir)
    print(f"  Wrote {summary_dir / 'detection_table.csv'} and .tex")

    # Diagnosis tables
    agg_diag = aggregate_diagnosis(runs_by_exp)
    if agg_diag:
        write_diagnosis_csv(agg_diag, summary_dir)
        write_diagnosis_tex(agg_diag, summary_dir)
        print(f"  Wrote {summary_dir / 'diagnosis_table.csv'} and .tex")
    else:
        print("  No diagnosis metrics found (skip diagnosis tables).")

    # Panels
    if PLOTTING_AVAILABLE:
        plot_loss_panel(runs_by_exp, summary_dir)
        plot_fi_rds_panel(runs_by_exp, summary_dir)
        print(f"  Wrote {summary_dir / 'loss_panel.png'} and .pdf")
        print(f"  Wrote {summary_dir / 'fi_rds_panel.png'} and .pdf")
    else:
        print("  matplotlib not available, skipping panels.")

    print(f"Summary written to {summary_dir}")


if __name__ == '__main__':
    main()
