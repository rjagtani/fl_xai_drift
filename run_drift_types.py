"""
Drift-type experiments: sudden, gradual.

Runs one experiment per (dataset, drift_type) combination:
  Sudden:    FedHeart (4 clients), Agrawal (10), Wine (5), Credit-G (5)
  Gradual:   Hyperplane (10 clients), Diabetes (6)

Results saved to  results/drift_types/<name>_seed<seed>/

Usage:
  python run_drift_types.py                       # all experiments
  python run_drift_types.py --experiments wine_sudden  # one
  python run_drift_types.py --redetect             # re-run detection only
  python run_drift_types.py --diagnose-only wine_sudden_seed42  # diagnosis only (no training)
"""

import sys
import csv
import time
import random
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np
import torch

sys.path.insert(0, '.')

from src.config import (
    ExperimentConfig, DatasetConfig, FLConfig, DriftConfig,
    TriggerConfig, DiagnosisConfig, MetricsConfig,
)
from src.fl_trainer import FLTrainer
from src.triggers.drift_detectors import MultiMethodDetector, bonferroni_k
from src.utils.visualization import plot_loss_and_rds_detection
from src.diagnosis.diagnosis_engine import DiagnosisEngine
from src.metrics.evaluation import compute_metrics, metrics_to_dict


SEED = 42
OUTPUT_DIR = Path('results') / 'drift_types'

EXPERIMENTS = {
    'fed_heart_sudden': {
        'dataset': 'fed_heart',
        'drift_type': 'sudden',
        'drift_mechanism': 'feature_noise',
        'n_clients': 4,
        'n_rounds': 300,
        'n_features': 13,
        'hidden_sizes': [64, 32],
        'learning_rate': 0.005,
        'warmup_rounds': 60,
        'calibration_start': 61,
        'calibration_end': 100,
        'noise_std': 2.0,
        'drifted_client_proportion': 0.5,
        'drifted_features': {2},
        't0_range': (105, 297),
    },
    'hyperplane_gradual': {
        'dataset': 'hyperplane',
        'drift_type': 'gradual',
        'drift_mechanism': 'coeff_change',
        'n_clients': 10,
        'n_rounds': 200,
        'n_features': 5,
        'n_samples_per_client': 500,
        'n_drift_features': 1,
        'hidden_sizes': [128, 64],
        'drift_magnitude': 0.5,
        'drifted_client_proportion': 0.5,
        'drifted_features': {0},
        't0_range': (85, 160),
        'transition_window': 100,
    },
    'hyperplane_gradual_low': {
        'dataset': 'hyperplane',
        'drift_type': 'gradual',
        'drift_mechanism': 'coeff_change',
        'n_clients': 10,
        'n_rounds': 200,
        'n_features': 5,
        'n_samples_per_client': 500,
        'n_drift_features': 1,
        'hidden_sizes': [128, 64],
        'drift_magnitude': 0.15,
        'drifted_client_proportion': 0.3,
        'drifted_features': {0},
        't0_range': (85, 160),
        'transition_window': 100,
    },
    'hyperplane_gradual_vlow': {
        'dataset': 'hyperplane',
        'drift_type': 'gradual',
        'drift_mechanism': 'coeff_change',
        'n_clients': 10,
        'n_rounds': 200,
        'n_features': 5,
        'n_samples_per_client': 500,
        'n_drift_features': 1,
        'hidden_sizes': [128, 64],
        'drift_magnitude': 0.05,
        'drifted_client_proportion': 0.1,
        'drifted_features': {0},
        't0_range': (85, 160),
        'transition_window': 100,
    },
    'agrawal_sudden': {
        'dataset': 'agrawal',
        'drift_type': 'sudden',
        'drift_mechanism': 'concept_switch',
        'n_clients': 10,
        'n_rounds': 200,
        'n_features': 9,
        'n_samples_per_client': 2000,
        'hidden_sizes': [128, 64],
        'classification_function_pre': 0,
        'classification_function_post': 1,
        'drifted_client_proportion': 0.5,
        'drifted_features': {0, 2},
        't0_range': (85, 160),
    },
    'wine_sudden': {
        'dataset': 'wine',
        'drift_type': 'sudden',
        'drift_mechanism': 'feature_noise',
        'n_clients': 5,
        'n_rounds': 200,
        'n_features': 11,
        'hidden_sizes': [64, 32],
        'noise_std': 2.0,
        'drifted_client_proportion': 0.5,
        'drifted_features': {10},
        't0_range': (85, 160),
    },
    'diabetes_gradual': {
        'dataset': 'diabetes',
        'drift_type': 'gradual',
        'drift_mechanism': 'feature_noise',
        'n_clients': 6,
        'n_rounds': 200,
        'n_features': 8,
        'hidden_sizes': [128, 64],
        'noise_std': 2.0,
        'drifted_client_proportion': 0.5,
        'drifted_features': {1},
        't0_range': (85, 160),
        'transition_window': 20,
    },
    'credit_sudden': {
        'dataset': 'credit',
        'drift_type': 'sudden',
        'drift_mechanism': 'feature_noise',
        'n_clients': 5,
        'n_rounds': 200,
        'n_features': 20,
        'hidden_sizes': [128, 64],
        'noise_std': 2.0,
        'drifted_client_proportion': 0.5,
        'drifted_features': {1},
        't0_range': (85, 160),
    },
}



def compute_drift_schedule(
    drift_type: str,
    round_num: int,
    t0: int,
    exp_cfg: dict,
) -> dict:
    """
    Return drift state for a given round.

    Returns dict with:
        'active': bool        - is drift on?
        'alpha':  float       - drift intensity 0..1 (for gradual; 1.0 for sudden/recurring)
        'flip_prob': float    - effective flip prob (for label-flip datasets)
    """
    target_flip = exp_cfg.get('drift_flip_prob', 0.5)
    target_mag = exp_cfg.get('drift_magnitude', 0.3)

    if drift_type == 'sudden':
        active = round_num >= t0
        return {'active': active, 'alpha': 1.0 if active else 0.0,
                'flip_prob': target_flip if active else 0.0}

    elif drift_type == 'gradual':
        W = exp_cfg.get('transition_window', 30)
        if round_num < t0:
            return {'active': False, 'alpha': 0.0, 'flip_prob': 0.0}
        elif round_num < t0 + W:
            alpha = (round_num - t0 + 1) / W
            return {'active': True, 'alpha': alpha,
                    'flip_prob': target_flip * alpha}
        else:
            return {'active': True, 'alpha': 1.0,
                    'flip_prob': target_flip}

    elif drift_type == 'recurring':
        period = exp_cfg.get('recurring_period', 100)
        t1 = t0 + period
        if t0 <= round_num < t1:
            return {'active': True, 'alpha': 1.0,
                    'flip_prob': target_flip}
        else:
            return {'active': False, 'alpha': 0.0, 'flip_prob': 0.0}

    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")



def get_round_data_custom(
    data_generator,
    round_num: int,
    drifted_clients: Set[int],
    schedule: dict,
    exp_cfg: dict,
    dataset_name: str,
):
    """
    Generate client data for one round, respecting the drift schedule.

    drift_mechanism (from exp_cfg):
      'feature_noise'  – add Gaussian noise to drifted features (covariate drift)
      'label_flip'     – flip labels conditioned on a feature (concept drift)
      'concept_switch' – Agrawal classification-function switch
      'coeff_change'   – Hyperplane coefficient change
    """
    active = schedule['active']
    alpha = schedule['alpha']
    flip_prob = schedule['flip_prob']
    mechanism = exp_cfg.get('drift_mechanism', 'label_flip')

    if dataset_name == 'hyperplane':
        return _get_hyperplane_gradual_data(
            data_generator, round_num, drifted_clients, alpha, exp_cfg)
    elif dataset_name == 'agrawal':
        return data_generator.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=active,
        )
    elif mechanism == 'feature_noise':
        client_data = data_generator.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=False,
        )
        if active:
            noise_std = alpha * exp_cfg.get('noise_std', 2.0)
            drift_features = sorted(exp_cfg.get('drifted_features', set()))
            for client_id in drifted_clients:
                cd = client_data[client_id]
                rng = np.random.default_rng(
                    data_generator.seed + client_id * 1000 + round_num)
                for f_idx in drift_features:
                    cd.X_train[:, f_idx] += rng.normal(
                        0, noise_std, size=len(cd.X_train)).astype(cd.X_train.dtype)
                    cd.X_val[:, f_idx] += rng.normal(
                        0, noise_std, size=len(cd.X_val)).astype(cd.X_val.dtype)
        return client_data
    else:
        if not active:
            return data_generator.generate_static_client_data(
                drifted_clients=drifted_clients,
                generate_drifted=False,
            )
        else:
            return data_generator.generate_static_client_data(
                drifted_clients=drifted_clients,
                generate_drifted=True,
                flip_prob_override=flip_prob,
            )


def _get_hyperplane_gradual_data(
    gen, round_num, drifted_clients, alpha, exp_cfg,
):
    """
    For Hyperplane gradual drift: blend baseline and drifted pools.

    alpha=0: pure baseline.  alpha=1: pure drifted.
    Intermediate: sample (1-alpha)*N from baseline, alpha*N from drifted, concatenate.
    """
    from src.data.base import ClientDataset
    from sklearn.model_selection import train_test_split

    mag = exp_cfg.get('drift_magnitude', 0.3)
    pool_size = max(gen.n_samples_per_client * 20, 10000)
    baseline_pool = gen._get_baseline_pool(pool_size)
    drifted_pool = gen._get_drifted_pool(mag, pool_size)

    client_datasets = {}
    for client_id in range(gen.n_clients):
        is_drifted = client_id in drifted_clients
        if is_drifted and alpha > 0:
            n_drifted = int(alpha * gen.n_samples_per_client)
            n_baseline = gen.n_samples_per_client - n_drifted
            rng = np.random.default_rng(gen.seed + client_id * 1000 + round_num)
            if n_baseline > 0:
                idx_b = rng.choice(len(baseline_pool), size=n_baseline, replace=True)
                Xb = baseline_pool[idx_b, :-1]
                yb = baseline_pool[idx_b, -1].astype(int)
            else:
                Xb = np.empty((0, gen.n_features))
                yb = np.empty(0, dtype=int)
            if n_drifted > 0:
                idx_d = rng.choice(len(drifted_pool), size=n_drifted, replace=True)
                Xd = drifted_pool[idx_d, :-1]
                yd = drifted_pool[idx_d, -1].astype(int)
            else:
                Xd = np.empty((0, gen.n_features))
                yd = np.empty(0, dtype=int)
            X = np.vstack([Xb, Xd])
            y = np.concatenate([yb, yd])
            perm = rng.permutation(len(y))
            X, y = X[perm], y[perm]
        else:
            X, y = gen._sample_client_data(baseline_pool, client_id, round_offset=round_num)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=gen.test_size,
            random_state=gen.seed + client_id,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        client_datasets[client_id] = ClientDataset(
            client_id=client_id,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            is_drifted=is_drifted and alpha > 0,
        )
    return client_datasets



def run_single_experiment(exp_name: str, exp_cfg: dict, seed: int = SEED):
    """Run FL training with custom drift schedule, then detect drift and diagnose."""
    dataset_name = exp_cfg['dataset']
    drift_type = exp_cfg['drift_type']
    n_rounds = exp_cfg['n_rounds']
    n_clients = exp_cfg['n_clients']

    rng = random.Random(seed)
    t0 = rng.randint(exp_cfg['t0_range'][0], exp_cfg['t0_range'][1])

    t1 = None
    if drift_type == 'recurring':
        t1 = t0 + exp_cfg.get('recurring_period', 100)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    timings: Dict[str, float] = {}

    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp_name}")
    print(f"  Dataset: {dataset_name}, Drift: {drift_type}")
    print(f"  Clients: {n_clients}, Rounds: {n_rounds}")
    print(f"  t0={t0}" + (f", t1={t1}" if t1 else ""))
    print(f"{'=' * 60}")

    dataset_config = DatasetConfig(
        name=dataset_name,
        n_features=exp_cfg['n_features'],
        n_samples_per_client=exp_cfg.get('n_samples_per_client', 500),
    )
    if dataset_name in ('fed_heart', 'wine'):
        dataset_config.drift_condition_feature = exp_cfg.get('drift_condition_feature', 'age')
        dataset_config.drift_condition_threshold = exp_cfg.get('drift_condition_threshold', None)
        dataset_config.drift_flip_prob = exp_cfg.get('drift_flip_prob', 0.3)
    if dataset_name in ('wine', 'diabetes', 'credit'):
        dataset_config.use_all_data_per_client = exp_cfg.get('use_all_data_per_client', True)
    if dataset_name == 'hyperplane':
        dataset_config.n_drift_features = exp_cfg.get('n_drift_features', 1)
    if dataset_name == 'agrawal':
        dataset_config.classification_function_pre = exp_cfg.get('classification_function_pre', 0)
        dataset_config.classification_function_post = exp_cfg.get('classification_function_post', 1)

    dummy_t0 = n_rounds + 1000

    config = ExperimentConfig(
        seed=seed,
        experiment_name=f"{exp_name}_seed{seed}",
        dataset=dataset_config,
        fl=FLConfig(
            n_clients=n_clients,
            n_rounds=n_rounds,
            hidden_sizes=exp_cfg.get('hidden_sizes', [128, 64]),
            learning_rate=exp_cfg.get('learning_rate', 0.05),
        ),
        drift=DriftConfig(
            t0=dummy_t0,
            drifted_client_proportion=exp_cfg.get('drifted_client_proportion', 0.5),
            drift_magnitude=exp_cfg.get('drift_magnitude', 1.0),
            drifted_features=exp_cfg.get('drifted_features', set()),
        ),
        trigger=TriggerConfig(),
        diagnosis=DiagnosisConfig(),
        metrics=MetricsConfig(
            k=len(exp_cfg.get('drifted_features', set())),
            use_ground_truth_k=True,
        ),
        base_output_dir=OUTPUT_DIR,
    )

    exp_dir = config.output_dir

    trainer = FLTrainer(config)
    drifted_clients = set()
    n_drifted = int(n_clients * exp_cfg.get('drifted_client_proportion', 0.5))
    drifted_clients = set(range(n_clients - n_drifted, n_clients))

    data_gen = trainer.data_generator

    if hasattr(data_gen, '_ensure_data'):
        data_gen._ensure_data()
    if hasattr(data_gen, '_ensure_pools'):
        data_gen._ensure_pools()

    original_get_round_data = trainer.get_round_data

    pre_drift_cache = [None]

    def custom_get_round_data(round_num):
        schedule = compute_drift_schedule(drift_type, round_num, t0, exp_cfg)
        if drift_type == 'gradual' and round_num < t0:
            if pre_drift_cache[0] is None:
                schedule_pre = compute_drift_schedule(drift_type, 1, t0, exp_cfg)
                pre_drift_cache[0] = get_round_data_custom(
                    data_gen, 1, drifted_clients, schedule_pre, exp_cfg, dataset_name
                )
            return pre_drift_cache[0]
        return get_round_data_custom(
            data_gen, round_num, drifted_clients, schedule, exp_cfg, dataset_name
        )

    trainer.get_round_data = custom_get_round_data

    config.create_directories()
    config.save()

    print(f"Starting FL training: {exp_name}")
    print(f"  Drifted clients: {drifted_clients}")

    if hasattr(data_gen, '_raw_condition_per_hospital') and data_gen._raw_condition_per_hospital is not None:
        thr = data_gen.drift_condition_threshold
        print(f"  Drift condition: {data_gen.drift_condition_feature} >= {thr}")
        for cid in range(n_clients):
            raw = data_gen._raw_condition_per_hospital[cid]
            n_total = len(raw)
            n_cond = int((raw >= thr).sum())
            tag = " [DRIFTED]" if cid in drifted_clients else ""
            print(f"    Client {cid}: {n_cond}/{n_total} samples meet condition{tag}")

    round_times: List[float] = []
    fl_train_start = time.perf_counter()
    for round_num in range(1, n_rounds + 1):
        t_rnd0 = time.perf_counter()
        round_log = trainer.train_round(round_num)
        round_times.append(time.perf_counter() - t_rnd0)
        if round_num % 20 == 0 or round_num == 1:
            sch = compute_drift_schedule(drift_type, round_num, t0, exp_cfg)
            drift_info = f"drift_alpha={sch['alpha']:.2f}" if sch['active'] else "no drift"
            print(
                f"  Round {round_num}/{n_rounds}: "
                f"Loss={round_log['global_loss']:.4f}, "
                f"Acc={round_log['global_accuracy']:.2%} [{drift_info}]"
            )
    fl_train_total = time.perf_counter() - fl_train_start
    timings['fl_train_total_s'] = fl_train_total
    timings['fl_train_per_round_avg_s'] = float(np.mean(round_times))

    loss_history = trainer.global_loss_series
    loss_matrix = trainer.client_loss_matrix
    np.save(exp_dir / 'loss_history.npy', loss_history)
    np.save(exp_dir / 'loss_matrix.npy', loss_matrix)

    round_data_for_weights = trainer.get_round_data(1)
    _client_sizes = np.array([len(round_data_for_weights[i].X_train)
                              for i in range(n_clients)], dtype=np.float64)
    client_weights = _client_sizes / _client_sizes.sum()

    meta = {
        'exp_name': exp_name,
        'dataset': dataset_name,
        'drift_type': drift_type,
        'seed': seed,
        't0': t0,
        't1': t1,
        'n_rounds': n_rounds,
        'n_clients': n_clients,
        'drifted_clients': sorted(drifted_clients),
        'exp_cfg': {k: (list(v) if isinstance(v, set) else v)
                    for k, v in exp_cfg.items()},
    }
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("  Running drift detection (change point at t0) ...")
    warmup = exp_cfg.get('warmup_rounds', 40)
    cal_start = exp_cfg.get('calibration_start', 41)
    cal_end = exp_cfg.get('calibration_end', 80)
    k = bonferroni_k(n_rounds, cal_end, 0.05)

    detector = MultiMethodDetector(
        warmup_rounds=warmup,
        calibration_start=cal_start,
        calibration_end=cal_end,
        n_rounds=n_rounds,
        rds_window=5,
        rds_alpha=k,
        confirm_consecutive=3,
        min_instances=5,
        cusum_k_ref=0.5,
        cusum_h=7.0,
        fwer_p=0.05,
    )
    detect_start = time.perf_counter()
    results_t0 = detector.detect(loss_matrix, np.array(loss_history), t0,
                                  client_weights=client_weights)
    timings['detection_rds_loss_s'] = time.perf_counter() - detect_start

    results_t1 = None
    if drift_type == 'recurring' and t1 is not None:
        print(f"  Running drift detection (reversion at t1={t1}) ...")
        recal_start = min(t0 + 10, t1 - 30)
        recal_end = min(t0 + 40, t1 - 5)
        if recal_end > recal_start + 5:
            k2 = bonferroni_k(n_rounds, recal_end, 0.05)
            detector2 = MultiMethodDetector(
                warmup_rounds=recal_start - 1,
                calibration_start=recal_start,
                calibration_end=recal_end,
                n_rounds=n_rounds,
                rds_window=5,
                rds_alpha=k2,
                confirm_consecutive=3,
                min_instances=5,
                cusum_k_ref=0.5,
                cusum_h=7.0,
                fwer_p=0.05,
            )
            results_t1 = detector2.detect(loss_matrix, np.array(loss_history), t1,
                                          client_weights=client_weights)

    def _format_result(res_dict, ground_truth_t):
        out = {}
        for method, r in res_dict.items():
            rnd = (r.trigger_round + 1) if (r.triggered and r.trigger_round is not None) else None
            delay = (rnd - ground_truth_t) if rnd is not None else None
            out[method] = {
                'triggered': r.triggered,
                'round': rnd,
                'delay': delay,
            }
        return out

    trigger_data = {
        'dataset': dataset_name,
        'drift_type': drift_type,
        'seed': seed,
        't0': t0,
        't1': t1,
        'detection_t0': _format_result(results_t0, t0),
    }
    if results_t1 is not None:
        trigger_data['detection_t1'] = _format_result(results_t1, t1)

    with open(exp_dir / 'trigger_results.json', 'w') as f:
        json.dump(trigger_data, f, indent=2)

    rds = results_t0.get('rds')
    if rds and rds.all_scores is not None:
        np.save(exp_dir / 'rds_scores.npy', rds.all_scores)
    if rds and rds.threshold_series is not None:
        np.save(exp_dir / 'rds_thresholds.npy', rds.threshold_series)

    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    candidates = []
    for method, r in results_t0.items():
        if r.triggered and r.trigger_round is not None:
            rnd = r.trigger_round + 1
            delay = rnd - t0
            if delay >= 0:
                candidates.append((method, rnd, delay))
    if candidates:
        candidates.sort(key=lambda x: x[2])
        best_method, best_round, best_delay = candidates[0]
    else:
        best_method, best_round, best_delay = None, None, None

    plot_loss_and_rds_detection(
        loss_series=loss_history,
        rds_scores=rds.all_scores if rds else None,
        t0=t0,
        trigger_round=best_round,
        threshold_series=rds.threshold_series if rds else None,
        calibration_start=cal_start,
        calibration_end=cal_end,
        warmup_rounds=warmup,
        save_path=plots_dir / 'loss_rds_detection.png',
    )

    print(f"\n  Results for {exp_name}:")
    print(f"    t0={t0}" + (f", t1={t1}" if t1 else ""))
    for method, r in results_t0.items():
        if r.triggered and r.trigger_round is not None:
            rnd = r.trigger_round + 1
            delay = rnd - t0
            if delay >= 0:
                print(f"    {method:12s}: TRIGGERED      round={rnd} delay={delay}")
            else:
                print(f"    {method:12s}: TRIGGERED (FP) round={rnd}")
        else:
            print(f"    {method:12s}: not triggered")

    if results_t1:
        print(f"  Reversion detection (t1={t1}):")
        for method, r in results_t1.items():
            if r.triggered and r.trigger_round is not None:
                rnd = r.trigger_round + 1
                delay = rnd - t1
                if delay >= 0:
                    print(f"    {method:12s}: TRIGGERED      round={rnd} delay={delay}")
                else:
                    print(f"    {method:12s}: TRIGGERED (FP) round={rnd}")
            else:
                print(f"    {method:12s}: not triggered")

    print(f"  Saved to: {exp_dir}")

    rds_result = results_t0.get('rds')
    if rds_result and rds_result.triggered and rds_result.trigger_round is not None:
        trigger_round = rds_result.trigger_round + 1
    else:
        trigger_round = None

    if trigger_round is None:
        print("  RDS did not trigger — skipping diagnosis.")
    else:
        print(f"\n  Running diagnosis at trigger round {trigger_round} ...")
        ground_truth = exp_cfg.get('drifted_features', set())
        if isinstance(ground_truth, list):
            ground_truth = set(ground_truth)
        feature_names = trainer.get_feature_names()

        mechanism = exp_cfg.get('drift_mechanism', 'label_flip')
        _diag_get_round_data = trainer.get_round_data

        if mechanism in ('label_flip', 'concept_switch'):
            def _ref_get_round_data(round_num, _trigger=trigger_round,
                                     _inner=_diag_get_round_data):
                if round_num == _trigger:
                    return _inner(round_num - 1)
                return _inner(round_num)
            trainer.get_round_data = _ref_get_round_data

        diag_start = time.perf_counter()
        engine = DiagnosisEngine(config, trainer)
        diag_results = engine.run_diagnosis(trigger_round,
                                            client_weights=client_weights)
        diag_total = time.perf_counter() - diag_start
        timings['diagnosis_total_s'] = diag_total

        sage_client_times = diag_results.get('sage_trigger_client_times', [])
        n_diag_rounds = len(diag_results.get('diagnosis_rounds', []))
        timings['sage_fi_compute_total_s'] = diag_results.get('fi_compute_time_s', 0.0)
        timings['sage_n_rounds_in_window'] = float(n_diag_rounds)
        if sage_client_times:
            arr = np.array([t for t in sage_client_times if not np.isnan(t)])
            if len(arr) > 0:
                timings['sage_trigger_client_max_s'] = float(arr.max())
                timings['sage_trigger_client_mean_s'] = float(arr.mean())
                timings['sage_trigger_client_median_s'] = float(np.median(arr))

        trainer.get_round_data = _diag_get_round_data

        rankings = engine.get_feature_rankings(diag_results)

        diag_dir = exp_dir / 'diagnosis'
        diag_dir.mkdir(parents=True, exist_ok=True)

        engine.save_results(diag_results, diag_dir)

        print(f"\n  Diagnosis results (ground truth drifted features: {ground_truth}):")
        print(f"  {'Method':<25s} {'Ranking (top 5)':<40s} {'Hits@1':>7s} {'Hits@2':>7s} {'MRR':>7s}")
        print(f"  {'-'*90}")

        all_metrics = {}
        for method_name, ranking in sorted(rankings.items()):
            m1 = compute_metrics(ranking, ground_truth, k=1)
            m2 = compute_metrics(ranking, ground_truth, k=2)
            m_full = compute_metrics(ranking, ground_truth, k=len(ground_truth))

            top5 = [f"{feature_names[i]}" for i in ranking[:5]]
            print(f"  {method_name:<25s} {str(top5):<40s} {m1.hits_at_k:>7.0f} {m2.hits_at_k:>7.0f} {m_full.mrr:>7.3f}")

            all_metrics[method_name] = {
                'ranking': ranking.tolist(),
                'ranking_names': [feature_names[i] for i in ranking],
                'hits_at_1': m1.hits_at_k,
                'hits_at_2': m2.hits_at_k,
                'mrr': m_full.mrr,
                'scores': None,
            }
            if method_name in diag_results.get('dist_fi', {}):
                result_obj = diag_results['dist_fi'][method_name]
                all_metrics[method_name]['scores'] = result_obj.rds_scores.tolist()

        if drift_type == 'recurring' and results_t1 is not None:
            rds_t1 = results_t1.get('rds')
            if rds_t1 and rds_t1.triggered and rds_t1.trigger_round is not None:
                trigger_round_t1 = rds_t1.trigger_round + 1
                print(f"\n  Running diagnosis at reversion round {trigger_round_t1} (t1) ...")
                diag_results_t1 = engine.run_diagnosis(trigger_round_t1,
                                                      client_weights=client_weights)
                rankings_t1 = engine.get_feature_rankings(diag_results_t1)
                all_metrics_t1 = {}
                for method_name, ranking in sorted(rankings_t1.items()):
                    m1 = compute_metrics(ranking, ground_truth, k=1)
                    m2 = compute_metrics(ranking, ground_truth, k=2)
                    m_full = compute_metrics(ranking, ground_truth, k=len(ground_truth))
                    all_metrics_t1[method_name] = {
                        'hits_at_1': m1.hits_at_k,
                        'hits_at_2': m2.hits_at_k,
                        'mrr': m_full.mrr,
                    }
                for method_name in all_metrics:
                    if method_name in all_metrics_t1:
                        all_metrics[method_name]['hits_at_1'] = (
                            all_metrics[method_name]['hits_at_1'] + all_metrics_t1[method_name]['hits_at_1']
                        ) / 2.0
                        all_metrics[method_name]['hits_at_2'] = (
                            all_metrics[method_name]['hits_at_2'] + all_metrics_t1[method_name]['hits_at_2']
                        ) / 2.0
                        all_metrics[method_name]['mrr'] = (
                            all_metrics[method_name]['mrr'] + all_metrics_t1[method_name]['mrr']
                        ) / 2.0
                print(f"  Averaged diagnosis metrics over t0 and t1.")

        with open(diag_dir / 'diagnosis_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Diagnosis saved to: {diag_dir}")

    timings_path = exp_dir / 'timings.csv'
    with open(timings_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k_name, v in sorted(timings.items()):
            writer.writerow([k_name, f"{v:.6f}"])
    print(f"  Timings saved to: {timings_path}")

    return trigger_data



def _restore_exp_cfg(meta_cfg: dict) -> dict:
    """Restore exp_cfg from metadata JSON (lists -> sets/tuples where needed)."""
    cfg = dict(meta_cfg)
    if 'drifted_features' in cfg and isinstance(cfg['drifted_features'], list):
        cfg['drifted_features'] = set(cfg['drifted_features'])
    if 't0_range' in cfg and isinstance(cfg['t0_range'], list):
        cfg['t0_range'] = tuple(cfg['t0_range'])
    return cfg


def run_diagnosis_only(exp_dir: Path) -> None:
    """
    Run diagnosis (SAGE/PFI/SHAP + metrics) on an existing run.
    Loads trigger_results.json and metadata.json from exp_dir, uses checkpoints
    from OUTPUT_DIR / exp_name / checkpoints (same layout as full run).
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = OUTPUT_DIR / exp_dir
    trigger_path = exp_dir / 'trigger_results.json'
    meta_path = exp_dir / 'metadata.json'
    if not trigger_path.exists():
        raise FileNotFoundError(f"Missing {trigger_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    with open(trigger_path) as f:
        trigger_data = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    exp_name = meta['exp_name']
    exp_cfg = _restore_exp_cfg(meta['exp_cfg'])
    dataset_name = exp_cfg['dataset']
    drift_type = meta['drift_type']
    seed = meta['seed']
    t0 = meta['t0']
    n_rounds = meta['n_rounds']
    n_clients = meta['n_clients']
    drifted_clients = set(meta['drifted_clients'])

    rds_info = trigger_data.get('detection_t0', {}).get('rds', {})
    if rds_info.get('triggered') and rds_info.get('round') is not None:
        trigger_round = rds_info['round']
    else:
        trigger_round = None
    if trigger_round is None:
        print("  RDS did not trigger — skipping diagnosis.")
        return

    seed_exp_name = exp_dir.name
    print(f"\nDiagnosis-only: {seed_exp_name}")
    print(f"  Trigger round: {trigger_round}, t0={t0}")
    print(f"  Checkpoints: {OUTPUT_DIR / seed_exp_name / 'checkpoints'}")

    dataset_config = DatasetConfig(
        name=dataset_name,
        n_features=exp_cfg['n_features'],
        n_samples_per_client=exp_cfg.get('n_samples_per_client', 500),
    )
    if dataset_name in ('fed_heart', 'wine'):
        dataset_config.drift_condition_feature = exp_cfg.get('drift_condition_feature', 'age')
        dataset_config.drift_condition_threshold = exp_cfg.get('drift_condition_threshold', None)
        dataset_config.drift_flip_prob = exp_cfg.get('drift_flip_prob', 0.3)
    if dataset_name in ('wine', 'diabetes', 'credit'):
        dataset_config.use_all_data_per_client = exp_cfg.get('use_all_data_per_client', True)
    if dataset_name == 'hyperplane':
        dataset_config.n_drift_features = exp_cfg.get('n_drift_features', 1)
    if dataset_name == 'agrawal':
        dataset_config.classification_function_pre = exp_cfg.get('classification_function_pre', 0)
        dataset_config.classification_function_post = exp_cfg.get('classification_function_post', 1)

    dummy_t0 = n_rounds + 1000
    config = ExperimentConfig(
        seed=seed,
        experiment_name=seed_exp_name,
        dataset=dataset_config,
        fl=FLConfig(
            n_clients=n_clients,
            n_rounds=n_rounds,
            hidden_sizes=exp_cfg.get('hidden_sizes', [128, 64]),
            learning_rate=exp_cfg.get('learning_rate', 0.05),
        ),
        drift=DriftConfig(
            t0=dummy_t0,
            drifted_client_proportion=exp_cfg.get('drifted_client_proportion', 0.5),
            drift_magnitude=exp_cfg.get('drift_magnitude', 1.0),
            drifted_features=exp_cfg.get('drifted_features', set()),
        ),
        trigger=TriggerConfig(),
        diagnosis=DiagnosisConfig(),
        metrics=MetricsConfig(
            k=len(exp_cfg.get('drifted_features', set())),
            use_ground_truth_k=True,
        ),
        base_output_dir=OUTPUT_DIR,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = FLTrainer(config)
    data_gen = trainer.data_generator
    if hasattr(data_gen, '_ensure_data'):
        data_gen._ensure_data()
    if hasattr(data_gen, '_ensure_pools'):
        data_gen._ensure_pools()

    def custom_get_round_data(round_num):
        schedule = compute_drift_schedule(drift_type, round_num, t0, exp_cfg)
        return get_round_data_custom(
            data_gen, round_num, drifted_clients, schedule, exp_cfg, dataset_name)

    trainer.get_round_data = custom_get_round_data

    ground_truth = exp_cfg.get('drifted_features', set())
    if isinstance(ground_truth, list):
        ground_truth = set(ground_truth)
    feature_names = trainer.get_feature_names()

    mechanism = exp_cfg.get('drift_mechanism', 'label_flip')
    _diag_get_round_data = trainer.get_round_data

    if mechanism in ('label_flip', 'concept_switch'):
        def _ref_get_round_data(round_num, _trigger=trigger_round, _inner=_diag_get_round_data):
            if round_num == _trigger:
                return _inner(round_num - 1)
            return _inner(round_num)
        trainer.get_round_data = _ref_get_round_data

    _rd = trainer.get_round_data(1)
    _csz = np.array([len(_rd[i].X_train) for i in range(n_clients)], dtype=np.float64)
    client_weights = _csz / _csz.sum()

    engine = DiagnosisEngine(config, trainer)
    print(f"  Running diagnosis at trigger round {trigger_round} ...")
    diag_results = engine.run_diagnosis(trigger_round,
                                        client_weights=client_weights)
    trainer.get_round_data = _diag_get_round_data

    rankings = engine.get_feature_rankings(diag_results)
    diag_dir = exp_dir / 'diagnosis'
    diag_dir.mkdir(parents=True, exist_ok=True)

    engine.save_results(diag_results, diag_dir)

    print(f"\n  Diagnosis results (ground truth drifted features: {ground_truth}):")
    print(f"  {'Method':<25s} {'Ranking (top 5)':<40s} {'Hits@1':>7s} {'Hits@2':>7s} {'MRR':>7s}")
    print(f"  {'-'*90}")

    all_metrics = {}
    for method_name, ranking in sorted(rankings.items()):
        m1 = compute_metrics(ranking, ground_truth, k=1)
        m2 = compute_metrics(ranking, ground_truth, k=2)
        m_full = compute_metrics(ranking, ground_truth, k=len(ground_truth))
        top5 = [f"{feature_names[i]}" for i in ranking[:5]]
        print(f"  {method_name:<25s} {str(top5):<40s} {m1.hits_at_k:>7.0f} {m2.hits_at_k:>7.0f} {m_full.mrr:>7.3f}")
        all_metrics[method_name] = {
            'ranking': ranking.tolist(),
            'ranking_names': [feature_names[i] for i in ranking],
            'hits_at_1': m1.hits_at_k,
            'hits_at_2': m2.hits_at_k,
            'mrr': m_full.mrr,
            'scores': None,
        }
        if method_name in diag_results.get('dist_fi', {}):
            result_obj = diag_results['dist_fi'][method_name]
            all_metrics[method_name]['scores'] = result_obj.rds_scores.tolist()

    with open(diag_dir / 'diagnosis_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Diagnosis saved to: {diag_dir}")



def main():
    parser = argparse.ArgumentParser(description='Drift-type experiments.')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='Run only these experiments (default: all)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single random seed (shorthand for --seeds <seed>)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='One or more seeds to run (e.g. --seeds 42 123 456 789 13)')
    parser.add_argument('--redetect', action='store_true',
                        help='Re-run detection only (no training)')
    parser.add_argument('--diagnose-only', type=str, default=None, metavar='DIR',
                        help='Run diagnosis only on existing run (e.g. wine_sudden_seed42)')
    parser.add_argument('--tiny', action='store_true',
                        help='Quick sanity check: override n_rounds=30, force t0=15')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.diagnose_only:
        try:
            run_diagnosis_only(args.diagnose_only)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        return

    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]

    exp_names = args.experiments if args.experiments else list(EXPERIMENTS.keys())
    print("Drift-Type Experiments")
    print(f"  Experiments: {exp_names}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {OUTPUT_DIR}")

    for seed in seeds:
        for name in exp_names:
            if name not in EXPERIMENTS:
                print(f"  WARNING: unknown experiment '{name}', skipping")
                continue
            exp_cfg = dict(EXPERIMENTS[name])
            if args.tiny:
                exp_cfg['n_rounds'] = 30
                exp_cfg['t0_range'] = (15, 15)
                exp_cfg['n_samples_per_client'] = exp_cfg.get('n_samples_per_client', 500)
                if exp_cfg['n_samples_per_client'] > 500:
                    exp_cfg['n_samples_per_client'] = 500
                if exp_cfg.get('recurring_period'):
                    exp_cfg['recurring_period'] = 10
                if exp_cfg.get('transition_window'):
                    exp_cfg['transition_window'] = 5
                exp_cfg['warmup_rounds'] = exp_cfg.get('warmup_rounds', 5)
                if exp_cfg['warmup_rounds'] > 5:
                    exp_cfg['warmup_rounds'] = 5
                exp_cfg['calibration_start'] = 6
                exp_cfg['calibration_end'] = 12
            try:
                run_single_experiment(name, exp_cfg, seed=seed)
            except Exception as e:
                print(f"  ERROR on {name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
