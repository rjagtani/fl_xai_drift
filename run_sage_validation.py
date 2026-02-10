"""
Federated SAGE Approximation Quality Experiment.

Evaluates how closely Federated SAGE (local + aggregate) approximates
Centralized SAGE (pooled data), comparing IID subsampling vs. compressed
(compress++ / K-means) summaries.

Per run (dataset/config/seed), the following are saved in the run directory:
  - global_model_final.pt: final trained global model (state_dict + architecture)
  - fi_scores.json: feature importance (SAGE) for each method (centralized,
    federated_iid, federated_compressed) with feature names
  - result.json: full metrics and SAGE vectors

Datasets (in CONFIGS, ready to run):
    hyperplane, wine, diabetes, credit, fed_heart
    (elec2, adult, agrawal, bank_marketing excluded to save time.)

Usage:
    python run_sage_validation.py                     # run all 5 datasets
    python run_sage_validation.py --dataset wine      # run one
    python run_sage_validation.py --summarize               # tables only
    python run_sage_validation.py --tiny                    # fast pipeline test
"""

from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
from typing import Dict, List, Set, Any
import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.cluster import KMeans

sys.path.insert(0, '.')

# -- Local imports --
from src.config import ExperimentConfig, DatasetConfig, FLConfig, DriftConfig
from src.models.mlp import MLP, ModelWrapper
from src.fl_trainer.trainer import FLTrainer
from src.data.base import ClientDataset

# SAGE
try:
    from sage import MarginalImputer, KernelEstimator
    SAGE_OK = True
except ImportError:
    SAGE_OK = False
    print("ERROR: sage-importance package required.  pip install sage-importance")

# Parallel client processing
from joblib import Parallel, delayed

# compress++ (goodpoints) -- optional, falls back to K-means
try:
    from goodpoints import compress as gp_compress
    COMPRESS_OK = True
except ImportError:
    COMPRESS_OK = False
    print("INFO: goodpoints not available; using K-means coreset as fallback.")

# Multivariate Wasserstein via POT (required for D_het)
try:
    import ot
    OT_OK = True
except ImportError:
    OT_OK = False
    print("WARNING: POT not found. Install with: pip install POT. D_het will use per-feature fallback.")

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
OUTPUT_DIR = Path('results') / 'sage_validation'
SEEDS = [42, 43, 44, 45, 46]
N_ROUNDS = 100

# -----------------------------------------------------------------
# Dataset x config definitions
# -----------------------------------------------------------------
# Rules: (1) No-drift (iid) run per dataset; (2) drifted_client_proportion = 0.5 for all drifted runs;
# (3) Drift type: magnitude (hyperplane), noise (wine, diabetes, credit, fed_heart on cp);
# (4) hyperplane, wine, diabetes, credit, fed_heart, agrawal (no elec2, adult, bank_marketing).
CONFIGS: Dict[str, List[dict]] = {
    # -- Hyperplane (synthetic): 10 clients x 1000 = 10,000 samples. Drift = magnitude change (River: [0, 1]). --
    'hyperplane': [
        {
            'tag': 'iid',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 5,
            'hidden_sizes': [128, 64],
            'drift_magnitude': 0.0,
            'drifted_client_proportion': 0.0,
        },
        {
            'tag': 'low',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 5,
            'hidden_sizes': [128, 64],
            'drift_magnitude': 0.25,
            'drifted_client_proportion': 0.5,
        },
        {
            'tag': 'mid',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 5,
            'hidden_sizes': [128, 64],
            'drift_magnitude': 0.5,
            'drifted_client_proportion': 0.5,
        },
        {
            'tag': 'high',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 5,
            'hidden_sizes': [128, 64],
            'drift_magnitude': 0.75,
            'drifted_client_proportion': 0.5,
        },
    ],

    # -- Wine Quality: noise on alcohol (index 10). Proportion fixed 0.5. --
    'wine': [
        {
            'tag': 'iid',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 11,
            'hidden_sizes': [64, 32],
            'noise_std': 0.0,
            'drifted_client_proportion': 0.0,
            'drifted_features': {10},
        },
        {
            'tag': 'low',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 11,
            'hidden_sizes': [64, 32],
            'noise_std': 2.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {10},
        },
        {
            'tag': 'mid',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 11,
            'hidden_sizes': [64, 32],
            'noise_std': 4.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {10},
        },
        {
            'tag': 'high',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 11,
            'hidden_sizes': [64, 32],
            'noise_std': 6.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {10},
        },
    ],

    # -- Diabetes: noise on Glucose (index 1). Proportion 0.5. --
    'diabetes': [
        {
            'tag': 'iid',
            'n_clients': 6, 'n_samples_per_client': 500, 'n_features': 8,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 0.0,
            'drifted_client_proportion': 0.0,
            'drifted_features': {1},
        },
        {
            'tag': 'low',
            'n_clients': 6, 'n_samples_per_client': 500, 'n_features': 8,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 2.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
        {
            'tag': 'mid',
            'n_clients': 6, 'n_samples_per_client': 500, 'n_features': 8,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 4.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
        {
            'tag': 'high',
            'n_clients': 6, 'n_samples_per_client': 500, 'n_features': 8,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 6.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
    ],

    # -- Credit: noise on duration (index 1). Proportion 0.5. --
    'credit': [
        {
            'tag': 'iid',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 20,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 0.0,
            'drifted_client_proportion': 0.0,
            'drifted_features': {1},
        },
        {
            'tag': 'low',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 20,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 2.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
        {
            'tag': 'mid',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 20,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 4.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
        {
            'tag': 'high',
            'n_clients': 5, 'n_samples_per_client': 500, 'n_features': 20,
            'hidden_sizes': [64, 32],
            'use_all_data_per_client': True,
            'noise_std': 6.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {1},
        },
    ],

    # -- Fed-Heart: noise on cp (chest pain, index 2). Proportion 0.5. No label flip. --
    'fed_heart': [
        {
            'tag': 'iid',
            'n_clients': 4, 'n_samples_per_client': 300, 'n_features': 13,
            'hidden_sizes': [64, 32],
            'noise_std': 0.0,
            'drifted_client_proportion': 0.0,
            'drifted_features': {2},   # cp
        },
        {
            'tag': 'low',
            'n_clients': 4, 'n_samples_per_client': 300, 'n_features': 13,
            'hidden_sizes': [64, 32],
            'noise_std': 2.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {2},
        },
        {
            'tag': 'mid',
            'n_clients': 4, 'n_samples_per_client': 300, 'n_features': 13,
            'hidden_sizes': [64, 32],
            'noise_std': 4.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {2},
        },
        {
            'tag': 'high',
            'n_clients': 4, 'n_samples_per_client': 300, 'n_features': 13,
            'hidden_sizes': [64, 32],
            'noise_std': 6.0,
            'drifted_client_proportion': 0.5,
            'drifted_features': {2},
        },
    ],

    # -- Agrawal (synthetic) -------------------------
    # Function 4 uses 4 features: age, salary, commission, hvalue
    # (fn 9 was too imbalanced, causing stratify failures in train_test_split)
    # Drift fn 4 -> fn 2: drifted features = sym_diff({2,0,1,6}, {2,3,0}) = {1,3,6}
    # Severity = drifted_client_proportion (0.1 low, 0.3 mid, 0.5 high).
    'agrawal': [
        {
            'tag': 'iid',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 9,
            'hidden_sizes': [128, 64],
            'drifted_client_proportion': 0.0,
            'classification_function_pre': 4,
            'classification_function_post': 2,
        },
        {
            'tag': 'low',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 9,
            'hidden_sizes': [128, 64],
            'drifted_client_proportion': 0.1,
            'classification_function_pre': 4,
            'classification_function_post': 2,
        },
        {
            'tag': 'mid',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 9,
            'hidden_sizes': [128, 64],
            'drifted_client_proportion': 0.3,
            'classification_function_pre': 4,
            'classification_function_post': 2,
        },
        {
            'tag': 'high',
            'n_clients': 10, 'n_samples_per_client': 1000, 'n_features': 9,
            'hidden_sizes': [128, 64],
            'drifted_client_proportion': 0.5,
            'classification_function_pre': 4,
            'classification_function_post': 2,
        },
    ],
}


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _make_config(dataset: str, cfg: dict, seed: int,
                 n_rounds_override: int = None) -> ExperimentConfig:
    """Build an ExperimentConfig for one (dataset, config, seed) tuple."""
    n_clients  = cfg['n_clients']
    n_features = cfg['n_features']
    hidden     = cfg['hidden_sizes']
    n_rounds   = n_rounds_override or N_ROUNDS

    ds = DatasetConfig(
        name=dataset,
        n_features=n_features,
        n_samples_per_client=cfg.get('n_samples_per_client', 500),
        classification_function_pre=cfg.get('classification_function_pre', 0),
        classification_function_post=cfg.get('classification_function_post', 2),
    )

    # Set real-world dataset-specific fields
    if dataset in ('wine', 'elec2'):
        ds.drift_condition_feature = {
            'wine': 'alcohol', 'elec2': 'nswprice',
        }.get(dataset, 'alcohol')
        ds.drift_condition_threshold = None  # auto median
        ds.drift_flip_prob = 0.3
    if dataset == 'wine':
        ds.use_all_data_per_client = False
    if dataset in ('adult', 'bank_marketing', 'diabetes', 'credit'):
        ds.use_all_data_per_client = cfg.get('use_all_data_per_client', True)
    if dataset == 'fed_heart':
        ds.drift_condition_feature = cfg.get('drift_condition_feature', 'chol')
        ds.drift_condition_threshold = None
        ds.drift_flip_prob = cfg.get('drift_flip_prob', 0.3)

    fl = FLConfig(
        n_clients=n_clients,
        n_rounds=n_rounds,
        hidden_sizes=hidden,
        n_classes=2,
        local_epochs=1,
        batch_size=32,
        learning_rate=cfg.get('learning_rate', 0.01),
    )

    # Non-IID from round 0: set t0=0 so trainer uses post-drift data immediately
    # For real-world datasets, heterogeneity is natural; feature noise adds more.
    prop = cfg.get('drifted_client_proportion', 0.0)
    drift = DriftConfig(
        t0=0 if prop > 0 else n_rounds + 999,   # drift from start or never
        drifted_client_proportion=prop,
        drift_magnitude=cfg.get('drift_magnitude', 0.0),
    )

    return ExperimentConfig(
        seed=seed,
        experiment_name=f"sage_val_{dataset}_{cfg['tag']}_s{seed}",
        dataset=ds,
        fl=fl,
        drift=drift,
        base_output_dir=OUTPUT_DIR,
    )


def _get_client_data(trainer: FLTrainer, cfg: dict, dataset: str, seed: int
                     ) -> Dict[int, ClientDataset]:
    """Generate client data respecting the non-IID config.

    Synthetic (hyperplane, agrawal): use built-in drift mechanisms.
    Real-world (wine, elec2): generate clean data, then add Gaussian noise
      to drifted features for the specified proportion of clients.
    """
    dg = trainer.data_generator
    n_clients = trainer.config.fl.n_clients
    prop = cfg.get('drifted_client_proportion', 0.0)
    n_drifted = int(round(prop * n_clients))
    drifted_clients = set(range(n_clients - n_drifted, n_clients))

    if dataset == 'hyperplane':
        mag = cfg.get('drift_magnitude', 0.0)
        return dg.generate_static_client_data(
            drifted_clients=drifted_clients,
            drift_magnitude=mag,
            generate_drifted=(mag > 0),
        )
    elif dataset == 'agrawal':
        # Force pre/post classification functions from this run's config
        # (avoids stale fn=9 or wrong defaults; fn 9 is too imbalanced for stratify)
        pre = cfg.get('classification_function_pre', 0)
        post = cfg.get('classification_function_post', 2)
        if hasattr(dg, 'classification_function_pre'):
            dg.classification_function_pre = pre
            dg.classification_function_post = post
            # Clear cached pools so they are regenerated with the correct functions
            if hasattr(dg, '_pre_pool') and dg._pre_pool is not None:
                dg._pre_pool = dg._pre_pool_y = dg._post_pool = dg._post_pool_y = None
            if hasattr(dg, '_drifted_feature_indices'):
                from src.data.agrawal import AgrawalDataGenerator
                pre_f = AgrawalDataGenerator.FUNCTION_RELEVANT_FEATURES.get(pre, set())
                post_f = AgrawalDataGenerator.FUNCTION_RELEVANT_FEATURES.get(post, set())
                dg._drifted_feature_indices = pre_f.symmetric_difference(post_f)
        if hasattr(dg, '_ensure_pools'):
            dg._ensure_pools()
        return dg.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=(prop > 0),
        )
    elif dataset in ('wine', 'elec2', 'adult', 'bank_marketing', 'diabetes', 'credit', 'fed_heart'):
        # Real-world (and fed_heart): generate clean data, then add Gaussian noise
        # on specified features for drifted clients. Fed_heart uses noise on cp (index 2).
        if hasattr(dg, '_ensure_data'):
            dg._ensure_data()
        client_data = dg.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=False,
        )
        noise_std = cfg.get('noise_std', 0.0)
        drift_feats = sorted(cfg.get('drifted_features', set()))
        if noise_std > 0 and drift_feats:
            rng = np.random.default_rng(seed + 7777)
            for cid in drifted_clients:
                if cid not in client_data:
                    continue
                cd = client_data[cid]
                for f in drift_feats:
                    cd.X_train[:, f] += rng.normal(
                        0, noise_std, len(cd.X_train)).astype(cd.X_train.dtype)
                    cd.X_val[:, f] += rng.normal(
                        0, noise_std, len(cd.X_val)).astype(cd.X_val.dtype)
        return client_data
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# -----------------------------------------------------------------
# Compression / Subsampling
# -----------------------------------------------------------------

def compresspp(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return indices of compress++ coreset.

    Uses goodpoints when available, otherwise K-means medoids.
    """
    n, d = X.shape
    sigma = np.sqrt(2.0 * d)

    if COMPRESS_OK:
        idx = gp_compress.compresspp_kt(
            X.astype(np.float64),
            kernel_type=b"gaussian",
            k_params=np.array([sigma ** 2]),
            g=4,
            seed=seed,
        )
        return idx

    # Fallback: K-means -> pick closest actual point to each centroid
    target = max(int(np.sqrt(n)), 10)
    km = KMeans(n_clusters=target, random_state=seed, n_init=3, max_iter=100)
    km.fit(X)
    from scipy.spatial.distance import cdist
    dists = cdist(km.cluster_centers_, X)
    idx = np.unique(dists.argmin(axis=1))
    return idx


def subsample(X: np.ndarray, n_target: int, seed: int = 42) -> np.ndarray:
    """Return IID random subsample indices."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_target = min(n_target, n)
    return rng.choice(n, size=n_target, replace=False)


# -----------------------------------------------------------------
# SAGE computation
# -----------------------------------------------------------------

def compute_sage(model_fn, X_bg, X_est, y_est, bar: bool = True):
    """Compute SAGE values using KernelEstimator (batch_size=512).

    Returns (values, elapsed_seconds).
    """
    X_bg  = np.asarray(X_bg, dtype=np.float32)
    X_est = np.asarray(X_est, dtype=np.float32)
    y_est = np.asarray(y_est, dtype=np.int64)
    imputer   = MarginalImputer(model_fn, X_bg)
    estimator = KernelEstimator(imputer, 'cross entropy')
    t0 = time.perf_counter()
    sv = estimator(X_est, y_est, batch_size=512, verbose=False, bar=bar)
    elapsed = time.perf_counter() - t0
    return sv.values, elapsed


# -----------------------------------------------------------------
# Top-k precision
# -----------------------------------------------------------------

TOP_K = 3  # precision @ top-k (correctly identifying the k most important features)

def topk_precision(values_pred: np.ndarray, values_true: np.ndarray,
                   k: int = TOP_K) -> float:
    """Precision of correctly identifying the top-k most important features.

    Ranks features by absolute SAGE value and returns the fraction of
    the true top-k features that appear in the predicted top-k.
    """
    k = min(k, len(values_true))
    top_true = set(np.argsort(np.abs(values_true))[-k:])
    top_pred = set(np.argsort(np.abs(values_pred))[-k:])
    return len(top_true & top_pred) / k


def mae_topk(sage_cent: np.ndarray, sage_fed: np.ndarray, k: int = TOP_K) -> float:
    """MAE restricted to the top-k features by |centralized SAGE|."""
    k = min(k, len(sage_cent))
    top_idx = np.argsort(np.abs(sage_cent))[-k:]
    return float(np.mean(np.abs(sage_cent[top_idx] - np.asarray(sage_fed)[top_idx])))


# -----------------------------------------------------------------
# Multivariate Wasserstein (heterogeneity distance)
# -----------------------------------------------------------------

def wasserstein_mv(X_a: np.ndarray, X_b: np.ndarray, max_n: int = 500) -> float:
    """Compute multivariate Wasserstein-1 (Earth Mover's distance) between two point clouds using POT."""
    rng = np.random.default_rng(0)
    if len(X_a) > max_n:
        X_a = X_a[rng.choice(len(X_a), max_n, replace=False)]
    if len(X_b) > max_n:
        X_b = X_b[rng.choice(len(X_b), max_n, replace=False)]

    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)

    if OT_OK:
        na, nb = len(X_a), len(X_b)
        a = np.ones(na) / na
        b = np.ones(nb) / nb
        M = ot.dist(X_a, X_b, metric='euclidean')
        return float(ot.emd2(a, b, M))

    # Fallback only if POT not installed (per-feature W1 average; scale not comparable to joint)
    d = X_a.shape[1]
    return float(np.mean([
        wasserstein_distance(X_a[:, j], X_b[:, j]) for j in range(d)
    ]))


# ===============================================================
# Per-client SAGE helper (for joblib parallelism)
# ===============================================================

def _process_client(cid, cd, weight, model_fn, seed, d,
                    sage_cent=None, bar=True):
    """Compute 3 federated SAGE variants for one client.

    Variants:
        comp     - compressed bg (train coreset) + compressed est (val coreset)
        comp_bg  - compressed bg (train coreset) + full client validation
        iid      - IID sampled bg (4x comp bg size) + full client validation

    Args:
        sage_cent: centralized SAGE values (if provided, per-client MAE is printed).

    Returns a dict with all per-client results ready for aggregation.
    """
    X_tr = cd.X_train
    X_va = cd.X_val
    y_va = cd.y_val

    # -- Compression step (shared by comp and comp_bg) --
    t_comp_start = time.perf_counter()
    idx_comp_tr = compresspp(X_tr, seed=seed + cid)
    idx_comp_va = compresspp(X_va, seed=seed + cid + 5000)
    t_compression = time.perf_counter() - t_comp_start

    X_bg_comp = X_tr[idx_comp_tr].astype(np.float32)

    # --- Method 1: comp (compressed bg + compressed est) ---
    X_est_comp = X_va[idx_comp_va].astype(np.float32)
    y_est_comp = y_va[idx_comp_va]
    sage_comp_i, t_sage_comp = compute_sage(
        model_fn, X_bg_comp, X_est_comp, y_est_comp, bar=bar)

    # --- Method 2: comp_bg (compressed bg + full validation) ---
    X_est_full = X_va.astype(np.float32)
    y_est_full = y_va
    sage_compbg_i, t_sage_compbg = compute_sage(
        model_fn, X_bg_comp, X_est_full, y_est_full, bar=bar)

    # --- Method 3: iid (IID bg 4x comp bg + full validation) ---
    n_bg_iid = min(4 * len(idx_comp_tr), len(X_tr))
    idx_iid_bg = subsample(X_tr, n_bg_iid, seed=seed + cid + 1000)
    X_bg_iid = X_tr[idx_iid_bg].astype(np.float32)
    sage_iid_i, t_sage_iid = compute_sage(
        model_fn, X_bg_iid, X_est_full, y_est_full, bar=bar)

    # Per-client MAE
    mae_str = ""
    if sage_cent is not None:
        mae_c = float(np.mean(np.abs(sage_comp_i   - sage_cent)))
        mae_b = float(np.mean(np.abs(sage_compbg_i - sage_cent)))
        mae_i = float(np.mean(np.abs(sage_iid_i    - sage_cent)))
        mae_str = (f"  MAE comp={mae_c:.5f} comp_bg={mae_b:.5f} "
                   f"iid={mae_i:.5f}")

    print(f"    Client {cid}: "
          f"comp={len(idx_comp_tr)}bg/{len(idx_comp_va)}est ({t_sage_comp + t_compression:.1f}s)  "
          f"comp_bg={len(idx_comp_tr)}bg/{len(X_va)}est ({t_sage_compbg + t_compression:.1f}s)  "
          f"iid={n_bg_iid}bg/{len(X_va)}est ({t_sage_iid:.1f}s)"
          f"{mae_str}")

    return {
        'cid': cid,
        'weight': weight,
        # SAGE values
        'sage_comp': sage_comp_i,
        'sage_comp_bg': sage_compbg_i,
        'sage_iid': sage_iid_i,
        # Timings (comp and comp_bg include compression time)
        'time_comp': t_sage_comp + t_compression,
        'time_comp_bg': t_sage_compbg + t_compression,
        'time_iid': t_sage_iid,
        'compression_time': t_compression,
        # Sizes
        'bg_size_comp': len(idx_comp_tr),
        'est_size_comp': len(idx_comp_va),
        'bg_size_comp_bg': len(idx_comp_tr),
        'est_size_comp_bg': len(X_va),
        'bg_size_iid': n_bg_iid,
        'est_size_iid': len(X_va),
    }


# ===============================================================
# Single run
# ===============================================================

def run_single(dataset: str, cfg: dict, seed: int,
               n_rounds_override: int = None,
               n_client_jobs: int = 1) -> dict:
    tag = cfg['tag']
    n_rounds = n_rounds_override or N_ROUNDS
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} | {tag} | seed={seed} | rounds={n_rounds}")
    if n_client_jobs > 1:
        print(f"  clients_parallel={n_client_jobs}")
    print(f"{'='*60}")

    # -- Step 0: build config, trainer, generate data --
    exp_cfg = _make_config(dataset, cfg, seed, n_rounds_override=n_rounds)
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = FLTrainer(exp_cfg)

    # Create directories so checkpoints can be saved during training
    exp_cfg.create_directories()

    client_data = _get_client_data(trainer, cfg, dataset, seed)

    # Monkey-patch get_round_data so the trainer always serves this static data
    trainer.get_round_data = lambda r, _cd=client_data: _cd

    # -- Step 1: FL Training --
    print(f"  Training {n_rounds} FL rounds ...")
    t_fl_start = time.perf_counter()
    for rnd in range(1, n_rounds + 1):
        trainer.train_round(rnd)
    time_fl_training_seconds = time.perf_counter() - t_fl_start
    global_model = trainer.global_model
    wrapper = ModelWrapper(global_model, device=trainer.device)
    model_fn = wrapper.predict_proba
    print(f"  Training done ({time_fl_training_seconds:.1f}s).  Final loss ~ {trainer.global_loss_series[-1]:.4f}")

    # -- Pool all data --
    all_X_train = np.concatenate([cd.X_train for cd in client_data.values()])
    all_y_train = np.concatenate([cd.y_train for cd in client_data.values()])
    all_X_val   = np.concatenate([cd.X_val   for cd in client_data.values()])
    all_y_val   = np.concatenate([cd.y_val   for cd in client_data.values()])
    d = all_X_train.shape[1]

    client_sizes = {cid: len(cd.X_train) for cid, cd in client_data.items()}
    total_n = sum(client_sizes.values())
    weights = {cid: n / total_n for cid, n in client_sizes.items()}

    # -- Step 2: Centralized SAGE (256 IID bg, full pooled validation) --
    print("  Computing Centralized SAGE ...")
    bg_cent_size = min(256, len(all_X_train))
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(all_X_train), bg_cent_size, replace=False)
    X_bg_cent = all_X_train[bg_idx].astype(np.float32)
    X_est_cent = all_X_val.astype(np.float32)
    y_est_cent = all_y_val

    sage_cent, time_cent = compute_sage(model_fn, X_bg_cent, X_est_cent, y_est_cent)
    print(f"    Centralized SAGE done ({time_cent:.1f}s)  "
          f"bg={len(X_bg_cent)} est={len(X_est_cent)}")

    # -- Step 3: Federated SAGE (3 variants: comp, comp_bg, iid) --
    if n_client_jobs == 1:
        print("  Computing Federated SAGE (sequential) ...")
        client_results = []
        for cid, cd in client_data.items():
            res = _process_client(cid, cd, weights[cid], model_fn, seed, d,
                                  sage_cent=sage_cent, bar=True)
            client_results.append(res)
    else:
        effective_jobs = min(n_client_jobs, len(client_data))
        print(f"  Computing Federated SAGE in parallel ({effective_jobs} workers) ...")
        client_results = Parallel(n_jobs=effective_jobs, prefer='threads')(
            delayed(_process_client)(
                cid, cd, weights[cid], model_fn, seed, d,
                sage_cent=sage_cent, bar=False)
            for cid, cd in client_data.items()
        )

    # Aggregate client results for all 3 methods
    fed_sage_comp    = np.zeros(d)
    fed_sage_comp_bg = np.zeros(d)
    fed_sage_iid     = np.zeros(d)
    ct_comp, ct_comp_bg, ct_iid, ct_compression = [], [], [], []
    sz_bg_comp, sz_est_comp = [], []
    sz_bg_comp_bg, sz_est_comp_bg = [], []
    sz_bg_iid, sz_est_iid = [], []

    for res in client_results:
        w_i = res['weight']
        fed_sage_comp    += w_i * res['sage_comp']
        fed_sage_comp_bg += w_i * res['sage_comp_bg']
        fed_sage_iid     += w_i * res['sage_iid']
        ct_comp.append(res['time_comp'])
        ct_comp_bg.append(res['time_comp_bg'])
        ct_iid.append(res['time_iid'])
        ct_compression.append(res['compression_time'])
        sz_bg_comp.append(res['bg_size_comp'])
        sz_est_comp.append(res['est_size_comp'])
        sz_bg_comp_bg.append(res['bg_size_comp_bg'])
        sz_est_comp_bg.append(res['est_size_comp_bg'])
        sz_bg_iid.append(res['bg_size_iid'])
        sz_est_iid.append(res['est_size_iid'])

    # -- Step 4: Metrics for all 3 methods --
    def _safe_rho(a, b):
        r, _ = spearmanr(a, b)
        return float(r) if not np.isnan(r) else 0.0

    mae_comp    = float(np.mean(np.abs(fed_sage_comp    - sage_cent)))
    mae_comp_bg = float(np.mean(np.abs(fed_sage_comp_bg - sage_cent)))
    mae_iid     = float(np.mean(np.abs(fed_sage_iid     - sage_cent)))

    mae_top3_comp    = mae_topk(sage_cent, fed_sage_comp)
    mae_top3_comp_bg = mae_topk(sage_cent, fed_sage_comp_bg)
    mae_top3_iid     = mae_topk(sage_cent, fed_sage_iid)

    rho_comp    = _safe_rho(fed_sage_comp,    sage_cent)
    rho_comp_bg = _safe_rho(fed_sage_comp_bg, sage_cent)
    rho_iid     = _safe_rho(fed_sage_iid,     sage_cent)

    topk_comp    = topk_precision(fed_sage_comp,    sage_cent)
    topk_comp_bg = topk_precision(fed_sage_comp_bg, sage_cent)
    topk_iid     = topk_precision(fed_sage_iid,     sage_cent)

    cent_mag = float(np.mean(np.abs(sage_cent)))
    eff_comp    = mae_comp    / cent_mag if cent_mag > 1e-12 else float('nan')
    eff_comp_bg = mae_comp_bg / cent_mag if cent_mag > 1e-12 else float('nan')
    eff_iid     = mae_iid     / cent_mag if cent_mag > 1e-12 else float('nan')

    # Timing stats per method
    def _tstats(times):
        return {
            'max': float(max(times)), 'mean': float(np.mean(times)),
            'median': float(np.median(times)), 'total': float(sum(times)),
        }
    ts_comp    = _tstats(ct_comp)
    ts_comp_bg = _tstats(ct_comp_bg)
    ts_iid     = _tstats(ct_iid)
    max_compression_time = float(max(ct_compression))

    # -- Heterogeneity distance (multivariate W1 via POT when available) --
    # D_het = sum_i w_i * W_1(X_train^i, X_pooled). Even for IID data, finite-sample
    # W1 between two 500-pt samples in 9D is non-zero (~1-2 scale); compare across configs.
    print("  Computing heterogeneity distance (POT multivariate W1) ..." if OT_OK else "  Computing heterogeneity distance (per-feature fallback) ...")
    per_client_dists = []
    for cid, cd in client_data.items():
        w_dist = wasserstein_mv(cd.X_train, all_X_train, max_n=500)
        per_client_dists.append(float(weights[cid] * w_dist))
    het_distance = float(sum(per_client_dists))

    print(f"\n  Results:")
    print(f"    MAE     comp={mae_comp:.6f}  comp_bg={mae_comp_bg:.6f}  iid={mae_iid:.6f}")
    print(f"    rho     comp={rho_comp:.4f}  comp_bg={rho_comp_bg:.4f}  iid={rho_iid:.4f}")
    print(f"    Top-{TOP_K}   comp={topk_comp:.4f}  comp_bg={topk_comp_bg:.4f}  iid={topk_iid:.4f}")
    print(f"    Time    comp   max={ts_comp['max']:.2f}s  mean={ts_comp['mean']:.2f}s  median={ts_comp['median']:.2f}s")
    print(f"            comp_bg max={ts_comp_bg['max']:.2f}s  mean={ts_comp_bg['mean']:.2f}s  median={ts_comp_bg['median']:.2f}s")
    print(f"            iid    max={ts_iid['max']:.2f}s  mean={ts_iid['mean']:.2f}s  median={ts_iid['median']:.2f}s")
    print(f"    D_het = {het_distance:.6f}")

    # -- Save --
    result = {
        'dataset': dataset,
        'config': tag,
        'seed': seed,
        'sage_centralized': sage_cent.tolist(),
        'sage_federated_comp': fed_sage_comp.tolist(),
        'sage_federated_comp_bg': fed_sage_comp_bg.tolist(),
        'sage_federated_iid': fed_sage_iid.tolist(),
        # Accuracy metrics
        'mae_comp': mae_comp,
        'mae_comp_bg': mae_comp_bg,
        'mae_iid': mae_iid,
        'mae_top3_comp': mae_top3_comp,
        'mae_top3_comp_bg': mae_top3_comp_bg,
        'mae_top3_iid': mae_top3_iid,
        'spearman_comp': rho_comp,
        'spearman_comp_bg': rho_comp_bg,
        'spearman_iid': rho_iid,
        'topk_comp': float(topk_comp),
        'topk_comp_bg': float(topk_comp_bg),
        'topk_iid': float(topk_iid),
        'efficiency_residual_comp': eff_comp,
        'efficiency_residual_comp_bg': eff_comp_bg,
        'efficiency_residual_iid': eff_iid,
        # Timing
        'max_time_comp_seconds': ts_comp['max'],
        'mean_time_comp_seconds': ts_comp['mean'],
        'median_time_comp_seconds': ts_comp['median'],
        'max_time_comp_bg_seconds': ts_comp_bg['max'],
        'mean_time_comp_bg_seconds': ts_comp_bg['mean'],
        'median_time_comp_bg_seconds': ts_comp_bg['median'],
        'max_time_iid_seconds': ts_iid['max'],
        'mean_time_iid_seconds': ts_iid['mean'],
        'median_time_iid_seconds': ts_iid['median'],
        'max_compression_time_seconds': max_compression_time,
        'time_centralized_seconds': time_cent,
        'time_fl_training_seconds': time_fl_training_seconds,
        # Sizes
        'n_clients': cfg['n_clients'],
        'samples_per_client': cfg.get('n_samples_per_client'),
        'bg_size_centralized': int(len(X_bg_cent)),
        'est_size_centralized': int(len(X_est_cent)),
        'bg_size_comp': int(np.mean(sz_bg_comp)),
        'est_size_comp': int(np.mean(sz_est_comp)),
        'bg_size_comp_bg': int(np.mean(sz_bg_comp_bg)),
        'est_size_comp_bg': int(np.mean(sz_est_comp_bg)),
        'bg_size_iid': int(np.mean(sz_bg_iid)),
        'est_size_iid': int(np.mean(sz_est_iid)),
        'het_distance': het_distance,
        'per_client_distances': per_client_dists,
        'n_client_jobs': n_client_jobs,
    }

    out_dir = OUTPUT_DIR / f"{dataset}_{tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save final global model
    model_path = out_dir / 'global_model_final.pt'
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'input_size': d,
        'hidden_sizes': cfg['hidden_sizes'],
        'n_classes': 2,
    }, model_path)
    print(f"  Saved model -> {model_path}")

    # Save FI scores (SAGE values) per method
    try:
        feature_names = list(exp_cfg.dataset.feature_names)
    except Exception:
        feature_names = [f'f{j}' for j in range(d)]
    if len(feature_names) != d:
        feature_names = [f'f{j}' for j in range(d)]
    fi_scores = {
        'feature_names': feature_names,
        'centralized': sage_cent.tolist(),
        'federated_comp': fed_sage_comp.tolist(),
        'federated_comp_bg': fed_sage_comp_bg.tolist(),
        'federated_iid': fed_sage_iid.tolist(),
    }
    with open(out_dir / 'fi_scores.json', 'w') as f:
        json.dump(fi_scores, f, indent=2)
    print(f"  Saved FI scores -> {out_dir / 'fi_scores.json'}")

    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_dir / 'result.json'}")

    return result


# ===============================================================
# Summary tables
# ===============================================================

def generate_summary():
    """Aggregate per-seed JSONs into summary CSV and LaTeX."""
    # Keys to aggregate (must match result.json fields)
    METHODS = ['comp', 'comp_bg', 'iid']
    agg_keys = []
    for m in METHODS:
        agg_keys += [f'mae_{m}', f'mae_top3_{m}', f'spearman_{m}', f'topk_{m}',
                     f'max_time_{m}_seconds', f'mean_time_{m}_seconds',
                     f'median_time_{m}_seconds',
                     f'bg_size_{m}', f'est_size_{m}']
    agg_keys += ['max_compression_time_seconds', 'time_centralized_seconds',
                 'time_fl_training_seconds', 'het_distance',
                 'bg_size_centralized', 'est_size_centralized']

    rows = []
    for ds, cfgs in CONFIGS.items():
        for cfg in cfgs:
            tag = cfg['tag']
            vals = {k: [] for k in agg_keys}
            for seed in SEEDS:
                p = OUTPUT_DIR / f"{ds}_{tag}_seed{seed}" / 'result.json'
                if not p.exists():
                    continue
                with open(p) as f:
                    r = json.load(f)
                # Compute MAE@top3 from vectors if not already in result (backfill for old runs)
                if 'mae_top3_comp' not in r and 'sage_centralized' in r:
                    cent = np.array(r['sage_centralized'])
                    for m in METHODS:
                        key = f'mae_top3_{m}'
                        fed_key = f'sage_federated_{m}'
                        if fed_key in r:
                            r[key] = mae_topk(cent, np.array(r[fed_key]))
                for k in vals:
                    if k in r:
                        vals[k].append(r[k])
            n = len(vals['mae_iid'])
            if n == 0:
                continue
            row = {'Dataset': ds.capitalize(), 'Config': tag, 'N': n}
            for k in vals:
                arr = np.array(vals[k]) if vals[k] else np.array([float('nan')])
                row[k + '_mean'] = float(np.nanmean(arr))
                row[k + '_std']  = float(np.nanstd(arr))
            rows.append(row)

    if not rows:
        print("No results found.")
        return

    # -- CSV --
    summary_dir = OUTPUT_DIR / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / 'sage_validation_summary.csv'
    cols = ['Dataset', 'Config']
    for m in METHODS:
        cols += [f'mae_{m}_mean', f'mae_{m}_std', f'mae_top3_{m}_mean', f'mae_top3_{m}_std',
                 f'spearman_{m}_mean', f'spearman_{m}_std',
                 f'topk_{m}_mean', f'topk_{m}_std',
                 f'max_time_{m}_seconds_mean', f'max_time_{m}_seconds_std',
                 f'mean_time_{m}_seconds_mean',
                 f'median_time_{m}_seconds_mean',
                 f'bg_size_{m}_mean', f'est_size_{m}_mean']
    cols += ['max_compression_time_seconds_mean',
             'time_centralized_seconds_mean',
             'time_fl_training_seconds_mean', 'time_fl_training_seconds_std',
             'het_distance_mean', 'het_distance_std',
             'bg_size_centralized_mean', 'est_size_centralized_mean', 'N']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"  CSV -> {csv_path}")

    # -- Console: main metrics table --
    print(f"\n{'Dataset':<15s} {'Cfg':<5s} "
          f"{'MAE(comp)':>12s} {'MAE(cbg)':>12s} {'MAE(iid)':>12s} "
          f"{'rho(comp)':>10s} {'rho(cbg)':>10s} {'rho(iid)':>10s} "
          f"{'Tk(comp)':>10s} {'Tk(cbg)':>10s} {'Tk(iid)':>10s} "
          f"{'D_het':>10s}")
    print('-' * 140)
    for r in rows:
        def _f(key, fmt='.4f'):
            m = r.get(key + '_mean', 0)
            s = r.get(key + '_std', 0)
            return f"{m:{fmt}}+/-{s:{fmt}}"
        print(f"{r['Dataset']:<15s} {r['Config']:<5s} "
              f"{_f('mae_comp', '.5f'):>12s} {_f('mae_comp_bg', '.5f'):>12s} {_f('mae_iid', '.5f'):>12s} "
              f"{_f('spearman_comp', '.3f'):>10s} {_f('spearman_comp_bg', '.3f'):>10s} {_f('spearman_iid', '.3f'):>10s} "
              f"{_f('topk_comp'):>10s} {_f('topk_comp_bg'):>10s} {_f('topk_iid'):>10s} "
              f"{_f('het_distance', '.5f'):>10s}")

    # -- Console: sample sizes --
    print(f"\n  Sample sizes (bg / est):")
    print(f"  {'Dataset':<15s} {'Cfg':<5s}  {'Cent':>10s}  {'Comp':>10s}  {'Comp_bg':>10s}  {'IID':>10s}")
    print("  " + "-" * 70)
    for r in rows:
        def _s(bg_key, est_key):
            bg = int(r.get(bg_key + '_mean', 0) or 0)
            est = int(r.get(est_key + '_mean', 0) or 0)
            return f"{bg}/{est}"
        print(f"  {r['Dataset']:<15s} {r['Config']:<5s}  "
              f"{_s('bg_size_centralized', 'est_size_centralized'):>10s}  "
              f"{_s('bg_size_comp', 'est_size_comp'):>10s}  "
              f"{_s('bg_size_comp_bg', 'est_size_comp_bg'):>10s}  "
              f"{_s('bg_size_iid', 'est_size_iid'):>10s}")

    # -- Console: timing --
    print(f"\n  Per-client SAGE timing (max/mean/median seconds, averaged across seeds):")
    print(f"  {'Dataset':<15s} {'Cfg':<5s}  "
          f"{'Comp max':>9s} {'mean':>6s} {'med':>6s}  "
          f"{'CBg max':>8s} {'mean':>6s} {'med':>6s}  "
          f"{'IID max':>8s} {'mean':>6s} {'med':>6s}")
    print("  " + "-" * 90)
    for r in rows:
        def _t(key):
            return r.get(key + '_mean', 0) or 0
        print(f"  {r['Dataset']:<15s} {r['Config']:<5s}  "
              f"{_t('max_time_comp_seconds'):>9.2f} {_t('mean_time_comp_seconds'):>6.2f} {_t('median_time_comp_seconds'):>6.2f}  "
              f"{_t('max_time_comp_bg_seconds'):>8.2f} {_t('mean_time_comp_bg_seconds'):>6.2f} {_t('median_time_comp_bg_seconds'):>6.2f}  "
              f"{_t('max_time_iid_seconds'):>8.2f} {_t('mean_time_iid_seconds'):>6.2f} {_t('median_time_iid_seconds'):>6.2f}")

    # -- LaTeX: publication table (Dataset, Config | MAE | rho | Max runtime; std in parens; best bold) --
    METHODS_ORDER = ['iid', 'comp_bg', 'comp']  # IID, Comp bg, Comp fg
    tex_path = summary_dir / 'sage_validation_table.tex'
    with open(tex_path, 'w') as f:
        f.write(r'\begin{table}[t]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Federated SAGE approximation: MAE vs.\ centralized, Spearman $\rho$, and max runtime per client (s). '
                r'IID: IID-sampled bg + full est; Comp\,bg: compressed bg + full est; Comp: compressed bg + compressed est. '
                r'Std in parentheses; best per row in bold. Mean over 5 seeds.}' + '\n')
        f.write(r'\label{tab:sage_validation}' + '\n')
        f.write(r'\small' + '\n')
        f.write(r'\begin{tabular}{@{}ll ccc ccc ccc@{}}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r' & & \multicolumn{3}{c}{MAE} & \multicolumn{3}{c}{$\rho$} & \multicolumn{3}{c}{Max time (s)} \\' + '\n')
        f.write(r'\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}' + '\n')
        f.write(r'Dataset & Cfg & IID & Comp\,bg & Comp & IID & Comp\,bg & Comp & IID & Comp\,bg & Comp \\' + '\n')
        f.write(r'\midrule' + '\n')
        prev_ds = None
        for r in rows:
            ds = r['Dataset']
            cfg = r['Config']
            ds_cell = ds if prev_ds != ds else ''
            prev_ds = ds

            def _cell(key, fmt='.4f', best_is_min=True):
                m = r.get(key + '_mean', 0) or 0
                s = r.get(key + '_std', 0) or 0
                return m, s, fmt

            # MAE: full MAE for all datasets. Best = min.
            mae_vals = [(r.get(f'mae_{m}_mean') or 0, r.get(f'mae_{m}_std') or 0) for m in METHODS_ORDER]
            best_mae = min(range(3), key=lambda i: mae_vals[i][0])
            # rho: best = max
            rho_vals = [(r.get(f'spearman_{m}_mean') or 0, r.get(f'spearman_{m}_std') or 0) for m in METHODS_ORDER]
            best_rho = max(range(3), key=lambda i: rho_vals[i][0])
            # max time: best = min
            time_vals = [(r.get(f'max_time_{m}_seconds_mean') or 0, r.get(f'max_time_{m}_seconds_std') or 0) for m in METHODS_ORDER]
            best_time = min(range(3), key=lambda i: time_vals[i][0])

            def fmt_val(val, std, decimals=4):
                if decimals == 3:
                    return f'{val:.3f} ({std:.3f})'
                if decimals == 2:
                    return f'{val:.2f} ({std:.2f})'
                return f'{val:.4f} ({std:.4f})'

            cells = [ds_cell, cfg]
            for i in range(3):
                v, s = mae_vals[i]
                cell = fmt_val(v, s, 4)
                if i == best_mae:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
            for i in range(3):
                v, s = rho_vals[i]
                cell = fmt_val(v, s, 3)
                if i == best_rho:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
            for i in range(3):
                v, s = time_vals[i]
                cell = fmt_val(v, s, 2)
                if i == best_time:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
            f.write(' & '.join(cells) + r' \\' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"  LaTeX -> {tex_path}")

    # -- LaTeX: IID-only publication table (one row per dataset; no Config column) --
    rows_iid = [r for r in rows if r['Config'] == 'iid']
    if rows_iid:
        tex_iid_path = summary_dir / 'sage_validation_table_iid.tex'
        with open(tex_iid_path, 'w') as f:
            f.write(r'\begin{table}[t]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Federated SAGE approximation (IID setting): MAE vs.\ centralized, Spearman $\rho$, and max runtime per client (s). '
                    r'Methods: IID (IID-sampled bg + full est), Comp\,bg (compressed bg + full est), Comp (compressed bg + compressed est). '
                    r'Std in parentheses; best per row in bold. Mean over 5 seeds.}' + '\n')
            f.write(r'\label{tab:sage_validation_iid}' + '\n')
            f.write(r'\small' + '\n')
            f.write(r'\begin{tabular}{@{}l ccc ccc ccc@{}}' + '\n')
            f.write(r'\toprule' + '\n')
            f.write(r' & \multicolumn{3}{c}{MAE} & \multicolumn{3}{c}{$\rho$} & \multicolumn{3}{c}{Max time (s)} \\' + '\n')
            f.write(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}' + '\n')
            f.write(r'Dataset & IID & Comp\,bg & Comp & IID & Comp\,bg & Comp & IID & Comp\,bg & Comp \\' + '\n')
            f.write(r'\midrule' + '\n')
            for r in rows_iid:
                ds = r['Dataset']
                mae_vals = [(r.get(f'mae_{m}_mean') or 0, r.get(f'mae_{m}_std') or 0) for m in METHODS_ORDER]
                best_mae = min(range(3), key=lambda i: mae_vals[i][0])
                rho_vals = [(r.get(f'spearman_{m}_mean') or 0, r.get(f'spearman_{m}_std') or 0) for m in METHODS_ORDER]
                best_rho = max(range(3), key=lambda i: rho_vals[i][0])
                time_vals = [(r.get(f'max_time_{m}_seconds_mean') or 0, r.get(f'max_time_{m}_seconds_std') or 0) for m in METHODS_ORDER]
                best_time = min(range(3), key=lambda i: time_vals[i][0])

                def fmt_val(val, std, decimals=4):
                    if decimals == 3:
                        return f'{val:.3f} ({std:.3f})'
                    if decimals == 2:
                        return f'{val:.2f} ({std:.2f})'
                    return f'{val:.4f} ({std:.4f})'

                cells = [ds]
                for i in range(3):
                    v, s = mae_vals[i]
                    cell = fmt_val(v, s, 4)
                    if i == best_mae:
                        cell = r'\textbf{' + cell + '}'
                    cells.append(cell)
                for i in range(3):
                    v, s = rho_vals[i]
                    cell = fmt_val(v, s, 3)
                    if i == best_rho:
                        cell = r'\textbf{' + cell + '}'
                    cells.append(cell)
                for i in range(3):
                    v, s = time_vals[i]
                    cell = fmt_val(v, s, 2)
                    if i == best_time:
                        cell = r'\textbf{' + cell + '}'
                    cells.append(cell)
                f.write(' & '.join(cells) + r' \\' + '\n')
            f.write(r'\bottomrule' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        print(f"  LaTeX (IID only) -> {tex_iid_path}")

        # IID-only summary CSV for plotting
        csv_iid_path = summary_dir / 'sage_validation_summary_iid.csv'
        with open(csv_iid_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows_iid:
                f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f"  CSV (IID only) -> {csv_iid_path}")


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description='Federated SAGE Validation')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run only this dataset (hyperplane, wine, diabetes, credit, fed_heart)')
    parser.add_argument('--config', type=str, default=None,
                        help='Run only this config tag (iid, low, mid, high)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed')
    parser.add_argument('--summarize', action='store_true',
                        help='Only generate summary from existing results')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip runs whose result.json already exists')
    parser.add_argument('--tiny', action='store_true',
                        help='Fast pipeline test: 5 FL rounds, 1 seed, 1 config per dataset')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers for per-client SAGE '
                             '(default: 1 = sequential)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Log parallelism settings
    if args.n_jobs != 1:
        print(f"Parallelism: n_client_jobs={args.n_jobs}")

    if args.summarize:
        generate_summary()
        return

    # -- Tiny mode: quick end-to-end pipeline test --
    if args.tiny:
        tiny_seeds = [42]
        tiny_datasets = [args.dataset] if args.dataset else list(CONFIGS.keys())
        print(f"*** TINY MODE: 5 rounds, 1 seed, 1 config per dataset ***")
        print(f"    Datasets: {tiny_datasets}\n")
        for ds in tiny_datasets:
            cfg = CONFIGS[ds][0]
            if args.config and cfg['tag'] != args.config:
                cfg = next((c for c in CONFIGS[ds] if c['tag'] == args.config), cfg)
            for seed in tiny_seeds:
                try:
                    run_single(ds, cfg, seed, n_rounds_override=5,
                               n_client_jobs=args.n_jobs)
                except Exception as e:
                    print(f"  ERROR in {ds}/{cfg['tag']}/s{seed}: {e}")
                    import traceback; traceback.print_exc()
        print("\n" + "=" * 60)
        print("Generating summary ...")
        print("=" * 60)
        generate_summary()
        return

    datasets = [args.dataset] if args.dataset else list(CONFIGS.keys())
    seeds    = [args.seed]    if args.seed    else SEEDS

    for ds in datasets:
        for cfg in CONFIGS[ds]:
            if args.config and cfg['tag'] != args.config:
                continue
            for seed in seeds:
                out_dir = OUTPUT_DIR / f"{ds}_{cfg['tag']}_seed{seed}"
                if args.skip_existing and (out_dir / 'result.json').exists():
                    print(f"  Skipping {ds}/{cfg['tag']}/s{seed} (exists)")
                    continue
                try:
                    run_single(ds, cfg, seed,
                               n_client_jobs=args.n_jobs)
                except Exception as e:
                    print(f"  ERROR in {ds}/{cfg['tag']}/s{seed}: {e}")
                    import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("Generating summary ...")
    print("=" * 60)
    generate_summary()


if __name__ == '__main__':
    main()
