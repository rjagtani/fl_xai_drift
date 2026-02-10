"""
Dist(FI) diagnosis method.

Computes RDS (Relative Distribution Shift) across client distributions
using client-weighted Wasserstein distance.

RDS[j] = W_weighted(current, past) / (weighted_std_past + eps)

Client weights w_i = n_i / N (proportion of total training samples).
- Current distribution: mass w_i on each client's FI value at trigger round.
- Past distribution: mass w_i / R on each (round, client) pair, where R is
  the number of past rounds, so each client's total mass is still w_i.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.stats import wasserstein_distance
from dataclasses import dataclass


@dataclass
class DistFIResult:
    """Result of Dist(FI) diagnosis."""
    
    method: str  # 'sage', 'pfi', or 'shap'
    rds_scores: np.ndarray  # Shape: (n_features,) - Final RDS at trigger round
    feature_ranking: np.ndarray  # Indices sorted by RDS (descending)
    wasserstein_distances: np.ndarray  # Raw Wasserstein distances
    past_stds: np.ndarray  # Standard deviations of past distributions
    
    # Within-window calibration results (optional)
    rds_series: Optional[np.ndarray] = None  # Shape: (n_check_rounds, n_features) - RDS over time
    rds_rounds: Optional[List[int]] = None  # Round indices where RDS was computed
    thresholds: Optional[np.ndarray] = None  # Shape: (n_features,) - Calibrated thresholds per feature
    triggered_features: Optional[np.ndarray] = None  # Boolean array of features that exceeded threshold
    calibration_mu: Optional[np.ndarray] = None  # Per-feature mu from calibration
    calibration_sigma: Optional[np.ndarray] = None  # Per-feature sigma from calibration


def _weighted_wasserstein(
    cur_vals: np.ndarray,
    past_vals: np.ndarray,
    cur_weights: Optional[np.ndarray],
    past_weights: Optional[np.ndarray],
) -> float:
    """Wasserstein-1 distance, optionally weighted."""
    return wasserstein_distance(
        cur_vals, past_vals,
        u_weights=cur_weights, v_weights=past_weights,
    )


def _weighted_std(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Standard deviation, optionally weighted."""
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        wmean = np.average(values, weights=w)
        wvar = np.average((values - wmean) ** 2, weights=w)
        return float(np.sqrt(wvar))
    return float(np.std(values))


def _build_weights_for_round(
    fi_row: np.ndarray,        # (n_clients,) FI for one feature at one round
    client_weights: Optional[np.ndarray],
):
    """Return (clean_values, clean_weights) with NaNs removed and weights renormalised."""
    mask = ~np.isnan(fi_row)
    vals = fi_row[mask]
    if client_weights is not None:
        w = client_weights[mask]
        w = w / w.sum() if w.sum() > 0 else w
    else:
        w = None
    return vals, w


class DistFIDiagnosis:
    """
    Dist(FI) diagnosis using distributional analysis of feature importance.
    
    For each feature, computes Relative Distribution Shift (RDS):
    RDS[j] = W(current, past) / (std_past + eps)
    
    where W is the (optionally client-weighted) Wasserstein distance between:
    - current: FI values for feature j at current round across clients
    - past: FI values for feature j from past window across all clients
    
    Features are ranked by RDS (descending) — higher RDS = more likely drifted.
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        rds_window: int = 5,  # Window size for computing RDS (past rounds)
    ):
        self.eps = eps
        self.rds_window = rds_window
    
    def _rds_for_feature(
        self,
        current_fi_j: np.ndarray,   # (n_clients,)
        past_fi_j: np.ndarray,      # (R, n_clients)
        client_weights: Optional[np.ndarray],
    ) -> tuple:
        """Compute RDS, W-dist, past-std for a single feature j."""
        n_past, n_clients = past_fi_j.shape

        # Current
        cur_mask = ~np.isnan(current_fi_j)
        cur_vals = current_fi_j[cur_mask]
        if client_weights is not None:
            cur_w = client_weights[cur_mask]
            cur_w = cur_w / cur_w.sum() if cur_w.sum() > 0 else cur_w
        else:
            cur_w = None

        # Past (flatten)
        past_flat = past_fi_j.flatten()
        past_mask = ~np.isnan(past_flat)
        past_vals = past_flat[past_mask]
        if client_weights is not None:
            past_w_2d = np.tile(client_weights, (n_past, 1))
            past_w = past_w_2d.flatten()[past_mask]
            past_w = past_w / past_w.sum() if past_w.sum() > 0 else past_w
        else:
            past_w = None

        if len(cur_vals) == 0 or len(past_vals) == 0:
            return 0.0, 0.0, 0.0

        w_dist = _weighted_wasserstein(cur_vals, past_vals, cur_w, past_w)
        std_past = _weighted_std(past_vals, past_w)
        rds = w_dist / (std_past + self.eps)
        return rds, w_dist, std_past

    def compute_rds_at_round(
        self,
        fi_matrix: np.ndarray,  # Shape: (n_rounds, n_clients, n_features)
        current_idx: int,  # Index of current round
        past_start_idx: int,  # Start index of past window
        client_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute RDS for all features at a single round.
        
        Returns:
            RDS scores per feature, shape (n_features,)
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        rds_scores = np.zeros(n_features)
        
        current_fi = fi_matrix[current_idx]            # (n_clients, n_features)
        past_fi = fi_matrix[past_start_idx:current_idx] # (R, n_clients, n_features)

        for j in range(n_features):
            rds_scores[j], _, _ = self._rds_for_feature(
                current_fi[:, j], past_fi[:, :, j], client_weights)
        
        return rds_scores
    
    def compute_rds_with_calibration(
        self,
        fi_matrix: np.ndarray,  # Shape: (window_size, n_clients, n_features)
        diagnosis_rounds: List[int],  # Actual round numbers
        client_weights: Optional[np.ndarray] = None,
    ) -> DistFIResult:
        """
        Compute RDS with within-window calibration.
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        first_rds_idx = self.rds_window
        n_rds_rounds = n_rounds - first_rds_idx
        
        if n_rds_rounds < 1:
            raise ValueError(f"Not enough rounds for RDS computation. "
                           f"Need at least {self.rds_window + 1} rounds, got {n_rounds}")
        
        # Compute RDS for all available rounds
        rds_series = np.zeros((n_rds_rounds, n_features))
        rds_round_indices = []
        
        for i in range(n_rds_rounds):
            current_idx = first_rds_idx + i
            past_start_idx = i
            rds_series[i] = self.compute_rds_at_round(
                fi_matrix, current_idx, past_start_idx,
                client_weights=client_weights)
            rds_round_indices.append(diagnosis_rounds[current_idx])
        
        # Calibration: use all but last RDS value for mu, sigma
        n_calibration_actual = min(
            getattr(self, 'n_calibration', n_rds_rounds - 1),
            n_rds_rounds - 1,
        )
        if n_calibration_actual < 1:
            calibration_mu = np.zeros(n_features)
            calibration_sigma = np.ones(n_features) * 0.1
        else:
            calibration_rds = rds_series[-(n_calibration_actual + 1):-1]
            calibration_mu = np.mean(calibration_rds, axis=0)
            calibration_sigma = np.std(calibration_rds, axis=0, ddof=1)
            calibration_sigma = np.maximum(calibration_sigma, 0.01)
        
        thresholds = calibration_mu + getattr(self, 'alpha', 3.0) * calibration_sigma
        
        final_rds = rds_series[-1]
        triggered_features = final_rds > thresholds
        feature_ranking = np.argsort(final_rds)[::-1]
        
        # Raw Wasserstein / past-std for trigger round
        trigger_idx = n_rounds - 1
        past_start = trigger_idx - self.rds_window
        current_fi = fi_matrix[trigger_idx]
        past_fi = fi_matrix[past_start:trigger_idx]
        
        wasserstein_dists = np.zeros(n_features)
        past_stds = np.zeros(n_features)
        for j in range(n_features):
            _, wasserstein_dists[j], past_stds[j] = self._rds_for_feature(
                current_fi[:, j], past_fi[:, :, j], client_weights)
        
        return DistFIResult(
            method='',
            rds_scores=final_rds,
            feature_ranking=feature_ranking,
            wasserstein_distances=wasserstein_dists,
            past_stds=past_stds,
            rds_series=rds_series,
            rds_rounds=rds_round_indices,
            thresholds=thresholds,
            triggered_features=triggered_features,
            calibration_mu=calibration_mu,
            calibration_sigma=calibration_sigma,
        )
    
    def compute_rds(
        self,
        fi_matrix: np.ndarray,  # Shape: (window_size+1, n_clients, n_features)
        trigger_idx: int = -1,
        client_weights: Optional[np.ndarray] = None,
    ) -> DistFIResult:
        """
        Compute RDS scores for all features (simple version without calibration).
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        if trigger_idx == -1:
            trigger_idx = n_rounds - 1
        
        current_fi = fi_matrix[trigger_idx]
        past_fi = fi_matrix[:trigger_idx]
        
        rds_scores = np.zeros(n_features)
        wasserstein_dists = np.zeros(n_features)
        past_stds = np.zeros(n_features)
        
        for j in range(n_features):
            rds_scores[j], wasserstein_dists[j], past_stds[j] = \
                self._rds_for_feature(current_fi[:, j], past_fi[:, :, j],
                                      client_weights)
        
        feature_ranking = np.argsort(rds_scores)[::-1]
        
        return DistFIResult(
            method='',
            rds_scores=rds_scores,
            feature_ranking=feature_ranking,
            wasserstein_distances=wasserstein_dists,
            past_stds=past_stds,
        )
    
    def diagnose(
        self,
        fi_matrices: Dict[str, np.ndarray],
        trigger_idx: int = -1,
        client_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, DistFIResult]:
        """Run Dist(FI) diagnosis for multiple FI methods (simple version)."""
        results = {}
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_rds(fi_matrix, trigger_idx,
                                      client_weights=client_weights)
            result.method = method
            results[f"dist_{method}"] = result
        return results
    
    def diagnose_with_calibration(
        self,
        fi_matrices: Dict[str, np.ndarray],
        diagnosis_rounds: List[int],
        client_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, DistFIResult]:
        """Run Dist(FI) diagnosis with within-window calibration."""
        results = {}
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_rds_with_calibration(
                fi_matrix, diagnosis_rounds,
                client_weights=client_weights)
            result.method = method
            results[f"dist_{method}"] = result
        return results


class DistFIWithClients(DistFIDiagnosis):
    """
    Extended Dist(FI) that also identifies which clients show the most change.
    """
    
    def compute_client_contributions(
        self,
        fi_matrix: np.ndarray,
        trigger_idx: int = -1,
        client_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute how much each client contributes to the distribution shift.
        
        Returns:
            Array of shape (n_clients, n_features) with contribution scores
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        if trigger_idx == -1:
            trigger_idx = n_rounds - 1
        
        current_fi = fi_matrix[trigger_idx]  # (n_clients, n_features)
        past_fi = fi_matrix[:trigger_idx]    # (past_rounds, n_clients, n_features)
        
        if client_weights is not None:
            n_past = past_fi.shape[0]
            w_2d = np.tile(client_weights, (n_past, 1))
            w_flat = w_2d.flatten()
            w_flat = w_flat / w_flat.sum()
            past_flat = past_fi.reshape(-1, n_features)
            past_mean = np.average(past_flat, axis=0, weights=w_flat)
            past_var = np.average((past_flat - past_mean) ** 2, axis=0, weights=w_flat)
            past_std = np.sqrt(past_var)
        else:
            past_mean = np.nanmean(past_fi, axis=(0, 1))
            past_std = np.nanstd(past_fi, axis=(0, 1))
        
        contributions = (current_fi - past_mean) / (past_std + self.eps)
        return contributions
