"""
Dist(FI) diagnosis method.

Computes RDS (Relative Distribution Shift) across client distributions
using Wasserstein distance.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.stats import wasserstein_distance
from dataclasses import dataclass


@dataclass
class DistFIResult:
    """Result of Dist(FI) diagnosis."""
    
    method: str
    rds_scores: np.ndarray
    feature_ranking: np.ndarray
    wasserstein_distances: np.ndarray
    past_stds: np.ndarray
    
    rds_series: Optional[np.ndarray] = None
    rds_rounds: Optional[List[int]] = None
    thresholds: Optional[np.ndarray] = None
    triggered_features: Optional[np.ndarray] = None
    calibration_mu: Optional[np.ndarray] = None
    calibration_sigma: Optional[np.ndarray] = None


class DistFIDiagnosis:
    """
    Dist(FI) diagnosis using distributional analysis of feature importance.
    
    For each feature, computes Relative Distribution Shift (RDS):
    RDS[j] = W(current, past) / (std_past + eps)
    
    where W is the Wasserstein distance between:
    - current: FI values for feature j at current round across clients
    - past: FI values for feature j from past window across all clients
    
    Within-window calibration:
    - With rds_window=5 and n_calibration=3, we need FI for 9 rounds (e.g., 242-250)
    - RDS at rounds 247, 248, 249 used for calibration (compute mu, sigma per feature)
    - RDS at round 250 (trigger) checked against threshold = mu + alpha * sigma
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        rds_window: int = 5,
        n_calibration: int = 3,
        alpha: float = 2.33,
    ):
        self.eps = eps
        self.rds_window = rds_window
        self.n_calibration = n_calibration
        self.alpha = alpha
    
    def compute_rds_at_round(
        self,
        fi_matrix: np.ndarray,
        current_idx: int,
        past_start_idx: int,
    ) -> np.ndarray:
        """
        Compute RDS for all features at a single round.
        
        Args:
            fi_matrix: FI values array (rounds, clients, features)
            current_idx: Index of current round
            past_start_idx: Start index of past window (past = [past_start_idx, current_idx))
        
        Returns:
            RDS scores per feature, shape (n_features,)
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        current_fi = fi_matrix[current_idx]
        past_fi = fi_matrix[past_start_idx:current_idx]
        
        rds_scores = np.zeros(n_features)
        
        for j in range(n_features):
            current_dist = current_fi[:, j]
            past_dist = past_fi[:, :, j].flatten()
            
            current_dist = current_dist[~np.isnan(current_dist)]
            past_dist = past_dist[~np.isnan(past_dist)]
            
            if len(current_dist) == 0 or len(past_dist) == 0:
                rds_scores[j] = 0.0
                continue
            
            w_dist = wasserstein_distance(current_dist, past_dist)
            std_past = np.std(past_dist)
            rds_scores[j] = w_dist / (std_past + self.eps)
        
        return rds_scores
    
    def compute_rds_with_calibration(
        self,
        fi_matrix: np.ndarray,
        diagnosis_rounds: List[int],
    ) -> DistFIResult:
        """
        Compute RDS with within-window calibration.
        
        Example with rds_window=5, n_calibration=3, window_size=9:
        - FI computed for rounds 242-250 (indices 0-8)
        - RDS at index 5 (round 247): past = indices 0-4 (rounds 242-246)
        - RDS at index 6 (round 248): past = indices 1-5 (rounds 243-247)
        - RDS at index 7 (round 249): past = indices 2-6 (rounds 244-248)
        - RDS at index 8 (round 250, trigger): past = indices 3-7 (rounds 245-249)
        
        Calibration: mu, sigma from RDS at indices 5,6,7 (rounds 247,248,249)
        Check: RDS at index 8 (round 250) against threshold = mu + alpha * sigma
        
        Args:
            fi_matrix: FI values, shape (window_size, n_clients, n_features)
            diagnosis_rounds: List of round numbers corresponding to fi_matrix indices
        
        Returns:
            DistFIResult with calibration info
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        first_rds_idx = self.rds_window
        n_rds_rounds = n_rounds - first_rds_idx
        
        if n_rds_rounds < 1:
            raise ValueError(f"Not enough rounds for RDS computation. "
                           f"Need at least {self.rds_window + 1} rounds, got {n_rounds}")
        
        rds_series = np.zeros((n_rds_rounds, n_features))
        rds_round_indices = []
        
        for i in range(n_rds_rounds):
            current_idx = first_rds_idx + i
            past_start_idx = i
            rds_series[i] = self.compute_rds_at_round(fi_matrix, current_idx, past_start_idx)
            rds_round_indices.append(diagnosis_rounds[current_idx])
        
        n_calibration_actual = min(self.n_calibration, n_rds_rounds - 1)
        if n_calibration_actual < 1:
            calibration_mu = np.zeros(n_features)
            calibration_sigma = np.ones(n_features) * 0.1
        else:
            calibration_rds = rds_series[-(n_calibration_actual + 1):-1]
            calibration_mu = np.mean(calibration_rds, axis=0)
            calibration_sigma = np.std(calibration_rds, axis=0, ddof=1)
            calibration_sigma = np.maximum(calibration_sigma, 0.01)
        
        thresholds = calibration_mu + self.alpha * calibration_sigma
        
        final_rds = rds_series[-1]
        
        triggered_features = final_rds > thresholds
        
        feature_ranking = np.argsort(final_rds)[::-1]
        
        trigger_idx = n_rounds - 1
        past_start = trigger_idx - self.rds_window
        current_fi = fi_matrix[trigger_idx]
        past_fi = fi_matrix[past_start:trigger_idx]
        
        wasserstein_dists = np.zeros(n_features)
        past_stds = np.zeros(n_features)
        for j in range(n_features):
            current_dist = current_fi[:, j]
            past_dist = past_fi[:, :, j].flatten()
            current_dist = current_dist[~np.isnan(current_dist)]
            past_dist = past_dist[~np.isnan(past_dist)]
            if len(current_dist) > 0 and len(past_dist) > 0:
                wasserstein_dists[j] = wasserstein_distance(current_dist, past_dist)
                past_stds[j] = np.std(past_dist)
        
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
        fi_matrix: np.ndarray,
        trigger_idx: int = -1,
    ) -> DistFIResult:
        """
        Compute RDS scores for all features (simple version without calibration).
        
        Args:
            fi_matrix: FI values array (rounds, clients, features)
            trigger_idx: Index of trigger round (-1 for last)
        
        Returns:
            DistFIResult with RDS scores and ranking
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
            current_dist = current_fi[:, j]
            
            past_dist = past_fi[:, :, j].flatten()
            
            current_dist = current_dist[~np.isnan(current_dist)]
            past_dist = past_dist[~np.isnan(past_dist)]
            
            if len(current_dist) == 0 or len(past_dist) == 0:
                rds_scores[j] = 0.0
                wasserstein_dists[j] = 0.0
                past_stds[j] = 0.0
                continue
            
            w_dist = wasserstein_distance(current_dist, past_dist)
            
            std_past = np.std(past_dist)
            
            rds = w_dist / (std_past + self.eps)
            
            rds_scores[j] = rds
            wasserstein_dists[j] = w_dist
            past_stds[j] = std_past
        
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
    ) -> Dict[str, DistFIResult]:
        """
        Run Dist(FI) diagnosis for multiple FI methods (simple version).
        
        Args:
            fi_matrices: Dictionary mapping method name to FI matrix
            trigger_idx: Index of trigger round
        
        Returns:
            Dictionary mapping method name to DistFIResult
        """
        results = {}
        
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_rds(fi_matrix, trigger_idx)
            result.method = method
            results[f"dist_{method}"] = result
        
        return results
    
    def diagnose_with_calibration(
        self,
        fi_matrices: Dict[str, np.ndarray],
        diagnosis_rounds: List[int],
    ) -> Dict[str, DistFIResult]:
        """
        Run Dist(FI) diagnosis with within-window calibration.
        
        Args:
            fi_matrices: Dictionary mapping method name to FI matrix
            diagnosis_rounds: List of round numbers corresponding to FI matrix indices
        
        Returns:
            Dictionary mapping method name to DistFIResult with calibration
        """
        results = {}
        
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_rds_with_calibration(fi_matrix, diagnosis_rounds)
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
    ) -> np.ndarray:
        """
        Compute how much each client contributes to the distribution shift.
        
        Returns:
            Array of shape (n_clients, n_features) with contribution scores
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        if trigger_idx == -1:
            trigger_idx = n_rounds - 1
        
        current_fi = fi_matrix[trigger_idx]
        past_fi = fi_matrix[:trigger_idx]
        
        past_mean = np.nanmean(past_fi, axis=(0, 1))
        past_std = np.nanstd(past_fi, axis=(0, 1))
        
        contributions = (current_fi - past_mean) / (past_std + self.eps)
        
        return contributions
