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
    rds_scores: np.ndarray  # Shape: (n_features,)
    feature_ranking: np.ndarray  # Indices sorted by RDS (descending)
    wasserstein_distances: np.ndarray  # Raw Wasserstein distances
    past_stds: np.ndarray  # Standard deviations of past distributions


class DistFIDiagnosis:
    """
    Dist(FI) diagnosis using distributional analysis of feature importance.
    
    For each feature, computes Relative Distribution Shift (RDS):
    RDS[j] = W(current, past) / (std_past + eps)
    
    where W is the (optionally client-weighted) Wasserstein distance between:
    - current: FI values for feature j at trigger round across clients
    - past: FI values for feature j from past rounds across all clients
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
    ):
        self.eps = eps
    
    def compute_rds(
        self,
        fi_matrix: np.ndarray,  # Shape: (window_size+1, n_clients, n_features)
        trigger_idx: int = -1,  # Index of trigger round in the window (default: last)
        client_weights: Optional[np.ndarray] = None,  # (n_clients,) probability weights
    ) -> DistFIResult:
        """
        Compute RDS scores for all features.
        
        Args:
            fi_matrix: FI values array (rounds, clients, features)
            trigger_idx: Index of trigger round (-1 for last)
            client_weights: Per-client weights summing to 1.  When provided,
                Wasserstein distance and past std are client-weighted so that
                clients with more data contribute proportionally more.
                If None, all clients are weighted equally.
        
        Returns:
            DistFIResult with RDS scores and ranking
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        # Separate current (trigger) and past rounds
        if trigger_idx == -1:
            trigger_idx = n_rounds - 1
        
        current_fi = fi_matrix[trigger_idx]  # Shape: (n_clients, n_features)
        past_fi = fi_matrix[:trigger_idx]  # Shape: (past_rounds, n_clients, n_features)
        n_past = past_fi.shape[0]
        
        rds_scores = np.zeros(n_features)
        wasserstein_dists = np.zeros(n_features)
        past_stds = np.zeros(n_features)
        
        for j in range(n_features):
            # Current distribution: FI values across clients at trigger
            current_vals = current_fi[:, j]  # Shape: (n_clients,)
            
            # Past distribution: FI values across all past rounds and clients
            past_vals_2d = past_fi[:, :, j]  # Shape: (past_rounds, n_clients)
            past_vals = past_vals_2d.flatten()  # Shape: (past_rounds * n_clients,)
            
            # Build weight vectors (handle NaNs by masking + renormalising)
            cur_mask = ~np.isnan(current_vals)
            past_mask = ~np.isnan(past_vals)
            cur_clean = current_vals[cur_mask]
            past_clean = past_vals[past_mask]
            
            if len(cur_clean) == 0 or len(past_clean) == 0:
                rds_scores[j] = 0.0
                wasserstein_dists[j] = 0.0
                past_stds[j] = 0.0
                continue
            
            if client_weights is not None:
                # Current weights: w_i for each non-NaN client
                cur_w = client_weights[cur_mask]
                cur_w = cur_w / cur_w.sum()  # renormalise after NaN removal
                
                # Past weights: tile w_i across R past rounds, then mask
                # Each (round, client) gets weight w_i / R so total per
                # client = w_i.  After NaN removal we renormalise.
                past_w_2d = np.tile(client_weights, (n_past, 1))  # (R, C)
                past_w = past_w_2d.flatten()[past_mask]
                past_w = past_w / past_w.sum()
            else:
                cur_w = None
                past_w = None
            
            # Compute Wasserstein distance (scipy normalises weights internally)
            w_dist = wasserstein_distance(
                cur_clean, past_clean,
                u_weights=cur_w, v_weights=past_w,
            )
            
            # Compute std of past distribution (weighted if applicable)
            if past_w is not None:
                pw = past_w / past_w.sum()
                wmean = np.average(past_clean, weights=pw)
                wvar = np.average((past_clean - wmean) ** 2, weights=pw)
                std_past = np.sqrt(wvar)
            else:
                std_past = np.std(past_clean)
            
            # Compute RDS
            rds = w_dist / (std_past + self.eps)
            
            rds_scores[j] = rds
            wasserstein_dists[j] = w_dist
            past_stds[j] = std_past
        
        # Rank features by RDS (descending)
        feature_ranking = np.argsort(rds_scores)[::-1]
        
        return DistFIResult(
            method='',  # To be set by caller
            rds_scores=rds_scores,
            feature_ranking=feature_ranking,
            wasserstein_distances=wasserstein_dists,
            past_stds=past_stds,
        )
    
    def diagnose(
        self,
        fi_matrices: Dict[str, np.ndarray],  # method -> (rounds, clients, features)
        trigger_idx: int = -1,
        client_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, DistFIResult]:
        """
        Run Dist(FI) diagnosis for multiple FI methods.
        
        Args:
            fi_matrices: Dictionary mapping method name to FI matrix
            trigger_idx: Index of trigger round
            client_weights: Per-client weights (see compute_rds)
        
        Returns:
            Dictionary mapping method name to DistFIResult
        """
        results = {}
        
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_rds(fi_matrix, trigger_idx,
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
        past_fi = fi_matrix[:trigger_idx]  # (past_rounds, n_clients, n_features)
        
        if client_weights is not None:
            # Weighted mean / std across (rounds, clients) in past
            n_past = past_fi.shape[0]
            # Weights: w_i / R per (round, client), broadcast over features
            w_2d = np.tile(client_weights, (n_past, 1))  # (R, C)
            w_flat = w_2d.flatten()
            w_flat = w_flat / w_flat.sum()
            
            past_flat = past_fi.reshape(-1, n_features)  # (R*C, F)
            past_mean = np.average(past_flat, axis=0, weights=w_flat)
            past_var = np.average((past_flat - past_mean) ** 2, axis=0, weights=w_flat)
            past_std = np.sqrt(past_var)
        else:
            past_mean = np.nanmean(past_fi, axis=(0, 1))  # (n_features,)
            past_std = np.nanstd(past_fi, axis=(0, 1))  # (n_features,)
        
        # Z-score of current values relative to past
        contributions = (current_fi - past_mean) / (past_std + self.eps)
        
        return contributions
