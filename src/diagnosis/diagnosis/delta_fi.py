"""
Delta(FI) diagnosis method.

Ranks features by the absolute change in client-weighted mean FI between
the trigger round and the immediately preceding round:

    delta_j = |wmean_j(trigger) - wmean_j(trigger - 1)|

where wmean_j(r) = sum_i w_i * FI[r, i, j]  (weighted average across clients,
with w_i = n_i / N proportional to client training-set size).

Features with the largest delta are most likely to be drifted.
"""

from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class DeltaFIResult:
    """Result of Delta(FI) diagnosis."""

    method: str  # 'sage', 'pfi', or 'shap'
    delta_scores: np.ndarray  # |wmean_FI(trigger) - wmean_FI(prev)| per feature
    feature_ranking: np.ndarray  # Indices sorted by delta (descending)
    mean_trigger: np.ndarray  # Weighted mean FI at trigger round per feature
    mean_prev: np.ndarray  # Weighted mean FI at previous round per feature


def _weighted_nanmean(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Compute weighted mean along axis 0, ignoring NaNs.

    Args:
        values: (n_clients, n_features)
        weights: (n_clients,) or None for uniform

    Returns:
        (n_features,) weighted mean
    """
    mask = ~np.isnan(values)  # (C, F)
    if weights is None:
        return np.nanmean(values, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    result = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        m = mask[:, j]
        if not m.any():
            result[j] = 0.0
            continue
        wj = w[m]
        wj = wj / wj.sum()
        result[j] = np.average(values[m, j], weights=wj)
    return result


class DeltaFIDiagnosis:
    """
    Delta(FI) diagnosis: rank features by absolute change in weighted mean FI.

    For each feature j:
        1. Compute weighted mean FI across clients at trigger round: wmean_j(T)
        2. Compute weighted mean FI across clients at round T-1: wmean_j(T-1)
        3. delta_j = |wmean_j(T) - wmean_j(T-1)|

    Rank features by delta (descending).  Simple, interpretable, and only
    requires two rounds of FI values.
    """

    def compute_delta_scores(
        self,
        fi_matrix: np.ndarray,  # Shape: (n_rounds, n_clients, n_features)
        client_weights: Optional[np.ndarray] = None,  # (n_clients,)
    ) -> DeltaFIResult:
        """
        Compute delta scores for all features.

        Uses the last two rounds in the matrix (trigger and previous).

        Args:
            fi_matrix: FI values array (rounds, clients, features)
            client_weights: Per-client weights summing to 1.  If None,
                all clients are weighted equally (simple mean).

        Returns:
            DeltaFIResult with delta scores and ranking
        """
        n_rounds, n_clients, n_features = fi_matrix.shape

        if n_rounds < 2:
            raise ValueError(
                f"Need at least 2 rounds for delta computation, got {n_rounds}"
            )

        # Weighted mean across clients at trigger round (last) and previous round
        mean_trigger = _weighted_nanmean(fi_matrix[-1], client_weights)
        mean_prev = _weighted_nanmean(fi_matrix[-2], client_weights)

        # Absolute delta
        delta_scores = np.abs(mean_trigger - mean_prev)

        # Rank features by delta (descending)
        feature_ranking = np.argsort(delta_scores)[::-1]

        return DeltaFIResult(
            method='',
            delta_scores=delta_scores,
            feature_ranking=feature_ranking,
            mean_trigger=mean_trigger,
            mean_prev=mean_prev,
        )

    def diagnose(
        self,
        fi_matrices: Dict[str, np.ndarray],
        client_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, DeltaFIResult]:
        """
        Run Delta(FI) diagnosis for multiple FI methods.

        Args:
            fi_matrices: Dictionary mapping method name to FI matrix
            client_weights: Per-client weights (see compute_delta_scores)

        Returns:
            Dictionary mapping method name to DeltaFIResult
        """
        results = {}

        for method, fi_matrix in fi_matrices.items():
            result = self.compute_delta_scores(fi_matrix,
                                               client_weights=client_weights)
            result.method = method
            results[f"delta_{method}"] = result

        return results
