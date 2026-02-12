"""
Mean(FI)+PH diagnosis method.

Uses per-feature Page-Hinkley detection on mean FI time series.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from ..triggers.page_hinkley import FeaturePageHinkley


@dataclass
class MeanFIPHResult:
    """Result of Mean(FI)+PH diagnosis."""
    
    method: str
    ph_scores: np.ndarray
    feature_ranking: np.ndarray
    triggered_features: np.ndarray
    mean_series: np.ndarray


class MeanFIPHDiagnosis:
    """
    Mean(FI)+PH diagnosis using Page-Hinkley detection on mean FI.
    
    For each feature:
    1. Compute mean FI across clients for each round: mean_j[r] = avg_i FI[r][i][j]
    2. Run Page-Hinkley detection on the mean_j time series
    3. Use final PH statistic as feature score
    
    This method may fail in "cancellation scenarios" where increases in some
    clients cancel out decreases in others.
    """
    
    def __init__(
        self,
        ph_delta: float = 0.005,
        ph_lambda: float = 50.0,
        warmup: int = 3,
    ):
        self.ph_delta = ph_delta
        self.ph_lambda = ph_lambda
        self.warmup = warmup
    
    def compute_mean_series(
        self,
        fi_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean FI per feature across clients for each round.
        
        Args:
            fi_matrix: FI values array
        
        Returns:
            Array of shape (n_rounds, n_features) with mean FI values
        """
        return np.nanmean(fi_matrix, axis=1)
    
    def compute_ph_scores(
        self,
        fi_matrix: np.ndarray,
    ) -> MeanFIPHResult:
        """
        Compute PH scores for all features.
        
        Args:
            fi_matrix: FI values array (rounds, clients, features)
        
        Returns:
            MeanFIPHResult with PH scores and ranking
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        mean_series = self.compute_mean_series(fi_matrix)
        
        feature_ph = FeaturePageHinkley(
            n_features=n_features,
            delta=self.ph_delta,
            lambda_=self.ph_lambda,
            warmup=self.warmup,
        )
        
        triggered, ph_scores = feature_ph.detect(mean_series)
        
        feature_ranking = np.argsort(ph_scores)[::-1]
        
        return MeanFIPHResult(
            method='',
            ph_scores=ph_scores,
            feature_ranking=feature_ranking,
            triggered_features=triggered,
            mean_series=mean_series,
        )
    
    def diagnose(
        self,
        fi_matrices: Dict[str, np.ndarray],
    ) -> Dict[str, MeanFIPHResult]:
        """
        Run Mean(FI)+PH diagnosis for multiple FI methods.
        
        Args:
            fi_matrices: Dictionary mapping method name to FI matrix
        
        Returns:
            Dictionary mapping method name to MeanFIPHResult
        """
        results = {}
        
        for method, fi_matrix in fi_matrices.items():
            result = self.compute_ph_scores(fi_matrix)
            result.method = method
            results[f"meanph_{method}"] = result
        
        return results


class MeanFIPHWithVariance(MeanFIPHDiagnosis):
    """
    Extended Mean(FI)+PH that also considers variance in FI values.
    
    Can be used as an alternative score that combines mean change and variance.
    """
    
    def compute_combined_score(
        self,
        fi_matrix: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Compute combined score using both PH and variance.
        
        Args:
            fi_matrix: FI values array
            alpha: Weight for PH score (1-alpha for variance)
        
        Returns:
            Array of combined scores per feature
        """
        n_rounds, n_clients, n_features = fi_matrix.shape
        
        ph_result = self.compute_ph_scores(fi_matrix)
        ph_scores_norm = ph_result.ph_scores / (np.max(ph_result.ph_scores) + 1e-8)
        
        mid_point = n_rounds // 2
        early_var = np.nanvar(fi_matrix[:mid_point], axis=(0, 1))
        late_var = np.nanvar(fi_matrix[mid_point:], axis=(0, 1))
        var_change = np.abs(late_var - early_var) / (early_var + 1e-8)
        var_scores_norm = var_change / (np.max(var_change) + 1e-8)
        
        combined = alpha * ph_scores_norm + (1 - alpha) * var_scores_norm
        
        return combined
