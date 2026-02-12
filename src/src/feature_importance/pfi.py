"""
Permutation Feature Importance (PFI) computation.

Loss-based feature importance using permutation of feature values.
"""

from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn

from .base import BaseImportanceComputer, FeatureImportanceResult


class PFIComputer(BaseImportanceComputer):
    """
    Permutation Feature Importance computation.
    
    PFI measures feature importance by permuting each feature column
    and measuring the increase in prediction loss.
    
    PFI[j] = Loss(permute feature j) - Loss(original)
    
    Higher values indicate more important features.
    """
    
    def __init__(
        self,
        n_permutations: int = 5,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.n_permutations = n_permutations
    
    def _compute_loss(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute cross-entropy loss."""
        preds = model_fn(X)
        eps = 1e-10
        preds = np.clip(preds, eps, 1 - eps)
        n_samples = len(y)
        loss = -np.mean(np.log(preds[np.arange(n_samples), y]))
        return loss
    
    def compute(
        self,
        model_fn: Callable,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PFI values.
        
        Args:
            model_fn: Function that takes X and returns probabilities
            X_background: Not used (kept for API consistency)
            X_estimation: Data to compute importance on
            y_estimation: Labels for estimation data
        
        Returns:
            Array of PFI values per feature
        """
        X = np.asarray(X_estimation, dtype=np.float32)
        y = np.asarray(y_estimation, dtype=np.int64)
        
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        
        baseline_loss = self._compute_loss(model_fn, X, y)
        
        pfi_values = np.zeros(n_features)
        
        for j in range(n_features):
            permuted_losses = []
            
            for _ in range(self.n_permutations):
                X_permuted = X.copy()
                X_permuted[:, j] = rng.permutation(X_permuted[:, j])
                
                perm_loss = self._compute_loss(model_fn, X_permuted, y)
                permuted_losses.append(perm_loss)
            
            pfi_values[j] = np.mean(permuted_losses) - baseline_loss
        
        return pfi_values
    
    def compute_for_client(
        self,
        model,
        client_data,
        estimation_fraction: float = 0.2,
    ) -> FeatureImportanceResult:
        """
        Compute PFI values for a specific client.
        
        Args:
            model: Model with predict_proba method
            client_data: ClientDataset instance
            estimation_fraction: Fraction of validation data to use
        
        Returns:
            FeatureImportanceResult with PFI values
        """
        X_est, y_est = self.create_estimation_set(
            client_data.X_val,
            client_data.y_val,
            fraction=estimation_fraction,
        )
        
        if hasattr(model, 'predict_proba'):
            model_fn = model.predict_proba
        elif hasattr(model, '__call__'):
            model_fn = lambda x: model(x)
        else:
            raise ValueError("Model must have predict_proba method or be callable")
        
        pfi_values = self.compute(
            model_fn,
            None,
            X_est,
            y_est,
        )
        
        return FeatureImportanceResult(
            client_id=client_data.client_id,
            round_num=-1,
            values=pfi_values,
            method='pfi',
        )


class PFIWithBaseline(PFIComputer):
    """
    PFI computation that also returns baseline loss.
    """
    
    def compute_with_loss(
        self,
        model_fn: Callable,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> tuple:
        """
        Compute PFI values along with baseline loss.
        
        Returns:
            Tuple of (pfi_values, baseline_loss)
        """
        X = np.asarray(X_estimation, dtype=np.float32)
        y = np.asarray(y_estimation, dtype=np.int64)
        
        baseline_loss = self._compute_loss(model_fn, X, y)
        pfi_values = self.compute(model_fn, None, X, y)
        
        return pfi_values, baseline_loss
