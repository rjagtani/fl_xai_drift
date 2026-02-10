"""
SAGE (Shapley Additive Global importancE) computation using sage-importance package.

Uses KernelEstimator for efficient Shapley value approximation.
"""

from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseImportanceComputer, FeatureImportanceResult

# Import SAGE components
try:
    from sage import MarginalImputer, KernelEstimator
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    print("Warning: sage-importance package not available. Install with: pip install sage-importance")


class SAGEComputer(BaseImportanceComputer):
    """
    SAGE feature importance using the sage-importance package.
    
    SAGE values measure how much each feature contributes to model predictions
    using Shapley values with a loss-based objective (cross-entropy by default).
    
    Uses KernelEstimator (KernelSAGE) for efficient computation.
    """
    
    def __init__(
        self,
        background_size: int = 50,
        use_kmeans: bool = True,
        loss_function: str = 'cross entropy',
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            background_size=background_size,
            use_kmeans=use_kmeans,
            random_state=random_state,
        )
        self.loss_function = loss_function
        self.verbose = verbose
    
    def compute(
        self,
        model_fn: Callable,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SAGE values using KernelEstimator.
        
        Args:
            model_fn: Function that takes X and returns probabilities
            X_background: Background data for marginal imputer
            X_estimation: Data to compute importance on
            y_estimation: Labels for estimation data
        
        Returns:
            Array of SAGE values per feature
        """
        if not SAGE_AVAILABLE:
            raise ImportError("sage-importance package required for SAGE computation")
        
        # Ensure proper dtypes
        X_background = np.asarray(X_background, dtype=np.float32)
        X_estimation = np.asarray(X_estimation, dtype=np.float32)
        y_estimation = np.asarray(y_estimation, dtype=np.int64)
        
        # Create marginal imputer
        imputer = MarginalImputer(model_fn, X_background)
        
        # Create kernel estimator
        estimator = KernelEstimator(imputer, self.loss_function)
        
        # Compute SAGE values
        sage_values = estimator(
            X_estimation,
            y_estimation,
            verbose=self.verbose,
        )
        
        return sage_values.values
    
    def compute_for_client(
        self,
        model,
        client_data,
        estimation_fraction: float = 0.2,
    ) -> FeatureImportanceResult:
        """
        Compute SAGE values for a specific client.
        
        Args:
            model: PyTorch model (or wrapper with predict_proba)
            client_data: ClientDataset instance
            estimation_fraction: Fraction of validation data to use
        
        Returns:
            FeatureImportanceResult with SAGE values
        """
        # Create background set from training data
        X_background = self.create_background_set(
            client_data.X_train,
            max_size=self.background_size,
        )
        
        # Create estimation set from validation data
        X_est, y_est = self.create_estimation_set(
            client_data.X_val,
            client_data.y_val,
            fraction=estimation_fraction,
        )
        
        # Get model prediction function
        if hasattr(model, 'predict_proba'):
            model_fn = model.predict_proba
        elif hasattr(model, '__call__'):
            model_fn = lambda x: model(x)
        else:
            raise ValueError("Model must have predict_proba method or be callable")
        
        # Compute SAGE values
        sage_values = self.compute(
            model_fn,
            X_background,
            X_est,
            y_est,
        )
        
        return FeatureImportanceResult(
            client_id=client_data.client_id,
            round_num=-1,  # To be filled by caller
            values=sage_values,
            method='sage',
        )


class SAGEWithLoss(SAGEComputer):
    """
    SAGE computation that also returns loss information for validation.
    """
    
    def compute_with_loss(
        self,
        model_fn: Callable,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> tuple:
        """
        Compute SAGE values and associated losses.
        
        Returns:
            Tuple of (sage_values, baseline_loss, total_loss)
        """
        if not SAGE_AVAILABLE:
            raise ImportError("sage-importance package required")
        
        X_background = np.asarray(X_background, dtype=np.float32)
        X_estimation = np.asarray(X_estimation, dtype=np.float32)
        y_estimation = np.asarray(y_estimation, dtype=np.int64)
        
        # Create imputer and estimator
        imputer = MarginalImputer(model_fn, X_background)
        estimator = KernelEstimator(imputer, self.loss_function)
        
        # Compute SAGE values
        sage_values = estimator(X_estimation, y_estimation, verbose=self.verbose)
        
        # Compute baseline loss (all features masked)
        n_samples, n_features = X_estimation.shape
        mask = np.zeros((n_samples, n_features), dtype=bool)
        baseline_preds = imputer(X_estimation, mask)
        
        # Compute cross-entropy loss
        eps = 1e-10
        baseline_preds = np.clip(baseline_preds, eps, 1 - eps)
        baseline_loss = -np.mean(
            np.log(baseline_preds[np.arange(n_samples), y_estimation])
        )
        
        # Compute total loss (no features masked)
        preds = model_fn(X_estimation)
        preds = np.clip(preds, eps, 1 - eps)
        total_loss = -np.mean(
            np.log(preds[np.arange(n_samples), y_estimation])
        )
        
        return sage_values.values, baseline_loss, total_loss
