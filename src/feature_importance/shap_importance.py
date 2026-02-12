"""
SHAP (SHapley Additive exPlanations) feature importance using KernelSHAP.

Uses the shap package for computing prediction-based Shapley values.
"""

from typing import Callable, Optional
import numpy as np

from .base import BaseImportanceComputer, FeatureImportanceResult

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap package not available. Install with: pip install shap")


class SHAPComputer(BaseImportanceComputer):
    """
    SHAP feature importance using KernelSHAP.
    
    KernelSHAP is a model-agnostic method that approximates SHAP values
    using a weighted linear regression approach.
    
    For diagnosis, we use mean absolute SHAP values across samples:
    |SHAP|[j] = mean(|shap_s,j|) for all samples s
    """
    
    def __init__(
        self,
        background_size: int = 32,
        use_kmeans: bool = False,
        n_samples_explain: int = 1000,
        random_state: int = 42,
        silent: bool = True,
    ):
        super().__init__(
            background_size=background_size,
            use_kmeans=use_kmeans,
            random_state=random_state,
        )
        self.n_samples_explain = n_samples_explain
        self.silent = silent
    
    def compute(
        self,
        model_fn: Callable,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean absolute SHAP values using KernelSHAP.
        
        Args:
            model_fn: Function that takes X and returns probabilities
            X_background: Background data for KernelSHAP
            X_estimation: Data to explain
            y_estimation: Labels (used for selecting correct class probabilities)
        
        Returns:
            Array of mean absolute SHAP values per feature
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap package required for SHAP computation")
        
        X_background = np.asarray(X_background, dtype=np.float32)
        X_estimation = np.asarray(X_estimation, dtype=np.float32)
        y_estimation = np.asarray(y_estimation, dtype=np.int64)
        
        n_explain = min(self.n_samples_explain, len(X_estimation))
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X_estimation), size=n_explain, replace=False)
        X_explain = X_estimation[indices]
        y_explain = y_estimation[indices]
        
        explainer = shap.KernelExplainer(model_fn, X_background, silent=self.silent)
        
        shap_values = explainer.shap_values(X_explain, silent=self.silent)
        
        if isinstance(shap_values, list):
            all_shap = np.zeros((n_explain, X_estimation.shape[1]))
            for i in range(n_explain):
                all_shap[i] = shap_values[y_explain[i]][i]
        else:
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                all_shap = np.mean(np.abs(shap_values), axis=-1)
            else:
                all_shap = shap_values
        
        mean_abs_shap = np.mean(np.abs(all_shap), axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = np.mean(mean_abs_shap, axis=-1)
        return mean_abs_shap.ravel()
    
    def compute_raw_shap(
        self,
        model_fn: Callable,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute raw SHAP values for all samples.
        
        Returns:
            Array of shape (n_samples, n_features) with SHAP values
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap package required")
        
        X_background = np.asarray(X_background, dtype=np.float32)
        X_estimation = np.asarray(X_estimation, dtype=np.float32)
        
        explainer = shap.KernelExplainer(model_fn, X_background, silent=self.silent)
        shap_values = explainer.shap_values(X_estimation, silent=self.silent)
        
        if isinstance(shap_values, list):
            return shap_values[1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            return np.mean(shap_values, axis=-1)
        return shap_values
    
    def compute_for_client(
        self,
        model,
        client_data,
        max_samples: int = 1000,
    ) -> FeatureImportanceResult:
        """
        Compute SHAP values for a specific client.
        
        Args:
            model: Model with predict_proba method
            client_data: ClientDataset instance
            max_samples: Max estimation samples (use full val if smaller)
        
        Returns:
            FeatureImportanceResult with mean absolute SHAP values
        """
        X_background = self.create_background_set(
            client_data.X_train,
            max_size=self.background_size,
        )
        
        X_est, y_est = self.create_estimation_set(
            client_data.X_val,
            client_data.y_val,
            max_samples=max_samples,
        )
        
        if hasattr(model, 'predict_proba'):
            model_fn = model.predict_proba
        elif hasattr(model, '__call__'):
            model_fn = lambda x: model(x)
        else:
            raise ValueError("Model must have predict_proba method or be callable")
        
        shap_values = self.compute(
            model_fn,
            X_background,
            X_est,
            y_est,
        )
        
        return FeatureImportanceResult(
            client_id=client_data.client_id,
            round_num=-1,
            values=shap_values,
            method='shap',
        )
