"""
Base classes for feature importance computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class FeatureImportanceResult:
    """Result of feature importance computation for one client at one round."""
    
    client_id: int
    round_num: int
    values: np.ndarray  # Shape: (n_features,)
    method: str  # 'sage', 'pfi', or 'shap'
    baseline_loss: Optional[float] = None
    total_loss: Optional[float] = None


class BaseImportanceComputer(ABC):
    """Abstract base class for feature importance computation."""
    
    def __init__(
        self,
        background_size: int = 50,
        use_kmeans: bool = True,
        random_state: int = 42,
    ):
        self.background_size = background_size
        self.use_kmeans = use_kmeans
        self.random_state = random_state
    
    def create_background_set(
        self,
        X: np.ndarray,
        max_size: int = None,
    ) -> np.ndarray:
        """
        Create background dataset using K-means clustering.
        
        Args:
            X: Input data array
            max_size: Maximum size of background set
        
        Returns:
            Background data array of shape (background_size, n_features)
        """
        if max_size is None:
            max_size = self.background_size
        
        n_samples = X.shape[0]
        target_size = min(max_size, n_samples)
        
        if not self.use_kmeans or n_samples <= target_size:
            # Random sampling
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=target_size, replace=False)
            return X[indices]
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=target_size,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(X)
        
        return kmeans.cluster_centers_
    
    def create_estimation_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fraction: float = 0.2,
    ) -> tuple:
        """
        Create estimation subset from validation data.
        
        Args:
            X: Feature array
            y: Label array
            fraction: Fraction of data to use
        
        Returns:
            Tuple of (X_est, y_est)
        """
        n_samples = X.shape[0]
        est_size = max(1, int(n_samples * fraction))
        
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=est_size, replace=False)
        
        return X[indices], y[indices]
    
    @abstractmethod
    def compute(
        self,
        model,
        X_background: np.ndarray,
        X_estimation: np.ndarray,
        y_estimation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute feature importance values.
        
        Args:
            model: Model to explain (with predict_proba method)
            X_background: Background data for reference
            X_estimation: Data to compute importance on
            y_estimation: Labels for estimation data
        
        Returns:
            Array of feature importance values
        """
        pass
