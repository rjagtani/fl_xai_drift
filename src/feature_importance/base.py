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
    values: np.ndarray
    method: str
    baseline_loss: Optional[float] = None
    total_loss: Optional[float] = None


class BaseImportanceComputer(ABC):
    """Abstract base class for feature importance computation."""
    
    def __init__(
        self,
        background_size: int = 32,
        use_kmeans: bool = False,
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
        Create background dataset via IID sampling (or K-means if enabled).
        
        Args:
            X: Input data array
            max_size: Maximum size of background set (default: self.background_size)
        
        Returns:
            Background data array of shape (<=max_size, n_features)
        """
        if max_size is None:
            max_size = self.background_size
        
        n_samples = X.shape[0]
        target_size = min(max_size, n_samples)
        
        if not self.use_kmeans or n_samples <= target_size:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=target_size, replace=False)
            return X[indices]
        
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
        max_samples: int = 1000,
    ) -> tuple:
        """
        Create estimation subset from validation data.
        
        Uses the full validation set or IID-samples max_samples from it,
        whichever is lower.
        
        Args:
            X: Feature array
            y: Label array
            max_samples: Maximum number of estimation samples
        
        Returns:
            Tuple of (X_est, y_est)
        """
        n_samples = X.shape[0]
        
        if n_samples <= max_samples:
            return X, y
        
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=max_samples, replace=False)
        
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
