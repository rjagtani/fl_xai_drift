"""
Base classes for data generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ClientDataset:
    """Container for a single client's data."""
    
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    is_drifted: bool = False
    
    @property
    def n_train(self) -> int:
        return len(self.y_train)
    
    @property
    def n_val(self) -> int:
        return len(self.y_val)
    
    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


class TorchDataset(Dataset):
    """PyTorch Dataset wrapper for client data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class BaseDataGenerator(ABC):
    """Abstract base class for synthetic data generators."""
    
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        n_features: int,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.n_clients = n_clients
        self.n_samples_per_client = n_samples_per_client
        self.n_features = n_features
        self.test_size = test_size
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
    
    @abstractmethod
    def generate_round_data(
        self,
        round_num: int,
        t0: int,
        drifted_clients: Set[int],
        drift_magnitude: float,
    ) -> Dict[int, ClientDataset]:
        """
        Generate data for all clients for a given round.
        
        Args:
            round_num: Current FL round (1-indexed)
            t0: Drift onset round
            drifted_clients: Set of client IDs that experience drift
            drift_magnitude: Magnitude of drift
        
        Returns:
            Dictionary mapping client_id to ClientDataset
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        pass
    
    @abstractmethod
    def get_drifted_feature_indices(self) -> Set[int]:
        """Return indices of features that drift."""
        pass
    
    def create_dataloaders(
        self,
        client_data: ClientDataset,
        batch_size: int,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for a client's data."""
        train_dataset = TorchDataset(client_data.X_train, client_data.y_train)
        val_dataset = TorchDataset(client_data.X_val, client_data.y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader
