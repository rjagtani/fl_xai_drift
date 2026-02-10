"""
Hyperplane dataset generator for federated learning with concept drift.
"""

from typing import Dict, List, Optional, Set
import numpy as np
from river.datasets import synth
from sklearn.model_selection import train_test_split

from .base import BaseDataGenerator, ClientDataset

# Fixed seed for Hyperplane coefficient generation.
# seed=123 gives weights approx [0.4, 0.7, 4.0, 1.2, 10.1] -- x4 always dominates.
# This is separated from the experiment seed so that feature importance rankings
# are stable across experiments while data sampling varies with the experiment seed.
HYPERPLANE_COEFF_SEED = 123


class HyperplaneDataGenerator(BaseDataGenerator):
    """
    Generates Hyperplane datasets for FL with configurable concept drift.
    
    The Hyperplane dataset is a classic synthetic dataset where the decision boundary
    is defined by a hyperplane. Drift is simulated by rotating the hyperplane coefficients.
    
    The coefficient seed is fixed (HYPERPLANE_COEFF_SEED=123) so that x4 is always
    the most important feature regardless of the experiment seed. The experiment seed
    controls data sampling randomness (which samples each client gets via shuffled
    assignment from a pre-generated pool).
    """
    
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        n_features: int = 5,
        n_drift_features: int = 2,
        noise_percentage: float = 0.05,
        sigma: float = 0.1,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=n_features,
            test_size=test_size,
            seed=seed,
        )
        
        self.n_drift_features = n_drift_features
        self.noise_percentage = noise_percentage
        self.sigma = sigma
        
        # River's Hyperplane drifts the FIRST n_drift_features features (x0, x1).
        # With HYPERPLANE_COEFF_SEED=123 baseline: w approx [0.4, 0.7, 4.0, 1.2, 10.1]
        # x4 is the most important feature; x0/x1 are the ones whose coefficients rotate.
        self._drifted_feature_indices = set(range(n_drift_features))  # {0, 1}
        
        # Pre-generated data pools (lazy init)
        self._baseline_pool: Optional[np.ndarray] = None  # (N_total, n_features+1)
        self._drifted_pools: Dict[float, np.ndarray] = {}
    
    def _generate_pool(self, mag_change: float, n_total: int) -> np.ndarray:
        """Generate a pool of n_total samples from a fixed-seed Hyperplane stream."""
        stream = synth.Hyperplane(
            seed=HYPERPLANE_COEFF_SEED,
            n_features=self.n_features,
            n_drift_features=self.n_drift_features if mag_change > 0 else 0,
            mag_change=mag_change,
            noise_percentage=self.noise_percentage,
            sigma=self.sigma,
        )
        data = []
        for x, y in stream.take(n_total):
            data.append(list(x.values()) + [y])
        return np.array(data)
    
    def _get_baseline_pool(self, n_total: int) -> np.ndarray:
        """Get or create the baseline (no drift) data pool."""
        if self._baseline_pool is None or len(self._baseline_pool) < n_total:
            self._baseline_pool = self._generate_pool(0.0, n_total)
        return self._baseline_pool
    
    def _get_drifted_pool(self, mag_change: float, n_total: int) -> np.ndarray:
        """Get or create a drifted data pool for a given magnitude."""
        if mag_change not in self._drifted_pools or len(self._drifted_pools[mag_change]) < n_total:
            self._drifted_pools[mag_change] = self._generate_pool(mag_change, n_total)
        return self._drifted_pools[mag_change]
    
    def _sample_client_data(
        self, pool: np.ndarray, client_id: int, round_offset: int = 0
    ) -> tuple:
        """
        Sample n_samples_per_client from the pool for a specific client.
        
        Uses the experiment seed + client_id + round_offset to shuffle the pool
        and pick a slice, ensuring different clients/rounds get different data
        while the underlying hyperplane coefficients remain fixed.
        """
        rng = np.random.default_rng(self.seed + client_id * 1000 + round_offset)
        indices = rng.choice(len(pool), size=self.n_samples_per_client, replace=True)
        data = pool[indices]
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [f'x{i}' for i in range(self.n_features)]
    
    def get_drifted_feature_indices(self) -> Set[int]:
        """Return indices of features that drift."""
        return self._drifted_feature_indices

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
            drift_magnitude: Magnitude of drift (mag_change parameter)
        
        Returns:
            Dictionary mapping client_id to ClientDataset
        """
        is_drift_phase = round_num > t0
        
        # Ensure pools exist (generate once, reuse across rounds)
        pool_size = max(self.n_samples_per_client * 20, 10000)
        baseline_pool = self._get_baseline_pool(pool_size)
        
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            is_drifted_client = client_id in drifted_clients
            
            if is_drift_phase and is_drifted_client:
                drifted_pool = self._get_drifted_pool(drift_magnitude, pool_size)
                X, y = self._sample_client_data(drifted_pool, client_id, round_offset=round_num)
            else:
                X, y = self._sample_client_data(baseline_pool, client_id, round_offset=round_num)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.seed + client_id,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_drifted=is_drifted_client and is_drift_phase,
            )
        
        return client_datasets
    
    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        drift_magnitude: float,
        generate_drifted: bool = False,
    ) -> Dict[int, ClientDataset]:
        """
        Generate static data for clients (for pre-drift or post-drift phases).
        
        Uses HYPERPLANE_COEFF_SEED for coefficient generation to ensure x4
        is always the most important feature. The experiment seed controls
        which samples each client gets from the pool.
        
        Args:
            drifted_clients: Set of client IDs that will experience drift
            drift_magnitude: Magnitude of drift
            generate_drifted: If True, generate drifted data; else generate baseline
        
        Returns:
            Dictionary mapping client_id to ClientDataset
        """
        pool_size = max(self.n_samples_per_client * 20, 10000)
        baseline_pool = self._get_baseline_pool(pool_size)
        
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            is_drifted_client = client_id in drifted_clients
            
            if generate_drifted and is_drifted_client:
                drifted_pool = self._get_drifted_pool(drift_magnitude, pool_size)
                X, y = self._sample_client_data(drifted_pool, client_id)
            else:
                X, y = self._sample_client_data(baseline_pool, client_id)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.seed + client_id,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_drifted=generate_drifted and is_drifted_client,
            )
        
        return client_datasets
