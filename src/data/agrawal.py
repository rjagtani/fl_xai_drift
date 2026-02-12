"""
Agrawal dataset generator for federated learning with concept drift.

Pre-generates large pools for pre- and post-drift classification functions
on first use, then samples from the cached arrays each round (fast).
"""

from typing import Dict, List, Set, Optional
import numpy as np
from river.datasets import synth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseDataGenerator, ClientDataset


class AgrawalDataGenerator(BaseDataGenerator):
    """
    Generates Agrawal datasets for FL with configurable concept drift.
    
    The Agrawal dataset simulates salary classification based on person attributes.
    Concept drift is simulated by switching the classification function.
    
    Classification functions define different decision rules:
    - Function 0: age < 40 OR age >= 60
    - Function 1: age >= 40 AND age < 60 AND elevel >= 3
    - Function 2: age >= 40 AND age < 60 AND elevel >= 2 AND salary >= 50000
    - etc.
    
    Different functions emphasize different features, enabling controlled drift.
    """
    
    FEATURE_NAMES = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
    
    FUNCTION_RELEVANT_FEATURES = {
        0: {2},
        1: {2, 3},
        2: {2, 3, 0},
        3: {2, 3, 0, 1},
        4: {2, 0, 1, 6},
        5: {2, 0, 4},
        6: {2, 0, 3, 7},
        7: {2, 0, 6, 8},
        8: {2, 0, 1, 5},
        9: {2, 0, 1, 6, 8},
    }
    
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        classification_function_pre: int = 0,
        classification_function_post: int = 2,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(self.FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        
        self.classification_function_pre = classification_function_pre
        self.classification_function_post = classification_function_post
        
        pre_features = self.FUNCTION_RELEVANT_FEATURES.get(classification_function_pre, set())
        post_features = self.FUNCTION_RELEVANT_FEATURES.get(classification_function_post, set())
        self._drifted_feature_indices = pre_features.symmetric_difference(post_features)
        
        self._label_encoders: Dict[str, LabelEncoder] = {}
        
        self._scaler: Optional[StandardScaler] = None
        self._scaler_fitted = False

        self._pre_pool: Optional[np.ndarray] = None
        self._pre_pool_y: Optional[np.ndarray] = None
        self._post_pool: Optional[np.ndarray] = None
        self._post_pool_y: Optional[np.ndarray] = None
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.FEATURE_NAMES.copy()
    
    def get_drifted_feature_indices(self) -> Set[int]:
        """Return indices of features that drift (change in importance)."""
        return self._drifted_feature_indices
    
    def _generate_pool(self, clf_fn: int, pool_size: int, stream_seed: int) -> tuple:
        """Generate a large pool of (X, y) from the Agrawal stream (vectorised)."""
        stream = synth.Agrawal(classification_function=clf_fn, seed=stream_seed)

        raw_X = {fname: [] for fname in self.FEATURE_NAMES}
        labels = []
        for x, y in stream.take(pool_size):
            for fname in self.FEATURE_NAMES:
                raw_X[fname].append(x[fname])
            labels.append(y)

        cols = []
        for fname in self.FEATURE_NAMES:
            arr = np.array(raw_X[fname])
            if fname in ('elevel', 'car', 'zipcode'):
                arr = arr.astype(np.float64)
            else:
                arr = arr.astype(np.float64)
            cols.append(arr)

        X = np.column_stack(cols)
        y = np.array(labels, dtype=int)
        return X, y

    def _ensure_pools(self):
        """Pre-generate pre-drift and post-drift pools (once)."""
        if self._pre_pool is not None:
            return

        pool_size = max(self.n_samples_per_client * self.n_clients * 2, 20000)

        print(f"  Agrawal: generating pre-drift pool ({pool_size} samples, fn={self.classification_function_pre}) ...")
        X_pre, y_pre = self._generate_pool(
            self.classification_function_pre, pool_size, stream_seed=self.seed)

        print(f"  Agrawal: generating post-drift pool ({pool_size} samples, fn={self.classification_function_post}) ...")
        X_post, y_post = self._generate_pool(
            self.classification_function_post, pool_size, stream_seed=self.seed + 999)

        self._scaler = StandardScaler()
        self._scaler.fit(X_pre)
        self._scaler_fitted = True

        self._pre_pool = self._scaler.transform(X_pre).astype(np.float32)
        self._pre_pool_y = y_pre
        self._post_pool = self._scaler.transform(X_post).astype(np.float32)
        self._post_pool_y = y_post

    def _sample_client_data(self, X_pool: np.ndarray, y_pool: np.ndarray,
                            client_id: int, extra_seed: int = 0) -> tuple:
        """Sample n_samples_per_client from a pool, seeded per client."""
        rng = np.random.default_rng(self.seed + client_id * 1000 + extra_seed)
        idx = rng.choice(len(y_pool), size=self.n_samples_per_client, replace=False)
        return X_pool[idx], y_pool[idx]

    def generate_round_data(
        self,
        round_num: int,
        t0: int,
        drifted_clients: Set[int],
        drift_magnitude: float,
    ) -> Dict[int, ClientDataset]:
        """
        Generate data for all clients for a given round.
        
        For Agrawal, concept drift affects ALL clients simultaneously (global concept shift).
        The drifted_clients parameter is ignored - all clients switch classification function at t0.
        """
        self._ensure_pools()
        is_drift_phase = round_num >= t0
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            if is_drift_phase:
                X, y = self._sample_client_data(
                    self._post_pool, self._post_pool_y, client_id, extra_seed=round_num)
            else:
                X, y = self._sample_client_data(
                    self._pre_pool, self._pre_pool_y, client_id, extra_seed=round_num)
            
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
                is_drifted=is_drift_phase,
            )
        
        return client_datasets
    
    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
    ) -> Dict[int, ClientDataset]:
        """
        Generate static data for clients (used by run_drift_types.py).
        Samples from pre-generated pools.
        """
        self._ensure_pools()
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            is_drifted_client = client_id in drifted_clients
            
            if generate_drifted and is_drifted_client:
                X, y = self._sample_client_data(
                    self._post_pool, self._post_pool_y, client_id)
            else:
                X, y = self._sample_client_data(
                    self._pre_pool, self._pre_pool_y, client_id)
            
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
